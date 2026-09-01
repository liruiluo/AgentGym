from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import resource
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .actions import (
    OpenMLEFastActionError,
    OpenMLEFastProtectedPathError,
    ParsedPolicyAction,
    apply_workspace_patch,
)
from .bounded_text import bound_text
from .deadline import DeadlineExceeded, MonotonicDeadline
from .runtime_attestation import exact_runtime_identity_is_attested

EXECUTOR_CONTRACT = "openmle_fast_executor_v1"
EXTERNAL_RUNNER_CONTRACT = "openmle_fast_linux_cgroup_namespace_runner_v1"
FIT_HOOK_CONTRACT = "openmle_fast_fit_hook_v1"
# Keep the existing policy-time reserve stable, but give the host-side exact
# runner longer to drain pipes and reap cgroup-v1 state under fully-async fanout.
EXTERNAL_RUNNER_COMPLETION_GRACE_MS = 3_000
EXTERNAL_RUNNER_PROCESS_GRACE_MS = 34_000


def _external_runner_environment() -> dict[str, str]:
    # Policy-authored shell text may contain arbitrary Unicode.  The exact
    # runner is a Python 3.6 executable, so its filesystem encoding is fixed at
    # interpreter startup from LC_ALL; the plain C locale makes argv encoding
    # ASCII and turns valid policy commands into infrastructure exclusions.
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class OpenMLEFastExecutorError(RuntimeError):
    pass


class WorkspaceInvariantError(OpenMLEFastExecutorError):
    pass


class OpenMLEFastExecutionDeadlineExceeded(DeadlineExceeded):
    def __init__(self, backend: BackendExecution) -> None:
        super().__init__("shell wrapper deadline expired")
        self.backend = backend


@dataclass(frozen=True)
class OpenMLEFastResourceLimits:
    max_policy_actions: int
    cpu_vcpus: int
    memory_bytes: int
    swap_bytes: int
    workspace_bytes: int
    tmp_bytes: int
    max_processes: int
    max_open_files: int
    max_files: int
    max_file_bytes: int
    max_submission_bytes: int
    shell_wall_ms: int
    managed_runtime_per_action_ms: int
    managed_runtime_per_episode_ms: int
    episode_wall_ms: int
    grader_cpu_vcpus: int
    grader_memory_bytes: int
    grader_max_processes: int
    grader_worker_wall_ms: int
    grader_total_wall_ms: int
    grader_max_concurrent_requests: int
    grader_input_bytes: int
    raw_output_bytes: int
    observation_bytes: int
    observation_head_bytes: int
    observation_tail_bytes: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"resource limit {name} must be a non-negative integer"
                )
            if name != "swap_bytes" and value == 0:
                raise ValueError(f"resource limit {name} must be positive")
        if self.max_policy_actions != 30:
            raise ValueError("OpenMLE-fast v1 has exactly 30 policy actions")
        if (
            self.observation_head_bytes + self.observation_tail_bytes
            != self.observation_bytes
        ):
            raise ValueError(
                "observation head/tail bytes must fill the observation cap"
            )
        if self.max_submission_bytes > self.max_file_bytes:
            raise ValueError("submission cap cannot exceed the regular-file cap")
        if self.managed_runtime_per_action_ms > self.shell_wall_ms:
            raise ValueError("managed runtime cannot exceed one shell wall limit")
        if self.grader_worker_wall_ms > self.grader_total_wall_ms:
            raise ValueError("grader worker wall cannot exceed total grader wall")

    @classmethod
    def frozen_v1(cls) -> OpenMLEFastResourceLimits:
        gib = 1024**3
        mib = 1024**2
        kib = 1024
        return cls(
            max_policy_actions=30,
            cpu_vcpus=2,
            memory_bytes=4 * gib,
            swap_bytes=0,
            workspace_bytes=2 * gib,
            tmp_bytes=256 * mib,
            max_processes=64,
            max_open_files=256,
            max_files=100_000,
            max_file_bytes=256 * mib,
            max_submission_bytes=64 * mib,
            shell_wall_ms=20_000,
            managed_runtime_per_action_ms=15_000,
            managed_runtime_per_episode_ms=120_000,
            episode_wall_ms=180_000,
            grader_cpu_vcpus=1,
            grader_memory_bytes=2 * gib,
            grader_max_processes=32,
            grader_worker_wall_ms=4_000,
            grader_total_wall_ms=5_000,
            grader_max_concurrent_requests=8,
            grader_input_bytes=64 * mib,
            raw_output_bytes=8 * mib,
            observation_bytes=64 * kib,
            observation_head_bytes=32 * kib,
            observation_tail_bytes=32 * kib,
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class BackendExecution:
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    timed_out: bool
    wall_seconds: float
    managed_runtime_wall_seconds: float
    cpu_seconds: float | None
    peak_rss_bytes: int | None
    bytes_read: int | None
    bytes_written: int | None
    process_peak: int | None
    execution_attempt_delta: int
    execution_completed_delta: int
    nested_subprocess_delta: int
    fit_delta: int
    fit_counter_coverage: str
    failure_class: str | None = None
    infrastructure_fault: bool = False


class ExecutionBackend(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def run(
        self,
        workspace: Path,
        *,
        command: str,
        timeout_ms: int,
        managed_runtime_budget_ms: int,
    ) -> BackendExecution: ...

    def freeze(self, workspace: Path, *, timeout_ms: int) -> BackendLifecycle: ...

    def teardown(self, workspace: Path) -> BackendLifecycle: ...


@dataclass(frozen=True)
class BackendLifecycle:
    schema: str
    operation: str
    success: bool
    processes_reaped: bool
    workspace_read_only: bool
    cgroup_empty: bool
    failure_class: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionReceipt:
    schema: str
    status: str
    action_kind: str
    exit_code: int | None
    timed_out: bool
    failure_class: str | None
    stdout: str
    stderr: str
    raw_output_bytes: int
    output_sha256: str
    visible_output_truncated: bool
    wall_seconds: float
    managed_runtime_wall_seconds: float
    cpu_seconds: float | None
    peak_rss_bytes: int | None
    bytes_read: int | None
    bytes_written: int | None
    process_peak: int | None
    execution_action_delta: int
    execution_attempt_delta: int
    execution_completed_delta: int
    nested_subprocess_delta: int
    fit_delta: int
    fit_counter_coverage: str
    fit_hook_digest: str
    tree_sha256_before: str
    tree_sha256_after: str
    changed_paths: tuple[str, ...]
    backend_contract: str

    @property
    def policy_terminal(self) -> bool:
        return self.status == "policy_violation"

    @property
    def infrastructure_fault(self) -> bool:
        return self.status == "infrastructure_fault"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_paths"] = list(self.changed_paths)
        return value


@dataclass(frozen=True)
class _TreeSnapshot:
    tree_sha256: str
    protected_sha256: str
    paths: frozenset[str]
    fingerprints: Mapping[str, str]


class LocalCPUExecutionBackend:
    """Host subprocess backend for CPU tests only.

    It intentionally reports partial counters and ``formal_eligible=False``.
    Production launch code never promotes this backend to exact-runtime status.
    """

    def __init__(self, limits: OpenMLEFastResourceLimits) -> None:
        self.limits = limits
        self._process_groups: set[int] = set()
        self._process_lock = threading.Lock()

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "contract": "openmle_fast_local_cpu_test_backend_v1",
            "formal_eligible": False,
            "network_isolation": "not_implemented",
            "namespace_coverage": "none",
            "cgroup_coverage": "none",
            "execution_counter_coverage": "partial_static_shell_scan",
            "fit_counter_coverage": "partial",
            "cumulative_managed_runtime_budget": False,
            "adapter_completion_grace_ms": 0,
            "resource_limits": self.limits.as_dict(),
        }

    def run(
        self,
        workspace: Path,
        *,
        command: str,
        timeout_ms: int,
        managed_runtime_budget_ms: int,
    ) -> BackendExecution:
        attempts = len(_PYTHON_ENTRYPOINT_RE.findall(command))
        if attempts:
            timeout_ms = min(timeout_ms, managed_runtime_budget_ms)
        nested = _estimated_subprocesses(command)
        tmp = workspace / ".openmle-tmp"
        tmp.mkdir(mode=0o700, exist_ok=True)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/nonexistent",
            "TMPDIR": str(tmp),
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                ["/bin/sh", "-c", command],
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            with self._process_lock:
                self._process_groups.add(process.pid)
        except OSError:
            return BackendExecution(
                stdout=b"",
                stderr=b"",
                exit_code=None,
                timed_out=False,
                wall_seconds=max(0.0, time.monotonic() - started),
                managed_runtime_wall_seconds=0.0,
                cpu_seconds=None,
                peak_rss_bytes=None,
                bytes_read=None,
                bytes_written=None,
                process_peak=None,
                execution_attempt_delta=0,
                execution_completed_delta=0,
                nested_subprocess_delta=0,
                fit_delta=0,
                fit_counter_coverage="partial",
                failure_class="backend_start_failure",
                infrastructure_fault=True,
            )
        timed_out = False
        surviving_background = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_ms / 1000.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process.pid)
            stdout, stderr = process.communicate()
        finally:
            surviving_background = _process_group_exists(process.pid)
            _kill_process_group(process.pid)
            with self._process_lock:
                self._process_groups.discard(process.pid)
        wall_seconds = max(0.0, time.monotonic() - started)
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_seconds = max(
            0.0,
            (usage_after.ru_utime + usage_after.ru_stime)
            - (usage_before.ru_utime + usage_before.ru_stime),
        )
        raw_bytes = len(stdout) + len(stderr)
        failure_class: str | None = None
        exit_code = 124 if timed_out else int(process.returncode)
        if timed_out:
            failure_class = "wall_timeout"
        elif surviving_background:
            failure_class = "surviving_background_process"
        elif raw_bytes > self.limits.raw_output_bytes:
            failure_class = "output_limit"
        completed = attempts if exit_code == 0 and failure_class is None else 0
        return BackendExecution(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            wall_seconds=wall_seconds,
            managed_runtime_wall_seconds=wall_seconds if attempts else 0.0,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=_rss_bytes(usage_after.ru_maxrss),
            bytes_read=None,
            bytes_written=None,
            process_peak=max(1, nested + 1),
            execution_attempt_delta=attempts,
            execution_completed_delta=completed,
            nested_subprocess_delta=nested,
            fit_delta=0,
            fit_counter_coverage="partial",
            failure_class=failure_class,
        )

    def freeze(self, workspace: Path, *, timeout_ms: int) -> BackendLifecycle:
        deadline = MonotonicDeadline.after_ms(timeout_ms)
        reaped = self._kill_all_process_groups()
        deadline.check()
        try:
            _make_workspace_read_only(workspace, deadline=deadline)
            deadline.check()
        except OSError:
            return BackendLifecycle(
                schema="openmle_fast_backend_lifecycle_v1",
                operation="freeze",
                success=False,
                processes_reaped=reaped,
                workspace_read_only=False,
                cgroup_empty=reaped,
                failure_class="workspace_freeze_failure",
            )
        return BackendLifecycle(
            schema="openmle_fast_backend_lifecycle_v1",
            operation="freeze",
            success=reaped,
            processes_reaped=reaped,
            workspace_read_only=True,
            cgroup_empty=reaped,
            failure_class=None if reaped else "process_reap_failure",
        )

    def teardown(self, workspace: Path) -> BackendLifecycle:
        del workspace
        reaped = self._kill_all_process_groups()
        return BackendLifecycle(
            schema="openmle_fast_backend_lifecycle_v1",
            operation="teardown",
            success=reaped,
            processes_reaped=reaped,
            workspace_read_only=False,
            cgroup_empty=reaped,
            failure_class=None if reaped else "process_reap_failure",
        )

    def _kill_all_process_groups(self) -> bool:
        with self._process_lock:
            groups = tuple(self._process_groups)
        for process_group in groups:
            _kill_process_group(process_group)
        alive = tuple(
            process_group
            for process_group in groups
            if _process_group_exists(process_group)
        )
        with self._process_lock:
            self._process_groups.difference_update(groups)
        return not alive


class ExternalSandboxRunnerBackend:
    """Strict adapter for a separately installed exact Linux sandbox runner."""

    def __init__(
        self,
        *,
        runner_path: Path | str,
        expected_runner_sha256: str,
        expected_runtime_digest: str,
        expected_artifact_lock_sha256: str,
        limits: OpenMLEFastResourceLimits,
    ) -> None:
        self.runner_path = Path(runner_path).expanduser().absolute()
        if self.runner_path.is_symlink() or not self.runner_path.is_file():
            raise OpenMLEFastExecutorError("sandbox runner must be a real file")
        if not os.access(self.runner_path, os.X_OK):
            raise OpenMLEFastExecutorError("sandbox runner is not executable")
        actual = _file_sha256(self.runner_path)
        if actual != expected_runner_sha256:
            raise OpenMLEFastExecutorError("sandbox runner SHA256 mismatch")
        self.limits = limits
        self.expected_runtime_digest = expected_runtime_digest
        self.expected_artifact_lock_sha256 = expected_artifact_lock_sha256
        self._metadata = self._load_metadata()

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def _load_metadata(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [str(self.runner_path), "metadata"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
                env=_external_runner_environment(),
            )
            value = _strict_json_loads(result.stdout)
        except (
            OSError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise OpenMLEFastExecutorError(
                "cannot attest sandbox runner metadata"
            ) from exc
        required_true = (
            "formal_eligible",
            "network_namespace",
            "network_no_egress",
            "dns_disabled",
            "metadata_service_blocked",
            "external_unix_sockets_blocked",
            "pid_namespace",
            "ipc_namespace",
            "mount_namespace",
            "workspace_quota",
            "tmpfs_limit",
            "file_size_limit",
            "open_files_limit",
            "fresh_unprivileged_uid_gid",
            "capabilities_dropped",
            "no_new_privs",
            "seccomp",
            "read_only_rootfs",
            "workspace_noexec",
            "instrumented_python_only",
            "cumulative_managed_runtime_budget",
            "isolated_proc",
            "minimal_devices",
            "gpu_devices_absent",
            "core_dumps_disabled",
            "mount_denied",
            "ptrace_denied",
            "setuid_denied",
            "user_namespace_creation_denied",
            "ebpf_denied",
            "raw_sockets_denied",
            "kernel_module_denied",
            "container_engine_absent",
            "background_process_detection",
            "descendant_kill_reap",
            "parent_death_cleanup_watchdog",
            "freeze_reap",
            "read_only_workspace_freeze",
            "teardown_cgroup_empty",
            "teardown_mount_empty",
            "idempotent_teardown",
        )
        if (
            result.returncode != 0
            or not isinstance(value, dict)
            or value.get("contract") != EXTERNAL_RUNNER_CONTRACT
            or value.get("runtime_digest") != self.expected_runtime_digest
            or value.get("resource_limits") != self.limits.as_dict()
            or any(value.get(key) is not True for key in required_true)
            or not exact_runtime_identity_is_attested(
                value,
                expected_artifact_lock_sha256=self.expected_artifact_lock_sha256,
            )
            or value.get("execution_counter_coverage") != "complete"
            or value.get("fit_counter_coverage") != "partial"
            or value.get("fit_hook_contract") != FIT_HOOK_CONTRACT
            or value.get("fit_hook_digest")
            != hashlib.sha256(FIT_HOOK_CONTRACT.encode()).hexdigest()
        ):
            raise OpenMLEFastExecutorError("sandbox runner attestation is incomplete")
        value = dict(value)
        value["adapter_completion_grace_ms"] = EXTERNAL_RUNNER_COMPLETION_GRACE_MS
        value["adapter_host_grace_ms"] = EXTERNAL_RUNNER_PROCESS_GRACE_MS
        return value

    def run(
        self,
        workspace: Path,
        *,
        command: str,
        timeout_ms: int,
        managed_runtime_budget_ms: int,
    ) -> BackendExecution:
        # The exact runner reserves this interval for admission and cleanup.
        # A shorter policy deadline is a policy timeout, not an infrastructure
        # fault that should drop the trajectory.
        if timeout_ms <= EXTERNAL_RUNNER_COMPLETION_GRACE_MS:
            return BackendExecution(
                stdout=b"",
                stderr=b"",
                exit_code=124,
                timed_out=True,
                wall_seconds=0.0,
                managed_runtime_wall_seconds=0.0,
                cpu_seconds=0.0,
                peak_rss_bytes=0,
                bytes_read=0,
                bytes_written=0,
                process_peak=0,
                execution_attempt_delta=0,
                execution_completed_delta=0,
                nested_subprocess_delta=0,
                fit_delta=0,
                fit_counter_coverage="partial",
                failure_class="wall_timeout",
                infrastructure_fault=False,
            )
        request = {
            "schema": "openmle_fast_runner_request_v1",
            "workspace": str(workspace),
            "command": command,
            "timeout_ms": timeout_ms,
            "managed_runtime_budget_ms": managed_runtime_budget_ms,
        }
        started = time.monotonic()
        try:
            result = subprocess.run(
                [str(self.runner_path), "execute"],
                input=json.dumps(
                    request, sort_keys=True, separators=(",", ":")
                ).encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=(timeout_ms + EXTERNAL_RUNNER_PROCESS_GRACE_MS) / 1000.0,
                check=False,
                env=_external_runner_environment(),
            )
            value = _strict_json_loads(result.stdout)
            return _decode_runner_execution(value, result.returncode)
        except subprocess.TimeoutExpired:
            return BackendExecution(
                stdout=b"",
                stderr=b"",
                exit_code=124,
                timed_out=False,
                wall_seconds=max(0.0, time.monotonic() - started),
                managed_runtime_wall_seconds=0.0,
                cpu_seconds=None,
                peak_rss_bytes=None,
                bytes_read=None,
                bytes_written=None,
                process_peak=None,
                execution_attempt_delta=0,
                execution_completed_delta=0,
                nested_subprocess_delta=0,
                fit_delta=0,
                fit_counter_coverage="unknown",
                failure_class="runner_protocol_fault",
                infrastructure_fault=True,
            )
        except (OSError, json.JSONDecodeError, ValueError):
            return BackendExecution(
                stdout=b"",
                stderr=b"",
                exit_code=None,
                timed_out=False,
                wall_seconds=0.0,
                managed_runtime_wall_seconds=0.0,
                cpu_seconds=None,
                peak_rss_bytes=None,
                bytes_read=None,
                bytes_written=None,
                process_peak=None,
                execution_attempt_delta=0,
                execution_completed_delta=0,
                nested_subprocess_delta=0,
                fit_delta=0,
                fit_counter_coverage="unknown",
                failure_class="runner_protocol_fault",
                infrastructure_fault=True,
            )

    def freeze(self, workspace: Path, *, timeout_ms: int) -> BackendLifecycle:
        return self._lifecycle("freeze", workspace, timeout_ms=timeout_ms)

    def teardown(self, workspace: Path) -> BackendLifecycle:
        return self._lifecycle("teardown", workspace, timeout_ms=5_000)

    def _lifecycle(
        self,
        operation: str,
        workspace: Path,
        *,
        timeout_ms: int,
    ) -> BackendLifecycle:
        if timeout_ms <= EXTERNAL_RUNNER_COMPLETION_GRACE_MS:
            return BackendLifecycle(
                schema="openmle_fast_backend_lifecycle_v1",
                operation=operation,
                success=False,
                processes_reaped=False,
                workspace_read_only=False,
                cgroup_empty=False,
                failure_class="runner_lifecycle_fault",
            )
        effective_timeout_ms = min(
            timeout_ms,
            self.limits.shell_wall_ms + EXTERNAL_RUNNER_COMPLETION_GRACE_MS,
        )
        request = {
            "schema": "openmle_fast_runner_lifecycle_request_v1",
            "workspace": str(workspace),
            "operation": operation,
            "timeout_ms": (
                effective_timeout_ms - EXTERNAL_RUNNER_COMPLETION_GRACE_MS
            ),
        }
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                [str(self.runner_path), operation],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=_external_runner_environment(),
                start_new_session=True,
            )
            request_bytes = json.dumps(
                request, sort_keys=True, separators=(",", ":")
            ).encode()
            try:
                stdout, _ = process.communicate(
                    request_bytes,
                    timeout=effective_timeout_ms / 1000.0,
                )
            except subprocess.TimeoutExpired:
                _kill_process_group(process.pid)
                process.communicate()
                raise
            if _process_group_exists(process.pid):
                _kill_process_group(process.pid)
                raise ValueError("runner lifecycle left surviving descendants")
            value = _strict_json_loads(stdout)
            required = {
                "schema",
                "operation",
                "success",
                "processes_reaped",
                "workspace_read_only",
                "cgroup_empty",
                "failure_class",
            }
            if (
                process.returncode != 0
                or not isinstance(value, dict)
                or set(value) != required
                or value["schema"] != "openmle_fast_backend_lifecycle_v1"
                or value["operation"] != operation
                or any(
                    type(value[key]) is not bool
                    for key in (
                        "success",
                        "processes_reaped",
                        "workspace_read_only",
                        "cgroup_empty",
                    )
                )
                or (
                    value["failure_class"] is not None
                    and not isinstance(value["failure_class"], str)
                )
            ):
                raise ValueError("runner lifecycle receipt is invalid")
            return BackendLifecycle(**value)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            return BackendLifecycle(
                schema="openmle_fast_backend_lifecycle_v1",
                operation=operation,
                success=False,
                processes_reaped=False,
                workspace_read_only=False,
                cgroup_empty=False,
                failure_class="runner_lifecycle_fault",
            )
        finally:
            if process is not None and process.poll() is None:
                _kill_process_group(process.pid)
                process.communicate()


class OpenMLEFastExecutor:
    def __init__(
        self,
        *,
        limits: OpenMLEFastResourceLimits,
        backend: ExecutionBackend,
    ) -> None:
        self.limits = limits
        self.backend = backend
        backend_limits = backend.metadata.get("resource_limits")
        if backend_limits != limits.as_dict():
            raise OpenMLEFastExecutorError("executor/backend resource limits disagree")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "contract": EXECUTOR_CONTRACT,
            "backend": dict(self.backend.metadata),
            "resource_limits": self.limits.as_dict(),
            "fit_hook_contract": FIT_HOOK_CONTRACT,
        }

    @property
    def runner_owns_workspace_lifecycle(self) -> bool:
        """Whether the attested backend adopts and unmounts policy workspaces."""

        metadata = self.backend.metadata
        return bool(
            metadata.get("teardown_mount_empty") is True
            and metadata.get("workspace_storage_contract")
            == "owned_dedicated_tmpfs_at_most_2g_v1"
        )

    def freeze_for_grading(
        self,
        workspace: Path | str,
        *,
        deadline: MonotonicDeadline,
    ) -> BackendLifecycle:
        root = Path(workspace).absolute()
        deadline.check()
        receipt = self.backend.freeze(
            root,
            timeout_ms=deadline.remaining_milliseconds(),
        )
        deadline.check()
        if (
            receipt.operation != "freeze"
            or not receipt.success
            or not receipt.processes_reaped
            or not receipt.workspace_read_only
            or not receipt.cgroup_empty
        ):
            raise OpenMLEFastExecutorError("sandbox freeze/reap failed closed")
        return receipt

    def close(self, workspace: Path | str) -> BackendLifecycle:
        root = Path(workspace).absolute()
        receipt = self.backend.teardown(root)
        if (
            receipt.operation != "teardown"
            or not receipt.success
            or not receipt.processes_reaped
            or not receipt.cgroup_empty
        ):
            raise OpenMLEFastExecutorError("sandbox teardown failed closed")
        return receipt

    def execute(
        self,
        workspace: Path | str,
        action: ParsedPolicyAction,
        *,
        deadline: MonotonicDeadline | None = None,
        managed_runtime_budget_ms: int | None = None,
    ) -> ExecutionReceipt:
        started = time.monotonic()
        root = Path(workspace).absolute()
        effective_deadline = deadline
        if action.kind == "shell_command":
            host_grace_ms = self.backend.metadata.get("adapter_host_grace_ms", 0)
            if type(host_grace_ms) is not int or host_grace_ms < 0:
                raise OpenMLEFastExecutorError("backend host grace is invalid")
            effective_deadline = MonotonicDeadline.after_ms(
                self.limits.shell_wall_ms + host_grace_ms,
                cap=deadline,
            )
            if managed_runtime_budget_ms is not None and (
                type(managed_runtime_budget_ms) is not int
                or managed_runtime_budget_ms <= 0
            ):
                raise ValueError("managed runtime budget must be a positive integer")
        before = _snapshot_tree(
            root,
            self.limits,
            deadline=effective_deadline,
        )
        if action.kind == "apply_patch":
            return self._execute_patch(
                root,
                action,
                before,
                deadline=effective_deadline,
                started=started,
            )
        if action.kind != "shell_command" or action.arguments is None:
            raise OpenMLEFastExecutorError(
                "executor accepts parsed shell_command or apply_patch actions only"
            )
        assert effective_deadline is not None
        return self._execute_shell(
            root,
            action,
            before,
            deadline=effective_deadline,
            started=started,
            managed_runtime_budget_ms=managed_runtime_budget_ms,
        )

    def _execute_patch(
        self,
        root: Path,
        action: ParsedPolicyAction,
        before: _TreeSnapshot,
        *,
        deadline: MonotonicDeadline | None,
        started: float,
    ) -> ExecutionReceipt:
        if deadline is not None:
            deadline.check()
        changed: tuple[str, ...] = ()
        status = "completed"
        failure_class = None
        try:
            result = apply_workspace_patch(root, action.patch or "")
            changed = result.changed_paths
        except OpenMLEFastProtectedPathError:
            status = "policy_violation"
            failure_class = "immutable_public_tree_mutation_attempt"
        except OpenMLEFastActionError:
            status = "rejected"
            failure_class = "patch_rejected"
        if deadline is not None:
            deadline.check()
        after = _snapshot_tree(root, self.limits, deadline=deadline)
        receipt = self._receipt(
            status=status,
            action_kind=action.kind,
            before=before,
            after=after,
            changed=changed,
            backend=None,
            failure_class=failure_class,
            wall_seconds=max(0.0, time.monotonic() - started),
            deadline=deadline,
        )
        if deadline is not None:
            deadline.check()
        return receipt

    def _execute_shell(
        self,
        root: Path,
        action: ParsedPolicyAction,
        before: _TreeSnapshot,
        *,
        deadline: MonotonicDeadline,
        started: float,
        managed_runtime_budget_ms: int | None,
    ) -> ExecutionReceipt:
        assert action.arguments is not None
        deadline.check()
        completion_grace_ms = self.backend.metadata.get(
            "adapter_completion_grace_ms", 0
        )
        if type(completion_grace_ms) is not int or completion_grace_ms < 0:
            raise OpenMLEFastExecutorError("backend completion grace is invalid")
        remaining_ms = deadline.remaining_milliseconds()
        if remaining_ms <= completion_grace_ms:
            raise DeadlineExceeded("no time remains for runner cleanup")
        timeout_ms = min(
            int(action.arguments["timeout_ms"]),
            remaining_ms - completion_grace_ms,
        )
        command = str(action.arguments["command"])
        effective_managed_runtime_budget_ms = min(
            self.limits.managed_runtime_per_action_ms,
            (
                self.limits.managed_runtime_per_episode_ms
                if managed_runtime_budget_ms is None
                else managed_runtime_budget_ms
            ),
        )
        backend = self.backend.run(
            root,
            command=command,
            timeout_ms=timeout_ms,
            managed_runtime_budget_ms=effective_managed_runtime_budget_ms,
        )
        try:
            deadline.check()
            status = "completed"
            failure_class = backend.failure_class
            after = _snapshot_tree(root, self.limits, deadline=deadline)
            if after.protected_sha256 != before.protected_sha256:
                status = "policy_violation"
                failure_class = "immutable_public_tree_changed"
        except DeadlineExceeded as exc:
            raise OpenMLEFastExecutionDeadlineExceeded(backend) from exc
        except WorkspaceInvariantError:
            _clear_invalid_workspace_for_teardown(root)
            after = before
            status = "policy_violation"
            failure_class = "workspace_invariant_violation"
        try:
            raw_output_bytes = len(backend.stdout) + len(backend.stderr)
            if raw_output_bytes > self.limits.raw_output_bytes:
                status = "policy_violation"
                failure_class = "output_limit"
            if backend.infrastructure_fault:
                # A policy-authored workspace invariant violation is detected
                # after the runner returns.  The runner may also report an
                # infrastructure fault while handling that invalid tree, but
                # it must not turn the policy-caused terminal into a dropped
                # trajectory.
                if status != "policy_violation":
                    status = "infrastructure_fault"
            elif (
                effective_managed_runtime_budget_ms is not None
                and backend.managed_runtime_wall_seconds * 1000.0
                >= effective_managed_runtime_budget_ms
            ):
                status = "policy_violation"
                failure_class = "managed_runtime_limit"
            elif backend.failure_class in {
                "wall_timeout",
                "output_limit",
                "memory_limit",
                "pid_limit",
                "disk_limit",
                "file_limit",
                "security_violation",
                "background_process",
                "surviving_background_process",
                "managed_runtime_limit",
            }:
                status = "policy_violation"
            changed = tuple(
                sorted(
                    path
                    for path in before.paths | after.paths
                    if before.fingerprints.get(path) != after.fingerprints.get(path)
                )
            )
            receipt = self._receipt(
                status=status,
                action_kind=action.kind,
                before=before,
                after=after,
                changed=changed,
                backend=backend,
                failure_class=failure_class,
                wall_seconds=max(0.0, time.monotonic() - started),
                deadline=deadline,
            )
            deadline.check()
            return receipt
        except DeadlineExceeded as exc:
            raise OpenMLEFastExecutionDeadlineExceeded(backend) from exc

    def _receipt(
        self,
        *,
        status: str,
        action_kind: str,
        before: _TreeSnapshot,
        after: _TreeSnapshot,
        changed: tuple[str, ...],
        backend: BackendExecution | None,
        failure_class: str | None,
        wall_seconds: float,
        deadline: MonotonicDeadline | None,
    ) -> ExecutionReceipt:
        if deadline is not None:
            deadline.check()
        stdout_raw = b"" if backend is None else backend.stdout
        stderr_raw = b"" if backend is None else backend.stderr
        raw = stdout_raw + b"\0" + stderr_raw
        stdout, stdout_truncated = _visible_text(
            stdout_raw, self.limits.observation_bytes // 2
        )
        stderr, stderr_truncated = _visible_text(
            stderr_raw, self.limits.observation_bytes // 2
        )
        if deadline is not None:
            deadline.check()
        attempts = 0 if backend is None else backend.execution_attempt_delta
        return ExecutionReceipt(
            schema="openmle_fast_execution_receipt_v1",
            status=status,
            action_kind=action_kind,
            exit_code=None if backend is None else backend.exit_code,
            timed_out=False if backend is None else backend.timed_out,
            failure_class=failure_class,
            stdout=stdout,
            stderr=stderr,
            raw_output_bytes=len(stdout_raw) + len(stderr_raw),
            output_sha256=hashlib.sha256(raw).hexdigest(),
            visible_output_truncated=stdout_truncated or stderr_truncated,
            wall_seconds=wall_seconds,
            managed_runtime_wall_seconds=(
                0.0 if backend is None else backend.managed_runtime_wall_seconds
            ),
            cpu_seconds=None if backend is None else backend.cpu_seconds,
            peak_rss_bytes=None if backend is None else backend.peak_rss_bytes,
            bytes_read=None if backend is None else backend.bytes_read,
            bytes_written=None if backend is None else backend.bytes_written,
            process_peak=None if backend is None else backend.process_peak,
            execution_action_delta=int(attempts > 0),
            execution_attempt_delta=attempts,
            execution_completed_delta=(
                0 if backend is None else backend.execution_completed_delta
            ),
            nested_subprocess_delta=(
                0 if backend is None else backend.nested_subprocess_delta
            ),
            fit_delta=0 if backend is None else backend.fit_delta,
            fit_counter_coverage=(
                "not_applicable" if backend is None else backend.fit_counter_coverage
            ),
            fit_hook_digest=hashlib.sha256(FIT_HOOK_CONTRACT.encode()).hexdigest(),
            tree_sha256_before=before.tree_sha256,
            tree_sha256_after=after.tree_sha256,
            changed_paths=changed,
            backend_contract=str(self.backend.metadata.get("contract")),
        )


_PYTHON_ENTRYPOINT_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[^\s;&|]*/)*python(?:3(?:\.\d+)?)?(?=\s|$)"
)


def _estimated_subprocesses(command: str) -> int:
    segments = re.split(r"(?:&&|\|\||[;|])", command)
    return sum(1 for segment in segments if segment.strip())


def _clear_invalid_workspace_for_teardown(root: Path) -> None:
    """Empty a terminally invalid policy tree without following its entries."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    original_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
    try:
        _clear_directory_descriptor(descriptor, flags)
    finally:
        os.fchmod(descriptor, original_mode)
        os.close(descriptor)


def _clear_directory_descriptor(descriptor: int, directory_flags: int) -> None:
    os.fchmod(descriptor, stat.S_IRWXU)
    parent_device = os.fstat(descriptor).st_dev
    with os.scandir(descriptor) as iterator:
        names = sorted(entry.name for entry in iterator)
    for name in names:
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            if info.st_dev != parent_device:
                raise WorkspaceInvariantError(
                    "workspace contains an unexpected nested mount"
                )
            child = os.open(name, directory_flags, dir_fd=descriptor)
            try:
                _clear_directory_descriptor(child, directory_flags)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _snapshot_tree(
    root: Path,
    limits: OpenMLEFastResourceLimits,
    *,
    deadline: MonotonicDeadline | None = None,
) -> _TreeSnapshot:
    if deadline is not None:
        deadline.check()
    if root.is_symlink() or not root.is_dir():
        raise WorkspaceInvariantError("workspace root is not a real directory")
    entries: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    total_bytes = 0
    file_count = 0
    paths: set[str] = set()
    fingerprints: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if deadline is not None:
            deadline.check()
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise WorkspaceInvariantError("workspace contains a non-independent file")
        if info.st_size > limits.max_file_bytes:
            raise WorkspaceInvariantError("workspace regular-file limit exceeded")
        file_count += 1
        total_bytes += info.st_size
        if file_count > limits.max_files or total_bytes > limits.workspace_bytes:
            raise WorkspaceInvariantError("workspace capacity limit exceeded")
        digest = _file_sha256(path, deadline=deadline)
        entry = {
            "path": relative,
            "size": info.st_size,
            "sha256": digest,
            "mode": stat.S_IMODE(info.st_mode),
        }
        entries.append(entry)
        paths.add(relative)
        fingerprints[relative] = _canonical_digest(entry)
        if relative == "TASK.md" or relative.startswith("data/"):
            protected.append(entry)
    if deadline is not None:
        deadline.check()
    return _TreeSnapshot(
        tree_sha256=_canonical_digest(entries),
        protected_sha256=_canonical_digest(protected),
        paths=frozenset(paths),
        fingerprints=fingerprints,
    )


def _canonical_digest(value: Any) -> str:
    payload = (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("runner JSON contains a duplicate key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)


def _visible_text(payload: bytes, limit: int) -> tuple[str, bool]:
    normalized = payload.decode("utf-8", errors="replace")
    bounded = bound_text(
        normalized,
        max_bytes=limit,
        marker="\n...[output truncated]...\n",
    )
    return bounded, bounded != normalized


def _file_sha256(
    path: Path,
    *,
    deadline: MonotonicDeadline | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if deadline is not None:
                deadline.check()
            digest.update(chunk)
    if deadline is not None:
        deadline.check()
    return digest.hexdigest()


def _rss_bytes(raw: float) -> int:
    # Linux reports KiB while Darwin reports bytes.
    return int(raw if os.uname().sysname == "Darwin" else raw * 1024)


def _make_workspace_read_only(
    root: Path,
    *,
    deadline: MonotonicDeadline | None = None,
) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if deadline is not None:
            deadline.check()
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            path.chmod(0o444)
        elif stat.S_ISDIR(info.st_mode):
            path.chmod(0o555)
        else:
            continue
    if deadline is not None:
        deadline.check()
    root.chmod(0o555)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_process_group(process_group: int) -> None:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _decode_runner_execution(value: Any, returncode: int) -> BackendExecution:
    if returncode != 0 or not isinstance(value, dict):
        raise ValueError("runner returned an invalid receipt")
    if value.get("schema") != "openmle_fast_runner_receipt_v1":
        raise ValueError("runner receipt schema mismatch")
    stdout = base64.b64decode(value["stdout_b64"], validate=True)
    stderr = base64.b64decode(value["stderr_b64"], validate=True)
    integers = {
        key: value.get(key)
        for key in (
            "execution_attempt_delta",
            "execution_completed_delta",
            "nested_subprocess_delta",
            "fit_delta",
        )
    }
    if any(type(item) is not int or item < 0 for item in integers.values()):
        raise ValueError("runner counters are invalid")
    if integers["execution_completed_delta"] > integers["execution_attempt_delta"]:
        raise ValueError("runner completed count exceeds attempts")
    numeric_names = (
        "wall_seconds",
        "managed_runtime_wall_seconds",
        "cpu_seconds",
    )
    numeric = {name: float(value[name]) for name in numeric_names}
    if any(not math.isfinite(item) or item < 0 for item in numeric.values()):
        raise ValueError("runner timing telemetry is invalid")
    count_names = ("peak_rss_bytes", "bytes_read", "bytes_written", "process_peak")
    counts = {name: value.get(name) for name in count_names}
    if any(type(item) is not int or item < 0 for item in counts.values()):
        raise ValueError("runner resource telemetry is invalid")
    exit_code = value.get("exit_code")
    if exit_code is not None and type(exit_code) is not int:
        raise ValueError("runner exit code is invalid")
    if (
        type(value.get("timed_out")) is not bool
        or type(value.get("infrastructure_fault")) is not bool
    ):
        raise ValueError("runner terminal flags are invalid")
    coverage = value.get("fit_counter_coverage")
    if coverage not in {"complete", "partial"}:
        raise ValueError("runner fit counter coverage is invalid")
    failure_class = value.get("failure_class")
    allowed_failures = {
        None,
        "wall_timeout",
        "output_limit",
        "memory_limit",
        "pid_limit",
        "disk_limit",
        "file_limit",
        "security_violation",
        "background_process",
        "managed_runtime_limit",
        "runner_infrastructure_fault",
    }
    if failure_class not in allowed_failures:
        raise ValueError("runner failure class is invalid")
    return BackendExecution(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=value["timed_out"],
        wall_seconds=numeric["wall_seconds"],
        managed_runtime_wall_seconds=numeric["managed_runtime_wall_seconds"],
        cpu_seconds=numeric["cpu_seconds"],
        peak_rss_bytes=counts["peak_rss_bytes"],
        bytes_read=counts["bytes_read"],
        bytes_written=counts["bytes_written"],
        process_peak=counts["process_peak"],
        execution_attempt_delta=integers["execution_attempt_delta"],
        execution_completed_delta=integers["execution_completed_delta"],
        nested_subprocess_delta=integers["nested_subprocess_delta"],
        fit_delta=integers["fit_delta"],
        fit_counter_coverage=coverage,
        failure_class=failure_class,
        infrastructure_fault=value["infrastructure_fault"],
    )
