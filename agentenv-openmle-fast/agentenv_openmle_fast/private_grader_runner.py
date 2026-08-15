from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .deadline import DeadlineExceeded, MonotonicDeadline

PRIVATE_RUNNER_CONTRACT = "openmle_fast_private_grader_runner_v1"
PRIVATE_WORKER_REQUEST_SCHEMA = "openmle_fast_private_worker_request_v1"
PRIVATE_WORKER_RESULT_SCHEMA = "openmle_fast_private_worker_result_v1"
PRIVATE_RUNNER_COMPLETION_GRACE_MS = 250


class PrivateGraderRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrivateGraderLimits:
    cpu_vcpus: int
    memory_bytes: int
    max_processes: int
    wall_ms: int
    input_bytes: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("private grader limits must be positive integers")

    @classmethod
    def frozen_v1(cls) -> PrivateGraderLimits:
        return cls(
            cpu_vcpus=1,
            memory_bytes=2 * 1024**3,
            max_processes=32,
            wall_ms=4_000,
            input_bytes=64 * 1024**2,
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateGradeExecutionRequest:
    task_id: str
    grader_binding_sha256: str
    package_identity_sha256: str
    metric_sha256: str
    answer_sha256: str
    higher_is_better: bool
    validator_success_forms: tuple[str, ...]
    metric: bytes
    answer: bytes
    submission: bytes


@dataclass(frozen=True)
class PrivateGradeExecution:
    classification: str
    native_score: float | None
    higher_is_better: bool


class PrivateGraderBackend(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def grade(
        self,
        request: PrivateGradeExecutionRequest,
        *,
        timeout_ms: int,
    ) -> PrivateGradeExecution: ...


class LocalCPUPrivateGraderBackend:
    """Fresh host worker for unit tests only; never formal-runtime eligible."""

    def __init__(self, limits: PrivateGraderLimits) -> None:
        self.limits = limits

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "contract": "openmle_fast_local_private_worker_v1",
            "formal_eligible": False,
            "fresh_worker_per_grade": True,
            "service_environment_forwarded": False,
            "selected_task_staging": True,
            "namespace_coverage": "none",
            "cgroup_coverage": "none",
            "network_isolation": "not_implemented",
            "resource_limits": self.limits.as_dict(),
        }

    def grade(
        self,
        request: PrivateGradeExecutionRequest,
        *,
        timeout_ms: int,
    ) -> PrivateGradeExecution:
        deadline = MonotonicDeadline.after_ms(timeout_ms)
        try:
            return self._grade_before_deadline(request, deadline)
        except DeadlineExceeded:
            return _infrastructure_fault(request.higher_is_better)

    def _grade_before_deadline(
        self,
        request: PrivateGradeExecutionRequest,
        deadline: MonotonicDeadline,
    ) -> PrivateGradeExecution:
        deadline.check()
        if (
            len(request.submission) > self.limits.input_bytes
            or _bytes_sha256(request.metric, deadline=deadline) != request.metric_sha256
            or _bytes_sha256(request.answer, deadline=deadline) != request.answer_sha256
        ):
            return _infrastructure_fault(request.higher_is_better)
        with tempfile.TemporaryDirectory(prefix="openmle-private-worker-") as raw_root:
            root = Path(raw_root)
            metric = root / "metric.py"
            answer = root / "answer.csv"
            submission = root / "submission.csv"
            temporary = root / "tmp"
            temporary.mkdir(mode=0o700)
            _write_private(metric, request.metric, deadline=deadline)
            _write_private(answer, request.answer, deadline=deadline)
            _write_private(submission, request.submission, deadline=deadline)
            worker_limits = self.limits.as_dict()
            remaining_ms = deadline.remaining_milliseconds()
            if remaining_ms <= PRIVATE_RUNNER_COMPLETION_GRACE_MS:
                raise DeadlineExceeded("no private-worker completion margin remains")
            worker_limits["wall_ms"] = min(
                worker_limits["wall_ms"],
                remaining_ms - PRIVATE_RUNNER_COMPLETION_GRACE_MS,
            )
            payload = {
                "schema": PRIVATE_WORKER_REQUEST_SCHEMA,
                "metric_path": str(metric),
                "answer_path": str(answer),
                "submission_path": str(submission),
                "higher_is_better": request.higher_is_better,
                "validator_success_forms": list(request.validator_success_forms),
                "resource_limits": worker_limits,
            }
            worker = Path(__file__).with_name("_private_grader_worker.py")
            environment = {
                "PATH": "/usr/bin:/bin",
                "HOME": "/nonexistent",
                "TMPDIR": str(temporary),
                "TZ": "UTC",
                "LC_ALL": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            try:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(worker)],
                    cwd=root,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    stdout, _ = process.communicate(
                        json.dumps(
                            payload, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8"),
                        timeout=min(
                            deadline.remaining_seconds(),
                            (
                                worker_limits["wall_ms"]
                                + PRIVATE_RUNNER_COMPLETION_GRACE_MS
                            )
                            / 1000.0,
                        ),
                    )
                except subprocess.TimeoutExpired:
                    _kill_process_group(process.pid)
                    process.communicate()
                    return _infrastructure_fault(request.higher_is_better)
                finally:
                    _kill_process_group(process.pid)
            except OSError:
                return _infrastructure_fault(request.higher_is_better)
            if process.returncode != 0 or len(stdout) > 64 * 1024:
                return _infrastructure_fault(request.higher_is_better)
            try:
                result = _decode_result(
                    _strict_json_loads(stdout), request.higher_is_better
                )
                deadline.check()
                return result
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                return _infrastructure_fault(request.higher_is_better)


class ExternalPrivateGraderRunnerBackend:
    """Adapter for the independently installed exact private worker sandbox."""

    def __init__(
        self,
        *,
        runner_path: Path | str,
        expected_runner_sha256: str,
        expected_runtime_digest: str,
        limits: PrivateGraderLimits,
    ) -> None:
        self.runner_path = Path(runner_path).expanduser().absolute()
        if self.runner_path.is_symlink() or not self.runner_path.is_file():
            raise PrivateGraderRunnerError("private runner must be a real file")
        if not os.access(self.runner_path, os.X_OK):
            raise PrivateGraderRunnerError("private runner is not executable")
        if _file_sha256(self.runner_path) != expected_runner_sha256:
            raise PrivateGraderRunnerError("private runner SHA256 mismatch")
        self.expected_runtime_digest = expected_runtime_digest
        self.limits = limits
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
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            value = _strict_json_loads(result.stdout)
        except (
            OSError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise PrivateGraderRunnerError(
                "cannot attest private runner metadata"
            ) from exc
        required_true = (
            "formal_eligible",
            "fresh_worker_per_grade",
            "selected_task_only_mounts",
            "submission_passed_by_fd",
            "result_sanitized_ipc",
            "service_environment_hidden",
            "network_namespace",
            "network_no_egress",
            "dns_disabled",
            "metadata_service_blocked",
            "external_unix_sockets_blocked",
            "pid_namespace",
            "ipc_namespace",
            "mount_namespace",
            "cgroup_v2_cpu",
            "cgroup_v2_memory",
            "cgroup_v2_pids",
            "fresh_unprivileged_uid_gid",
            "capabilities_dropped",
            "no_new_privs",
            "seccomp",
            "read_only_rootfs",
            "isolated_proc",
            "minimal_devices",
            "gpu_devices_absent",
            "core_dumps_disabled",
            "hard_wall_supervision",
            "descendant_kill_reap",
            "worker_teardown_verified",
            "validate_submission_once",
            "evaluate_once_after_validation",
        )
        if (
            result.returncode != 0
            or not isinstance(value, dict)
            or value.get("contract") != PRIVATE_RUNNER_CONTRACT
            or value.get("runtime_digest") != self.expected_runtime_digest
            or value.get("resource_limits") != self.limits.as_dict()
            or any(value.get(key) is not True for key in required_true)
        ):
            raise PrivateGraderRunnerError("private runner attestation is incomplete")
        return value

    def grade(
        self,
        request: PrivateGradeExecutionRequest,
        *,
        timeout_ms: int,
    ) -> PrivateGradeExecution:
        deadline = MonotonicDeadline.after_ms(timeout_ms)
        try:
            return self._grade_before_deadline(request, deadline)
        except DeadlineExceeded:
            return _infrastructure_fault(request.higher_is_better)

    def _grade_before_deadline(
        self,
        request: PrivateGradeExecutionRequest,
        deadline: MonotonicDeadline,
    ) -> PrivateGradeExecution:
        deadline.check()
        with tempfile.TemporaryDirectory(prefix="openmle-private-runner-input-") as raw:
            root = Path(raw)
            paths = (
                root / "metric.py",
                root / "answer.csv",
                root / "submission.csv",
            )
            for path, content in zip(
                paths,
                (request.metric, request.answer, request.submission),
                strict=True,
            ):
                _write_private(path, content, deadline=deadline)
            remaining_ms = deadline.remaining_milliseconds()
            if remaining_ms <= PRIVATE_RUNNER_COMPLETION_GRACE_MS:
                raise DeadlineExceeded("no private-runner completion margin remains")
            runner_timeout_ms = min(
                self.limits.wall_ms,
                remaining_ms - PRIVATE_RUNNER_COMPLETION_GRACE_MS,
            )
            descriptors = tuple(os.open(path, os.O_RDONLY) for path in paths)
            payload = {
                "schema": "openmle_fast_private_runner_request_v1",
                "task_id": request.task_id,
                "grader_binding_sha256": request.grader_binding_sha256,
                "package_identity_sha256": request.package_identity_sha256,
                "metric_sha256": request.metric_sha256,
                "answer_sha256": request.answer_sha256,
                "submission_sha256": _bytes_sha256(
                    request.submission, deadline=deadline
                ),
                "higher_is_better": request.higher_is_better,
                "validator_success_forms": list(request.validator_success_forms),
                "metric_fd": descriptors[0],
                "answer_fd": descriptors[1],
                "submission_fd": descriptors[2],
                "timeout_ms": runner_timeout_ms,
            }
            try:
                result = subprocess.run(
                    [str(self.runner_path), "grade"],
                    input=json.dumps(
                        payload, sort_keys=True, separators=(",", ":")
                    ).encode(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=min(
                        deadline.remaining_seconds(),
                        (
                            runner_timeout_ms + PRIVATE_RUNNER_COMPLETION_GRACE_MS
                        )
                        / 1000.0,
                    ),
                    check=False,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                    pass_fds=descriptors,
                )
                if result.returncode != 0 or len(result.stdout) > 64 * 1024:
                    raise ValueError("private runner failed")
                decoded = _decode_result(
                    _strict_json_loads(result.stdout), request.higher_is_better
                )
                deadline.check()
                return decoded
            except (
                OSError,
                subprocess.TimeoutExpired,
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
                TypeError,
            ):
                return _infrastructure_fault(request.higher_is_better)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)


def _decode_result(value: Any, expected_direction: bool) -> PrivateGradeExecution:
    required = {"schema", "classification", "native_score", "higher_is_better"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("private worker result has unexpected fields")
    if value["schema"] != PRIVATE_WORKER_RESULT_SCHEMA:
        raise ValueError("private worker result schema mismatch")
    classification = value["classification"]
    if classification not in {
        "graded",
        "invalid_submission",
        "infrastructure_fault",
    }:
        raise ValueError("private worker result classification is invalid")
    direction = value["higher_is_better"]
    if type(direction) is not bool or direction != expected_direction:
        raise ValueError("private worker direction drifted")
    score_value = value["native_score"]
    score: float | None
    if classification == "graded":
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise ValueError("private worker score is invalid")
        score = float(score_value)
        if not math.isfinite(score):
            raise ValueError("private worker score is not finite")
    else:
        if score_value is not None:
            raise ValueError("ungraded private worker result carried a score")
        score = None
    return PrivateGradeExecution(
        classification=classification,
        native_score=score,
        higher_is_better=direction,
    )


def _infrastructure_fault(direction: bool) -> PrivateGradeExecution:
    return PrivateGradeExecution(
        classification="infrastructure_fault",
        native_score=None,
        higher_is_better=direction,
    )


def _write_private(
    path: Path,
    payload: bytes,
    *,
    deadline: MonotonicDeadline | None = None,
) -> None:
    if deadline is not None:
        deadline.check()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            if deadline is not None:
                deadline.check()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if deadline is not None:
        deadline.check()


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes, *, deadline: MonotonicDeadline) -> str:
    digest = hashlib.sha256()
    view = memoryview(payload)
    for offset in range(0, len(view), 1024 * 1024):
        deadline.check()
        digest.update(view[offset : offset + 1024 * 1024])
    deadline.check()
    return digest.hexdigest()


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("private runner JSON contains a duplicate key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)
