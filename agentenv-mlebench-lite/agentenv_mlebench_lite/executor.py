from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .resources import (
    RESOURCE_USAGE_KEYS,
    resource_contract_sha256,
    validate_resource_usage,
    zero_resource_usage,
)
from .resources import (
    validate_resource_contract as _validate_resource_contract,
)
from .workspace import EXTERNAL_MEMORY_PATH, MODE_AMG_MEMORY, EpisodeWorkspace

ATTESTATION_SCHEMA = "mlebench_lite_sandbox_attestation_v3"
EXECUTION_SCHEMA = "mlebench_lite_sandbox_execution_v3"
FREEZE_SCHEMA = "mlebench_lite_sandbox_freeze_v2"
TEARDOWN_SCHEMA = "mlebench_lite_sandbox_teardown_v2"
EXTERNAL_MEMORY_ACCESS_SCHEMA = "amg_external_memory_access_v1"


class MLEBenchLiteExecutorError(RuntimeError):
    """Formal isolation could not be proved; execution must stop."""


@dataclass(frozen=True)
class BackendExecution:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    receipt: Mapping[str, Any]


class SandboxBackend(Protocol):
    formal_isolation: bool

    def attest(self, workspace: EpisodeWorkspace) -> Mapping[str, Any]: ...

    def execute(
        self,
        *,
        workspace: EpisodeWorkspace,
        command: str,
        timeout_ms: int,
        operation_id: str,
    ) -> BackendExecution: ...

    def freeze_and_reap(
        self,
        *,
        workspace: EpisodeWorkspace,
        operation_id: str,
    ) -> Mapping[str, Any]: ...

    def teardown(
        self,
        *,
        workspace: EpisodeWorkspace,
        operation_id: str,
    ) -> Mapping[str, Any]: ...


class SandboxExecutor:
    """Validate one attested sandbox and its monotonic operation receipts."""

    def __init__(
        self,
        backend: SandboxBackend,
        *,
        expected_runner_sha256: str,
        expected_runtime_digest: str,
        expected_resource_contract_sha256: str | None = None,
    ) -> None:
        _require_sha256(expected_runner_sha256, "runner SHA256")
        _require_sha256(expected_runtime_digest, "runtime digest")
        if expected_resource_contract_sha256 is not None:
            _require_sha256(
                expected_resource_contract_sha256,
                "resource contract SHA256",
            )
        self.backend = backend
        self.expected_runner_sha256 = expected_runner_sha256
        self.expected_runtime_digest = expected_runtime_digest
        self.expected_resource_contract_sha256 = expected_resource_contract_sha256
        self._attestations: dict[str, str] = {}
        self._resource_usage: dict[str, dict[str, int]] = {}
        self._freeze_operations: dict[str, str] = {}
        self._teardown_operations: dict[str, str] = {}
        self._freeze_receipts: dict[str, Mapping[str, Any]] = {}
        self._teardown_receipts: dict[str, Mapping[str, Any]] = {}

    def preflight(self, workspace: EpisodeWorkspace) -> str:
        if getattr(self.backend, "formal_isolation", None) is not True:
            raise MLEBenchLiteExecutorError(
                "formal execution requires an attested sandbox backend"
            )
        self._validate_workspace_contract(workspace)
        cached = self._attestations.get(workspace.episode_id)
        if cached is not None:
            return cached
        expected = self._expected_attestation(workspace)
        try:
            attestation = self.backend.attest(workspace)
        except Exception as exc:
            raise MLEBenchLiteExecutorError(
                "cannot obtain sandbox attestation"
            ) from exc
        if not isinstance(attestation, Mapping) or not _strict_equal(
            dict(attestation), expected
        ):
            raise MLEBenchLiteExecutorError("sandbox attestation drifted")
        digest = _canonical_sha256(expected)
        self._attestations[workspace.episode_id] = digest
        self._resource_usage[workspace.episode_id] = zero_resource_usage()
        return digest

    def validate_resource_contract(
        self,
        contract: Mapping[str, Any],
        contract_sha256: str,
    ) -> None:
        try:
            canonical = _validate_resource_contract(contract)
            computed_sha256 = resource_contract_sha256(canonical)
        except ValueError as exc:
            raise MLEBenchLiteExecutorError("resource contract is invalid") from exc
        _require_sha256(contract_sha256, "resource contract SHA256")
        if computed_sha256 != contract_sha256:
            raise MLEBenchLiteExecutorError("resource contract SHA256 mismatch")
        if (
            self.expected_resource_contract_sha256 is not None
            and contract_sha256 != self.expected_resource_contract_sha256
        ):
            raise MLEBenchLiteExecutorError("resource contract SHA256 mismatch")

    def run(
        self,
        workspace: EpisodeWorkspace,
        command: str,
        *,
        timeout_ms: int,
        operation_id: str | None = None,
    ) -> BackendExecution:
        self._validate_workspace_contract(workspace)
        if workspace.episode_id in self._freeze_operations:
            raise MLEBenchLiteExecutorError("sandbox workspace is already frozen")
        attestation_digest = self._cached_attestation(workspace)
        operation = _canonical_uuid(operation_id or str(uuid.uuid4()))
        try:
            result = self.backend.execute(
                workspace=workspace,
                command=command,
                timeout_ms=timeout_ms,
                operation_id=operation,
            )
        except Exception as exc:
            raise MLEBenchLiteExecutorError("sandbox execution failed") from exc
        cumulative = self._validate_execution_receipt(
            result,
            workspace=workspace,
            command=command,
            timeout_ms=timeout_ms,
            operation_id=operation,
            attestation_digest=attestation_digest,
            prior=self.resource_usage(workspace),
            resource_contract_sha256=workspace.resource_contract_sha256,
        )
        self._resource_usage[workspace.episode_id] = cumulative
        return result

    def freeze_and_reap(self, workspace: EpisodeWorkspace) -> Mapping[str, Any]:
        self._validate_workspace_contract(workspace)
        cached = self._freeze_receipts.get(workspace.episode_id)
        if cached is not None:
            return cached
        attestation_digest = self._cached_attestation(workspace)
        operation = self._freeze_operations.setdefault(
            workspace.episode_id, str(uuid.uuid4())
        )
        try:
            receipt = self.backend.freeze_and_reap(
                workspace=workspace,
                operation_id=operation,
            )
        except Exception as exc:
            raise MLEBenchLiteExecutorError("sandbox freeze/reap failed") from exc
        expected = {
            "schema": FREEZE_SCHEMA,
            "operation_id": operation,
            "runner_sha256": self.expected_runner_sha256,
            "runtime_digest": self.expected_runtime_digest,
            "resource_contract_sha256": workspace.resource_contract_sha256,
            "mount_attestation_sha256": attestation_digest,
            "resource_cumulative": self.resource_usage(workspace),
            "processes_reaped": True,
            "workspace_frozen": True,
            "descendant_process_count": 0,
        }
        if not isinstance(receipt, Mapping) or not _strict_equal(
            dict(receipt), expected
        ):
            raise MLEBenchLiteExecutorError("sandbox freeze/reap receipt drifted")
        self._freeze_receipts[workspace.episode_id] = dict(receipt)
        return self._freeze_receipts[workspace.episode_id]

    def teardown(self, workspace: EpisodeWorkspace) -> Mapping[str, Any]:
        cached = self._teardown_receipts.get(workspace.episode_id)
        if cached is not None:
            return cached
        self._validate_workspace_contract(workspace)
        attestation_digest = self._attestations.get(
            workspace.episode_id,
            _canonical_sha256(self._expected_attestation(workspace)),
        )
        operation = self._teardown_operations.setdefault(
            workspace.episode_id, str(uuid.uuid4())
        )
        cumulative = self._resource_usage.get(
            workspace.episode_id, zero_resource_usage()
        )
        try:
            receipt = self.backend.teardown(
                workspace=workspace,
                operation_id=operation,
            )
        except Exception as exc:
            raise MLEBenchLiteExecutorError("sandbox teardown failed") from exc
        expected = {
            "schema": TEARDOWN_SCHEMA,
            "operation_id": operation,
            "runner_sha256": self.expected_runner_sha256,
            "runtime_digest": self.expected_runtime_digest,
            "resource_contract_sha256": workspace.resource_contract_sha256,
            "mount_attestation_sha256": attestation_digest,
            "resource_cumulative": cumulative,
            "processes_reaped": True,
            "mounts_released": True,
            "descendant_process_count": 0,
            "mount_count": 0,
            "sandbox_present": False,
        }
        if not isinstance(receipt, Mapping) or not _strict_equal(
            dict(receipt), expected
        ):
            raise MLEBenchLiteExecutorError("sandbox teardown receipt drifted")
        self._teardown_receipts[workspace.episode_id] = dict(receipt)
        self._freeze_operations.setdefault(workspace.episode_id, operation)
        return self._teardown_receipts[workspace.episode_id]

    def resource_usage(self, workspace: EpisodeWorkspace) -> dict[str, int]:
        return dict(
            self._resource_usage.get(workspace.episode_id, zero_resource_usage())
        )

    def _cached_attestation(self, workspace: EpisodeWorkspace) -> str:
        try:
            return self._attestations[workspace.episode_id]
        except KeyError as exc:
            raise MLEBenchLiteExecutorError(
                "sandbox preflight has not completed"
            ) from exc

    def _validate_workspace_contract(self, workspace: EpisodeWorkspace) -> None:
        if (workspace.mode == MODE_AMG_MEMORY) != (workspace.memory_root is not None):
            raise MLEBenchLiteExecutorError(
                "external-memory workspace contract drifted"
            )
        self.validate_resource_contract(
            workspace.resource_contract,
            workspace.resource_contract_sha256,
        )

    def _expected_attestation(self, workspace: EpisodeWorkspace) -> dict[str, Any]:
        mounts = [
            {
                "source": str(workspace.public_root),
                "target": "/home/data",
                "read_only": True,
                "source_tree_sha256": workspace.public_tree_sha256,
            },
            {
                "source": str(workspace.workspace_root),
                "target": "/home/workspace",
                "read_only": False,
            },
            {
                "source": str(workspace.submission_root),
                "target": "/home/submission",
                "read_only": False,
            },
        ]
        if workspace.memory_root is not None:
            mounts.append(
                {
                    "source": str(workspace.memory_root),
                    "target": EXTERNAL_MEMORY_PATH,
                    "read_only": False,
                }
            )
        return {
            "schema": ATTESTATION_SCHEMA,
            "runner_sha256": self.expected_runner_sha256,
            "runtime_digest": self.expected_runtime_digest,
            "resource_contract": dict(workspace.resource_contract),
            "resource_contract_sha256": workspace.resource_contract_sha256,
            "mount_namespace": True,
            "network_disabled": True,
            "non_root": True,
            "read_only_rootfs": True,
            "execution_scope": {
                "scope": "episode_cgroup_descendants",
                "cgroup_enforced": True,
                "isolated_process_group": True,
            },
            "external_memory_isolation": {
                "sandbox_access": (
                    "read_write_mount_v1"
                    if workspace.mode == MODE_AMG_MEMORY
                    else "none"
                ),
                "native_tool_surface": "inspect_edit_shell_v1",
                "private_root_state": (
                    "allocated"
                    if workspace.mode == MODE_AMG_MEMORY
                    else "absent"
                ),
            },
            "mounts": mounts,
            "denied_mount_prefixes": ["/host", "/private"],
        }

    def _validate_execution_receipt(
        self,
        result: Any,
        *,
        workspace: EpisodeWorkspace,
        command: str,
        timeout_ms: int,
        operation_id: str,
        attestation_digest: str,
        prior: Mapping[str, int],
        resource_contract_sha256: str,
    ) -> dict[str, int]:
        if (
            type(getattr(result, "returncode", None)) is not int
            or type(getattr(result, "timed_out", None)) is not bool
            or not isinstance(getattr(result, "stdout", None), str)
            or not isinstance(getattr(result, "stderr", None), str)
        ):
            raise MLEBenchLiteExecutorError("sandbox execution result drifted")
        receipt = getattr(result, "receipt", None)
        if not isinstance(receipt, Mapping):
            raise MLEBenchLiteExecutorError("sandbox execution receipt drifted")
        try:
            delta = validate_resource_usage(
                receipt.get("resource_delta"), label="resource delta"
            )
            cumulative = validate_resource_usage(
                receipt.get("resource_cumulative"), label="resource cumulative"
            )
        except ValueError as exc:
            raise MLEBenchLiteExecutorError(
                "sandbox execution resource receipt drifted"
            ) from exc
        expected_cumulative = {
            key: int(prior[key]) + delta[key] for key in RESOURCE_USAGE_KEYS
        }
        if not _strict_equal(cumulative, expected_cumulative):
            raise MLEBenchLiteExecutorError("sandbox resource ledger is not monotonic")
        expected = {
            "schema": EXECUTION_SCHEMA,
            "operation_id": operation_id,
            "runner_sha256": self.expected_runner_sha256,
            "runtime_digest": self.expected_runtime_digest,
            "resource_contract_sha256": resource_contract_sha256,
            "mount_attestation_sha256": attestation_digest,
            "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "timeout_ms": timeout_ms,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "resource_delta": delta,
            "resource_cumulative": cumulative,
            "containment": {
                "scope": "episode_cgroup_descendants",
                "cgroup_enforced": True,
                "isolated_process_group": True,
                "descendant_process_count": 0,
            },
        }
        access = receipt.get("external_memory_access")
        if access is not None:
            if (
                workspace.mode != MODE_AMG_MEMORY
                or not isinstance(access, Mapping)
                or set(access) != {"schema", "operation"}
                or access.get("schema") != EXTERNAL_MEMORY_ACCESS_SCHEMA
                or access.get("operation") not in {"read", "write", "read_write"}
            ):
                raise MLEBenchLiteExecutorError(
                    "external-memory access receipt drifted"
                )
            expected["external_memory_access"] = dict(access)
        if not _strict_equal(dict(receipt), expected):
            raise MLEBenchLiteExecutorError("sandbox execution receipt drifted")
        return cumulative


class ExternalSandboxRunnerBackend:
    """Protocol adapter that executes a verified open runner file descriptor."""

    formal_isolation = True

    def __init__(
        self,
        runner_path: Path,
        *,
        expected_runner_sha256: str,
        expected_runtime_digest: str,
        expected_runner_uid: int,
        protocol_timeout_seconds: float = 10.0,
    ) -> None:
        _require_sha256(expected_runner_sha256, "runner SHA256")
        _require_sha256(expected_runtime_digest, "runtime digest")
        if type(expected_runner_uid) is not int or expected_runner_uid < 0:
            raise MLEBenchLiteExecutorError("runner UID must be a nonnegative integer")
        self.runner_path = Path(runner_path)
        self.expected_runner_sha256 = expected_runner_sha256
        self.expected_runtime_digest = expected_runtime_digest
        self.expected_runner_uid = expected_runner_uid
        self.protocol_timeout_seconds = protocol_timeout_seconds
        descriptor = self._open_verified_runner()
        os.close(descriptor)

    def attest(self, workspace: EpisodeWorkspace) -> Mapping[str, Any]:
        return self._invoke("attest", _workspace_request(workspace))

    def execute(
        self,
        *,
        workspace: EpisodeWorkspace,
        command: str,
        timeout_ms: int,
        operation_id: str,
    ) -> BackendExecution:
        value = self._invoke(
            "execute",
            {
                **_workspace_request(workspace),
                "operation_id": operation_id,
                "command": command,
                "timeout_ms": timeout_ms,
            },
            timeout=max(self.protocol_timeout_seconds, timeout_ms / 1000.0 + 5.0),
        )
        if not isinstance(value, dict) or set(value) != {
            "returncode",
            "stdout",
            "stderr",
            "timed_out",
            "receipt",
        }:
            raise MLEBenchLiteExecutorError("sandbox runner execution response drifted")
        return BackendExecution(
            returncode=value["returncode"],
            stdout=value["stdout"],
            stderr=value["stderr"],
            timed_out=value["timed_out"],
            receipt=value["receipt"],
        )

    def freeze_and_reap(
        self,
        *,
        workspace: EpisodeWorkspace,
        operation_id: str,
    ) -> Mapping[str, Any]:
        return self._invoke(
            "freeze",
            {**_workspace_request(workspace), "operation_id": operation_id},
        )

    def teardown(
        self,
        *,
        workspace: EpisodeWorkspace,
        operation_id: str,
    ) -> Mapping[str, Any]:
        return self._invoke(
            "teardown",
            {**_workspace_request(workspace), "operation_id": operation_id},
        )

    def _invoke(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        descriptor = self._open_verified_runner()
        executable = _descriptor_path(descriptor)
        try:
            completed = subprocess.run(
                [executable, operation],
                input=_canonical_bytes(payload),
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=self.protocol_timeout_seconds if timeout is None else timeout,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                close_fds=True,
                pass_fds=(descriptor,),
                start_new_session=True,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MLEBenchLiteExecutorError("sandbox runner protocol failed") from exc
        finally:
            os.close(descriptor)
        if completed.returncode != 0 or len(completed.stdout) > 2_000_000:
            raise MLEBenchLiteExecutorError("sandbox runner rejected the request")
        try:
            return _strict_json_loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise MLEBenchLiteExecutorError(
                "sandbox runner returned invalid JSON"
            ) from exc

    def _open_verified_runner(self) -> int:
        parent_descriptor: int | None = None
        descriptor: int | None = None
        try:
            parent_descriptor = os.open(
                self.runner_path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            parent_metadata = os.fstat(parent_descriptor)
            if (
                not stat.S_ISDIR(parent_metadata.st_mode)
                or parent_metadata.st_uid != self.expected_runner_uid
                or parent_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise MLEBenchLiteExecutorError(
                    "sandbox runner parent identity is unsafe"
                )
            descriptor = os.open(
                self.runner_path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not metadata.st_mode & stat.S_IXUSR
                or metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                or metadata.st_uid != self.expected_runner_uid
                or _file_descriptor_sha256(descriptor) != self.expected_runner_sha256
            ):
                raise MLEBenchLiteExecutorError("sandbox runner identity is unsafe")
            os.lseek(descriptor, 0, os.SEEK_SET)
            return descriptor
        except MLEBenchLiteExecutorError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise MLEBenchLiteExecutorError("sandbox runner is unavailable") from exc
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)


def _workspace_request(workspace: EpisodeWorkspace) -> dict[str, Any]:
    try:
        contract = _validate_resource_contract(workspace.resource_contract)
        contract_sha256 = resource_contract_sha256(contract)
    except ValueError as exc:
        raise MLEBenchLiteExecutorError("resource contract is invalid") from exc
    if contract_sha256 != workspace.resource_contract_sha256:
        raise MLEBenchLiteExecutorError("resource contract SHA256 mismatch")
    memory_enabled = workspace.mode == MODE_AMG_MEMORY
    if memory_enabled != (workspace.memory_root is not None):
        raise MLEBenchLiteExecutorError("external-memory workspace contract drifted")
    request = {
        "schema": "mlebench_lite_sandbox_request_v3",
        "episode_id": workspace.episode_id,
        "competition_id": workspace.competition_id,
        "mode": workspace.mode,
        "resource_contract": contract,
        "resource_contract_sha256": workspace.resource_contract_sha256,
        "public_root": str(workspace.public_root),
        "public_tree_sha256": workspace.public_tree_sha256,
        "workspace_root": str(workspace.workspace_root),
        "submission_root": str(workspace.submission_root),
    }
    if workspace.memory_root is not None:
        request["external_memory_root"] = str(workspace.memory_root)
    return request


def _descriptor_path(descriptor: int) -> str:
    proc = Path(f"/proc/self/fd/{descriptor}")
    if proc.exists():
        return str(proc)
    return f"/dev/fd/{descriptor}"


def _file_descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _strict_json_loads(payload: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
    )


def _canonical_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise MLEBenchLiteExecutorError("operation id must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise MLEBenchLiteExecutorError("operation id must be a UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise MLEBenchLiteExecutorError("operation id must be a canonical UUID4")
    return value


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MLEBenchLiteExecutorError(f"{label} must be lowercase SHA256")
