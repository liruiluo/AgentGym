#!/usr/bin/python3
"""Strict MLE-bench Lite external-runner protocol engine.

The protocol layer deliberately knows nothing about competitions, grading, or
policy arms beyond the optional external-memory mount invariant.  Linux
isolation is supplied by :mod:`runtime_bridge.linux_runtime` and is injected
in unit tests so protocol claims never depend on the host running the tests.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import fcntl
import hashlib
import hmac
import importlib.util
import json
import os
import posixpath
import re
import stat
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

REQUEST_SCHEMA = "mlebench_lite_sandbox_request_v3"
RESOURCE_SCHEMA = "mlebench_lite_resource_contract_v2"
ATTESTATION_SCHEMA = "mlebench_lite_sandbox_attestation_v3"
EXECUTION_SCHEMA = "mlebench_lite_sandbox_execution_v3"
FREEZE_SCHEMA = "mlebench_lite_sandbox_freeze_v2"
TEARDOWN_SCHEMA = "mlebench_lite_sandbox_teardown_v2"
EXTERNAL_MEMORY_ACCESS_SCHEMA = "amg_external_memory_access_v1"

MODES = frozenset(("native", "amg_compaction_only", "amg_memory"))
MEMORY_MODE = "amg_memory"
RESOURCE_USAGE_KEYS = (
    "execution_time_ms",
    "cpu_time_ms",
    "writable_bytes",
    "writable_inodes",
    "processes_started",
)
FIXED_RESOURCE_VALUES = {
    "cpu_limit_cores": 36,
    "memory_limit_bytes": 440_000_000_000,
    "pids_limit": 4096,
    "writable_bytes_limit": 500_000_000_000,
    "writable_inodes_limit": 2_000_000,
    "gpu_count": 1,
}
_NUMERIC_RESOURCE_FIELDS = frozenset(
    (
        "max_actions",
        "max_submission_bytes",
        "max_shell_timeout_ms",
        "max_visible_output_bytes",
        "episode_timeout_ms",
        "max_total_execution_ms",
        "cpu_limit_cores",
        "memory_limit_bytes",
        "pids_limit",
        "writable_bytes_limit",
        "writable_inodes_limit",
        "gpu_count",
        "max_step_response_ms",
    )
)
_RESOURCE_FIELDS = frozenset(
    {
        "schema",
        *_NUMERIC_RESOURCE_FIELDS,
        "submission_path",
        "network_disabled",
        "read_only_public_data",
        "process_scope",
        "cgroup_required",
        "isolated_process_group_required",
    }
)
_BASE_FIELDS = frozenset(
    (
        "schema",
        "episode_id",
        "competition_id",
        "mode",
        "resource_contract",
        "resource_contract_sha256",
        "public_root",
        "public_tree_sha256",
        "workspace_root",
        "submission_root",
    )
)
_EPISODE_ID = re.compile(r"^[0-9a-f]{32}$")
_COMPETITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_GPU_UUID = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
OPENMLE_V7_ARTIFACT_LOCK_SHA256 = (
    "f04f269d39f66c025d70620f41016fb3a555fb175b9feb6c8977fed6debae1f6"
)
OPENMLE_V7_SUPERVISOR_SHA256 = (
    "25a93be7ec835df83c2100bede5743c66dee18246cd32aa44ffaa67f8c625032"
)
RUNTIME_AUDIT_RELATIVE_PATH = "lib/mlebench-lite-runtime-audit.so"
ROOTFS_ELF_CLOSURE_RELATIVE_PATH = "rootfs-elf-closure.json"
BUNDLE_ROOT_FD = 197


class BridgeError(RuntimeError):
    """A request or runtime proof failed and must not produce a success receipt."""


@dataclass(frozen=True)
class BridgeIdentity:
    runner_sha256: str
    runtime_digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.runner_sha256, "runner SHA256")
        _require_sha256(self.runtime_digest, "runtime digest")


@dataclass(frozen=True)
class BundleIdentity:
    identity: BridgeIdentity
    bundle_root: str
    deployment: dict[str, Any]
    supervisor_path: str
    expected_uid: int
    bundle_fd: int | None = None


@dataclass(frozen=True)
class RuntimeAttestation:
    cpu_limit_cores: int
    memory_limit_bytes: int
    pids_limit: int
    gpu_count: int
    gpu_uuid: str
    mount_namespace: bool
    network_disabled: bool
    non_root: bool
    read_only_rootfs: bool
    runtime_identity: dict[str, Any]


@dataclass(frozen=True)
class ExecutionOutcome:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    execution_time_ms: int
    cpu_time_ms: int
    writable_bytes: int
    writable_inodes: int
    processes_started: int
    descendant_process_count: int
    external_memory_access: str | None = None


@dataclass(frozen=True)
class LifecycleOutcome:
    processes_reaped: bool
    workspace_frozen: bool
    mounts_released: bool
    descendant_process_count: int
    mount_count: int
    sandbox_present: bool


class RuntimeFacade(Protocol):
    def attest(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any] | None = None,
    ) -> RuntimeAttestation: ...

    def execute(
        self, request: Mapping[str, Any], state: Mapping[str, Any]
    ) -> ExecutionOutcome: ...

    def freeze(
        self, request: Mapping[str, Any], state: Mapping[str, Any]
    ) -> LifecycleOutcome: ...

    def teardown(
        self, request: Mapping[str, Any], state: Mapping[str, Any]
    ) -> LifecycleOutcome: ...

    def reconcile(
        self,
        operation: str,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> LifecycleOutcome: ...


class StateStore(Protocol):
    def get(self, episode_id: str) -> dict[str, Any] | None: ...

    def set(self, episode_id: str, state: Mapping[str, Any]) -> None: ...


class MemoryStateStore:
    """In-memory store used only by protocol tests."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def get(self, episode_id: str) -> dict[str, Any] | None:
        state = self._states.get(episode_id)
        return None if state is None else copy.deepcopy(state)

    def set(self, episode_id: str, state: Mapping[str, Any]) -> None:
        self._states[episode_id] = copy.deepcopy(dict(state))


class SealedFileStateStore:
    """Owner-only HMAC state and per-episode interprocess serialization."""

    _KEY_NAME = "owner.key"
    _KEY_LOCK_NAME = "owner.key.lock"
    _MAX_STATE_BYTES = 8 * 1024 * 1024

    def __init__(self, root: os.PathLike[str] | str, *, expected_uid: int) -> None:
        if type(expected_uid) is not int or expected_uid < 0:
            raise BridgeError("state owner UID drifted")
        self.root = os.fspath(root)
        try:
            descriptor = os.open(
                self.root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise BridgeError("state root is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != expected_uid
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise BridgeError("state root identity is unsafe")
        except BaseException:
            os.close(descriptor)
            raise
        self._directory_fd = descriptor
        self.expected_uid = expected_uid
        self._key = self._load_or_create_key()

    def close(self) -> None:
        descriptor = getattr(self, "_directory_fd", None)
        if descriptor is not None:
            os.close(descriptor)
            self._directory_fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    @contextlib.contextmanager
    def episode_lock(self, episode_id: str):
        if not isinstance(episode_id, str) or not _EPISODE_ID.fullmatch(episode_id):
            raise BridgeError("episode id drifted")
        descriptor = self._open_owned_file(
            self._lock_name(episode_id),
            os.O_RDWR | os.O_CREAT,
            allow_create=True,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except OSError as exc:
            raise BridgeError("episode lock failed") from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def get(self, episode_id: str) -> dict[str, Any] | None:
        name = self._state_name(episode_id)
        try:
            descriptor = self._open_owned_file(name, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            payload = _read_bounded(descriptor, self._MAX_STATE_BYTES)
        finally:
            os.close(descriptor)
        try:
            envelope = strict_json_loads(payload)
        except ValueError as exc:
            raise BridgeError("episode state is not strict JSON") from exc
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"schema", "body", "hmac_sha256"}
            or envelope.get("schema") != "mlebench_lite_bridge_state_envelope_v1"
            or not isinstance(envelope.get("body"), dict)
        ):
            raise BridgeError("episode state envelope drifted")
        expected = hmac.new(
            self._key, canonical_json_bytes(envelope["body"]), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, str(envelope.get("hmac_sha256"))):
            raise BridgeError("episode state authentication failed")
        return copy.deepcopy(envelope["body"])

    def set(self, episode_id: str, state: Mapping[str, Any]) -> None:
        body = copy.deepcopy(dict(state))
        envelope = {
            "schema": "mlebench_lite_bridge_state_envelope_v1",
            "body": body,
            "hmac_sha256": hmac.new(
                self._key, canonical_json_bytes(body), hashlib.sha256
            ).hexdigest(),
        }
        payload = canonical_json_bytes(envelope)
        if len(payload) > self._MAX_STATE_BYTES:
            raise BridgeError("episode state exceeds byte cap")
        temporary = "tmp-" + uuid.uuid4().hex
        descriptor: int | None = None
        try:
            descriptor = self._open_owned_file(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                allow_create=True,
            )
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                temporary,
                self._state_name(episode_id),
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            os.fsync(self._directory_fd)
        except OSError as exc:
            raise BridgeError("cannot persist episode state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=self._directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def _load_or_create_key(self) -> bytes:
        lock_descriptor = self._open_owned_file(
            self._KEY_LOCK_NAME,
            os.O_RDWR | os.O_CREAT,
            allow_create=True,
        )
        temporary: str | None = None
        descriptor: int | None = None
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            try:
                descriptor = self._open_owned_file(self._KEY_NAME, os.O_RDONLY)
            except FileNotFoundError:
                temporary = "tmp-owner-key-" + uuid.uuid4().hex
                descriptor = self._open_owned_file(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    allow_create=True,
                )
                _write_all(descriptor, os.urandom(32))
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                os.replace(
                    temporary,
                    self._KEY_NAME,
                    src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd,
                )
                temporary = None
                os.fsync(self._directory_fd)
                descriptor = self._open_owned_file(self._KEY_NAME, os.O_RDONLY)
            key = _read_bounded(descriptor, 32)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=self._directory_fd)
                except FileNotFoundError:
                    pass
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        if len(key) != 32:
            raise BridgeError("state owner key drifted")
        return key

    def _open_owned_file(
        self,
        name: str,
        flags: int,
        *,
        allow_create: bool = False,
    ) -> int:
        if not name or "/" in name or name in {".", ".."}:
            raise BridgeError("state filename drifted")
        open_flags = flags | getattr(os, "O_NOFOLLOW", 0) | getattr(
            os, "O_CLOEXEC", 0
        )
        try:
            descriptor = os.open(
                name,
                open_flags,
                0o600,
                dir_fd=self._directory_fd,
            )
        except FileNotFoundError:
            raise
        except FileExistsError:
            raise
        except OSError as exc:
            raise BridgeError("state file is unavailable") from exc
        if allow_create:
            try:
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                os.close(descriptor)
                raise BridgeError("state file mode cannot be sealed") from exc
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != self.expected_uid
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise BridgeError("state file identity is unsafe")
        return descriptor

    @staticmethod
    def _state_name(episode_id: str) -> str:
        if not isinstance(episode_id, str) or not _EPISODE_ID.fullmatch(episode_id):
            raise BridgeError("episode id drifted")
        return "state-" + episode_id + ".json"

    @staticmethod
    def _lock_name(episode_id: str) -> str:
        if not isinstance(episode_id, str) or not _EPISODE_ID.fullmatch(episode_id):
            raise BridgeError("episode id drifted")
        return "lock-" + episode_id


class BridgeEngine:
    """Validate requests and translate runtime facts into exact MLE receipts."""

    def __init__(
        self,
        identity: BridgeIdentity,
        runtime: RuntimeFacade,
        store: StateStore,
    ) -> None:
        self.identity = identity
        self.runtime = runtime
        self.store = store
        self.bundle_identity_sha256 = canonical_sha256(
            {
                "runner_sha256": identity.runner_sha256,
                "runtime_digest": identity.runtime_digest,
            }
        )

    def handle(self, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        canonical = validate_request(operation, request)
        if operation == "attest":
            return self._attest(canonical)
        state = self._matching_state(canonical)
        if operation == "execute":
            return self._execute(canonical, state)
        if operation == "freeze":
            return self._freeze(canonical, state)
        if operation == "teardown":
            return self._teardown(canonical, state)
        raise BridgeError("unsupported operation")

    def _attest(self, request: dict[str, Any]) -> dict[str, Any]:
        attestation = expected_attestation(request, self.identity)
        base_sha256 = canonical_sha256(_base_request(request))
        state = self.store.get(request["episode_id"])
        if state is None:
            state = {
                "schema": "mlebench_lite_bridge_state_v3",
                "bundle_identity_sha256": self.bundle_identity_sha256,
                "base_sha256": base_sha256,
                "attestation": copy.deepcopy(attestation),
                "mount_attestation_sha256": canonical_sha256(attestation),
                "runtime_identity": None,
                "resource_cumulative": zero_resource_usage(),
                "lifecycle": "attesting",
                "operations": {},
            }
            # The pending attestation is durable before the persistent mount is
            # created.  A retry may only adopt the exact same mount identity.
            self.store.set(request["episode_id"], state)
        else:
            self._require_state_identity(state, base_sha256)
            if state["lifecycle"] == "torn_down":
                raise BridgeError("episode is already torn down")
            if state["lifecycle"] == "frozen":
                raise BridgeError("frozen episode cannot be re-attested")
            if state["attestation"] != attestation:
                raise BridgeError("attestation identity drifted")
        runtime_attestation = self.runtime.attest(request, copy.deepcopy(state))
        _validate_runtime_attestation(runtime_attestation, request["resource_contract"])
        if state["runtime_identity"] is not None and (
            state["runtime_identity"] != runtime_attestation.runtime_identity
        ):
            raise BridgeError("runtime attestation identity drifted")
        state["runtime_identity"] = copy.deepcopy(
            runtime_attestation.runtime_identity
        )
        state["lifecycle"] = "active"
        self.store.set(request["episode_id"], state)
        return copy.deepcopy(attestation)

    def _matching_state(self, request: Mapping[str, Any]) -> dict[str, Any]:
        state = self.store.get(request["episode_id"])
        if state is None:
            raise BridgeError("episode has not been attested")
        self._require_state_identity(
            state, canonical_sha256(_base_request(request))
        )
        return state

    def _require_state_identity(
        self, state: Mapping[str, Any], base_sha256: str
    ) -> None:
        if (
            set(state)
            != {
                "schema",
                "bundle_identity_sha256",
                "base_sha256",
                "attestation",
                "mount_attestation_sha256",
                "runtime_identity",
                "resource_cumulative",
                "lifecycle",
                "operations",
            }
            or state.get("schema") != "mlebench_lite_bridge_state_v3"
            or state.get("bundle_identity_sha256")
            != self.bundle_identity_sha256
            or state.get("base_sha256") != base_sha256
        ):
            raise BridgeError("episode state identity drifted")
        _validate_usage(state.get("resource_cumulative"), "resource cumulative")
        if state.get("lifecycle") not in {
            "attesting",
            "active",
            "frozen",
            "torn_down",
        }:
            raise BridgeError("episode lifecycle drifted")
        if not isinstance(state.get("attestation"), dict) or not isinstance(
            state.get("operations"), dict
        ):
            raise BridgeError("episode state shape drifted")
        if canonical_sha256(state["attestation"]) != state.get(
            "mount_attestation_sha256"
        ):
            raise BridgeError("mount attestation state drifted")
        runtime_identity = state.get("runtime_identity")
        if state.get("lifecycle") == "attesting":
            if runtime_identity is not None:
                raise BridgeError("pending attestation identity drifted")
        elif state.get("lifecycle") == "torn_down" and runtime_identity is None:
            pass
        elif (
            not isinstance(runtime_identity, dict)
            or not isinstance(runtime_identity.get("schema"), str)
            or not runtime_identity["schema"]
        ):
            raise BridgeError("runtime identity state drifted")
        for operation_id, entry in state["operations"].items():
            _canonical_uuid4(operation_id)
            if (
                not isinstance(entry, dict)
                or set(entry)
                != {"operation", "request_sha256", "status", "response"}
                or entry.get("operation") not in {"execute", "freeze", "teardown"}
                or entry.get("status") not in {"pending", "final"}
            ):
                raise BridgeError("operation ledger shape drifted")
            _require_sha256(entry.get("request_sha256"), "operation request SHA256")
            if entry["status"] == "pending" and entry["response"] is not None:
                raise BridgeError("pending operation contains a response")
            if entry["status"] == "final" and not isinstance(
                entry["response"], dict
            ):
                raise BridgeError("final operation response drifted")

    def _operation_status(
        self,
        operation: str,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        operation_id = request["operation_id"]
        cached = state["operations"].get(operation_id)
        if cached is None:
            return "new", None
        request_sha256 = canonical_sha256(request)
        if (
            not isinstance(cached, dict)
            or set(cached)
            != {"operation", "request_sha256", "status", "response"}
            or cached["operation"] != operation
            or cached["request_sha256"] != request_sha256
        ):
            raise BridgeError("operation replay identity drifted")
        if cached["status"] == "pending":
            if cached["response"] is not None:
                raise BridgeError("pending operation response drifted")
            return "pending", None
        if cached["status"] != "final" or not isinstance(
            cached["response"], dict
        ):
            raise BridgeError("operation replay state drifted")
        return "final", copy.deepcopy(cached["response"])

    def _begin_operation(
        self,
        operation: str,
        request: Mapping[str, Any],
        state: dict[str, Any],
    ) -> None:
        state["operations"][request["operation_id"]] = {
            "operation": operation,
            "request_sha256": canonical_sha256(request),
            "status": "pending",
            "response": None,
        }
        self.store.set(request["episode_id"], state)

    def _complete_operation(
        self,
        operation: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        state: dict[str, Any],
    ) -> None:
        entry = state["operations"].get(request["operation_id"])
        if (
            not isinstance(entry, dict)
            or entry.get("operation") != operation
            or entry.get("request_sha256") != canonical_sha256(request)
            or entry.get("status") != "pending"
            or entry.get("response") is not None
        ):
            raise BridgeError("pending operation state drifted")
        entry["status"] = "final"
        entry["response"] = copy.deepcopy(dict(response))
        self.store.set(request["episode_id"], state)

    def _execute(
        self, request: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        status, replay = self._operation_status("execute", request, state)
        if status == "final":
            assert replay is not None
            return replay
        if status == "pending":
            # Re-running a command after losing its durable response could
            # duplicate arbitrary policy side effects.  Only lifecycle cleanup
            # may reconcile this episode now.
            raise BridgeError("execute operation outcome is indeterminate")
        if state["lifecycle"] != "active":
            raise BridgeError("episode is not executable")
        if any(
            isinstance(entry, dict)
            and entry.get("operation") == "execute"
            and entry.get("status") == "pending"
            for entry in state["operations"].values()
        ):
            raise BridgeError("episode contains an indeterminate execute")
        self._begin_operation("execute", request, state)
        outcome = self.runtime.execute(request, copy.deepcopy(state))
        _validate_execution_outcome(outcome, request)
        additive_delta = {
            "execution_time_ms": outcome.execution_time_ms,
            "cpu_time_ms": outcome.cpu_time_ms,
            "processes_started": outcome.processes_started,
        }
        prior = _validate_usage(state["resource_cumulative"], "resource cumulative")
        delta = {
            **additive_delta,
            "writable_bytes": max(0, outcome.writable_bytes - prior["writable_bytes"]),
            "writable_inodes": max(
                0, outcome.writable_inodes - prior["writable_inodes"]
            ),
        }
        cumulative = {
            "execution_time_ms": prior["execution_time_ms"]
            + delta["execution_time_ms"],
            "cpu_time_ms": prior["cpu_time_ms"] + delta["cpu_time_ms"],
            "writable_bytes": max(prior["writable_bytes"], outcome.writable_bytes),
            "writable_inodes": max(
                prior["writable_inodes"], outcome.writable_inodes
            ),
            "processes_started": prior["processes_started"]
            + delta["processes_started"],
        }
        contract = request["resource_contract"]
        if (
            cumulative["execution_time_ms"] > contract["max_total_execution_ms"]
            or cumulative["writable_bytes"] > contract["writable_bytes_limit"]
            or cumulative["writable_inodes"] > contract["writable_inodes_limit"]
        ):
            raise BridgeError("resource ledger exceeded its contract")
        receipt: dict[str, Any] = {
            "schema": EXECUTION_SCHEMA,
            "operation_id": request["operation_id"],
            "runner_sha256": self.identity.runner_sha256,
            "runtime_digest": self.identity.runtime_digest,
            "resource_contract_sha256": request["resource_contract_sha256"],
            "mount_attestation_sha256": state["mount_attestation_sha256"],
            "command_sha256": hashlib.sha256(
                request["command"].encode("utf-8")
            ).hexdigest(),
            "timeout_ms": request["timeout_ms"],
            "returncode": outcome.returncode,
            "timed_out": outcome.timed_out,
            "resource_delta": delta,
            "resource_cumulative": cumulative,
            "containment": {
                "scope": "episode_cgroup_descendants",
                "cgroup_enforced": True,
                "isolated_process_group": True,
                "descendant_process_count": 0,
            },
        }
        if outcome.external_memory_access is not None:
            if request["mode"] != MEMORY_MODE or outcome.external_memory_access not in {
                "read",
                "write",
                "read_write",
            }:
                raise BridgeError("external-memory access proof drifted")
            receipt["external_memory_access"] = {
                "schema": EXTERNAL_MEMORY_ACCESS_SCHEMA,
                "operation": outcome.external_memory_access,
            }
        response = {
            "returncode": outcome.returncode,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
            "timed_out": outcome.timed_out,
            "receipt": receipt,
        }
        state["resource_cumulative"] = cumulative
        self._complete_operation("execute", request, response, state)
        return response

    def _freeze(
        self, request: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        status, replay = self._operation_status("freeze", request, state)
        if status == "final":
            assert replay is not None
            return replay
        if state["lifecycle"] not in {"active", "frozen"}:
            raise BridgeError("episode cannot be frozen")
        if status == "new":
            self._begin_operation("freeze", request, state)
            outcome = self.runtime.freeze(request, copy.deepcopy(state))
        else:
            outcome = self.runtime.reconcile(
                "freeze", request, copy.deepcopy(state)
            )
        if (
            type(outcome) is not LifecycleOutcome
            or outcome.processes_reaped is not True
            or outcome.workspace_frozen is not True
            or outcome.mounts_released is not False
            or outcome.descendant_process_count != 0
            or type(outcome.mount_count) is not int
            or outcome.mount_count <= 0
            or outcome.sandbox_present is not True
        ):
            raise BridgeError("runtime freeze proof drifted")
        response = {
            "schema": FREEZE_SCHEMA,
            "operation_id": request["operation_id"],
            "runner_sha256": self.identity.runner_sha256,
            "runtime_digest": self.identity.runtime_digest,
            "resource_contract_sha256": request["resource_contract_sha256"],
            "mount_attestation_sha256": state["mount_attestation_sha256"],
            "resource_cumulative": copy.deepcopy(state["resource_cumulative"]),
            "processes_reaped": True,
            "workspace_frozen": True,
            "descendant_process_count": 0,
        }
        state["lifecycle"] = "frozen"
        self._complete_operation("freeze", request, response, state)
        return response

    def _teardown(
        self, request: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        status, replay = self._operation_status("teardown", request, state)
        if status == "final":
            assert replay is not None
            return replay
        already_torn_down = state["lifecycle"] == "torn_down"
        if status == "new":
            self._begin_operation("teardown", request, state)
            if already_torn_down:
                # A new teardown UUID is accepted only after the runtime
                # independently validates the bundle-bound tombstone and
                # proves that mounts, cgroups, processes, and run state remain
                # absent.  It is never answered from protocol state alone.
                outcome = self.runtime.reconcile(
                    "teardown", request, copy.deepcopy(state)
                )
            else:
                outcome = self.runtime.teardown(request, copy.deepcopy(state))
        else:
            outcome = self.runtime.reconcile(
                "teardown", request, copy.deepcopy(state)
            )
        if (
            type(outcome) is not LifecycleOutcome
            or outcome.processes_reaped is not True
            or outcome.mounts_released is not True
            or outcome.descendant_process_count != 0
            or outcome.mount_count != 0
            or outcome.sandbox_present is not False
        ):
            raise BridgeError("runtime teardown proof drifted")
        response = {
            "schema": TEARDOWN_SCHEMA,
            "operation_id": request["operation_id"],
            "runner_sha256": self.identity.runner_sha256,
            "runtime_digest": self.identity.runtime_digest,
            "resource_contract_sha256": request["resource_contract_sha256"],
            "mount_attestation_sha256": state["mount_attestation_sha256"],
            "resource_cumulative": copy.deepcopy(state["resource_cumulative"]),
            "processes_reaped": True,
            "mounts_released": True,
            "descendant_process_count": 0,
            "mount_count": 0,
            "sandbox_present": False,
        }
        state["lifecycle"] = "torn_down"
        self._complete_operation("teardown", request, response, state)
        return response


def validate_request(operation: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if operation not in {"attest", "execute", "freeze", "teardown"}:
        raise BridgeError("unsupported operation")
    if not isinstance(value, Mapping):
        raise BridgeError("request must be an object")
    request = copy.deepcopy(dict(value))
    expected = set(_BASE_FIELDS)
    mode = request.get("mode")
    if mode == MEMORY_MODE:
        expected.add("external_memory_root")
    if operation == "execute":
        expected.update(("operation_id", "command", "timeout_ms"))
    elif operation in {"freeze", "teardown"}:
        expected.add("operation_id")
    if set(request) != expected:
        raise BridgeError("request fields drifted")
    if request.get("schema") != REQUEST_SCHEMA:
        raise BridgeError("request schema drifted")
    episode_id = request.get("episode_id")
    competition_id = request.get("competition_id")
    if not isinstance(episode_id, str) or not _EPISODE_ID.fullmatch(episode_id):
        raise BridgeError("episode id drifted")
    if not isinstance(competition_id, str) or not _COMPETITION_ID.fullmatch(
        competition_id
    ):
        raise BridgeError("competition id drifted")
    if mode not in MODES:
        raise BridgeError("mode drifted")
    contract = _validate_resource_contract(request.get("resource_contract"))
    contract_sha256 = request.get("resource_contract_sha256")
    _require_sha256(contract_sha256, "resource contract SHA256")
    if canonical_sha256(contract) != contract_sha256:
        raise BridgeError("resource contract SHA256 mismatch")
    _require_sha256(request.get("public_tree_sha256"), "public tree SHA256")
    public = _canonical_path(request.get("public_root"), "public root")
    workspace = _canonical_path(request.get("workspace_root"), "workspace root")
    submission = _canonical_path(
        request.get("submission_root"), "submission root"
    )
    episode_root = posixpath.dirname(workspace)
    if (
        posixpath.basename(workspace) != "workspace"
        or posixpath.basename(submission) != "submission"
        or posixpath.dirname(submission) != episode_root
        or posixpath.basename(episode_root) != episode_id
    ):
        raise BridgeError("episode mount roots drifted")
    if _paths_overlap(public, episode_root):
        raise BridgeError("public and writable roots overlap")
    if mode == MEMORY_MODE:
        memory = _canonical_path(
            request.get("external_memory_root"), "external memory root"
        )
        if (
            posixpath.basename(memory) != "external-memory"
            or posixpath.dirname(memory) != episode_root
        ):
            raise BridgeError("external-memory root drifted")
    elif "external_memory_root" in request:
        raise BridgeError("external-memory capability drifted")
    if operation in {"execute", "freeze", "teardown"}:
        _canonical_uuid4(request.get("operation_id"))
    if operation == "execute":
        command = request.get("command")
        timeout_ms = request.get("timeout_ms")
        if (
            not isinstance(command, str)
            or "\x00" in command
            or len(command.encode("utf-8")) > 1024 * 1024
        ):
            raise BridgeError("command drifted")
        if (
            type(timeout_ms) is not int
            or timeout_ms <= 0
            or timeout_ms > contract["max_shell_timeout_ms"]
        ):
            raise BridgeError("timeout drifted")
    request["resource_contract"] = contract
    return request


def validate_deployment(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "rootfs",
        "rootfs_digest",
        "rootfs_tree_lock",
        "rootfs_tree_lock_sha256",
        "state_root",
        "episodes_root",
        "sandbox_host_uid",
        "sandbox_host_gid",
        "rootfs_loader_path",
        "rootfs_python_path",
        "rootfs_python_home",
        "rootfs_library_paths",
        "rootfs_nvidia_smi_path",
        "gpu",
        "openmle_v7_provenance",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BridgeError("deployment fields drifted")
    deployment = copy.deepcopy(dict(value))
    if deployment.get("schema") != "mlebench_lite_runtime_bridge_deployment_v1":
        raise BridgeError("deployment schema drifted")
    for field in (
        "rootfs",
        "rootfs_tree_lock",
        "state_root",
        "episodes_root",
    ):
        deployment[field] = _canonical_path(
            deployment.get(field), field.replace("_", " ")
        )
        if any(character.isspace() for character in deployment[field]) or "\\" in deployment[field]:
            raise BridgeError("deployment trusted path contains unsafe bytes")
    if deployment["rootfs"] == "/":
        raise BridgeError("deployment rootfs must be a dedicated sealed mount")
    for field in (
        "rootfs_loader_path",
        "rootfs_python_path",
        "rootfs_python_home",
        "rootfs_nvidia_smi_path",
    ):
        deployment[field] = _canonical_rootfs_member(
            deployment.get(field), field.replace("_", " ")
        )
    library_paths = deployment.get("rootfs_library_paths")
    if (
        not isinstance(library_paths, list)
        or not library_paths
        or len(library_paths) > 16
    ):
        raise BridgeError("rootfs library-path inventory drifted")
    deployment["rootfs_library_paths"] = [
        _canonical_rootfs_member(value, "rootfs library path")
        for value in library_paths
    ]
    if len(set(deployment["rootfs_library_paths"])) != len(library_paths):
        raise BridgeError("rootfs library-path inventory contains duplicates")
    if _paths_overlap(deployment["state_root"], deployment["episodes_root"]):
        raise BridgeError("deployment state and episode roots overlap")
    for writable_root in (deployment["state_root"], deployment["episodes_root"]):
        if _paths_overlap(deployment["rootfs"], writable_root):
            raise BridgeError("deployment rootfs overlaps a writable runtime root")
    _require_sha256(deployment.get("rootfs_digest"), "rootfs digest")
    _require_sha256(
        deployment.get("rootfs_tree_lock_sha256"), "rootfs tree-lock SHA256"
    )
    for field in ("sandbox_host_uid", "sandbox_host_gid"):
        if type(deployment.get(field)) is not int or deployment[field] <= 0:
            raise BridgeError("deployment sandbox identity drifted")
    provenance = deployment.get("openmle_v7_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "artifact_lock_sha256",
        "supervisor_sha256",
    }:
        raise BridgeError("OpenMLE v7 provenance fields drifted")
    if (
        provenance.get("artifact_lock_sha256")
        != OPENMLE_V7_ARTIFACT_LOCK_SHA256
        or provenance.get("supervisor_sha256") != OPENMLE_V7_SUPERVISOR_SHA256
    ):
        raise BridgeError("OpenMLE v7 provenance identity drifted")
    gpu = deployment.get("gpu")
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "uuid",
        "device",
        "control_devices",
    }:
        raise BridgeError("GPU deployment fields drifted")
    if not isinstance(gpu.get("uuid"), str) or not _GPU_UUID.fullmatch(gpu["uuid"]):
        raise BridgeError("GPU UUID drifted")
    controls = gpu.get("control_devices")
    if not isinstance(controls, list) or not controls:
        raise BridgeError("GPU control-device inventory drifted")
    devices = [_validate_gpu_device(gpu.get("device"), primary=True)]
    devices.extend(_validate_gpu_device(item, primary=False) for item in controls)
    sources = [item["source"] for item in devices]
    targets = [item["target"] for item in devices]
    numbers = [(item["major"], item["minor"]) for item in devices]
    if (
        len(set(sources)) != len(sources)
        or len(set(targets)) != len(targets)
        or len(set(numbers)) != len(numbers)
    ):
        raise BridgeError("GPU device inventory contains duplicates")
    if not {"/dev/nvidiactl", "/dev/nvidia-uvm"}.issubset(set(targets)):
        raise BridgeError("GPU control-device inventory is incomplete")
    allowed_controls = re.compile(
        r"^/dev/(?:nvidiactl|nvidia-uvm|nvidia-uvm-tools|"
        r"nvidia-modeset|nvidia-caps/nvidia-cap[0-9]+)$"
    )
    if any(not allowed_controls.fullmatch(item["target"]) for item in devices[1:]):
        raise BridgeError("GPU control-device projection is too broad")
    deployment["gpu"] = {
        "uuid": gpu["uuid"],
        "device": devices[0],
        "control_devices": devices[1:],
    }
    deployment["openmle_v7_provenance"] = dict(provenance)
    return deployment


def load_bundle_identity(
    runner_path: os.PathLike[str] | str,
    *,
    expected_uid: int,
    bundle_fd: int | None = None,
) -> BundleIdentity:
    if type(expected_uid) is not int or expected_uid < 0:
        raise BridgeError("bundle owner UID drifted")
    runner_value = os.fspath(runner_path)
    if not isinstance(runner_value, str) or "\x00" in runner_value:
        raise BridgeError("runner entrypoint path drifted")
    if bundle_fd is None:
        runner_value = os.path.abspath(runner_value)
    runner_name = os.path.basename(runner_value)
    if runner_name != "sandbox-runner":
        raise BridgeError("runner entrypoint name drifted")
    bundle_root = os.path.dirname(runner_value)
    if bundle_fd is not None:
        if type(bundle_fd) is not int or bundle_fd < 0:
            raise BridgeError("bundle anchor descriptor drifted")
        if bundle_root != f"/proc/self/fd/{bundle_fd}":
            raise BridgeError("bundle anchor path drifted")
        try:
            directory_fd = os.dup(bundle_fd)
        except OSError as exc:
            raise BridgeError("bundle anchor descriptor is unavailable") from exc
        retained_bundle_fd: int | None = bundle_fd
    else:
        directory_fd = _open_bundle_directory(bundle_root)
        retained_bundle_fd = None
    try:
        _validate_bundle_directory_fd(directory_fd, expected_uid)
        lock_payload = _read_safe_regular_file_at(
            directory_fd,
            "artifact-lock.json",
            expected_uid=expected_uid,
            maximum=2 * 1024 * 1024,
        )
        try:
            lock = strict_json_loads(lock_payload)
        except ValueError as exc:
            raise BridgeError("artifact lock is not strict JSON") from exc
        if (
            not isinstance(lock, dict)
            or set(lock)
            != {
                "schema",
                "files",
                "deployment_schema",
                "openmle_v7_provenance",
            }
            or lock.get("schema") != "mlebench_lite_runtime_artifact_lock_v1"
            or lock.get("deployment_schema")
            != "mlebench_lite_runtime_bridge_deployment_v1"
        ):
            raise BridgeError("artifact-lock shape drifted")
        provenance = lock.get("openmle_v7_provenance")
        if provenance != {
            "artifact_lock_sha256": OPENMLE_V7_ARTIFACT_LOCK_SHA256,
            "supervisor_sha256": OPENMLE_V7_SUPERVISOR_SHA256,
        }:
            raise BridgeError("artifact-lock provenance drifted")
        members = lock.get("files")
        if not isinstance(members, list) or not members:
            raise BridgeError("artifact-lock member inventory drifted")
        expected_paths: list[str] = []
        member_hashes: dict[str, str] = {}
        for member in members:
            if not isinstance(member, dict) or set(member) != {"path", "sha256"}:
                raise BridgeError("artifact-lock member fields drifted")
            relative = _safe_relative_member(member.get("path"))
            digest = _require_sha256(member.get("sha256"), "artifact member SHA256")
            if relative in member_hashes:
                raise BridgeError("artifact-lock member is duplicated")
            expected_paths.append(relative)
            member_hashes[relative] = digest
        if expected_paths != sorted(expected_paths):
            raise BridgeError("artifact-lock members are not sorted")
        required = {
            "sandbox-runner",
            "runner.py",
            "runner_launcher.c",
            "runtime_audit.c",
            "linux_runtime.py",
            "sandbox_supervisor.c",
            "bin/mlebench-lite-sandbox-supervisor",
            RUNTIME_AUDIT_RELATIVE_PATH,
            "deployment.json",
            ROOTFS_ELF_CLOSURE_RELATIVE_PATH,
            "build-provenance.json",
        }
        if not required.issubset(member_hashes):
            raise BridgeError("artifact-lock required member is absent")
        actual_paths = _bundle_file_inventory_fd(directory_fd, expected_uid)
        if actual_paths != set(member_hashes) | {"artifact-lock.json"}:
            raise BridgeError("bundle contains an unlocked member")
        for relative, expected in member_hashes.items():
            payload = _read_safe_regular_file_at(
                directory_fd,
                relative,
                expected_uid=expected_uid,
                maximum=None,
            )
            if hashlib.sha256(payload).hexdigest() != expected:
                raise BridgeError("artifact member hash drifted")
        deployment_payload = _read_safe_regular_file_at(
            directory_fd,
            "deployment.json",
            expected_uid=expected_uid,
            maximum=1024 * 1024,
        )
        try:
            deployment_value = strict_json_loads(deployment_payload)
        except ValueError as exc:
            raise BridgeError("deployment is not strict JSON") from exc
        deployment = validate_deployment(deployment_value)
        return BundleIdentity(
            identity=BridgeIdentity(
                runner_sha256=member_hashes["sandbox-runner"],
                runtime_digest=canonical_sha256(lock),
            ),
            bundle_root=bundle_root,
            deployment=deployment,
            supervisor_path=os.path.join(
                bundle_root, "bin", "mlebench-lite-sandbox-supervisor"
            ),
            expected_uid=expected_uid,
            bundle_fd=retained_bundle_fd,
        )
    finally:
        os.close(directory_fd)


def expected_attestation(
    request: Mapping[str, Any], identity: BridgeIdentity
) -> dict[str, Any]:
    mounts = [
        {
            "source": request["public_root"],
            "target": "/home/data",
            "read_only": True,
            "source_tree_sha256": request["public_tree_sha256"],
        },
        {
            "source": request["workspace_root"],
            "target": "/home/workspace",
            "read_only": False,
        },
        {
            "source": request["submission_root"],
            "target": "/home/submission",
            "read_only": False,
        },
    ]
    if request["mode"] == MEMORY_MODE:
        mounts.append(
            {
                "source": request["external_memory_root"],
                "target": "/run/amg_memory",
                "read_only": False,
            }
        )
    return {
        "schema": ATTESTATION_SCHEMA,
        "runner_sha256": identity.runner_sha256,
        "runtime_digest": identity.runtime_digest,
        "resource_contract": copy.deepcopy(request["resource_contract"]),
        "resource_contract_sha256": request["resource_contract_sha256"],
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
                "read_write_mount_v1" if request["mode"] == MEMORY_MODE else "none"
            ),
            "native_tool_surface": "inspect_edit_shell_v1",
            "private_root_state": (
                "allocated" if request["mode"] == MEMORY_MODE else "absent"
            ),
        },
        "mounts": mounts,
        "denied_mount_prefixes": ["/host", "/private"],
    }


def _validate_resource_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_FIELDS:
        raise BridgeError("resource contract fields drifted")
    contract = copy.deepcopy(dict(value))
    if contract.get("schema") != RESOURCE_SCHEMA:
        raise BridgeError("resource contract schema drifted")
    for field in _NUMERIC_RESOURCE_FIELDS:
        if type(contract[field]) is not int or contract[field] <= 0:
            raise BridgeError("resource contract numeric field drifted")
    for field, expected in FIXED_RESOURCE_VALUES.items():
        if contract[field] != expected:
            raise BridgeError("fixed resource target drifted")
    fixed = {
        "submission_path": "/home/submission/submission.csv",
        "network_disabled": True,
        "read_only_public_data": True,
        "process_scope": "episode_cgroup_descendants",
        "cgroup_required": True,
        "isolated_process_group_required": True,
    }
    for field, expected in fixed.items():
        if type(expected) is bool and type(contract.get(field)) is not bool:
            raise BridgeError("resource contract boolean field drifted")
        if contract.get(field) != expected:
            raise BridgeError("resource contract isolation field drifted")
    if (
        contract["max_shell_timeout_ms"] > contract["episode_timeout_ms"]
        or contract["max_total_execution_ms"] > contract["episode_timeout_ms"]
        or contract["max_submission_bytes"] > contract["writable_bytes_limit"]
        or contract["max_step_response_ms"] != contract["episode_timeout_ms"] + 30_000
    ):
        raise BridgeError("resource contract relationship drifted")
    return contract


def _validate_runtime_attestation(
    value: RuntimeAttestation, contract: Mapping[str, Any]
) -> None:
    if type(value) is not RuntimeAttestation:
        raise BridgeError("runtime attestation shape drifted")
    if (
        value.cpu_limit_cores != contract["cpu_limit_cores"]
        or value.memory_limit_bytes != contract["memory_limit_bytes"]
        or value.pids_limit != contract["pids_limit"]
        or value.gpu_count != 1
        or not isinstance(value.gpu_uuid, str)
        or not value.gpu_uuid.startswith("GPU-")
        or value.mount_namespace is not True
        or value.network_disabled is not True
        or value.non_root is not True
        or value.read_only_rootfs is not True
        or not isinstance(value.runtime_identity, dict)
        or not isinstance(value.runtime_identity.get("schema"), str)
        or not value.runtime_identity["schema"]
    ):
        raise BridgeError("runtime capability proof drifted")
    canonical_json_bytes(value.runtime_identity)


def _validate_execution_outcome(
    value: ExecutionOutcome, request: Mapping[str, Any]
) -> None:
    if type(value) is not ExecutionOutcome:
        raise BridgeError("runtime execution shape drifted")
    if (
        type(value.returncode) is not int
        or not isinstance(value.stdout, str)
        or not isinstance(value.stderr, str)
        or type(value.timed_out) is not bool
        or type(value.descendant_process_count) is not int
        or value.descendant_process_count != 0
    ):
        raise BridgeError("runtime execution proof drifted")
    for field in RESOURCE_USAGE_KEYS:
        item = getattr(value, field)
        if type(item) is not int or item < 0:
            raise BridgeError("runtime resource proof drifted")
    try:
        output_bytes = len(value.stdout.encode("utf-8")) + len(
            value.stderr.encode("utf-8")
        )
    except UnicodeEncodeError as exc:
        raise BridgeError("runtime output encoding drifted") from exc
    if output_bytes > request["resource_contract"]["max_visible_output_bytes"]:
        raise BridgeError("runtime output cap drifted")


def _base_request(request: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(_BASE_FIELDS)
    if request.get("mode") == MEMORY_MODE:
        fields.add("external_memory_root")
    return {key: copy.deepcopy(request[key]) for key in sorted(fields)}


def zero_resource_usage() -> dict[str, int]:
    return {key: 0 for key in RESOURCE_USAGE_KEYS}


def _validate_usage(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(RESOURCE_USAGE_KEYS):
        raise BridgeError(label + " fields drifted")
    result = dict(value)
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise BridgeError(label + " values drifted")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BridgeError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_loads(payload: bytes) -> Any:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid strict JSON") from exc


def _canonical_uuid4(value: Any) -> str:
    if not isinstance(value, str):
        raise BridgeError("operation id drifted")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise BridgeError("operation id drifted") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise BridgeError("operation id drifted")
    return value


def _canonical_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\x00" in value
        or posixpath.normpath(value) != value
        or "//" in value
    ):
        raise BridgeError(label + " drifted")
    parts = tuple(part for part in value.split("/") if part)
    if any(part in {".", "..", "host", "private"} for part in parts):
        raise BridgeError(label + " uses a denied component")
    return value


def _canonical_rootfs_member(value: Any, label: str) -> str:
    value = _canonical_path(value, label)
    if (
        value == "/"
        or len(value.encode("utf-8")) > 1024
        or any(character.isspace() for character in value)
        or "\\" in value
    ):
        raise BridgeError(f"{label} is not a safe rootfs member")
    return value


def _paths_overlap(first: str, second: str) -> bool:
    left = first.rstrip("/") + "/"
    right = second.rstrip("/") + "/"
    return first == second or left.startswith(right) or right.startswith(left)


def _validate_gpu_device(value: Any, *, primary: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "source",
        "target",
        "major",
        "minor",
    }:
        raise BridgeError("GPU device fields drifted")
    result = copy.deepcopy(dict(value))
    result["source"] = _canonical_path(result.get("source"), "GPU device source")
    result["target"] = _canonical_path(result.get("target"), "GPU device target")
    if not result["source"].startswith("/dev/nvidia") or not result[
        "target"
    ].startswith("/dev/nvidia"):
        raise BridgeError("GPU device path drifted")
    if primary and result["target"] != "/dev/nvidia0":
        raise BridgeError("primary GPU projection drifted")
    if not primary and result["target"] == "/dev/nvidia0":
        raise BridgeError("GPU control projection drifted")
    for field in ("major", "minor"):
        if type(result.get(field)) is not int or result[field] < 0:
            raise BridgeError("GPU device number drifted")
    return result


def _open_bundle_directory(path: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise BridgeError("bundle directory is unavailable") from exc


def _validate_bundle_directory_fd(descriptor: int, expected_uid: int) -> None:
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise BridgeError("bundle directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise BridgeError("bundle directory identity is unsafe")


def _open_bundle_member_at(
    bundle_fd: int,
    relative: str,
    *,
    expected_uid: int,
    directory: bool,
) -> int:
    relative = _safe_relative_member(relative)
    current = os.dup(bundle_fd)
    try:
        for index, component in enumerate(relative.split("/")):
            final = index == len(relative.split("/")) - 1
            flags = os.O_RDONLY
            if not final or directory:
                flags |= os.O_DIRECTORY
            opened = os.open(
                component,
                flags | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = opened
            if not final:
                _validate_bundle_directory_fd(current, expected_uid)
        if directory:
            _validate_bundle_directory_fd(current, expected_uid)
        return current
    except OSError as exc:
        os.close(current)
        raise BridgeError("bundle member is unavailable") from exc
    except BaseException:
        os.close(current)
        raise


def open_bundle_regular_file(
    bundle_fd: int, relative: str, *, expected_uid: int
) -> int:
    descriptor = _open_bundle_member_at(
        bundle_fd, relative, expected_uid=expected_uid, directory=False
    )
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_nlink != 1
        or metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        os.close(descriptor)
        raise BridgeError("bundle member identity is unsafe")
    return descriptor


def _read_safe_regular_file_at(
    bundle_fd: int,
    relative: str,
    *,
    expected_uid: int,
    maximum: int | None,
) -> bytes:
    descriptor = open_bundle_regular_file(
        bundle_fd, relative, expected_uid=expected_uid
    )
    try:
        metadata = os.fstat(descriptor)
        if maximum is not None and metadata.st_size > maximum:
            raise BridgeError("bundle member exceeds byte cap")
        limit = metadata.st_size if maximum is None else maximum
        payload = _read_bounded(descriptor, limit)
        after = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise BridgeError("bundle member changed while hashing")
        return payload
    finally:
        os.close(descriptor)


def _safe_relative_member(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\x00" in value
        or posixpath.normpath(value) != value
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise BridgeError("artifact member path drifted")
    return value


def _bundle_file_inventory_fd(bundle_fd: int, expected_uid: int) -> set[str]:
    result: set[str] = set()
    seen_directories: set[tuple[int, int]] = set()

    def visit(directory_fd: int, prefix: str) -> None:
        _validate_bundle_directory_fd(directory_fd, expected_uid)
        before = os.fstat(directory_fd)
        identity = (before.st_dev, before.st_ino)
        if identity in seen_directories:
            raise BridgeError("bundle contains a directory alias")
        seen_directories.add(identity)
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise BridgeError("bundle directory cannot be inventoried") from exc
        for name in names:
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\x00" in name
                or any(0xD800 <= ord(character) <= 0xDFFF for character in name)
            ):
                raise BridgeError("bundle member name is unsafe")
            relative = name if not prefix else prefix + "/" + name
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise BridgeError("bundle member cannot be stated") from exc
            if stat.S_ISDIR(metadata.st_mode):
                child = _open_bundle_member_at(
                    directory_fd,
                    name,
                    expected_uid=expected_uid,
                    directory=True,
                )
                try:
                    if _stable_file_identity(metadata) != _stable_file_identity(
                        os.fstat(child)
                    ):
                        raise BridgeError("bundle directory changed while opening")
                    visit(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise BridgeError("bundle contains a non-regular member")
            _safe_relative_member(relative)
            _read_safe_regular_file_at(
                directory_fd, name, expected_uid=expected_uid, maximum=None
            )
            result.add(relative)
        if _stable_file_identity(before) != _stable_file_identity(
            os.fstat(directory_fd)
        ):
            raise BridgeError("bundle directory changed while hashing")

    visit(bundle_fd, "")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BridgeError(label + " must be lowercase SHA256")
    return value


def _read_bounded(descriptor: int, maximum: int) -> bytes:
    chunks = bytearray()
    while len(chunks) <= maximum:
        chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(chunks)))
        if not chunk:
            return bytes(chunks)
        chunks.extend(chunk)
    raise BridgeError("bounded file exceeds byte cap")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise BridgeError("bounded file write failed")
        view = view[written:]


def main(argv: list[str] | None = None) -> int:
    """Run one strict, serialized external-runner protocol operation."""

    operation_started = time.monotonic()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1 or arguments[0] not in {
        "attest",
        "execute",
        "freeze",
        "teardown",
    }:
        raise BridgeError("runner operation argument drifted")
    operation = arguments[0]
    payload = _read_stdin_bounded(2 * 1024 * 1024)
    try:
        value = strict_json_loads(payload)
    except ValueError as exc:
        raise BridgeError("runner request is not strict JSON") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise BridgeError("runner request is not canonical JSON")
    request = validate_request(operation, value)
    operation_deadline = operation_started + (
        request["resource_contract"]["max_step_response_ms"] / 1000.0
    )
    bundle_root = os.environ.get("MLE_BRIDGE_BUNDLE_ROOT")
    expected_bundle_root = f"/proc/self/fd/{BUNDLE_ROOT_FD}"
    if bundle_root != expected_bundle_root:
        raise BridgeError("native launcher bundle identity is absent")
    try:
        _validate_bundle_directory_fd(BUNDLE_ROOT_FD, os.geteuid())
        if not os.get_inheritable(BUNDLE_ROOT_FD):
            raise BridgeError("native launcher bundle anchor is not inherited")
    except OSError as exc:
        raise BridgeError("native launcher bundle anchor is unavailable") from exc
    launcher_path = os.path.join(bundle_root, "sandbox-runner")
    bundle = load_bundle_identity(
        launcher_path,
        expected_uid=os.geteuid(),
        bundle_fd=BUNDLE_ROOT_FD,
    )
    expected_runtime_digest = os.environ.get(
        "MLE_BRIDGE_EXPECTED_RUNTIME_DIGEST"
    )
    if (
        expected_runtime_digest is None
        or _require_sha256(expected_runtime_digest, "expected runtime digest")
        != bundle.identity.runtime_digest
    ):
        raise BridgeError("configured runtime digest drifted")
    _verify_running_python(bundle.deployment)
    if time.monotonic() >= operation_deadline:
        raise BridgeError("operation deadline expired during pre-exec verification")
    sys.dont_write_bytecode = True
    runtime_path = os.path.join(bundle.bundle_root, "linux_runtime.py")
    specification = importlib.util.spec_from_file_location(
        "mlebench_lite_locked_linux_runtime", runtime_path
    )
    if specification is None or specification.loader is None:
        raise BridgeError("locked Linux runtime cannot be imported")
    module = importlib.util.module_from_spec(specification)
    try:
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        runtime_mapping_identity = module.verify_audited_runtime_mappings(
            bundle.deployment, bundle.bundle_root, bundle.bundle_fd
        )
        runtime = module.LinuxRuntime(
            bundle,
            operation_started=operation_started,
            operation_deadline=operation_deadline,
            runtime_mapping_identity=runtime_mapping_identity,
        )
    except BridgeError:
        raise
    except BaseException as exc:
        raise BridgeError("locked Linux runtime cannot be loaded") from exc
    store = SealedFileStateStore(
        bundle.deployment["state_root"], expected_uid=os.geteuid()
    )
    try:
        engine = BridgeEngine(bundle.identity, runtime, store)
        with store.episode_lock(request["episode_id"]):
            response = engine.handle(operation, request)
        output = canonical_json_bytes(response)
        _write_all(sys.stdout.fileno(), output)
    finally:
        store.close()
    return 0


def _read_stdin_bounded(maximum: int) -> bytes:
    payload = bytearray()
    while len(payload) <= maximum:
        chunk = os.read(
            sys.stdin.fileno(), min(1024 * 1024, maximum + 1 - len(payload))
        )
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
    raise BridgeError("runner request exceeds byte cap")


def _verify_running_python(deployment: Mapping[str, Any]) -> None:
    if sys.version_info < (3, 11):
        raise BridgeError("runtime bridge requires Python 3.11 or newer")
    configured = os.path.join(
        deployment["rootfs"], deployment["rootfs_python_path"].lstrip("/")
    )
    descriptor: int | None = None
    try:
        running_metadata = os.stat("/proc/self/exe")
        descriptor = _open_absolute_regular_nofollow(configured)
    except OSError as exc:
        raise BridgeError("sealed rootfs Python identity is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != 0
            or opened.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or (opened.st_dev, opened.st_ino)
            != (running_metadata.st_dev, running_metadata.st_ino)
        ):
            raise BridgeError("sealed rootfs Python identity drifted")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_absolute_regular_nofollow(path: str) -> int:
    components = [component for component in path.split("/") if component]
    if not path.startswith("/") or not components:
        raise OSError(errno.EINVAL, "unsafe absolute path")
    current = os.open(
        "/",
        getattr(os, "O_PATH", os.O_RDONLY)
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC,
    )
    try:
        for index, component in enumerate(components):
            if component in {".", ".."}:
                raise OSError(errno.EINVAL, "unsafe path component")
            final = index == len(components) - 1
            flags = os.O_RDONLY if final else (
                getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY
            )
            opened = os.open(
                component,
                flags | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = opened
        return current
    except BaseException:
        os.close(current)
        raise


def _stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:  # noqa: BLE001 - protocol failures must be silent.
        os._exit(2)
    raise SystemExit(exit_code)
