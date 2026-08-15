from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import stat
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset import canonical_sha256
from .deadline import DeadlineExceeded, MonotonicDeadline
from .grader_client import read_credential
from .grader_protocol import (
    CONTRACT_VERSION,
    GradeRequest,
    GradeResult,
    GraderProtocolError,
    authenticated_message,
    receive_frame,
    send_frame,
    verify_authenticated_message,
)
from .private_grader_runner import (
    PrivateGradeExecutionRequest,
    PrivateGraderBackend,
)

PRIVATE_MANIFEST_SCHEMA = "openmle_fast_fullpool_private_grader_manifest_v1"
PUBLIC_MANIFEST_BINDING_KEYS = frozenset({"g64", "train", "heldout"})


class PrivateGraderError(RuntimeError):
    pass


class InvalidSubmission(PrivateGraderError):
    pass


@dataclass(frozen=True)
class _PrivateTask:
    task_id: str
    binding_id: str
    binding_sha256: str
    archive_path: Path
    archive_sha256: str
    metric_path: Path
    metric_sha256: str
    answer_path: Path
    answer_sha256: str
    public_tree_sha256: str
    task_spec_sha256: str
    package_identity_sha256: str
    baseline_score: float
    ideal_score: float
    higher_is_better: bool
    validator_success_forms: tuple[str, ...]


class PrivateGraderService:
    """Authenticated AF_UNIX service and sole owner of private task material."""

    def __init__(
        self,
        *,
        private_manifest_path: Path | str,
        expected_manifest_sha256: str,
        package_root: Path | str,
        archive_root: Path | str,
        expected_release_revision: str,
        expected_runtime_digest: str,
        socket_path: Path | str,
        credential_path: Path | str,
        audit_root: Path | str,
        total_wall_ms: int,
        max_concurrent_requests: int,
        backend: PrivateGraderBackend,
        max_submission_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.private_manifest_path = _real_file(
            Path(private_manifest_path), "private grader manifest"
        )
        manifest_bytes = self.private_manifest_path.read_bytes()
        if _sha256(manifest_bytes) != _digest(expected_manifest_sha256):
            raise PrivateGraderError("private grader manifest SHA256 mismatch")
        try:
            manifest = _strict_json_loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrivateGraderError("private grader manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise PrivateGraderError("private grader manifest must be an object")
        if manifest.get("schema") != PRIVATE_MANIFEST_SCHEMA:
            raise PrivateGraderError("private grader manifest schema mismatch")
        if manifest.get("contract_version") != CONTRACT_VERSION:
            raise PrivateGraderError("private grader contract mismatch")
        if manifest.get("openmle_tasks_revision") != expected_release_revision:
            raise PrivateGraderError("private grader release mismatch")
        runtime_id = manifest.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id.strip():
            raise PrivateGraderError("private grader runtime identity is missing")
        manifest_runtime_digest = _runtime_digest(manifest.get("runtime_digest"))
        configured_runtime_digest = _runtime_digest(expected_runtime_digest)
        if manifest_runtime_digest != configured_runtime_digest:
            raise PrivateGraderError("private grader runtime digest mismatch")
        self.runtime_digest = configured_runtime_digest
        self.package_root = _real_directory(Path(package_root), "private package root")
        self.archive_root = _real_directory(Path(archive_root), "private archive root")
        tasks = manifest.get("records")
        task_count = manifest.get("task_count")
        if (
            type(task_count) is not int
            or task_count <= 0
            or not isinstance(tasks, list)
            or len(tasks) != task_count
        ):
            raise PrivateGraderError("private grader manifest has no tasks")
        public_manifests = manifest.get("public_manifest_sha256")
        if (
            not isinstance(public_manifests, dict)
            or set(public_manifests) != PUBLIC_MANIFEST_BINDING_KEYS
        ):
            raise PrivateGraderError("private grader public-manifest bindings drifted")
        for digest in public_manifests.values():
            if _digest(digest) != digest:
                raise PrivateGraderError("private grader public-manifest binding drift")
        loaded: dict[str, _PrivateTask] = {}
        for expected_index, raw in enumerate(tasks):
            if (
                not isinstance(raw, dict)
                or raw.get("private_data_idx") != expected_index
            ):
                raise PrivateGraderError("private task indices are not contiguous")
            if raw.get("reward_eligible") is False:
                if (
                    raw.get("reward_block_reason")
                    != "nonpositive_baseline_to_ideal_gap"
                ):
                    raise PrivateGraderError("private reward exclusion reason drift")
                continue
            if raw.get("reward_eligible") is not True:
                raise PrivateGraderError("private reward eligibility is invalid")
            task = self._load_task(raw)
            if task.task_id in loaded:
                raise PrivateGraderError("private grader manifest has duplicate tasks")
            loaded[task.task_id] = task
        self.tasks = loaded
        self.credential = read_credential(Path(credential_path))
        self.socket_path = Path(socket_path).expanduser().absolute()
        parent = self.socket_path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise PrivateGraderError("private grader socket parent is invalid")
        self.audit_root = Path(audit_root).expanduser().absolute()
        self.audit_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.audit_root.is_symlink() or not self.audit_root.is_dir():
            raise PrivateGraderError("private grader audit root is invalid")
        self.audit_root.chmod(0o700)
        if type(total_wall_ms) is not int or total_wall_ms <= 0:
            raise ValueError("private grader total wall must be positive")
        if type(max_concurrent_requests) is not int or max_concurrent_requests <= 0:
            raise ValueError("private grader concurrency must be positive")
        if type(max_submission_bytes) is not int or max_submission_bytes <= 0:
            raise ValueError("private grader input cap must be positive")
        self.total_wall_ms = total_wall_ms
        self.max_concurrent_requests = max_concurrent_requests
        self.max_submission_bytes = max_submission_bytes
        backend_limits = backend.metadata.get("resource_limits")
        if (
            not isinstance(backend_limits, Mapping)
            or backend_limits.get("input_bytes") != max_submission_bytes
        ):
            raise PrivateGraderError("private backend input limit mismatch")
        self.backend = backend
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._server: socket.socket | None = None
        self._admission = threading.BoundedSemaphore(max_concurrent_requests)
        self._worker_lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()

    def _load_task(self, raw: Any) -> _PrivateTask:
        if not isinstance(raw, dict):
            raise PrivateGraderError("private task binding must be an object")
        task_id = _text(raw, "task_id")
        binding_id = _text(raw, "grader_binding")
        metric_sha256 = _digest(raw.get("metric_sha256"))
        binding_sha256 = _digest(raw.get("grader_binding_sha256"))
        if binding_id != f"openmlefast-grader-{binding_sha256[:24]}":
            raise PrivateGraderError("private grader binding identifier drift")
        archive_relative = raw.get("archive_relpath", f"{task_id}.tar.zst")
        if not isinstance(archive_relative, str):
            raise PrivateGraderError("archive_relpath must be text")
        archive = _contained_file(
            self.archive_root,
            self.archive_root / _relative(archive_relative),
            "private task archive",
        )
        archive_sha256 = _digest(raw.get("archive_sha256"))
        if _file_sha256(archive) != archive_sha256:
            raise PrivateGraderError("private task archive SHA256 mismatch")
        metric = _contained_file(
            self.package_root,
            self.package_root / _relative(_text(raw, "metric_relpath")),
            "native scoring module",
        )
        if _file_sha256(metric) != metric_sha256:
            raise PrivateGraderError("native scoring module SHA256 mismatch")
        answer = _contained_file(
            self.package_root,
            self.package_root / _relative(_text(raw, "answer_relpath")),
            "private truth file",
        )
        answer_sha256 = _digest(raw.get("answer_sha256"))
        if _file_sha256(answer) != answer_sha256:
            raise PrivateGraderError("private truth SHA256 mismatch")
        normalization = raw.get("normalization")
        if not isinstance(normalization, dict):
            raise PrivateGraderError("private normalization contract is missing")
        baseline = _finite(normalization.get("baseline_score"), "baseline_score")
        ideal = _finite(normalization.get("ideal_score"), "ideal_score")
        direction = normalization.get("higher_is_better")
        if type(direction) is not bool:
            raise PrivateGraderError("private task direction must be Boolean")
        directed_gap = (1.0 if direction else -1.0) * (ideal - baseline)
        scale = max(1.0, abs(baseline), abs(ideal))
        if directed_gap <= 1e-6 * scale:
            raise PrivateGraderError("private task has no normalization gap")
        validator = raw.get("validator_contract")
        forms = _validator_success_forms(validator)
        public_tree_sha256 = _digest(raw.get("public_tree_sha256"))
        task_spec_sha256 = _digest(raw.get("task_spec_sha256"))
        binding = raw.get("grader_binding_payload")
        if not isinstance(binding, dict) or canonical_sha256(binding) != binding_sha256:
            raise PrivateGraderError("private grader binding SHA256 mismatch")
        expected_binding_fields = {
            "schema": "openmle_fast_grader_binding_v1",
            "task_id": task_id,
            "archive_sha256": archive_sha256,
            "metric_sha256": metric_sha256,
            "answer_sha256": answer_sha256,
            "public_tree_sha256": public_tree_sha256,
            "task_spec_sha256": task_spec_sha256,
            "normalization": normalization,
            "validator_contract": validator,
        }
        if any(
            binding.get(key) != value for key, value in expected_binding_fields.items()
        ):
            raise PrivateGraderError("private grader binding payload drift")
        package_identity_sha256 = _digest(raw.get("package_identity_sha256"))
        identity = {
            "task_id": task_id,
            "archive_sha256": archive_sha256,
            "public_tree_sha256": public_tree_sha256,
            "metric_sha256": metric_sha256,
            "private_grader_binding_sha256": binding_sha256,
            "task_spec_sha256": task_spec_sha256,
        }
        if canonical_sha256(identity) != package_identity_sha256:
            raise PrivateGraderError("private package identity SHA256 mismatch")
        return _PrivateTask(
            task_id=task_id,
            binding_id=binding_id,
            binding_sha256=binding_sha256,
            archive_path=archive,
            archive_sha256=archive_sha256,
            metric_path=metric,
            metric_sha256=metric_sha256,
            answer_path=answer,
            answer_sha256=answer_sha256,
            public_tree_sha256=public_tree_sha256,
            task_spec_sha256=task_spec_sha256,
            package_identity_sha256=package_identity_sha256,
            baseline_score=baseline,
            ideal_score=ideal,
            higher_is_better=direction,
            validator_success_forms=forms,
        )

    def wait_until_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def serve_forever(self) -> None:
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise PrivateGraderError("private grader socket path already exists")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        try:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(socket.SOMAXCONN)
            server.settimeout(0.2)
            self._ready.set()
            while not self._stopping.is_set():
                if not self._admission.acquire(timeout=0.2):
                    continue
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    self._admission.release()
                    continue
                except OSError:
                    self._admission.release()
                    if self._stopping.is_set():
                        break
                    raise
                deadline = MonotonicDeadline.after_ms(self.total_wall_ms)
                worker = threading.Thread(
                    target=self._serve_admitted_connection,
                    args=(connection, deadline),
                    daemon=True,
                    name="openmle-private-grade",
                )
                with self._worker_lock:
                    self._workers.add(worker)
                    self._connections.add(connection)
                try:
                    worker.start()
                except BaseException:
                    with self._worker_lock:
                        self._workers.discard(worker)
                        self._connections.discard(connection)
                    connection.close()
                    self._admission.release()
                    raise
        finally:
            self._ready.clear()
            try:
                server.close()
            finally:
                self._server = None
                self._close_active_connections()
                self._join_workers()
                if self.socket_path.exists() and not self.socket_path.is_symlink():
                    self.socket_path.unlink()

    def shutdown(self) -> None:
        self._stopping.set()
        server = self._server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        self._close_active_connections()

    def _serve_admitted_connection(
        self,
        connection: socket.socket,
        deadline: MonotonicDeadline,
    ) -> None:
        try:
            with connection:
                self._serve_connection(connection, deadline)
        finally:
            with self._worker_lock:
                self._connections.discard(connection)
                self._workers.discard(threading.current_thread())
            self._admission.release()

    def _serve_connection(
        self,
        connection: socket.socket,
        deadline: MonotonicDeadline,
    ) -> None:
        try:
            connection.settimeout(deadline.remaining_seconds())
            payload = verify_authenticated_message(
                receive_frame(connection, deadline=deadline),
                self.credential,
                deadline=deadline,
            )
            deadline.check()
            request = GradeRequest.from_payload(
                payload,
                max_submission_bytes=self.max_submission_bytes,
                deadline=deadline,
            )
            deadline.check()
        except (OSError, TimeoutError, GraderProtocolError, DeadlineExceeded):
            return
        try:
            result = self._grade_or_fault(request, deadline)
            deadline.check()
            response = authenticated_message(
                result.payload(),
                self.credential,
                deadline=deadline,
            )
            deadline.check()
            connection.settimeout(deadline.remaining_seconds())
            send_frame(
                connection,
                response,
                deadline=deadline,
            )
            deadline.check()
        except (OSError, TimeoutError, GraderProtocolError, DeadlineExceeded):
            return

    def _close_active_connections(self) -> None:
        with self._worker_lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _join_workers(self) -> None:
        stop_at = time.monotonic() + self.total_wall_ms / 1000.0
        while True:
            with self._worker_lock:
                workers = tuple(self._workers)
            if not workers:
                return
            for worker in workers:
                remaining = stop_at - time.monotonic()
                if remaining <= 0:
                    return
                worker.join(remaining)

    def _grade_or_fault(
        self,
        request: GradeRequest,
        deadline: MonotonicDeadline,
    ) -> GradeResult:
        deadline.check()
        task = self.tasks.get(request.task_id)
        if task is None:
            return self._with_audit(
                self._fault(request, higher_is_better=False), deadline
            )
        if (
            request.grader_binding_sha256 != task.binding_sha256
            or request.package_identity_sha256 != task.package_identity_sha256
            or request.baseline_score != task.baseline_score
            or request.ideal_score != task.ideal_score
            or request.higher_is_better != task.higher_is_better
        ):
            return self._with_audit(
                self._fault(request, higher_is_better=request.higher_is_better),
                deadline,
            )
        try:
            result = self._grade(request, task, deadline)
        except DeadlineExceeded:
            raise
        except InvalidSubmission:
            result = self._invalid(request, task)
        except BaseException:  # noqa: BLE001 - untrusted native code must not stop service
            result = self._fault(request, higher_is_better=task.higher_is_better)
        try:
            return self._with_audit(result, deadline)
        except DeadlineExceeded:
            raise
        except Exception:  # noqa: BLE001 - audit failure is sanitized infrastructure
            fault = self._fault(
                request,
                higher_is_better=(False if task is None else task.higher_is_better),
            )
            record = fault.payload()
            record.pop("audit_digest")
            deadline.check()
            return GradeResult(
                request_id=fault.request_id,
                episode_id=fault.episode_id,
                task_id=fault.task_id,
                grader_binding_sha256=fault.grader_binding_sha256,
                package_identity_sha256=fault.package_identity_sha256,
                baseline_score=fault.baseline_score,
                ideal_score=fault.ideal_score,
                submission_sha256=fault.submission_sha256,
                submission_valid=False,
                native_score=None,
                higher_is_better=fault.higher_is_better,
                normalized_reward=None,
                improved_over_baseline=False,
                runtime_success=False,
                terminal_reason="grader_infrastructure_fault",
                classification="infrastructure_fault",
                audit_digest=canonical_sha256(record),
            )

    def _grade(
        self,
        request: GradeRequest,
        task: _PrivateTask,
        deadline: MonotonicDeadline,
    ) -> GradeResult:
        deadline.check()
        if (
            _file_sha256(task.archive_path, deadline=deadline) != task.archive_sha256
            or _file_sha256(task.metric_path, deadline=deadline) != task.metric_sha256
            or _file_sha256(task.answer_path, deadline=deadline) != task.answer_sha256
        ):
            raise PrivateGraderError("trusted private input changed")
        if not request.submission:
            raise InvalidSubmission("submission is empty")
        metric = _read_verified_bytes(
            task.metric_path,
            task.metric_sha256,
            deadline=deadline,
        )
        answer = _read_verified_bytes(
            task.answer_path,
            task.answer_sha256,
            deadline=deadline,
        )
        execution = self.backend.grade(
            PrivateGradeExecutionRequest(
                task_id=task.task_id,
                grader_binding_sha256=task.binding_sha256,
                package_identity_sha256=task.package_identity_sha256,
                metric_sha256=task.metric_sha256,
                answer_sha256=task.answer_sha256,
                higher_is_better=task.higher_is_better,
                validator_success_forms=task.validator_success_forms,
                metric=metric,
                answer=answer,
                submission=request.submission,
            ),
            timeout_ms=deadline.remaining_milliseconds(),
        )
        deadline.check()
        if execution.classification == "invalid_submission":
            raise InvalidSubmission("native validation rejected submission")
        if execution.classification != "graded" or execution.native_score is None:
            raise PrivateGraderError("private worker infrastructure fault")
        score = execution.native_score
        reward, improved = _normalize_score(task, score)
        return GradeResult(
            request_id=request.request_id,
            episode_id=request.episode_id,
            task_id=request.task_id,
            grader_binding_sha256=request.grader_binding_sha256,
            package_identity_sha256=request.package_identity_sha256,
            baseline_score=request.baseline_score,
            ideal_score=request.ideal_score,
            submission_sha256=request.submission_sha256,
            submission_valid=True,
            native_score=score,
            higher_is_better=task.higher_is_better,
            normalized_reward=reward,
            improved_over_baseline=improved,
            runtime_success=True,
            terminal_reason="graded_submission",
            classification="graded",
            audit_digest="0" * 64,
        )

    def _invalid(self, request: GradeRequest, task: _PrivateTask) -> GradeResult:
        return GradeResult(
            request_id=request.request_id,
            episode_id=request.episode_id,
            task_id=request.task_id,
            grader_binding_sha256=request.grader_binding_sha256,
            package_identity_sha256=request.package_identity_sha256,
            baseline_score=request.baseline_score,
            ideal_score=request.ideal_score,
            submission_sha256=request.submission_sha256,
            submission_valid=False,
            native_score=None,
            higher_is_better=task.higher_is_better,
            normalized_reward=-1.0,
            improved_over_baseline=False,
            runtime_success=False,
            terminal_reason="invalid_submission",
            classification="invalid_submission",
            audit_digest="0" * 64,
        )

    def _fault(self, request: GradeRequest, *, higher_is_better: bool) -> GradeResult:
        return GradeResult(
            request_id=request.request_id,
            episode_id=request.episode_id,
            task_id=request.task_id,
            grader_binding_sha256=request.grader_binding_sha256,
            package_identity_sha256=request.package_identity_sha256,
            baseline_score=request.baseline_score,
            ideal_score=request.ideal_score,
            submission_sha256=request.submission_sha256,
            submission_valid=False,
            native_score=None,
            higher_is_better=higher_is_better,
            normalized_reward=None,
            improved_over_baseline=False,
            runtime_success=False,
            terminal_reason="grader_infrastructure_fault",
            classification="infrastructure_fault",
            audit_digest="0" * 64,
        )

    def _with_audit(
        self,
        result: GradeResult,
        deadline: MonotonicDeadline,
    ) -> GradeResult:
        deadline.check()
        record = result.payload()
        record.pop("audit_digest")
        digest = canonical_sha256(record)
        destination = self.audit_root / f"grade-{uuid.uuid4().hex}.json"
        payload = (
            json.dumps(
                {**record, "audit_digest": digest},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        deadline.check()
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                deadline.check()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        deadline.check()
        return GradeResult(
            request_id=result.request_id,
            episode_id=result.episode_id,
            task_id=result.task_id,
            grader_binding_sha256=result.grader_binding_sha256,
            package_identity_sha256=result.package_identity_sha256,
            baseline_score=result.baseline_score,
            ideal_score=result.ideal_score,
            submission_sha256=result.submission_sha256,
            submission_valid=result.submission_valid,
            native_score=result.native_score,
            higher_is_better=result.higher_is_better,
            normalized_reward=result.normalized_reward,
            improved_over_baseline=result.improved_over_baseline,
            runtime_success=result.runtime_success,
            terminal_reason=result.terminal_reason,
            classification=result.classification,
            audit_digest=digest,
        )


def _normalize_score(task: _PrivateTask, score: float) -> tuple[float, bool]:
    direction = 1.0 if task.higher_is_better else -1.0
    scale = max(1.0, abs(task.baseline_score), abs(task.ideal_score))
    tolerance = 1e-9 * scale
    gap = direction * (task.ideal_score - task.baseline_score)
    if not math.isfinite(gap) or gap <= 1e-6 * scale:
        raise PrivateGraderError("normalization gap drift")
    delta_raw = direction * (score - task.baseline_score)
    delta = 0.0 if abs(delta_raw) <= tolerance else delta_raw
    reward = max(-1.0, min(1.0, delta / gap))
    return reward, delta_raw > tolerance


def _real_file(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise PrivateGraderError(f"{label} must be a real file")
    return absolute.resolve()


def _real_directory(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise PrivateGraderError(f"{label} must be a real directory")
    return absolute.resolve()


def _contained_file(root: Path, path: Path, label: str) -> Path:
    resolved = _contained(root, path, label)
    info = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PrivateGraderError(f"{label} must be an independent regular file")
    return resolved


def _contained_directory(root: Path, path: Path, label: str) -> Path:
    resolved = _contained(root, path, label)
    if not resolved.is_dir():
        raise PrivateGraderError(f"{label} must be a real directory")
    return resolved


def _contained(root: Path, path: Path, label: str) -> Path:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise PrivateGraderError(f"{label} escapes its configured root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise PrivateGraderError(f"{label} traverses a symlink")
    resolved = absolute.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PrivateGraderError(f"{label} escapes its configured root") from exc
    return resolved


def _relative(value: str) -> str:
    if "\x00" in value or "\\" in value:
        raise PrivateGraderError("private manifest path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PrivateGraderError("private manifest path is invalid")
    return path.as_posix()


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PrivateGraderError(f"private manifest field {key!r} is invalid")
    return value.strip()


def _validator_success_forms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict) or value.get("success") is not True:
        raise PrivateGraderError("validator contract is not an admitted success form")
    kind = value.get("kind")
    success_value = value.get("success_value")
    if not isinstance(success_value, str) or not success_value:
        raise PrivateGraderError("validator success value must be non-empty text")
    if kind == "string":
        return (success_value,)
    if kind == "bool_message_tuple":
        return ()
    raise PrivateGraderError("validator return convention is not admitted")


def _runtime_digest(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise PrivateGraderError("private grader runtime digest is invalid")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str):
        raise PrivateGraderError("private manifest digest must be text")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise PrivateGraderError("private manifest digest is invalid")
    return normalized


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivateGraderError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PrivateGraderError(f"{label} must be finite")
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _read_verified_bytes(
    path: Path,
    expected_sha256: str,
    *,
    deadline: MonotonicDeadline,
) -> bytes:
    deadline.check()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PrivateGraderError("private worker input is not an independent file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            chunks: list[bytes] = []
            while True:
                deadline.check()
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if _sha256(payload) != expected_sha256:
        raise PrivateGraderError("private worker input changed")
    deadline.check()
    return payload


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PrivateGraderError("private manifest contains a duplicate key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)
