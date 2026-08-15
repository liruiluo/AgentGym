from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .dataset import PublicTaskRecord
from .resources import (
    DEFAULT_CPU_LIMIT_CORES,
    DEFAULT_EPISODE_TIMEOUT_MS,
    DEFAULT_GPU_COUNT,
    DEFAULT_MAX_SHELL_TIMEOUT_MS,
    DEFAULT_MAX_TOTAL_EXECUTION_MS,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_WRITABLE_BYTES_LIMIT,
    DEFAULT_WRITABLE_INODES_LIMIT,
    build_resource_contract,
    validate_resource_contract,
)
from .resources import (
    resource_contract_sha256 as _resource_contract_sha256,
)

MODE_NATIVE = "native"
MODE_AMG_COMPACTION_ONLY = "amg_compaction_only"
MODE_AMG_MEMORY = "amg_memory"
MODES = (MODE_NATIVE, MODE_AMG_COMPACTION_ONLY, MODE_AMG_MEMORY)
COMPACTION_MODES = (MODE_AMG_COMPACTION_ONLY, MODE_AMG_MEMORY)
SUBMISSION_PATH = "/home/submission/submission.csv"

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class MLEBenchLiteWorkspaceError(RuntimeError):
    """A policy path or workspace boundary is unavailable."""


class MLEBenchLitePolicyPathError(MLEBenchLiteWorkspaceError):
    """A policy-selected path is absent, outside the mount, or unsafe."""


class PendingCreationCleanup:
    """Opaque manager-issued authority to retry one exact create rollback."""


class MLEBenchLiteWorkspaceRollbackError(MLEBenchLiteWorkspaceError):
    """A failed create still owns an exact, retryable cleanup target."""

    def __init__(self, pending_cleanup: PendingCreationCleanup) -> None:
        super().__init__("episode workspace rollback failed")
        self.pending_cleanup = pending_cleanup


@dataclass(frozen=True)
class EpisodeWorkspace:
    episode_id: str
    competition_id: str
    mode: str
    episode_root: Path
    workspace_root: Path
    submission_root: Path
    public_root: Path
    public_tree_sha256: str
    resource_contract: Mapping[str, Any]
    resource_contract_sha256: str

    @property
    def submission_path(self) -> Path:
        return self.submission_root / "submission.csv"

    def resolve_policy_path(self, value: str, *, write: bool) -> Path:
        """Return a diagnostic path after a no-symlink walk.

        Host reads/writes use the dirfd helpers below. This method remains for
        boundary checks and tests; callers must not open its returned path.
        """

        root, parts = self._policy_route(value, write=write)
        target = root.joinpath(*parts)
        _reject_symlink_path(root, target)
        return target

    def read_policy_file(self, value: str, *, offset: int, max_bytes: int) -> bytes:
        root, parts = self._policy_route(value, write=False)
        if not parts:
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        parent_fd = _open_directory_chain(root, parts[:-1], create=False)
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise MLEBenchLitePolicyPathError("policy path is unavailable")
                os.lseek(descriptor, offset, os.SEEK_SET)
                return os.read(descriptor, max_bytes)
            finally:
                os.close(descriptor)
        except MLEBenchLiteWorkspaceError:
            raise
        except OSError as exc:
            raise _policy_or_storage_error(exc) from exc
        finally:
            os.close(parent_fd)

    def atomic_write_policy_file(self, value: str, payload: bytes) -> None:
        root, parts = self._policy_route(value, write=True)
        if not parts:
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        parent_fd = _open_directory_chain(root, parts[:-1], create=True)
        temporary = f".{parts[-1]}.tmp-{uuid.uuid4().hex}"
        descriptor: int | None = None
        try:
            try:
                existing = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise MLEBenchLitePolicyPathError("policy path is unavailable")
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                temporary,
                parts[-1],
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        except MLEBenchLiteWorkspaceError:
            raise
        except OSError as exc:
            raise _policy_or_storage_error(exc) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(parent_fd)

    def _policy_route(self, value: str, *, write: bool) -> tuple[Path, tuple[str, ...]]:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        virtual = PurePosixPath(value)
        if not virtual.is_absolute() or ".." in virtual.parts:
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        routes = {
            ("/", "home", "data"): (self.public_root, True),
            ("/", "home", "workspace"): (self.workspace_root, False),
            ("/", "home", "submission"): (self.submission_root, False),
        }
        selected: tuple[Path, bool] | None = None
        parts: tuple[str, ...] = ()
        for prefix, route in routes.items():
            if virtual.parts[: len(prefix)] == prefix:
                selected = route
                parts = tuple(virtual.parts[len(prefix) :])
                break
        if selected is None or any(part in {"", ".", ".."} for part in parts):
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        root, read_only = selected
        if write and read_only:
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        if (
            self.mode != MODE_AMG_MEMORY
            and parts
            and parts[0] == ".agent_memory"
            and root == self.workspace_root
        ):
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        if write and root == self.submission_root and parts != ("submission.csv",):
            raise MLEBenchLitePolicyPathError("policy path is unavailable")
        return root, parts


@dataclass(frozen=True)
class HandoffStaging:
    episode_id: str
    directory: Path
    submission_sha256: str


@dataclass
class _TrackedStaging:
    value: HandoffStaging
    current_path: Path


@dataclass(frozen=True)
class _TrackedCreation:
    episode_id: str
    staging: Path
    final: Path


StageHook = Callable[[str, Path], None]


class WorkspaceManager:
    def __init__(
        self,
        episodes_root: Path,
        handoff_root: Path | None = None,
        *,
        stage_hook: StageHook | None = None,
    ) -> None:
        requested = Path(episodes_root)
        self.episodes_root = _private_root(requested, "episode root")
        requested_handoff = (
            requested.parent / f"{requested.name}-handoffs"
            if handoff_root is None
            else Path(handoff_root)
        )
        self.handoff_root = _private_root(requested_handoff, "handoff root")
        if (
            self.handoff_root == self.episodes_root
            or _is_relative_to(self.handoff_root, self.episodes_root)
            or _is_relative_to(self.episodes_root, self.handoff_root)
        ):
            raise MLEBenchLiteWorkspaceError(
                "episode and handoff roots must be disjoint"
            )
        self._stage_hook = stage_hook
        self._creation_lock = threading.Lock()
        self._pending_creations: dict[PendingCreationCleanup, _TrackedCreation] = {}
        self._staging_lock = threading.Lock()
        self._tracked_staging: dict[Path, _TrackedStaging] = {}

    def create(
        self,
        record: PublicTaskRecord,
        mode: str,
        *,
        resource_contract: Mapping[str, Any] | None = None,
        resource_contract_sha256: str | None = None,
    ) -> EpisodeWorkspace:
        if mode not in MODES:
            raise MLEBenchLiteWorkspaceError("unsupported evaluation mode")
        if resource_contract is None:
            resource_contract = build_resource_contract(
                max_actions=30,
                max_submission_bytes=100_000_000,
                max_shell_timeout_ms=DEFAULT_MAX_SHELL_TIMEOUT_MS,
                max_visible_output_bytes=65_536,
                submission_path=SUBMISSION_PATH,
                episode_timeout_ms=DEFAULT_EPISODE_TIMEOUT_MS,
                max_total_execution_ms=DEFAULT_MAX_TOTAL_EXECUTION_MS,
                cpu_limit_cores=DEFAULT_CPU_LIMIT_CORES,
                memory_limit_bytes=DEFAULT_MEMORY_LIMIT_BYTES,
                pids_limit=DEFAULT_PIDS_LIMIT,
                writable_bytes_limit=DEFAULT_WRITABLE_BYTES_LIMIT,
                writable_inodes_limit=DEFAULT_WRITABLE_INODES_LIMIT,
                gpu_count=DEFAULT_GPU_COUNT,
            )
        try:
            canonical_contract = validate_resource_contract(resource_contract)
            canonical_sha256 = _resource_contract_sha256(canonical_contract)
        except ValueError as exc:
            raise MLEBenchLiteWorkspaceError("resource contract is invalid") from exc
        if (
            resource_contract_sha256 is not None
            and resource_contract_sha256 != canonical_sha256
        ):
            raise MLEBenchLiteWorkspaceError("resource contract SHA256 mismatch")
        episode_id = uuid.uuid4().hex
        final_root = self.episodes_root / episode_id
        staging = self.episodes_root / f".creating-{episode_id}-{uuid.uuid4().hex}"
        pending_cleanup = self._track_pending_creation(
            _TrackedCreation(
                episode_id=episode_id,
                staging=staging,
                final=final_root,
            )
        )
        try:
            staging.mkdir(mode=0o700)
            self._stage("episode_created", staging)
            workspace_staging = staging / "workspace"
            submission_staging = staging / "submission"
            workspace_staging.mkdir(mode=0o700)
            self._stage("workspace_created", workspace_staging)
            submission_staging.mkdir(mode=0o700)
            self._stage("submission_created", submission_staging)
            if mode == MODE_AMG_MEMORY:
                memory = workspace_staging / ".agent_memory"
                memory.mkdir(mode=0o700)
                self._stage("memory_created", memory)
            self._stage("before_publish", staging)
            os.rename(staging, final_root)
            _fsync_directory(self.episodes_root)
        except Exception as exc:
            try:
                self.cleanup_pending_creation(pending_cleanup)
            except MLEBenchLiteWorkspaceError as cleanup_exc:
                raise MLEBenchLiteWorkspaceRollbackError(
                    pending_cleanup
                ) from cleanup_exc
            raise MLEBenchLiteWorkspaceError("cannot create episode workspace") from exc
        self._untrack_pending_creation(pending_cleanup)
        return EpisodeWorkspace(
            episode_id=episode_id,
            competition_id=record.competition_id,
            mode=mode,
            episode_root=final_root,
            workspace_root=final_root / "workspace",
            submission_root=final_root / "submission",
            public_root=record.public_root,
            public_tree_sha256=record.public_tree_sha256,
            resource_contract=MappingProxyType(canonical_contract),
            resource_contract_sha256=canonical_sha256,
        )

    def cleanup_pending_creation(
        self,
        pending_cleanup: PendingCreationCleanup,
    ) -> None:
        tracked = self._get_pending_creation(pending_cleanup)
        if tracked is None or not self._safe_creation_cleanup_target(tracked):
            raise MLEBenchLiteWorkspaceError("pending episode cleanup target is unsafe")
        failures: list[OSError] = []
        for candidate in (tracked.staging, tracked.final):
            try:
                _remove_private_tree(candidate)
            except OSError as exc:
                failures.append(exc)
        try:
            _fsync_directory(self.episodes_root)
        except OSError as exc:
            failures.append(exc)
        if failures:
            raise MLEBenchLiteWorkspaceError(
                "pending episode cleanup failed"
            ) from failures[0]
        self._untrack_pending_creation(pending_cleanup)

    def _track_pending_creation(
        self,
        tracked: _TrackedCreation,
    ) -> PendingCreationCleanup:
        handle = PendingCreationCleanup()
        with self._creation_lock:
            self._pending_creations[handle] = tracked
        return handle

    def _get_pending_creation(
        self,
        pending_cleanup: PendingCreationCleanup,
    ) -> _TrackedCreation | None:
        with self._creation_lock:
            return self._pending_creations.get(pending_cleanup)

    def _untrack_pending_creation(
        self,
        pending_cleanup: PendingCreationCleanup,
    ) -> None:
        with self._creation_lock:
            self._pending_creations.pop(pending_cleanup, None)

    def _safe_creation_cleanup_target(self, tracked: _TrackedCreation) -> bool:
        prefix = f".creating-{tracked.episode_id}-"
        suffix = tracked.staging.name.removeprefix(prefix)
        return (
            len(tracked.episode_id) == 32
            and all(character in "0123456789abcdef" for character in tracked.episode_id)
            and tracked.final == self.episodes_root / tracked.episode_id
            and tracked.final.parent == self.episodes_root
            and tracked.staging.parent == self.episodes_root
            and tracked.staging.name.startswith(prefix)
            and len(suffix) == 32
            and all(character in "0123456789abcdef" for character in suffix)
        )

    def remove(self, workspace: EpisodeWorkspace) -> None:
        root = workspace.episode_root
        if (
            root.parent != self.episodes_root
            or root.name != workspace.episode_id
            or len(root.name) != 32
        ):
            raise MLEBenchLiteWorkspaceError("episode cleanup target is unsafe")
        try:
            _remove_private_tree(root)
            _fsync_directory(self.episodes_root)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise MLEBenchLiteWorkspaceError(
                "episode workspace cleanup failed"
            ) from exc

    def stage_submission(
        self,
        workspace: EpisodeWorkspace,
        payload: bytes,
        submission_sha256: str,
    ) -> HandoffStaging:
        if hashlib.sha256(payload).hexdigest() != submission_sha256:
            raise MLEBenchLiteWorkspaceError("submission handoff digest drifted")
        staging = self.handoff_root / (
            f".staging-{workspace.episode_id}-{uuid.uuid4().hex}"
        )
        value = HandoffStaging(workspace.episode_id, staging, submission_sha256)
        self._track_staging(value)
        try:
            staging.mkdir(mode=0o700)
            _write_new_file(staging / "submission.csv", payload)
            _fsync_directory(staging)
        except OSError as exc:
            try:
                self.discard_staging(value)
            except MLEBenchLiteWorkspaceError as cleanup_exc:
                raise MLEBenchLiteWorkspaceError(
                    "host submission staging rollback failed"
                ) from cleanup_exc
            raise MLEBenchLiteWorkspaceError("host submission staging failed") from exc
        return value

    def discard_staging(self, staging: HandoffStaging) -> None:
        if (
            staging.directory.parent != self.handoff_root
            or not staging.directory.name.startswith(f".staging-{staging.episode_id}-")
        ):
            raise MLEBenchLiteWorkspaceError("handoff staging target is unsafe")
        tracked = self._get_tracked_staging(staging)
        target = staging.directory if tracked is None else tracked.current_path
        if not self._safe_staging_cleanup_target(staging, target):
            raise MLEBenchLiteWorkspaceError("handoff staging target is unsafe")
        try:
            _remove_private_tree(target)
            _fsync_directory(self.handoff_root)
        except OSError as exc:
            raise MLEBenchLiteWorkspaceError("handoff staging cleanup failed") from exc
        self._untrack_staging(staging)

    def publish_submission(
        self,
        staging: HandoffStaging,
        manifest_value: Mapping[str, Any],
    ) -> Path:
        final = self.handoff_root / staging.episode_id
        try:
            if self._get_tracked_staging(staging) is None:
                raise OSError("handoff staging is not tracked")
            if final.exists() or final.is_symlink():
                raise OSError("handoff already exists")
            manifest_payload = json.dumps(
                manifest_value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _write_new_file(staging.directory / "handoff.json", manifest_payload)
            for name in ("submission.csv", "handoff.json"):
                os.chmod(staging.directory / name, 0o400, follow_symlinks=False)
            _fsync_directory(staging.directory)
            # Darwin refuses to rename a directory after its owner write bit is
            # removed.  The staging directory remains owner-only and outside
            # every policy mount; its files are already sealed before publish.
            os.rename(staging.directory, final)
            self._move_tracked_staging(staging, final)
            os.chmod(final, 0o500, follow_symlinks=False)
            _fsync_directory(final)
            _fsync_directory(self.handoff_root)
        except (OSError, TypeError, ValueError) as exc:
            try:
                self.discard_staging(staging)
            except MLEBenchLiteWorkspaceError as cleanup_exc:
                raise MLEBenchLiteWorkspaceError(
                    "host submission publish rollback failed"
                ) from cleanup_exc
            raise MLEBenchLiteWorkspaceError("host submission publish failed") from exc
        self._untrack_staging(staging)
        return final / "submission.csv"

    def tracked_staging(self, episode_id: str) -> tuple[HandoffStaging, ...]:
        with self._staging_lock:
            return tuple(
                tracked.value
                for tracked in self._tracked_staging.values()
                if tracked.value.episode_id == episode_id
            )

    def _track_staging(self, staging: HandoffStaging) -> None:
        with self._staging_lock:
            if staging.directory in self._tracked_staging:
                raise MLEBenchLiteWorkspaceError("handoff staging identity collided")
            self._tracked_staging[staging.directory] = _TrackedStaging(
                value=staging,
                current_path=staging.directory,
            )

    def _get_tracked_staging(self, staging: HandoffStaging) -> _TrackedStaging | None:
        with self._staging_lock:
            tracked = self._tracked_staging.get(staging.directory)
            if tracked is None or tracked.value != staging:
                return None
            return _TrackedStaging(tracked.value, tracked.current_path)

    def _move_tracked_staging(self, staging: HandoffStaging, target: Path) -> None:
        with self._staging_lock:
            tracked = self._tracked_staging.get(staging.directory)
            if tracked is None or tracked.value != staging:
                raise MLEBenchLiteWorkspaceError("handoff staging is not tracked")
            tracked.current_path = target

    def _untrack_staging(self, staging: HandoffStaging) -> None:
        with self._staging_lock:
            tracked = self._tracked_staging.get(staging.directory)
            if tracked is not None and tracked.value == staging:
                self._tracked_staging.pop(staging.directory, None)

    def _safe_staging_cleanup_target(
        self, staging: HandoffStaging, target: Path
    ) -> bool:
        return target.parent == self.handoff_root and (
            target.name.startswith(f".staging-{staging.episode_id}-")
            or target.name == staging.episode_id
        )

    def verify_handoff_submission(
        self,
        workspace: EpisodeWorkspace,
        expected_manifest: Mapping[str, Any],
    ) -> Path:
        directory = self.handoff_root / workspace.episode_id
        try:
            metadata = directory.lstat()
            _require_owned_mode(metadata, expected_mode=0o500, label="handoff")
            directory_fd = os.open(directory, _DIRECTORY_FLAGS)
            try:
                manifest_payload = _read_regular_at(directory_fd, "handoff.json", 0o400)
                submission_payload = _read_regular_at(
                    directory_fd,
                    "submission.csv",
                    0o400,
                )
            finally:
                os.close(directory_fd)
            manifest = json.loads(
                manifest_payload.decode("utf-8"), object_pairs_hook=_unique_object
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise MLEBenchLiteWorkspaceError(
                "host handoff verification failed"
            ) from exc
        if not _strict_equal(manifest, dict(expected_manifest)):
            raise MLEBenchLiteWorkspaceError("host handoff manifest drifted")
        if hashlib.sha256(submission_payload).hexdigest() != manifest.get(
            "submission_sha256"
        ):
            raise MLEBenchLiteWorkspaceError("host handoff payload drifted")
        return directory / "submission.csv"

    def _stage(self, stage: str, path: Path) -> None:
        if self._stage_hook is not None:
            self._stage_hook(stage, path)


def _open_directory_chain(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
    try:
        descriptor = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise _policy_or_storage_error(exc) from exc
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise _policy_or_storage_error(exc) from exc


def _private_root(path: Path, label: str) -> Path:
    existed = False
    try:
        path.lstat()
        existed = True
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise MLEBenchLiteWorkspaceError(f"{label} is unavailable") from exc
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not existed:
            os.chmod(path, 0o700, follow_symlinks=False)
        metadata = path.lstat()
        _require_owned_mode(metadata, expected_mode=0o700, label=label)
        # ``lstat`` above rejects a symlink at the managed root itself.  Return
        # the canonical path so later parent/containment checks use one stable
        # spelling, but do not reject a trusted platform ancestor such as
        # macOS' ``/var -> /private/var`` mapping.
        return path.resolve(strict=True)
    except MLEBenchLiteWorkspaceError:
        raise
    except OSError as exc:
        raise MLEBenchLiteWorkspaceError(f"{label} is unavailable") from exc


def _require_owned_mode(
    metadata: os.stat_result,
    *,
    expected_mode: int,
    label: str,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode) and label in {
        "episode root",
        "handoff root",
        "handoff",
    }:
        raise MLEBenchLiteWorkspaceError(f"{label} must be a directory")
    if stat.S_ISLNK(metadata.st_mode):
        raise MLEBenchLiteWorkspaceError(f"{label} must not be a symlink")
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise MLEBenchLiteWorkspaceError(f"{label} must be owner-only")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise MLEBenchLiteWorkspaceError(f"{label} owner is unsafe")


def _remove_private_tree(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("private tree target is unsafe")
    if stat.S_IMODE(metadata.st_mode) == 0o500:
        os.chmod(path, 0o700, follow_symlinks=False)
    shutil.rmtree(path)


def _reject_symlink_path(root: Path, target: Path) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise MLEBenchLitePolicyPathError("policy path is unavailable") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _policy_or_storage_error(exc) from exc
        if stat.S_ISLNK(mode):
            raise MLEBenchLitePolicyPathError("policy path is unavailable")


def _policy_or_storage_error(exc: OSError) -> MLEBenchLiteWorkspaceError:
    if exc.errno in {
        errno.ENOENT,
        errno.ENOTDIR,
        errno.ELOOP,
        errno.EISDIR,
    }:
        return MLEBenchLitePolicyPathError("policy path is unavailable")
    return MLEBenchLiteWorkspaceError("workspace storage is unavailable")


def _write_new_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _read_regular_at(directory_fd: int, name: str, expected_mode: int) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != expected_mode
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise OSError("unsafe handoff file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
