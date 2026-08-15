from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .dataset import OpenMLEFastRecord, tree_sha256

WORKSPACE_CONTRACT = "openmle_fast_public_workspace_v1"
_MOUNT_OWNERSHIP_CONTRACT = "openmle_fast_tmpfs_mount_ownership_v1"
_MOUNT_MARKER = ".openmle-fast-workspace-owner.json"
_REQUIRED_MOUNT_OPTIONS = frozenset({"noexec", "nosuid", "nodev"})


class OpenMLEFastMaterializerError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceMountIdentity:
    mount_id: str
    device: int
    filesystem: str
    options: tuple[str, ...]
    capacity_bytes: int
    inode_capacity: int


class WorkspaceMountBackend(Protocol):
    def mount(
        self,
        path: Path,
        *,
        workspace_bytes: int,
        max_files: int,
    ) -> WorkspaceMountIdentity: ...

    def inspect(self, path: Path) -> WorkspaceMountIdentity | None: ...

    def unmount(
        self,
        path: Path,
        *,
        expected: WorkspaceMountIdentity,
        workspace_bytes: int,
        max_files: int,
    ) -> None: ...


@dataclass(frozen=True)
class OpenMLEFastWorkspace:
    episode_id: str
    episode_root: Path
    policy_root: Path
    task_id: str
    public_tree_sha256: str
    contract: str = WORKSPACE_CONTRACT


class OpenMLEFastWorkspaceMaterializer:
    def __init__(
        self,
        episodes_root: Path | str,
        *,
        runner_workspace_parent: Path | str,
        workspace_bytes: int,
        max_files: int,
        mount_backend: WorkspaceMountBackend | None = None,
    ) -> None:
        if type(workspace_bytes) is not int or workspace_bytes <= 0:
            raise OpenMLEFastMaterializerError("workspace byte cap must be positive")
        if type(max_files) is not int or max_files <= 0:
            raise OpenMLEFastMaterializerError("workspace inode cap must be positive")
        parent = _real_directory(
            Path(runner_workspace_parent), "runner workspace parent"
        )
        root = Path(episodes_root).expanduser().absolute()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = _real_directory(root, "episodes root")
        try:
            contained = os.path.commonpath((str(root), str(parent))) == str(parent)
        except ValueError:
            contained = False
        if not contained:
            raise OpenMLEFastMaterializerError(
                "episodes root must be inside the attested runner workspace parent"
            )
        self.runner_workspace_parent = parent
        self.episodes_root = root
        self.workspace_bytes = workspace_bytes
        self.max_files = max_files
        self.mount_backend = mount_backend or LinuxTmpfsWorkspaceMountBackend()

    def materialize(self, record: OpenMLEFastRecord) -> OpenMLEFastWorkspace:
        episode_id = uuid.uuid4().hex
        episode_root = self.episodes_root / f"openmle-fast-episode-{episode_id}"
        policy_root = episode_root / "workspace"
        mounted_identity: WorkspaceMountIdentity | None = None
        try:
            episode_root.mkdir(mode=0o700)
            marker = _MountOwnership(
                contract=_MOUNT_OWNERSHIP_CONTRACT,
                episode_id=episode_id,
                policy_root_name="workspace",
                workspace_bytes=self.workspace_bytes,
                max_files=self.max_files,
                adopted_by_runner=False,
                mount=None,
            )
            _write_marker_new(episode_root / _MOUNT_MARKER, marker)
            policy_root.mkdir(mode=0o700)
            identity = self.mount_backend.mount(
                policy_root,
                workspace_bytes=self.workspace_bytes,
                max_files=self.max_files,
            )
            mounted_identity = identity
            _validate_mount_identity(
                identity,
                workspace_bytes=self.workspace_bytes,
                max_files=self.max_files,
                require_writable=True,
            )
            _replace_marker(
                episode_root / _MOUNT_MARKER,
                replace(marker, mount=identity),
            )
            for relative, expected_sha256 in sorted(record.public_file_sha256.items()):
                destination = policy_root / PurePosixPath(relative)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = record.public_source_root / PurePosixPath(relative)
                payload = _read_attested_file(source, expected_sha256)
                _write_new_regular(destination, payload, 0o444)
            digest = _visible_tree_sha256(policy_root)
            if digest != record.public_tree_sha256:
                raise OpenMLEFastMaterializerError(
                    "materialized public-tree SHA256 does not match the frozen task"
                )
            for directory in sorted(
                (path for path in policy_root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            _require_same_mount(
                self.mount_backend.inspect(policy_root),
                identity,
                workspace_bytes=self.workspace_bytes,
                max_files=self.max_files,
                require_writable=True,
            )
            return OpenMLEFastWorkspace(
                episode_id=episode_id,
                episode_root=episode_root,
                policy_root=policy_root,
                task_id=record.task_id,
                public_tree_sha256=digest,
            )
        except BaseException:
            self._cleanup_failed_materialization(
                episode_root,
                policy_root,
                mounted_identity=mounted_identity,
            )
            raise

    def mark_adopted_by_runner(self, workspace: OpenMLEFastWorkspace) -> None:
        root = self._validate_workspace_root(workspace)
        marker_path = root / _MOUNT_MARKER
        marker = _read_marker(marker_path)
        identity = self.mount_backend.inspect(workspace.policy_root)
        if marker.mount is None:
            raise OpenMLEFastMaterializerError("workspace mount identity is missing")
        _require_same_mount(
            identity,
            marker.mount,
            workspace_bytes=self.workspace_bytes,
            max_files=self.max_files,
            require_writable=True,
        )
        if not marker.adopted_by_runner:
            _replace_marker(marker_path, replace(marker, adopted_by_runner=True))

    def is_adopted_by_runner(self, workspace: OpenMLEFastWorkspace) -> bool:
        root = self._validate_workspace_root(workspace)
        return _read_marker(root / _MOUNT_MARKER).adopted_by_runner

    def close(self, workspace: OpenMLEFastWorkspace) -> None:
        root = self._validate_workspace_root(workspace)
        self._remove_owned_root(root)

    def reconcile_orphans(self) -> int:
        removed = 0
        for root in tuple(self.episodes_root.iterdir()):
            if root.is_symlink() or not root.is_dir():
                raise OpenMLEFastMaterializerError(
                    "episodes root contains an unowned entry"
                )
            if not root.name.startswith("openmle-fast-episode-"):
                raise OpenMLEFastMaterializerError(
                    "episodes root contains an unrecognized directory"
                )
            self._remove_owned_root(root)
            removed += 1
        return removed

    def _validate_workspace_root(self, workspace: OpenMLEFastWorkspace) -> Path:
        root = workspace.episode_root.absolute()
        if root.parent != self.episodes_root:
            raise OpenMLEFastMaterializerError(
                "refusing to clean an episode outside the configured root"
            )
        if root.name != f"openmle-fast-episode-{workspace.episode_id}":
            raise OpenMLEFastMaterializerError("unrecognized OpenMLE-fast episode path")
        if workspace.policy_root.absolute() != root / "workspace":
            raise OpenMLEFastMaterializerError("workspace policy root drifted")
        if root.is_symlink():
            raise OpenMLEFastMaterializerError("episode root became a symlink")
        return root

    def _cleanup_failed_materialization(
        self,
        episode_root: Path,
        policy_root: Path,
        *,
        mounted_identity: WorkspaceMountIdentity | None,
    ) -> None:
        if not episode_root.exists() or episode_root.is_symlink():
            return
        marker_path = episode_root / _MOUNT_MARKER
        if marker_path.exists() and not marker_path.is_symlink():
            marker = _read_marker(marker_path)
            actual = self.mount_backend.inspect(policy_root)
            if actual is not None and marker.mount is None:
                if mounted_identity is None:
                    raise OpenMLEFastMaterializerError(
                        "failed materialization left an unattested workspace mount"
                    )
                _require_same_mount(
                    actual,
                    mounted_identity,
                    workspace_bytes=self.workspace_bytes,
                    max_files=self.max_files,
                    require_writable=False,
                )
                self.mount_backend.unmount(
                    policy_root,
                    expected=mounted_identity,
                    workspace_bytes=self.workspace_bytes,
                    max_files=self.max_files,
                )
                if self.mount_backend.inspect(policy_root) is not None:
                    raise OpenMLEFastMaterializerError(
                        "failed materialization workspace unmount did not complete"
                    )
            self._remove_owned_root(episode_root)
            return
        if self.mount_backend.inspect(policy_root) is not None:
            raise OpenMLEFastMaterializerError(
                "failed materialization left an unattested workspace mount"
            )
        _make_tree_owner_writable(episode_root)
        shutil.rmtree(episode_root)

    def _remove_owned_root(self, root: Path) -> None:
        if not root.exists():
            return
        if root.parent != self.episodes_root or not root.name.startswith(
            "openmle-fast-episode-"
        ):
            raise OpenMLEFastMaterializerError("refusing to remove an unowned root")
        if root.is_symlink() or not root.is_dir():
            raise OpenMLEFastMaterializerError("owned episode root is unsafe")
        marker = _read_marker(root / _MOUNT_MARKER)
        if root.name != f"openmle-fast-episode-{marker.episode_id}":
            raise OpenMLEFastMaterializerError("workspace ownership marker drifted")
        if (
            marker.workspace_bytes != self.workspace_bytes
            or marker.max_files != self.max_files
        ):
            raise OpenMLEFastMaterializerError(
                "workspace ownership marker caps differ from this materializer"
            )
        policy_root = root / marker.policy_root_name
        identity = self.mount_backend.inspect(policy_root)
        if identity is not None:
            if marker.mount is None:
                raise OpenMLEFastMaterializerError(
                    "workspace mount has no recorded ownership identity"
                )
            _require_same_mount(
                identity,
                marker.mount,
                workspace_bytes=self.workspace_bytes,
                max_files=self.max_files,
                require_writable=False,
            )
            if marker.adopted_by_runner:
                raise OpenMLEFastMaterializerError(
                    "adopted workspace must be torn down by the exact runner first"
                )
            self.mount_backend.unmount(
                policy_root,
                expected=marker.mount,
                workspace_bytes=self.workspace_bytes,
                max_files=self.max_files,
            )
            if self.mount_backend.inspect(policy_root) is not None:
                raise OpenMLEFastMaterializerError("workspace unmount did not complete")
        _make_tree_owner_writable(root)
        shutil.rmtree(root)


@dataclass(frozen=True)
class _MountOwnership:
    contract: str
    episode_id: str
    policy_root_name: str
    workspace_bytes: int
    max_files: int
    adopted_by_runner: bool
    mount: WorkspaceMountIdentity | None


class LinuxTmpfsWorkspaceMountBackend:
    def mount(
        self,
        path: Path,
        *,
        workspace_bytes: int,
        max_files: int,
    ) -> WorkspaceMountIdentity:
        if os.name != "posix" or not Path("/proc/self/mountinfo").is_file():
            raise OpenMLEFastMaterializerError(
                "dedicated tmpfs workspaces require Linux mount namespaces"
            )
        if self.inspect(path) is not None:
            raise OpenMLEFastMaterializerError("workspace path is already a mount")
        options = (
            f"size={workspace_bytes},nr_inodes={max_files},mode=0777,"
            "noexec,nosuid,nodev"
        )
        try:
            subprocess.run(
                [
                    "/usr/bin/mount",
                    "-t",
                    "tmpfs",
                    "-o",
                    options,
                    "openmle-fast-tmpfs",
                    str(path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10.0,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OpenMLEFastMaterializerError(
                "cannot mount the dedicated OpenMLE-fast tmpfs"
            ) from exc
        try:
            identity = self.inspect(path)
            if identity is None:
                raise OpenMLEFastMaterializerError("tmpfs mount did not appear")
            _validate_mount_identity(
                identity,
                workspace_bytes=workspace_bytes,
                max_files=max_files,
                require_writable=True,
            )
            return identity
        except BaseException as admission_error:
            try:
                self._rollback_new_mount(path)
            except BaseException as rollback_error:
                raise OpenMLEFastMaterializerError(
                    "invalid workspace tmpfs could not be rolled back"
                ) from rollback_error
            raise admission_error

    def _rollback_new_mount(self, path: Path) -> None:
        """Undo the exact mount just created before it can acquire a marker."""

        try:
            subprocess.run(
                ["/usr/bin/umount", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10.0,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OpenMLEFastMaterializerError(
                "cannot roll back the newly mounted OpenMLE-fast tmpfs"
            ) from exc
        if self.inspect(path) is not None:
            raise OpenMLEFastMaterializerError(
                "newly mounted OpenMLE-fast tmpfs remains after rollback"
            )

    def inspect(self, path: Path) -> WorkspaceMountIdentity | None:
        try:
            target = str(path.resolve(strict=True))
        except OSError:
            return None
        matches: list[tuple[str, str, str, set[str]]] = []
        try:
            lines = (
                Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
            )
        except OSError as exc:
            raise OpenMLEFastMaterializerError("cannot inspect Linux mounts") from exc
        for line in lines:
            fields = line.split()
            if "-" not in fields or len(fields) < 10:
                continue
            separator = fields.index("-")
            mountpoint = _decode_mount_field(fields[4])
            if os.path.realpath(mountpoint) != target:
                continue
            options = set(fields[5].split(","))
            options.update(fields[separator + 3].split(","))
            matches.append((fields[0], fields[2], fields[separator + 1], options))
        if not matches:
            return None
        if len(matches) != 1:
            raise OpenMLEFastMaterializerError(
                "workspace must have one exact dedicated mount"
            )
        mount_id, _major_minor, filesystem, options = matches[0]
        try:
            info = os.stat(target, follow_symlinks=False)
            filesystem_info = os.statvfs(target)
        except OSError as exc:
            raise OpenMLEFastMaterializerError(
                "cannot inspect workspace tmpfs"
            ) from exc
        return WorkspaceMountIdentity(
            mount_id=mount_id,
            device=int(info.st_dev),
            filesystem=filesystem,
            options=tuple(sorted(options)),
            capacity_bytes=int(filesystem_info.f_frsize * filesystem_info.f_blocks),
            inode_capacity=int(filesystem_info.f_files),
        )

    def unmount(
        self,
        path: Path,
        *,
        expected: WorkspaceMountIdentity,
        workspace_bytes: int,
        max_files: int,
    ) -> None:
        _require_same_mount(
            self.inspect(path),
            expected,
            workspace_bytes=workspace_bytes,
            max_files=max_files,
            require_writable=False,
        )
        try:
            subprocess.run(
                ["/usr/bin/umount", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10.0,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OpenMLEFastMaterializerError(
                "cannot unmount the owned OpenMLE-fast tmpfs"
            ) from exc
        if self.inspect(path) is not None:
            raise OpenMLEFastMaterializerError("workspace tmpfs remains mounted")


def _real_directory(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise OpenMLEFastMaterializerError(f"{label} must be a real directory")
    return absolute.resolve()


def _validate_mount_identity(
    identity: WorkspaceMountIdentity,
    *,
    workspace_bytes: int,
    max_files: int,
    require_writable: bool,
) -> None:
    options = set(identity.options)
    if identity.filesystem != "tmpfs":
        raise OpenMLEFastMaterializerError("workspace mount is not tmpfs")
    if not _REQUIRED_MOUNT_OPTIONS.issubset(options):
        raise OpenMLEFastMaterializerError("workspace tmpfs lacks hardened options")
    if require_writable and "rw" not in options:
        raise OpenMLEFastMaterializerError("active workspace tmpfs is not writable")
    if (
        identity.capacity_bytes <= 0
        or identity.capacity_bytes > workspace_bytes
        or identity.inode_capacity <= 0
        or identity.inode_capacity > max_files + 1024
    ):
        raise OpenMLEFastMaterializerError("workspace tmpfs exceeds its frozen cap")
    if not identity.mount_id or identity.device < 0:
        raise OpenMLEFastMaterializerError("workspace mount identity is invalid")


def _require_same_mount(
    actual: WorkspaceMountIdentity | None,
    expected: WorkspaceMountIdentity,
    *,
    workspace_bytes: int,
    max_files: int,
    require_writable: bool,
) -> None:
    if actual is None:
        raise OpenMLEFastMaterializerError("owned workspace mount disappeared")
    _validate_mount_identity(
        actual,
        workspace_bytes=workspace_bytes,
        max_files=max_files,
        require_writable=require_writable,
    )
    if actual.mount_id != expected.mount_id or actual.device != expected.device:
        raise OpenMLEFastMaterializerError("workspace mount identity drifted")


def _marker_payload(marker: _MountOwnership) -> dict[str, Any]:
    value = asdict(marker)
    return value


def _marker_from_payload(value: Any) -> _MountOwnership:
    required = {
        "contract",
        "episode_id",
        "policy_root_name",
        "workspace_bytes",
        "max_files",
        "adopted_by_runner",
        "mount",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise OpenMLEFastMaterializerError("workspace ownership marker is invalid")
    if value["contract"] != _MOUNT_OWNERSHIP_CONTRACT:
        raise OpenMLEFastMaterializerError("workspace ownership contract drifted")
    episode_id = value["episode_id"]
    if (
        not isinstance(episode_id, str)
        or len(episode_id) != 32
        or any(character not in "0123456789abcdef" for character in episode_id)
    ):
        raise OpenMLEFastMaterializerError("workspace episode identity is invalid")
    if value["policy_root_name"] != "workspace":
        raise OpenMLEFastMaterializerError("workspace mountpoint name drifted")
    if type(value["workspace_bytes"]) is not int or value["workspace_bytes"] <= 0:
        raise OpenMLEFastMaterializerError("workspace marker byte cap is invalid")
    if type(value["max_files"]) is not int or value["max_files"] <= 0:
        raise OpenMLEFastMaterializerError("workspace marker inode cap is invalid")
    if type(value["adopted_by_runner"]) is not bool:
        raise OpenMLEFastMaterializerError("workspace adoption marker is invalid")
    mount_value = value["mount"]
    mount = None
    if mount_value is not None:
        mount_required = {
            "mount_id",
            "device",
            "filesystem",
            "options",
            "capacity_bytes",
            "inode_capacity",
        }
        if not isinstance(mount_value, Mapping) or set(mount_value) != mount_required:
            raise OpenMLEFastMaterializerError("workspace mount marker is invalid")
        options = mount_value["options"]
        if not isinstance(options, list) or any(
            not isinstance(option, str) for option in options
        ):
            raise OpenMLEFastMaterializerError("workspace mount options are invalid")
        mount = WorkspaceMountIdentity(
            mount_id=str(mount_value["mount_id"]),
            device=int(mount_value["device"]),
            filesystem=str(mount_value["filesystem"]),
            options=tuple(options),
            capacity_bytes=int(mount_value["capacity_bytes"]),
            inode_capacity=int(mount_value["inode_capacity"]),
        )
    return _MountOwnership(
        contract=value["contract"],
        episode_id=episode_id,
        policy_root_name=value["policy_root_name"],
        workspace_bytes=value["workspace_bytes"],
        max_files=value["max_files"],
        adopted_by_runner=value["adopted_by_runner"],
        mount=mount,
    )


def _read_marker(path: Path) -> _MountOwnership:
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OpenMLEFastMaterializerError(
                "workspace ownership marker is not an independent file"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenMLEFastMaterializerError(
            "cannot read workspace ownership marker"
        ) from exc
    marker = _marker_from_payload(value)
    return marker


def _write_marker_new(path: Path, marker: _MountOwnership) -> None:
    payload = (
        json.dumps(_marker_payload(marker), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    _write_new_regular(path, payload, 0o600)
    _fsync_directory(path.parent)


def _replace_marker(path: Path, marker: _MountOwnership) -> None:
    current = _read_marker(path)
    if (
        current.contract != marker.contract
        or current.episode_id != marker.episode_id
        or current.policy_root_name != marker.policy_root_name
        or current.workspace_bytes != marker.workspace_bytes
        or current.max_files != marker.max_files
    ):
        raise OpenMLEFastMaterializerError("workspace marker identity changed")
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (
        json.dumps(_marker_payload(marker), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    try:
        _write_new_regular(temporary, payload, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _make_tree_owner_writable(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(current) / name
            info = os.lstat(path)
            if not stat.S_ISLNK(info.st_mode) and (
                not stat.S_ISREG(info.st_mode) or info.st_nlink == 1
            ):
                path.chmod(0o600)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(0o700)
    root.chmod(0o700)


def _read_attested_file(path: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OpenMLEFastMaterializerError(
            "cannot open an attested public file"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OpenMLEFastMaterializerError(
                "attested public input is not an independent regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise OpenMLEFastMaterializerError(
            "public input changed after dataset attestation"
        )
    return payload


def _write_new_regular(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _visible_tree_sha256(policy_root: Path) -> str:
    entries: list[dict[str, object]] = []
    for path in sorted(policy_root.rglob("*")):
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OpenMLEFastMaterializerError(
                "materialized policy tree contains a non-regular file"
            )
        relative = path.relative_to(policy_root).as_posix()
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mode": stat.S_IMODE(info.st_mode),
            }
        )
    return tree_sha256(entries)
