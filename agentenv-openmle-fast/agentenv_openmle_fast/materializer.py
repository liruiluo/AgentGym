from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .dataset import OpenMLEFastRecord, tree_sha256

WORKSPACE_CONTRACT = "openmle_fast_public_workspace_v1"


class OpenMLEFastMaterializerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenMLEFastWorkspace:
    episode_id: str
    episode_root: Path
    policy_root: Path
    task_id: str
    public_tree_sha256: str
    contract: str = WORKSPACE_CONTRACT


class OpenMLEFastWorkspaceMaterializer:
    def __init__(self, episodes_root: Path | str) -> None:
        root = Path(episodes_root).expanduser().absolute()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or not root.is_dir():
            raise OpenMLEFastMaterializerError("episodes root must be a real directory")
        self.episodes_root = root.resolve()

    def materialize(self, record: OpenMLEFastRecord) -> OpenMLEFastWorkspace:
        episode_id = uuid.uuid4().hex
        episode_root = self.episodes_root / f"openmle-fast-episode-{episode_id}"
        policy_root = episode_root / "workspace"
        try:
            episode_root.mkdir(mode=0o700)
            policy_root.mkdir(mode=0o700)
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
            return OpenMLEFastWorkspace(
                episode_id=episode_id,
                episode_root=episode_root,
                policy_root=policy_root,
                task_id=record.task_id,
                public_tree_sha256=digest,
            )
        except BaseException:
            if episode_root.exists() and not episode_root.is_symlink():
                shutil.rmtree(episode_root)
            raise

    def close(self, workspace: OpenMLEFastWorkspace) -> None:
        root = workspace.episode_root.absolute()
        if root.parent != self.episodes_root:
            raise OpenMLEFastMaterializerError(
                "refusing to clean an episode outside the configured root"
            )
        if not root.name.startswith("openmle-fast-episode-"):
            raise OpenMLEFastMaterializerError("unrecognized OpenMLE-fast episode path")
        if root.is_symlink():
            raise OpenMLEFastMaterializerError("episode root became a symlink")
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

    def _remove_owned_root(self, root: Path) -> None:
        if not root.exists():
            return
        for current, directories, files in os.walk(
            root, topdown=False, followlinks=False
        ):
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
        shutil.rmtree(root)


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
