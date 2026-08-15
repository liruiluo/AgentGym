from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .identity import OfficialLiteIdentity

PUBLIC_MANIFEST_SCHEMA = "mlebench_lite_public_manifest_v1"


class MLEBenchLiteDatasetError(RuntimeError):
    """Prepared-data identity or isolation is not safe to expose."""


@dataclass(frozen=True)
class PublicTaskRecord:
    competition_id: str
    public_root: Path
    public_tree_sha256: str


class MLEBenchLiteDataset(Sequence[PublicTaskRecord]):
    def __init__(
        self,
        *,
        identity: OfficialLiteIdentity,
        records: tuple[PublicTaskRecord, ...],
        public_manifest_sha256: str,
        data_root: Path,
    ) -> None:
        self.identity = identity
        self._records = records
        self.public_manifest_sha256 = public_manifest_sha256
        self._data_root = data_root

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]

    def __iter__(self) -> Iterator[PublicTaskRecord]:
        return iter(self._records)


def load_lite_dataset(
    *,
    identity: OfficialLiteIdentity,
    manifest_path: Path,
    expected_manifest_sha256: str,
    data_root: Path,
    forbidden_roots: Iterable[Path] = (),
) -> MLEBenchLiteDataset:
    """Load public records from the official prepared-cache layout.

    Official preparation stores each task at
    ``<data>/<competition>/prepared/public`` with a private sibling. The
    adapter validates both but returns records containing only the public
    source and its externally pinned tree attestation.
    """

    _require_sha256(expected_manifest_sha256, "expected manifest SHA256")
    manifest = _regular_file(Path(manifest_path), "public manifest")
    payload = manifest.read_bytes()
    manifest_digest = hashlib.sha256(payload).hexdigest()
    if manifest_digest != expected_manifest_sha256:
        raise MLEBenchLiteDatasetError("public manifest SHA256 mismatch")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MLEBenchLiteDatasetError("public manifest is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "upstream_commit",
        "split_sha256",
        "tasks",
    }:
        raise MLEBenchLiteDatasetError("public manifest fields drifted")
    if value["schema"] != PUBLIC_MANIFEST_SCHEMA:
        raise MLEBenchLiteDatasetError("public manifest schema drifted")
    if value["upstream_commit"] != identity.upstream_commit:
        raise MLEBenchLiteDatasetError("public manifest upstream commit drifted")
    if value["split_sha256"] != identity.split_sha256:
        raise MLEBenchLiteDatasetError("public manifest split SHA256 drifted")
    tasks = value["tasks"]
    if not isinstance(tasks, list) or len(tasks) != len(identity.competition_ids):
        raise MLEBenchLiteDatasetError("public manifest task count drifted")

    root = _regular_directory(Path(data_root), "prepared data root")
    forbidden = tuple(
        _regular_directory(Path(path), "forbidden root") for path in forbidden_roots
    )
    records: list[PublicTaskRecord] = []
    public_paths: list[Path] = []
    private_paths: list[Path] = []
    for index, (task, expected_id) in enumerate(zip(tasks, identity.competition_ids)):
        if not isinstance(task, dict) or set(task) != {
            "competition_id",
            "public_relative_path",
            "private_relative_path",
            "public_files",
            "public_tree_sha256",
        }:
            raise MLEBenchLiteDatasetError(f"task manifest fields drifted at {index}")
        if task["competition_id"] != expected_id:
            raise MLEBenchLiteDatasetError("task manifest membership or order drifted")
        expected_public = f"{expected_id}/prepared/public"
        expected_private = f"{expected_id}/prepared/private"
        if task["public_relative_path"] != expected_public:
            raise MLEBenchLiteDatasetError("task public path crossed its competition")
        if task["private_relative_path"] != expected_private:
            raise MLEBenchLiteDatasetError("task private path crossed its competition")
        public_relative = _safe_relative(task["public_relative_path"])
        private_relative = _safe_relative(task["private_relative_path"])
        requested_public = root / public_relative
        requested_private = root / private_relative
        _reject_symlink_components(root, requested_public, "task public root")
        _reject_symlink_components(root, requested_private, "task private root")
        public_path = _regular_directory(requested_public, "task public root")
        private_path = _regular_directory(requested_private, "task private root")
        _require_disjoint(public_path, private_path, "task public/private roots")
        for forbidden_root in forbidden:
            _require_disjoint(public_path, forbidden_root, "public/forbidden roots")
        _reject_tree_symlinks(public_path, "task public source")
        _reject_tree_symlinks(private_path, "task private source")
        public_files = _verify_public_inventory(public_path, task["public_files"])
        public_tree_sha256 = task["public_tree_sha256"]
        _require_sha256(public_tree_sha256, "public tree SHA256")
        if public_tree_sha256 != _canonical_sha256(public_files):
            raise MLEBenchLiteDatasetError("public tree attestation drifted")
        records.append(
            PublicTaskRecord(
                competition_id=expected_id,
                public_root=public_path,
                public_tree_sha256=public_tree_sha256,
            )
        )
        public_paths.append(public_path)
        private_paths.append(private_path)

    for index, public_path in enumerate(public_paths):
        for other_index, other_path in enumerate((*public_paths, *private_paths)):
            if other_index == index:
                continue
            _require_disjoint(public_path, other_path, "cross-task source roots")
    return MLEBenchLiteDataset(
        identity=identity,
        records=tuple(records),
        public_manifest_sha256=manifest_digest,
        data_root=root,
    )


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MLEBenchLiteDatasetError("manifest path must be non-empty text")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise MLEBenchLiteDatasetError("manifest path must be a safe relative path")
    if relative.as_posix() != value:
        raise MLEBenchLiteDatasetError("manifest path must be canonical POSIX text")
    return Path(*relative.parts)


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MLEBenchLiteDatasetError(f"{label} must be lowercase SHA256")


def _regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise MLEBenchLiteDatasetError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise MLEBenchLiteDatasetError(f"{label} must be a non-symlink regular file")
    return path.resolve(strict=True)


def _regular_directory(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise MLEBenchLiteDatasetError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise MLEBenchLiteDatasetError(f"{label} must be a non-symlink directory")
    return path.resolve(strict=True)


def _reject_tree_symlinks(root: Path, label: str) -> None:
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise MLEBenchLiteDatasetError(f"{label} contains a symlink")
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif not entry.is_file(follow_symlinks=False):
                            raise MLEBenchLiteDatasetError(
                                f"{label} contains a special file"
                            )
                    except OSError as exc:
                        raise MLEBenchLiteDatasetError(
                            f"{label} cannot be attested"
                        ) from exc
        except OSError as exc:
            raise MLEBenchLiteDatasetError(f"{label} cannot be scanned") from exc


def _verify_public_inventory(root: Path, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise MLEBenchLiteDatasetError("public file inventory must be non-empty")
    expected: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise MLEBenchLiteDatasetError(
                f"public file inventory fields drifted at {index}"
            )
        relative = _safe_relative(item["path"])
        relative_posix = PurePosixPath(*relative.parts).as_posix()
        if relative_posix in expected_paths:
            raise MLEBenchLiteDatasetError("public file inventory contains duplicates")
        size = item["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise MLEBenchLiteDatasetError("public file size must be non-negative")
        _require_sha256(item["sha256"], "public file SHA256")
        expected_paths.add(relative_posix)
        expected.append(
            {"path": relative_posix, "size": size, "sha256": item["sha256"]}
        )
    if expected != sorted(expected, key=lambda item: item["path"]):
        raise MLEBenchLiteDatasetError("public file inventory must be sorted")

    actual_paths: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            continue
        actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise MLEBenchLiteDatasetError("public file inventory membership drifted")
    for item in expected:
        path = root / Path(*PurePosixPath(item["path"]).parts)
        descriptor: int | None = None
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != item["size"]
            ):
                raise MLEBenchLiteDatasetError("public file size/type/link drifted")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        except OSError as exc:
            raise MLEBenchLiteDatasetError("public file cannot be hashed") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if _stable_stat(before) != _stable_stat(after):
            raise MLEBenchLiteDatasetError("public file changed while hashing")
        if digest.hexdigest() != item["sha256"]:
            raise MLEBenchLiteDatasetError("public file SHA256 drifted")
    return expected


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_disjoint(first: Path, second: Path, label: str) -> None:
    if (
        first == second
        or _is_relative_to(first, second)
        or _is_relative_to(second, first)
    ):
        raise MLEBenchLiteDatasetError(f"{label} overlap or nest")


def _reject_symlink_components(root: Path, target: Path, label: str) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise MLEBenchLiteDatasetError(f"{label} escaped prepared data") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise MLEBenchLiteDatasetError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise MLEBenchLiteDatasetError(f"{label} contains a symlink")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
