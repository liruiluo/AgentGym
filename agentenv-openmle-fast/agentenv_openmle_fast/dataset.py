from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "openmle_fast_public_manifest_v1"
CONTRACT_VERSION = "openmle_fast_v1"
GATE_ONLY_ROLE = "gate_only"


class OpenMLEFastDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetProvenance:
    schema: str
    contract_version: str
    panel_id: str
    release_revision: str
    role: str
    manifest_path: Path
    manifest_sha256: str
    task_id_list_sha256: str
    compact_panel_sha256: str
    task_count: int

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": self.contract_version,
            "panel_id": self.panel_id,
            "release_revision": self.release_revision,
            "role": self.role,
            "manifest_sha256": self.manifest_sha256,
            "task_id_list_sha256": self.task_id_list_sha256,
            "compact_panel_sha256": self.compact_panel_sha256,
            "task_count": self.task_count,
        }


@dataclass(frozen=True)
class OpenMLEFastRecord:
    data_idx: int
    task_id: str
    source_family: str
    package_root: Path
    public_source_root: Path
    archive_path: Path
    archive_sha256: str
    public_tree_sha256: str
    package_identity_sha256: str
    grader_binding: str
    grader_binding_sha256: str
    baseline_score: float
    ideal_score: float
    higher_is_better: bool
    license: Mapping[str, Any]
    provenance: Mapping[str, Any]
    public_task: Mapping[str, Any]
    task_spec_sha256: str
    public_file_sha256: Mapping[str, str]

    @property
    def task_markdown(self) -> str:
        return str(self.public_task["task_markdown"])

    def public_metadata(self) -> dict[str, Any]:
        return {
            "data_idx": self.data_idx,
            "task_id": self.task_id,
            "source_family": self.source_family,
            "archive_sha256": self.archive_sha256,
            "public_tree_sha256": self.public_tree_sha256,
            "package_identity_sha256": self.package_identity_sha256,
            "grader_binding_sha256": self.grader_binding_sha256,
            "baseline_score": self.baseline_score,
            "ideal_score": self.ideal_score,
            "higher_is_better": self.higher_is_better,
            "license": dict(self.license),
            "provenance": dict(self.provenance),
            "public_task": dict(self.public_task),
            "task_spec_sha256": self.task_spec_sha256,
        }


class OpenMLEFastDataset:
    """Read and attest an ordered, externally frozen OpenMLE-fast manifest."""

    def __init__(
        self,
        *,
        manifest_path: Path | str,
        package_root: Path | str,
        archive_root: Path | str,
        expected_manifest_sha256: str,
        expected_release_revision: str,
        expected_role: str,
    ) -> None:
        self.manifest_path = _real_file(Path(manifest_path), "task manifest")
        self.package_root = _real_directory(Path(package_root), "package root")
        self.archive_root = _real_directory(Path(archive_root), "archive root")
        raw_manifest = _read_regular_bytes(self.manifest_path, "task manifest")
        actual_manifest_sha256 = _sha256(raw_manifest)
        if actual_manifest_sha256 != _sha256_text(
            expected_manifest_sha256, "expected task-manifest SHA256"
        ):
            raise OpenMLEFastDatasetError("task-manifest SHA256 mismatch")
        try:
            manifest = _strict_json_loads(raw_manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenMLEFastDatasetError(
                "task manifest is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(manifest, dict):
            raise OpenMLEFastDatasetError("task manifest must be a JSON object")
        schema = _text(manifest, "schema")
        if schema != MANIFEST_SCHEMA:
            raise OpenMLEFastDatasetError(f"unsupported task-manifest schema: {schema}")
        contract_version = _text(manifest, "contract_version")
        if contract_version != CONTRACT_VERSION:
            raise OpenMLEFastDatasetError(
                f"unsupported OpenMLE-fast contract: {contract_version}"
            )
        panel_id = _text(manifest, "panel_id")
        release_revision = _git_revision(manifest.get("openmle_tasks_revision"))
        if _git_revision(manifest.get("release_revision")) != release_revision:
            raise OpenMLEFastDatasetError("manifest release identities disagree")
        if release_revision != _git_revision(expected_release_revision):
            raise OpenMLEFastDatasetError("OpenMLE release revision mismatch")
        role = _text(manifest, "role")
        if role != expected_role:
            raise OpenMLEFastDatasetError("task-manifest role mismatch")
        task_count = manifest.get("task_count")
        if type(task_count) is not int or task_count <= 0:
            raise OpenMLEFastDatasetError("task_count must be a positive integer")
        if manifest.get("max_policy_actions") != 30:
            raise OpenMLEFastDatasetError("manifest action limit must be 30")
        tasks = manifest.get("records")
        if not isinstance(tasks, list) or len(tasks) != task_count:
            raise OpenMLEFastDatasetError("task manifest records must match task_count")

        records: list[OpenMLEFastRecord] = []
        seen_task_ids: set[str] = set()
        seen_source_families: set[str] = set()
        for expected_index, value in enumerate(tasks):
            if not isinstance(value, dict):
                raise OpenMLEFastDatasetError(
                    f"task row {expected_index} must be a JSON object"
                )
            data_idx = value.get("data_idx")
            if type(data_idx) is not int or data_idx != expected_index:
                raise OpenMLEFastDatasetError(
                    "task data_idx values must be non-boolean contiguous integers"
                )
            task_id = _text(value, "task_id")
            source_family = _text(value, "source_family")
            if _text(value, "role") != role:
                raise OpenMLEFastDatasetError("task row role differs from manifest")
            if task_id in seen_task_ids:
                raise OpenMLEFastDatasetError(f"duplicate task identity: {task_id}")
            if role == GATE_ONLY_ROLE and source_family in seen_source_families:
                raise OpenMLEFastDatasetError(
                    f"duplicate source family in gate-only manifest: {source_family}"
                )
            seen_task_ids.add(task_id)
            seen_source_families.add(source_family)
            records.append(self._load_record(expected_index, value))

        task_ids = [record.task_id for record in records]
        source_families = sorted({record.source_family for record in records})
        task_id_list_sha256 = _sha256_text(
            manifest.get("task_id_list_sha256"), "task-id-list SHA256"
        )
        if (
            _sha256(("\n".join(task_ids) + "\n").encode("utf-8"))
            != task_id_list_sha256
        ):
            raise OpenMLEFastDatasetError("ordered task identity digest drift")
        if _sha256(("\n".join(source_families) + "\n").encode("utf-8")) != _sha256_text(
            manifest.get("source_family_list_sha256"),
            "source-family-list SHA256",
        ):
            raise OpenMLEFastDatasetError("source-family digest drift")
        if manifest.get("source_family_count") != len(source_families):
            raise OpenMLEFastDatasetError("source-family count drift")
        compact = [
            {
                "task_id": value["task_id"],
                "source_family": value["source_family"],
                "archive_sha256": value["archive_sha256"],
                "public_tree_sha256": value["public_tree_sha256"],
                "package_identity_sha256": value["package_identity_sha256"],
                "reward_eligible": value["reward_eligible"],
                "engineering_gate_member": value["engineering_gate_member"],
            }
            for value in tasks
        ]
        compact_panel_sha256 = _sha256_text(
            manifest.get("compact_panel_sha256"), "compact-panel SHA256"
        )
        if canonical_sha256(compact) != compact_panel_sha256:
            raise OpenMLEFastDatasetError("compact panel digest drift")

        self._records = tuple(records)
        self.provenance = DatasetProvenance(
            schema=schema,
            contract_version=contract_version,
            panel_id=panel_id,
            release_revision=release_revision,
            role=role,
            manifest_path=self.manifest_path,
            manifest_sha256=actual_manifest_sha256,
            task_id_list_sha256=task_id_list_sha256,
            compact_panel_sha256=compact_panel_sha256,
            task_count=len(records),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, data_idx: int) -> OpenMLEFastRecord:
        if isinstance(data_idx, bool):
            raise TypeError("OpenMLE-fast data_idx must not be bool")
        try:
            index = operator.index(data_idx)
        except TypeError as exc:
            raise TypeError("OpenMLE-fast data_idx must be an integer") from exc
        if index < 0 or index >= len(self._records):
            raise IndexError(
                f"OpenMLE-fast data_idx {index} is outside [0, {len(self._records)})"
            )
        return self._records[index]

    def _load_record(
        self, data_idx: int, value: Mapping[str, Any]
    ) -> OpenMLEFastRecord:
        task_id = _text(value, "task_id")
        package_relative = _safe_relative(_text(value, "public_task_relpath"))
        archive_relative = _safe_relative(_text(value, "archive_relpath"))
        public_root = _contained_directory(
            self.package_root,
            self.package_root / package_relative,
            "public task root",
        )
        archive = _contained_file(
            self.archive_root, self.archive_root / archive_relative, "task archive"
        )
        archive_sha256 = _sha256_text(value.get("archive_sha256"), "archive SHA256")
        if _sha256_file(archive) != archive_sha256:
            raise OpenMLEFastDatasetError(f"archive SHA256 mismatch for {task_id}")

        task_spec_sha256 = _sha256_text(
            value.get("task_spec_sha256"), "task-spec SHA256"
        )
        public_inventory = value.get("public_files")
        if not isinstance(public_inventory, list) or not public_inventory:
            raise OpenMLEFastDatasetError("public_files must be a non-empty list")
        normalized_public_names: list[str] = []
        collision_keys: set[str] = set()
        entries: list[dict[str, Any]] = []
        public_file_sha256: dict[str, str] = {}
        previous_name: bytes | None = None
        for item in public_inventory:
            if not isinstance(item, dict) or set(item) != {
                "path",
                "size",
                "sha256",
                "mode",
            }:
                raise OpenMLEFastDatasetError("public file record schema drift")
            name = _text(item, "path")
            normalized = _safe_relative(name)
            encoded_name = normalized.encode("utf-8")
            if previous_name is not None and encoded_name <= previous_name:
                raise OpenMLEFastDatasetError(
                    "public_files must be UTF-8 sorted and unique"
                )
            previous_name = encoded_name
            collision_key = unicodedata.normalize("NFC", normalized).casefold()
            if collision_key in collision_keys:
                raise OpenMLEFastDatasetError(
                    "public file inventory has a Unicode/case collision"
                )
            collision_keys.add(collision_key)
            lowered_parts = {
                part.casefold() for part in PurePosixPath(normalized).parts
            }
            if any("private" in part or "answer" in part for part in lowered_parts):
                raise OpenMLEFastDatasetError(
                    f"policy-visible filename is not public-safe: {name}"
                )
            if normalized != "TASK.md" and not normalized.startswith("data/"):
                raise OpenMLEFastDatasetError(
                    f"policy-visible file is outside TASK.md/data: {name}"
                )
            size = item.get("size")
            mode = item.get("mode")
            if type(size) is not int or size < 0 or mode != 0o444:
                raise OpenMLEFastDatasetError("public file size/mode drift")
            digest = _sha256_text(item.get("sha256"), "public-file SHA256")
            source = _contained_file(
                public_root,
                public_root / PurePosixPath(normalized),
                f"public file {normalized}",
            )
            info = os.stat(source, follow_symlinks=False)
            if info.st_nlink != 1 or info.st_size != size:
                raise OpenMLEFastDatasetError(
                    f"public file identity drift for {normalized}"
                )
            if stat.S_IMODE(info.st_mode) != mode or _sha256_file(source) != digest:
                raise OpenMLEFastDatasetError(
                    f"public file hash/mode drift for {normalized}"
                )
            normalized_public_names.append(normalized)
            public_file_sha256[normalized] = digest
            entries.append(
                {"path": normalized, "size": size, "sha256": digest, "mode": mode}
            )

        if "TASK.md" not in public_file_sha256:
            raise OpenMLEFastDatasetError("public task is missing TASK.md")
        required_data = {
            "data/train.csv",
            "data/test.csv",
            "data/sample_submission.csv",
        }
        if not required_data.issubset(public_file_sha256):
            raise OpenMLEFastDatasetError("standard public data files are missing")
        discovered = _discover_public_files(public_root)
        if discovered != normalized_public_names:
            raise OpenMLEFastDatasetError(f"public file inventory drift for {task_id}")
        task_path = _contained_file(public_root, public_root / "TASK.md", "TASK.md")
        task_bytes = task_path.read_bytes()
        if _sha256(task_bytes) != task_spec_sha256:
            raise OpenMLEFastDatasetError(f"task-spec SHA256 mismatch for {task_id}")
        try:
            task_markdown = task_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OpenMLEFastDatasetError("TASK.md must be UTF-8") from exc
        if not task_markdown.strip():
            raise OpenMLEFastDatasetError("TASK.md must be non-empty")
        public_tree_sha256 = _sha256_text(
            value.get("public_tree_sha256"), "public-tree SHA256"
        )
        if tree_sha256(entries) != public_tree_sha256:
            raise OpenMLEFastDatasetError(f"public-tree SHA256 mismatch for {task_id}")

        grader_binding = _text(value, "grader_binding")
        grader_binding_sha256 = _sha256_text(
            value.get("grader_binding_sha256"), "grader-binding SHA256"
        )
        if grader_binding != f"openmlefast-grader-{grader_binding_sha256[:24]}":
            raise OpenMLEFastDatasetError("grader binding identifier drift")
        package_identity_sha256 = _sha256_text(
            value.get("package_identity_sha256"), "package-identity SHA256"
        )

        baseline = _finite_number(value, "baseline_score")
        ideal = _finite_number(value, "ideal_score")
        direction = value.get("higher_is_better")
        if type(direction) is not bool:
            raise OpenMLEFastDatasetError("higher_is_better must be Boolean")
        directed_gap = (1.0 if direction else -1.0) * (ideal - baseline)
        scale = max(1.0, abs(baseline), abs(ideal))
        if directed_gap <= 1e-6 * scale:
            raise OpenMLEFastDatasetError(
                f"task has no admissible baseline-to-ideal gap: {task_id}"
            )
        public_direction = _text(value, "metric_direction")
        expected_direction = "higher" if direction else "lower"
        if public_direction != expected_direction:
            raise OpenMLEFastDatasetError(f"metric direction mismatch for {task_id}")
        if value.get("reward_eligible") is not True:
            raise OpenMLEFastDatasetError("manifest row is not reward eligible")
        license_name = _text(value, "license_name_or_permission")
        source_urls = value.get("source_urls")
        if (
            not isinstance(source_urls, list)
            or not source_urls
            or any(
                not isinstance(item, str) or not item.strip() for item in source_urls
            )
        ):
            raise OpenMLEFastDatasetError("source_urls must be non-empty text")
        provenance_root = _contained_directory(
            self.package_root,
            self.package_root / _safe_relative(_text(value, "provenance_relpath")),
            "task provenance root",
        )
        _validate_attested_files(provenance_root, value.get("provenance_files"))
        public_task = {
            "task_markdown": task_markdown,
            "metric_name": _text(value, "metric_name"),
            "metric_direction": public_direction,
            "public_files": list(normalized_public_names),
        }

        return OpenMLEFastRecord(
            data_idx=data_idx,
            task_id=task_id,
            source_family=_text(value, "source_family"),
            package_root=public_root,
            public_source_root=public_root,
            archive_path=archive,
            archive_sha256=archive_sha256,
            public_tree_sha256=public_tree_sha256,
            package_identity_sha256=package_identity_sha256,
            grader_binding=grader_binding,
            grader_binding_sha256=grader_binding_sha256,
            baseline_score=baseline,
            ideal_score=ideal,
            higher_is_better=direction,
            license={"name_or_permission": license_name},
            provenance={
                "source_urls": list(source_urls),
                "source_family_sha256": _sha256_text(
                    value.get("source_family_sha256"), "source-family SHA256"
                ),
                "evidence_sha256": _sha256_text(
                    value.get("evidence_sha256"), "evidence SHA256"
                ),
            },
            public_task=public_task,
            task_spec_sha256=task_spec_sha256,
            public_file_sha256=public_file_sha256,
        )


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return _sha256(payload)


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OpenMLEFastDatasetError("task manifest contains a duplicate key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)


def tree_sha256(entries: list[dict[str, Any]]) -> str:
    payload = (
        json.dumps(
            sorted(entries, key=lambda item: item["path"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return _sha256(payload)


def _discover_public_files(root: Path) -> list[str]:
    discovered: list[str] = []
    for path in root.rglob("*"):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise OpenMLEFastDatasetError("public data may not contain symlinks")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise OpenMLEFastDatasetError("public data may contain regular files only")
        discovered.append(path.relative_to(root).as_posix())
    return sorted(discovered, key=lambda value: value.encode("utf-8"))


def _validate_attested_files(root: Path, inventory: Any) -> None:
    if not isinstance(inventory, list) or not inventory:
        raise OpenMLEFastDatasetError("provenance file inventory is empty")
    expected_names: list[str] = []
    previous_name: bytes | None = None
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size",
            "sha256",
            "mode",
        }:
            raise OpenMLEFastDatasetError("provenance file record schema drift")
        relative = _safe_relative(_text(item, "path"))
        encoded = relative.encode("utf-8")
        if previous_name is not None and encoded <= previous_name:
            raise OpenMLEFastDatasetError(
                "provenance files must be UTF-8 sorted and unique"
            )
        previous_name = encoded
        size = item.get("size")
        mode = item.get("mode")
        if type(size) is not int or size < 0 or mode != 0o444:
            raise OpenMLEFastDatasetError("provenance file size/mode drift")
        digest = _sha256_text(item.get("sha256"), "provenance-file SHA256")
        source = _contained_file(root, root / relative, "task provenance file")
        info = os.stat(source, follow_symlinks=False)
        if (
            info.st_nlink != 1
            or info.st_size != size
            or stat.S_IMODE(info.st_mode) != mode
            or _sha256_file(source) != digest
        ):
            raise OpenMLEFastDatasetError("provenance file identity drift")
        expected_names.append(relative)
    if _discover_public_files(root) != expected_names:
        raise OpenMLEFastDatasetError("provenance file inventory drift")


def _safe_relative(value: str) -> str:
    if "\x00" in value or "\\" in value:
        raise OpenMLEFastDatasetError("manifest path is not a safe POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise OpenMLEFastDatasetError("manifest path is not a safe POSIX relative path")
    return path.as_posix()


def _contained_directory(root: Path, path: Path, label: str) -> Path:
    candidate = _resolve_beneath_without_symlink(root, path, label)
    if not candidate.is_dir():
        raise OpenMLEFastDatasetError(f"{label} must be a real directory")
    _require_contained(root, candidate, label)
    return candidate


def _contained_file(root: Path, path: Path, label: str) -> Path:
    candidate = _resolve_beneath_without_symlink(root, path, label)
    if not candidate.is_file():
        raise OpenMLEFastDatasetError(f"{label} must be a regular file")
    _require_contained(root, candidate, label)
    return candidate


def _resolve_beneath_without_symlink(root: Path, path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise OpenMLEFastDatasetError(f"{label} escapes its configured root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise OpenMLEFastDatasetError(f"cannot inspect {label}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise OpenMLEFastDatasetError(f"{label} may not traverse a symlink")
    resolved = absolute.resolve()
    _require_contained(root, resolved, label)
    return resolved


def _require_contained(root: Path, candidate: Path, label: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OpenMLEFastDatasetError(f"{label} escapes its configured root") from exc


def _real_file(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink():
        raise OpenMLEFastDatasetError(f"{label} may not be a symlink")
    resolved = absolute.resolve()
    if not resolved.is_file():
        raise OpenMLEFastDatasetError(f"{label} must be a regular file")
    return resolved


def _real_directory(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink():
        raise OpenMLEFastDatasetError(f"{label} may not be a symlink")
    resolved = absolute.resolve()
    if not resolved.is_dir():
        raise OpenMLEFastDatasetError(f"{label} must be a real directory")
    return resolved


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise OpenMLEFastDatasetError(f"{label} must be a regular file")
        return path.read_bytes()
    except OSError as exc:
        raise OpenMLEFastDatasetError(f"cannot read {label}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise OpenMLEFastDatasetError(f"manifest field {key!r} must be an object")
    return value


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OpenMLEFastDatasetError(f"manifest field {key!r} must be non-empty text")
    return value.strip()


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise OpenMLEFastDatasetError(f"{label} must be text")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise OpenMLEFastDatasetError(f"{label} must be a SHA256 digest")
    return normalized


def _git_revision(value: Any) -> str:
    if not isinstance(value, str):
        raise OpenMLEFastDatasetError("release revision must be text")
    normalized = value.lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise OpenMLEFastDatasetError("release revision must be a full Git commit")
    return normalized


def _finite_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenMLEFastDatasetError(f"manifest field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OpenMLEFastDatasetError(f"manifest field {key!r} must be finite")
    return result
