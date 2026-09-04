from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DATASET_MANIFEST_SCHEMA = "swesmith_jsonl_manifest_v1"
UPSTREAM_REPOSITORY = "SWE-bench/SWE-smith"
DATASET_ROLES = frozenset({"plumbing", "train", "heldout", "formal_heldout"})


class SwesmithDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShardProvenance:
    path: Path
    sha256: str
    physical_rows: int
    usable_rows: int


@dataclass(frozen=True)
class DatasetProvenance:
    schema_version: str
    dataset_id: str
    upstream_repository: str
    upstream_revision: str
    role: str
    selection_mode: str
    manifest_path: Path
    manifest_sha256: str
    physical_rows: int
    usable_rows: int
    selected_rows: int
    blank_rows: int
    shards: tuple[ShardProvenance, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "upstream_repository": self.upstream_repository,
            "upstream_revision": self.upstream_revision,
            "role": self.role,
            "selection_mode": self.selection_mode,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "physical_rows": self.physical_rows,
            "usable_rows": self.usable_rows,
            "selected_rows": self.selected_rows,
            "blank_rows": self.blank_rows,
            "shards": [
                {
                    "path": str(shard.path),
                    "sha256": shard.sha256,
                    "physical_rows": shard.physical_rows,
                    "usable_rows": shard.usable_rows,
                }
                for shard in self.shards
            ],
        }


@dataclass(frozen=True)
class SwesmithRecord:
    data_idx: int
    physical_index: int
    shard_index: int
    shard_line: int
    shard_sha256: str
    instance: Mapping[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.instance["instance_id"])

    @property
    def problem_statement(self) -> str:
        return str(self.instance["problem_statement"])

    @property
    def base_repository(self) -> str:
        repo = str(self.instance["repo"]).strip()
        prefix = "swesmith/"
        if not repo.startswith(prefix):
            raise SwesmithDatasetError(
                f"SWE-smith repo has no {prefix!r} prefix: {repo!r}"
            )
        repository, separator, revision = repo[len(prefix) :].rpartition(".")
        if (
            not separator
            or not repository
            or len(revision) != 8
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise SwesmithDatasetError(
                f"SWE-smith repo has no terminal 8-hex revision: {repo!r}"
            )
        expected_prefix = f"{repository}.{revision}."
        if not self.instance_id.startswith(expected_prefix):
            raise SwesmithDatasetError(
                "SWE-smith repo and instance_id repository/revision disagree"
            )
        return repository


@dataclass(frozen=True)
class _IndexedLine:
    shard_index: int
    byte_offset: int
    byte_length: int
    shard_line: int
    physical_index: int
    instance_id: str


class SwesmithDataset:
    """A frozen, offset-indexed JSONL dataset.

    Startup verifies the complete byte identity and row contract of every
    shard, but retains only byte offsets and opaque instance identities. The
    full private row is parsed again only when its explicit integer data index
    is requested.
    """

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        raw_manifest = _read_regular_bytes(self.manifest_path, "dataset manifest")
        self.manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        try:
            manifest = json.loads(raw_manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SwesmithDatasetError("dataset manifest is not valid UTF-8 JSON") from exc
        if not isinstance(manifest, dict):
            raise SwesmithDatasetError("dataset manifest must be a JSON object")

        self.dataset_id = _required_text(manifest, "dataset_id")
        schema_version = _required_text(manifest, "schema_version")
        if schema_version != DATASET_MANIFEST_SCHEMA:
            raise SwesmithDatasetError(
                f"unsupported dataset manifest schema: {schema_version!r}"
            )
        upstream = _required_mapping(manifest, "upstream")
        upstream_repository = _required_text(upstream, "repository")
        if upstream_repository != UPSTREAM_REPOSITORY:
            raise SwesmithDatasetError(
                f"unexpected upstream repository: {upstream_repository!r}"
            )
        upstream_revision = _required_git_commit(upstream, "revision")
        role = _required_text(manifest, "role")
        if role not in DATASET_ROLES:
            raise SwesmithDatasetError(
                f"dataset role must be one of {sorted(DATASET_ROLES)}"
            )

        selection = _required_mapping(manifest, "selection")
        selection_mode = _required_text(selection, "mode")
        selected_ids = self._load_selection(selection_mode, selection)
        shard_specs = manifest.get("shards")
        if not isinstance(shard_specs, list) or not shard_specs:
            raise SwesmithDatasetError("dataset manifest shards must be a non-empty list")

        self._shard_paths: list[Path] = []
        self._shard_sha256: list[str] = []
        self._index: list[_IndexedLine] = []
        shard_provenance: list[ShardProvenance] = []
        seen_instance_ids: set[str] = set()
        found_selected_ids: set[str] = set()
        physical_index = 0
        blank_rows = 0

        for shard_index, raw_spec in enumerate(shard_specs):
            if not isinstance(raw_spec, dict):
                raise SwesmithDatasetError(f"shard {shard_index} must be an object")
            shard_path = self._resolve_manifest_path(
                _required_text(raw_spec, "path"),
                label=f"shard {shard_index}",
            )
            expected_sha256 = _required_sha256(raw_spec, "sha256")
            expected_physical = _required_nonnegative_int(raw_spec, "physical_rows")
            expected_usable = _required_nonnegative_int(raw_spec, "usable_rows")

            digest = hashlib.sha256()
            shard_physical = 0
            shard_usable = 0
            try:
                handle = shard_path.open("rb")
            except OSError as exc:
                raise SwesmithDatasetError(
                    f"cannot open dataset shard {shard_path}"
                ) from exc
            with handle:
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    digest.update(raw_line)
                    shard_physical += 1
                    try:
                        instance = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SwesmithDatasetError(
                            f"invalid JSON at shard {shard_index} line {shard_physical}"
                        ) from exc
                    _validate_instance(instance, shard_index, shard_physical)
                    instance_id = str(instance["instance_id"])
                    if instance_id in seen_instance_ids:
                        raise SwesmithDatasetError(
                            f"duplicate instance_id across shards: {instance_id}"
                        )
                    seen_instance_ids.add(instance_id)
                    usable = bool(str(instance["problem_statement"]).strip())
                    if usable:
                        shard_usable += 1
                        if selected_ids is None or instance_id in selected_ids:
                            self._index.append(
                                _IndexedLine(
                                    shard_index=shard_index,
                                    byte_offset=offset,
                                    byte_length=len(raw_line),
                                    shard_line=shard_physical,
                                    physical_index=physical_index,
                                    instance_id=instance_id,
                                )
                            )
                            found_selected_ids.add(instance_id)
                    else:
                        blank_rows += 1
                    physical_index += 1

            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise SwesmithDatasetError(
                    f"shard {shard_index} SHA256 mismatch: expected "
                    f"{expected_sha256}, got {actual_sha256}"
                )
            if shard_physical != expected_physical:
                raise SwesmithDatasetError(
                    f"shard {shard_index} physical row mismatch: expected "
                    f"{expected_physical}, got {shard_physical}"
                )
            if shard_usable != expected_usable:
                raise SwesmithDatasetError(
                    f"shard {shard_index} usable row mismatch: expected "
                    f"{expected_usable}, got {shard_usable}"
                )
            self._shard_paths.append(shard_path)
            self._shard_sha256.append(actual_sha256)
            shard_provenance.append(
                ShardProvenance(
                    path=shard_path,
                    sha256=actual_sha256,
                    physical_rows=shard_physical,
                    usable_rows=shard_usable,
                )
            )

        if selected_ids is not None:
            missing = sorted(selected_ids - found_selected_ids)
            if missing:
                sample = ", ".join(missing[:3])
                raise SwesmithDatasetError(
                    f"selection contains {len(missing)} missing or blank instance IDs: {sample}"
                )
            expected_count = _required_nonnegative_int(selection, "count")
            if len(self._index) != expected_count:
                raise SwesmithDatasetError(
                    f"selected row mismatch: expected {expected_count}, got {len(self._index)}"
                )

        usable_rows = sum(shard.usable_rows for shard in shard_provenance)
        self.provenance = DatasetProvenance(
            schema_version=schema_version,
            dataset_id=self.dataset_id,
            upstream_repository=upstream_repository,
            upstream_revision=upstream_revision,
            role=role,
            selection_mode=selection_mode,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            physical_rows=physical_index,
            usable_rows=usable_rows,
            selected_rows=len(self._index),
            blank_rows=blank_rows,
            shards=tuple(shard_provenance),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, data_idx: int) -> SwesmithRecord:
        resolved = _resolve_data_idx(data_idx)
        if resolved >= len(self._index):
            raise IndexError(
                f"SWE-smith data_idx {resolved} is outside [0, {len(self._index)})"
            )
        indexed = self._index[resolved]
        shard_path = self._shard_paths[indexed.shard_index]
        try:
            with shard_path.open("rb") as handle:
                handle.seek(indexed.byte_offset)
                raw_line = handle.read(indexed.byte_length)
        except OSError as exc:
            raise SwesmithDatasetError(
                f"cannot read indexed dataset row from {shard_path}"
            ) from exc
        try:
            instance = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SwesmithDatasetError(
                "indexed dataset row changed after manifest attestation"
            ) from exc
        _validate_instance(instance, indexed.shard_index, indexed.shard_line)
        if str(instance["instance_id"]) != indexed.instance_id:
            raise SwesmithDatasetError(
                "indexed dataset identity changed after manifest attestation"
            )
        if not str(instance["problem_statement"]).strip():
            raise SwesmithDatasetError(
                "indexed dataset problem statement became blank after attestation"
            )
        return SwesmithRecord(
            data_idx=resolved,
            physical_index=indexed.physical_index,
            shard_index=indexed.shard_index,
            shard_line=indexed.shard_line,
            shard_sha256=self._shard_sha256[indexed.shard_index],
            instance=instance,
        )

    def _load_selection(
        self,
        mode: str,
        selection: Mapping[str, Any],
    ) -> set[str] | None:
        if mode == "all_usable":
            unexpected = set(selection) - {"mode"}
            if unexpected:
                raise SwesmithDatasetError(
                    f"all_usable selection has unexpected keys: {sorted(unexpected)}"
                )
            return None
        if mode != "instance_ids":
            raise SwesmithDatasetError(
                f"unsupported dataset selection mode: {mode!r}"
            )
        path = self._resolve_manifest_path(
            _required_text(selection, "path"),
            label="selection file",
        )
        expected_sha256 = _required_sha256(selection, "sha256")
        raw = _read_regular_bytes(path, "selection file")
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SwesmithDatasetError(
                "selection file SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        try:
            values = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SwesmithDatasetError(
                "selection file is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise SwesmithDatasetError(
                "selection file must be a JSON list of non-empty instance IDs"
            )
        if len(values) != len(set(values)):
            raise SwesmithDatasetError("selection file contains duplicate instance IDs")
        expected_count = _required_nonnegative_int(selection, "count")
        if len(values) != expected_count:
            raise SwesmithDatasetError(
                f"selection count mismatch: expected {expected_count}, got {len(values)}"
            )
        return set(values)

    def _resolve_manifest_path(self, raw: str, *, label: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        resolved = path.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise SwesmithDatasetError(f"{label} must be a real file: {resolved}")
        return resolved


def _validate_instance(instance: Any, shard_index: int, shard_line: int) -> None:
    label = f"shard {shard_index} line {shard_line}"
    if not isinstance(instance, dict):
        raise SwesmithDatasetError(f"{label} must contain a JSON object")
    for key in ("instance_id", "repo", "problem_statement"):
        value = instance.get(key)
        if not isinstance(value, str):
            raise SwesmithDatasetError(f"{label} field {key!r} must be a string")
    if not instance["instance_id"]:
        raise SwesmithDatasetError(f"{label} instance_id must not be empty")
    if not instance["repo"]:
        raise SwesmithDatasetError(f"{label} repo must not be empty")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = instance.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise SwesmithDatasetError(
                f"{label} field {key!r} must be a list of non-empty strings"
            )


def _resolve_data_idx(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("SWE-smith data_idx must not be bool")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError("SWE-smith data_idx must be an integer") from exc
    if resolved < 0:
        raise IndexError("SWE-smith data_idx must be non-negative")
    return int(resolved)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise SwesmithDatasetError(f"{label} must be a real file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SwesmithDatasetError(f"cannot read {label}: {path}") from exc


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise SwesmithDatasetError(f"manifest field {key!r} must be an object")
    return value


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SwesmithDatasetError(f"manifest field {key!r} must be non-empty text")
    return value.strip()


def _required_sha256(mapping: Mapping[str, Any], key: str) -> str:
    value = _required_text(mapping, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SwesmithDatasetError(
            f"manifest field {key!r} must be exactly 64 hexadecimal characters"
        )
    return value


def _required_git_commit(mapping: Mapping[str, Any], key: str) -> str:
    value = _required_text(mapping, key).lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise SwesmithDatasetError(
            f"manifest field {key!r} must be a full 40-character Git commit"
        )
    return value


def _required_nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SwesmithDatasetError(
            f"manifest field {key!r} must be a non-negative integer"
        )
    return value
