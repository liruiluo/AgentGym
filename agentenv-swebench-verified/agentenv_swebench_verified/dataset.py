from __future__ import annotations

import copy
import hashlib
import json
import operator
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .protocol import (
    POLICY_FIELDS,
    PRODUCTION_DATASET_PINS,
    FrozenDatasetPins,
    policy_projection,
    require_nonempty_text,
)


DATASET_MANIFEST_SCHEMA = "swebench_verified_frozen_jsonl_manifest_v1"
_MANIFEST_FIELDS = {"schema_version", "dataset", "canonical_jsonl"}
_DATASET_FIELDS = {"repository", "revision", "split"}
_JSONL_FIELDS = {"path", "sha256", "rows", "id_ledger_sha256"}
_TESTSPEC_REQUIRED_FIELDS = {
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "version",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}


class VerifiedDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetProvenance:
    schema_version: str
    repository: str
    revision: str
    split: str
    row_count: int
    canonical_jsonl_path: Path
    canonical_jsonl_sha256: str
    id_ledger_sha256: str
    manifest_path: Path
    manifest_sha256: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "revision": self.revision,
            "split": self.split,
            "row_count": self.row_count,
            "canonical_jsonl_sha256": self.canonical_jsonl_sha256,
            "id_ledger_sha256": self.id_ledger_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class VerifiedRecord:
    data_idx: int
    _policy_instance: Mapping[str, str]
    _private_row: Mapping[str, Any]

    @property
    def instance_id(self) -> str:
        return self._policy_instance["instance_id"]

    @property
    def problem_statement(self) -> str:
        return self._policy_instance["problem_statement"]

    @property
    def policy_instance(self) -> dict[str, str]:
        return dict(self._policy_instance)

    def private_instance(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._private_row))


class VerifiedDataset:
    """Load one externally materialized, byte-pinned Verified JSONL.

    Complete rows remain server-private. The only policy-facing representation
    is ``VerifiedRecord.policy_instance``, whose four keys are fixed by the
    audit protocol.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        pins: FrozenDatasetPins = PRODUCTION_DATASET_PINS,
    ) -> None:
        self.manifest_path = require_regular_file(
            Path(manifest_path).expanduser(), "dataset manifest"
        )
        raw_manifest = self.manifest_path.read_bytes()
        manifest = parse_json_object(raw_manifest, "dataset manifest")
        require_exact_fields(manifest, _MANIFEST_FIELDS, "dataset manifest")
        if manifest["schema_version"] != DATASET_MANIFEST_SCHEMA:
            raise VerifiedDatasetError("dataset manifest schema is unsupported")

        dataset_spec = require_mapping(manifest, "dataset")
        require_exact_fields(dataset_spec, _DATASET_FIELDS, "dataset identity")
        expected_dataset = {
            "repository": pins.repository,
            "revision": pins.revision,
            "split": pins.split,
        }
        if dict(dataset_spec) != expected_dataset:
            mismatched = sorted(
                key
                for key, expected in expected_dataset.items()
                if dataset_spec.get(key) != expected
            )
            raise VerifiedDatasetError(
                "dataset manifest identity does not match frozen "
                + ", ".join(mismatched)
            )

        jsonl_spec = require_mapping(manifest, "canonical_jsonl")
        require_exact_fields(jsonl_spec, _JSONL_FIELDS, "canonical JSONL")
        expected_jsonl = {
            "sha256": pins.canonical_jsonl_sha256,
            "rows": pins.row_count,
            "id_ledger_sha256": pins.id_ledger_sha256,
        }
        for key, expected in expected_jsonl.items():
            if jsonl_spec.get(key) != expected:
                raise VerifiedDatasetError(
                    f"canonical JSONL {key} does not match the frozen pins"
                )

        jsonl_path = resolve_manifest_path(
            self.manifest_path, jsonl_spec.get("path"), "canonical JSONL"
        )
        payload = jsonl_path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != pins.canonical_jsonl_sha256:
            raise VerifiedDatasetError(
                "canonical JSONL SHA-256 mismatch: expected "
                f"{pins.canonical_jsonl_sha256}, got {actual_sha256}"
            )

        rows = parse_canonical_rows(payload)
        if len(rows) != pins.row_count:
            raise VerifiedDatasetError(
                f"canonical JSONL row count mismatch: expected {pins.row_count}, "
                f"got {len(rows)}"
            )
        instance_ids = [require_nonempty_text(row, "instance_id") for row in rows]
        if len(set(instance_ids)) != len(instance_ids):
            raise VerifiedDatasetError("instance IDs must be unique")
        if instance_ids != sorted(instance_ids):
            raise VerifiedDatasetError("instance IDs must be sorted")
        ledger = "".join(f"{instance_id}\n" for instance_id in instance_ids).encode(
            "utf-8"
        )
        actual_ledger_sha256 = hashlib.sha256(ledger).hexdigest()
        if actual_ledger_sha256 != pins.id_ledger_sha256:
            raise VerifiedDatasetError(
                "instance-ID ledger SHA-256 does not match the frozen pins"
            )

        records: list[VerifiedRecord] = []
        for data_idx, row in enumerate(rows):
            validate_private_instance(row, data_idx)
            projected = policy_projection(row)
            if tuple(projected) != POLICY_FIELDS:
                raise VerifiedDatasetError("policy projection order drifted")
            records.append(
                VerifiedRecord(
                    data_idx=data_idx,
                    _policy_instance=MappingProxyType(projected),
                    _private_row=MappingProxyType(copy.deepcopy(row)),
                )
            )

        self._records = tuple(records)
        self.instance_ids = tuple(instance_ids)
        self.provenance = DatasetProvenance(
            schema_version=DATASET_MANIFEST_SCHEMA,
            repository=pins.repository,
            revision=pins.revision,
            split=pins.split,
            row_count=len(records),
            canonical_jsonl_path=jsonl_path,
            canonical_jsonl_sha256=actual_sha256,
            id_ledger_sha256=actual_ledger_sha256,
            manifest_path=self.manifest_path,
            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, data_idx: int) -> VerifiedRecord:
        try:
            normalized = operator.index(data_idx)
        except TypeError as exc:
            raise TypeError("data_idx must be an integer") from exc
        if isinstance(data_idx, bool):
            raise TypeError("data_idx must be an integer")
        if not 0 <= normalized < len(self._records):
            raise IndexError("data_idx is outside the frozen dataset")
        return self._records[normalized]


def parse_canonical_rows(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        raise VerifiedDatasetError("canonical JSONL is empty")
    if not payload.endswith(b"\n"):
        raise VerifiedDatasetError("canonical JSONL must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        if raw_line == b"\n":
            raise VerifiedDatasetError(
                f"canonical JSONL contains a blank row at line {line_number}"
            )
        row = parse_json_object(raw_line, f"canonical JSONL line {line_number}")
        canonical = (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if raw_line != canonical:
            raise VerifiedDatasetError(
                f"canonical JSONL line {line_number} is not canonically serialized"
            )
        rows.append(row)
    return rows


def validate_private_instance(row: Mapping[str, Any], data_idx: int) -> None:
    missing = sorted(_TESTSPEC_REQUIRED_FIELDS - set(row))
    if missing:
        raise VerifiedDatasetError(
            f"row {data_idx} lacks TestSpec fields: {', '.join(missing)}"
        )
    try:
        policy_projection(row)
        require_nonempty_text(row, "version")
    except (TypeError, ValueError) as exc:
        raise VerifiedDatasetError(f"row {data_idx} is invalid: {exc}") from exc
    if not isinstance(row.get("test_patch"), str):
        raise VerifiedDatasetError(f"row {data_idx} test_patch must be text")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if not isinstance(row.get(key), (str, list)):
            raise VerifiedDatasetError(f"row {data_idx} {key} has invalid type")


def parse_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedDatasetError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise VerifiedDatasetError(f"{label} must be a JSON object")
    return value


def require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise VerifiedDatasetError(f"{key} must be a JSON object")
    return item


def require_exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise VerifiedDatasetError(f"{label} has unexpected or missing fields")


def resolve_manifest_path(manifest_path: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise VerifiedDatasetError(f"{label} path must be non-empty text")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return require_regular_file(candidate, label)


def require_regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise VerifiedDatasetError(f"{label} is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise VerifiedDatasetError(f"{label} must be a real regular file")
    return path.resolve(strict=True)
