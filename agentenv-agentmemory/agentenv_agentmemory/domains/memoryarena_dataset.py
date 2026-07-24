from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MEMORYARENA_HF_REPO_ID = "ZexueHe/memoryarena"
MEMORYARENA_HF_REVISION = "da1a37c8b19280e18627ca01cf368195a5e1d92e"
MEMORYARENA_HF_SPLIT = "test"


@dataclass(frozen=True)
class FrozenMemoryArenaDatasetSpec:
    config: str
    repo_path: str
    sha256: str
    record_count: int
    phase_count: int
    phase_field: str


FROZEN_MEMORYARENA_DATASETS = {
    "formal_reasoning_math": FrozenMemoryArenaDatasetSpec(
        config="formal_reasoning_math",
        repo_path="formal_reasoning_math/data.jsonl",
        sha256="ff5b0ad575847c7476a02d1e35661592a833bd0cff384cb54bc6f35b46de7803",
        record_count=40,
        phase_count=354,
        phase_field="questions",
    ),
    "formal_reasoning_phys": FrozenMemoryArenaDatasetSpec(
        config="formal_reasoning_phys",
        repo_path="formal_reasoning_phys/data.jsonl",
        sha256="580862006af2ff2bfc8c5d2d2b9a60bf33a46cbb64f27d60a2bfe039aec61cf6",
        record_count=20,
        phase_count=86,
        phase_field="questions",
    ),
    "group_travel_planner": FrozenMemoryArenaDatasetSpec(
        config="group_travel_planner",
        repo_path="group_travel_planner/data.jsonl",
        sha256="2f955d444f6f3ad3c5da2064359ab19f8fc1f90621ff9d00723a450a009c3732",
        record_count=270,
        phase_count=1869,
        phase_field="questions",
    ),
    "progressive_search": FrozenMemoryArenaDatasetSpec(
        config="progressive_search",
        repo_path="progressive_search/data.jsonl",
        sha256="b445ee36fa3ccb9ad08eae9e7adda86bbc64f14f1e2a0682a8b2085cdb8e4c0e",
        record_count=221,
        phase_count=1641,
        phase_field="questions",
    ),
}


@dataclass(frozen=True)
class MemoryArenaDatasetProvenance:
    mode: str
    dataset_config: str
    split: str
    repo_id: str | None
    revision: str | None
    repo_path: str | None
    sha256: str
    record_count: int
    phase_count: int
    phase_field: str
    attestation_sha256: str

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def attest_frozen_memoryarena_dataset(
    path: str | Path,
    *,
    config: str,
) -> MemoryArenaDatasetProvenance:
    """Bind a local JSONL byte-for-byte to a frozen public HF config."""

    try:
        spec = FROZEN_MEMORYARENA_DATASETS[config]
    except KeyError as exc:
        raise RuntimeError(
            f"No frozen MemoryArena dataset spec for config: {config}"
        ) from exc
    observed = _inspect_jsonl(Path(path), phase_field=spec.phase_field)
    mismatches = []
    for field in ("sha256", "record_count", "phase_count"):
        expected_value = getattr(spec, field)
        observed_value = observed[field]
        if observed_value != expected_value:
            mismatches.append(
                f"{field}: expected {expected_value}, observed {observed_value}"
            )
    if mismatches:
        raise RuntimeError(
            f"MemoryArena dataset does not match frozen HF config {config}: "
            + "; ".join(mismatches)
        )
    payload = {
        "mode": "frozen_public_hf_dataset",
        "dataset_config": spec.config,
        "split": MEMORYARENA_HF_SPLIT,
        "repo_id": MEMORYARENA_HF_REPO_ID,
        "revision": MEMORYARENA_HF_REVISION,
        "repo_path": spec.repo_path,
        "sha256": observed["sha256"],
        "record_count": observed["record_count"],
        "phase_count": observed["phase_count"],
        "phase_field": spec.phase_field,
    }
    return MemoryArenaDatasetProvenance(
        **payload,
        attestation_sha256=_sha256_json(payload),
    )


def attest_injected_test_dataset(
    path: str | Path,
    *,
    config: str,
    phase_field: str = "questions",
) -> MemoryArenaDatasetProvenance:
    """Create an explicit, non-production provenance record for a unit fixture."""

    observed = _inspect_jsonl(Path(path), phase_field=phase_field)
    payload = {
        "mode": "injected_test_fixture",
        "dataset_config": config,
        "split": "injected_test",
        "repo_id": None,
        "revision": None,
        "repo_path": None,
        "sha256": observed["sha256"],
        "record_count": observed["record_count"],
        "phase_count": observed["phase_count"],
        "phase_field": phase_field,
    }
    return MemoryArenaDatasetProvenance(
        **payload,
        attestation_sha256=_sha256_json(payload),
    )


def verify_memoryarena_dataset_provenance(
    path: str | Path,
    *,
    expected_config: str,
    provenance: MemoryArenaDatasetProvenance,
) -> None:
    if not isinstance(provenance, MemoryArenaDatasetProvenance):
        raise TypeError("dataset_provenance must be a MemoryArenaDatasetProvenance")
    if provenance.dataset_config != expected_config:
        raise RuntimeError(
            "MemoryArena dataset config mismatch: "
            f"surface requires {expected_config}, provenance is "
            f"{provenance.dataset_config}"
        )
    observed = _inspect_jsonl(Path(path), phase_field=provenance.phase_field)
    for field in ("sha256", "record_count", "phase_count"):
        if observed[field] != getattr(provenance, field):
            raise RuntimeError(
                "MemoryArena dataset changed after provenance attestation: "
                f"{field} expected {getattr(provenance, field)}, "
                f"observed {observed[field]}"
            )
    payload = provenance.metadata()
    attestation_sha256 = payload.pop("attestation_sha256")
    if _sha256_json(payload) != attestation_sha256:
        raise RuntimeError(
            "MemoryArena dataset provenance record is internally inconsistent"
        )
    if provenance.mode == "frozen_public_hf_dataset":
        expected = attest_frozen_memoryarena_dataset(path, config=expected_config)
        if provenance != expected:
            raise RuntimeError(
                "MemoryArena frozen dataset provenance does not match its spec"
            )
    elif provenance.mode != "injected_test_fixture":
        raise RuntimeError(
            f"Unsupported MemoryArena dataset provenance mode: {provenance.mode}"
        )


def _inspect_jsonl(path: Path, *, phase_field: str) -> dict[str, int | str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"MemoryArena dataset file does not exist: {resolved}")
    digest = hashlib.sha256()
    records = 0
    phases = 0
    with resolved.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise RuntimeError(
                    f"Blank MemoryArena dataset row at line {line_number}: {resolved}"
                )
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Invalid MemoryArena JSONL row at line {line_number}: {resolved}"
                ) from exc
            phase_values = row.get(phase_field) if isinstance(row, dict) else None
            if not isinstance(phase_values, list) or not phase_values:
                raise RuntimeError(
                    f"MemoryArena row {line_number} requires non-empty {phase_field}"
                )
            records += 1
            phases += len(phase_values)
    if records == 0:
        raise RuntimeError(f"MemoryArena dataset file is empty: {resolved}")
    return {
        "sha256": digest.hexdigest(),
        "record_count": records,
        "phase_count": phases,
    }


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
