from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

GAIA_TEXT_PROTOCOL_ID = (
    "gaia_text_2023_validation_no_attachment@682dd723ee1e1697e00360edccf2366dc8418dd9"
)
GAIA_TEXT_DATASET_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"
GAIA_TEXT_MANIFEST_SHA256 = (
    "06f6da09978555c39f70f2794499012a1d07eb391e01a0f3d498957b09a1fda7"
)
GAIA_TEXT_TASK_IDS_SHA256 = (
    "57e76233b8b12d8d9ea18639d1d52616449cf521559cd9d103c76ff399a842ad"
)
GAIA_TEXT_SCORER_REVISION = "9f133d71362e77b3539f1514f31b9c101a545fec"
GAIA_TEXT_SCORER_SHA256 = (
    "0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvaluationArm(str, Enum):
    NATIVE = "native"
    AMG_MEMORY = "amg_memory"


@dataclass(frozen=True)
class ProtocolContract:
    protocol_id: str
    dataset_revision: str
    split: str
    task_count: int
    level_counts: tuple[tuple[int, int], ...]
    manifest_sha256: str
    task_ids_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty text")
        if not re.fullmatch(r"[0-9a-f]{40}", self.dataset_revision):
            raise ValueError("dataset_revision must be a full lowercase Git revision")
        if self.split != "validation":
            raise ValueError("GAIA-Text protocol split must be validation")
        if type(self.task_count) is not int or self.task_count <= 0:
            raise ValueError("task_count must be a positive integer")
        counts = tuple(self.level_counts)
        if not counts or any(
            type(level) is not int
            or level not in {1, 2, 3}
            or type(count) is not int
            or count <= 0
            for level, count in counts
        ):
            raise ValueError(
                "level_counts must contain positive counts for levels 1..3"
            )
        if tuple(sorted(counts)) != counts or len(
            {level for level, _ in counts}
        ) != len(counts):
            raise ValueError("level_counts must be unique and sorted by level")
        if tuple(level for level, _ in counts) != (1, 2, 3):
            raise ValueError("level_counts must cover GAIA levels 1, 2, and 3")
        if sum(count for _, count in counts) != self.task_count:
            raise ValueError("level_counts must sum to task_count")
        for name, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("task_ids_sha256", self.task_ids_sha256),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def public_metadata(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "task_count": self.task_count,
            "level_counts": {str(level): count for level, count in self.level_counts},
            "manifest_sha256": self.manifest_sha256,
            "task_ids_sha256": self.task_ids_sha256,
        }


PRODUCTION_PROTOCOL = ProtocolContract(
    protocol_id=GAIA_TEXT_PROTOCOL_ID,
    dataset_revision=GAIA_TEXT_DATASET_REVISION,
    split="validation",
    task_count=127,
    level_counts=((1, 42), (2, 66), (3, 19)),
    manifest_sha256=GAIA_TEXT_MANIFEST_SHA256,
    task_ids_sha256=GAIA_TEXT_TASK_IDS_SHA256,
)
