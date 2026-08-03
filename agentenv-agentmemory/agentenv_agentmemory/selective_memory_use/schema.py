from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import (
    LatentPreferenceTask,
    canonical_sha256,
    require_id,
    require_sha256,
)


SPLITS = ("train", "dev", "test")
MEMORY_REQUIREMENTS = ("memory_required", "memory_not_required")
BRANCH_SPECS = (
    ("memory_required_a", "memory_required", 0),
    ("memory_not_required_a", "memory_not_required", 0),
    ("memory_required_b", "memory_required", 1),
    ("memory_not_required_b", "memory_not_required", 1),
)
TASK_SCHEMA = "agentmemory_selective_memory_use_task_v1"
ORBIT_SCHEMA = "agentmemory_selective_memory_use_factorial_orbit_v1"
PROOF_SCHEMA = "agentmemory_selective_memory_use_proof_v1"


class SelectiveMemoryUseDataError(ValueError):
    """Raised when a memory-use decision task is not machine-verifiable."""


@dataclass(frozen=True)
class SeededProfileMemory:
    key: str
    value: str
    state: str

    def __post_init__(self) -> None:
        if self.state not in {"current", "stale"}:
            raise SelectiveMemoryUseDataError(
                f"invalid seeded profile state {self.state!r}."
            )
        for field_name in ("key", "value"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise SelectiveMemoryUseDataError(
                    f"seeded profile {field_name} must be non-empty text."
                )

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "value": self.value, "state": self.state}


@dataclass(frozen=True)
class SelectiveMemoryUseTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    branch_kind: str
    memory_requirement: str
    preference_coordinate: int
    preferred_attribute_value: str
    source_task: LatentPreferenceTask
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    initial_memory: SeededProfileMemory
    canonical_query: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        require_id(self.task_id, field="task_id")
        require_id(self.orbit_id, field="orbit_id")
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise SelectiveMemoryUseDataError("orbit_index must be non-negative.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise SelectiveMemoryUseDataError("semantic_epoch must be non-negative.")
        if self.split not in SPLITS or self.source_task.split != self.split:
            raise SelectiveMemoryUseDataError("invalid or inconsistent task split.")
        expected_specs = {
            branch: (requirement, coordinate)
            for branch, requirement, coordinate in BRANCH_SPECS
        }
        if self.branch_kind not in expected_specs:
            raise SelectiveMemoryUseDataError(
                f"invalid branch kind {self.branch_kind!r}."
            )
        if expected_specs[self.branch_kind] != (
            self.memory_requirement,
            self.preference_coordinate,
        ):
            raise SelectiveMemoryUseDataError(
                "branch kind disagrees with its factorial coordinates."
            )
        if self.memory_requirement not in MEMORY_REQUIREMENTS:
            raise SelectiveMemoryUseDataError("invalid memory requirement.")
        if self.preference_coordinate not in (0, 1):
            raise SelectiveMemoryUseDataError(
                "preference_coordinate must be zero or one."
            )
        if (
            self.source_task.preferred_attribute_value
            != self.preferred_attribute_value
        ):
            raise SelectiveMemoryUseDataError(
                "preferred value disagrees with the certified source task."
            )
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise SelectiveMemoryUseDataError(
                "selective memory task requires six shopping sessions."
            )
        if self.target_asins != self.source_task.target_asins:
            raise SelectiveMemoryUseDataError(
                "targets must preserve the certified source product choices."
            )
        expected_memory_state = (
            "current" if self.memory_requirement == "memory_required" else "stale"
        )
        if self.initial_memory.state != expected_memory_state:
            raise SelectiveMemoryUseDataError(
                "seeded memory state disagrees with the task requirement."
            )
        if not isinstance(self.canonical_query, str) or not self.canonical_query.strip():
            raise SelectiveMemoryUseDataError(
                "canonical_query must be non-empty text."
            )
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise SelectiveMemoryUseDataError(
                "generator_version must be non-empty."
            )
        if isinstance(self.generator_seed, bool) or not isinstance(
            self.generator_seed, int
        ):
            raise SelectiveMemoryUseDataError("generator_seed must be an integer.")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
        if self.source_task.product_pool_sha256 != self.product_pool_sha256:
            raise SelectiveMemoryUseDataError("source task product pool mismatch.")

    @property
    def budget_cents(self) -> int:
        return self.source_task.budget_cents

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "split": self.split,
            "branch_kind": self.branch_kind,
            "memory_requirement": self.memory_requirement,
            "preference_coordinate": self.preference_coordinate,
            "preferred_attribute_value": self.preferred_attribute_value,
            "source_task": self.source_task.as_dict(include_targets=True),
            "questions": list(self.questions),
            "target_asins": list(self.target_asins),
            "initial_memory": self.initial_memory.as_dict(),
            "canonical_query": self.canonical_query,
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }


@dataclass(frozen=True)
class SelectiveMemoryUseOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    source_preference_orbit_id: str
    tasks: tuple[
        SelectiveMemoryUseTask,
        SelectiveMemoryUseTask,
        SelectiveMemoryUseTask,
        SelectiveMemoryUseTask,
    ]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(
            self.source_preference_orbit_id,
            field="source_preference_orbit_id",
        )
        actual_specs = tuple(
            (
                task.branch_kind,
                task.memory_requirement,
                task.preference_coordinate,
            )
            for task in self.tasks
        )
        if actual_specs != BRANCH_SPECS:
            raise SelectiveMemoryUseDataError(
                "orbit tasks must use the canonical 2x2 factorial order."
            )
        for task in self.tasks:
            if (
                task.orbit_id,
                task.orbit_index,
                task.semantic_epoch,
                task.source_task.orbit_id,
            ) != (
                self.orbit_id,
                self.orbit_index,
                self.semantic_epoch,
                self.source_preference_orbit_id,
            ):
                raise SelectiveMemoryUseDataError(
                    "task identity disagrees with its orbit."
                )

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ORBIT_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "source_preference_orbit_id": self.source_preference_orbit_id,
            "tasks": [task.as_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class SelectiveMemoryUseBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    branch_kind: str
    memory_requirement: str
    preferred_attribute_value: str
    initial_memory: SeededProfileMemory
    canonical_query: str
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise SelectiveMemoryUseDataError("bundle requires six shopping phases.")
        if self.split not in SPLITS:
            raise SelectiveMemoryUseDataError("invalid bundle split.")
        if self.memory_requirement not in MEMORY_REQUIREMENTS:
            raise SelectiveMemoryUseDataError("invalid bundle memory requirement.")
        if self.branch_kind not in {item[0] for item in BRANCH_SPECS}:
            raise SelectiveMemoryUseDataError("invalid bundle branch kind.")
        require_sha256(self.proof_sha256, field="proof_sha256")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
