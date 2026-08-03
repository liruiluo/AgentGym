from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import canonical_sha256, require_id, require_sha256
from ..recency_override.schema import RecencyOverrideTask


SPLITS = ("train", "dev", "test")
BRANCH_KINDS = ("clean", "distracted")
SIMILARITY_TIERS = ("high", "medium", "low")
TASK_SCHEMA = "agentmemory_distractor_robustness_task_v1"
ORBIT_SCHEMA = "agentmemory_distractor_robustness_orbit_v1"
PROOF_SCHEMA = "agentmemory_distractor_robustness_proof_v1"


class DistractorRobustnessDataError(ValueError):
    """Raised when a selective-retrieval task is not machine-verifiable."""


@dataclass(frozen=True)
class InitialMemory:
    key: str
    value: str
    distractor_kind: str
    similarity_tier: str

    def __post_init__(self) -> None:
        for field_name in ("key", "value", "distractor_kind"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistractorRobustnessDataError(
                    f"initial-memory {field_name} must be non-empty text."
                )
        if self.similarity_tier not in SIMILARITY_TIERS:
            raise DistractorRobustnessDataError(
                f"invalid similarity tier {self.similarity_tier!r}."
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "value": self.value,
            "distractor_kind": self.distractor_kind,
            "similarity_tier": self.similarity_tier,
        }


@dataclass(frozen=True)
class DistractorRobustnessTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    branch_kind: str
    source_task: RecencyOverrideTask
    initial_memories: tuple[InitialMemory, ...]
    canonical_memory_key: str
    canonical_memory_value: str
    canonical_query: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("task_id", "orbit_id"):
            require_id(getattr(self, field_name), field=field_name)
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise DistractorRobustnessDataError(
                "orbit_index must be a non-negative integer."
            )
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise DistractorRobustnessDataError(
                "semantic_epoch must be a non-negative integer."
            )
        if self.split not in SPLITS or self.source_task.split != self.split:
            raise DistractorRobustnessDataError("task split is invalid or inconsistent.")
        if self.branch_kind not in BRANCH_KINDS:
            raise DistractorRobustnessDataError(
                f"invalid branch kind {self.branch_kind!r}."
            )
        if self.source_task.branch_kind != "stay":
            raise DistractorRobustnessDataError(
                "distractor tasks must reuse the no-change recency branch."
            )
        expected_count = 0 if self.branch_kind == "clean" else len(self.initial_memories)
        if self.branch_kind == "clean" and self.initial_memories:
            raise DistractorRobustnessDataError(
                "clean branch cannot preload distractor memories."
            )
        if self.branch_kind == "distracted" and expected_count < 1:
            raise DistractorRobustnessDataError(
                "distracted branch must preload at least one memory."
            )
        for field_name in (
            "canonical_memory_key",
            "canonical_memory_value",
            "canonical_query",
            "generator_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DistractorRobustnessDataError(
                    f"{field_name} must be non-empty text."
                )
        if isinstance(self.generator_seed, bool) or not isinstance(
            self.generator_seed, int
        ):
            raise DistractorRobustnessDataError("generator_seed must be an integer.")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
        if self.source_task.product_pool_sha256 != self.product_pool_sha256:
            raise DistractorRobustnessDataError(
                "source task and distractor task disagree on product pool."
            )

    @property
    def questions(self) -> tuple[str, ...]:
        return self.source_task.questions

    @property
    def target_asins(self) -> tuple[str, ...]:
        return self.source_task.target_asins

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
            "source_task": self.source_task.as_dict(include_targets=True),
            "initial_memories": [item.as_dict() for item in self.initial_memories],
            "canonical_memory_key": self.canonical_memory_key,
            "canonical_memory_value": self.canonical_memory_value,
            "canonical_query": self.canonical_query,
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }


@dataclass(frozen=True)
class DistractorRobustnessOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    source_recency_orbit_id: str
    tasks: tuple[DistractorRobustnessTask, DistractorRobustnessTask]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.source_recency_orbit_id, field="source_recency_orbit_id")
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise DistractorRobustnessDataError("invalid orbit_index.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise DistractorRobustnessDataError("invalid semantic_epoch.")
        if tuple(task.branch_kind for task in self.tasks) != BRANCH_KINDS:
            raise DistractorRobustnessDataError(
                "orbit tasks must be ordered clean then distracted."
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
                self.source_recency_orbit_id,
            ):
                raise DistractorRobustnessDataError(
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
            "source_recency_orbit_id": self.source_recency_orbit_id,
            "tasks": [task.as_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class DistractorRobustnessBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    branch_kind: str
    initial_memories: tuple[InitialMemory, ...]
    canonical_query: str
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise DistractorRobustnessDataError("bundle requires six shopping phases.")
        if self.split not in SPLITS or self.branch_kind not in BRANCH_KINDS:
            raise DistractorRobustnessDataError("invalid bundle split or branch.")
        require_sha256(self.proof_sha256, field="proof_sha256")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
