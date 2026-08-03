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
BRANCH_KINDS = ("preference_a", "preference_b")
TASK_SCHEMA = "agentmemory_intent_clarification_task_v1"
ORBIT_SCHEMA = "agentmemory_intent_clarification_counterfactual_orbit_v1"
PROOF_SCHEMA = "agentmemory_intent_clarification_proof_v1"


class IntentClarificationDataError(ValueError):
    """Raised when an intent-clarification task is not machine-verifiable."""


@dataclass(frozen=True)
class ClarificationMemoryFact:
    key: str
    value: str
    query: str

    def __post_init__(self) -> None:
        for field_name in ("key", "value", "query"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise IntentClarificationDataError(
                    f"clarification memory {field_name} must be non-empty text."
                )

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "value": self.value, "query": self.query}


@dataclass(frozen=True)
class IntentClarificationTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    branch_kind: str
    source_task: LatentPreferenceTask
    clarification_field: str
    clarification_answer: str
    preferred_attribute_value: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    canonical_memory: ClarificationMemoryFact
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
            raise IntentClarificationDataError("invalid orbit_index.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise IntentClarificationDataError("invalid semantic_epoch.")
        if self.split not in SPLITS or self.source_task.split != self.split:
            raise IntentClarificationDataError("invalid or inconsistent split.")
        if self.branch_kind not in BRANCH_KINDS:
            raise IntentClarificationDataError("invalid clarification branch kind.")
        if self.source_task.preferred_attribute_value != self.preferred_attribute_value:
            raise IntentClarificationDataError(
                "preferred value disagrees with the certified source task."
            )
        require_id(self.clarification_field, field="clarification_field")
        if not isinstance(self.clarification_answer, str) or not self.clarification_answer.strip():
            raise IntentClarificationDataError(
                "clarification_answer must be non-empty text."
            )
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise IntentClarificationDataError("task requires six shopping sessions.")
        if self.target_asins != self.source_task.target_asins:
            raise IntentClarificationDataError(
                "targets must preserve the certified source product choices."
            )
        if isinstance(self.generator_seed, bool) or not isinstance(
            self.generator_seed, int
        ):
            raise IntentClarificationDataError("generator_seed must be an integer.")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise IntentClarificationDataError("generator_version must be non-empty.")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
        if self.source_task.product_pool_sha256 != self.product_pool_sha256:
            raise IntentClarificationDataError("source task product pool mismatch.")

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
            "clarification_field": self.clarification_field,
            "clarification_answer": self.clarification_answer,
            "preferred_attribute_value": self.preferred_attribute_value,
            "questions": list(self.questions),
            "target_asins": list(self.target_asins),
            "canonical_memory": self.canonical_memory.as_dict(),
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }


@dataclass(frozen=True)
class IntentClarificationOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    source_preference_orbit_id: str
    tasks: tuple[IntentClarificationTask, IntentClarificationTask]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(
            self.source_preference_orbit_id,
            field="source_preference_orbit_id",
        )
        if tuple(task.branch_kind for task in self.tasks) != BRANCH_KINDS:
            raise IntentClarificationDataError(
                "orbit tasks must use the canonical counterfactual order."
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
                raise IntentClarificationDataError(
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
class IntentClarificationBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    branch_kind: str
    clarification_field: str
    clarification_answer: str
    canonical_memory: ClarificationMemoryFact
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise IntentClarificationDataError("bundle requires six shopping sessions.")
        if self.split not in SPLITS:
            raise IntentClarificationDataError("invalid bundle split.")
        if self.branch_kind not in BRANCH_KINDS:
            raise IntentClarificationDataError("invalid bundle branch kind.")
        require_id(self.clarification_field, field="clarification_field")
        require_sha256(self.proof_sha256, field="proof_sha256")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
