from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..latent_preference.schema import (
    PreferenceCandidate,
    PreferenceProductPool,
    PreferenceRecipe,
    canonical_sha256,
    normalize_native_title,
    require_id,
    require_sha256,
)


SPLITS = ("train", "dev", "test")
PHASE_KINDS = ("evidence", "application", "override")
TASK_SCHEMA = "agentmemory_recency_override_task_v1"
ORBIT_SCHEMA = "agentmemory_recency_override_counterfactual_orbit_v1"
PROOF_SCHEMA = "agentmemory_recency_override_proof_v1"
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


class RecencyOverrideDataError(ValueError):
    """Raised when a recency-override example is not machine-verifiable."""


@dataclass(frozen=True)
class RecencyPhase:
    phase_index: int
    phase_kind: str
    category_id: str
    category_display_name: str
    candidates: tuple[PreferenceCandidate, PreferenceCandidate]
    question: str
    target_asin: str
    confirmed_attribute_value: str | None
    active_attribute_value: str

    def __post_init__(self) -> None:
        if self.phase_index not in range(6):
            raise RecencyOverrideDataError("phase index must be in [0, 6).")
        if self.phase_kind not in PHASE_KINDS:
            raise RecencyOverrideDataError(f"invalid phase kind {self.phase_kind!r}.")
        if len(self.candidates) != 2 or len({item.asin for item in self.candidates}) != 2:
            raise RecencyOverrideDataError("each phase requires two distinct candidates.")
        require_id(self.category_id, field="phase category_id")
        if not isinstance(self.category_display_name, str) or not self.category_display_name.strip():
            raise RecencyOverrideDataError("phase category display name must be non-empty.")
        for candidate in self.candidates:
            if candidate.category_id != self.category_id:
                raise RecencyOverrideDataError("candidate category disagrees with phase.")
            if candidate.category_display_name != self.category_display_name:
                raise RecencyOverrideDataError("candidate category display disagrees with phase.")
        if not isinstance(self.question, str) or not self.question.strip():
            raise RecencyOverrideDataError("phase question must be non-empty.")
        if self.target_asin not in {item.asin for item in self.candidates}:
            raise RecencyOverrideDataError("phase target must be an approved candidate.")
        if not isinstance(self.active_attribute_value, str) or not self.active_attribute_value:
            raise RecencyOverrideDataError("active preference value must be non-empty.")
        values = {item.attribute_value for item in self.candidates}
        if self.active_attribute_value not in values:
            raise RecencyOverrideDataError("active preference value is absent from candidates.")
        if self.phase_kind in {"evidence", "override"}:
            if self.confirmed_attribute_value not in values:
                raise RecencyOverrideDataError("confirmation must identify one candidate value.")
        elif self.confirmed_attribute_value is not None:
            raise RecencyOverrideDataError("application phase cannot expose a confirmed value.")

    def as_dict(self, *, include_target: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase_index": self.phase_index,
            "phase_kind": self.phase_kind,
            "category_id": self.category_id,
            "category_display_name": self.category_display_name,
            "candidates": [item.as_dict() for item in self.candidates],
            "question": self.question,
            "confirmed_attribute_value": self.confirmed_attribute_value,
            "active_attribute_value": self.active_attribute_value,
        }
        if include_target:
            payload["target_asin"] = self.target_asin
        return payload


@dataclass(frozen=True)
class RecencyOverrideTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    user_id: str
    split: str
    branch_kind: str
    old_attribute_value: str
    new_attribute_value: str
    canonical_memory_key: str
    override_mode: str
    budget_cents: int
    phases: tuple[RecencyPhase, ...]
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("task_id", "orbit_id", "recipe_id", "user_id"):
            require_id(getattr(self, field_name), field=field_name)
        if isinstance(self.orbit_index, bool) or not isinstance(self.orbit_index, int) or self.orbit_index < 0:
            raise RecencyOverrideDataError("orbit_index must be a non-negative integer.")
        if isinstance(self.semantic_epoch, bool) or not isinstance(self.semantic_epoch, int) or self.semantic_epoch < 0:
            raise RecencyOverrideDataError("semantic_epoch must be a non-negative integer.")
        if self.split not in SPLITS:
            raise RecencyOverrideDataError(f"invalid split {self.split!r}.")
        if self.branch_kind not in {"stay", "flip"}:
            raise RecencyOverrideDataError("branch_kind must be stay or flip.")
        if self.old_attribute_value == self.new_attribute_value:
            raise RecencyOverrideDataError("old and new preference values must differ.")
        require_id(self.canonical_memory_key, field="canonical_memory_key")
        expected_mode = "none" if self.branch_kind == "stay" else "update_or_delete_add"
        if self.override_mode != expected_mode:
            raise RecencyOverrideDataError("override_mode disagrees with branch_kind.")
        if isinstance(self.budget_cents, bool) or not isinstance(self.budget_cents, int) or self.budget_cents <= 0:
            raise RecencyOverrideDataError("budget must be a positive integer number of cents.")
        if len(self.phases) != 6 or tuple(item.phase_index for item in self.phases) != tuple(range(6)):
            raise RecencyOverrideDataError("phases must be ordered from 0 through 5.")
        if any(item.active_attribute_value not in {self.old_attribute_value, self.new_attribute_value} for item in self.phases):
            raise RecencyOverrideDataError("phase active value is outside the old/new pair.")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
        if isinstance(self.generator_seed, bool) or not isinstance(self.generator_seed, int):
            raise RecencyOverrideDataError("generator_seed must be an integer.")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise RecencyOverrideDataError("generator_version must be non-empty.")

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(item.question for item in self.phases)

    @property
    def target_asins(self) -> tuple[str, ...]:
        return tuple(item.target_asin for item in self.phases)

    @property
    def active_values(self) -> tuple[str, ...]:
        return tuple(item.active_attribute_value for item in self.phases)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_targets=True))

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "recipe_id": self.recipe_id,
            "user_id": self.user_id,
            "split": self.split,
            "branch_kind": self.branch_kind,
            "old_attribute_value": self.old_attribute_value,
            "new_attribute_value": self.new_attribute_value,
            "canonical_memory_key": self.canonical_memory_key,
            "override_mode": self.override_mode,
            "budget_cents": self.budget_cents,
            "phases": [item.as_dict(include_target=include_targets) for item in self.phases],
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }
        return payload


@dataclass(frozen=True)
class RecencyOverrideOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    user_id: str
    old_attribute_value: str
    new_attribute_value: str
    tasks: tuple[RecencyOverrideTask, RecencyOverrideTask]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.recipe_id, field="recipe_id")
        require_id(self.user_id, field="user_id")
        if isinstance(self.orbit_index, bool) or not isinstance(self.orbit_index, int) or self.orbit_index < 0:
            raise RecencyOverrideDataError("orbit_index must be non-negative.")
        if isinstance(self.semantic_epoch, bool) or not isinstance(self.semantic_epoch, int) or self.semantic_epoch < 0:
            raise RecencyOverrideDataError("semantic_epoch must be non-negative.")
        if self.old_attribute_value == self.new_attribute_value:
            raise RecencyOverrideDataError("orbit old/new values must differ.")
        if len(self.tasks) != 2 or tuple(item.branch_kind for item in self.tasks) != ("stay", "flip"):
            raise RecencyOverrideDataError("orbit tasks must be ordered stay then flip.")
        for task in self.tasks:
            if (task.orbit_id, task.orbit_index, task.semantic_epoch, task.recipe_id, task.user_id) != (
                self.orbit_id, self.orbit_index, self.semantic_epoch, self.recipe_id, self.user_id
            ):
                raise RecencyOverrideDataError("task identity disagrees with orbit.")
            if (task.old_attribute_value, task.new_attribute_value) != (
                self.old_attribute_value, self.new_attribute_value
            ):
                raise RecencyOverrideDataError("task old/new values disagree with orbit.")

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_targets=True))

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": ORBIT_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "recipe_id": self.recipe_id,
            "user_id": self.user_id,
            "old_attribute_value": self.old_attribute_value,
            "new_attribute_value": self.new_attribute_value,
            "tasks": [item.as_dict(include_targets=include_targets) for item in self.tasks],
        }
        return payload


@dataclass(frozen=True)
class RecencyOverrideBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    active_values: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    recipe_id: str
    user_id: str
    branch_kind: str
    old_attribute_value: str
    new_attribute_value: str
    canonical_memory_key: str
    override_mode: str
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6 or len(self.active_values) != 6:
            raise RecencyOverrideDataError("bundle requires six phases.")
        if self.split not in SPLITS:
            raise RecencyOverrideDataError("invalid bundle split.")
        if self.branch_kind not in {"stay", "flip"}:
            raise RecencyOverrideDataError("invalid bundle branch.")
        require_id(self.canonical_memory_key, field="bundle canonical_memory_key")
        require_sha256(self.proof_sha256, field="bundle proof_sha256")
        require_sha256(self.product_pool_sha256, field="bundle product_pool_sha256")


def candidate_from_product(product, *, product_pool_sha256: str) -> PreferenceCandidate:
    """Build the policy-visible candidate while keeping pool identity attached."""

    return PreferenceCandidate.from_product(product, product_pool_sha256=product_pool_sha256)
