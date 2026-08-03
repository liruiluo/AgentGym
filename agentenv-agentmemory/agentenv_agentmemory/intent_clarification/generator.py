from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..latent_preference import LatentPreferenceGenerator
from ..latent_preference.schema import PreferenceProductPool, canonical_json_bytes
from .question_format import render_intent_clarification_question
from .schema import (
    BRANCH_KINDS,
    ClarificationMemoryFact,
    IntentClarificationDataError,
    IntentClarificationOrbit,
    IntentClarificationTask,
)


DEFAULT_GENERATOR_VERSION = "intent_clarification_counterfactual_top1_v1"
CANONICAL_MEMORY_KEY = "customer clarification"


@dataclass(frozen=True)
class IntentClarificationGenerator:
    pool: PreferenceProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise IntentClarificationDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise IntentClarificationDataError("generator version must be non-empty.")

    @property
    def source_generator(self) -> LatentPreferenceGenerator:
        return LatentPreferenceGenerator(pool=self.pool, seed=self.seed)

    @property
    def semantic_period_orbits(self) -> int:
        return self.source_generator.semantic_period_orbits

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * 2

    def generate_orbit(
        self,
        orbit_index: int,
        *,
        split: str,
    ) -> IntentClarificationOrbit:
        source_orbit = self.source_generator.generate_orbit(orbit_index, split=split)
        recipe = self.pool.recipe_by_id(source_orbit.recipe_id)
        digest = self._digest("orbit", source_orbit.orbit_id).hex()[:16]
        orbit_id = (
            f"amgic.{split}.{orbit_index}.e{source_orbit.semantic_epoch}.{digest}"
        )
        tasks = tuple(
            self._build_task(
                orbit_id=orbit_id,
                branch_kind=BRANCH_KINDS[branch_index],
                source_task=source_task,
                recipe=recipe,
            )
            for branch_index, source_task in enumerate(source_orbit.tasks)
        )
        return IntentClarificationOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=source_orbit.semantic_epoch,
            source_preference_orbit_id=source_orbit.orbit_id,
            tasks=tasks,  # type: ignore[arg-type]
        )

    def _build_task(
        self,
        *,
        orbit_id,
        branch_kind,
        source_task,
        recipe,
    ) -> IntentClarificationTask:
        questions = tuple(
            render_intent_clarification_question(
                user_id=source_task.user_id,
                phase_index=phase.phase_index,
                recipe=recipe,
                category_id=phase.category_id,
                candidates=phase.candidates,
                budget_cents=source_task.budget_cents,
            )
            for phase in source_task.phases
        )
        display_value = recipe.value_display_name(
            source_task.preferred_attribute_value
        )
        clarification_answer = (
            f"For {recipe.axis_display_name}, I want {display_value}."
        )
        canonical_memory = ClarificationMemoryFact(
            key=CANONICAL_MEMORY_KEY,
            value=(
                f"Customer {source_task.user_id} clarified their "
                f"{recipe.axis_display_name} preference as {display_value}."
            ),
            query=(
                f"customer {source_task.user_id} clarified "
                f"{recipe.axis_display_name} preference"
            ),
        )
        task_digest = self._digest(
            "task",
            orbit_id,
            branch_kind,
            source_task.preferred_attribute_value,
        ).hex()[:16]
        return IntentClarificationTask(
            task_id=f"{orbit_id}.t.{branch_kind}.{task_digest}",
            orbit_id=orbit_id,
            orbit_index=source_task.orbit_index,
            semantic_epoch=source_task.semantic_epoch,
            split=source_task.split,
            branch_kind=branch_kind,
            source_task=source_task,
            clarification_field=recipe.axis,
            clarification_answer=clarification_answer,
            preferred_attribute_value=source_task.preferred_attribute_value,
            questions=questions,
            target_asins=source_task.target_asins,
            canonical_memory=canonical_memory,
            generator_version=self.version,
            generator_seed=self.seed,
            product_pool_sha256=self.pool.semantic_sha256,
        )

    def _digest(self, *parts: object) -> bytes:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "version": self.version,
                    "seed": self.seed,
                    "pool": self.pool.semantic_sha256,
                    "parts": list(parts),
                }
            )
        ).digest()
