from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..latent_preference import LatentPreferenceGenerator
from ..latent_preference.schema import PreferenceProductPool, canonical_json_bytes
from .question_format import render_selective_memory_question
from .schema import (
    BRANCH_SPECS,
    SeededProfileMemory,
    SelectiveMemoryUseDataError,
    SelectiveMemoryUseOrbit,
    SelectiveMemoryUseTask,
)


DEFAULT_GENERATOR_VERSION = "selective_memory_use_top1_v2"
CANONICAL_MEMORY_KEY = "customer profile"


@dataclass(frozen=True)
class SelectiveMemoryUseGenerator:
    pool: PreferenceProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SelectiveMemoryUseDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise SelectiveMemoryUseDataError(
                "generator version must be non-empty."
            )

    @property
    def source_generator(self) -> LatentPreferenceGenerator:
        return LatentPreferenceGenerator(pool=self.pool, seed=self.seed)

    @property
    def semantic_period_orbits(self) -> int:
        return self.source_generator.semantic_period_orbits

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * len(BRANCH_SPECS)

    def generate_orbit(
        self,
        orbit_index: int,
        *,
        split: str,
    ) -> SelectiveMemoryUseOrbit:
        source_orbit = self.source_generator.generate_orbit(
            orbit_index,
            split=split,
        )
        digest = self._digest("orbit", source_orbit.orbit_id).hex()[:16]
        orbit_id = (
            f"amgsmu.{split}.{orbit_index}.e{source_orbit.semantic_epoch}.{digest}"
        )
        recipe = self.pool.recipe_by_id(source_orbit.recipe_id)
        tasks = []
        for branch_kind, memory_requirement, coordinate in BRANCH_SPECS:
            source_task = source_orbit.tasks[coordinate]
            preferred_value = source_task.preferred_attribute_value
            opposite_value = source_orbit.tasks[1 - coordinate].preferred_attribute_value
            stored_value = (
                preferred_value
                if memory_requirement == "memory_required"
                else opposite_value
            )
            stored_display = recipe.value_display_name(stored_value)
            if memory_requirement == "memory_required":
                memory_value = (
                    f"Current confirmed profile for customer {source_task.user_id}: "
                    f"{recipe.axis_display_name} is {stored_display}."
                )
                memory_state = "current"
            else:
                memory_value = (
                    f"Saved profile for customer {source_task.user_id}: "
                    f"{recipe.axis_display_name} is {stored_display}."
                )
                memory_state = "stale"
            questions = tuple(
                render_selective_memory_question(
                    user_id=source_task.user_id,
                    phase_index=phase.phase_index,
                    memory_requirement=memory_requirement,
                    recipe=recipe,
                    category_id=phase.category_id,
                    candidates=phase.candidates,
                    budget_cents=source_task.budget_cents,
                    preferred_attribute_value=preferred_value,
                )
                for phase in source_task.phases
            )
            task_digest = self._digest(
                "task",
                orbit_id,
                branch_kind,
            ).hex()[:16]
            tasks.append(
                SelectiveMemoryUseTask(
                    task_id=f"{orbit_id}.t.{branch_kind}.{task_digest}",
                    orbit_id=orbit_id,
                    orbit_index=orbit_index,
                    semantic_epoch=source_orbit.semantic_epoch,
                    split=split,
                    branch_kind=branch_kind,
                    memory_requirement=memory_requirement,
                    preference_coordinate=coordinate,
                    preferred_attribute_value=preferred_value,
                    source_task=source_task,
                    questions=questions,
                    target_asins=source_task.target_asins,
                    initial_memory=SeededProfileMemory(
                        key=CANONICAL_MEMORY_KEY,
                        value=memory_value,
                        state=memory_state,
                    ),
                    canonical_query=(
                        f"current profile customer {source_task.user_id} "
                        f"{recipe.axis_display_name}"
                    ),
                    generator_version=self.version,
                    generator_seed=self.seed,
                    product_pool_sha256=self.pool.semantic_sha256,
                )
            )
        return SelectiveMemoryUseOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=source_orbit.semantic_epoch,
            source_preference_orbit_id=source_orbit.orbit_id,
            tasks=tuple(tasks),  # type: ignore[arg-type]
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
