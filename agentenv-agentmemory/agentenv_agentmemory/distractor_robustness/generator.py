from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..latent_preference.schema import PreferenceProductPool, canonical_json_bytes
from ..recency_override import RecencyOverrideGenerator
from .schema import (
    BRANCH_KINDS,
    DistractorRobustnessDataError,
    DistractorRobustnessOrbit,
    DistractorRobustnessTask,
    InitialMemory,
)


DEFAULT_GENERATOR_VERSION = "distractor_robustness_top1_v1"
DEFAULT_DISTRACTOR_COUNT = 8
CANONICAL_MEMORY_KEY = "customer profile"
AUXILIARY_PROFILE_FACTS = (
    ("delivery cadence", "weekly"),
    ("packaging preference", "minimal"),
    ("pickup window", "evening"),
    ("warranty preference", "extended"),
)


@dataclass(frozen=True)
class DistractorRobustnessGenerator:
    pool: PreferenceProductPool
    seed: int
    distractor_count: int = DEFAULT_DISTRACTOR_COUNT
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DistractorRobustnessDataError("generator seed must be an integer.")
        if (
            isinstance(self.distractor_count, bool)
            or not isinstance(self.distractor_count, int)
            or not 1 <= self.distractor_count <= 64
        ):
            raise DistractorRobustnessDataError(
                "distractor_count must be an integer from 1 through 64."
            )
        if not isinstance(self.version, str) or not self.version:
            raise DistractorRobustnessDataError(
                "generator version must be non-empty."
            )

    @property
    def semantic_period_orbits(self) -> int:
        return self.source_generator.semantic_period_orbits

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * 2

    @property
    def source_generator(self) -> RecencyOverrideGenerator:
        return RecencyOverrideGenerator(pool=self.pool, seed=self.seed)

    def generate_orbit(
        self,
        orbit_index: int,
        *,
        split: str,
    ) -> DistractorRobustnessOrbit:
        source_orbit = self.source_generator.generate_orbit(orbit_index, split=split)
        source_task = source_orbit.tasks[0]
        recipe = self.pool.recipe_by_id(source_task.recipe_id)
        current_value = recipe.value_display_name(source_task.old_attribute_value)
        canonical_value = (
            f"Current confirmed preference for customer {source_task.user_id}: "
            f"{recipe.axis_display_name} is {current_value}."
        )
        canonical_query = (
            f"current confirmed preference customer {source_task.user_id} "
            f"{recipe.axis_display_name}"
        )
        distractors = self._build_distractors(
            source_task=source_task,
            recipe=recipe,
        )
        digest = self._digest(
            "orbit",
            source_orbit.orbit_id,
            self.distractor_count,
        ).hex()[:16]
        orbit_id = (
            f"amgdr.{split}.{orbit_index}.e{source_orbit.semantic_epoch}.{digest}"
        )
        tasks = tuple(
            DistractorRobustnessTask(
                task_id=(
                    f"{orbit_id}.t.{branch}."
                    f"{self._digest('task', orbit_id, branch).hex()[:16]}"
                ),
                orbit_id=orbit_id,
                orbit_index=orbit_index,
                semantic_epoch=source_orbit.semantic_epoch,
                split=split,
                branch_kind=branch,
                source_task=source_task,
                initial_memories=() if branch == "clean" else distractors,
                canonical_memory_key=CANONICAL_MEMORY_KEY,
                canonical_memory_value=canonical_value,
                canonical_query=canonical_query,
                generator_version=self.version,
                generator_seed=self.seed,
                product_pool_sha256=self.pool.semantic_sha256,
            )
            for branch in BRANCH_KINDS
        )
        return DistractorRobustnessOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=source_orbit.semantic_epoch,
            source_recency_orbit_id=source_orbit.orbit_id,
            tasks=tasks,  # type: ignore[arg-type]
        )

    def _build_distractors(self, *, source_task, recipe) -> tuple[InitialMemory, ...]:
        opposite_value = recipe.value_display_name(source_task.new_attribute_value)
        other_recipes = tuple(
            item
            for item in self.pool.recipes
            if item.recipe_id != recipe.recipe_id and item.axis != recipe.axis
        )
        memories = []
        for ordinal in range(self.distractor_count):
            kind_index = ordinal % 4
            nonce = self._digest(
                "distractor",
                source_task.task_id,
                ordinal,
            ).hex()[:10]
            other_user = f"shopper.{source_task.split}.other.{nonce}"
            if other_recipes:
                other_recipe = other_recipes[ordinal % len(other_recipes)]
                other_axis = other_recipe.axis_display_name
                other_value = other_recipe.value_display_name(
                    other_recipe.values[ordinal % len(other_recipe.values)]
                )
            else:
                other_axis, other_value = AUXILIARY_PROFILE_FACTS[
                    ordinal % len(AUXILIARY_PROFILE_FACTS)
                ]
            if kind_index == 0:
                kind = "same_customer_superseded_same_axis"
                tier = "high"
                value = (
                    f"Superseded historical preference for customer "
                    f"{source_task.user_id}: {recipe.axis_display_name} was "
                    f"{opposite_value}. Archive record {nonce}."
                )
            elif kind_index == 1:
                kind = "other_customer_current_same_axis"
                tier = "high"
                value = (
                    f"Current confirmed preference for customer {other_user}: "
                    f"{recipe.axis_display_name} is {opposite_value}. "
                    f"Profile record {nonce}."
                )
            elif kind_index == 2:
                kind = "same_customer_current_other_axis"
                tier = "medium"
                value = (
                    f"Current confirmed preference for customer "
                    f"{source_task.user_id}: {other_axis} is "
                    f"{other_value}. Profile record {nonce}."
                )
            else:
                kind = "other_customer_historical_other_axis"
                tier = "low"
                value = (
                    f"Historical profile for customer {other_user}: "
                    f"{other_axis} was {other_value}. "
                    f"Archive record {nonce}."
                )
            memories.append(
                InitialMemory(
                    key=CANONICAL_MEMORY_KEY,
                    value=value,
                    distractor_kind=kind,
                    similarity_tier=tier,
                )
            )
        if len({(item.key, item.value) for item in memories}) != len(memories):
            raise AssertionError("distractor generator produced duplicate memories")
        return tuple(memories)

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
