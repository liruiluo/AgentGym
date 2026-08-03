from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..latent_preference.schema import PreferenceProductPool, canonical_json_bytes
from ..recency_override import RecencyOverrideGenerator
from .question_format import render_compositional_question
from .schema import (
    BRANCH_COORDINATES,
    CanonicalMemoryFact,
    CompositionalRecallDataError,
    CompositionalRecallOrbit,
    CompositionalRecallTask,
)


DEFAULT_GENERATOR_VERSION = "compositional_recall_factorial_top1_v1"
CANONICAL_MEMORY_KEY = "memory record"


@dataclass(frozen=True)
class CompositionalRecallGenerator:
    pool: PreferenceProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise CompositionalRecallDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise CompositionalRecallDataError("generator version must be non-empty.")

    @property
    def source_generator(self) -> RecencyOverrideGenerator:
        return RecencyOverrideGenerator(pool=self.pool, seed=self.seed)

    @property
    def semantic_period_orbits(self) -> int:
        return self.source_generator.semantic_period_orbits

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * len(BRANCH_COORDINATES)

    def generate_orbit(
        self,
        orbit_index: int,
        *,
        split: str,
    ) -> CompositionalRecallOrbit:
        source_orbit = self.source_generator.generate_orbit(orbit_index, split=split)
        source_task = source_orbit.tasks[0]
        recipe = self.pool.recipe_by_id(source_task.recipe_id)
        profile_tokens = (
            f"pt.{self._digest('profile_token_a', source_orbit.orbit_id).hex()[:16]}",
            f"pt.{self._digest('profile_token_b', source_orbit.orbit_id).hex()[:16]}",
        )
        if profile_tokens[0] == profile_tokens[1]:
            raise AssertionError("profile token hash collision")
        digest = self._digest("orbit", source_orbit.orbit_id).hex()[:16]
        orbit_id = (
            f"amgcr.{split}.{orbit_index}.e{source_orbit.semantic_epoch}.{digest}"
        )
        tasks = tuple(
            self._build_task(
                orbit_id=orbit_id,
                source_task=source_task,
                profile_tokens=profile_tokens,
                mapping_branch=mapping_branch,
                directory_branch=directory_branch,
                recipe=recipe,
            )
            for mapping_branch, directory_branch in BRANCH_COORDINATES
        )
        return CompositionalRecallOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=source_orbit.semantic_epoch,
            source_recency_orbit_id=source_orbit.orbit_id,
            profile_tokens=profile_tokens,
            tasks=tasks,  # type: ignore[arg-type]
        )

    def _build_task(
        self,
        *,
        orbit_id,
        source_task,
        profile_tokens,
        mapping_branch,
        directory_branch,
        recipe,
    ) -> CompositionalRecallTask:
        old_value = source_task.old_attribute_value
        new_value = source_task.new_attribute_value
        directory_values = (
            (old_value, new_value)
            if directory_branch == "identity"
            else (new_value, old_value)
        )
        profile_directory = tuple(zip(profile_tokens, directory_values))
        active_profile_token = profile_tokens[
            0 if mapping_branch == "token_a" else 1
        ]
        preferred_value = dict(profile_directory)[active_profile_token]
        questions = tuple(
            render_compositional_question(
                user_id=source_task.user_id,
                phase_index=phase.phase_index,
                recipe=recipe,
                category_id=phase.category_id,
                candidates=phase.candidates,
                budget_cents=source_task.budget_cents,
                profile_tokens=profile_tokens,
                active_profile_token=active_profile_token,
                profile_directory=profile_directory,  # type: ignore[arg-type]
                one_time_attribute_value=old_value,
            )
            for phase in source_task.phases
        )
        target_asins = []
        for phase in source_task.phases:
            target_value = old_value if phase.phase_index == 0 else preferred_value
            matches = tuple(
                item for item in phase.candidates if item.attribute_value == target_value
            )
            if len(matches) != 1:
                raise CompositionalRecallDataError(
                    "target value must identify exactly one candidate."
                )
            target_asins.append(matches[0].asin)
        mapping_fact = CanonicalMemoryFact(
            role="customer_to_profile",
            key=CANONICAL_MEMORY_KEY,
            value=(
                f"Customer {source_task.user_id} routes to profile token "
                f"{active_profile_token}."
            ),
            query=f"customer {source_task.user_id} route",
        )
        directory_text = "; ".join(
            f"profile token {token} has {recipe.axis_display_name} "
            f"{recipe.value_display_name(value)}"
            for token, value in profile_directory
        )
        directory_fact = CanonicalMemoryFact(
            role="profile_directory",
            key=CANONICAL_MEMORY_KEY,
            value=f"Profile directory: {directory_text}.",
            query=(
                f"profile token {active_profile_token} "
                f"{recipe.axis_display_name}"
            ),
        )
        branch_kind = f"active_{mapping_branch}.directory_{directory_branch}"
        return CompositionalRecallTask(
            task_id=(
                f"{orbit_id}.t.{mapping_branch}.{directory_branch}."
                f"{self._digest('task', orbit_id, mapping_branch, directory_branch).hex()[:16]}"
            ),
            orbit_id=orbit_id,
            orbit_index=source_task.orbit_index,
            semantic_epoch=source_task.semantic_epoch,
            split=source_task.split,
            branch_kind=branch_kind,
            mapping_branch=mapping_branch,
            directory_branch=directory_branch,
            source_task=source_task,
            profile_tokens=profile_tokens,
            active_profile_token=active_profile_token,
            profile_directory=profile_directory,  # type: ignore[arg-type]
            preferred_attribute_value=preferred_value,
            questions=questions,
            target_asins=tuple(target_asins),
            canonical_memories=(mapping_fact, directory_fact),
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
