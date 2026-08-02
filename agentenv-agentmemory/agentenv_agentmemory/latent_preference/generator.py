from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .question_format import render_preference_question
from .schema import (
    SPLITS,
    LatentPreferenceDataError,
    LatentPreferenceOrbit,
    LatentPreferenceTask,
    CertifiedPreferenceProduct,
    PreferenceCandidate,
    PreferencePhase,
    PreferenceProductPool,
    PreferenceRecipe,
    canonical_json_bytes,
)


DEFAULT_GENERATOR_VERSION = "native_same_axis_latent_preference_v1"
CORE_BUDGET_MARGIN_CENTS = 5_000
CORE_BUDGET_ROUNDING_CENTS = 1_000
EVIDENCE_COUNTS = (1, 2, 3)
CATEGORY_SCHEDULES = {
    1: (0, 1, 2, 3, 1, 2),
    2: (0, 1, 2, 3, 2, 3),
    3: (0, 1, 2, 3, 0, 3),
}


@dataclass(frozen=True)
class LatentPreferenceGenerator:
    """Generate a deterministic counterfactual stream over real products.

    Each semantic coordinate chooses one recipe, an evidence curriculum, twelve
    non-reused native products, and six candidate-order bits. An affine seed
    permutation changes stream order without collisions inside the complete
    semantic period.
    """

    pool: PreferenceProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise LatentPreferenceDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise LatentPreferenceDataError("generator version must be non-empty.")
        capacities = {
            self.recipe_semantic_capacity(recipe, split)
            for recipe in self.pool.recipes
            for split in SPLITS
        }
        if len(capacities) != 1:
            raise LatentPreferenceDataError(
                "balanced preference pools must give every recipe and split one "
                "common semantic period."
            )

    @property
    def semantic_period_orbits(self) -> int:
        return len(self.pool.recipes) * self.recipe_semantic_capacity(
            self.pool.recipes[0], "train"
        )

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * 2

    @property
    def conservative_task_capacity_without_candidate_order(self) -> int:
        return self.semantic_period_tasks // (2**6)

    def recipe_semantic_capacity(
        self,
        recipe: PreferenceRecipe,
        split: str,
    ) -> int:
        if split not in SPLITS:
            raise LatentPreferenceDataError(f"invalid split {split!r}.")
        for evidence_count in EVIDENCE_COUNTS:
            schedule = CATEGORY_SCHEDULES[evidence_count]
            assignment_count = 1
            for category_position, category_id in enumerate(recipe.categories):
                occurrences = schedule.count(category_position)
                for value in recipe.values:
                    product_count = len(
                        self.pool.products_for(
                            axis=recipe.axis,
                            category_id=category_id,
                            attribute_value=value,
                            split=split,
                        )
                    )
                    assignment_count *= math.perm(product_count, occurrences)
            if evidence_count == EVIDENCE_COUNTS[0]:
                first_assignment_count = assignment_count
            elif assignment_count != first_assignment_count:
                raise LatentPreferenceDataError(
                    "canonical evidence schedules must have equal product capacity."
                )
        return len(EVIDENCE_COUNTS) * (2**6) * first_assignment_count

    def generate_orbit(
        self,
        orbit_index: int,
        *,
        split: str,
    ) -> LatentPreferenceOrbit:
        if (
            isinstance(orbit_index, bool)
            or not isinstance(orbit_index, int)
            or orbit_index < 0
        ):
            raise LatentPreferenceDataError("orbit_index must be non-negative.")
        if split not in SPLITS:
            raise LatentPreferenceDataError(
                f"invalid generator split {split!r}; expected one of {SPLITS}."
            )

        recipe_count = len(self.pool.recipes)
        recipe_offset = self._integer("recipe_offset", modulo=recipe_count)
        recipe = self.pool.recipes[(orbit_index + recipe_offset) % recipe_count]
        local_position = orbit_index // recipe_count
        capacity = self.recipe_semantic_capacity(recipe, split)
        semantic_epoch, local_index = divmod(local_position, capacity)
        coordinate = self._permute_index(
            local_index,
            capacity=capacity,
            recipe_id=recipe.recipe_id,
            split=split,
        )
        evidence_count, selected, remaining = self._decode_coordinate(
            coordinate,
            recipe=recipe,
            split=split,
        )
        if remaining != 0:
            raise AssertionError("mixed-radix decoder left an impossible remainder")

        max_path_cost = sum(
            max(candidate.price_cents for candidate in candidates)
            for candidates in selected
        )
        rounded_max = (
            (max_path_cost + CORE_BUDGET_ROUNDING_CENTS - 1)
            // CORE_BUDGET_ROUNDING_CENTS
            * CORE_BUDGET_ROUNDING_CENTS
        )
        budget_cents = rounded_max + CORE_BUDGET_MARGIN_CENTS
        orbit_digest = self._digest(
            "orbit_id",
            split,
            recipe.recipe_id,
            orbit_index,
            semantic_epoch,
            coordinate,
        ).hex()[:16]
        orbit_id = (
            f"amglp.{split}.{recipe.recipe_id}.{orbit_index}."
            f"e{semantic_epoch}.{orbit_digest}"
        )
        user_digest = self._digest("user_id", orbit_id).hex()[:16]
        user_id = f"shopper.{split}.{user_digest}"
        tasks = tuple(
            self._build_task(
                orbit_id=orbit_id,
                orbit_index=orbit_index,
                semantic_epoch=semantic_epoch,
                user_id=user_id,
                recipe=recipe,
                split=split,
                preferred_attribute_value=preferred_value,
                supporting_evidence_count=evidence_count,
                budget_cents=budget_cents,
                selected=selected,
            )
            for preferred_value in recipe.values
        )
        return LatentPreferenceOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            preferred_attribute_values=recipe.values,
            tasks=tasks,  # type: ignore[arg-type]
        )

    def _decode_coordinate(
        self,
        coordinate: int,
        *,
        recipe: PreferenceRecipe,
        split: str,
    ) -> tuple[
        int,
        tuple[tuple[PreferenceCandidate, PreferenceCandidate], ...],
        int,
    ]:
        coordinate, evidence_coordinate = divmod(coordinate, len(EVIDENCE_COUNTS))
        evidence_count = EVIDENCE_COUNTS[evidence_coordinate]
        coordinate, candidate_order_mask = divmod(coordinate, 2**6)
        schedule = CATEGORY_SCHEDULES[evidence_count]

        selected_by_cell: dict[
            tuple[int, str], tuple[CertifiedPreferenceProduct, ...]
        ] = {}
        for category_position, category_id in enumerate(recipe.categories):
            occurrences = schedule.count(category_position)
            for value in recipe.values:
                products = self._ordered_products(
                    recipe=recipe,
                    category_id=category_id,
                    attribute_value=value,
                    split=split,
                )
                radix = math.perm(len(products), occurrences)
                coordinate, selection_coordinate = divmod(coordinate, radix)
                selected_by_cell[(category_position, value)] = _unrank_permutation(
                    products,
                    selection_coordinate,
                    occurrences,
                )

        seen_per_category = {position: 0 for position in range(4)}
        selected_phases = []
        pool_sha256 = self.pool.semantic_sha256
        for phase_index, category_position in enumerate(schedule):
            occurrence_index = seen_per_category[category_position]
            seen_per_category[category_position] += 1
            pair = tuple(
                PreferenceCandidate.from_product(
                    selected_by_cell[(category_position, value)][occurrence_index],
                    product_pool_sha256=pool_sha256,
                )
                for value in recipe.values
            )
            if (candidate_order_mask >> phase_index) & 1:
                pair = (pair[1], pair[0])
            selected_phases.append(pair)
        return evidence_count, tuple(selected_phases), coordinate  # type: ignore[return-value]

    def _build_task(
        self,
        *,
        orbit_id: str,
        orbit_index: int,
        semantic_epoch: int,
        user_id: str,
        recipe: PreferenceRecipe,
        split: str,
        preferred_attribute_value: str,
        supporting_evidence_count: int,
        budget_cents: int,
        selected: tuple[tuple[PreferenceCandidate, PreferenceCandidate], ...],
    ) -> LatentPreferenceTask:
        schedule = CATEGORY_SCHEDULES[supporting_evidence_count]
        phases = []
        for phase_index, (category_position, candidates) in enumerate(
            zip(schedule, selected)
        ):
            category_id = recipe.categories[category_position]
            target_matches = tuple(
                candidate
                for candidate in candidates
                if candidate.attribute_value == preferred_attribute_value
            )
            if len(target_matches) != 1:
                raise LatentPreferenceDataError(
                    f"generated phase {phase_index} has {len(target_matches)} matches "
                    f"for preference {preferred_attribute_value!r}."
                )
            phase_kind = (
                "evidence"
                if phase_index < supporting_evidence_count
                else "application"
            )
            confirmed_value = (
                preferred_attribute_value if phase_kind == "evidence" else None
            )
            question = render_preference_question(
                user_id=user_id,
                phase_index=phase_index,
                phase_kind=phase_kind,
                supporting_evidence_count=supporting_evidence_count,
                recipe=recipe,
                category_id=category_id,
                candidates=candidates,
                budget_cents=budget_cents,
                confirmed_attribute_value=confirmed_value,
            )
            phases.append(
                PreferencePhase(
                    phase_index=phase_index,
                    phase_kind=phase_kind,
                    category_id=category_id,
                    category_display_name=recipe.category_display_name(category_id),
                    candidates=candidates,
                    question=question,
                    target_asin=target_matches[0].asin,
                    confirmed_attribute_value=confirmed_value,
                )
            )

        task_digest = self._digest(
            "task_id",
            orbit_id,
            preferred_attribute_value,
        ).hex()[:16]
        return LatentPreferenceTask(
            task_id=f"{orbit_id}.t.{task_digest}",
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            split=split,
            preferred_attribute_value=preferred_attribute_value,
            supporting_evidence_count=supporting_evidence_count,
            resolution_step=1,
            budget_cents=budget_cents,
            phases=tuple(phases),
            generator_version=self.version,
            generator_seed=self.seed,
            product_pool_sha256=self.pool.semantic_sha256,
        )

    def _ordered_products(
        self,
        *,
        recipe: PreferenceRecipe,
        category_id: str,
        attribute_value: str,
        split: str,
    ):
        products = self.pool.products_for(
            axis=recipe.axis,
            category_id=category_id,
            attribute_value=attribute_value,
            split=split,
        )
        return tuple(
            sorted(
                products,
                key=lambda product: (
                    self._digest(
                        "product_order",
                        split,
                        recipe.axis,
                        category_id,
                        attribute_value,
                        product.asin,
                    ),
                    product.asin,
                ),
            )
        )

    def _permute_index(
        self,
        index: int,
        *,
        capacity: int,
        recipe_id: str,
        split: str,
    ) -> int:
        multiplier = self._integer(
            "affine_multiplier",
            recipe_id,
            split,
            modulo=capacity,
        )
        if multiplier == 0:
            multiplier = 1
        while math.gcd(multiplier, capacity) != 1:
            multiplier = (multiplier + 1) % capacity
            if multiplier == 0:
                multiplier = 1
        offset = self._integer(
            "affine_offset",
            recipe_id,
            split,
            modulo=capacity,
        )
        return (multiplier * index + offset) % capacity

    def _integer(self, *parts: object, modulo: int) -> int:
        if modulo <= 0:
            raise LatentPreferenceDataError("choice modulo must be positive.")
        return int.from_bytes(self._digest(*parts), "big") % modulo

    def _digest(self, *parts: object) -> bytes:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "generator_version": self.version,
                    "generator_seed": self.seed,
                    "product_pool_sha256": self.pool.semantic_sha256,
                    "parts": list(parts),
                }
            )
        ).digest()


def _unrank_permutation(items: tuple[object, ...], rank: int, count: int):
    if count < 0 or count > len(items):
        raise LatentPreferenceDataError("invalid permutation sample size.")
    capacity = math.perm(len(items), count)
    if rank < 0 or rank >= capacity:
        raise LatentPreferenceDataError("permutation rank is outside its domain.")
    available = list(items)
    chosen = []
    remainder = rank
    for position in range(count):
        suffix_count = count - position - 1
        block = math.perm(len(available) - 1, suffix_count)
        choice_index, remainder = divmod(remainder, block)
        chosen.append(available.pop(choice_index))
    return tuple(chosen)
