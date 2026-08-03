from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from ..latent_preference.schema import canonical_json_bytes
from .question_format import render_negative_constraint_question
from .schema import (
    NegativeConstraintDataError,
    NegativeConstraintOrbit,
    NegativeConstraintPhase,
    NegativeConstraintProductPool,
    NegativeConstraintRecipe,
    NegativeConstraintTask,
)


DEFAULT_GENERATOR_VERSION = "negative_constraint_three_way_v1"
CANONICAL_MEMORY_KEY = "standing_constraints"
BUDGET_CENTS = 10_000_000


def phase_category_positions(recipe: NegativeConstraintRecipe) -> tuple[int, ...]:
    return tuple(index % len(recipe.categories) for index in range(6))


@dataclass(frozen=True)
class NegativeConstraintGenerator:
    pool: NegativeConstraintProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise NegativeConstraintDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise NegativeConstraintDataError("generator version must be non-empty.")
        capacities = tuple(
            self.recipe_semantic_capacity(recipe, split)
            for recipe in self.pool.recipes
            for split in ("train", "dev", "test")
        )
        if any(capacity <= 0 for capacity in capacities):
            raise NegativeConstraintDataError(
                "negative recipe/split capacities must be positive."
            )

    @property
    def semantic_period_orbits(self) -> int:
        return len(self.pool.recipes) * min(
            self.recipe_semantic_capacity(recipe, split)
            for recipe in self.pool.recipes
            for split in ("train", "dev", "test")
        )

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * 3

    def recipe_semantic_capacity(
        self,
        recipe: NegativeConstraintRecipe,
        split: str,
    ) -> int:
        if split not in ("train", "dev", "test"):
            raise NegativeConstraintDataError(f"invalid split {split!r}.")
        capacity = math.factorial(3) ** 6
        schedule = phase_category_positions(recipe)
        for category_position in range(4):
            if category_position >= len(recipe.categories):
                continue
            occurrence_count = schedule.count(category_position)
            for value in recipe.values:
                products = self.pool.candidates_for(
                    axis=recipe.axis,
                    category_id=recipe.categories[category_position],
                    attribute_value=value,
                    split=split,
                )
                capacity *= math.perm(len(products), occurrence_count)
        return capacity

    def generate_orbit(
        self,
        orbit_index: int,
        *,
        split: str,
    ) -> NegativeConstraintOrbit:
        if (
            isinstance(orbit_index, bool)
            or not isinstance(orbit_index, int)
            or orbit_index < 0
        ):
            raise NegativeConstraintDataError(
                "orbit_index must be a non-negative integer."
            )
        if split not in ("train", "dev", "test"):
            raise NegativeConstraintDataError(f"invalid split {split!r}.")
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
        selected = self._decode_coordinate(
            coordinate,
            recipe=recipe,
            split=split,
        )
        digest = self._digest(
            "orbit_id",
            split,
            recipe.recipe_id,
            orbit_index,
            semantic_epoch,
            coordinate,
        ).hex()[:16]
        orbit_id = (
            f"amgnc.{split}.{recipe.recipe_id}.{orbit_index}."
            f"e{semantic_epoch}.{digest}"
        )
        user_id = f"shopper.{split}.{self._digest('user_id', orbit_id).hex()[:16]}"
        tasks = tuple(
            self._build_task(
                orbit_id=orbit_id,
                orbit_index=orbit_index,
                semantic_epoch=semantic_epoch,
                user_id=user_id,
                recipe=recipe,
                split=split,
                allowed_value=allowed_value,
                selected=selected,
            )
            for allowed_value in recipe.values
        )
        return NegativeConstraintOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            tasks=tasks,  # type: ignore[arg-type]
        )

    def _decode_coordinate(
        self,
        coordinate: int,
        *,
        recipe: NegativeConstraintRecipe,
        split: str,
    ):
        coordinate, order_code = divmod(coordinate, math.factorial(3) ** 6)
        selected_by_cell: dict[tuple[int, str], tuple[object, ...]] = {}
        schedule = phase_category_positions(recipe)
        for category_position in range(len(recipe.categories)):
            occurrence_count = schedule.count(category_position)
            for value in recipe.values:
                products = self._ordered_products(
                    recipe=recipe,
                    category_position=category_position,
                    value=value,
                    split=split,
                )
                radix = math.perm(len(products), occurrence_count)
                coordinate, rank = divmod(coordinate, radix)
                selected_by_cell[(category_position, value)] = _unrank_permutation(
                    products,
                    rank,
                    occurrence_count,
                )
        if coordinate != 0:
            raise AssertionError("negative coordinate decoder left a remainder")
        seen_by_category = {position: 0 for position in range(len(recipe.categories))}
        phases = []
        for category_position in schedule:
            occurrence = seen_by_category[category_position]
            seen_by_category[category_position] += 1
            candidates = tuple(
                selected_by_cell[(category_position, value)][occurrence]
                for value in recipe.values
            )
            order_code, order_rank = divmod(order_code, math.factorial(3))
            phases.append(_unrank_permutation(candidates, order_rank, 3))
        if order_code != 0:
            raise AssertionError("negative candidate-order decoder left a remainder")
        return tuple(phases)

    def _build_task(
        self,
        *,
        orbit_id,
        orbit_index,
        semantic_epoch,
        user_id,
        recipe,
        split,
        allowed_value,
        selected,
    ) -> NegativeConstraintTask:
        forbidden_values = tuple(
            value for value in recipe.values if value != allowed_value
        )
        memory_value = (
            f"Customer {user_id} has standing never-accept constraints for "
            f"{recipe.axis_display_name}: "
            f"{recipe.value_display_name(forbidden_values[0])} and "
            f"{recipe.value_display_name(forbidden_values[1])}."
        )
        retrieval_query = f"customer {user_id} standing never accept constraints"
        phases = []
        for phase_index, (category_position, candidates) in enumerate(
            zip(phase_category_positions(recipe), selected)
        ):
            target = tuple(
                item
                for item in candidates
                if item.attribute_value == allowed_value
            )
            if len(target) != 1:
                raise NegativeConstraintDataError(
                    "each negative phase must have one allowed candidate."
                )
            category_id = recipe.categories[category_position]
            question = render_negative_constraint_question(
                user_id=user_id,
                phase_index=phase_index,
                recipe=recipe,
                category_id=category_id,
                candidates=candidates,
                budget_cents=BUDGET_CENTS,
                allowed_attribute_value=allowed_value,
                forbidden_attribute_values=forbidden_values,  # type: ignore[arg-type]
            )
            phases.append(
                NegativeConstraintPhase(
                    phase_index=phase_index,
                    phase_kind=(
                        "constraint_evidence" if phase_index == 0 else "application"
                    ),
                    category_id=category_id,
                    category_display_name=recipe.category_display_name(category_id),
                    candidates=candidates,
                    question=question,
                    target_asin=target[0].asin,
                    allowed_attribute_value=allowed_value,
                )
            )
        task_digest = self._digest(
            "task_id",
            orbit_id,
            allowed_value,
        ).hex()[:16]
        return NegativeConstraintTask(
            task_id=f"{orbit_id}.t.allow_{allowed_value}.{task_digest}",
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            split=split,
            branch_kind=f"allow_{allowed_value}",
            allowed_attribute_value=allowed_value,
            forbidden_attribute_values=forbidden_values,  # type: ignore[arg-type]
            canonical_memory_key=CANONICAL_MEMORY_KEY,
            canonical_memory_value=memory_value,
            canonical_retrieval_query=retrieval_query,
            budget_cents=BUDGET_CENTS,
            phases=tuple(phases),
            generator_version=self.version,
            generator_seed=self.seed,
            product_pool_sha256=self.pool.semantic_sha256,
        )

    def _ordered_products(
        self,
        *,
        recipe,
        category_position,
        value,
        split,
    ):
        products = self.pool.candidates_for(
            axis=recipe.axis,
            category_id=recipe.categories[category_position],
            attribute_value=value,
            split=split,
        )
        return tuple(
            sorted(
                products,
                key=lambda item: (
                    self._digest(
                        "product_order",
                        split,
                        recipe.axis,
                        recipe.categories[category_position],
                        value,
                        item.asin,
                    ),
                    item.asin,
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
        ) or 1
        while math.gcd(multiplier, capacity) != 1:
            multiplier = (multiplier + 1) % capacity or 1
        offset = self._integer(
            "affine_offset",
            recipe_id,
            split,
            modulo=capacity,
        )
        return (multiplier * index + offset) % capacity

    def _integer(self, *parts: object, modulo: int) -> int:
        return int.from_bytes(self._digest(*parts), "big") % modulo

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


def _unrank_permutation(items: tuple[object, ...], rank: int, count: int):
    capacity = math.perm(len(items), count)
    if count < 0 or count > len(items) or rank < 0 or rank >= capacity:
        raise NegativeConstraintDataError("invalid negative permutation rank.")
    available = list(items)
    result = []
    remainder = rank
    for position in range(count):
        block = math.perm(len(available) - 1, count - position - 1)
        choice, remainder = divmod(remainder, block)
        result.append(available.pop(choice))
    return tuple(result)
