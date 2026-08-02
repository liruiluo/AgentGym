from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from ..latent_preference.schema import PreferenceProductPool, PreferenceRecipe, canonical_json_bytes
from .question_format import render_recency_question
from .schema import (
    RecencyOverrideDataError,
    RecencyOverrideOrbit,
    RecencyOverrideTask,
    RecencyPhase,
    candidate_from_product,
)


DEFAULT_GENERATOR_VERSION = "recency_override_native_v1"
PHASE_CATEGORY_POSITIONS = (0, 1, 2, 3, 0, 1)
CANONICAL_MEMORY_KEY = "user_preference"
CORE_BUDGET_MARGIN_CENTS = 5_000
CORE_BUDGET_ROUNDING_CENTS = 1_000


@dataclass(frozen=True)
class RecencyOverrideGenerator:
    pool: PreferenceProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RecencyOverrideDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise RecencyOverrideDataError("generator version must be non-empty.")
        capacities = {
            self.recipe_semantic_capacity(recipe, split)
            for recipe in self.pool.recipes
            for split in ("train", "dev", "test")
        }
        if len(capacities) != 1:
            raise RecencyOverrideDataError("all recipe/split cells must have one semantic capacity.")

    @property
    def semantic_period_orbits(self) -> int:
        return len(self.pool.recipes) * self.recipe_semantic_capacity(self.pool.recipes[0], "train")

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * 2

    def recipe_semantic_capacity(self, recipe: PreferenceRecipe, split: str) -> int:
        if split not in ("train", "dev", "test"):
            raise RecencyOverrideDataError(f"invalid split {split!r}.")
        capacity = 2 * (2**6)
        for category_position in range(4):
            occurrence_count = PHASE_CATEGORY_POSITIONS.count(category_position)
            for value in recipe.values:
                cell = self.pool.products_for(
                    axis=recipe.axis,
                    category_id=recipe.categories[category_position],
                    attribute_value=value,
                    split=split,
                )
                capacity *= math.perm(len(cell), occurrence_count)
        return capacity

    def generate_orbit(self, orbit_index: int, *, split: str) -> RecencyOverrideOrbit:
        if isinstance(orbit_index, bool) or not isinstance(orbit_index, int) or orbit_index < 0:
            raise RecencyOverrideDataError("orbit_index must be non-negative.")
        if split not in ("train", "dev", "test"):
            raise RecencyOverrideDataError(f"invalid split {split!r}.")
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
        orientation, selected = self._decode_coordinate(coordinate, recipe=recipe, split=split)
        old_value = recipe.values[orientation]
        new_value = recipe.values[1 - orientation]
        max_path_cost = sum(max(item.price_cents for item in pair) for pair in selected)
        rounded_max = ((max_path_cost + CORE_BUDGET_ROUNDING_CENTS - 1) // CORE_BUDGET_ROUNDING_CENTS) * CORE_BUDGET_ROUNDING_CENTS
        budget_cents = rounded_max + CORE_BUDGET_MARGIN_CENTS
        orbit_digest = self._digest("orbit_id", split, recipe.recipe_id, orbit_index, semantic_epoch, coordinate).hex()[:16]
        orbit_id = f"amgro.{split}.{recipe.recipe_id}.{orbit_index}.e{semantic_epoch}.{orbit_digest}"
        user_id = f"shopper.{split}.{self._digest('user_id', orbit_id).hex()[:16]}"
        tasks = tuple(
            self._build_task(
                orbit_id=orbit_id,
                orbit_index=orbit_index,
                semantic_epoch=semantic_epoch,
                user_id=user_id,
                recipe=recipe,
                split=split,
                branch_kind=branch,
                old_value=old_value,
                new_value=new_value,
                budget_cents=budget_cents,
                selected=selected,
            )
            for branch in ("stay", "flip")
        )
        return RecencyOverrideOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            old_attribute_value=old_value,
            new_attribute_value=new_value,
            tasks=tasks,  # type: ignore[arg-type]
        )

    def _decode_coordinate(self, coordinate: int, *, recipe: PreferenceRecipe, split: str):
        coordinate, orientation = divmod(coordinate, 2)
        coordinate, order_mask = divmod(coordinate, 2**6)
        selected_by_cell: dict[tuple[int, str], tuple[object, ...]] = {}
        for category_position in range(4):
            occurrence_count = PHASE_CATEGORY_POSITIONS.count(category_position)
            for value in recipe.values:
                products = self._ordered_products(recipe=recipe, category_position=category_position, value=value, split=split)
                radix = math.perm(len(products), occurrence_count)
                coordinate, rank = divmod(coordinate, radix)
                selected_by_cell[(category_position, value)] = _unrank_permutation(products, rank, occurrence_count)
        if coordinate != 0:
            raise AssertionError("recency coordinate decoder left a remainder")
        seen_by_category: dict[int, int] = {position: 0 for position in range(4)}
        selected: list[tuple[object, object]] = []
        ordered_values = (recipe.values[orientation], recipe.values[1 - orientation])
        for phase_index, category_position in enumerate(PHASE_CATEGORY_POSITIONS):
            occurrence = seen_by_category[category_position]
            seen_by_category[category_position] += 1
            pair = tuple(
                selected_by_cell[(category_position, value)][occurrence]
                for value in ordered_values
            )
            if (order_mask >> phase_index) & 1:
                pair = (pair[1], pair[0])
            selected.append(pair)  # type: ignore[arg-type]
        return orientation, tuple(selected)

    def _build_task(self, *, orbit_id, orbit_index, semantic_epoch, user_id, recipe, split, branch_kind, old_value, new_value, budget_cents, selected):
        phases = []
        for phase_index, (category_position, candidates) in enumerate(zip(PHASE_CATEGORY_POSITIONS, selected)):
            active_value = old_value if (branch_kind == "stay" or phase_index < 2) else new_value
            if phase_index == 2 and branch_kind == "flip":
                active_value = new_value
            if phase_index == 2 and branch_kind == "stay":
                active_value = old_value
            phase_kind = "evidence" if phase_index == 0 else ("override" if phase_index == 2 else "application")
            confirmed = active_value if phase_kind in {"evidence", "override"} else None
            target = tuple(item for item in candidates if item.attribute_value == active_value)
            if len(target) != 1:
                raise RecencyOverrideDataError("each phase must contain one active-value candidate.")
            category_id = recipe.categories[category_position]
            question = render_recency_question(
                user_id=user_id,
                phase_index=phase_index,
                phase_kind=phase_kind,
                recipe=recipe,
                category_id=category_id,
                candidates=tuple(candidate_from_product(item, product_pool_sha256=self.pool.semantic_sha256) for item in candidates),
                budget_cents=budget_cents,
                old_attribute_value=old_value,
                new_attribute_value=new_value,
                active_attribute_value=active_value,
                confirmed_attribute_value=confirmed,
            )
            phases.append(
                RecencyPhase(
                    phase_index=phase_index,
                    phase_kind=phase_kind,
                    category_id=category_id,
                    category_display_name=recipe.category_display_name(category_id),
                    candidates=tuple(candidate_from_product(item, product_pool_sha256=self.pool.semantic_sha256) for item in candidates),
                    question=question,
                    target_asin=target[0].asin,
                    confirmed_attribute_value=confirmed,
                    active_attribute_value=active_value,
                )
            )
        task_digest = self._digest("task_id", orbit_id, branch_kind).hex()[:16]
        return RecencyOverrideTask(
            task_id=f"{orbit_id}.t.{branch_kind}.{task_digest}",
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            recipe_id=recipe.recipe_id,
            user_id=user_id,
            split=split,
            branch_kind=branch_kind,
            old_attribute_value=old_value,
            new_attribute_value=new_value,
            canonical_memory_key=CANONICAL_MEMORY_KEY,
            override_mode="none" if branch_kind == "stay" else "update_or_delete_add",
            budget_cents=budget_cents,
            phases=tuple(phases),
            generator_version=self.version,
            generator_seed=self.seed,
            product_pool_sha256=self.pool.semantic_sha256,
        )

    def _ordered_products(self, *, recipe, category_position, value, split):
        products = self.pool.products_for(
            axis=recipe.axis,
            category_id=recipe.categories[category_position],
            attribute_value=value,
            split=split,
        )
        return tuple(sorted(products, key=lambda item: (self._digest("product_order", split, recipe.axis, recipe.categories[category_position], value, item.asin), item.asin)))

    def _permute_index(self, index: int, *, capacity: int, recipe_id: str, split: str) -> int:
        multiplier = self._integer("affine_multiplier", recipe_id, split, modulo=capacity) or 1
        while math.gcd(multiplier, capacity) != 1:
            multiplier = (multiplier + 1) % capacity or 1
        offset = self._integer("affine_offset", recipe_id, split, modulo=capacity)
        return (multiplier * index + offset) % capacity

    def _integer(self, *parts: object, modulo: int) -> int:
        return int.from_bytes(self._digest(*parts), "big") % modulo

    def _digest(self, *parts: object) -> bytes:
        return hashlib.sha256(canonical_json_bytes({"version": self.version, "seed": self.seed, "pool": self.pool.semantic_sha256, "parts": list(parts)})).digest()


def _unrank_permutation(items: tuple[object, ...], rank: int, count: int):
    capacity = math.perm(len(items), count)
    if count < 0 or count > len(items) or rank < 0 or rank >= capacity:
        raise RecencyOverrideDataError("invalid permutation rank.")
    available = list(items)
    result = []
    remainder = rank
    for position in range(count):
        block = math.perm(len(available) - 1, count - position - 1)
        choice, remainder = divmod(remainder, block)
        result.append(available.pop(choice))
    return tuple(result)
