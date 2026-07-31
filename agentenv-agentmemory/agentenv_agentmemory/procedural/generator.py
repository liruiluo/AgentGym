from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .question_format import render_question
from .schema import (
    SPLITS,
    AttributeTransition,
    CounterfactualOrbit,
    ProceduralCandidate,
    ProceduralMemoryDataError,
    ProceduralPhase,
    ProceduralTask,
    ProductPool,
    canonical_json_bytes,
)
from .scenarios import ScenarioSpec


DEFAULT_GENERATOR_VERSION = "natural_attribute_approved_shortlist_chain_v5"
CORE_BUDGET_MARGIN_CENTS = 5_000
CORE_BUDGET_ROUNDING_CENTS = 1_000


@dataclass(frozen=True)
class NaturalAttributeChainGenerator:
    """Enumerate a deterministic, collision-free practical task stream.

    Every scenario-local semantic coordinate covers six real product pairs,
    five binary customer-plan bijections, and six candidate order bits. The
    mixed-radix coordinate is permuted by the seed, but no semantic orbit is
    repeated before the full period is exhausted.
    """

    pool: ProductPool
    seed: int
    version: str = DEFAULT_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ProceduralMemoryDataError("generator seed must be an integer.")
        if not isinstance(self.version, str) or not self.version:
            raise ProceduralMemoryDataError("generator version must be non-empty.")
        capacities = {
            self.scenario_semantic_capacity(scenario, split)
            for scenario in self.pool.scenarios
            for split in SPLITS
        }
        if len(capacities) != 1:
            raise ProceduralMemoryDataError(
                "balanced product pools must give every scenario and split one "
                "common semantic period."
            )

    @property
    def semantic_period_orbits(self) -> int:
        scenario = self.pool.scenarios[0]
        return len(self.pool.scenarios) * self.scenario_semantic_capacity(
            scenario, "train"
        )

    @property
    def semantic_period_tasks(self) -> int:
        return self.semantic_period_orbits * 2

    @property
    def conservative_task_capacity_without_candidate_order(self) -> int:
        pair_choices = self.pool.products_per_cell**2
        return len(self.pool.scenarios) * (pair_choices**6) * (2**5) * 2

    def scenario_semantic_capacity(self, scenario: ScenarioSpec, split: str) -> int:
        if split not in SPLITS:
            raise ProceduralMemoryDataError(f"invalid split {split!r}.")
        capacity = (2**5) * (2**6)
        for slot in scenario.slots:
            for value in slot.values:
                capacity *= len(
                    self.pool.products_for_split(
                        scenario.scenario_id,
                        slot.slot_id,
                        value.value_id,
                        split,
                    )
                )
        return capacity

    def generate_orbit(self, orbit_index: int, *, split: str) -> CounterfactualOrbit:
        if (
            isinstance(orbit_index, bool)
            or not isinstance(orbit_index, int)
            or orbit_index < 0
        ):
            raise ProceduralMemoryDataError("orbit_index must be non-negative.")
        if split not in SPLITS:
            raise ProceduralMemoryDataError(
                f"invalid generator split {split!r}; expected one of {SPLITS}."
            )

        scenario_count = len(self.pool.scenarios)
        scenario_offset = self._integer("scenario_offset", modulo=scenario_count)
        scenario = self.pool.scenarios[(orbit_index + scenario_offset) % scenario_count]
        local_position = orbit_index // scenario_count
        capacity = self.scenario_semantic_capacity(scenario, split)
        semantic_epoch, local_index = divmod(local_position, capacity)
        coordinate = self._permute_index(
            local_index,
            capacity=capacity,
            scenario_id=scenario.scenario_id,
            split=split,
        )
        selected, transitions, remaining = self._decode_coordinate(
            coordinate,
            scenario=scenario,
            split=split,
        )
        if remaining != 0:
            raise AssertionError("mixed-radix decoder left an impossible remainder")

        max_path_cost_cents = sum(
            max(candidate.price_cents for candidate in candidates)
            for candidates in selected
        )
        rounded_max = (
            (max_path_cost_cents + CORE_BUDGET_ROUNDING_CENTS - 1)
            // CORE_BUDGET_ROUNDING_CENTS
            * CORE_BUDGET_ROUNDING_CENTS
        )
        budget_cents = rounded_max + CORE_BUDGET_MARGIN_CENTS
        orbit_digest = self._digest(
            "orbit_id",
            split,
            scenario.scenario_id,
            orbit_index,
            semantic_epoch,
            coordinate,
        ).hex()[:16]
        orbit_id = (
            f"amgpm.{split}.{scenario.scenario_id}.{orbit_index}."
            f"e{semantic_epoch}.{orbit_digest}"
        )
        roots = scenario.slots[0].value_ids
        tasks = tuple(
            self._build_task(
                orbit_id=orbit_id,
                orbit_index=orbit_index,
                semantic_epoch=semantic_epoch,
                scenario=scenario,
                split=split,
                root_attribute_value=root,
                budget_cents=budget_cents,
                selected=selected,
                transitions=transitions,
            )
            for root in roots
        )
        return CounterfactualOrbit(
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            scenario_id=scenario.scenario_id,
            root_attribute_values=roots,
            tasks=tasks,
        )

    def _decode_coordinate(
        self,
        coordinate: int,
        *,
        scenario: ScenarioSpec,
        split: str,
    ) -> tuple[
        tuple[tuple[ProceduralCandidate, ProceduralCandidate], ...],
        tuple[AttributeTransition, ...],
        int,
    ]:
        coordinate, transition_mask = divmod(coordinate, 2**5)
        coordinate, candidate_order_mask = divmod(coordinate, 2**6)
        selected: list[tuple[ProceduralCandidate, ProceduralCandidate]] = []
        for phase_index, slot in enumerate(scenario.slots):
            value_products = [
                self._ordered_products(
                    scenario_id=scenario.scenario_id,
                    slot_id=slot.slot_id,
                    attribute_value=value.value_id,
                    split=split,
                )
                for value in slot.values
            ]
            pair_radix = len(value_products[0]) * len(value_products[1])
            coordinate, pair_coordinate = divmod(coordinate, pair_radix)
            first_index = pair_coordinate % len(value_products[0])
            second_index = pair_coordinate // len(value_products[0])
            pair = (
                ProceduralCandidate(value_products[0][first_index]),
                ProceduralCandidate(value_products[1][second_index]),
            )
            if (candidate_order_mask >> phase_index) & 1:
                pair = (pair[1], pair[0])
            selected.append(pair)

        transitions = []
        for phase_index in range(1, 6):
            previous_slot = scenario.slots[phase_index - 1]
            current_slot = scenario.slots[phase_index]
            current_values = current_slot.value_ids
            if (transition_mask >> (phase_index - 1)) & 1:
                current_values = (current_values[1], current_values[0])
            transitions.append(
                AttributeTransition(
                    previous_slot_id=previous_slot.slot_id,
                    previous_attribute_name=previous_slot.attribute_name,
                    current_attribute_name=current_slot.attribute_name,
                    pairs=tuple(zip(previous_slot.value_ids, current_values)),  # type: ignore[arg-type]
                )
            )
        return tuple(selected), tuple(transitions), coordinate

    def _build_task(
        self,
        *,
        orbit_id: str,
        orbit_index: int,
        semantic_epoch: int,
        scenario: ScenarioSpec,
        split: str,
        root_attribute_value: str,
        budget_cents: int,
        selected: tuple[tuple[ProceduralCandidate, ProceduralCandidate], ...],
        transitions: tuple[AttributeTransition, ...],
    ) -> ProceduralTask:
        expected_value = root_attribute_value
        phases = []
        for phase_index, (slot, candidates) in enumerate(zip(scenario.slots, selected)):
            target_matches = tuple(
                candidate
                for candidate in candidates
                if candidate.attribute_value == expected_value
            )
            if len(target_matches) != 1:
                raise ProceduralMemoryDataError(
                    f"generated phase {phase_index} has {len(target_matches)} candidates "
                    f"for attribute value {expected_value!r}; expected one."
                )
            transition = None if phase_index == 0 else transitions[phase_index - 1]
            question = render_question(
                scenario_id=scenario.scenario_id,
                phase_index=phase_index,
                slot_id=slot.slot_id,
                candidate_rows=tuple(
                    (
                        candidate.title,
                        candidate.attribute_value,
                        candidate.attribute_display_name,
                    )
                    for candidate in candidates
                ),
                budget_cents=budget_cents,
                root_attribute_value=(
                    root_attribute_value if phase_index == 0 else None
                ),
                transition=transition,
            )
            phases.append(
                ProceduralPhase(
                    phase_index=phase_index,
                    scenario_id=scenario.scenario_id,
                    slot_id=slot.slot_id,
                    display_name=slot.display_name,
                    attribute_name=slot.attribute_name,
                    candidates=candidates,
                    question=question,
                    target_asin=target_matches[0].asin,
                    root_attribute_value=(
                        root_attribute_value if phase_index == 0 else None
                    ),
                    transition=transition,
                )
            )
            if phase_index < 5:
                expected_value = transitions[phase_index].resolve(expected_value)

        task_digest = self._digest(
            "task_id",
            orbit_id,
            root_attribute_value,
        ).hex()[:16]
        return ProceduralTask(
            task_id=f"{orbit_id}.t.{task_digest}",
            orbit_id=orbit_id,
            orbit_index=orbit_index,
            semantic_epoch=semantic_epoch,
            scenario_id=scenario.scenario_id,
            root_attribute_value=root_attribute_value,
            split=split,
            budget_cents=budget_cents,
            phases=tuple(phases),
            generator_version=self.version,
            generator_seed=self.seed,
            product_pool_sha256=self.pool.semantic_sha256,
        )

    def _ordered_products(
        self,
        *,
        scenario_id: str,
        slot_id: str,
        attribute_value: str,
        split: str,
    ):
        products = self.pool.products_for_split(
            scenario_id,
            slot_id,
            attribute_value,
            split,
        )
        return tuple(
            sorted(
                products,
                key=lambda product: (
                    self._digest(
                        "product_order",
                        split,
                        scenario_id,
                        slot_id,
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
        scenario_id: str,
        split: str,
    ) -> int:
        multiplier = self._integer(
            "affine_multiplier",
            scenario_id,
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
            scenario_id,
            split,
            modulo=capacity,
        )
        return (multiplier * index + offset) % capacity

    def _integer(self, *parts: object, modulo: int) -> int:
        if modulo <= 0:
            raise ProceduralMemoryDataError("choice modulo must be positive.")
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
