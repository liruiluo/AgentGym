from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from .generator import CORE_BUDGET_MARGIN_CENTS, CORE_BUDGET_ROUNDING_CENTS
from .question_format import render_question, transition_rows
from .schema import (
    PROOF_SCHEMA,
    CounterfactualOrbit,
    ProceduralMemoryDataError,
    ProceduralTask,
    ProductPool,
    canonical_sha256,
)
from .scenarios import scenario_by_id


VERIFIER_VERSION = "exhaustive_approved_shortlist_chain_v4"


@dataclass(frozen=True)
class OrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    scenario_id: str
    split: str
    verifier_version: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str]
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int]
    valid_solution_vectors: tuple[tuple[int, ...], tuple[int, ...]]
    counterfactual_target_flip_count: int
    complete_bijection_count: int
    later_observation_identity_checks: int
    future_mapping_exclusion_checks: int
    certified_candidate_checks: int
    budget_cents: int
    min_path_cost_cents: int
    max_path_cost_cents: int

    def payload(self) -> dict[str, Any]:
        return {
            "schema": PROOF_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "scenario_id": self.scenario_id,
            "split": self.split,
            "verifier_version": self.verifier_version,
            "generator": {
                "version": self.generator_version,
                "seed": self.generator_seed,
                "product_pool_sha256": self.product_pool_sha256,
            },
            "task_semantic_sha256": list(self.task_semantic_sha256),
            "enumeration": {
                "paths_per_branch": self.enumerated_path_count_per_branch,
                "valid_solution_counts": list(self.valid_solution_counts),
                "valid_solution_vectors": [
                    list(vector) for vector in self.valid_solution_vectors
                ],
            },
            "checks": {
                "counterfactual_target_flip_count": self.counterfactual_target_flip_count,
                "complete_bijection_count": self.complete_bijection_count,
                "later_observation_identity_checks": (
                    self.later_observation_identity_checks
                ),
                "future_mapping_exclusion_checks": (
                    self.future_mapping_exclusion_checks
                ),
                "certified_candidate_checks": self.certified_candidate_checks,
            },
            "budget": {
                "budget_cents": self.budget_cents,
                "min_path_cost_cents": self.min_path_cost_cents,
                "max_path_cost_cents": self.max_path_cost_cents,
                "paths_pruned": 0,
            },
            "verification": {
                "phase_count_per_task": 6,
                "candidate_count_per_phase": 2,
                "answer_domain": (
                    "current_phase_approved_titles_resolved_by_hidden_asin_receipt"
                ),
                "approved_candidate_titles_in_task_prompt": True,
                "approved_candidate_asins_in_task_prompt": False,
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "native_purchase_receipt_asin_verification": True,
                "out_of_shortlist_purchase_is_legal": False,
                "global_catalog_attribute_uniqueness_required": False,
                "global_catalog_attribute_uniqueness_claimed": False,
                "global_catalog_normalized_title_uniqueness_required": True,
                "global_catalog_normalized_title_uniqueness_claimed": True,
                "natural_attribute_values_per_phase": 2,
                "previous_purchase_attribute_required_after_phase_one": True,
                "complete_order_specific_bijections": True,
                "unique_legal_purchase_vector_per_branch": True,
                "all_64_purchase_vectors_within_budget": True,
                "paired_root_changes_only_first_observation": True,
                "paired_targets_flip_all_six_phases": True,
                "later_observations_byte_identical": True,
                "future_mapping_absent_from_current_observation": True,
                "category_and_attribute_native_certified": True,
                "asin_split_isolated": True,
                "wrong_buy_feedback_contract": "terminal_without_answer",
                "human_review_required": False,
                "llm_judge_required": False,
                "paper_eligible": False,
            },
        }

    @property
    def proof_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["proof_sha256"] = self.proof_sha256
        return payload


def verify_counterfactual_orbit(
    orbit: CounterfactualOrbit,
    *,
    pool: ProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> OrbitProof:
    """Independently enumerate all 64 purchase vectors for both root branches."""

    if orbit.scenario_id not in pool.scenario_ids:
        raise ProceduralMemoryDataError("orbit scenario is outside the certified pool.")
    left, right = orbit.tasks
    split = left.split
    if right.split != split:
        raise ProceduralMemoryDataError("counterfactual tasks must share one split.")
    pool_sha256 = pool.semantic_sha256
    for task in orbit.tasks:
        if task.product_pool_sha256 != pool_sha256:
            raise ProceduralMemoryDataError("task product-pool hash mismatch.")
        if task.generator_version != left.generator_version:
            raise ProceduralMemoryDataError("orbit generator versions disagree.")
        if task.generator_seed != left.generator_seed:
            raise ProceduralMemoryDataError("orbit generator seeds disagree.")
    if (
        expected_generator_version is not None
        and left.generator_version != expected_generator_version
    ):
        raise ProceduralMemoryDataError("unexpected generator version.")
    if expected_generator_seed is not None and left.generator_seed != expected_generator_seed:
        raise ProceduralMemoryDataError("unexpected generator seed.")

    certified_checks = sum(_verify_task_structure(task, pool=pool) for task in orbit.tasks)
    identity_checks = _verify_pair_symmetry(orbit)
    bijection_count = _verify_transition_bijections(left)
    future_checks = sum(_verify_question_locality(task) for task in orbit.tasks)

    all_costs: list[int] = []
    valid_vectors: list[tuple[int, ...]] = []
    for task in orbit.tasks:
        task_valid = []
        for vector in itertools.product((0, 1), repeat=6):
            chosen = tuple(
                phase.candidates[choice]
                for phase, choice in zip(task.phases, vector)
            )
            cost = sum(candidate.price_cents for candidate in chosen)
            all_costs.append(cost)
            if cost > task.budget_cents:
                raise ProceduralMemoryDataError(
                    "budget prunes a purchase vector and could become an answer shortcut."
                )
            if _is_structurally_legal(task, chosen):
                task_valid.append(tuple(vector))
        if len(task_valid) != 1:
            raise ProceduralMemoryDataError(
                "independent exhaustive verifier expected exactly one legal purchase "
                f"vector, observed {len(task_valid)} for task {task.task_id}."
            )
        declared_vector = tuple(
            next(
                index
                for index, candidate in enumerate(phase.candidates)
                if candidate.asin == phase.target_asin
            )
            for phase in task.phases
        )
        if declared_vector != task_valid[0]:
            raise ProceduralMemoryDataError(
                "declared targets disagree with the independently enumerated solution."
            )
        valid_vectors.append(task_valid[0])

    if any(left_choice == right_choice for left_choice, right_choice in zip(*valid_vectors)):
        raise ProceduralMemoryDataError(
            "counterfactual roots must flip the correct candidate in all six phases."
        )
    max_path_cost = max(all_costs)
    min_path_cost = min(all_costs)
    expected_budget = (
        (max_path_cost + CORE_BUDGET_ROUNDING_CENTS - 1)
        // CORE_BUDGET_ROUNDING_CENTS
        * CORE_BUDGET_ROUNDING_CENTS
        + CORE_BUDGET_MARGIN_CENTS
    )
    if left.budget_cents != right.budget_cents or left.budget_cents != expected_budget:
        raise ProceduralMemoryDataError(
            "task budget must use the canonical rounded maximum plus fixed margin."
        )

    return OrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        scenario_id=orbit.scenario_id,
        split=split,
        verifier_version=VERIFIER_VERSION,
        generator_version=left.generator_version,
        generator_seed=left.generator_seed,
        product_pool_sha256=pool_sha256,
        task_semantic_sha256=(left.semantic_sha256, right.semantic_sha256),
        enumerated_path_count_per_branch=64,
        valid_solution_counts=(1, 1),
        valid_solution_vectors=(valid_vectors[0], valid_vectors[1]),
        counterfactual_target_flip_count=6,
        complete_bijection_count=bijection_count,
        later_observation_identity_checks=identity_checks,
        future_mapping_exclusion_checks=future_checks,
        certified_candidate_checks=certified_checks,
        budget_cents=left.budget_cents,
        min_path_cost_cents=min_path_cost,
        max_path_cost_cents=max_path_cost,
    )


def _verify_task_structure(task: ProceduralTask, *, pool: ProductPool) -> int:
    scenario = scenario_by_id(task.scenario_id)
    pool_by_asin = {product.asin: product for product in pool.products}
    checks = 0
    for phase_index, (phase, slot) in enumerate(zip(task.phases, scenario.slots)):
        if phase.phase_index != phase_index:
            raise ProceduralMemoryDataError("phase index does not match scenario order.")
        if (
            phase.scenario_id != scenario.scenario_id
            or phase.slot_id != slot.slot_id
            or phase.display_name != slot.display_name
            or phase.attribute_name != slot.attribute_name
        ):
            raise ProceduralMemoryDataError("phase metadata disagrees with its scenario.")
        for candidate in phase.candidates:
            checks += 1
            certified = pool_by_asin.get(candidate.asin)
            if certified is None or certified != candidate.product:
                raise ProceduralMemoryDataError(
                    f"candidate {candidate.asin!r} is not an exact certified pool record."
                )
            if candidate.product.split != task.split:
                raise ProceduralMemoryDataError(
                    f"candidate {candidate.asin!r} leaks across product splits."
                )
            if (
                candidate.product.native_title_catalog_match_count != 1
                or candidate.product.native_title_globally_unique is not True
            ):
                raise ProceduralMemoryDataError(
                    f"candidate {candidate.asin!r} lacks catalog-wide title uniqueness."
                )
        if phase_index == 0:
            if phase.root_attribute_value != task.root_attribute_value:
                raise ProceduralMemoryDataError("first-phase root mismatch.")
        else:
            transition = phase.transition
            if transition is None:
                raise ProceduralMemoryDataError("later phase is missing its transition.")
            previous_slot = scenario.slots[phase_index - 1]
            if transition.previous_slot_id != previous_slot.slot_id:
                raise ProceduralMemoryDataError("transition points to the wrong prior slot.")
            if set(value for value, _ in transition.pairs) != set(
                previous_slot.value_ids
            ):
                raise ProceduralMemoryDataError(
                    "transition does not cover both previous natural attributes."
                )
            if set(value for _, value in transition.pairs) != set(slot.value_ids):
                raise ProceduralMemoryDataError(
                    "transition does not cover both current natural attributes."
                )
        expected_question = render_question(
            scenario_id=task.scenario_id,
            phase_index=phase.phase_index,
            slot_id=phase.slot_id,
                candidate_rows=tuple(
                    (
                        candidate.title,
                        candidate.attribute_value,
                        candidate.attribute_display_name,
                )
                for candidate in phase.candidates
            ),
            budget_cents=task.budget_cents,
            root_attribute_value=phase.root_attribute_value,
            transition=phase.transition,
        )
        if phase.question != expected_question:
            raise ProceduralMemoryDataError(
                f"phase {phase.phase_index} question is not canonical visible text."
            )
        for candidate in phase.candidates:
            if phase.question.count(candidate.title) != 1:
                raise ProceduralMemoryDataError(
                    "each candidate title must appear exactly once without singling out "
                    "the target."
                )
            if candidate.asin.casefold() in phase.question.casefold():
                raise ProceduralMemoryDataError(
                    "internal candidate ASINs must remain hidden from policy-visible text."
                )
    return checks


def _verify_pair_symmetry(orbit: CounterfactualOrbit) -> int:
    left, right = orbit.tasks
    if left.phases[0].question == right.phases[0].question:
        raise ProceduralMemoryDataError(
            "session one must expose the changed customer starting request."
        )
    checks = 0
    for phase_index in range(6):
        left_phase = left.phases[phase_index]
        right_phase = right.phases[phase_index]
        if left_phase.candidates != right_phase.candidates:
            raise ProceduralMemoryDataError(
                "counterfactual tasks must keep the real candidate products fixed."
            )
        if phase_index > 0:
            checks += 1
            if left_phase.as_dict(include_target=False) != right_phase.as_dict(
                include_target=False
            ):
                raise ProceduralMemoryDataError(
                    "later policy observations must be byte-identical across the pair."
                )
    return checks


def _verify_transition_bijections(task: ProceduralTask) -> int:
    scenario = scenario_by_id(task.scenario_id)
    count = 0
    for phase_index in range(1, 6):
        transition = task.phases[phase_index].transition
        if transition is None:
            raise ProceduralMemoryDataError("missing transition")
        previous_values = set(scenario.slots[phase_index - 1].value_ids)
        current_values = set(scenario.slots[phase_index].value_ids)
        if {value for value, _ in transition.pairs} != previous_values:
            raise ProceduralMemoryDataError("transition input is not complete.")
        if {value for _, value in transition.pairs} != current_values:
            raise ProceduralMemoryDataError("transition output is not bijective.")
        count += 1
    return count


def _verify_question_locality(task: ProceduralTask) -> int:
    """Check that a session contains no exact row from a future pairing table."""

    checks = 0
    for phase_index, phase in enumerate(task.phases):
        for future_phase in task.phases[phase_index + 1 :]:
            if future_phase.transition is None:
                continue
            for row in transition_rows(
                scenario_id=task.scenario_id,
                current_slot_id=future_phase.slot_id,
                transition=future_phase.transition,
            ):
                checks += 1
                if row in phase.question:
                    raise ProceduralMemoryDataError(
                        f"phase {phase_index} leaks a future customer pairing row."
                    )
    return checks


def _is_structurally_legal(task: ProceduralTask, chosen) -> bool:
    if chosen[0].attribute_value != task.root_attribute_value:
        return False
    previous_value = chosen[0].attribute_value
    for phase_index in range(1, 6):
        transition = task.phases[phase_index].transition
        if transition is None:
            return False
        expected = transition.resolve(previous_value)
        if chosen[phase_index].attribute_value != expected:
            return False
        previous_value = chosen[phase_index].attribute_value
    return True
