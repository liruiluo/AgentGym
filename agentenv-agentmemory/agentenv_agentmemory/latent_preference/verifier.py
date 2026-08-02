from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from .question_format import render_preference_question
from .schema import (
    PROOF_SCHEMA,
    LatentPreferenceDataError,
    LatentPreferenceOrbit,
    LatentPreferenceTask,
    PreferenceCandidate,
    PreferenceProductPool,
    PreferenceRecipe,
    canonical_sha256,
    normalize_native_title,
)


VERIFIER_VERSION = "latent_preference_exhaustive_v1"
CORE_BUDGET_MARGIN_CENTS = 5_000
CORE_BUDGET_ROUNDING_CENTS = 1_000
EXPECTED_CATEGORY_SCHEDULES = {
    1: (0, 1, 2, 3, 1, 2),
    2: (0, 1, 2, 3, 2, 3),
    3: (0, 1, 2, 3, 0, 3),
}


@dataclass(frozen=True)
class LatentPreferenceOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    split: str
    supporting_evidence_count: int
    verifier_version: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str]
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int]
    valid_solution_vectors: tuple[tuple[int, ...], tuple[int, ...]]
    hypothesis_counts_after_evidence: tuple[tuple[int, ...], tuple[int, ...]]
    derived_preference_values: tuple[str, str]
    evidence_target_flip_count: int
    application_target_flip_count: int
    application_observation_identity_checks: int
    evidence_observation_divergence_checks: int
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
            "recipe_id": self.recipe_id,
            "split": self.split,
            "supporting_evidence_count": self.supporting_evidence_count,
            "verifier_version": self.verifier_version,
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
            "task_semantic_sha256": list(self.task_semantic_sha256),
            "inference": {
                "hypothesis_counts_after_evidence": [
                    list(counts) for counts in self.hypothesis_counts_after_evidence
                ],
                "derived_preference_values": list(self.derived_preference_values),
                "resolution_step": 1,
            },
            "enumeration": {
                "paths_per_branch": self.enumerated_path_count_per_branch,
                "valid_solution_counts": list(self.valid_solution_counts),
                "valid_solution_vectors": [
                    list(vector) for vector in self.valid_solution_vectors
                ],
            },
            "counterfactual_checks": {
                "evidence_target_flip_count": self.evidence_target_flip_count,
                "application_target_flip_count": self.application_target_flip_count,
                "application_observation_identity_checks": (
                    self.application_observation_identity_checks
                ),
                "evidence_observation_divergence_checks": (
                    self.evidence_observation_divergence_checks
                ),
            },
            "catalog_checks": {
                "certified_candidate_checks": self.certified_candidate_checks,
                "same_asin_or_title_reuse_within_task": False,
            },
            "budget": {
                "budget_cents": self.budget_cents,
                "min_path_cost_cents": self.min_path_cost_cents,
                "max_path_cost_cents": self.max_path_cost_cents,
                "paths_pruned": 0,
            },
            "verification": {
                "real_frozen_webshop_records_only": True,
                "phase_count_per_task": 6,
                "candidate_count_per_phase": 2,
                "preference_axis_count": 1,
                "resolution_step": 1,
                "supporting_evidence_count": self.supporting_evidence_count,
                "evidence_categories_are_distinct": True,
                "first_application_category_unseen_in_evidence": True,
                "profile_matching_candidate_count_per_phase": 1,
                "counterfactual_application_observation_identity": True,
                "counterfactual_target_flip_for_every_application": True,
                "unique_legal_purchase_vector_per_branch": True,
                "all_64_purchase_vectors_within_budget": True,
                "approved_candidate_titles_in_task_prompt": True,
                "approved_candidate_asins_in_task_prompt": False,
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_eligibility_scope": "current_phase_two_approved_listings",
                "global_catalog_attribute_uniqueness_required": False,
                "global_catalog_normalized_title_uniqueness_required": True,
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


def verify_latent_preference_orbit(
    orbit: LatentPreferenceOrbit,
    *,
    pool: PreferenceProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> LatentPreferenceOrbitProof:
    """Reconstruct the preference and exhaustively verify both 64-way branches."""

    recipe = pool.recipe_by_id(orbit.recipe_id)
    left, right = orbit.tasks
    if left.split != right.split:
        raise LatentPreferenceDataError("counterfactual tasks must share one split.")
    pool_sha256 = pool.semantic_sha256
    for task in orbit.tasks:
        if task.product_pool_sha256 != pool_sha256:
            raise LatentPreferenceDataError("task product-pool hash mismatch.")
        if task.generator_version != left.generator_version:
            raise LatentPreferenceDataError("orbit generator versions disagree.")
        if task.generator_seed != left.generator_seed:
            raise LatentPreferenceDataError("orbit generator seeds disagree.")
    if (
        expected_generator_version is not None
        and left.generator_version != expected_generator_version
    ):
        raise LatentPreferenceDataError("unexpected generator version.")
    if expected_generator_seed is not None and left.generator_seed != expected_generator_seed:
        raise LatentPreferenceDataError("unexpected generator seed.")

    task_results = tuple(
        _verify_task_structure(task, recipe=recipe, pool=pool) for task in orbit.tasks
    )
    certified_checks = sum(result.certified_candidate_checks for result in task_results)

    all_costs: list[int] = []
    valid_vectors: list[tuple[int, ...]] = []
    for task, result in zip(orbit.tasks, task_results):
        task_valid = []
        for vector in itertools.product((0, 1), repeat=6):
            chosen = tuple(
                phase.candidates[choice]
                for phase, choice in zip(task.phases, vector)
            )
            cost = sum(candidate.price_cents for candidate in chosen)
            all_costs.append(cost)
            if cost > task.budget_cents:
                raise LatentPreferenceDataError(
                    "budget prunes a purchase vector and could reveal the answer."
                )
            if _is_structurally_legal(
                task,
                chosen,
                derived_preference=result.derived_preference,
            ):
                task_valid.append(tuple(vector))
        if len(task_valid) != 1:
            raise LatentPreferenceDataError(
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
            raise LatentPreferenceDataError(
                "declared targets disagree with the independently reconstructed solution."
            )
        valid_vectors.append(task_valid[0])

    evidence_divergence, application_identity = _verify_pair_symmetry(
        orbit,
        recipe=recipe,
    )
    evidence_count = left.supporting_evidence_count
    if any(
        left_choice == right_choice
        for left_choice, right_choice in zip(*valid_vectors)
    ):
        raise LatentPreferenceDataError(
            "counterfactual histories must flip the correct candidate in every phase."
        )
    min_path_cost = min(all_costs)
    max_path_cost = max(all_costs)
    expected_budget = (
        (max_path_cost + CORE_BUDGET_ROUNDING_CENTS - 1)
        // CORE_BUDGET_ROUNDING_CENTS
        * CORE_BUDGET_ROUNDING_CENTS
        + CORE_BUDGET_MARGIN_CENTS
    )
    if left.budget_cents != right.budget_cents or left.budget_cents != expected_budget:
        raise LatentPreferenceDataError(
            "task budget must use the canonical rounded maximum plus fixed margin."
        )

    return LatentPreferenceOrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        recipe_id=orbit.recipe_id,
        split=left.split,
        supporting_evidence_count=evidence_count,
        verifier_version=VERIFIER_VERSION,
        generator_version=left.generator_version,
        generator_seed=left.generator_seed,
        product_pool_sha256=pool_sha256,
        task_semantic_sha256=(left.semantic_sha256, right.semantic_sha256),
        enumerated_path_count_per_branch=64,
        valid_solution_counts=(1, 1),
        valid_solution_vectors=(valid_vectors[0], valid_vectors[1]),
        hypothesis_counts_after_evidence=(
            task_results[0].hypothesis_counts,
            task_results[1].hypothesis_counts,
        ),
        derived_preference_values=(
            task_results[0].derived_preference,
            task_results[1].derived_preference,
        ),
        evidence_target_flip_count=evidence_count,
        application_target_flip_count=6 - evidence_count,
        application_observation_identity_checks=application_identity,
        evidence_observation_divergence_checks=evidence_divergence,
        certified_candidate_checks=certified_checks,
        budget_cents=left.budget_cents,
        min_path_cost_cents=min_path_cost,
        max_path_cost_cents=max_path_cost,
    )


@dataclass(frozen=True)
class _TaskVerification:
    derived_preference: str
    hypothesis_counts: tuple[int, ...]
    certified_candidate_checks: int


def _verify_task_structure(
    task: LatentPreferenceTask,
    *,
    recipe: PreferenceRecipe,
    pool: PreferenceProductPool,
) -> _TaskVerification:
    if task.recipe_id != recipe.recipe_id:
        raise LatentPreferenceDataError("task recipe identity mismatch.")
    schedule = EXPECTED_CATEGORY_SCHEDULES[task.supporting_evidence_count]
    pool_sha256 = pool.semantic_sha256
    pool_by_asin = {product.asin: product for product in pool.products}
    seen_asins: set[str] = set()
    seen_titles: set[str] = set()
    certified_checks = 0
    hypotheses = set(recipe.values)
    hypothesis_counts = []
    evidence_categories = []

    for phase_index, (phase, category_position) in enumerate(
        zip(task.phases, schedule)
    ):
        expected_category = recipe.categories[category_position]
        if phase.phase_index != phase_index:
            raise LatentPreferenceDataError("phase index does not match its position.")
        if (
            phase.category_id != expected_category
            or phase.category_display_name
            != recipe.category_display_name(expected_category)
        ):
            raise LatentPreferenceDataError("phase category disagrees with its schedule.")
        expected_kind = (
            "evidence"
            if phase_index < task.supporting_evidence_count
            else "application"
        )
        if phase.phase_kind != expected_kind:
            raise LatentPreferenceDataError("phase kind disagrees with evidence schedule.")
        if {candidate.attribute_value for candidate in phase.candidates} != set(
            recipe.values
        ):
            raise LatentPreferenceDataError(
                "approved pair must contain exactly one product for each recipe value."
            )
        for candidate in phase.candidates:
            certified_checks += 1
            certified = pool_by_asin.get(candidate.asin)
            if certified is None:
                raise LatentPreferenceDataError(
                    f"candidate {candidate.asin!r} is outside the certified pool."
                )
            expected_candidate = PreferenceCandidate.from_product(
                certified,
                product_pool_sha256=pool_sha256,
            )
            if candidate != expected_candidate:
                raise LatentPreferenceDataError(
                    f"candidate {candidate.asin!r} is not an exact certified record."
                )
            if (
                candidate.split != task.split
                or candidate.axis != recipe.axis
                or candidate.category_id != expected_category
            ):
                raise LatentPreferenceDataError(
                    f"candidate {candidate.asin!r} has wrong split/axis/category."
                )
            if candidate.asin in seen_asins:
                raise LatentPreferenceDataError("one task reuses a product ASIN.")
            normalized_title = normalize_native_title(candidate.title)
            if normalized_title in seen_titles:
                raise LatentPreferenceDataError("one task reuses a product title.")
            seen_asins.add(candidate.asin)
            seen_titles.add(normalized_title)

        expected_question = render_preference_question(
            user_id=task.user_id,
            phase_index=phase_index,
            phase_kind=phase.phase_kind,
            supporting_evidence_count=task.supporting_evidence_count,
            recipe=recipe,
            category_id=expected_category,
            candidates=phase.candidates,
            budget_cents=task.budget_cents,
            confirmed_attribute_value=phase.confirmed_attribute_value,
        )
        if phase.question != expected_question:
            raise LatentPreferenceDataError(
                f"phase {phase_index} question is not canonical visible text."
            )
        for candidate in phase.candidates:
            if phase.question.count(candidate.title) != 1:
                raise LatentPreferenceDataError(
                    "each approved title must appear exactly once in the question."
                )
            if candidate.asin.casefold() in phase.question.casefold():
                raise LatentPreferenceDataError(
                    "candidate ASINs must remain hidden from the task prompt."
                )

        if phase.phase_kind == "evidence":
            confirmed = phase.confirmed_attribute_value
            if confirmed not in recipe.values:
                raise LatentPreferenceDataError("evidence confirms an unknown value.")
            matching = tuple(
                candidate
                for candidate in phase.candidates
                if candidate.attribute_value == confirmed
            )
            if len(matching) != 1:
                raise LatentPreferenceDataError(
                    "confirmed evidence does not identify exactly one approved product."
                )
            hypotheses.intersection_update({confirmed})
            hypothesis_counts.append(len(hypotheses))
            evidence_categories.append(expected_category)

    if len(seen_asins) != 12 or len(seen_titles) != 12:
        raise LatentPreferenceDataError(
            "a six-phase task must contain twelve distinct products and titles."
        )
    if len(set(evidence_categories)) != task.supporting_evidence_count:
        raise LatentPreferenceDataError("evidence categories must be distinct.")
    first_application_category = recipe.categories[
        schedule[task.supporting_evidence_count]
    ]
    if first_application_category in set(evidence_categories):
        raise LatentPreferenceDataError(
            "first application category must be unseen in the evidence history."
        )
    if len(hypotheses) != 1:
        raise LatentPreferenceDataError(
            "visible evidence does not resolve exactly one preference hypothesis."
        )
    derived_preference = next(iter(hypotheses))
    if task.preferred_attribute_value != derived_preference:
        raise LatentPreferenceDataError(
            "declared hidden preference disagrees with visible evidence."
        )
    resolution_step = next(
        (index + 1 for index, count in enumerate(hypothesis_counts) if count == 1),
        None,
    )
    if task.resolution_step != resolution_step or resolution_step != 1:
        raise LatentPreferenceDataError("preference resolution step is incorrect.")
    return _TaskVerification(
        derived_preference=derived_preference,
        hypothesis_counts=tuple(hypothesis_counts),
        certified_candidate_checks=certified_checks,
    )


def _verify_pair_symmetry(
    orbit: LatentPreferenceOrbit,
    *,
    recipe: PreferenceRecipe,
) -> tuple[int, int]:
    left, right = orbit.tasks
    shared_fields = (
        "recipe_id",
        "user_id",
        "split",
        "supporting_evidence_count",
        "resolution_step",
        "budget_cents",
        "generator_version",
        "generator_seed",
        "product_pool_sha256",
    )
    for field in shared_fields:
        if getattr(left, field) != getattr(right, field):
            raise LatentPreferenceDataError(
                f"counterfactual tasks disagree on shared field {field}."
            )
    if set(orbit.preferred_attribute_values) != set(recipe.values):
        raise LatentPreferenceDataError("orbit does not cover both recipe values.")
    if tuple(task.preferred_attribute_value for task in orbit.tasks) != tuple(
        orbit.preferred_attribute_values
    ):
        raise LatentPreferenceDataError("task preferences disagree with orbit order.")

    evidence_divergence = 0
    application_identity = 0
    for phase_index, (left_phase, right_phase) in enumerate(
        zip(left.phases, right.phases)
    ):
        if left_phase.candidates != right_phase.candidates:
            raise LatentPreferenceDataError(
                "counterfactual tasks must keep candidates and order fixed."
            )
        if left_phase.target_asin == right_phase.target_asin:
            raise LatentPreferenceDataError(
                "counterfactual tasks must flip every target product."
            )
        if phase_index < left.supporting_evidence_count:
            evidence_divergence += 1
            if (
                left_phase.confirmed_attribute_value
                == right_phase.confirmed_attribute_value
                or left_phase.question == right_phase.question
            ):
                raise LatentPreferenceDataError(
                    "counterfactual evidence histories must visibly diverge."
                )
        else:
            application_identity += 1
            if left_phase.as_dict(include_target=False) != right_phase.as_dict(
                include_target=False
            ):
                raise LatentPreferenceDataError(
                    "application observations must be byte-identical across the pair."
                )
    return evidence_divergence, application_identity


def _is_structurally_legal(
    task: LatentPreferenceTask,
    chosen: Sequence[PreferenceCandidate],
    *,
    derived_preference: str,
) -> bool:
    for phase, candidate in zip(task.phases, chosen):
        expected_value = (
            phase.confirmed_attribute_value
            if phase.phase_kind == "evidence"
            else derived_preference
        )
        if candidate.attribute_value != expected_value:
            return False
    return True
