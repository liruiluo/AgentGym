from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from ..latent_preference.schema import PreferenceProductPool, PreferenceRecipe, canonical_sha256, normalize_native_title
from .generator import PHASE_CATEGORY_POSITIONS
from .question_format import render_recency_question
from .schema import PROOF_SCHEMA, RecencyOverrideDataError, RecencyOverrideOrbit, RecencyOverrideTask


VERIFIER_VERSION = "recency_override_exhaustive_v1"
CORE_BUDGET_MARGIN_CENTS = 5_000
CORE_BUDGET_ROUNDING_CENTS = 1_000


@dataclass(frozen=True)
class RecencyOverrideOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    split: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str]
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int]
    valid_solution_vectors: tuple[tuple[int, ...], tuple[int, ...]]
    application_observation_identity_checks: int
    application_target_flip_count: int
    override_transition_checks: int
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
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
            "task_semantic_sha256": list(self.task_semantic_sha256),
            "enumeration": {
                "paths_per_branch": self.enumerated_path_count_per_branch,
                "valid_solution_counts": list(self.valid_solution_counts),
                "valid_solution_vectors": [list(item) for item in self.valid_solution_vectors],
            },
            "counterfactual_checks": {
                "application_observation_identity_checks": self.application_observation_identity_checks,
                "application_target_flip_count": self.application_target_flip_count,
                "override_transition_checks": self.override_transition_checks,
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
                "counterfactual_stay_flip": True,
                "override_phase_index": 2,
                "application_observation_identity_after_override": True,
                "application_target_flip_after_override": True,
                "unique_legal_purchase_vector_per_branch": True,
                "all_64_purchase_vectors_within_budget": True,
                "canonical_memory_update_required_for_flip": True,
                "old_memory_absent_after_flip_update": True,
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
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


def verify_recency_override_orbit(
    orbit: RecencyOverrideOrbit,
    *,
    pool: PreferenceProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> RecencyOverrideOrbitProof:
    recipe = pool.recipe_by_id(orbit.recipe_id)
    stay, flip = orbit.tasks
    if expected_generator_version is not None and stay.generator_version != expected_generator_version:
        raise RecencyOverrideDataError("unexpected generator version.")
    if expected_generator_seed is not None and stay.generator_seed != expected_generator_seed:
        raise RecencyOverrideDataError("unexpected generator seed.")
    if stay.product_pool_sha256 != pool.semantic_sha256 or flip.product_pool_sha256 != pool.semantic_sha256:
        raise RecencyOverrideDataError("task product-pool hash mismatch.")
    results = tuple(_verify_task(task, recipe=recipe, pool=pool) for task in orbit.tasks)
    costs: list[int] = []
    vectors: list[tuple[int, ...]] = []
    for task, result in zip(orbit.tasks, results):
        valid = []
        for vector in itertools.product((0, 1), repeat=6):
            chosen = tuple(phase.candidates[index] for phase, index in zip(task.phases, vector))
            cost = sum(item.price_cents for item in chosen)
            costs.append(cost)
            if cost > task.budget_cents:
                raise RecencyOverrideDataError("budget prunes a legal purchase vector.")
            if all(item.attribute_value == phase.active_attribute_value for phase, item in zip(task.phases, chosen)):
                valid.append(tuple(vector))
        if len(valid) != 1 or valid[0] != result.declared_vector:
            raise RecencyOverrideDataError("each branch must have one declared legal vector.")
        vectors.append(valid[0])
    identity_checks, flip_count, transition_checks = _verify_pair(orbit, recipe=recipe)
    if stay.budget_cents != flip.budget_cents:
        raise RecencyOverrideDataError("counterfactual branches must share a budget.")
    max_path = max(costs)
    expected_budget = ((max_path + CORE_BUDGET_ROUNDING_CENTS - 1) // CORE_BUDGET_ROUNDING_CENTS) * CORE_BUDGET_ROUNDING_CENTS + CORE_BUDGET_MARGIN_CENTS
    if stay.budget_cents != expected_budget:
        raise RecencyOverrideDataError("budget must equal rounded maximum path plus margin.")
    return RecencyOverrideOrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        recipe_id=orbit.recipe_id,
        split=stay.split,
        generator_version=stay.generator_version,
        generator_seed=stay.generator_seed,
        product_pool_sha256=pool.semantic_sha256,
        task_semantic_sha256=(stay.semantic_sha256, flip.semantic_sha256),
        enumerated_path_count_per_branch=64,
        valid_solution_counts=(1, 1),
        valid_solution_vectors=(vectors[0], vectors[1]),
        application_observation_identity_checks=identity_checks,
        application_target_flip_count=flip_count,
        override_transition_checks=transition_checks,
        certified_candidate_checks=sum(item.certified_candidate_checks for item in results),
        budget_cents=stay.budget_cents,
        min_path_cost_cents=min(costs),
        max_path_cost_cents=max_path,
    )


@dataclass(frozen=True)
class _TaskVerification:
    declared_vector: tuple[int, ...]
    certified_candidate_checks: int


def _verify_task(task: RecencyOverrideTask, *, recipe: PreferenceRecipe, pool: PreferenceProductPool) -> _TaskVerification:
    if task.recipe_id != recipe.recipe_id or task.canonical_memory_key != "user_preference":
        raise RecencyOverrideDataError("task recipe or canonical memory key mismatch.")
    expected_values = {task.old_attribute_value, task.new_attribute_value}
    if expected_values != set(recipe.values):
        raise RecencyOverrideDataError("task old/new values must cover the recipe values.")
    pool_by_asin = {item.asin: item for item in pool.products}
    seen_asins: set[str] = set()
    seen_titles: set[str] = set()
    certified_checks = 0
    for phase_index, (phase, category_position) in enumerate(zip(task.phases, PHASE_CATEGORY_POSITIONS)):
        if phase.phase_index != phase_index or phase.category_id != recipe.categories[category_position]:
            raise RecencyOverrideDataError("phase category schedule mismatch.")
        expected_kind = "evidence" if phase_index == 0 else ("override" if phase_index == 2 else "application")
        if phase.phase_kind != expected_kind:
            raise RecencyOverrideDataError("phase kind schedule mismatch.")
        candidate_values = {item.attribute_value for item in phase.candidates}
        if candidate_values != expected_values:
            raise RecencyOverrideDataError("candidate pair must contain old and new values.")
        for candidate in phase.candidates:
            certified_checks += 1
            expected = pool_by_asin.get(candidate.asin)
            if expected is None:
                raise RecencyOverrideDataError("candidate is outside the certified pool.")
            from_product = candidate.__class__.from_product(expected, product_pool_sha256=pool.semantic_sha256)
            if candidate != from_product:
                raise RecencyOverrideDataError("candidate differs from certified product record.")
            if candidate.split != task.split or candidate.category_id != phase.category_id:
                raise RecencyOverrideDataError("candidate split/category mismatch.")
            normalized = normalize_native_title(candidate.title)
            if candidate.asin in seen_asins or normalized in seen_titles:
                raise RecencyOverrideDataError("task reuses an ASIN or normalized title.")
            seen_asins.add(candidate.asin)
            seen_titles.add(normalized)
        expected_question = render_recency_question(
            user_id=task.user_id,
            phase_index=phase_index,
            phase_kind=phase.phase_kind,
            recipe=recipe,
            category_id=phase.category_id,
            candidates=phase.candidates,
            budget_cents=task.budget_cents,
            old_attribute_value=task.old_attribute_value,
            new_attribute_value=task.new_attribute_value,
            active_attribute_value=phase.active_attribute_value,
            confirmed_attribute_value=phase.confirmed_attribute_value,
        )
        if phase.question != expected_question:
            raise RecencyOverrideDataError("phase question is not canonical.")
        if any(candidate.asin.casefold() in phase.question.casefold() for candidate in phase.candidates):
            raise RecencyOverrideDataError("candidate ASIN leaked into prompt.")
    if len(seen_asins) != 12 or len(seen_titles) != 12:
        raise RecencyOverrideDataError("task must contain twelve unique products.")
    declared = tuple(next(index for index, item in enumerate(phase.candidates) if item.asin == phase.target_asin) for phase in task.phases)
    return _TaskVerification(declared_vector=declared, certified_candidate_checks=certified_checks)


def _verify_pair(orbit: RecencyOverrideOrbit, *, recipe: PreferenceRecipe) -> tuple[int, int, int]:
    stay, flip = orbit.tasks
    shared = ("recipe_id", "user_id", "split", "old_attribute_value", "new_attribute_value", "canonical_memory_key", "budget_cents", "generator_version", "generator_seed", "product_pool_sha256")
    for field in shared:
        if getattr(stay, field) != getattr(flip, field):
            raise RecencyOverrideDataError(f"branches disagree on shared field {field}.")
    if (stay.old_attribute_value, stay.new_attribute_value) != (orbit.old_attribute_value, orbit.new_attribute_value):
        raise RecencyOverrideDataError("orbit old/new values disagree with tasks.")
    for task in (stay, flip):
        if task.phases[0].active_attribute_value != task.old_attribute_value:
            raise RecencyOverrideDataError("initial evidence must use the old value.")
        if task.phases[1].active_attribute_value != task.old_attribute_value:
            raise RecencyOverrideDataError("pre-override application must use the old value.")
    if stay.phases[2].active_attribute_value != stay.old_attribute_value:
        raise RecencyOverrideDataError("stay override must preserve the old value.")
    if flip.phases[2].active_attribute_value != flip.new_attribute_value:
        raise RecencyOverrideDataError("flip override must activate the new value.")
    identity_checks = 0
    flip_count = 0
    for index, (left, right) in enumerate(zip(stay.phases, flip.phases)):
        if left.candidates != right.candidates:
            raise RecencyOverrideDataError("branches must share candidate identity and order.")
        if index in (0, 1):
            if left.question != right.question or left.target_asin != right.target_asin:
                raise RecencyOverrideDataError("pre-override phases must be byte-identical.")
        elif index == 2:
            if left.question == right.question or left.target_asin == right.target_asin:
                raise RecencyOverrideDataError("override phase must distinguish stay and flip.")
            flip_count += 1
        else:
            identity_checks += 1
            if not _visible_phase_equal(left, right):
                raise RecencyOverrideDataError("post-override observations must be byte-identical.")
            if left.active_attribute_value != stay.old_attribute_value:
                raise RecencyOverrideDataError("stay application must retain the old value.")
            if right.active_attribute_value != flip.new_attribute_value:
                raise RecencyOverrideDataError("flip application must retain the new value.")
            if left.target_asin == right.target_asin:
                raise RecencyOverrideDataError("post-override target must flip.")
            flip_count += 1
    if stay.override_mode != "none" or flip.override_mode != "update_or_delete_add":
        raise RecencyOverrideDataError("override modes do not match stay/flip branches.")
    return identity_checks, flip_count, 1


def _visible_phase_equal(left, right) -> bool:
    """Compare only fields that can appear in the policy observation.

    ``active_attribute_value`` and ``target_asin`` are verifier-only labels. They
    must differ across the counterfactual arms after the override while the
    rendered question and candidate listings remain byte-identical.
    """

    return (
        left.phase_index == right.phase_index
        and left.phase_kind == right.phase_kind
        and left.category_id == right.category_id
        and left.category_display_name == right.category_display_name
        and left.candidates == right.candidates
        and left.question == right.question
        and left.confirmed_attribute_value == right.confirmed_attribute_value
    )
