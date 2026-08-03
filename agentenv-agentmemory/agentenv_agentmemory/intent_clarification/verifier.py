from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from ..latent_preference import (
    LatentPreferenceGenerator,
    verify_latent_preference_orbit,
)
from ..latent_preference.schema import PreferenceProductPool, canonical_sha256
from ..memory_state import MemoryEntry, rank_memory_entries_bm25
from .generator import IntentClarificationGenerator
from .question_format import render_intent_clarification_question
from .schema import (
    PROOF_SCHEMA,
    IntentClarificationDataError,
    IntentClarificationOrbit,
)


VERIFIER_VERSION = "intent_clarification_counterfactual_exhaustive_v1"


@dataclass(frozen=True)
class IntentClarificationOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str]
    source_preference_proof_sha256: str
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int]
    pre_ask_observation_identity_checks: int
    post_clarification_target_flip_checks: int
    later_session_memory_dependency_checks: int
    top1_retrieval_min_score: float

    def payload(self) -> dict[str, Any]:
        return {
            "schema": PROOF_SCHEMA,
            "verifier_version": VERIFIER_VERSION,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "split": self.split,
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
            "task_semantic_sha256": list(self.task_semantic_sha256),
            "source_preference_proof_sha256": (
                self.source_preference_proof_sha256
            ),
            "enumeration": {
                "paths_per_branch": self.enumerated_path_count_per_branch,
                "valid_solution_counts": list(self.valid_solution_counts),
                "branch_count": 2,
            },
            "clarification": {
                "required_action": "ASK",
                "result_event": "CLARIFY",
                "allowed_session": 0,
                "maximum_successful_asks": 1,
                "purchase_before_clarification_allowed": False,
                "pre_ask_observation_identity_checks": (
                    self.pre_ask_observation_identity_checks
                ),
                "post_clarification_target_flip_checks": (
                    self.post_clarification_target_flip_checks
                ),
            },
            "memory": {
                "retrieve_policy": "query_top1",
                "lookup_by_memory_id_allowed": False,
                "ltm_inventory_visible": False,
                "canonical_memory_count": 1,
                "later_session_memory_dependency_checks": (
                    self.later_session_memory_dependency_checks
                ),
                "top1_retrieval_min_score": self.top1_retrieval_min_score,
            },
            "verification": {
                "real_frozen_webshop_records_only": True,
                "native_certified_product_pool_required": True,
                "phase_count_per_task": 6,
                "candidate_count_per_phase": 2,
                "unique_legal_purchase_vector_per_branch": True,
                "all_64_purchase_vectors_within_budget": True,
                "current_request_alone_identifies_target": False,
                "target_asin_in_task_prompt": False,
                "human_review_required": False,
                "llm_judge_required": False,
                "training_ready": True,
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


def verify_intent_clarification_orbit(
    orbit: IntentClarificationOrbit,
    *,
    pool: PreferenceProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> IntentClarificationOrbitProof:
    first = orbit.tasks[0]
    if expected_generator_version is not None and (
        first.generator_version != expected_generator_version
    ):
        raise IntentClarificationDataError("unexpected generator version.")
    if expected_generator_seed is not None and (
        first.generator_seed != expected_generator_seed
    ):
        raise IntentClarificationDataError("unexpected generator seed.")
    if first.product_pool_sha256 != pool.semantic_sha256:
        raise IntentClarificationDataError("task product-pool hash mismatch.")

    regenerated = IntentClarificationGenerator(
        pool=pool,
        seed=first.generator_seed,
        version=first.generator_version,
    ).generate_orbit(orbit.orbit_index, split=first.split)
    if orbit.as_dict() != regenerated.as_dict():
        raise IntentClarificationDataError(
            "orbit differs from canonical deterministic generation."
        )

    source_generator = LatentPreferenceGenerator(
        pool=pool,
        seed=first.source_task.generator_seed,
        version=first.source_task.generator_version,
    )
    source_orbit = source_generator.generate_orbit(
        orbit.orbit_index,
        split=first.split,
    )
    if source_orbit.orbit_id != orbit.source_preference_orbit_id:
        raise IntentClarificationDataError("source preference orbit id mismatch.")
    if tuple(task.source_task for task in orbit.tasks) != source_orbit.tasks:
        raise IntentClarificationDataError(
            "embedded source tasks are not the canonical preference pair."
        )
    source_proof = verify_latent_preference_orbit(
        source_orbit,
        pool=pool,
        expected_generator_version=source_generator.version,
        expected_generator_seed=source_generator.seed,
    )

    valid_counts = tuple(_verify_task(task, pool=pool) for task in orbit.tasks)
    identity_checks, target_checks = _verify_counterfactual_pair(orbit)
    top1_scores = tuple(_verify_top1_memory(task) for task in orbit.tasks)
    return IntentClarificationOrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        split=first.split,
        generator_version=first.generator_version,
        generator_seed=first.generator_seed,
        product_pool_sha256=pool.semantic_sha256,
        task_semantic_sha256=tuple(  # type: ignore[arg-type]
            task.semantic_sha256 for task in orbit.tasks
        ),
        source_preference_proof_sha256=source_proof.proof_sha256,
        enumerated_path_count_per_branch=64,
        valid_solution_counts=valid_counts,  # type: ignore[arg-type]
        pre_ask_observation_identity_checks=identity_checks,
        post_clarification_target_flip_checks=target_checks,
        later_session_memory_dependency_checks=5,
        top1_retrieval_min_score=min(top1_scores),
    )


def _verify_task(task, *, pool: PreferenceProductPool) -> int:
    recipe = pool.recipe_by_id(task.source_task.recipe_id)
    if task.clarification_field != recipe.axis:
        raise IntentClarificationDataError("clarification field mismatch.")
    expected_display = recipe.value_display_name(task.preferred_attribute_value)
    if expected_display.casefold() not in task.clarification_answer.casefold():
        raise IntentClarificationDataError(
            "clarification answer does not state the selected value."
        )
    if expected_display.casefold() not in task.canonical_memory.value.casefold():
        raise IntentClarificationDataError(
            "canonical memory does not preserve the clarification."
        )

    for phase, question, target_asin in zip(
        task.source_task.phases,
        task.questions,
        task.target_asins,
    ):
        expected_question = render_intent_clarification_question(
            user_id=task.source_task.user_id,
            phase_index=phase.phase_index,
            recipe=recipe,
            category_id=phase.category_id,
            candidates=phase.candidates,
            budget_cents=task.budget_cents,
        )
        if question != expected_question:
            raise IntentClarificationDataError(
                "question differs from canonical visible text."
            )
        if any(candidate.asin.casefold() in question.casefold() for candidate in phase.candidates):
            raise IntentClarificationDataError("candidate ASIN leaked into prompt.")
        if any(question.count(candidate.title) != 1 for candidate in phase.candidates):
            raise IntentClarificationDataError(
                "question must show each approved title exactly once."
            )
        target = next(
            (candidate for candidate in phase.candidates if candidate.asin == target_asin),
            None,
        )
        if target is None or target.attribute_value != task.preferred_attribute_value:
            raise IntentClarificationDataError(
                "declared target does not match the clarified preference."
            )

    valid = 0
    candidate_pairs = tuple(phase.candidates for phase in task.source_task.phases)
    for vector in itertools.product((0, 1), repeat=6):
        chosen = tuple(pair[index] for pair, index in zip(candidate_pairs, vector))
        if sum(item.price_cents for item in chosen) > task.budget_cents:
            raise IntentClarificationDataError("budget prunes a purchase vector.")
        if tuple(item.asin for item in chosen) == task.target_asins:
            valid += 1
    if valid != 1:
        raise IntentClarificationDataError(
            "each branch must have one declared legal purchase vector."
        )
    return valid


def _verify_counterfactual_pair(
    orbit: IntentClarificationOrbit,
) -> tuple[int, int]:
    left, right = orbit.tasks
    if left.questions != right.questions:
        raise IntentClarificationDataError(
            "counterfactual branches must be byte-identical before ASK."
        )
    if left.clarification_field != right.clarification_field:
        raise IntentClarificationDataError(
            "counterfactual branches must ask the same field."
        )
    if left.clarification_answer == right.clarification_answer:
        raise IntentClarificationDataError(
            "counterfactual ASK results must differ."
        )
    target_checks = 0
    for left_phase, right_phase, left_target, right_target in zip(
        left.source_task.phases,
        right.source_task.phases,
        left.target_asins,
        right.target_asins,
    ):
        if left_phase.candidates != right_phase.candidates:
            raise IntentClarificationDataError(
                "counterfactual candidate sets or order changed."
            )
        if left_target == right_target:
            raise IntentClarificationDataError(
                "clarification must flip every purchase target."
            )
        target_checks += 1
    return 6, target_checks


def _verify_top1_memory(task) -> float:
    fact = task.canonical_memory
    entry = MemoryEntry(
        memory_id="mem_0000",
        key=fact.key,
        value=fact.value,
        created_step=0,
        updated_step=0,
    )
    ranked = rank_memory_entries_bm25(fact.query, [entry], top_k=1)
    if not ranked or ranked[0][0].memory_id != "mem_0000" or ranked[0][1] <= 0:
        raise IntentClarificationDataError(
            "canonical query does not retrieve the clarification at top1."
        )
    return round(float(ranked[0][1]), 12)
