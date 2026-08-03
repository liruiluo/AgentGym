from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

from ..latent_preference import (
    LatentPreferenceGenerator,
    verify_latent_preference_orbit,
)
from ..latent_preference.schema import (
    PreferenceProductPool,
    canonical_sha256,
)
from ..memory_state import MemoryEntry, rank_memory_entries_bm25
from .generator import SelectiveMemoryUseGenerator
from .question_format import render_selective_memory_question
from .schema import (
    BRANCH_SPECS,
    PROOF_SCHEMA,
    SelectiveMemoryUseDataError,
    SelectiveMemoryUseOrbit,
)


VERIFIER_VERSION = "selective_memory_use_top1_exhaustive_v3"

_FORBIDDEN_REQUEST_WORDS = re.compile(r"\b(?:add|retrieve|memory)\b", re.IGNORECASE)
_FORBIDDEN_REQUEST_PHRASES = (
    "saved current profile",
    "older profile history",
)


def _request_leaks_memory_use_instruction(question: str) -> bool:
    try:
        request_and_listings = question.split("\n\n", 1)[1]
        request = request_and_listings.split("\n\nApproved listings:", 1)[0]
    except IndexError as exc:
        raise SelectiveMemoryUseDataError(
            "question lacks the canonical natural-request section."
        ) from exc
    folded = request.casefold()
    return bool(_FORBIDDEN_REQUEST_WORDS.search(request)) or any(
        phrase in folded for phrase in _FORBIDDEN_REQUEST_PHRASES
    )


@dataclass(frozen=True)
class SelectiveMemoryUseOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str, str, str]
    source_preference_proof_sha256: str
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int, int, int]
    valid_solution_vectors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    required_observation_identity_checks: int
    required_target_flip_checks: int
    explicit_constraint_checks: int
    stale_memory_conflict_checks: int
    top1_positive_score_checks: int
    leak_checked_memory_count: int
    certified_candidate_checks: int

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
            "source_preference": {
                "proof_sha256": self.source_preference_proof_sha256,
                "real_frozen_webshop_records_only": True,
                "candidate_and_budget_contract_preserved": True,
            },
            "factorial": {
                "branch_specs": [list(item) for item in BRANCH_SPECS],
                "required_observation_identity_checks": (
                    self.required_observation_identity_checks
                ),
                "required_target_flip_checks": self.required_target_flip_checks,
                "explicit_constraint_checks": self.explicit_constraint_checks,
                "stale_memory_conflict_checks": self.stale_memory_conflict_checks,
            },
            "enumeration": {
                "paths_per_branch": self.enumerated_path_count_per_branch,
                "valid_solution_counts": list(self.valid_solution_counts),
                "valid_solution_vectors": [
                    list(item) for item in self.valid_solution_vectors
                ],
            },
            "retrieval": {
                "policy": "query_top1",
                "lookup_by_memory_id_allowed": False,
                "top_k_override_allowed": False,
                "top1_positive_score_checks": self.top1_positive_score_checks,
                "required_branch_memory_state": "current",
                "not_required_branch_memory_state": "stale_opposite",
            },
            "verification": {
                "phase_count_per_task": 6,
                "candidate_count_per_phase": 2,
                "four_way_factorial_orbit": True,
                "memory_required_without_memory_is_counterfactually_ambiguous": True,
                "memory_not_required_is_explicitly_identifiable": True,
                "memory_not_required_retrieval_is_counterevidence": True,
                "memory_action_positive_shaping_allowed": False,
                "unnecessary_memory_action_penalized": True,
                "unique_legal_purchase_vector_per_branch": True,
                "all_64_purchase_vectors_within_budget": True,
                "approved_candidate_titles_in_task_prompt": True,
                "target_asin_in_task_prompt": False,
                "leak_checked_memory_count": self.leak_checked_memory_count,
                "certified_candidate_checks": self.certified_candidate_checks,
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


def verify_selective_memory_use_orbit(
    orbit: SelectiveMemoryUseOrbit,
    *,
    pool: PreferenceProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> SelectiveMemoryUseOrbitProof:
    first = orbit.tasks[0]
    if expected_generator_version is not None and (
        first.generator_version != expected_generator_version
    ):
        raise SelectiveMemoryUseDataError("unexpected generator version.")
    if expected_generator_seed is not None and (
        first.generator_seed != expected_generator_seed
    ):
        raise SelectiveMemoryUseDataError("unexpected generator seed.")
    if any(task.generator_version != first.generator_version for task in orbit.tasks):
        raise SelectiveMemoryUseDataError("orbit generator versions disagree.")
    if any(task.generator_seed != first.generator_seed for task in orbit.tasks):
        raise SelectiveMemoryUseDataError("orbit generator seeds disagree.")
    if any(task.product_pool_sha256 != pool.semantic_sha256 for task in orbit.tasks):
        raise SelectiveMemoryUseDataError("task product-pool hash mismatch.")

    canonical = SelectiveMemoryUseGenerator(
        pool=pool,
        seed=first.generator_seed,
        version=first.generator_version,
    ).generate_orbit(orbit.orbit_index, split=first.split)
    if orbit.as_dict() != canonical.as_dict():
        raise SelectiveMemoryUseDataError(
            "orbit differs from canonical deterministic generation."
        )

    source_orbit = LatentPreferenceGenerator(
        pool=pool,
        seed=first.generator_seed,
    ).generate_orbit(orbit.orbit_index, split=first.split)
    source_proof = verify_latent_preference_orbit(
        source_orbit,
        pool=pool,
    )
    if orbit.source_preference_orbit_id != source_orbit.orbit_id:
        raise SelectiveMemoryUseDataError("source preference orbit mismatch.")

    recipe = pool.recipe_by_id(source_orbit.recipe_id)
    valid_vectors = []
    top1_checks = 0
    leak_checks = 0
    certified_checks = 0
    for task in orbit.tasks:
        source_task = source_orbit.tasks[task.preference_coordinate]
        if task.source_task != source_task:
            raise SelectiveMemoryUseDataError("certified source task mismatch.")
        if task.preferred_attribute_value != recipe.values[task.preference_coordinate]:
            raise SelectiveMemoryUseDataError("factorial preference coordinate mismatch.")
        for phase_index, (source_phase, question) in enumerate(
            zip(source_task.phases, task.questions)
        ):
            expected_question = render_selective_memory_question(
                user_id=source_task.user_id,
                phase_index=phase_index,
                memory_requirement=task.memory_requirement,
                recipe=recipe,
                category_id=source_phase.category_id,
                candidates=source_phase.candidates,
                budget_cents=source_task.budget_cents,
                preferred_attribute_value=task.preferred_attribute_value,
            )
            if question != expected_question:
                raise SelectiveMemoryUseDataError(
                    f"phase {phase_index} question is not canonical."
                )
            for candidate in source_phase.candidates:
                certified_checks += 1
                if question.count(candidate.title) != 1:
                    raise SelectiveMemoryUseDataError(
                        "each approved title must appear exactly once."
                    )
                if candidate.asin.casefold() in question.casefold():
                    raise SelectiveMemoryUseDataError("candidate ASIN leaked into prompt.")
                if candidate.asin.casefold() in task.initial_memory.value.casefold():
                    raise SelectiveMemoryUseDataError("candidate ASIN leaked into memory.")
        ranked = rank_memory_entries_bm25(
            task.canonical_query,
            [
                MemoryEntry(
                    memory_id="mem_0000",
                    key=task.initial_memory.key,
                    value=task.initial_memory.value,
                    created_step=0,
                    updated_step=0,
                )
            ],
            top_k=1,
        )
        if len(ranked) != 1 or ranked[0][1] <= 0.0:
            raise SelectiveMemoryUseDataError(
                "canonical query does not retrieve the seeded profile as top-1."
            )
        top1_checks += 1
        leak_checks += 1

        task_valid = []
        for vector in itertools.product((0, 1), repeat=6):
            chosen = tuple(
                phase.candidates[position]
                for phase, position in zip(source_task.phases, vector)
            )
            if sum(item.price_cents for item in chosen) > task.budget_cents:
                raise SelectiveMemoryUseDataError(
                    "budget prunes a path and could reveal the target."
                )
            if all(
                item.attribute_value == task.preferred_attribute_value
                for item in chosen
            ):
                task_valid.append(tuple(vector))
        if len(task_valid) != 1:
            raise SelectiveMemoryUseDataError(
                "independent exhaustive verifier expected one valid path."
            )
        declared = tuple(
            next(
                index
                for index, candidate in enumerate(phase.candidates)
                if candidate.asin == target_asin
            )
            for phase, target_asin in zip(source_task.phases, task.target_asins)
        )
        if declared != task_valid[0]:
            raise SelectiveMemoryUseDataError(
                "declared target vector disagrees with the explicit/profile value."
            )
        valid_vectors.append(task_valid[0])

    required_a, not_required_a, required_b, not_required_b = orbit.tasks
    required_identity = 0
    required_target_flips = 0
    explicit_checks = 0
    stale_conflicts = 0
    for phase_index in range(6):
        if required_a.questions[phase_index] != required_b.questions[phase_index]:
            raise SelectiveMemoryUseDataError(
                "required A/B observations must be byte-identical."
            )
        required_identity += 1
        if required_a.target_asins[phase_index] == required_b.target_asins[phase_index]:
            raise SelectiveMemoryUseDataError(
                "required A/B targets must flip in every session."
            )
        required_target_flips += 1
        for task in (not_required_a, not_required_b):
            display = recipe.value_display_name(task.preferred_attribute_value)
            marker = (
                f"required {recipe.axis_display_name} value is {display}"
            )
            if marker not in task.questions[phase_index]:
                raise SelectiveMemoryUseDataError(
                    "not-required request does not state its current constraint."
                )
            if _request_leaks_memory_use_instruction(
                task.questions[phase_index]
            ):
                raise SelectiveMemoryUseDataError(
                    "not-required request leaks a memory-use instruction."
                )
            explicit_checks += 1
            opposite = recipe.values[1 - task.preference_coordinate]
            opposite_display = recipe.value_display_name(opposite)
            if (
                opposite_display not in task.initial_memory.value
                or task.initial_memory.state != "stale"
            ):
                raise SelectiveMemoryUseDataError(
                    "not-required seeded memory is not stale counterevidence."
                )
            stale_conflicts += 1
    for task in (required_a, required_b):
        display = recipe.value_display_name(task.preferred_attribute_value)
        if display not in task.initial_memory.value or task.initial_memory.state != "current":
            raise SelectiveMemoryUseDataError(
                "required branch lacks its hidden current profile."
            )
        for question in task.questions:
            if _request_leaks_memory_use_instruction(question):
                raise SelectiveMemoryUseDataError(
                    "required request leaks a memory-use instruction."
                )
    if valid_vectors[0] != valid_vectors[1] or valid_vectors[2] != valid_vectors[3]:
        raise SelectiveMemoryUseDataError(
            "memory-use condition changed the target for a fixed preference."
        )
    if valid_vectors[0] == valid_vectors[2]:
        raise SelectiveMemoryUseDataError(
            "preference counterfactual failed to flip the target vector."
        )

    return SelectiveMemoryUseOrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        split=first.split,
        generator_version=first.generator_version,
        generator_seed=first.generator_seed,
        product_pool_sha256=pool.semantic_sha256,
        task_semantic_sha256=tuple(
            task.semantic_sha256 for task in orbit.tasks
        ),  # type: ignore[arg-type]
        source_preference_proof_sha256=source_proof.proof_sha256,
        enumerated_path_count_per_branch=64,
        valid_solution_counts=(1, 1, 1, 1),
        valid_solution_vectors=tuple(valid_vectors),  # type: ignore[arg-type]
        required_observation_identity_checks=required_identity,
        required_target_flip_checks=required_target_flips,
        explicit_constraint_checks=explicit_checks,
        stale_memory_conflict_checks=stale_conflicts,
        top1_positive_score_checks=top1_checks,
        leak_checked_memory_count=leak_checks,
        certified_candidate_checks=certified_checks,
    )
