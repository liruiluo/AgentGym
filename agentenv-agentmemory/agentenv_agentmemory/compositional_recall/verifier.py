from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import PreferenceProductPool, canonical_sha256
from ..memory_state import MemoryEntry, rank_memory_entries_bm25
from ..recency_override import (
    RecencyOverrideGenerator,
    verify_recency_override_orbit,
)
from .generator import CompositionalRecallGenerator
from .schema import (
    PROOF_SCHEMA,
    CompositionalRecallDataError,
    CompositionalRecallOrbit,
)


VERIFIER_VERSION = "compositional_recall_factorial_exhaustive_v1"
_SCORE_EPSILON = 1e-12


@dataclass(frozen=True)
class CompositionalRecallOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str, str, str]
    source_recency_proof_sha256: str
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int, int, int]
    hop1_min_top1_margin: float
    hop2_min_top1_margin: float
    sequential_token_bridge_checks: int
    mapping_leave_one_out_checks: int
    directory_leave_one_out_checks: int
    application_observation_identity_checks: int
    application_target_factorial_checks: int

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
            "source_recency_proof_sha256": self.source_recency_proof_sha256,
            "enumeration": {
                "paths_per_branch": self.enumerated_path_count_per_branch,
                "valid_solution_counts": list(self.valid_solution_counts),
                "branch_count": 4,
            },
            "retrieval": {
                "policy": "query_top1",
                "lookup_by_memory_id_allowed": False,
                "required_sequential_retrievals": 2,
                "hop1_role": "customer_to_profile",
                "hop2_role": "profile_directory",
                "hop1_min_top1_margin": self.hop1_min_top1_margin,
                "hop2_min_top1_margin": self.hop2_min_top1_margin,
                "sequential_token_bridge_checks": (
                    self.sequential_token_bridge_checks
                ),
            },
            "leave_one_memory_out": {
                "mapping_checks": self.mapping_leave_one_out_checks,
                "directory_checks": self.directory_leave_one_out_checks,
                "mapping_required": True,
                "directory_required": True,
                "proof_method": "two_by_two_factorial_counterfactual",
            },
            "counterfactual_checks": {
                "application_observation_identity_checks": (
                    self.application_observation_identity_checks
                ),
                "application_target_factorial_checks": (
                    self.application_target_factorial_checks
                ),
                "mapping_bit": "active_profile_token_a_or_b",
                "directory_bit": "identity_or_swapped_value_permutation",
            },
            "verification": {
                "real_frozen_webshop_records_only": True,
                "phase_count_per_task": 6,
                "candidate_count_per_phase": 2,
                "unique_legal_purchase_vector_per_branch": True,
                "all_64_purchase_vectors_within_budget": True,
                "profile_token_absent_from_application_observation": True,
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


def verify_compositional_recall_orbit(
    orbit: CompositionalRecallOrbit,
    *,
    pool: PreferenceProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> CompositionalRecallOrbitProof:
    first = orbit.tasks[0]
    if expected_generator_version is not None and (
        first.generator_version != expected_generator_version
    ):
        raise CompositionalRecallDataError("unexpected generator version.")
    if expected_generator_seed is not None and (
        first.generator_seed != expected_generator_seed
    ):
        raise CompositionalRecallDataError("unexpected generator seed.")
    if first.product_pool_sha256 != pool.semantic_sha256:
        raise CompositionalRecallDataError("task product-pool hash mismatch.")

    regenerated = CompositionalRecallGenerator(
        pool=pool,
        seed=first.generator_seed,
        version=first.generator_version,
    ).generate_orbit(orbit.orbit_index, split=first.split)
    if orbit.as_dict() != regenerated.as_dict():
        raise CompositionalRecallDataError(
            "orbit differs from canonical deterministic generation."
        )

    source_generator = RecencyOverrideGenerator(
        pool=pool,
        seed=first.source_task.generator_seed,
        version=first.source_task.generator_version,
    )
    source_orbit = source_generator.generate_orbit(
        orbit.orbit_index,
        split=first.split,
    )
    if source_orbit.orbit_id != orbit.source_recency_orbit_id:
        raise CompositionalRecallDataError("source recency orbit id mismatch.")
    if any(task.source_task != source_orbit.tasks[0] for task in orbit.tasks):
        raise CompositionalRecallDataError(
            "embedded source task is not the canonical stay branch."
        )
    source_proof = verify_recency_override_orbit(
        source_orbit,
        pool=pool,
        expected_generator_version=source_generator.version,
        expected_generator_seed=source_generator.seed,
    )

    valid_counts = tuple(_verify_task_paths(task) for task in orbit.tasks)
    factorial = _verify_factorial(orbit)
    margins = tuple(_verify_two_hop_top1(task, pool=pool) for task in orbit.tasks)
    return CompositionalRecallOrbitProof(
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
        source_recency_proof_sha256=source_proof.proof_sha256,
        enumerated_path_count_per_branch=64,
        valid_solution_counts=valid_counts,  # type: ignore[arg-type]
        hop1_min_top1_margin=min(item[0] for item in margins),
        hop2_min_top1_margin=min(item[1] for item in margins),
        sequential_token_bridge_checks=len(orbit.tasks),
        mapping_leave_one_out_checks=factorial.mapping_leave_one_out_checks,
        directory_leave_one_out_checks=factorial.directory_leave_one_out_checks,
        application_observation_identity_checks=(
            factorial.application_observation_identity_checks
        ),
        application_target_factorial_checks=(
            factorial.application_target_factorial_checks
        ),
    )


def _verify_task_paths(task) -> int:
    candidates_by_phase = tuple(phase.candidates for phase in task.source_task.phases)
    for question, phase in zip(task.questions, task.source_task.phases):
        if any(candidate.asin.casefold() in question.casefold() for candidate in phase.candidates):
            raise CompositionalRecallDataError("candidate ASIN leaked into prompt.")
    for question in task.questions[2:]:
        if any(token in question for token in task.profile_tokens):
            raise CompositionalRecallDataError(
                "profile token leaked into application observation."
            )
    valid = 0
    for vector in itertools.product((0, 1), repeat=6):
        chosen = tuple(pair[index] for pair, index in zip(candidates_by_phase, vector))
        if sum(item.price_cents for item in chosen) > task.budget_cents:
            raise CompositionalRecallDataError("budget prunes a legal purchase vector.")
        if tuple(item.asin for item in chosen) == task.target_asins:
            valid += 1
    if valid != 1:
        raise CompositionalRecallDataError(
            "each branch must have one declared legal purchase vector."
        )
    return valid


@dataclass(frozen=True)
class _FactorialProof:
    mapping_leave_one_out_checks: int
    directory_leave_one_out_checks: int
    application_observation_identity_checks: int
    application_target_factorial_checks: int


def _verify_factorial(orbit: CompositionalRecallOrbit) -> _FactorialProof:
    by_coordinate = {
        (task.mapping_branch, task.directory_branch): task for task in orbit.tasks
    }
    application_questions = {task.questions[2:] for task in orbit.tasks}
    if len(application_questions) != 1:
        raise CompositionalRecallDataError(
            "all factorial branches must share application observations."
        )
    application_identity_checks = 4 * len(orbit.tasks)
    target_checks = 0
    mapping_checks = 0
    directory_checks = 0

    for directory_branch in ("identity", "swapped"):
        left = by_coordinate[("token_a", directory_branch)]
        right = by_coordinate[("token_b", directory_branch)]
        if left.canonical_memories[1].value != right.canonical_memories[1].value:
            raise CompositionalRecallDataError(
                "mapping counterfactual must keep directory memory fixed."
            )
        if left.questions[1] != right.questions[1]:
            raise CompositionalRecallDataError(
                "mapping counterfactual must keep directory evidence fixed."
            )
        for phase_index in range(1, 6):
            if left.target_asins[phase_index] == right.target_asins[phase_index]:
                raise CompositionalRecallDataError(
                    "mapping bit must flip every profile-dependent target."
                )
            mapping_checks += 1
            target_checks += 1

    for mapping_branch in ("token_a", "token_b"):
        left = by_coordinate[(mapping_branch, "identity")]
        right = by_coordinate[(mapping_branch, "swapped")]
        if left.canonical_memories[0].value != right.canonical_memories[0].value:
            raise CompositionalRecallDataError(
                "directory counterfactual must keep mapping memory fixed."
            )
        if left.questions[0] != right.questions[0]:
            raise CompositionalRecallDataError(
                "directory counterfactual must keep mapping evidence fixed."
            )
        for phase_index in range(2, 6):
            if left.target_asins[phase_index] == right.target_asins[phase_index]:
                raise CompositionalRecallDataError(
                    "directory bit must flip every application target."
                )
            directory_checks += 1
            target_checks += 1

    first_targets = {task.target_asins[0] for task in orbit.tasks}
    if len(first_targets) != 1:
        raise CompositionalRecallDataError(
            "one-time first-session target must be branch invariant."
        )
    return _FactorialProof(
        mapping_leave_one_out_checks=mapping_checks,
        directory_leave_one_out_checks=directory_checks,
        application_observation_identity_checks=application_identity_checks,
        application_target_factorial_checks=target_checks,
    )


def _verify_two_hop_top1(
    task,
    *,
    pool: PreferenceProductPool,
) -> tuple[float, float]:
    entries = [
        MemoryEntry(
            memory_id=f"mem_{index:04d}",
            key=fact.key,
            value=fact.value,
            created_step=index,
            updated_step=index,
        )
        for index, fact in enumerate(task.canonical_memories)
    ]
    margins = []
    for expected_index, fact in enumerate(task.canonical_memories):
        ranked = rank_memory_entries_bm25(
            fact.query,
            entries,
            top_k=len(entries),
        )
        if not ranked or ranked[0][0].memory_id != f"mem_{expected_index:04d}":
            raise CompositionalRecallDataError(
                f"hop {expected_index + 1} canonical query does not retrieve top1."
            )
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = ranked[0][1] - runner_up
        if margin <= _SCORE_EPSILON:
            raise CompositionalRecallDataError(
                f"hop {expected_index + 1} lacks a strict top1 margin."
            )
        margins.append(round(float(margin), 12))
    mapping, directory = task.canonical_memories
    if task.active_profile_token not in mapping.value:
        raise CompositionalRecallDataError("hop1 output does not expose the bridge token.")
    if task.active_profile_token not in directory.query:
        raise CompositionalRecallDataError("hop2 query does not consume the bridge token.")
    recipe = pool.recipe_by_id(task.source_task.recipe_id)
    preferred_display = recipe.value_display_name(task.preferred_attribute_value)
    if preferred_display not in directory.value:
        raise CompositionalRecallDataError(
            "hop2 output does not contain the target attribute display name."
        )
    return margins[0], margins[1]
