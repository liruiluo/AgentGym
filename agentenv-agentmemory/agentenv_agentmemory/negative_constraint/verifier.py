from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import canonical_sha256
from ..memoryarena_webshop_env import MemoryEntry, rank_memory_entries_bm25
from .generator import NegativeConstraintGenerator
from .schema import (
    PROOF_SCHEMA,
    NegativeConstraintDataError,
    NegativeConstraintOrbit,
    NegativeConstraintProductPool,
)


VERIFIER_VERSION = "negative_constraint_exhaustive_v1"


@dataclass(frozen=True)
class NegativeConstraintOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    split: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str, str]
    enumerated_path_count_per_branch: int
    valid_solution_counts: tuple[int, int, int]
    valid_solution_vectors: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    application_observation_identity_checks: int
    application_three_target_permutation_checks: int
    certified_source_candidate_checks: int
    unique_candidate_checks: int
    top1_retrieval_min_score: float

    def payload(self) -> dict[str, Any]:
        return {
            "schema": PROOF_SCHEMA,
            "verifier_version": VERIFIER_VERSION,
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
                "valid_solution_vectors": [
                    list(item) for item in self.valid_solution_vectors
                ],
            },
            "counterfactual_checks": {
                "branch_count": 3,
                "application_observation_identity_checks": (
                    self.application_observation_identity_checks
                ),
                "application_three_target_permutation_checks": (
                    self.application_three_target_permutation_checks
                ),
                "leave_history_out_has_three_conflicting_targets": True,
            },
            "catalog_checks": {
                "source_candidate_checks": self.certified_source_candidate_checks,
                "unique_candidate_checks": self.unique_candidate_checks,
                "candidate_count_per_phase": 3,
                "distinct_attribute_values_per_phase": 3,
                "same_asin_or_title_reuse_within_task": False,
            },
            "retrieval_checks": {
                "policy": "query_top1",
                "canonical_memory_count": 1,
                "minimum_positive_top1_score": self.top1_retrieval_min_score,
            },
            "verification": {
                "rules_generated_from_frozen_webshop_catalog": True,
                "native_search_certified": False,
                "native_open_certified": False,
                "native_purchase_certified": False,
                "training_ready": False,
                "phase_count_per_task": 6,
                "three_way_counterfactual": True,
                "unique_legal_purchase_vector_per_branch": True,
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


def verify_negative_constraint_orbit(
    orbit: NegativeConstraintOrbit,
    *,
    pool: NegativeConstraintProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> NegativeConstraintOrbitProof:
    first = orbit.tasks[0]
    split = first.split
    if expected_generator_version is not None and (
        first.generator_version != expected_generator_version
    ):
        raise NegativeConstraintDataError("unexpected generator version.")
    if expected_generator_seed is not None and (
        first.generator_seed != expected_generator_seed
    ):
        raise NegativeConstraintDataError("unexpected generator seed.")
    if any(
        item.generator_version != first.generator_version
        or item.generator_seed != first.generator_seed
        or item.product_pool_sha256 != pool.semantic_sha256
        or item.split != split
        for item in orbit.tasks
    ):
        raise NegativeConstraintDataError(
            "negative orbit generator, split, or pool metadata is inconsistent."
        )
    regenerated = NegativeConstraintGenerator(
        pool=pool,
        seed=first.generator_seed,
        version=first.generator_version,
    ).generate_orbit(orbit.orbit_index, split=split)
    if orbit.as_dict(include_targets=True) != regenerated.as_dict(
        include_targets=True
    ):
        raise NegativeConstraintDataError(
            "negative orbit differs from canonical deterministic generation."
        )

    recipe = pool.recipe_by_id(orbit.recipe_id)
    if tuple(item.allowed_attribute_value for item in orbit.tasks) != recipe.values:
        raise NegativeConstraintDataError(
            "negative branch order must match the recipe value order."
        )
    reference = orbit.tasks[0]
    application_identity_checks = 0
    application_target_permutation_checks = 0
    for phase_index in range(1, 6):
        questions = {item.questions[phase_index] for item in orbit.tasks}
        if len(questions) != 1:
            raise NegativeConstraintDataError(
                "negative application observations must be byte-identical."
            )
        candidate_asins = {
            candidate.asin
            for candidate in reference.phases[phase_index].candidates
        }
        targets = {item.target_asins[phase_index] for item in orbit.tasks}
        if targets != candidate_asins or len(targets) != 3:
            raise NegativeConstraintDataError(
                "three negative branches must permute all application targets."
            )
        application_identity_checks += 1
        application_target_permutation_checks += 1
    if len({item.questions[0] for item in orbit.tasks}) != 3:
        raise NegativeConstraintDataError(
            "negative evidence observations must expose three constraint sets."
        )

    pool_by_asin = {item.asin: item for item in pool.candidates}
    source_candidate_checks = 0
    unique_candidate_checks = 0
    top1_scores: list[float] = []
    solution_counts: list[int] = []
    solution_vectors: list[tuple[int, ...]] = []
    for task in orbit.tasks:
        if set(task.forbidden_attribute_values) != set(recipe.values) - {
            task.allowed_attribute_value
        }:
            raise NegativeConstraintDataError(
                "negative forbidden values are not the allowed-value complement."
            )
        seen_asins: set[str] = set()
        seen_titles: set[str] = set()
        for phase in task.phases:
            if {item.attribute_value for item in phase.candidates} != set(
                recipe.values
            ):
                raise NegativeConstraintDataError(
                    "negative phase does not cover all recipe values."
                )
            for candidate in phase.candidates:
                source = pool_by_asin.get(candidate.asin)
                if source != candidate:
                    raise NegativeConstraintDataError(
                        "negative phase candidate is absent from the frozen rules pool."
                    )
                source_candidate_checks += 1
                if candidate.asin in phase.question:
                    raise NegativeConstraintDataError(
                        "negative question leaks a candidate ASIN."
                    )
                if candidate.asin in task.canonical_memory_value:
                    raise NegativeConstraintDataError(
                        "negative canonical memory leaks a candidate ASIN."
                    )
                if candidate.asin in seen_asins or (
                    candidate.normalized_title in seen_titles
                ):
                    raise NegativeConstraintDataError(
                        "negative task reuses an ASIN or full title."
                    )
                seen_asins.add(candidate.asin)
                seen_titles.add(candidate.normalized_title)
                unique_candidate_checks += 1
        entry = MemoryEntry(
            memory_id="mem_0000",
            key=task.canonical_memory_key,
            value=task.canonical_memory_value,
            created_step=1,
            updated_step=1,
        )
        ranked = rank_memory_entries_bm25(
            task.canonical_retrieval_query,
            [entry],
            top_k=1,
        )
        if len(ranked) != 1 or ranked[0][0].memory_id != "mem_0000":
            raise NegativeConstraintDataError(
                "canonical negative query does not retrieve its memory."
            )
        if ranked[0][1] <= 0:
            raise NegativeConstraintDataError(
                "canonical negative query must have a positive retrieval score."
            )
        top1_scores.append(float(ranked[0][1]))

        valid: list[tuple[int, ...]] = []
        for vector in itertools.product(range(3), repeat=6):
            selected = tuple(
                phase.candidates[position].asin
                for phase, position in zip(task.phases, vector)
            )
            if selected == task.target_asins:
                valid.append(tuple(vector))
        if len(valid) != 1:
            raise NegativeConstraintDataError(
                "negative branch must have one legal purchase vector."
            )
        solution_counts.append(len(valid))
        solution_vectors.append(valid[0])

    return NegativeConstraintOrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        recipe_id=orbit.recipe_id,
        split=split,
        generator_version=first.generator_version,
        generator_seed=first.generator_seed,
        product_pool_sha256=pool.semantic_sha256,
        task_semantic_sha256=tuple(
            item.semantic_sha256 for item in orbit.tasks
        ),  # type: ignore[arg-type]
        enumerated_path_count_per_branch=3**6,
        valid_solution_counts=tuple(solution_counts),  # type: ignore[arg-type]
        valid_solution_vectors=tuple(solution_vectors),  # type: ignore[arg-type]
        application_observation_identity_checks=application_identity_checks,
        application_three_target_permutation_checks=(
            application_target_permutation_checks
        ),
        certified_source_candidate_checks=source_candidate_checks,
        unique_candidate_checks=unique_candidate_checks,
        top1_retrieval_min_score=min(top1_scores),
    )
