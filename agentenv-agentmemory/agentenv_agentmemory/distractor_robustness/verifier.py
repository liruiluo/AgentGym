from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import (
    PreferenceProductPool,
    canonical_sha256,
    normalize_native_title,
)
from ..memory_state import MemoryEntry, rank_memory_entries_bm25
from ..recency_override import (
    RecencyOverrideGenerator,
    verify_recency_override_orbit,
)
from .generator import (
    CANONICAL_MEMORY_KEY,
    DistractorRobustnessGenerator,
)
from .schema import (
    PROOF_SCHEMA,
    SIMILARITY_TIERS,
    DistractorRobustnessDataError,
    DistractorRobustnessOrbit,
)


VERIFIER_VERSION = "distractor_robustness_top1_exhaustive_v1"
_SCORE_EPSILON = 1e-12


@dataclass(frozen=True)
class DistractorRobustnessOrbitProof:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    generator_version: str
    generator_seed: int
    product_pool_sha256: str
    task_semantic_sha256: tuple[str, str]
    source_recency_proof_sha256: str
    source_recency_valid_solution_counts: tuple[int, int]
    distractor_count: int
    similarity_tier_counts: tuple[tuple[str, int], ...]
    canonical_top1_score: float
    runner_up_score: float
    canonical_top1_margin: float
    similarity_tier_max_scores: tuple[tuple[str, float | None], ...]
    visible_question_identity_checks: int
    target_identity_checks: int
    leak_checked_memory_count: int

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
            "source_recency": {
                "proof_sha256": self.source_recency_proof_sha256,
                "valid_solution_counts": list(
                    self.source_recency_valid_solution_counts
                ),
                "six_phase_budget_exhaustive": True,
                "candidate_target_unique": True,
            },
            "retrieval": {
                "policy": "query_top1",
                "lookup_by_memory_id_allowed": False,
                "distractor_count": self.distractor_count,
                "similarity_tier_counts": dict(self.similarity_tier_counts),
                "canonical_top1_score": self.canonical_top1_score,
                "runner_up_score": self.runner_up_score,
                "canonical_top1_margin": self.canonical_top1_margin,
                "similarity_tier_max_scores": dict(
                    self.similarity_tier_max_scores
                ),
                "strict_top1": True,
            },
            "counterfactual_checks": {
                "clean_distracted_question_identity_checks": (
                    self.visible_question_identity_checks
                ),
                "clean_distracted_target_identity_checks": (
                    self.target_identity_checks
                ),
                "only_initial_memory_differs": True,
            },
            "leakage": {
                "checked_initial_memory_count": self.leak_checked_memory_count,
                "candidate_asin_in_initial_memory": False,
                "complete_candidate_title_in_initial_memory": False,
                "canonical_answer_record_preloaded": False,
                "memory_keys_identical": True,
            },
            "verification": {
                "real_frozen_webshop_records_only": True,
                "phase_count_per_task": 6,
                "candidate_count_per_phase": 2,
                "clean_distracted_pair": True,
                "correct_memory_policy_authored_after_evidence": True,
                "initial_memory_hidden_until_retrieve": True,
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


def verify_distractor_robustness_orbit(
    orbit: DistractorRobustnessOrbit,
    *,
    pool: PreferenceProductPool,
    expected_generator_version: str | None = None,
    expected_generator_seed: int | None = None,
) -> DistractorRobustnessOrbitProof:
    clean, distracted = orbit.tasks
    if expected_generator_version is not None and (
        clean.generator_version != expected_generator_version
    ):
        raise DistractorRobustnessDataError("unexpected generator version.")
    if expected_generator_seed is not None and (
        clean.generator_seed != expected_generator_seed
    ):
        raise DistractorRobustnessDataError("unexpected generator seed.")
    if clean.product_pool_sha256 != pool.semantic_sha256:
        raise DistractorRobustnessDataError("task product-pool hash mismatch.")

    regenerated = DistractorRobustnessGenerator(
        pool=pool,
        seed=clean.generator_seed,
        distractor_count=len(distracted.initial_memories),
        version=clean.generator_version,
    ).generate_orbit(orbit.orbit_index, split=clean.split)
    if orbit.as_dict() != regenerated.as_dict():
        raise DistractorRobustnessDataError(
            "orbit differs from canonical deterministic generation."
        )

    source_generator = RecencyOverrideGenerator(
        pool=pool,
        seed=clean.source_task.generator_seed,
        version=clean.source_task.generator_version,
    )
    source_orbit = source_generator.generate_orbit(
        orbit.orbit_index,
        split=clean.split,
    )
    if source_orbit.orbit_id != orbit.source_recency_orbit_id:
        raise DistractorRobustnessDataError("source recency orbit id mismatch.")
    if clean.source_task != source_orbit.tasks[0]:
        raise DistractorRobustnessDataError(
            "embedded source task is not the canonical stay branch."
        )
    source_proof = verify_recency_override_orbit(
        source_orbit,
        pool=pool,
        expected_generator_version=source_generator.version,
        expected_generator_seed=source_generator.seed,
    )

    _verify_pair(clean, distracted)
    leak_checked = _verify_initial_memory_leakage(distracted, pool=pool)
    ranking = _verify_strict_top1(distracted)
    tier_counts = tuple(
        (tier, sum(item.similarity_tier == tier for item in distracted.initial_memories))
        for tier in SIMILARITY_TIERS
    )
    tier_max_scores = tuple(
        (tier, ranking.tier_max_scores.get(tier)) for tier in SIMILARITY_TIERS
    )
    return DistractorRobustnessOrbitProof(
        orbit_id=orbit.orbit_id,
        orbit_index=orbit.orbit_index,
        semantic_epoch=orbit.semantic_epoch,
        split=clean.split,
        generator_version=clean.generator_version,
        generator_seed=clean.generator_seed,
        product_pool_sha256=pool.semantic_sha256,
        task_semantic_sha256=(clean.semantic_sha256, distracted.semantic_sha256),
        source_recency_proof_sha256=source_proof.proof_sha256,
        source_recency_valid_solution_counts=source_proof.valid_solution_counts,
        distractor_count=len(distracted.initial_memories),
        similarity_tier_counts=tier_counts,
        canonical_top1_score=ranking.correct_score,
        runner_up_score=ranking.runner_up_score,
        canonical_top1_margin=ranking.margin,
        similarity_tier_max_scores=tier_max_scores,
        visible_question_identity_checks=6,
        target_identity_checks=6,
        leak_checked_memory_count=leak_checked,
    )


def _verify_pair(clean, distracted) -> None:
    if clean.source_task != distracted.source_task:
        raise DistractorRobustnessDataError(
            "clean and distracted branches must share the exact source task."
        )
    if clean.initial_memories:
        raise DistractorRobustnessDataError("clean branch contains initial memory.")
    if not distracted.initial_memories:
        raise DistractorRobustnessDataError("distracted branch is empty.")
    shared = (
        "canonical_memory_key",
        "canonical_memory_value",
        "canonical_query",
        "generator_version",
        "generator_seed",
        "product_pool_sha256",
    )
    for field in shared:
        if getattr(clean, field) != getattr(distracted, field):
            raise DistractorRobustnessDataError(
                f"counterfactual branches disagree on {field}."
            )
    if clean.canonical_memory_key != CANONICAL_MEMORY_KEY:
        raise DistractorRobustnessDataError("canonical memory key mismatch.")
    if clean.questions != distracted.questions:
        raise DistractorRobustnessDataError("branch questions differ.")
    if clean.target_asins != distracted.target_asins:
        raise DistractorRobustnessDataError("branch targets differ.")
    if clean.budget_cents != distracted.budget_cents:
        raise DistractorRobustnessDataError("branch budgets differ.")


def _verify_initial_memory_leakage(task, *, pool: PreferenceProductPool) -> int:
    candidates = tuple(
        candidate
        for phase in task.source_task.phases
        for candidate in phase.candidates
    )
    normalized_titles = tuple(
        normalize_native_title(candidate.title) for candidate in candidates
    )
    for memory in task.initial_memories:
        if memory.key != task.canonical_memory_key:
            raise DistractorRobustnessDataError(
                "all initial keys must be identical to prevent an inventory shortcut."
            )
        text = f"{memory.key} {memory.value}".casefold()
        normalized_text = normalize_native_title(text)
        if any(candidate.asin.casefold() in text for candidate in candidates):
            raise DistractorRobustnessDataError(
                "candidate ASIN leaked into initial memory."
            )
        if any(title and title in normalized_text for title in normalized_titles):
            raise DistractorRobustnessDataError(
                "complete candidate title leaked into initial memory."
            )
        if memory.value == task.canonical_memory_value:
            raise DistractorRobustnessDataError(
                "canonical answer record was preloaded as a distractor."
            )
    return len(task.initial_memories)


@dataclass(frozen=True)
class _RankingProof:
    correct_score: float
    runner_up_score: float
    margin: float
    tier_max_scores: dict[str, float]


def _verify_strict_top1(task) -> _RankingProof:
    correct = MemoryEntry(
        memory_id="mem_correct",
        key=task.canonical_memory_key,
        value=task.canonical_memory_value,
        created_step=1,
        updated_step=1,
    )
    distractors = [
        MemoryEntry(
            memory_id=f"mem_distractor_{index:04d}",
            key=item.key,
            value=item.value,
            created_step=0,
            updated_step=0,
        )
        for index, item in enumerate(task.initial_memories)
    ]
    ranked = rank_memory_entries_bm25(
        task.canonical_query,
        [correct, *distractors],
        top_k=1 + len(distractors),
    )
    if len(ranked) != 1 + len(distractors):
        raise DistractorRobustnessDataError(
            "canonical query must assign a positive score to every proof entry."
        )
    if ranked[0][0].memory_id != correct.memory_id:
        raise DistractorRobustnessDataError(
            "canonical memory is not top1 for the canonical query."
        )
    correct_score = ranked[0][1]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if correct_score <= runner_up_score + _SCORE_EPSILON:
        raise DistractorRobustnessDataError(
            "canonical memory does not have a strict top1 margin."
        )
    tier_by_id = {
        f"mem_distractor_{index:04d}": item.similarity_tier
        for index, item in enumerate(task.initial_memories)
    }
    tier_max_scores: dict[str, float] = {}
    for entry, score in ranked[1:]:
        tier = tier_by_id[entry.memory_id]
        tier_max_scores[tier] = max(tier_max_scores.get(tier, 0.0), score)
    rounded_correct = _round_score(correct_score)
    rounded_runner_up = _round_score(runner_up_score)
    return _RankingProof(
        correct_score=rounded_correct,
        runner_up_score=rounded_runner_up,
        margin=_round_score(correct_score - runner_up_score),
        tier_max_scores={
            tier: _round_score(score) for tier, score in tier_max_scores.items()
        },
    )


def _round_score(value: float) -> float:
    return round(float(value), 12)
