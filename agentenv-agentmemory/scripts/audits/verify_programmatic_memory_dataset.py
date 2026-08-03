#!/usr/bin/env python3
"""Generate and verify a fixed window for a programmatic memory surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agentenv_agentmemory.compositional_recall import (
    CompositionalRecallGenerator,
    verify_compositional_recall_orbit,
)
from agentenv_agentmemory.compositional_recall.verifier import (
    VERIFIER_VERSION as COMPOSITIONAL_VERIFIER_VERSION,
)
from agentenv_agentmemory.distractor_robustness import (
    DistractorRobustnessGenerator,
    verify_distractor_robustness_orbit,
)
from agentenv_agentmemory.distractor_robustness.verifier import (
    VERIFIER_VERSION as DISTRACTOR_VERIFIER_VERSION,
)
from agentenv_agentmemory.intent_clarification import (
    IntentClarificationGenerator,
    verify_intent_clarification_orbit,
)
from agentenv_agentmemory.intent_clarification.verifier import (
    VERIFIER_VERSION as INTENT_VERIFIER_VERSION,
)
from agentenv_agentmemory.latent_preference import load_preference_product_pool
from agentenv_agentmemory.latent_preference.schema import (
    SPLITS,
    canonical_json_bytes,
    require_sha256,
)
from agentenv_agentmemory.selective_memory_use import (
    SelectiveMemoryUseGenerator,
    verify_selective_memory_use_orbit,
)
from agentenv_agentmemory.selective_memory_use.verifier import (
    VERIFIER_VERSION as SELECTIVE_MEMORY_VERIFIER_VERSION,
)


DISTRACTOR_SURFACE = (
    "agentmemory_webshop_distractor_robustness_top1_train_v1"
)
COMPOSITIONAL_SURFACE = (
    "agentmemory_webshop_compositional_recall_top1_train_v1"
)
INTENT_SURFACE = "agentmemory_webshop_intent_clarification_train_v1"
SELECTIVE_MEMORY_SURFACE = (
    "agentmemory_webshop_selective_memory_use_top1_train_v1"
)


@dataclass(frozen=True)
class SurfaceSpec:
    generator_type: type
    verifier: Callable[..., Any]
    verifier_version: str
    tasks_per_orbit: int


SURFACE_SPECS = {
    DISTRACTOR_SURFACE: SurfaceSpec(
        generator_type=DistractorRobustnessGenerator,
        verifier=verify_distractor_robustness_orbit,
        verifier_version=DISTRACTOR_VERIFIER_VERSION,
        tasks_per_orbit=2,
    ),
    COMPOSITIONAL_SURFACE: SurfaceSpec(
        generator_type=CompositionalRecallGenerator,
        verifier=verify_compositional_recall_orbit,
        verifier_version=COMPOSITIONAL_VERIFIER_VERSION,
        tasks_per_orbit=4,
    ),
    INTENT_SURFACE: SurfaceSpec(
        generator_type=IntentClarificationGenerator,
        verifier=verify_intent_clarification_orbit,
        verifier_version=INTENT_VERIFIER_VERSION,
        tasks_per_orbit=2,
    ),
    SELECTIVE_MEMORY_SURFACE: SurfaceSpec(
        generator_type=SelectiveMemoryUseGenerator,
        verifier=verify_selective_memory_use_orbit,
        verifier_version=SELECTIVE_MEMORY_VERIFIER_VERSION,
        tasks_per_orbit=4,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=tuple(SURFACE_SPECS), required=True)
    parser.add_argument("--product-pool", required=True, type=Path)
    parser.add_argument("--product-pool-sha256", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--generator-seed", required=True, type=int)
    parser.add_argument("--task-count", required=True, type=int)
    parser.add_argument("--start-orbit", default=0, type=int)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args()


def _surface_verification(surface: str, aggregates: dict[str, Any]) -> dict[str, Any]:
    if surface == DISTRACTOR_SURFACE:
        return {
            "capability": "selective_top1_retrieval_under_hidden_distractors",
            "counterfactual_branches": ["clean", "distracted"],
            "preloaded_distractor_count": aggregates["distractor_count"],
            "strict_top1_min_margin": aggregates["top1_min_margin"],
            "total_leak_checked_initial_memories": aggregates[
                "leak_checked_memories"
            ],
            "clean_distracted_question_identity_checks": aggregates[
                "question_identity_checks"
            ],
            "clean_distracted_target_identity_checks": aggregates[
                "target_identity_checks"
            ],
            "correct_memory_preloaded": False,
        }
    if surface == COMPOSITIONAL_SURFACE:
        return {
            "capability": "two_hop_compositional_recall",
            "factorial_branch_count": 4,
            "factorial_coordinates": [
                "active_token_a.directory_identity",
                "active_token_a.directory_swapped",
                "active_token_b.directory_identity",
                "active_token_b.directory_swapped",
            ],
            "required_sequential_retrievals": 2,
            "hop1_min_strict_top1_margin": aggregates["hop1_min_margin"],
            "hop2_min_strict_top1_margin": aggregates["hop2_min_margin"],
            "sequential_token_bridge_checks": aggregates[
                "sequential_token_bridge_checks"
            ],
            "mapping_leave_one_out_checks": aggregates[
                "mapping_leave_one_out_checks"
            ],
            "directory_leave_one_out_checks": aggregates[
                "directory_leave_one_out_checks"
            ],
            "application_observation_identity_checks": aggregates[
                "application_observation_identity_checks"
            ],
            "application_target_factorial_checks": aggregates[
                "application_target_factorial_checks"
            ],
        }
    if surface == SELECTIVE_MEMORY_SURFACE:
        return {
            "capability": "selective_memory_use_or_abstention",
            "factorial_branches": [
                "memory_required_a",
                "memory_not_required_a",
                "memory_required_b",
                "memory_not_required_b",
            ],
            "memory_required_fraction": 0.5,
            "memory_not_required_fraction": 0.5,
            "required_observation_identity_checks": aggregates[
                "required_observation_identity_checks"
            ],
            "required_target_flip_checks": aggregates[
                "required_target_flip_checks"
            ],
            "explicit_constraint_checks": aggregates[
                "explicit_constraint_checks"
            ],
            "stale_memory_conflict_checks": aggregates[
                "stale_memory_conflict_checks"
            ],
            "top1_positive_score_checks": aggregates[
                "top1_positive_score_checks"
            ],
            "memory_action_positive_shaping_allowed": False,
            "unnecessary_memory_action_penalty": -0.01,
        }
    return {
        "capability": "ask_then_remember_intent_clarification",
        "counterfactual_branches": ["preference_a", "preference_b"],
        "required_action": "ASK",
        "clarification_event": "CLARIFY",
        "ask_allowed_session": 0,
        "maximum_successful_asks": 1,
        "purchase_before_clarification_allowed": False,
        "pre_ask_observation_identity_checks": aggregates[
            "pre_ask_observation_identity_checks"
        ],
        "post_clarification_target_flip_checks": aggregates[
            "post_clarification_target_flip_checks"
        ],
        "later_session_memory_dependency_checks": aggregates[
            "later_session_memory_dependency_checks"
        ],
        "top1_retrieval_min_score": aggregates["top1_retrieval_min_score"],
    }


def _new_aggregates(surface: str) -> dict[str, Any]:
    if surface == DISTRACTOR_SURFACE:
        return {
            "distractor_count": None,
            "top1_min_margin": float("inf"),
            "leak_checked_memories": 0,
            "question_identity_checks": 0,
            "target_identity_checks": 0,
        }
    if surface == COMPOSITIONAL_SURFACE:
        return {
            "hop1_min_margin": float("inf"),
            "hop2_min_margin": float("inf"),
            "sequential_token_bridge_checks": 0,
            "mapping_leave_one_out_checks": 0,
            "directory_leave_one_out_checks": 0,
            "application_observation_identity_checks": 0,
            "application_target_factorial_checks": 0,
        }
    if surface == SELECTIVE_MEMORY_SURFACE:
        return {
            "required_observation_identity_checks": 0,
            "required_target_flip_checks": 0,
            "explicit_constraint_checks": 0,
            "stale_memory_conflict_checks": 0,
            "top1_positive_score_checks": 0,
        }
    return {
        "pre_ask_observation_identity_checks": 0,
        "post_clarification_target_flip_checks": 0,
        "later_session_memory_dependency_checks": 0,
        "top1_retrieval_min_score": float("inf"),
    }


def _accumulate_surface_proof(
    surface: str,
    aggregates: dict[str, Any],
    proof: Any,
) -> None:
    if surface == DISTRACTOR_SURFACE:
        if aggregates["distractor_count"] is None:
            aggregates["distractor_count"] = proof.distractor_count
        elif aggregates["distractor_count"] != proof.distractor_count:
            raise ValueError("distractor count changed inside one dataset window")
        aggregates["top1_min_margin"] = min(
            aggregates["top1_min_margin"],
            proof.canonical_top1_margin,
        )
        aggregates["leak_checked_memories"] += proof.leak_checked_memory_count
        aggregates["question_identity_checks"] += (
            proof.visible_question_identity_checks
        )
        aggregates["target_identity_checks"] += proof.target_identity_checks
        return
    if surface == COMPOSITIONAL_SURFACE:
        aggregates["hop1_min_margin"] = min(
            aggregates["hop1_min_margin"],
            proof.hop1_min_top1_margin,
        )
        aggregates["hop2_min_margin"] = min(
            aggregates["hop2_min_margin"],
            proof.hop2_min_top1_margin,
        )
        for field in (
            "sequential_token_bridge_checks",
            "mapping_leave_one_out_checks",
            "directory_leave_one_out_checks",
            "application_observation_identity_checks",
            "application_target_factorial_checks",
        ):
            aggregates[field] += getattr(proof, field)
        return
    if surface == SELECTIVE_MEMORY_SURFACE:
        for field in (
            "required_observation_identity_checks",
            "required_target_flip_checks",
            "explicit_constraint_checks",
            "stale_memory_conflict_checks",
            "top1_positive_score_checks",
        ):
            aggregates[field] += getattr(proof, field)
        return
    for field in (
        "pre_ask_observation_identity_checks",
        "post_clarification_target_flip_checks",
        "later_session_memory_dependency_checks",
    ):
        aggregates[field] += getattr(proof, field)
    aggregates["top1_retrieval_min_score"] = min(
        aggregates["top1_retrieval_min_score"],
        proof.top1_retrieval_min_score,
    )


def main() -> None:
    args = parse_args()
    spec = SURFACE_SPECS[args.surface]
    if (
        args.task_count <= 0
        or args.task_count % spec.tasks_per_orbit
    ):
        raise ValueError(
            "--task-count must be a positive multiple of the surface's "
            f"{spec.tasks_per_orbit}-task orbit"
        )
    if args.start_orbit < 0:
        raise ValueError("--start-orbit must be non-negative")

    pool_file_sha256 = require_sha256(
        args.product_pool_sha256,
        field="product_pool_sha256",
    )
    expected_manifest_sha256 = (
        require_sha256(
            args.expected_manifest_sha256,
            field="expected_manifest_sha256",
        )
        if args.expected_manifest_sha256 is not None
        else None
    )
    pool = load_preference_product_pool(
        args.product_pool,
        expected_file_sha256=pool_file_sha256,
    )
    generator = spec.generator_type(pool=pool, seed=args.generator_seed)
    orbit_count = args.task_count // spec.tasks_per_orbit
    if args.start_orbit + orbit_count > generator.semantic_period_orbits:
        raise ValueError(
            "requested window crosses the collision-free semantic period"
        )

    task_stream_sha256 = hashlib.sha256()
    proof_stream_sha256 = hashlib.sha256()
    task_ids: set[str] = set()
    orbit_ids: set[str] = set()
    task_semantic_hashes: set[str] = set()
    orbit_semantic_hashes: set[str] = set()
    product_use_counts: Counter[str] = Counter()
    recipe_orbit_counts: Counter[str] = Counter()
    branch_task_counts: Counter[str] = Counter()
    first_task_id = None
    last_task_id = None
    aggregates = _new_aggregates(args.surface)

    for offset in range(orbit_count):
        orbit_index = args.start_orbit + offset
        orbit = generator.generate_orbit(orbit_index, split=args.split)
        proof = spec.verifier(
            orbit,
            pool=pool,
            expected_generator_version=generator.version,
            expected_generator_seed=generator.seed,
        )
        if orbit.semantic_epoch != 0:
            raise ValueError(
                "dataset window unexpectedly entered a repeated semantic epoch"
            )
        if orbit.orbit_id in orbit_ids:
            raise ValueError(f"duplicate orbit ID: {orbit.orbit_id}")
        if orbit.semantic_sha256 in orbit_semantic_hashes:
            raise ValueError(
                f"duplicate semantic orbit: {orbit.semantic_sha256}"
            )
        orbit_ids.add(orbit.orbit_id)
        orbit_semantic_hashes.add(orbit.semantic_sha256)
        proof_stream_sha256.update(canonical_json_bytes(proof.as_dict()))
        proof_stream_sha256.update(b"\n")
        _accumulate_surface_proof(args.surface, aggregates, proof)

        first_source_task = orbit.tasks[0].source_task
        recipe_orbit_counts[first_source_task.recipe_id] += 1
        for phase in first_source_task.phases:
            product_use_counts.update(
                candidate.asin for candidate in phase.candidates
            )

        for task in orbit.tasks:
            if task.task_id in task_ids:
                raise ValueError(f"duplicate task ID: {task.task_id}")
            if task.semantic_sha256 in task_semantic_hashes:
                raise ValueError(
                    f"duplicate semantic task: {task.semantic_sha256}"
                )
            task_ids.add(task.task_id)
            task_semantic_hashes.add(task.semantic_sha256)
            branch_task_counts[task.branch_kind] += 1
            first_task_id = first_task_id or task.task_id
            last_task_id = task.task_id
            task_stream_sha256.update(canonical_json_bytes(task.as_dict()))
            task_stream_sha256.update(b"\n")

    split_pool_asins = {
        product.asin for product in pool.products if product.split == args.split
    }
    used_asins = set(product_use_counts)
    if not used_asins <= split_pool_asins:
        raise ValueError("generated window used a product from another split")

    manifest = {
        "schema": "agentmemory_verified_programmatic_memory_dataset_window_v1",
        "surface": args.surface,
        "product_pool": {
            "file_sha256": pool_file_sha256,
            "semantic_sha256": pool.semantic_sha256,
            "pool_id": pool.pool_id,
            "recipe_ids": [recipe.recipe_id for recipe in pool.recipes],
            "products_per_cell": pool.products_per_cell,
        },
        "generator": {
            "version": generator.version,
            "seed": generator.seed,
            "split": args.split,
            "start_orbit": args.start_orbit,
            "orbit_count": orbit_count,
            "task_count": args.task_count,
            "tasks_per_orbit": spec.tasks_per_orbit,
            "semantic_period_orbits": generator.semantic_period_orbits,
            "semantic_period_tasks": generator.semantic_period_tasks,
        },
        "stream": {
            "first_task_id": first_task_id,
            "last_task_id": last_task_id,
            "task_stream_sha256": task_stream_sha256.hexdigest(),
            "proof_stream_sha256": proof_stream_sha256.hexdigest(),
            "unique_task_semantic_sha256": len(task_semantic_hashes),
            "unique_orbit_semantic_sha256": len(orbit_semantic_hashes),
        },
        "coverage": {
            "branch_task_counts": dict(sorted(branch_task_counts.items())),
            "recipe_orbit_counts": dict(sorted(recipe_orbit_counts.items())),
            "split_pool_product_count": len(split_pool_asins),
            "used_product_count": len(used_asins),
            "unused_split_product_count": len(split_pool_asins - used_asins),
            "min_product_orbit_uses": min(product_use_counts.values(), default=0),
            "max_product_orbit_uses": max(product_use_counts.values(), default=0),
        },
        "verification": {
            "verifier_version": spec.verifier_version,
            "dataset_certification_only": True,
            "native_runtime_smoke_required_separately": True,
            "real_frozen_webshop_records_only": True,
            "phase_count_per_task": 6,
            "candidate_count_per_phase": 2,
            "enumerated_paths_per_task": 64,
            "total_enumerated_paths": args.task_count * 64,
            "unique_legal_paths_per_task": args.task_count,
            "query_top1_contract": {
                "required_fields": ["query"],
                "fixed_result_count": 1,
                "forbidden_fields": ["memory_id", "top_k"],
            },
            "target_asin_in_task_prompt": False,
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
            "unique_task_ids": len(task_ids) == args.task_count,
            "unique_orbit_ids": len(orbit_ids) == orbit_count,
            "unique_semantic_tasks": len(task_semantic_hashes) == args.task_count,
            "unique_semantic_orbits": len(orbit_semantic_hashes) == orbit_count,
            "asin_split_isolated": True,
            "human_review_required": False,
            "llm_judge_required": False,
            "training_ready": True,
            "paper_eligible": False,
            "surface_specific": _surface_verification(
                args.surface,
                aggregates,
            ),
        },
    }
    output_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if (
        expected_manifest_sha256 is not None
        and output_sha256 != expected_manifest_sha256
    ):
        raise ValueError(
            "replayed task-window manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {output_sha256}"
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_bytes(output_bytes)
    print(
        "AGENTMEMORY_PROGRAMMATIC_MEMORY_DATASET_VERIFIED "
        f"surface={args.surface} tasks={args.task_count} orbits={orbit_count} "
        f"enumerated_paths={args.task_count * 64} "
        f"manifest_sha256={output_sha256} human_review_required=false "
        "llm_judge_required=false"
    )


if __name__ == "__main__":
    main()
