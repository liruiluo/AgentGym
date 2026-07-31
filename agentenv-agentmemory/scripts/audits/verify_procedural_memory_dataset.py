#!/usr/bin/env python3
"""Generate and exhaustively verify a deterministic procedural task window."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from agentenv_agentmemory.procedural import (
    NaturalAttributeChainGenerator,
    ProceduralMemoryDataError,
    load_certified_product_pool,
    verify_counterfactual_orbit,
)
from agentenv_agentmemory.procedural.schema import (
    SPLITS,
    canonical_json_bytes,
    require_sha256,
)
from agentenv_agentmemory.procedural.verifier import VERIFIER_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-pool", required=True, type=Path)
    parser.add_argument("--product-pool-sha256", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--generator-seed", required=True, type=int)
    parser.add_argument("--task-count", required=True, type=int)
    parser.add_argument("--start-orbit", default=0, type=int)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_count <= 0 or args.task_count % 2:
        raise ProceduralMemoryDataError("--task-count must be a positive even integer.")
    if args.start_orbit < 0:
        raise ProceduralMemoryDataError("--start-orbit must be non-negative.")
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
    pool = load_certified_product_pool(
        args.product_pool,
        expected_file_sha256=pool_file_sha256,
    )
    generator = NaturalAttributeChainGenerator(pool=pool, seed=args.generator_seed)
    orbit_count = args.task_count // 2
    if args.start_orbit + orbit_count > generator.semantic_period_orbits:
        raise ProceduralMemoryDataError(
            "requested audit window crosses the generator's collision-free semantic period."
        )

    task_stream_sha256 = hashlib.sha256()
    proof_stream_sha256 = hashlib.sha256()
    task_ids: set[str] = set()
    orbit_ids: set[str] = set()
    task_semantic_hashes: set[str] = set()
    orbit_semantic_hashes: set[str] = set()
    product_use_counts: Counter[str] = Counter()
    scenario_orbit_counts: Counter[str] = Counter()
    first_task_id = None
    last_task_id = None
    total_future_mapping_checks = 0
    total_certified_candidate_checks = 0

    for offset in range(orbit_count):
        orbit_index = args.start_orbit + offset
        orbit = generator.generate_orbit(orbit_index, split=args.split)
        proof = verify_counterfactual_orbit(
            orbit,
            pool=pool,
            expected_generator_version=generator.version,
            expected_generator_seed=generator.seed,
        )
        if orbit.semantic_epoch != 0:
            raise ProceduralMemoryDataError(
                "audit window unexpectedly entered a repeated semantic epoch."
            )
        if orbit.orbit_id in orbit_ids:
            raise ProceduralMemoryDataError(f"duplicate orbit ID: {orbit.orbit_id}")
        if orbit.semantic_sha256 in orbit_semantic_hashes:
            raise ProceduralMemoryDataError(
                f"duplicate semantic orbit: {orbit.semantic_sha256}"
            )
        orbit_ids.add(orbit.orbit_id)
        orbit_semantic_hashes.add(orbit.semantic_sha256)
        scenario_orbit_counts[orbit.scenario_id] += 1
        proof_stream_sha256.update(canonical_json_bytes(proof.as_dict()))
        proof_stream_sha256.update(b"\n")
        total_future_mapping_checks += proof.future_mapping_exclusion_checks
        total_certified_candidate_checks += proof.certified_candidate_checks

        for task in orbit.tasks:
            if task.task_id in task_ids:
                raise ProceduralMemoryDataError(f"duplicate task ID: {task.task_id}")
            if task.semantic_sha256 in task_semantic_hashes:
                raise ProceduralMemoryDataError(
                    f"duplicate semantic task: {task.semantic_sha256}"
                )
            task_ids.add(task.task_id)
            task_semantic_hashes.add(task.semantic_sha256)
            first_task_id = first_task_id or task.task_id
            last_task_id = task.task_id
            task_stream_sha256.update(canonical_json_bytes(task.as_dict()))
            task_stream_sha256.update(b"\n")
        for phase in orbit.tasks[0].phases:
            product_use_counts.update(candidate.asin for candidate in phase.candidates)

    split_pool_asins = {
        product.asin for product in pool.products if product.split == args.split
    }
    used_asins = set(product_use_counts)
    if not used_asins <= split_pool_asins:
        raise ProceduralMemoryDataError("generated window used a product from another split.")
    manifest = {
        "schema": "agentmemory_verified_natural_attribute_dataset_window_v3",
        "product_pool": {
            "file_sha256": pool_file_sha256,
            "semantic_sha256": pool.semantic_sha256,
            "pool_id": pool.pool_id,
            "scenario_ids": list(pool.scenario_ids),
            "products_per_attribute_cell": pool.products_per_cell,
        },
        "generator": {
            "version": generator.version,
            "seed": generator.seed,
            "split": args.split,
            "start_orbit": args.start_orbit,
            "orbit_count": orbit_count,
            "task_count": args.task_count,
            "semantic_period_orbits": generator.semantic_period_orbits,
            "semantic_period_tasks": generator.semantic_period_tasks,
            "conservative_task_capacity_without_candidate_order": (
                generator.conservative_task_capacity_without_candidate_order
            ),
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
            "scenario_orbit_counts": dict(sorted(scenario_orbit_counts.items())),
            "split_pool_product_count": len(split_pool_asins),
            "used_product_count": len(used_asins),
            "unused_split_product_count": len(split_pool_asins - used_asins),
            "min_product_orbit_uses": min(product_use_counts.values(), default=0),
            "max_product_orbit_uses": max(product_use_counts.values(), default=0),
        },
        "verification": {
            "verifier_version": VERIFIER_VERSION,
            "candidate_count_per_phase": 2,
            "task_prompt_product_identity": "complete_native_title",
            "target_asin_in_task_prompt": False,
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
            "purchase_receipt_asin_verification": True,
            "catalog_wide_normalized_title_uniqueness": True,
            "phase_count_per_task": 6,
            "counterfactual_tasks_per_orbit": 2,
            "enumerated_paths_per_task": 64,
            "total_enumerated_paths": args.task_count * 64,
            "unique_legal_paths_per_task": args.task_count,
            "later_observations_pair_identical": True,
            "correct_target_flips_per_pair": 6,
            "natural_attribute_order_chain": True,
            "complete_bijections_per_task": 5,
            "future_mapping_exclusion_checks": total_future_mapping_checks,
            "certified_candidate_checks": total_certified_candidate_checks,
            "budget_prunes_zero_paths": True,
            "unique_task_ids": len(task_ids) == args.task_count,
            "unique_orbit_ids": len(orbit_ids) == orbit_count,
            "unique_semantic_tasks": len(task_semantic_hashes) == args.task_count,
            "unique_semantic_orbits": len(orbit_semantic_hashes) == orbit_count,
            "asin_split_isolated": True,
            "human_review_required": False,
            "llm_judge_required": False,
            "paper_eligible": False,
        },
    }
    output_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if expected_manifest_sha256 is not None and output_sha256 != expected_manifest_sha256:
        raise ProceduralMemoryDataError(
            "replayed task-window manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {output_sha256}."
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_bytes(output_bytes)
    print(
        "AGENTMEMORY_PROCEDURAL_DATASET_VERIFIED "
        f"tasks={args.task_count} orbits={orbit_count} "
        f"enumerated_paths={args.task_count * 64} "
        f"manifest_sha256={output_sha256} human_review_required=false "
        "llm_judge_required=false"
    )


if __name__ == "__main__":
    main()
