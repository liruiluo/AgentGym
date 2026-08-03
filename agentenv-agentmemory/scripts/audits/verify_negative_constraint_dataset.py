#!/usr/bin/env python3
"""Generate and exhaustively verify a negative-constraint dataset window."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from agentenv_agentmemory.latent_preference.schema import (
    SPLITS,
    canonical_json_bytes,
    require_sha256,
)
from agentenv_agentmemory.negative_constraint import (
    TASKS_PER_ORBIT,
    NegativeConstraintGenerator,
    load_negative_constraint_native_product_pool,
    verify_negative_constraint_orbit,
)
from agentenv_agentmemory.negative_constraint.verifier import VERIFIER_VERSION


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
    if args.task_count <= 0 or args.task_count % TASKS_PER_ORBIT:
        raise ValueError(
            f"--task-count must be a positive multiple of {TASKS_PER_ORBIT}"
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
    pool = load_negative_constraint_native_product_pool(
        args.product_pool,
        expected_file_sha256=pool_file_sha256,
    )
    generator = NegativeConstraintGenerator(pool=pool, seed=args.generator_seed)
    orbit_count = args.task_count // TASKS_PER_ORBIT
    if args.start_orbit + orbit_count > generator.semantic_period_orbits:
        raise ValueError("requested window crosses the collision-free semantic period")

    task_stream_sha256 = hashlib.sha256()
    proof_stream_sha256 = hashlib.sha256()
    task_ids: set[str] = set()
    orbit_ids: set[str] = set()
    task_semantic_hashes: set[str] = set()
    orbit_semantic_hashes: set[str] = set()
    product_use_counts: Counter[str] = Counter()
    recipe_orbit_counts: Counter[str] = Counter()
    branch_task_counts: Counter[str] = Counter()
    aggregates: Counter[str] = Counter()
    minimum_top1_score = float("inf")
    first_task_id: str | None = None
    last_task_id: str | None = None

    for offset in range(orbit_count):
        orbit_index = args.start_orbit + offset
        orbit = generator.generate_orbit(orbit_index, split=args.split)
        proof = verify_negative_constraint_orbit(
            orbit,
            pool=pool,
            expected_generator_version=generator.version,
            expected_generator_seed=generator.seed,
        )
        if not proof.native_certified:
            raise ValueError("negative dataset proof is not native certified")
        if orbit.semantic_epoch != 0:
            raise ValueError("dataset window entered a repeated semantic epoch")
        if orbit.orbit_id in orbit_ids or orbit.semantic_sha256 in orbit_semantic_hashes:
            raise ValueError("duplicate negative orbit in dataset window")
        orbit_ids.add(orbit.orbit_id)
        orbit_semantic_hashes.add(orbit.semantic_sha256)
        proof_stream_sha256.update(canonical_json_bytes(proof.as_dict()))
        proof_stream_sha256.update(b"\n")
        recipe_orbit_counts[orbit.recipe_id] += 1
        minimum_top1_score = min(minimum_top1_score, proof.top1_retrieval_min_score)
        aggregates["application_observation_identity_checks"] += (
            proof.application_observation_identity_checks
        )
        aggregates["application_three_target_permutation_checks"] += (
            proof.application_three_target_permutation_checks
        )
        aggregates["native_certificate_checks"] += proof.native_certificate_checks

        reference = orbit.tasks[0]
        for phase in reference.phases:
            product_use_counts.update(candidate.asin for candidate in phase.candidates)
        for task in orbit.tasks:
            if task.task_id in task_ids or task.semantic_sha256 in task_semantic_hashes:
                raise ValueError("duplicate negative task in dataset window")
            task_ids.add(task.task_id)
            task_semantic_hashes.add(task.semantic_sha256)
            branch_task_counts[task.branch_kind] += 1
            first_task_id = first_task_id or task.task_id
            last_task_id = task.task_id
            task_stream_sha256.update(canonical_json_bytes(task.as_dict()))
            task_stream_sha256.update(b"\n")

    split_pool_asins = {
        candidate.asin for candidate in pool.candidates if candidate.split == args.split
    }
    if not set(product_use_counts) <= split_pool_asins:
        raise ValueError("negative dataset used a product from another split")

    manifest = {
        "schema": "agentmemory_verified_negative_constraint_dataset_window_v1",
        "surface": "agentmemory_webshop_negative_constraint_top1_train_v1",
        "product_pool": {
            "file_sha256": pool_file_sha256,
            "semantic_sha256": pool.semantic_sha256,
            "pool_id": pool.pool_id,
            "rules_pool_sha256": pool.rules_pool_sha256,
            "source_manifest_sha256": pool.source_manifest_sha256,
            "native_certificate_count": len(pool.native_certificates),
        },
        "generator": {
            "version": generator.version,
            "seed": generator.seed,
            "split": args.split,
            "start_orbit": args.start_orbit,
            "orbit_count": orbit_count,
            "task_count": args.task_count,
            "tasks_per_orbit": TASKS_PER_ORBIT,
            "first_task_id": first_task_id,
            "last_task_id": last_task_id,
        },
        "verifier": {
            "version": VERIFIER_VERSION,
            "proof_stream_sha256": proof_stream_sha256.hexdigest(),
        },
        "streams": {
            "task_stream_sha256": task_stream_sha256.hexdigest(),
            "unique_task_ids": len(task_ids),
            "unique_orbit_ids": len(orbit_ids),
            "unique_task_semantics": len(task_semantic_hashes),
            "unique_orbit_semantics": len(orbit_semantic_hashes),
        },
        "distribution": {
            "recipe_orbit_counts": dict(sorted(recipe_orbit_counts.items())),
            "branch_task_counts": dict(sorted(branch_task_counts.items())),
            "unique_used_asins": len(product_use_counts),
            "product_use_counts": dict(sorted(product_use_counts.items())),
        },
        "verification": {
            "total_enumerated_paths": args.task_count * (3**6),
            "unique_legal_purchase_vector_per_branch": True,
            "three_way_counterfactual": True,
            "application_observation_identity_checks": aggregates[
                "application_observation_identity_checks"
            ],
            "application_three_target_permutation_checks": aggregates[
                "application_three_target_permutation_checks"
            ],
            "native_certificate_checks": aggregates["native_certificate_checks"],
            "top1_retrieval_min_score": minimum_top1_score,
            "split_isolation": True,
            "target_asin_in_task_prompt": False,
            "human_review_required": False,
            "llm_judge_required": False,
            "training_ready": True,
        },
    }
    data = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    observed_manifest_sha256 = hashlib.sha256(data).hexdigest()
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError(
            "negative dataset manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, "
            f"observed {observed_manifest_sha256}"
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_bytes(data)
    print(
        "AGENTMEMORY_NEGATIVE_CONSTRAINT_DATASET_VERIFIED "
        f"tasks={args.task_count} manifest_sha256={observed_manifest_sha256} "
        "training_ready=true"
    )


if __name__ == "__main__":
    main()
