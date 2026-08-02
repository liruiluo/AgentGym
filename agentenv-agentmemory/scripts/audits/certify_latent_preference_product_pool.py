#!/usr/bin/env python3
"""Build a no-human-review hidden-preference pool from native WebShop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agentenv_agentmemory.latent_preference import (
    PREFERENCE_RECIPES,
    NativePreferenceCertificationConfig,
    NativePreferencePoolCertificationError,
    certify_native_preference_product_pool,
    load_preference_product_pool,
    write_preference_product_pool_manifest,
)
from agentenv_agentmemory.latent_preference.schema import (
    LatentPreferenceDataError,
    require_sha256,
)
from agentenv_agentmemory.native_webshop_backend import (
    FROZEN_MEMORYARENA_COMMIT,
    MemoryArenaNativeWebShopBackend,
)
from agentenv_agentmemory.procedural import file_sha256, verify_lucene_index_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memoryarena-root", required=True, type=Path)
    parser.add_argument(
        "--memoryarena-base-commit",
        default=FROZEN_MEMORYARENA_COMMIT,
    )
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument("--attributes-file", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--java-home", required=True, type=Path)
    parser.add_argument("--lucene-index-manifest", required=True, type=Path)
    parser.add_argument("--candidate-artifact", required=True, type=Path)
    parser.add_argument("--expected-candidate-artifact-sha256", required=True)
    parser.add_argument("--expected-items-sha256", required=True)
    parser.add_argument("--expected-attributes-sha256", required=True)
    parser.add_argument("--expected-lucene-manifest-sha256", required=True)
    parser.add_argument("--expected-price-table-sha256")
    parser.add_argument("--expected-pool-file-sha256")
    parser.add_argument("--output-pool", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument(
        "--pool-id",
        default="memoryarena_latent_preference_mainline_v4",
    )
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=tuple(recipe.recipe_id for recipe in PREFERENCE_RECIPES),
        default=tuple(recipe.recipe_id for recipe in PREFERENCE_RECIPES),
    )
    parser.add_argument("--products-per-cell", type=int, default=2)
    parser.add_argument("--candidate-cap-per-cell", type=int, default=72)
    parser.add_argument("--max-search-rank", type=int, default=10)
    parser.add_argument("--price-seed", type=int, default=233)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_items = require_sha256(
        args.expected_items_sha256,
        field="expected_items_sha256",
    )
    observed_items = file_sha256(args.items_file)
    if observed_items != expected_items:
        raise LatentPreferenceDataError(
            "items file SHA256 mismatch: "
            f"expected {expected_items}, observed {observed_items}."
        )
    expected_attributes = require_sha256(
        args.expected_attributes_sha256,
        field="expected_attributes_sha256",
    )
    observed_attributes = file_sha256(args.attributes_file)
    if observed_attributes != expected_attributes:
        raise LatentPreferenceDataError(
            "attributes file SHA256 mismatch: "
            f"expected {expected_attributes}, observed {observed_attributes}."
        )
    expected_lucene_manifest = require_sha256(
        args.expected_lucene_manifest_sha256,
        field="expected_lucene_manifest_sha256",
    )
    observed_lucene_manifest = file_sha256(args.lucene_index_manifest)
    if observed_lucene_manifest != expected_lucene_manifest:
        raise LatentPreferenceDataError(
            "Lucene manifest SHA256 mismatch: "
            f"expected {expected_lucene_manifest}, "
            f"observed {observed_lucene_manifest}."
        )
    verified_index_files = verify_lucene_index_manifest(
        args.lucene_index_manifest,
        index_dir=args.search_root / "indexes-full",
    )

    backend = MemoryArenaNativeWebShopBackend(
        memoryarena_root=args.memoryarena_root,
        items_file=args.items_file,
        attributes_file=args.attributes_file,
        search_root=args.search_root,
        java_home=args.java_home,
        expected_memoryarena_commit=args.memoryarena_base_commit,
        price_seed=args.price_seed,
    )
    try:
        try:
            pool, audit = certify_native_preference_product_pool(
                backend,
                candidate_artifact=args.candidate_artifact,
                expected_candidate_artifact_sha256=(
                    args.expected_candidate_artifact_sha256
                ),
                catalog_sha256=observed_items,
                attributes_sha256=observed_attributes,
                lucene_index_sha256=observed_lucene_manifest,
                config=NativePreferenceCertificationConfig(
                    pool_id=args.pool_id,
                    recipe_ids=tuple(args.recipes),
                    products_per_cell=args.products_per_cell,
                    candidate_cap_per_cell=args.candidate_cap_per_cell,
                    max_search_rank=args.max_search_rank,
                ),
            )
        except NativePreferencePoolCertificationError as exc:
            failure_sha256 = _write_audit(exc.audit, args.output_audit)
            print(
                "AGENTMEMORY_LATENT_PREFERENCE_POOL_CERTIFICATION_FAILED "
                f"audit_sha256={failure_sha256} "
                "human_review_required=false llm_judge_required=false"
            )
            raise
    finally:
        backend.close()

    if args.expected_price_table_sha256 is not None:
        expected_price = require_sha256(
            args.expected_price_table_sha256,
            field="expected_price_table_sha256",
        )
        if pool.price_table_sha256 != expected_price:
            raise LatentPreferenceDataError(
                "native price table SHA256 mismatch: "
                f"expected {expected_price}, observed {pool.price_table_sha256}."
            )

    pool_file_sha256 = write_preference_product_pool_manifest(
        pool,
        args.output_pool,
    )
    load_preference_product_pool(
        args.output_pool,
        expected_file_sha256=pool_file_sha256,
    )
    if args.expected_pool_file_sha256 is not None:
        expected_pool = require_sha256(
            args.expected_pool_file_sha256,
            field="expected_pool_file_sha256",
        )
        if pool_file_sha256 != expected_pool:
            raise LatentPreferenceDataError(
                "replayed preference pool SHA256 mismatch: "
                f"expected {expected_pool}, observed {pool_file_sha256}."
            )

    audit["artifacts"] = {
        "product_pool": str(args.output_pool.resolve()),
        "product_pool_file_sha256": pool_file_sha256,
        "lucene_index_file_count": verified_index_files,
    }
    audit_sha256 = _write_audit(audit, args.output_audit)
    print(
        "AGENTMEMORY_LATENT_PREFERENCE_POOL_CERTIFIED "
        f"recipes={len(pool.recipes)} products={len(pool.products)} "
        f"pool_sha256={pool_file_sha256} audit_sha256={audit_sha256} "
        "human_review_required=false llm_judge_required=false"
    )


def _write_audit(audit: dict[str, object], path: Path) -> str:
    data = json.dumps(
        audit,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
