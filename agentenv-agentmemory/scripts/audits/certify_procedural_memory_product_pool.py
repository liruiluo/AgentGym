#!/usr/bin/env python3
"""Build a no-human-review procedural memory pool from native WebShop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agentenv_agentmemory.native_webshop_backend import MemoryArenaNativeWebShopBackend
from agentenv_agentmemory.procedural import (
    SCENARIOS,
    NativeCertificationConfig,
    NativeProductPoolCertificationError,
    certify_native_product_pool,
    file_sha256,
    load_certified_product_pool,
    verify_lucene_index_manifest,
)
from agentenv_agentmemory.procedural.pool_io import write_product_pool_manifest
from agentenv_agentmemory.procedural.schema import ProceduralMemoryDataError, require_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memoryarena-root", required=True, type=Path)
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument("--attributes-file", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--java-home", required=True, type=Path)
    parser.add_argument("--lucene-index-manifest", required=True, type=Path)
    parser.add_argument("--expected-items-sha256", required=True)
    parser.add_argument("--expected-attributes-sha256", required=True)
    parser.add_argument("--expected-lucene-manifest-sha256", required=True)
    parser.add_argument("--expected-price-table-sha256")
    parser.add_argument("--expected-pool-file-sha256")
    parser.add_argument("--output-pool", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    parser.add_argument(
        "--pool-id",
        default="memoryarena_natural_order_chains_v4",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(scenario.scenario_id for scenario in SCENARIOS),
        default=tuple(scenario.scenario_id for scenario in SCENARIOS),
    )
    parser.add_argument("--products-per-cell", type=int, default=4)
    parser.add_argument("--probe-cap-per-cell-split", type=int, default=24)
    parser.add_argument("--max-search-rank", type=int, default=10)
    parser.add_argument("--price-seed", type=int, default=233)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_items = require_sha256(
        args.expected_items_sha256,
        field="expected_items_sha256",
    )
    expected_lucene_manifest = require_sha256(
        args.expected_lucene_manifest_sha256,
        field="expected_lucene_manifest_sha256",
    )
    observed_items = file_sha256(args.items_file)
    if observed_items != expected_items:
        raise ProceduralMemoryDataError(
            "items file SHA256 mismatch: "
            f"expected {expected_items}, observed {observed_items}."
        )
    expected_attributes = require_sha256(
        args.expected_attributes_sha256,
        field="expected_attributes_sha256",
    )
    observed_attributes = file_sha256(args.attributes_file)
    if observed_attributes != expected_attributes:
        raise ProceduralMemoryDataError(
            "attributes file SHA256 mismatch: "
            f"expected {expected_attributes}, observed {observed_attributes}."
        )
    observed_lucene_manifest = file_sha256(args.lucene_index_manifest)
    if observed_lucene_manifest != expected_lucene_manifest:
        raise ProceduralMemoryDataError(
            "Lucene manifest SHA256 mismatch: "
            f"expected {expected_lucene_manifest}, observed {observed_lucene_manifest}."
        )
    index_dir = args.search_root / "indexes-full"
    verified_index_files = verify_lucene_index_manifest(
        args.lucene_index_manifest,
        index_dir=index_dir,
    )

    backend = MemoryArenaNativeWebShopBackend(
        memoryarena_root=args.memoryarena_root,
        items_file=args.items_file,
        attributes_file=args.attributes_file,
        search_root=args.search_root,
        java_home=args.java_home,
        price_seed=args.price_seed,
    )
    try:
        try:
            pool, audit = certify_native_product_pool(
                backend,
                catalog_sha256=observed_items,
                attributes_sha256=observed_attributes,
                lucene_index_sha256=observed_lucene_manifest,
                config=NativeCertificationConfig(
                    pool_id=args.pool_id,
                    scenario_ids=tuple(args.scenarios),
                    products_per_cell=args.products_per_cell,
                    probe_cap_per_cell_split=args.probe_cap_per_cell_split,
                    max_search_rank=args.max_search_rank,
                ),
            )
        except NativeProductPoolCertificationError as exc:
            failure_sha256 = _write_audit(exc.audit, args.output_audit)
            print(
                "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL_CERTIFICATION_FAILED "
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
            raise ProceduralMemoryDataError(
                "native price table SHA256 mismatch: "
                f"expected {expected_price}, observed {pool.price_table_sha256}."
            )

    pool_file_sha256 = write_product_pool_manifest(pool, args.output_pool)
    load_certified_product_pool(
        args.output_pool,
        expected_file_sha256=pool_file_sha256,
    )
    if args.expected_pool_file_sha256 is not None:
        expected_pool = require_sha256(
            args.expected_pool_file_sha256,
            field="expected_pool_file_sha256",
        )
        if pool_file_sha256 != expected_pool:
            raise ProceduralMemoryDataError(
                "replayed product pool SHA256 mismatch: "
                f"expected {expected_pool}, observed {pool_file_sha256}."
            )

    audit["artifacts"] = {
        "product_pool": str(args.output_pool.resolve()),
        "product_pool_file_sha256": pool_file_sha256,
        "lucene_index_file_count": verified_index_files,
    }
    audit_sha256 = _write_audit(audit, args.output_audit)
    print(
        "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL_CERTIFIED "
        f"scenarios={len(pool.scenario_ids)} products={len(pool.products)} "
        f"pool_sha256={pool_file_sha256} "
        f"audit_sha256={audit_sha256} human_review_required=false "
        "llm_judge_required=false"
    )


def _write_audit(audit: dict[str, object], path: Path) -> str:
    audit_bytes = json.dumps(
        audit,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit_bytes)
    return hashlib.sha256(audit_bytes).hexdigest()


if __name__ == "__main__":
    main()
