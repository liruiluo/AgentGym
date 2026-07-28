#!/usr/bin/env python3
"""Summarize full-catalog pool counts under active allow/deny contexts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


CHAIN_TO_FAMILY = {
    "baking_decoration_strict_aesthetic_6_step": "baking",
    "home_decor_living_room_style_harmony_6_step": "home",
    "electronics_home_theater_tier_6_step": "electronics",
    "grocery_flavor_profile_detailed_3_path_6_step": "grocery",
    "beauty_skincare_routine_oily_vs_dry_6_step": "beauty",
}


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def listify(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def label_sets(counter: Mapping[str, int]) -> list[tuple[frozenset[str], int]]:
    return [
        (frozenset(value for value in key.split("|") if value), int(count))
        for key, count in counter.items()
    ]


def context_maps(step: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, set[str]]]:
    result: dict[tuple[int, str], dict[str, set[str]]] = defaultdict(
        lambda: {"allowed": set(), "denied": set()}
    )
    for source_step, source_label, values in step.get("dependency_map") or []:
        result[(int(source_step), normalize(source_label))]["allowed"].update(
            normalize(value) for value in listify(values)
        )
    for source_step, source_label, values in step.get("reject_map") or []:
        result[(int(source_step), normalize(source_label))]["denied"].update(
            normalize(value) for value in listify(values)
        )
    return result


def compatible_count(
    pools: Iterable[tuple[frozenset[str], int]],
    allowed: set[str],
    denied: set[str],
) -> int:
    return sum(
        count
        for labels, count in pools
        if labels & allowed and not labels & denied
    )


def enumerate_strict_paths(
    chain: Mapping[str, Any], pool_rows: Mapping[tuple[str, int], Mapping[str, Any]]
) -> list[tuple[str, ...]]:
    chain_id = str(chain["chain_id"])
    steps = chain["path"]
    first_pool = pool_rows[(chain_id, 1)]["counts"]["strict_single_ontology_label_single_price"]
    paths = [(label,) for label, count in first_pool.items() if count > 0]
    for step in steps[1:]:
        strict_pool = pool_rows[(chain_id, int(step["step"]))]["counts"][
            "strict_single_ontology_label_single_price"
        ]
        next_paths: list[tuple[str, ...]] = []
        for path in paths:
            allowed: set[str] = set()
            denied: set[str] = set()
            for source_step, source_label, values in step.get("dependency_map") or []:
                if path[int(source_step) - 1] == normalize(source_label):
                    allowed.update(normalize(value) for value in listify(values))
            for source_step, source_label, values in step.get("reject_map") or []:
                if path[int(source_step) - 1] == normalize(source_label):
                    denied.update(normalize(value) for value in listify(values))
            for label, count in strict_pool.items():
                if count > 0 and label in allowed and label not in denied:
                    next_paths.append(path + (label,))
        paths = sorted(set(next_paths))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--domain-data", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    pools = json.loads(args.pool_audit.read_text(encoding="utf-8"))
    domain_data = json.loads(args.domain_data.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    pool_rows = {
        (str(row["chain_id"]), int(row["step"])): row for row in pools["steps"]
    }
    chains = {
        str(row["chain_id"]): row
        for row in domain_data
        if str(row.get("chain_id")) in CHAIN_TO_FAMILY
    }

    contexts: list[dict[str, Any]] = []
    edge_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for chain_id, chain in chains.items():
        for step in chain["path"]:
            step_index = int(step["step"])
            if step_index == 1:
                continue
            row = pool_rows[(chain_id, step_index)]
            price_pools = label_sets(row["counts"]["single_price_semantic_label_set"])
            rating_pools = label_sets(row["counts"]["single_price_rating_semantic_label_set"])
            for (source_step, source_label), rules in sorted(context_maps(step).items()):
                context = {
                    "chain_id": chain_id,
                    "family": CHAIN_TO_FAMILY[chain_id],
                    "step": step_index,
                    "description": step.get("description"),
                    "source_step": source_step,
                    "source_label": source_label,
                    "allowed": sorted(rules["allowed"]),
                    "denied": sorted(rules["denied"]),
                    "single_price_compatible_products": compatible_count(
                        price_pools, rules["allowed"], rules["denied"]
                    ),
                    "single_price_rating_compatible_products": compatible_count(
                        rating_pools, rules["allowed"], rules["denied"]
                    ),
                }
                contexts.append(context)
                edge_groups[(chain_id, step_index, source_step)].append(context)

    supported_edges = []
    for (chain_id, step_index, source_step), rows in sorted(edge_groups.items()):
        supported = [row for row in rows if row["single_price_compatible_products"] >= 2]
        distinct_allowed = {tuple(row["allowed"]) for row in supported}
        if len(supported) >= 2 and len(distinct_allowed) >= 2:
            supported_edges.append(
                {
                    "chain_id": chain_id,
                    "family": CHAIN_TO_FAMILY[chain_id],
                    "step": step_index,
                    "source_step": source_step,
                    "supported_source_label_count": len(supported),
                    "minimum_single_price_compatible_products": min(
                        row["single_price_compatible_products"] for row in supported
                    ),
                }
            )

    path_rows = []
    raw_path_reuse = dataset["reuse"]["semantic_path_reuse"]
    for chain_id, chain in chains.items():
        paths = enumerate_strict_paths(chain, pool_rows)
        family = CHAIN_TO_FAMILY[chain_id]
        path_rows.append(
            {
                "chain_id": chain_id,
                "family": family,
                "strict_single_label_full_catalog_path_count": len(paths),
                "existing_raw_semantic_path_count": raw_path_reuse[family][
                    "unique_semantic_path_count"
                ],
                "path_examples": [" -> ".join(path) for path in paths[:30]],
            }
        )

    step_rows = []
    for row in pools["steps"]:
        chain_id = str(row["chain_id"])
        counts = row["counts"]
        strict = counts["strict_single_ontology_label_single_price"]
        strict_rating = counts["strict_single_ontology_label_single_price_rating"]
        allowed = set(row["allowed_target_labels"])
        if int(row["step"]) == 1:
            allowed = set(strict)
        step_rows.append(
            {
                "chain_id": chain_id,
                "family": CHAIN_TO_FAMILY[chain_id],
                "step": int(row["step"]),
                "description": row["description"],
                "category_product_count": counts["category_product_count"],
                "strict_single_label_single_price_products": sum(strict.values()),
                "strict_single_label_single_price_rating_products": sum(strict_rating.values()),
                "allowed_target_label_count": len(allowed),
                "allowed_target_labels_with_zero_strict_products": sorted(
                    label for label in allowed if strict.get(label, 0) == 0
                ),
                "minimum_nonzero_allowed_label_pool": min(
                    (strict.get(label, 0) for label in allowed if strict.get(label, 0) > 0),
                    default=0,
                ),
            }
        )

    report = {
        "schema": "memoryarena_catalog_generation_feasibility_v2",
        "inputs": {
            "pool_audit": str(args.pool_audit.resolve()),
            "domain_data": str(args.domain_data.resolve()),
            "dataset_audit": str(args.dataset_audit.resolve()),
        },
        "catalog_product_count": pools["scanned_product_count"],
        "category_prefix_relevant_occurrences": pools["relevant_product_occurrence_count"],
        "step_pools": step_rows,
        "active_dependency_contexts": contexts,
        "active_dependency_context_count": len(contexts),
        "active_contexts_with_zero_single_price_products": sum(
            row["single_price_compatible_products"] == 0 for row in contexts
        ),
        "active_context_single_price_pool_summary": {
            "min": min(row["single_price_compatible_products"] for row in contexts),
            "max": max(row["single_price_compatible_products"] for row in contexts),
            "median": sorted(row["single_price_compatible_products"] for row in contexts)[
                len(contexts) // 2
            ],
        },
        "counterfactual_supported_edges": supported_edges,
        "counterfactual_supported_edge_count": len(supported_edges),
        "full_path_feasibility": path_rows,
        "boundary": [
            "Counts use canonical product-category prefix matching and title-label rules only.",
            "A product is context-compatible when it contains at least one active allowed label and no active denied label.",
            "Search rank, native page executability, unique metric optimum, budget, split isolation, and pair observation identity remain mandatory later gates.",
        ],
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    zero_contexts = [row for row in contexts if row["single_price_compatible_products"] == 0]
    lines = [
        "# Full Catalog Pool Feasibility",
        "",
        f"- Scanned products: {report['catalog_product_count']}",
        f"- Category-prefix relevant occurrences: {report['category_prefix_relevant_occurrences']}",
        f"- Active dependency contexts: {len(contexts)}; zero-product contexts: {len(zero_contexts)}",
        f"- Counterfactual-supported dependency edges: {len(supported_edges)}",
        "",
        "## Full-path expansion",
        "",
    ]
    for row in path_rows:
        lines.append(
            f"- {row['family']}: raw {row['existing_raw_semantic_path_count']} paths -> "
            f"at least {row['strict_single_label_full_catalog_path_count']} conservative full-catalog paths"
        )
    lines.extend(["", "## Zero-product active contexts", ""])
    if zero_contexts:
        for row in zero_contexts:
            lines.append(
                f"- {row['family']} step {row['step']} from {row['source_label']}: "
                f"allowed={row['allowed']} denied={row['denied']}"
            )
    else:
        lines.append("- None under the context-aware allow/deny rule.")
    lines.extend(["", "## Boundary", ""] + [f"- {value}" for value in report["boundary"]] + [""])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
