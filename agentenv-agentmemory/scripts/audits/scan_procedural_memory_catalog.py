#!/usr/bin/env python3
"""Count fail-closed natural-attribute cells in a raw WebShop catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from agentenv_agentmemory.procedural import SCENARIOS, classify_product_record, file_sha256
from agentenv_agentmemory.procedural.scenarios import (
    SCENARIO_DEFINITION_SHA256,
    SCENARIO_DEFINITION_VERSION,
    scenario_by_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(scenario.scenario_id for scenario in SCENARIOS),
        default=tuple(scenario.scenario_id for scenario in SCENARIOS),
    )
    parser.add_argument("--max-examples-per-cell", type=int, default=3)
    parser.add_argument("--max-catalog-values-per-cell", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _top_counts(counter: Counter[str], *, limit: int) -> list[dict[str, object]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def main() -> None:
    args = parse_args()
    if args.max_examples_per_cell < 0:
        raise SystemExit("--max-examples-per-cell must be non-negative")
    if args.max_catalog_values_per_cell < 0:
        raise SystemExit("--max-catalog-values-per-cell must be non-negative")
    items_sha256 = file_sha256(args.items_file)
    with args.items_file.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise RuntimeError("WebShop items file must be a JSON list")

    scenario_ids = tuple(args.scenarios)
    counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    examples: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    product_categories: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    queries: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    asins: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            counts["rejected_non_object"] += 1
            continue
        asin = str(item.get("asin") or "").upper()
        if not asin or asin in asins:
            counts["rejected_missing_or_duplicate_asin"] += 1
            continue
        asins.add(asin)
        record = {
            "Title": item.get("name"),
            "category": item.get("category"),
            "query": item.get("query"),
            "product_category": item.get("product_category"),
        }
        matches = classify_product_record(record, scenario_ids=scenario_ids)
        if not matches:
            counts["not_uniquely_classified"] += 1
            continue
        if len(matches) != 1:
            counts["ambiguous_across_cells"] += 1
            continue
        match = matches[0]
        cell = (match.scenario_id, match.slot_id, match.attribute_value)
        cell_counts[cell] += 1
        product_categories[cell][str(record["product_category"])] += 1
        queries[cell][str(record["query"])] += 1
        counts["uniquely_classified"] += 1
        if len(examples[cell]) < args.max_examples_per_cell:
            examples[cell].append(
                {
                    "asin": asin,
                    "title": str(record["Title"]),
                    "query": str(record["query"]),
                    "product_category": str(record["product_category"]),
                }
            )

    expected_cells = [
        (scenario_id, slot.slot_id, value.value_id)
        for scenario_id in scenario_ids
        for slot in scenario_by_id(scenario_id).slots
        for value in slot.values
    ]
    cells = {
        "/".join(cell): {
            "count": cell_counts[cell],
            "examples": examples[cell],
            "top_product_categories": _top_counts(
                product_categories[cell],
                limit=args.max_catalog_values_per_cell,
            ),
            "top_queries": _top_counts(
                queries[cell],
                limit=args.max_catalog_values_per_cell,
            ),
        }
        for cell in expected_cells
    }
    report = {
        "schema": "agentmemory_natural_attribute_catalog_scan_v1",
        "items_file": str(args.items_file.resolve()),
        "items_file_sha256": items_sha256,
        "scenario_definition_version": SCENARIO_DEFINITION_VERSION,
        "scenario_definition_sha256": SCENARIO_DEFINITION_SHA256,
        "scenario_ids": list(scenario_ids),
        "raw_product_count": len(items),
        "counts": dict(sorted(counts.items())),
        "cells": cells,
        "minimum_cell_count": min((cell_counts[cell] for cell in expected_cells), default=0),
        "all_cells_have_at_least_12_candidates": all(
            cell_counts[cell] >= 12 for cell in expected_cells
        ),
        "boundary": (
            "This scan proves deterministic title/category eligibility only. Native "
            "search, open, price, and purchase certification is a separate required gate."
        ),
    }
    output = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(
        "AGENTMEMORY_PROCEDURAL_CATALOG_SCANNED "
        f"products={len(items)} unique={counts['uniquely_classified']} "
        f"minimum_cell={report['minimum_cell_count']} output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
