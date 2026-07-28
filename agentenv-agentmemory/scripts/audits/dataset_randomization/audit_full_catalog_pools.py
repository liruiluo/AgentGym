#!/usr/bin/env python3
"""Stream-audit full MemoryArena WebShop catalog pools for structured generation.

Compatible with the Python 3.6 runtime on cpu9n. The scanner records counts and
bounded samples only; it does not materialize or modify catalog records.
"""

from __future__ import print_function

import argparse
import collections
import json
import os
import re
import sys
import time


USED_CHAIN_IDS = set([
    "baking_decoration_strict_aesthetic_6_step",
    "home_decor_living_room_style_harmony_6_step",
    "electronics_home_theater_tier_6_step",
    "grocery_flavor_profile_detailed_3_path_6_step",
    "beauty_skincare_routine_oily_vs_dry_6_step",
])


def iter_json_array(path, chunk_size=8 * 1024 * 1024):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            position = 0
            length = len(buffer)
            if not started:
                while position < length and buffer[position].isspace():
                    position += 1
                if position >= length:
                    if eof:
                        raise ValueError("empty catalog")
                    buffer = buffer[position:]
                    continue
                if buffer[position] != "[":
                    raise ValueError("catalog must be a JSON array")
                position += 1
                started = True

            need_more = False
            while True:
                while position < length and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position >= length:
                    need_more = True
                    break
                if buffer[position] == "]":
                    tail = buffer[position + 1:].strip()
                    if tail:
                        raise ValueError("trailing data after catalog array")
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except ValueError:
                    need_more = True
                    break
                if not isinstance(value, dict):
                    raise TypeError("catalog entry is not an object")
                yield value
                position = end

            buffer = buffer[position:]
            if eof:
                if need_more and buffer.strip():
                    raise ValueError("truncated or malformed catalog JSON")
                raise ValueError("catalog array missing closing bracket")


def listify(value):
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def parse_price_kind(value):
    if value is None or not value:
        return "missing_default_100", []
    if not isinstance(value, str):
        return "malformed", []
    values = []
    try:
        for fragment in value.split("$")[1:]:
            numeric = re.sub(r"[^\d.]", "", fragment)
            if numeric:
                values.append(float(numeric))
    except (TypeError, ValueError):
        return "malformed", []
    if not values:
        return "malformed", []
    if len(values) == 1:
        return "single", values[:1]
    return "range", values[:2]


def normalize_label(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def category_matches(canonical_category, product_category):
    """Match one canonical category and its descendants, not text prefixes."""
    canonical = str(canonical_category)
    category = str(product_category)
    return category == canonical or category.startswith(canonical + " › ")


def label_pattern(value):
    escaped = re.escape(str(value).strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE)


def build_specs(domain_data):
    specs = []
    for chain in domain_data:
        if chain.get("chain_id") not in USED_CHAIN_IDS:
            continue
        for step in chain.get("path") or []:
            ontology = set()
            allowed_labels = set()
            rejected_labels = set()
            for mapping_name in ("dependency_map", "reject_map"):
                for entry in step.get(mapping_name) or []:
                    for value in listify(entry[2]):
                        ontology.add(str(value).strip())
                        if mapping_name == "dependency_map":
                            allowed_labels.add(normalize_label(value))
                        else:
                            rejected_labels.add(normalize_label(value))
            specs.append({
                "chain_id": chain["chain_id"],
                "domain": chain.get("domain"),
                "step": int(step["step"]),
                "description": step.get("description"),
                "product_category": step.get("product_category"),
                "extract_pattern_text": step.get("extract_pattern"),
                "extract_pattern": re.compile(step.get("extract_pattern")),
                "ontology_patterns": [(normalize_label(value), label_pattern(value)) for value in sorted(ontology)],
                "allowed_labels": sorted(allowed_labels),
                "rejected_labels": sorted(rejected_labels),
                "dependency_map": step.get("dependency_map") or [],
                "reject_map": step.get("reject_map") or [],
            })
    return specs


def new_counter():
    return {
        "category_product_count": 0,
        "price_kind": collections.Counter(),
        "rating_present": 0,
        "extract_match_count": collections.Counter(),
        "ontology_match_count": collections.Counter(),
        "extract_primary_label": collections.Counter(),
        "ontology_label": collections.Counter(),
        "single_price_single_extract_label": collections.Counter(),
        "single_price_rating_single_extract_label": collections.Counter(),
        "single_price_semantic_label_set": collections.Counter(),
        "single_price_rating_semantic_label_set": collections.Counter(),
        "strict_single_ontology_label_single_price": collections.Counter(),
        "strict_single_ontology_label_single_price_rating": collections.Counter(),
        "samples": collections.defaultdict(list),
    }


def add_sample(counter, bucket, product, labels, limit):
    rows = counter["samples"][bucket]
    if len(rows) >= limit:
        return
    rows.append({
        "asin": product.get("asin"),
        "name": product.get("name"),
        "pricing": product.get("pricing"),
        "average_rating": product.get("average_rating"),
        "labels": labels,
    })


def finalize(counter):
    result = dict(counter)
    for key in (
        "price_kind",
        "extract_match_count",
        "ontology_match_count",
        "extract_primary_label",
        "ontology_label",
        "single_price_single_extract_label",
        "single_price_rating_single_extract_label",
        "single_price_semantic_label_set",
        "single_price_rating_semantic_label_set",
        "strict_single_ontology_label_single_price",
        "strict_single_ontology_label_single_price_rating",
    ):
        result[key] = dict(sorted(counter[key].items(), key=lambda row: str(row[0])))
    result["samples"] = dict(sorted(counter["samples"].items()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--domain-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    with open(args.domain_data, "r", encoding="utf-8") as handle:
        domain_data = json.load(handle)
    specs = build_specs(domain_data)
    by_category_root = collections.defaultdict(list)
    for spec in specs:
        root = str(spec["product_category"]).split(" › ", 1)[0]
        by_category_root[root].append(spec)
    counters = {(spec["chain_id"], spec["step"]): new_counter() for spec in specs}

    started = time.time()
    scanned = 0
    relevant = 0
    for product in iter_json_array(args.items):
        scanned += 1
        category = str(product.get("product_category") or "")
        category_root = category.split(" › ", 1)[0]
        category_specs = [
            spec
            for spec in by_category_root.get(category_root, ())
            if category_matches(spec["product_category"], category)
        ]
        if category_specs:
            relevant += 1
        title = str(product.get("name") or "")
        price_kind, _ = parse_price_kind(product.get("pricing"))
        rating_present = product.get("average_rating") not in (None, "")
        for spec in category_specs:
            counter = counters[(spec["chain_id"], spec["step"])]
            counter["category_product_count"] += 1
            counter["price_kind"][price_kind] += 1
            counter["rating_present"] += int(rating_present)

            extract_labels = []
            seen = set()
            for match in spec["extract_pattern"].finditer(title):
                label = normalize_label(match.group(0))
                if label not in seen:
                    seen.add(label)
                    extract_labels.append(label)
            ontology_labels = [label for label, pattern in spec["ontology_patterns"] if pattern.search(title)]
            semantic_labels = ontology_labels if spec["ontology_patterns"] else extract_labels
            counter["extract_match_count"][str(len(extract_labels))] += 1
            counter["ontology_match_count"][str(len(ontology_labels))] += 1
            if extract_labels:
                counter["extract_primary_label"][extract_labels[0]] += 1
            for label in ontology_labels:
                counter["ontology_label"][label] += 1

            if price_kind == "single" and len(extract_labels) == 1:
                counter["single_price_single_extract_label"][extract_labels[0]] += 1
                if rating_present:
                    counter["single_price_rating_single_extract_label"][extract_labels[0]] += 1
            if price_kind == "single" and semantic_labels:
                semantic_key = "|".join(sorted(semantic_labels))
                counter["single_price_semantic_label_set"][semantic_key] += 1
                if rating_present:
                    counter["single_price_rating_semantic_label_set"][semantic_key] += 1
            if price_kind == "single" and len(semantic_labels) == 1:
                label = semantic_labels[0]
                counter["strict_single_ontology_label_single_price"][label] += 1
                if rating_present:
                    counter["strict_single_ontology_label_single_price_rating"][label] += 1
                add_sample(counter, "strict:" + label, product, semantic_labels, args.sample_limit)
            elif extract_labels or semantic_labels:
                add_sample(counter, "rejected_multi_or_unresolved", product, semantic_labels, args.sample_limit)

        if args.progress_every and scanned % args.progress_every == 0:
            elapsed = time.time() - started
            print(
                "scanned={} relevant={} elapsed_s={:.1f} rate={:.0f}/s".format(
                    scanned, relevant, elapsed, scanned / max(elapsed, 1e-9)
                ),
                file=sys.stderr,
                flush=True,
            )

    rows = []
    for spec in specs:
        key = (spec["chain_id"], spec["step"])
        row = {
            "chain_id": spec["chain_id"],
            "domain": spec["domain"],
            "step": spec["step"],
            "description": spec["description"],
            "product_category": spec["product_category"],
            "extract_pattern": spec["extract_pattern_text"],
            "ontology_labels": [label for label, _ in spec["ontology_patterns"]],
            "allowed_target_labels": spec["allowed_labels"],
            "rejected_labels": spec["rejected_labels"],
            "counts": finalize(counters[key]),
        }
        rows.append(row)

    payload = {
        "schema": "memoryarena_full_catalog_pool_audit_v1",
        "items_path": os.path.abspath(args.items),
        "items_size_bytes": os.path.getsize(args.items),
        "domain_data_path": os.path.abspath(args.domain_data),
        "scanned_product_count": scanned,
        "relevant_product_occurrence_count": relevant,
        "elapsed_seconds": time.time() - started,
        "used_chain_ids": sorted(USED_CHAIN_IDS),
        "step_pool_count": len(rows),
        "steps": rows,
        "notes": [
            "product_category prefix match on canonical category boundaries.",
            "Strict pools require one ontology label and a single explicit price.",
            "Rating-strict pools additionally require average_rating.",
            "Search rank, exact option-page executability, and whole-chain uniqueness are not checked here.",
            "Counts are candidate-pool evidence, not generated-task clearance.",
        ],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote={} scanned={} elapsed_s={:.1f}".format(args.output, scanned, payload["elapsed_seconds"]))


if __name__ == "__main__":
    main()
