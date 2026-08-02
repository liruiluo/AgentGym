#!/usr/bin/env python3
"""Build deterministic latent-preference candidates from frozen WebShop."""

from __future__ import annotations

import argparse
import codecs
import collections
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


SCHEMA = "agentmemory_latent_preference_candidate_build_audit_v3"
CANDIDATE_SCHEMA = "agentmemory_latent_preference_rule_candidate_v2"
BUILDER_VERSION = "latent_preference_candidate_rules_v3"
ASIN_RE = re.compile(r"[A-Z0-9]{10}")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ValueRule:
    value_id: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class CategoryRule:
    category_id: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class AxisRule:
    rule_id: str
    axis: str
    values: tuple[ValueRule, ...]
    categories: tuple[CategoryRule, ...]


def _value(value_id: str, *patterns: str) -> ValueRule:
    return ValueRule(value_id=value_id, patterns=tuple(patterns))


def _category(category_id: str, *patterns: str) -> CategoryRule:
    return CategoryRule(category_id=category_id, patterns=tuple(patterns))


COLORS = (
    _value("black", r"\bblack\b"),
    _value("white", r"\bwhite\b"),
    _value("red", r"\bred\b"),
    _value("green", r"\bgreen\b"),
    _value("pink", r"\bpink\b"),
    _value("purple", r"\bpurple\b"),
    _value("beige", r"\bbeige\b"),
    _value("gray", r"\bgr[ae]y\b"),
)

TEXTILE_MATERIALS = (
    _value("cotton", r"\bcotton\b"),
    _value("polyester", r"\bpolyester\b"),
    _value("linen", r"\blinen\b"),
    _value("silk", r"\bsilk\b"),
    _value("velvet", r"\bvelvet\b"),
    _value("canvas", r"\bcanvas\b"),
)

PATTERNS = (
    _value("solid", r"\bsolid(?: color)?\b"),
    _value("striped", r"\bstrip(?:e|ed|es)\b"),
    _value("floral", r"\bfloral\b", r"\bflower print\b"),
    _value("plaid", r"\bplaid\b"),
    _value("geometric", r"\bgeometric\b"),
)

TEXTILE_CATEGORIES = (
    _category("shower_curtain", r"shower curtains?"),
    _category("window_curtain", r"curtains? & drapes", r"curtain panels?"),
    _category("pillowcase", r"pillowcases?", r"pillow covers?"),
    _category("duvet_cover", r"duvet cover sets?"),
    _category("comforter", r"comforters?(?: & sets)?"),
    _category("quilt", r"quilts?(?: & sets)?"),
    _category("bedspread", r"bedspreads?(?: & coverlets)?"),
    _category("blanket", r"bed blankets?", r"throw blankets?"),
    _category("tablecloth", r"tablecloths?"),
    _category("table_runner", r"table runners?"),
    _category("tote_bag", r"tote bags?"),
    _category("area_rug", r"area rugs?"),
    _category("apron", r"aprons?"),
)

FOOD_FLAVORS = (
    _value("chocolate", r"\bchocolate\b"),
    _value("vanilla", r"\bvanilla\b"),
    _value("strawberry", r"\bstrawberry\b"),
    _value("peanut_butter", r"\bpeanut butter\b"),
    _value("caramel", r"\bcaramel\b"),
)

DIETARY_PROFILES = (
    _value("gluten_free", r"\bgluten[ -]?free\b"),
    _value("sugar_free", r"\bsugar[ -]?free\b"),
    _value("organic", r"\borganic\b"),
    _value("vegan", r"\bvegan\b"),
    _value("keto", r"\bketo(?:genic)?\b"),
)

FOOD_CATEGORIES = (
    _category("protein_powder", r"protein powders?"),
    _category("nutrition_bar", r"nutrition bars?", r"protein bars?"),
    _category("cake_mix", r"baking mixes? .+ cakes?", r"cake mixes?"),
    _category("pancake_mix", r"pancake & waffle mixes?", r"pancake mixes?"),
    _category("cookies", r"breads? & bakery .+ cookies?", r"cookies?$"),
    _category("pudding", r"pudding(?: & gelatin)?"),
    _category("coffee_creamer", r"coffee creamers?"),
    _category("breakfast_cereal", r"breakfast cereals?"),
    _category("granola", r"granola"),
    _category("drink_mix", r"powdered drink mixes?", r"drink mixes?"),
)

SCENTS = (
    _value("fragrance_free", r"\bfragrance[ -]?free\b", r"\bunscented\b"),
    _value("lavender", r"\blavender\b"),
    _value("coconut", r"\bcoconut\b"),
    _value("vanilla", r"\bvanilla\b"),
    _value("rose", r"\brose(?: scented)?\b"),
    _value("citrus", r"\bcitrus\b", r"\blemon(?: scented)?\b"),
)

CARE_INGREDIENTS = (
    _value("aloe", r"\baloe(?: vera)?\b"),
    _value("shea_butter", r"\bshea butter\b"),
    _value("oatmeal", r"\boatmeal\b", r"\bcolloidal oat\b"),
    _value("argan_oil", r"\bargan oil\b"),
    _value("hyaluronic_acid", r"\bhyaluronic acid\b"),
)

CARE_CATEGORIES = (
    _category("hand_soap", r"hand soaps?"),
    _category("body_wash", r"body washes?"),
    _category("body_lotion", r"body lotions?", r"body .+ lotions?"),
    _category("hand_cream", r"hand creams?", r"hand lotions?"),
    _category("shampoo", r"shampoos?$"),
    _category("conditioner", r"conditioners?$"),
    _category("laundry_detergent", r"laundry detergents?"),
    _category("dish_soap", r"dish soaps?", r"dishwashing liquids?"),
)

ACCESSORY_MATERIALS = (
    _value("leather", r"\b(?:genuine |faux |pu )?leather\b"),
    _value("canvas", r"\bcanvas\b"),
    _value("nylon", r"\bnylon\b"),
    _value("silicone", r"\bsilicone\b"),
    _value("metal", r"\bmetal\b", r"\bstainless steel\b"),
)

ACCESSORY_CATEGORIES = (
    _category("tote_bag", r"tote bags?"),
    _category("handbag", r"handbags?"),
    _category("backpack", r"backpacks?"),
    _category("wallet", r"wallets?"),
    _category("laptop_sleeve", r"laptop sleeves?", r"laptop bags?"),
    _category("phone_case", r"cell phone cases?", r"basic cases?"),
    _category("watch_band", r"watch bands?"),
    _category("belt", r"belts?$"),
)

RULES = (
    AxisRule("textile_color", "color", COLORS, TEXTILE_CATEGORIES),
    AxisRule("textile_material", "material", TEXTILE_MATERIALS, TEXTILE_CATEGORIES),
    AxisRule("textile_pattern", "pattern", PATTERNS, TEXTILE_CATEGORIES),
    AxisRule("food_flavor", "flavor", FOOD_FLAVORS, FOOD_CATEGORIES),
    AxisRule(
        "food_dietary_profile",
        "dietary_profile",
        DIETARY_PROFILES,
        FOOD_CATEGORIES,
    ),
    AxisRule("care_scent", "scent", SCENTS, CARE_CATEGORIES),
    AxisRule("care_ingredient", "ingredient", CARE_INGREDIENTS, CARE_CATEGORIES),
    AxisRule("accessory_color", "color", COLORS, ACCESSORY_CATEGORIES),
    AxisRule(
        "accessory_material",
        "material",
        ACCESSORY_MATERIALS,
        ACCESSORY_CATEGORIES,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-candidates", required=True, type=Path)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", str(value or ""))
    ).strip()


def normalize_title(value: object) -> str:
    return normalize_text(value).casefold()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(
    handle: BinaryIO,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterable[Any]:
    """Incrementally decode one top-level JSON array using only stdlib."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    utf8_decoder = codecs.getincrementaldecoder("utf-8")("strict")
    json_decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    started = False
    finished = False
    expecting_value = True
    allow_array_end = True
    item_index = 0

    def read_more() -> bool:
        nonlocal buffer, eof
        raw = handle.read(chunk_size)
        if raw:
            buffer += utf8_decoder.decode(raw, final=False)
            return True
        if not eof:
            buffer += utf8_decoder.decode(b"", final=True)
            eof = True
        return False

    def skip_whitespace() -> None:
        nonlocal position
        while position < len(buffer) and buffer[position].isspace():
            position += 1

    while True:
        if position >= chunk_size:
            buffer = buffer[position:]
            position = 0

        skip_whitespace()

        if finished:
            if position < len(buffer):
                raise ValueError("trailing data after top-level JSON array")
            if eof:
                return
            read_more()
            continue

        if not started:
            if position == len(buffer):
                if eof:
                    raise ValueError("expected a top-level JSON array, found EOF")
                read_more()
                continue
            if buffer[position] != "[":
                raise ValueError("expected a top-level JSON array")
            position += 1
            started = True
            continue

        if position == len(buffer):
            if eof:
                raise ValueError("unterminated top-level JSON array")
            read_more()
            continue

        if expecting_value:
            if buffer[position] == "]":
                if not allow_array_end:
                    raise ValueError("trailing comma in top-level JSON array")
                position += 1
                finished = True
                continue
            try:
                value, end = json_decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(
                        f"invalid JSON array item {item_index}: {error.msg}"
                    ) from error
                read_more()
                continue
            position = end
            item_index += 1
            expecting_value = False
            allow_array_end = True
            yield value
            continue

        if buffer[position] == ",":
            position += 1
            expecting_value = True
            allow_array_end = False
            continue
        if buffer[position] == "]":
            position += 1
            finished = True
            continue
        raise ValueError(
            f"expected ',' or ']' after JSON array item {item_index - 1}"
        )


def iter_records(handle: BinaryIO) -> Iterable[Mapping[str, Any]]:
    for index, record in enumerate(iter_json_array(handle)):
        if not isinstance(record, Mapping):
            raise ValueError(f"catalog record {index} is not a JSON object")
        yield record


def category_for(rule: AxisRule, product_category: str) -> str | None:
    folded = product_category.casefold()
    for category in rule.categories:
        if any(re.search(pattern, folded) for pattern in category.patterns):
            return category.category_id
    return None


def matched_value(
    title: str,
    values: Sequence[ValueRule],
) -> tuple[str | None, tuple[str, ...], int]:
    matches: list[tuple[str, tuple[str, ...]]] = []
    for value in values:
        evidence = tuple(
            sorted(
                {
                    match.group(0)
                    for pattern in value.patterns
                    for match in re.finditer(pattern, title, flags=re.IGNORECASE)
                },
                key=lambda value: (value.casefold(), value),
            )
        )
        if evidence:
            matches.append((value.value_id, evidence))
    if len(matches) != 1:
        return None, (), len(matches)
    return matches[0][0], matches[0][1], 1


def record_asin(record: Mapping[str, Any]) -> str:
    raw = record.get("asin")
    if raw is None and isinstance(record.get("product_information"), Mapping):
        raw = record["product_information"].get("ASIN")
    return str(raw or "").upper()


def scan_candidates(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    with path.open("rb") as handle:
        for record in iter_records(handle):
            counts["catalog_records"] += 1
            asin = record_asin(record)
            if not ASIN_RE.fullmatch(asin):
                counts["rejected_invalid_asin"] += 1
                continue
            title = normalize_text(record.get("name"))
            product_category = normalize_text(record.get("product_category"))
            if not 8 <= len(title) <= 220 or not product_category:
                counts["rejected_title_or_category"] += 1
                continue
            normalized_title = normalize_title(title)
            matched_any_category = False
            for rule in RULES:
                category_id = category_for(rule, product_category)
                if category_id is None:
                    continue
                matched_any_category = True
                value, evidence, match_count = matched_value(title, rule.values)
                if match_count != 1:
                    counts[f"{rule.rule_id}:rejected_axis_match_{match_count}"] += 1
                    continue
                assert value is not None
                semantic = {
                    "category_id": category_id,
                    "axis": rule.axis,
                    "attribute_value": value,
                    "asin": asin,
                    "title": title,
                    "product_category": product_category,
                    "title_evidence": list(evidence),
                }
                semantic["classification_sha256"] = canonical_sha256(semantic)
                semantic["normalized_title"] = normalized_title
                candidates.append(semantic)
                counts[f"{rule.rule_id}:rule_candidates"] += 1
            if matched_any_category:
                counts["records_in_candidate_categories"] += 1
    return candidates, dict(sorted(counts.items()))


def catalog_title_counts(path: Path, wanted: set[str]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    with path.open("rb") as handle:
        for record in iter_records(handle):
            normalized = normalize_title(record.get("name"))
            if normalized in wanted:
                counts[normalized] += 1
    return dict(counts)


def valid_candidates(
    candidates: Sequence[dict[str, Any]],
    title_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejection_counts: collections.Counter[str] = collections.Counter()
    title_unique: list[dict[str, Any]] = []
    for item in candidates:
        if title_counts.get(item["normalized_title"], 0) != 1:
            rejection_counts["nonunique_catalog_title"] += 1
            continue
        title_unique.append(dict(item))

    assignments: dict[tuple[str, str], set[tuple[str, str]]] = (
        collections.defaultdict(set)
    )
    canonical_edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in title_unique:
        asin_axis = (str(item["asin"]), str(item["axis"]))
        cell = (str(item["category_id"]), str(item["attribute_value"]))
        assignments[asin_axis].add(cell)
        edge = (*asin_axis, *cell)
        prior = canonical_edges.get(edge)
        if prior is None:
            canonical_edges[edge] = item
        else:
            rejection_counts["duplicate_equivalent_axis_edge"] += 1
            if item["classification_sha256"] < prior["classification_sha256"]:
                canonical_edges[edge] = item

    valid: list[dict[str, Any]] = []
    for item in canonical_edges.values():
        asin_axis = (str(item["asin"]), str(item["axis"]))
        if len(assignments[asin_axis]) != 1:
            rejection_counts["asin_axis_matches_multiple_rule_cells"] += 1
            continue
        valid.append(item)
    valid.sort(
        key=lambda item: (
            item["axis"],
            item["category_id"],
            item["attribute_value"],
            item["classification_sha256"],
            item["asin"],
        )
    )
    return valid, dict(sorted(rejection_counts.items()))


def axis_candidate_counts(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = collections.Counter(str(item["axis"]) for item in candidates)
    return dict(sorted(counts.items()))


def write_candidates(path: Path, candidates: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in candidates:
            payload = {"schema": CANDIDATE_SCHEMA, **item}
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    if not args.items_file.is_file():
        raise SystemExit(f"items file does not exist: {args.items_file}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.catalog_sha256):
        raise SystemExit("--catalog-sha256 must be a lowercase SHA256")
    observed_catalog_sha256 = file_sha256(args.items_file)
    if observed_catalog_sha256 != args.catalog_sha256:
        raise SystemExit(
            "catalog SHA256 mismatch: "
            f"expected {args.catalog_sha256}, observed {observed_catalog_sha256}"
        )

    candidates, scan_counts = scan_candidates(args.items_file)
    title_counts = catalog_title_counts(
        args.items_file,
        {item["normalized_title"] for item in candidates},
    )
    valid, rejection_counts = valid_candidates(candidates, title_counts)
    write_candidates(args.output_candidates, valid)
    report = {
        "schema": SCHEMA,
        "input": {
            "catalog_sha256": args.catalog_sha256,
        },
        "rules_sha256": canonical_sha256(
            {
                "builder_version": BUILDER_VERSION,
                "evidence_ordering": "unicode_casefold_then_codepoint_v1",
                "axis_rules": [
                    {
                        "rule_id": rule.rule_id,
                        "axis": rule.axis,
                        "values": [value.__dict__ for value in rule.values],
                        "categories": [
                            category.__dict__ for category in rule.categories
                        ],
                    }
                    for rule in RULES
                ],
            }
        ),
        "builder_version": BUILDER_VERSION,
        "scan_counts": scan_counts,
        "postscan_rejections": rejection_counts,
        "globally_title_unique_rule_candidate_count": len(valid),
        "candidate_artifact": {
            "sha256": file_sha256(args.output_candidates),
            "rows": len(valid),
        },
        "axis_candidate_counts": axis_candidate_counts(valid),
        "verification_scope": {
            "rules_only": True,
            "independent_axis_candidate_rows": True,
            "global_normalized_title_uniqueness": True,
            "candidate_rows_are_machine_generated": True,
            "native_search_certified": False,
            "native_open_certified": False,
            "native_purchase_certified": False,
            "human_review_required": False,
            "llm_judge_required": False,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "AGENTMEMORY_LATENT_PREFERENCE_CANDIDATES_BUILT "
        f"candidates={len(valid)} "
        f"candidate_sha256={file_sha256(args.output_candidates)} "
        "human_review_required=false llm_judge_required=false"
    )


if __name__ == "__main__":
    main()
