#!/usr/bin/env python3
"""Audit MemoryArena bundled-shopping structure, reuse, leakage, and label evidence.

This script is deliberately read-only with respect to the source dataset. It
does not generate variants or bless provisional candidate-to-ASIN mappings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEPARATOR = "-" * 64
OPTIONS_MARKER = "**Available Options:**"
CATEGORY_RE = re.compile(r"^(?P<family>[a-z_]+)_item_(?P<index>\d+)$")
BUDGET_RE = re.compile(r"(\*\*Total Budget:\*\*\s*All items combined must not exceed\s*)\$[\d,.]+")
PRODUCT_RE = re.compile(r"Product\s+(\d+):", re.IGNORECASE)
METRIC_RE = re.compile(r"\b(highest|lowest)[- ](priced|rated|price|rating)\b", re.IGNORECASE)
FIELD_RE = {
    "title": re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE),
    "goal": re.compile(r"^\*\*Goal:\*\*\s*(.+?)\s*$", re.MULTILINE),
    "preference": re.compile(r"^\*\*Preference:\*\*\s*(.+?)\s*$", re.MULTILINE),
    "avoid": re.compile(r"^\*\*Avoid:\*\*\s*(.+?)\s*$", re.MULTILINE),
    "constraint": re.compile(r"^\*\*Constraint:\*\*\s*(.+?)\s*$", re.MULTILINE),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_split(source_id: int) -> str:
    bucket = source_id % 10
    return "train" if bucket < 8 else "dev" if bucket == 8 else "test"


def family_of(category: str) -> str:
    match = CATEGORY_RE.fullmatch(category)
    if not match:
        raise ValueError(f"unexpected category: {category!r}")
    return match.group("family")


def distribution(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items(), key=lambda row: str(row[0]))}


def quantiles(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": statistics.median(ordered),
        "p75": percentile(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def entropy_bits(counts: Mapping[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def field(text: str, name: str) -> str | None:
    match = FIELD_RE[name].search(text)
    return match.group(1).strip() if match else None


def normalize_budget(text: str) -> str:
    return BUDGET_RE.sub(r"\1$<BUDGET>", text)


def normalize_product_number(text: str) -> str:
    return PRODUCT_RE.sub("Product <STEP>:", text)


def structural_instruction(instruction: str) -> str:
    text = normalize_product_number(instruction)
    for name, pattern in FIELD_RE.items():
        text = pattern.sub(lambda _: f"**<{name.upper()}>**", text)
    return text.strip()


@dataclass(frozen=True)
class ParsedQuestion:
    preamble: str
    instruction: str
    options: tuple[str, ...]
    title: str | None
    goal: str | None
    preference: str | None
    avoid: str | None
    constraint: str | None
    metrics: tuple[str, ...]
    budget: float


def parse_question(question: str) -> ParsedQuestion:
    if question.count(SEPARATOR) != 1:
        raise ValueError("question does not contain exactly one canonical separator")
    preamble, section = question.split(SEPARATOR, maxsplit=1)
    if section.count(OPTIONS_MARKER) != 1:
        raise ValueError("question does not contain exactly one options marker")
    instruction, option_tail = section.split(OPTIONS_MARKER, maxsplit=1)
    options = tuple(
        line.strip()[2:].strip()
        for line in option_tail.splitlines()
        if line.strip().startswith("- ")
    )
    if not options:
        raise ValueError("question has no options")
    budget_match = re.search(r"\*\*Total Budget:\*\*.*?\$([\d,.]+)", preamble)
    if not budget_match:
        raise ValueError("question lacks budget")
    metrics = tuple(
        f"{direction.lower()}-{'price' if noun.lower() in {'priced', 'price'} else 'rating'}"
        for direction, noun in METRIC_RE.findall(instruction)
    )
    return ParsedQuestion(
        preamble=preamble.strip(),
        instruction=instruction.strip(),
        options=options,
        title=field(instruction, "title"),
        goal=field(instruction, "goal"),
        preference=field(instruction, "preference"),
        avoid=field(instruction, "avoid"),
        constraint=field(instruction, "constraint"),
        metrics=metrics,
        budget=float(budget_match.group(1).replace(",", "")),
    )


class UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def shared_component_report(
    rows: Sequence[Mapping[str, Any]], values_by_id: Mapping[int, set[str]]
) -> dict[str, Any]:
    ids = [int(row["id"]) for row in rows]
    uf = UnionFind(ids)
    owners: dict[str, int] = {}
    for source_id in ids:
        for value in values_by_id[source_id]:
            if value in owners:
                uf.union(source_id, owners[value])
            else:
                owners[value] = source_id
    components: dict[int, list[int]] = defaultdict(list)
    for source_id in ids:
        components[uf.find(source_id)].append(source_id)
    spanning = [
        members
        for members in components.values()
        if len({assign_split(source_id) for source_id in members}) > 1
    ]
    heldout = [source_id for source_id in ids if assign_split(source_id) != "train"]
    connected_to_train = [
        source_id
        for source_id in heldout
        if any(assign_split(peer) == "train" for peer in components[uf.find(source_id)])
    ]
    return {
        "component_count": len(components),
        "component_size": quantiles([len(members) for members in components.values()]),
        "cross_split_component_count": len(spanning),
        "bundles_in_cross_split_components": sum(len(members) for members in spanning),
        "heldout_bundle_count": len(heldout),
        "heldout_bundles_connected_to_train": len(connected_to_train),
        "heldout_connected_fraction": len(connected_to_train) / len(heldout) if heldout else 0.0,
    }


def overlap_report(values_by_split: Mapping[str, set[str]]) -> dict[str, Any]:
    train = values_by_split["train"]
    result: dict[str, Any] = {"unique_by_split": {split: len(values) for split, values in values_by_split.items()}}
    for split in ("dev", "test"):
        heldout = values_by_split[split]
        shared = train & heldout
        result[f"train_{split}_shared_unique"] = len(shared)
        result[f"{split}_unique_seen_in_train_fraction"] = len(shared) / len(heldout) if heldout else 0.0
    shared_all = set.intersection(*values_by_split.values())
    result["shared_across_all_splits"] = len(shared_all)
    return result


def majority_accuracy(records: Sequence[Mapping[str, Any]], key_names: Sequence[str]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    for row in records:
        groups[tuple(row[name] for name in key_names)][int(row["target_position"])] += 1
    correct = sum(max(counts.values()) for counts in groups.values())
    return {
        "features": list(key_names),
        "group_count": len(groups),
        "accuracy": correct / len(records) if records else 0.0,
        "correct": correct,
        "total": len(records),
    }


def role_dictionary_baseline(
    sessions: Sequence[Mapping[str, Any]], heldout_split: str
) -> dict[str, Any]:
    train_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in sessions:
        if row["split"] != "train":
            continue
        target = int(row["target_position"])
        for index, option in enumerate(row["options"]):
            train_counts[str(option)][0] += int(index == target)
            train_counts[str(option)][1] += 1
    total = covered = correct = ambiguous = unique_predictions = unique_correct = 0
    for row in sessions:
        if row["split"] != heldout_split:
            continue
        scores: list[float | None] = []
        for option in row["options"]:
            counts = train_counts.get(str(option))
            scores.append(counts[0] / counts[1] if counts else None)
        total += 1
        known = [(index, score) for index, score in enumerate(scores) if score is not None]
        if not known:
            continue
        covered += 1
        best = max(score for _, score in known)
        predictions = [index for index, score in known if score == best]
        if len(predictions) != 1:
            ambiguous += 1
        else:
            unique_predictions += 1
            unique_correct += int(predictions[0] == int(row["target_position"]))
        if int(row["target_position"]) in predictions:
            correct += 1
    return {
        "split": heldout_split,
        "total_sessions": total,
        "covered_sessions": covered,
        "coverage": covered / total if total else 0.0,
        "target_in_argmax_count": correct,
        "target_in_argmax_accuracy_on_covered": correct / covered if covered else 0.0,
        "tied_argmax_sessions": ambiguous,
        "unique_argmax_sessions": unique_predictions,
        "unique_argmax_correct": unique_correct,
        "unique_argmax_accuracy": unique_correct / unique_predictions if unique_predictions else 0.0,
        "note": "Uses provisional target-option indices; this is a leakage diagnostic, not label proof.",
    }


def dependency_summary(domain_data: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions = []
    counterfactual_edges = []
    for key, chain in enumerate(domain_data):
        path = chain.get("path") or []
        dependency_steps = 0
        source_lags: list[int] = []
        for step in path:
            current = int(step["step"])
            entries = step.get("dependency_map") or []
            if entries:
                dependency_steps += 1
            grouped: dict[int, dict[str, set[str]]] = defaultdict(dict)
            for source_step, source_label, allowed in entries:
                allowed_values = {str(value) for value in (allowed if isinstance(allowed, list) else [allowed])}
                grouped[int(source_step)][str(source_label)] = allowed_values
                source_lags.append(current - int(source_step))
            for source_step, mapping in grouped.items():
                distinct_allowed = {tuple(sorted(values)) for values in mapping.values()}
                if len(mapping) >= 2 and len(distinct_allowed) >= 2:
                    counterfactual_edges.append(
                        {
                            "chain_id": chain.get("chain_id"),
                            "current_step": current,
                            "source_step": source_step,
                            "source_label_count": len(mapping),
                            "distinct_allowed_sets": len(distinct_allowed),
                        }
                    )
        definitions.append(
            {
                "index": key,
                "chain_id": chain.get("chain_id"),
                "domain": chain.get("domain"),
                "step_count": len(path),
                "dependency_step_count": dependency_steps,
                "source_lag_distribution": distribution(source_lags),
            }
        )
    return {
        "definition_count": len(definitions),
        "step_count_distribution": distribution(row["step_count"] for row in definitions),
        "six_step_definition_count": sum(row["step_count"] == 6 for row in definitions),
        "five_step_definition_count": sum(row["step_count"] == 5 for row in definitions),
        "definitions": definitions,
        "structurally_counterfactual_dependency_edges": counterfactual_edges,
        "structurally_counterfactual_dependency_edge_count": len(counterfactual_edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--whole-steps", type=Path, required=True)
    parser.add_argument("--semantic-steps", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    raw_path = root / "raw.jsonl"
    domain_path = root / "domain_data.json"
    chain_path = root / "chains.jsonl"
    chain_summary_path = root / "chain_summary.json"
    annotation_summary_path = root / "annotation_summary.json"
    raw = read_jsonl(raw_path)
    domain_data = read_json(domain_path)
    whole_chains = read_jsonl(chain_path)
    whole_steps = read_jsonl(args.whole_steps.resolve())
    semantic_steps = read_jsonl(args.semantic_steps.resolve())
    chain_summary = read_json(chain_summary_path)
    annotation_summary = read_json(annotation_summary_path)

    whole_by_key = {(int(row["source_id"]), int(row["step_index"])): row for row in whole_steps}
    semantic_by_key = {(int(row["source_id"]), int(row["step_index"])): row for row in semantic_steps}
    chain_by_id = {int(row["source_id"]): row for row in whole_chains}
    if len(whole_by_key) != 900 or len(semantic_by_key) != 900 or len(chain_by_id) != 150:
        raise ValueError("expected 900 whole/semantic steps and 150 whole chains")

    sessions: list[dict[str, Any]] = []
    session_budgets: list[float] = []
    bundle_budgets: list[float] = []
    target_asins_by_id: dict[int, set[str]] = {}
    option_texts_by_id: dict[int, set[str]] = {}
    target_sequences: dict[int, tuple[str, ...]] = {}
    for row in raw:
        source_id = int(row["id"])
        category = str(row["category"])
        family = family_of(category)
        split = assign_split(source_id)
        answers = row["answers"]
        bundle_budgets.append(parse_question(str(row["questions"][0])).budget)
        target_sequences[source_id] = tuple(str(answer["target_asin"]).upper() for answer in answers)
        target_asins_by_id[source_id] = set(target_sequences[source_id])
        option_texts_by_id[source_id] = set()
        for zero_index, (question, answer) in enumerate(zip(row["questions"], answers)):
            parsed = parse_question(str(question))
            step_index = zero_index + 1
            whole = whole_by_key[(source_id, step_index)]
            target_position = int(whole["target"]["option_index"])
            if not 0 <= target_position < len(parsed.options):
                raise ValueError(f"target position out of range for {source_id}/{step_index}")
            option_texts_by_id[source_id].update(parsed.options)
            session_budgets.append(parsed.budget)
            normalized_preamble = normalize_budget(parsed.preamble)
            semantic_template = normalize_product_number(parsed.instruction)
            question_no_options = normalize_product_number(normalize_budget(parsed.preamble + "\n" + SEPARATOR + parsed.instruction))
            candidate_set = tuple(sorted(parsed.options))
            sessions.append(
                {
                    "source_id": source_id,
                    "category": category,
                    "family": family,
                    "split": split,
                    "step_index": step_index,
                    "question": question,
                    "question_chars": len(question),
                    "question_words": len(re.findall(r"\S+", question)),
                    "budget": parsed.budget,
                    "title": parsed.title,
                    "goal": parsed.goal,
                    "preference": parsed.preference,
                    "avoid_present": parsed.avoid is not None,
                    "constraint_present": parsed.constraint is not None,
                    "metrics": parsed.metrics,
                    "metric_signature": "+".join(parsed.metrics),
                    "options": parsed.options,
                    "option_count": len(parsed.options),
                    "candidate_set_hash": stable_hash(candidate_set),
                    "candidate_order_hash": stable_hash(parsed.options),
                    "normalized_preamble_hash": stable_hash(normalized_preamble),
                    "semantic_template_hash": stable_hash(semantic_template),
                    "structural_template_hash": stable_hash(structural_instruction(parsed.instruction)),
                    "question_no_options_hash": stable_hash(question_no_options),
                    "target_asin": str(answer["target_asin"]).upper(),
                    "attribute_count": len(answer.get("attributes") or []),
                    "target_position": target_position,
                    "target_option_text": parsed.options[target_position],
                    "target_alignment_status": whole["target"]["alignment"]["status"],
                    "target_label": whole["target"]["label_evidence"]["first_label"],
                    "target_distinct_label_count": len(whole["target"]["label_evidence"]["all_distinct_labels"]),
                    "effective_metric": f"{whole['metric']['direction']}-{whole['metric']['field']}",
                    "target_effective_metric_missing": whole["target"]["official_metadata"].get(whole["metric"]["field"]) in (None, ""),
                    "chain_status": chain_by_id[source_id]["status"],
                }
            )

    family_counts = Counter(family_of(str(row["category"])) for row in raw)
    split_counts = Counter(assign_split(int(row["id"])) for row in raw)
    provisional = [row for row in raw if chain_by_id[int(row["id"])]["status"] in {"pass", "unknown"}]
    provisional_split_counts = Counter(assign_split(int(row["id"])) for row in provisional)

    value_fields = {
        "target_asin": defaultdict(set),
        "option_text": defaultdict(set),
        "candidate_set": defaultdict(set),
        "semantic_template": defaultdict(set),
        "question_no_options": defaultdict(set),
    }
    for row in sessions:
        split = str(row["split"])
        value_fields["target_asin"][split].add(str(row["target_asin"]))
        value_fields["option_text"][split].update(str(value) for value in row["options"])
        value_fields["candidate_set"][split].add(str(row["candidate_set_hash"]))
        value_fields["semantic_template"][split].add(str(row["semantic_template_hash"]))
        value_fields["question_no_options"][split].add(str(row["question_no_options_hash"]))

    option_roles: dict[str, Counter[str]] = defaultdict(Counter)
    option_splits: dict[str, set[str]] = defaultdict(set)
    for row in sessions:
        target = int(row["target_position"])
        for index, option in enumerate(row["options"]):
            option_roles[str(option)]["target" if index == target else "distractor"] += 1
            option_splits[str(option)].add(str(row["split"]))

    target_frequencies = Counter(str(row["target_asin"]) for row in sessions)
    option_frequencies = Counter(option for row in sessions for option in row["options"])
    target_position_records = [row for row in sessions]
    alignment_pass_records = [row for row in sessions if row["target_alignment_status"] == "pass"]

    ngram_overlap: dict[str, Any] = {}
    for width in (1, 2, 3, 6):
        by_split: dict[str, set[tuple[str, ...]]] = {split: set() for split in ("train", "dev", "test")}
        for source_id, sequence in target_sequences.items():
            split = assign_split(source_id)
            for start in range(0, len(sequence) - width + 1):
                by_split[split].add(sequence[start : start + width])
        ngram_overlap[str(width)] = overlap_report(by_split)

    exact_current_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        if int(row["step_index"]) == 1:
            continue
        exact_current_groups[stable_hash([row["question"], row["options"]])].append(row)
    natural_counterfactual_groups = []
    for group in exact_current_groups.values():
        if len(group) < 2:
            continue
        previous = {target_sequences[int(row["source_id"])][int(row["step_index"]) - 2] for row in group}
        targets = {str(row["target_asin"]) for row in group}
        if len(previous) >= 2 and len(targets) >= 2:
            natural_counterfactual_groups.append(group)

    candidate_resolution_status = Counter()
    candidate_confidence = Counter()
    candidate_asin_present_count = 0
    non_target_candidate_asin_present_count = 0
    total_visible_candidates = 0
    for row in whole_steps:
        for candidate in row["candidates"]:
            total_visible_candidates += 1
            resolution = candidate["resolution"]
            candidate_resolution_status[str(resolution["status"])] += 1
            if resolution.get("asin"):
                candidate_asin_present_count += 1
                if not candidate["is_target"]:
                    non_target_candidate_asin_present_count += 1
    for row in semantic_steps:
        for candidate in row["candidate_records"]:
            candidate_confidence[str(candidate["metadata"]["confidence"])] += 1

    discrepancy_examples: list[dict[str, Any]] = []
    discrepancy_counts: dict[str, Counter[str]] = {
        "compatibility": Counter(),
        "ranking": Counter(),
        "bundle_budget": Counter(),
    }
    semantic_key = {
        "compatibility": "cross_session_compatibility",
        "ranking": "ranking",
        "bundle_budget": "bundle_budget",
    }
    for key, whole in whole_by_key.items():
        semantic = semantic_by_key[key]
        for check in discrepancy_counts:
            pair = f"semantic={semantic['checks'][semantic_key[check]]}|whole={whole['checks'][check]}"
            discrepancy_counts[check][pair] += 1
            if semantic["checks"][semantic_key[check]] != whole["checks"][check] and len(discrepancy_examples) < 30:
                discrepancy_examples.append(
                    {
                        "source_id": key[0],
                        "step_index": key[1],
                        "check": check,
                        "semantic": semantic["checks"][semantic_key[check]],
                        "whole": whole["checks"][check],
                        "target_asin": whole["target"]["asin"],
                        "whole_target_label": whole["target"]["label_evidence"]["first_label"],
                        "semantic_target_reason": (
                            semantic.get("candidate_compatibility", {})
                            .get(semantic.get("target_product_id"), {})
                            .get("reason")
                        ),
                    }
                )

    pool_by_family_step_label: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    fixed_price_target_asins: set[str] = set()
    range_price_target_asins: set[str] = set()
    missing_rating_target_asins: set[str] = set()
    for row in whole_steps:
        family = family_of(str(row["category"]))
        label = str(row["target"]["label_evidence"]["first_label"]).casefold()
        asin = str(row["target"]["asin"])
        pool_by_family_step_label[(family, int(row["step_index"]), label)].add(asin)
        metadata = row["target"]["official_metadata"]
        if metadata.get("price_semantics") == "webshop_runtime_random_uniform_range":
            range_price_target_asins.add(asin)
        else:
            fixed_price_target_asins.add(asin)
        if metadata.get("average_rating") in (None, ""):
            missing_rating_target_asins.add(asin)

    pool_rows = [
        {
            "family": key[0],
            "step_index": key[1],
            "label": key[2],
            "unique_target_asins": len(values),
        }
        for key, values in sorted(pool_by_family_step_label.items())
    ]
    family_step_reuse = []
    for family in sorted(family_counts):
        for step_index in range(1, 7):
            group = [
                row for row in sessions
                if row["family"] == family and int(row["step_index"]) == step_index
            ]
            group_options = [str(option) for row in group for option in row["options"]]
            group_roles: dict[str, Counter[str]] = defaultdict(Counter)
            label_counts = Counter(str(row["target_label"]).casefold() for row in group)
            for row in group:
                target = int(row["target_position"])
                for index, option in enumerate(row["options"]):
                    group_roles[str(option)]["target" if index == target else "distractor"] += 1
            family_step_reuse.append(
                {
                    "family": family,
                    "step_index": step_index,
                    "session_count": len(group),
                    "option_occurrences": len(group_options),
                    "unique_option_texts": len(set(group_options)),
                    "option_reuse_factor": len(group_options) / len(set(group_options)),
                    "max_option_frequency": max(Counter(group_options).values()),
                    "unique_target_asins": len({row["target_asin"] for row in group}),
                    "unique_target_option_texts_provisional": len({row["target_option_text"] for row in group}),
                    "target_only_option_texts_provisional": sum(
                        counts["target"] > 0 and counts["distractor"] == 0
                        for counts in group_roles.values()
                    ),
                    "distractor_only_option_texts_provisional": sum(
                        counts["target"] == 0 and counts["distractor"] > 0
                        for counts in group_roles.values()
                    ),
                    "role_switched_option_texts_provisional": sum(
                        counts["target"] > 0 and counts["distractor"] > 0
                        for counts in group_roles.values()
                    ),
                    "target_position_distribution_provisional": distribution(
                        row["target_position"] for row in group
                    ),
                    "target_label_distribution": dict(sorted(label_counts.items())),
                    "target_label_count": len(label_counts),
                    "target_label_majority_fraction": max(label_counts.values()) / len(group),
                    "target_label_entropy_bits": entropy_bits(label_counts),
                }
            )
    semantic_paths: dict[int, tuple[str, ...]] = {}
    for row in sessions:
        semantic_paths.setdefault(int(row["source_id"]), tuple())
    for source_id in semantic_paths:
        semantic_paths[source_id] = tuple(
            str(row["target_label"]).casefold()
            for row in sorted(
                (item for item in sessions if int(item["source_id"]) == source_id),
                key=lambda item: int(item["step_index"]),
            )
        )
    semantic_path_reuse = {}
    for family in sorted(family_counts):
        family_ids = [
            int(row["id"]) for row in raw if family_of(str(row["category"])) == family
        ]
        counts = Counter(semantic_paths[source_id] for source_id in family_ids)
        train_paths = {semantic_paths[source_id] for source_id in family_ids if assign_split(source_id) == "train"}
        semantic_path_reuse[family] = {
            "bundle_count": len(family_ids),
            "unique_semantic_path_count": len(counts),
            "majority_path_fraction": max(counts.values()) / len(family_ids),
            "path_frequency": {" -> ".join(path): count for path, count in sorted(counts.items())},
            "dev_paths_seen_in_train_fraction": sum(
                semantic_paths[source_id] in train_paths
                for source_id in family_ids if assign_split(source_id) == "dev"
            ) / sum(assign_split(source_id) == "dev" for source_id in family_ids),
            "test_paths_seen_in_train_fraction": sum(
                semantic_paths[source_id] in train_paths
                for source_id in family_ids if assign_split(source_id) == "test"
            ) / sum(assign_split(source_id) == "test" for source_id in family_ids),
        }
    budget_failures = [row for row in whole_chains if row["budget_status"] == "fail"]
    budget_passes = [row for row in whole_chains if row["budget_status"] == "pass"]
    used_chain_counts = Counter(str(row["domain_chain_id"]) for row in whole_chains)
    domain_chain_ids = {str(row["chain_id"]) for row in domain_data}

    report = {
        "schema": "memoryarena_dataset_structure_audit_v1",
        "inputs": {
            "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "domain_data": {"path": str(domain_path), "sha256": sha256_file(domain_path)},
            "whole_steps": {"path": str(args.whole_steps.resolve()), "sha256": sha256_file(args.whole_steps.resolve())},
            "semantic_steps": {"path": str(args.semantic_steps.resolve()), "sha256": sha256_file(args.semantic_steps.resolve())},
        },
        "raw_schema": {
            "bundle_count": len(raw),
            "session_count": len(sessions),
            "top_level_key_sets": distribution(tuple(sorted(row)) for row in raw),
            "id_range": [min(int(row["id"]) for row in raw), max(int(row["id"]) for row in raw)],
            "unique_category_count": len({row["category"] for row in raw}),
            "family_counts": dict(sorted(family_counts.items())),
            "sessions_per_bundle": distribution(len(row["questions"]) for row in raw),
            "question_chars": quantiles([int(row["question_chars"]) for row in sessions]),
            "question_words": quantiles([int(row["question_words"]) for row in sessions]),
            "option_count": distribution(int(row["option_count"]) for row in sessions),
            "option_count_by_step": {
                str(step): distribution(int(row["option_count"]) for row in sessions if int(row["step_index"]) == step)
                for step in range(1, 7)
            },
            "answer_attribute_count": distribution(int(row["attribute_count"]) for row in sessions),
            "budget": {"bundle_unique_count": len(set(bundle_budgets)), "distribution": distribution(bundle_budgets), "summary": quantiles(bundle_budgets)},
            "metric_signature": distribution(row["metric_signature"] for row in sessions),
            "effective_metric": distribution(row["effective_metric"] for row in sessions),
            "target_missing_effective_metric": sum(bool(row["target_effective_metric_missing"]) for row in sessions),
            "target_missing_effective_metric_by_type": distribution(
                row["effective_metric"] for row in sessions if row["target_effective_metric_missing"]
            ),
            "section_titles_by_family_step": {
                f"{family}:{step}": sorted({str(row["title"]) for row in sessions if row["family"] == family and int(row["step_index"]) == step})
                for family in sorted(family_counts)
                for step in range(1, 7)
            },
        },
        "template_reuse": {
            "exact_question_count": len({row["question"] for row in sessions}),
            "normalized_preamble_count": len({row["normalized_preamble_hash"] for row in sessions}),
            "semantic_instruction_template_count": len({row["semantic_template_hash"] for row in sessions}),
            "structural_instruction_template_count": len({row["structural_template_hash"] for row in sessions}),
            "question_without_options_template_count": len({row["question_no_options_hash"] for row in sessions}),
            "semantic_template_frequency": distribution(row["semantic_template_hash"] for row in sessions),
        },
        "split": {
            "strategy": "source_position_mod10_8_1_1_v1",
            "raw_bundle_counts": dict(sorted(split_counts.items())),
            "raw_bundle_counts_by_family": {
                family: distribution(assign_split(int(row["id"])) for row in raw if family_of(str(row["category"])) == family)
                for family in sorted(family_counts)
            },
            "provisional_gate_bundle_counts": dict(sorted(provisional_split_counts.items())),
            "provisional_gate_allowed_statuses": ["pass", "unknown"],
            "provisional_gate_excludes": distribution(chain_by_id[int(row["id"])]["status"] for row in raw if row not in provisional),
            "overlap": {name: overlap_report(values) for name, values in value_fields.items()},
            "target_asin_components": shared_component_report(raw, target_asins_by_id),
            "option_text_components": shared_component_report(raw, option_texts_by_id),
            "target_sequence_ngram_overlap": ngram_overlap,
        },
        "reuse": {
            "target_occurrence_count": len(sessions),
            "unique_target_asin_count": len(target_frequencies),
            "target_asin_frequency": quantiles(list(target_frequencies.values())),
            "target_asins_repeated": sum(count > 1 for count in target_frequencies.values()),
            "target_occurrences_on_repeated_asins": sum(count for count in target_frequencies.values() if count > 1),
            "visible_option_occurrence_count": sum(len(row["options"]) for row in sessions),
            "unique_visible_option_text_count": len(option_frequencies),
            "visible_option_text_frequency": quantiles(list(option_frequencies.values())),
            "option_texts_repeated": sum(count > 1 for count in option_frequencies.values()),
            "option_texts_cross_split": sum(len(splits) > 1 for splits in option_splits.values()),
            "option_text_role": {
                "target_only": sum(counts["target"] > 0 and counts["distractor"] == 0 for counts in option_roles.values()),
                "distractor_only": sum(counts["target"] == 0 and counts["distractor"] > 0 for counts in option_roles.values()),
                "both_target_and_distractor": sum(counts["target"] > 0 and counts["distractor"] > 0 for counts in option_roles.values()),
            },
            "family_step_reuse": family_step_reuse,
            "semantic_path_reuse": semantic_path_reuse,
            "family_step_cells_with_one_target_label": sum(
                row["target_label_count"] == 1 for row in family_step_reuse
            ),
            "family_step_cells_with_target_label_majority_at_least_90pct": sum(
                row["target_label_majority_fraction"] >= 0.9 for row in family_step_reuse
            ),
        },
        "shortcut_diagnostics": {
            "target_position_distribution_provisional": distribution(row["target_position"] for row in target_position_records),
            "target_position_distribution_alignment_pass_only": distribution(row["target_position"] for row in alignment_pass_records),
            "alignment_pass_session_count": len(alignment_pass_records),
            "majority_position_baselines_provisional": [
                majority_accuracy(target_position_records, []),
                majority_accuracy(target_position_records, ["step_index"]),
                majority_accuracy(target_position_records, ["family", "step_index"]),
                majority_accuracy(target_position_records, ["family", "step_index", "metric_signature"]),
                majority_accuracy(target_position_records, ["question_no_options_hash"]),
            ],
            "train_option_role_dictionary": {
                split: role_dictionary_baseline(sessions, split) for split in ("dev", "test")
            },
            "natural_exact_current_observation_counterfactual_group_count": len(natural_counterfactual_groups),
            "natural_exact_current_observation_counterfactual_session_count": sum(len(group) for group in natural_counterfactual_groups),
        },
        "label_evidence": {
            "canonical_whole_chain_summary": {
                "scope": "AMG stricter derived-data audit; not an upstream MemoryArena label verdict",
                "chain_status_counts": chain_summary["chain_status_counts"],
                "step_status_counts": chain_summary["step_status_counts"],
                "check_status_counts": chain_summary["check_status_counts"],
                "proven_correct_chain_count": chain_summary["proven_correct_chain_count"],
            },
            "older_semantic_summary": {
                "bundle_status_counts": annotation_summary["bundle_status_counts"],
                "step_status_counts": annotation_summary["step_status_counts"],
                "check_status_counts": annotation_summary["check_status_counts"],
            },
            "whole_candidate_resolution_status": dict(sorted(candidate_resolution_status.items())),
            "whole_candidate_resolution_asin_present_count": candidate_asin_present_count,
            "whole_non_target_candidate_resolution_asin_present_count": non_target_candidate_asin_present_count,
            "visible_candidate_count": total_visible_candidates,
            "semantic_candidate_resolver_confidence": dict(sorted(candidate_confidence.items())),
            "check_pair_counts": {key: dict(sorted(values.items())) for key, values in discrepancy_counts.items()},
            "discrepancy_examples": discrepancy_examples,
        },
        "structured_generation_assets": {
            "domain_data": dependency_summary(domain_data),
            "unique_exact_target_asins": len(target_frequencies),
            "fixed_price_target_asins": len(fixed_price_target_asins),
            "range_price_target_asins": len(range_price_target_asins),
            "missing_rating_target_asins": len(missing_rating_target_asins),
            "used_domain_chain_counts": dict(sorted(used_chain_counts.items())),
            "unused_domain_chain_ids": sorted(domain_chain_ids - set(used_chain_counts)),
            "budget_replay": {
                "scope": "AMG strict-budget replay; discrepancies are not confirmed upstream label failures",
                "pass_bundle_count": len(budget_passes),
                "fail_bundle_count": len(budget_failures),
                "failures": [
                    {
                        "source_id": int(row["source_id"]),
                        "category": row["category"],
                        "budget": row["budget"],
                        "annotated_total_min": row["annotated_bundle_total_min"],
                        "annotated_total_max": row["annotated_bundle_total_max"],
                        "minimum_over_budget": row["annotated_bundle_total_min"] - row["budget"],
                    }
                    for row in budget_failures
                ],
                "pass_worst_case_slack": quantiles(
                    [row["budget"] - row["annotated_bundle_total_max"] for row in budget_passes]
                ),
                "price_range_target_count": sum(
                    item.get("price_semantics") == "webshop_runtime_random_uniform_range"
                    for row in whole_chains for item in row["target_price_evidence"]
                ),
                "price_range_crosses_budget_boundary_count": sum(
                    row["annotated_bundle_total_min"] <= row["budget"] < row["annotated_bundle_total_max"]
                    for row in whole_chains
                ),
            },
            "family_step_label_pool_count": len(pool_rows),
            "family_step_label_pool_size": quantiles([row["unique_target_asins"] for row in pool_rows]),
            "family_step_label_pools": pool_rows,
            "target_sessions_with_multiple_extract_pattern_labels": sum(
                int(row["target_distinct_label_count"]) > 1 for row in sessions
            ),
            "note": "Pools use published target ASINs and exact official metadata; they are assets for a new sidecar-grounded generator, not proof that original visible options map correctly.",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_overlap = report["split"]["overlap"]
    shortcuts = report["shortcut_diagnostics"]
    label = report["label_evidence"]
    domain = report["structured_generation_assets"]["domain_data"]
    lines = [
        "# MemoryArena Bundled-Shopping Dataset Structure Audit",
        "",
        "## Scope",
        "",
        "This is a read-only structural and leakage audit. MemoryArena publishes these 150 rows as a test set; the train/dev/test split below is AMG's row-position development split. Provisional lexical mappings are used only for shortcut diagnostics and never promoted to annotation proof.",
        "",
        "## Core shape",
        "",
        f"- {len(raw)} bundles, {len(sessions)} sessions, {len(family_counts)} families, exactly six sessions per bundle.",
        f"- Raw split by row position: {dict(sorted(split_counts.items()))}; provisional gate: {dict(sorted(provisional_split_counts.items()))}.",
        f"- {len(target_frequencies)} unique target ASINs over {len(sessions)} target occurrences.",
        f"- {report['reuse']['unique_visible_option_text_count']} unique visible option strings over {report['reuse']['visible_option_occurrence_count']} occurrences.",
        f"- Option-count distribution: {report['raw_schema']['option_count']}.",
        f"- Normalized global preambles: {report['template_reuse']['normalized_preamble_count']}; semantic instruction templates: {report['template_reuse']['semantic_instruction_template_count']}.",
        "",
        "## AMG row-position split overlap",
        "",
        f"- Dev target ASINs seen in train: {split_overlap['target_asin']['train_dev_shared_unique']} unique ({split_overlap['target_asin']['dev_unique_seen_in_train_fraction']:.1%}).",
        f"- Test target ASINs seen in train: {split_overlap['target_asin']['train_test_shared_unique']} unique ({split_overlap['target_asin']['test_unique_seen_in_train_fraction']:.1%}).",
        f"- Dev option strings seen in train: {split_overlap['option_text']['train_dev_shared_unique']} unique ({split_overlap['option_text']['dev_unique_seen_in_train_fraction']:.1%}).",
        f"- Test option strings seen in train: {split_overlap['option_text']['train_test_shared_unique']} unique ({split_overlap['option_text']['test_unique_seen_in_train_fraction']:.1%}).",
        f"- Held-out bundles connected to train through a shared target ASIN: {report['split']['target_asin_components']['heldout_bundles_connected_to_train']}/{report['split']['target_asin_components']['heldout_bundle_count']}.",
        f"- Held-out bundles connected to train through any exact option text: {report['split']['option_text_components']['heldout_bundles_connected_to_train']}/{report['split']['option_text_components']['heldout_bundle_count']}.",
        "",
        "## Shortcut diagnostics",
        "",
        f"- Provisional target positions: {shortcuts['target_position_distribution_provisional']}.",
        f"- Alignment-proven positions cover only {shortcuts['alignment_pass_session_count']}/900 sessions.",
        f"- Naturally occurring exact current-observation counterfactual groups: {shortcuts['natural_exact_current_observation_counterfactual_group_count']}.",
        f"- Train option-role dictionary on dev: coverage {shortcuts['train_option_role_dictionary']['dev']['coverage']:.1%}, target-in-argmax {shortcuts['train_option_role_dictionary']['dev']['target_in_argmax_accuracy_on_covered']:.1%} on covered sessions.",
        f"- Train option-role dictionary on test: coverage {shortcuts['train_option_role_dictionary']['test']['coverage']:.1%}, target-in-argmax {shortcuts['train_option_role_dictionary']['test']['target_in_argmax_accuracy_on_covered']:.1%} on covered sessions.",
        "",
        "## Label evidence boundary",
        "",
        f"- AMG stricter whole-chain audit statuses: {label['canonical_whole_chain_summary']['chain_status_counts']}; proven under that stricter audit: {label['canonical_whole_chain_summary']['proven_correct_chain_count']}. These are not upstream MemoryArena label verdicts.",
        f"- Candidate records carrying an ASIN: {label['whole_candidate_resolution_asin_present_count']}/{label['visible_candidate_count']}; only {label['whole_non_target_candidate_resolution_asin_present_count']} are non-target slots. Target-slot ASIN attachment is conditional on provisional alignment and is not mapping proof.",
        f"- Strict target alignment: {label['canonical_whole_chain_summary']['check_status_counts']['target_alignment']}.",
        "- The older semantic audit can produce compatibility/ranking failures from fuzzy candidate-title resolution; the canonical audit refuses to turn those fuzzy mappings into formal ranking proof.",
        "",
        "## Structured assets",
        "",
        f"- domain_data contains {domain['definition_count']} chains: {domain['six_step_definition_count']} six-step and {domain['five_step_definition_count']} five-step.",
        f"- Structurally counterfactual dependency edges: {domain['structurally_counterfactual_dependency_edge_count']}.",
        f"- Published exact target pool: {report['structured_generation_assets']['unique_exact_target_asins']} unique ASINs across {report['structured_generation_assets']['family_step_label_pool_count']} family/step/label pools.",
        f"- Pool-size summary: {report['structured_generation_assets']['family_step_label_pool_size']}.",
        "",
        "## Immediate implication",
        "",
        "Option shuffling and wording perturbations can reduce surface-position cues, but they cannot remove product/chain memorization. The current AMG row-position split is row-disjoint but not product-, option-, template-, or dependency-disjoint. A useful expansion therefore needs explicit candidate-ASIN sidecars, a symbolic six-step oracle, grouped splitting, and paired counterfactual chains whose dependent observation is identical before RETRIEVE while the hidden prior and correct BUY flip.",
        "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
