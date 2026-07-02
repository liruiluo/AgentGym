from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentenv_agentmemory.environment import AgentMemoryEnv, Product


ACTION_RE = re.compile(r"^(?P<op>[A-Z_]+)\s+(?P<payload>\{.*\})$")
SEARCH_RESULT_RE = re.compile(
    r"^- (?P<title>.*) \((?P<attrs>average_rating=.*?, price_usd=.*?, total_reviews=.*?, match_score=.*?)\)$"
)
ATTR_RE = re.compile(r"(average_rating|price_usd|total_reviews|match_score)=([^,)]*)")
TOKEN_RE = re.compile(r"[a-z0-9]+")
PREFERENCE_RE = re.compile(r"\b(highest|lowest)[- ](rated|priced|price|rating)\b", re.IGNORECASE)
NON_SEMANTIC_ATTRIBUTE_KEYS = {"source_option"}
COLOR_LABELS = {
    "black",
    "blue",
    "brown",
    "gold",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
BROAD_COLOR_TOKENS = {"color", "colors", "colour", "colours", "colorful", "vibrant"}
POLICY_MODES = (
    "scripted-search-memory",
    "search-full-context",
    "search-no-memory",
)
POLICY_BOUNDARIES = {
    "scripted-search-memory": (
        "Scripted baseline only: uses visible candidate titles, current instruction text, its own ADD/RETRIEVE "
        "memory, and public metadata returned by SEARCH. It is not RL training and not memory-improvement evidence."
    ),
    "search-full-context": (
        "Scripted full-context diagnostic: uses visible candidate titles, current instruction text, prior accepted "
        "purchase notes kept in the runner context, and public metadata returned by SEARCH. It bypasses memory-tool "
        "decisions, so it is not a learned memory policy or RL evidence."
    ),
    "search-no-memory": (
        "Scripted no-memory diagnostic: uses only current visible candidate titles, current instruction text, and "
        "public metadata returned by SEARCH. It does not use ADD, RETRIEVE, or prior purchase notes, so compatibility "
        "failures reflect missing cross-session memory."
    ),
}


@dataclass
class SearchHit:
    title: str
    average_rating: float | None
    price_usd: float | None
    total_reviews: int | None
    match_score: int | None


@dataclass
class CandidateView:
    product_id: str
    title: str
    attributes: dict[str, Any]
    search_hit: SearchHit | None


@dataclass
class StepDecision:
    subtask_index: int
    attempt_number: int
    instruction_head: str
    preference: str
    compatibility_fallback: str
    active_compatibility_keys: list[str]
    allowed_values: list[str]
    compatible_product_ids: list[str]
    fallback_product_ids: list[str]
    ranked_product_ids: list[str]
    chosen_product_id: str
    chosen_title: str
    chosen_search_title: str | None
    chosen_rating: float | None
    chosen_price: float | None
    chosen_reviews: int | None
    target_product_id: str | None
    target_chosen: bool | None
    target_in_compatible_pool: bool | None
    target_rank: int | None
    buy_reward: float
    buy_accepted: bool
    progress_score: float


@dataclass
class EpisodeResult:
    task_id: str
    split: str
    data_idx: int
    episode_success: bool
    progress_score: float
    reward_sum: float
    env_steps: int
    search_calls: int
    add_calls: int
    retrieve_calls: int
    buy_calls: int
    rejected_buys: int
    decisions: list[StepDecision]


def action(op: str, **payload: Any) -> str:
    return f"{op} {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    return " ".join(normalize_tokens(text))


def label_matches(label: str, text: str) -> bool:
    label_norm = normalize_text(label)
    text_norm = normalize_text(text)
    if not label_norm or not text_norm:
        return False
    label_tokens = normalize_tokens(label)
    text_tokens = normalize_tokens(text)
    if not label_tokens or not text_tokens:
        return False
    if len(label_tokens) == 1:
        return bool(token_variants(label_tokens) & token_variants(text_tokens))
    return phrase_tokens_match(label_tokens, text_tokens)


def phrase_tokens_match(label_tokens: list[str], text_tokens: list[str]) -> bool:
    """Match label tokens as an adjacent phrase, allowing simple plural variants."""
    width = len(label_tokens)
    for start in range(0, len(text_tokens) - width + 1):
        window = text_tokens[start : start + width]
        if all(
            bool(token_variants([label_token]) & token_variants([text_token]))
            for label_token, text_token in zip(label_tokens, window)
        ):
            return True
    return False


def token_variants(tokens: list[str]) -> set[str]:
    variants = set(tokens)
    for token in tokens:
        if len(token) > 3 and token.endswith("s"):
            variants.add(token[:-1])
        elif len(token) > 2 and not token.endswith("s"):
            variants.add(f"{token}s")
    return variants


def broad_value_group_matches(values: list[str], text: str) -> bool:
    """Match broad set descriptions such as "9 vibrant colors" to a color list."""
    value_tokens = {
        token
        for value in values
        for token in normalize_tokens(value)
        if token in COLOR_LABELS
    }
    text_tokens = set(normalize_tokens(text))
    return len(value_tokens) >= 3 and bool(text_tokens & BROAD_COLOR_TOKENS)


def clean_label(text: str) -> str:
    text = re.sub(r"^\s*(?:and|or)\s+", "", text.strip(), flags=re.IGNORECASE)
    text = text.strip(" :;,.\n\t*")
    return text


def split_values(text: str) -> list[str]:
    text = text.replace(" and ", ", ")
    return [clean_label(item) for item in text.split(",") if clean_label(item)]


def extract_section(instruction: str, start: str, end_markers: tuple[str, ...]) -> str:
    lower = instruction.lower()
    start_idx = lower.find(start.lower())
    if start_idx < 0:
        return ""
    body = instruction[start_idx + len(start) :]
    end_positions = [body.lower().find(marker.lower()) for marker in end_markers]
    end_positions = [pos for pos in end_positions if pos >= 0]
    if end_positions:
        body = body[: min(end_positions)]
    return body.strip()


def parse_preference(instruction: str, subtask_index: int) -> str:
    goal = extract_section(instruction, "**Goal:**", ("**Preference:**", "**Avoid:**"))
    preference = extract_section(instruction, "**Preference:**", ("**Avoid:**",))
    search_order = [goal, preference, instruction] if subtask_index == 0 else [preference, goal, instruction]
    for text in search_order:
        match = PREFERENCE_RE.search(text)
        if not match:
            continue
        direction = match.group(1).lower()
        kind = match.group(2).lower()
        if kind in {"priced", "price"}:
            return f"{direction}-priced"
        return f"{direction}-rated"
    return "highest-rated"


def parse_pairs(section: str, verb: str) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = {}
    if not section:
        return pairs
    # Keep sentences short; generated MemoryArena notes use period-separated rules.
    for sentence in re.split(r"\.\s*", section):
        sentence = sentence.strip()
        if not sentence:
            continue
        one_of = re.match(rf"(?P<src>.+?)\s+{verb}\s+one of:\s*(?P<dst>.+)$", sentence, re.IGNORECASE)
        plain = re.match(rf"(?P<src>.+?)\s+{verb}\s+(?P<dst>.+)$", sentence, re.IGNORECASE)
        match = one_of or plain
        if not match:
            continue
        src = clean_label(match.group("src"))
        dst_values = split_values(match.group("dst"))
        if not src or not dst_values:
            continue
        pairs.setdefault(src, [])
        for value in dst_values:
            if value not in pairs[src]:
                pairs[src].append(value)
    return pairs


def parse_compatibility(instruction: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    compat_section = extract_section(instruction, "Compatibility notes:", ("**Preference:**", "**Avoid:**"))
    avoid_section = extract_section(instruction, "**Avoid:**", ("Use memory tools",))
    return parse_pairs(compat_section, "pairs well with"), parse_pairs(avoid_section, "avoids")


def parse_search_hits(observation: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for raw_line in observation.splitlines():
        line = raw_line.strip()
        match = SEARCH_RESULT_RE.match(line)
        if not match:
            continue
        attrs = {key: value.strip() for key, value in ATTR_RE.findall(match.group("attrs"))}
        hits.append(
            SearchHit(
                title=match.group("title"),
                average_rating=parse_optional_float(attrs.get("average_rating")),
                price_usd=parse_optional_float(attrs.get("price_usd")),
                total_reviews=parse_optional_int(attrs.get("total_reviews")),
                match_score=parse_optional_int(attrs.get("match_score")),
            )
        )
    return hits


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, "", "unknown"):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_optional_int(value: str | None) -> int | None:
    if value in (None, "", "unknown"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def candidate_text(candidate: CandidateView) -> str:
    semantic_attrs = {
        key: value for key, value in candidate.attributes.items() if key not in NON_SEMANTIC_ATTRIBUTE_KEYS
    }
    parts = [candidate.title, render_attrs(semantic_attrs)]
    if candidate.search_hit is not None:
        parts.append(candidate.search_hit.title)
    return " ".join(parts)


def render_attrs(attrs: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(attrs.items()))


def infer_allowed_values(
    compatibility: dict[str, list[str]],
    avoid: dict[str, list[str]],
    memory_text: str,
    candidates: list[CandidateView],
) -> tuple[list[str], list[str], list[CandidateView]]:
    if not compatibility:
        return [], [], candidates
    active_keys = [key for key in compatibility if label_matches(key, memory_text)]
    allowed_values: list[str] = []
    avoided_values: list[str] = []
    for key in active_keys:
        for value in compatibility.get(key, []):
            if value not in allowed_values:
                allowed_values.append(value)
        for value in avoid.get(key, []):
            if value not in avoided_values:
                avoided_values.append(value)
    if not active_keys:
        # Some MemoryArena generated products describe a set rather than a
        # specific attribute, e.g. "9 vibrant colors". In that case the previous
        # item is clearly in the compatibility dimension, but no single note key
        # appears verbatim. Filter to candidates that contain any allowed next
        # value instead of giving up to all candidates immediately.
        all_allowed_values = sorted({value for values in compatibility.values() for value in values})
        compatible = [
            candidate
            for candidate in candidates
            if any(label_matches(value, candidate_text(candidate)) for value in all_allowed_values)
        ]
        if compatible and any(token in normalize_tokens(memory_text) for token in ("color", "colors", "colour", "colours")):
            return [], all_allowed_values, compatible
        return [], [], candidates
    compatible = []
    for candidate in candidates:
        text = candidate_text(candidate)
        if any(label_matches(value, text) for value in allowed_values) or broad_value_group_matches(allowed_values, text):
            compatible.append(candidate)
    if avoided_values and len(active_keys) == 1:
        compatible = [
            candidate
            for candidate in compatible
            if not any(label_matches(value, candidate_text(candidate)) for value in avoided_values)
        ]
    if compatible:
        return active_keys, allowed_values, compatible
    # Generated tasks often use broad color-set descriptions such as "9 vibrant colors".
    # If exact compatibility labels do not appear, keep all candidates instead of inventing
    # hidden attributes; the metric still uses public SEARCH metadata.
    return active_keys, allowed_values, candidates




def filter_explicit_attribute_compatibility(candidates: list[CandidateView], memory_text: str) -> list[CandidateView]:
    constrained = [candidate for candidate in candidates if has_explicit_constraints(candidate.attributes)]
    if not constrained:
        return candidates
    compatible = [candidate for candidate in candidates if explicit_attrs_compatible(candidate.attributes, memory_text)]
    return compatible or candidates


def has_explicit_constraints(attrs: dict[str, Any]) -> bool:
    return any(
        key in attrs
        for key in (
            "compatible_tv_min",
            "compatible_monitor_min",
            "compatible_laptop_min",
            "max_weight_kg",
            "supported_vesa",
            "required_port",
        )
    )


def explicit_attrs_compatible(attrs: dict[str, Any], memory_text: str) -> bool:
    checks = []
    if "compatible_tv_min" in attrs:
        checks.append(in_range(extract_memory_number(memory_text, "tv_size_in"), attrs.get("compatible_tv_min"), attrs.get("compatible_tv_max")))
    if "compatible_monitor_min" in attrs:
        checks.append(in_range(extract_memory_number(memory_text, "monitor_size_in"), attrs.get("compatible_monitor_min"), attrs.get("compatible_monitor_max")))
    if "compatible_laptop_min" in attrs:
        checks.append(in_range(extract_memory_number(memory_text, "laptop_size_in"), attrs.get("compatible_laptop_min"), attrs.get("compatible_laptop_max")))
    if "max_weight_kg" in attrs:
        weight = extract_memory_number(memory_text, "tv_weight_kg")
        if weight is None:
            weight = extract_memory_number(memory_text, "monitor_weight_kg")
        checks.append(weight is not None and weight <= float(attrs.get("max_weight_kg")))
    if "supported_vesa" in attrs:
        vesa = extract_memory_value(memory_text, "vesa")
        supported = attrs.get("supported_vesa")
        checks.append(vesa is not None and vesa in str(supported))
    if "required_port" in attrs:
        ports = extract_memory_value(memory_text, "laptop_ports") or extract_memory_value(memory_text, "monitor_ports")
        checks.append(ports is not None and str(attrs.get("required_port")) in str(ports).split(","))
    return all(checks) if checks else True


def in_range(value: float | None, min_value: Any, max_value: Any) -> bool:
    return value is not None and float(min_value) <= value <= float(max_value)


def extract_memory_number(memory_text: str, key: str) -> float | None:
    value = extract_memory_value(memory_text, key)
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def extract_memory_value(memory_text: str, key: str) -> str | None:
    match = re.search(rf"(?:^|[;\s]){re.escape(key)}=([^;]+)", memory_text)
    return match.group(1).strip() if match else None


def metric_key(candidate: CandidateView, preference: str) -> tuple[float, float, float, float, str]:
    hit = candidate.search_hit
    rating = hit.average_rating if hit and hit.average_rating is not None else None
    price = hit.price_usd if hit and hit.price_usd is not None else None
    reviews = hit.total_reviews if hit and hit.total_reviews is not None else None
    match_score = hit.match_score if hit and hit.match_score is not None else None
    order_tiebreak = -float(ord(str(candidate.attributes.get("source_option", "z"))[:1]))
    if preference == "highest-rated":
        return (rating if rating is not None else -1e9, reviews if reviews is not None else -1, match_score or -1, -(price or 0), order_tiebreak)
    if preference == "lowest-rated":
        return (-(rating if rating is not None else 1e9), order_tiebreak, match_score or -1, -(price or 0), order_tiebreak)
    if preference == "highest-priced":
        return (price if price is not None else -1e9, rating if rating is not None else -1, reviews if reviews is not None else -1, match_score or -1, order_tiebreak)
    if preference == "lowest-priced":
        return (-(price if price is not None else 1e9), order_tiebreak, rating if rating is not None else -1, reviews if reviews is not None else -1, match_score or -1)
    raise ValueError(f"Unsupported preference: {preference}")


def choose_candidate(candidates: list[CandidateView], preference: str) -> CandidateView:
    if not candidates:
        raise ValueError("No candidates to choose from.")
    return max(candidates, key=lambda item: metric_key(item, preference))


def rank_candidates(candidates: list[CandidateView], preference: str) -> list[CandidateView]:
    return sorted(candidates, key=lambda item: metric_key(item, preference), reverse=True)


def build_ranked_candidate_pool(
    *,
    compatible_candidates: list[CandidateView],
    fallback_candidates: list[CandidateView],
    preference: str,
    compatibility_fallback: str,
) -> tuple[list[CandidateView], list[CandidateView]]:
    ranked_compatible = rank_candidates(compatible_candidates, preference)
    if compatibility_fallback == "none":
        return ranked_compatible, []
    if compatibility_fallback != "ranked-all-after-compatible":
        raise ValueError(f"Unsupported compatibility_fallback: {compatibility_fallback}")
    compatible_ids = {candidate.product_id for candidate in ranked_compatible}
    fallback_pool = [candidate for candidate in fallback_candidates if candidate.product_id not in compatible_ids]
    ranked_fallback = rank_candidates(fallback_pool, preference)
    return ranked_compatible + ranked_fallback, ranked_fallback


def search_candidate(env: AgentMemoryEnv, product: Product) -> tuple[SearchHit | None, str, float, bool, dict[str, Any]]:
    obs, reward, done, _, info = env.step(action("SEARCH", query=product.title, top_k=1))
    hits = parse_search_hits(obs)
    return (hits[0] if hits else None), obs, reward, done, info


def run_episode(
    env: AgentMemoryEnv,
    *,
    data_idx: int,
    include_target_audit: bool = True,
    max_buy_attempts: int = 1,
    compatibility_fallback: str = "none",
    policy_mode: str = "scripted-search-memory",
) -> tuple[EpisodeResult, list[dict[str, Any]]]:
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"Unsupported policy_mode={policy_mode!r}; expected one of {POLICY_MODES}.")
    use_env_memory_tools = policy_mode == "scripted-search-memory"
    use_context_memory = policy_mode in {"scripted-search-memory", "search-full-context"}
    observation, info = env.reset(data_idx=data_idx)
    task_id = info["task_id"]
    split = info["split"]
    reward_sum = 0.0
    actions_log: list[dict[str, Any]] = []
    decisions: list[StepDecision] = []
    memory_texts: list[str] = []
    op_counts = {"SEARCH": 0, "ADD": 0, "RETRIEVE": 0, "BUY": 0}
    rejected_buys = 0
    env_steps = 0

    while not env.done and env.current_subtask_index < len(env.require_task().subtasks):
        subtask_index = env.current_subtask_index
        subtask = env.current_subtask()
        if use_env_memory_tools and memory_texts:
            query = memory_texts[-1][:240]
            act = action("RETRIEVE", query=query, top_k=3)
            observation, reward, done, _, info = env.step(act)
            reward_sum += reward
            env_steps += 1
            op_counts["RETRIEVE"] += 1
            actions_log.append(log_action(env_steps, task_id, subtask_index, act, reward, done, info))

        candidates: list[CandidateView] = []
        for product in subtask.candidate_products:
            hit, observation, reward, done, info = search_candidate(env, product)
            reward_sum += reward
            env_steps += 1
            op_counts["SEARCH"] += 1
            actions_log.append(log_action(env_steps, task_id, subtask_index, action("SEARCH", query=product.title, top_k=1), reward, done, info))
            candidates.append(CandidateView(product_id=product.product_id, title=product.title, attributes=dict(product.attributes), search_hit=hit))

        preference = parse_preference(subtask.instruction, subtask_index)
        compatibility, avoid = parse_compatibility(subtask.instruction)
        memory_text = "\n".join(memory_texts) if use_context_memory else ""
        compatibility_memory_text = memory_texts[-1] if use_context_memory and memory_texts else ""
        explicit_candidates = filter_explicit_attribute_compatibility(candidates, memory_text)
        active_keys, allowed_values, compatible_candidates = infer_allowed_values(
            compatibility,
            avoid,
            compatibility_memory_text,
            explicit_candidates,
        )
        ranked_candidates, fallback_candidates = build_ranked_candidate_pool(
            compatible_candidates=compatible_candidates,
            fallback_candidates=explicit_candidates,
            preference=preference,
            compatibility_fallback=compatibility_fallback,
        )
        compatible_product_ids = [item.product_id for item in compatible_candidates]
        fallback_product_ids = [item.product_id for item in fallback_candidates]
        ranked_product_ids = [item.product_id for item in ranked_candidates]
        chosen: CandidateView | None = None
        buy_accepted = False
        attempts = 0
        done = False
        target_product_id = subtask.target_product_id if include_target_audit else None
        for candidate in ranked_candidates[: max(1, max_buy_attempts)]:
            attempts += 1
            buy_act = action("BUY", product_id=candidate.product_id)
            observation, buy_reward, done, _, info = env.step(buy_act)
            reward_sum += buy_reward
            env_steps += 1
            op_counts["BUY"] += 1
            buy_accepted = buy_reward > 0 and not info.get("compatibility_violations")
            if not buy_accepted:
                rejected_buys += 1
            actions_log.append(log_action(env_steps, task_id, subtask_index, buy_act, buy_reward, done, info))
            decisions.append(
                StepDecision(
                    subtask_index=subtask_index,
                    attempt_number=attempts,
                    instruction_head=subtask.instruction.splitlines()[0][:160],
                    preference=preference,
                    compatibility_fallback=compatibility_fallback,
                    active_compatibility_keys=active_keys,
                    allowed_values=allowed_values,
                    compatible_product_ids=compatible_product_ids,
                    fallback_product_ids=fallback_product_ids,
                    ranked_product_ids=ranked_product_ids,
                    chosen_product_id=candidate.product_id,
                    chosen_title=candidate.title,
                    chosen_search_title=candidate.search_hit.title if candidate.search_hit else None,
                    chosen_rating=candidate.search_hit.average_rating if candidate.search_hit else None,
                    chosen_price=candidate.search_hit.price_usd if candidate.search_hit else None,
                    chosen_reviews=candidate.search_hit.total_reviews if candidate.search_hit else None,
                    target_product_id=target_product_id,
                    target_chosen=(candidate.product_id == subtask.target_product_id) if include_target_audit else None,
                    target_in_compatible_pool=(subtask.target_product_id in compatible_product_ids) if include_target_audit else None,
                    target_rank=(ranked_product_ids.index(subtask.target_product_id) + 1)
                    if include_target_audit and subtask.target_product_id in ranked_product_ids
                    else None,
                    buy_reward=buy_reward,
                    buy_accepted=buy_accepted,
                    progress_score=float(info.get("progress_score", 0.0)),
                )
            )
            if buy_accepted:
                chosen = candidate
                break
        if not buy_accepted or chosen is None:
            break

        memory_value = format_memory_value(subtask_index, chosen)
        if use_context_memory:
            memory_texts.append(memory_value)
        if use_env_memory_tools and not done:
            add_act = action("ADD", key=f"selected_step_{subtask_index + 1}", value=memory_value)
            observation, reward, done, _, info = env.step(add_act)
            reward_sum += reward
            env_steps += 1
            op_counts["ADD"] += 1
            actions_log.append(log_action(env_steps, task_id, env.current_subtask_index, add_act, reward, done, info))

    result = EpisodeResult(
        task_id=task_id,
        split=split,
        data_idx=data_idx,
        episode_success=bool(info.get("episode_success", False)),
        progress_score=float(info.get("progress_score", 0.0)),
        reward_sum=reward_sum,
        env_steps=env_steps,
        search_calls=op_counts["SEARCH"],
        add_calls=op_counts["ADD"],
        retrieve_calls=op_counts["RETRIEVE"],
        buy_calls=op_counts["BUY"],
        rejected_buys=rejected_buys,
        decisions=decisions,
    )
    return result, actions_log


def format_memory_value(subtask_index: int, candidate: CandidateView) -> str:
    hit = candidate.search_hit
    fields = [f"step={subtask_index + 1}", f"product_id={candidate.product_id}", f"visible_title={candidate.title}"]
    for key, value in sorted(candidate.attributes.items()):
        fields.append(f"{key}={value}")
    if hit is not None:
        fields.extend(
            [
                f"catalog_title={hit.title}",
                f"average_rating={hit.average_rating}",
                f"price_usd={hit.price_usd}",
                f"total_reviews={hit.total_reviews}",
            ]
        )
    return "; ".join(fields)


def log_action(
    step_number: int,
    task_id: str,
    subtask_index: int,
    chosen_action: str,
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    match = ACTION_RE.match(chosen_action)
    payload: dict[str, Any] = {}
    op = chosen_action.split(maxsplit=1)[0]
    if match:
        op = match.group("op")
        payload = json.loads(match.group("payload"))
    return {
        "step_number": step_number,
        "task_id": task_id,
        "subtask_index": subtask_index,
        "op": op,
        "payload": payload,
        "reward": reward,
        "done": done,
        "progress_score": info.get("progress_score"),
        "episode_success": info.get("episode_success"),
        "compatibility_violations": info.get("compatibility_violations", []),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(results: list[EpisodeResult], *, args: argparse.Namespace, started_at: float) -> dict[str, Any]:
    successes = sum(1 for item in results if item.episode_success)
    total = len(results)
    total_steps = sum(item.env_steps for item in results)
    return {
        "marker": "AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK",
        "policy": args.policy_mode,
        "compatibility_fallback": args.compatibility_fallback,
        "data_path": args.data,
        "split": args.split,
        "split_dir": args.split_dir,
        "catalog_index_path": args.catalog_index,
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "mean_progress_score": sum(item.progress_score for item in results) / total if total else 0.0,
        "total_env_steps": total_steps,
        "search_calls": sum(item.search_calls for item in results),
        "add_calls": sum(item.add_calls for item in results),
        "retrieve_calls": sum(item.retrieve_calls for item in results),
        "buy_calls": sum(item.buy_calls for item in results),
        "rejected_buys": sum(item.rejected_buys for item in results),
        "max_buy_attempts": args.max_buy_attempts,
        "task_id_filter": args.task_id or [],
        "elapsed_seconds": round(time.time() - started_at, 3),
        "task_ids": [item.task_id for item in results],
        "per_episode": [
            {
                "task_id": item.task_id,
                "episode_success": item.episode_success,
                "progress_score": item.progress_score,
                "env_steps": item.env_steps,
                "search_calls": item.search_calls,
                "rejected_buys": item.rejected_buys,
            }
            for item in results
        ],
        "boundary": POLICY_BOUNDARIES[args.policy_mode],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted SEARCH baseline for AgentMemoryGym bundled shopping.")
    parser.add_argument("--data", default=os.environ.get("AGENTMEMORY_DATA_PATH"), required=os.environ.get("AGENTMEMORY_DATA_PATH") is None)
    parser.add_argument("--split", default=os.environ.get("AGENTMEMORY_SPLIT", "dev"))
    parser.add_argument("--split-dir", default=os.environ.get("AGENTMEMORY_SPLIT_DIR"))
    parser.add_argument("--catalog-index", default=os.environ.get("AGENTMEMORY_CATALOG_INDEX_PATH"), required=os.environ.get("AGENTMEMORY_CATALOG_INDEX_PATH") is None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", help="Run only matching task_id values. Repeatable.")
    parser.add_argument(
        "--policy-mode",
        choices=POLICY_MODES,
        default="scripted-search-memory",
        help=(
            "Which scripted diagnostic baseline to run. 'scripted-search-memory' is the default heuristic "
            "memory-manager baseline with ADD/RETRIEVE. 'search-full-context' keeps prior accepted purchase "
            "notes in runner context without memory-tool actions. 'search-no-memory' ignores prior purchase notes."
        ),
    )
    parser.add_argument("--max-buy-attempts", type=int, default=1, help="Try up to this many ranked candidate BUY actions before ending the episode.")
    parser.add_argument(
        "--compatibility-fallback",
        choices=("none", "ranked-all-after-compatible"),
        default="none",
        help=(
            "Diagnostic fallback for brittle compatibility labels. The default 'none' preserves the strict "
            "scripted baseline. 'ranked-all-after-compatible' tries compatible candidates first, then other "
            "visible candidates ranked by the same SEARCH metadata."
        ),
    )
    parser.add_argument("--include-target-audit", action="store_true", help="Include target ids in saved audit rows, never in action selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = AgentMemoryEnv(data_path=args.data, split=args.split, split_dir=args.split_dir, catalog_index_path=args.catalog_index)
    data_indices = list(range(len(env.tasks)))
    if args.task_id:
        requested_task_ids = set(args.task_id)
        data_indices = [idx for idx, task in enumerate(env.tasks) if task.task_id in requested_task_ids]
        missing_task_ids = sorted(requested_task_ids - {env.tasks[idx].task_id for idx in data_indices})
        if missing_task_ids:
            raise SystemExit(f"Requested task_id not found in split: {', '.join(missing_task_ids)}")
    if args.limit is not None:
        data_indices = data_indices[: args.limit]
    results: list[EpisodeResult] = []
    action_rows: list[dict[str, Any]] = []
    for data_idx in data_indices:
        result, actions = run_episode(
            env,
            data_idx=data_idx,
            include_target_audit=args.include_target_audit,
            max_buy_attempts=args.max_buy_attempts,
            compatibility_fallback=args.compatibility_fallback,
            policy_mode=args.policy_mode,
        )
        results.append(result)
        action_rows.extend(actions)
        print(
            "EPISODE",
            result.task_id,
            "success=",
            result.episode_success,
            "progress=",
            result.progress_score,
            "steps=",
            result.env_steps,
            "search=",
            result.search_calls,
            flush=True,
        )
    summary = summarize(results, args=args, started_at=started_at)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "episodes.jsonl", [asdict(item) for item in results])
    write_jsonl(output_dir / "actions.jsonl", action_rows)
    print(
        summary["marker"],
        f"episodes={summary['episodes']}",
        f"successes={summary['successes']}",
        f"success_rate={summary['success_rate']:.4f}",
        f"mean_progress={summary['mean_progress_score']:.4f}",
        f"search_calls={summary['search_calls']}",
    )
    print("SUMMARY_PATH", output_dir / "summary.json")


if __name__ == "__main__":
    main()
