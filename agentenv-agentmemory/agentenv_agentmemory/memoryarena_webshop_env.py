from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from .formal_native_contract import build_reward_components, infer_raw_action_op
from .memoryarena_dataset import MemoryArenaBundle
from .memory_state import MemoryEntry, rank_memory_entries_bm25
from .native_webshop_backend import NativePage, NativeWebShopBackend
from .reward_hierarchy import (
    EXACT_REPEAT_ACTION_PENALTY,
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
    INVALID_ACTION_PENALTY,
    WRONG_BUY_TERMINAL_FAILURE,
    build_memoryarena_reward_contract,
)


MEMORY_TOOL_OPS = {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
NATIVE_ACTION_RE = re.compile(r"\A(search|click)\[([^\[\]\r\n]+)\]\Z")
MEMORY_ACTION_RE = re.compile(r"\A(ADD|UPDATE|DELETE|RETRIEVE|SUMMARY|FILTER)\s+(\{.*\})\Z", re.DOTALL)


class InvalidNativeAction(ValueError):
    pass


class NativeWebShopInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedAction:
    op: str
    raw_action: str
    payload: dict[str, Any] | None = None

    @property
    def is_native(self) -> bool:
        return self.payload is None


class MemoryArenaWebShopEnv:
    """Six-session MemoryArena task on the original WebShop action surface."""

    surface = "memoryarena_webshop_native_v1"

    def __init__(
        self,
        *,
        bundles: Sequence[MemoryArenaBundle],
        backend: NativeWebShopBackend,
        env_uid: str | None = None,
        first_valid_add_reward: float = FIRST_VALID_ADD_BONUS,
        first_valid_later_session_retrieve_reward: float = FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
    ) -> None:
        if not bundles:
            raise ValueError("MemoryArenaWebShopEnv requires at least one bundle.")
        self.bundles = tuple(bundles)
        self.backend = backend
        self.env_uid = env_uid or uuid.uuid4().hex[:12]
        self._reward_contract = build_memoryarena_reward_contract(
            first_valid_add_reward=first_valid_add_reward,
            first_valid_later_session_retrieve_reward=(
                first_valid_later_session_retrieve_reward
            ),
        )
        self.first_valid_add_reward = float(
            self._reward_contract["first_valid_add_reward"]
        )
        self.first_valid_later_session_retrieve_reward = float(
            self._reward_contract["first_valid_later_session_retrieve_reward"]
        )
        self.episode_counter = 0
        self.bundle: MemoryArenaBundle | None = None
        self.data_idx = 0
        self.current_session_index = 0
        self.step_count = 0
        self.spent_cents = 0
        self.status = "idle"
        self.done = False
        self.native_session_token: str | None = None
        self.native_page: NativePage | None = None
        self.long_term_memory: dict[str, MemoryEntry] = {}
        self.memory_id_counter = 0
        self.active_context: list[str] = []
        self.session_trace: list[str] = []
        self.purchase_ledger: list[dict[str, Any]] = []
        self.last_tool_ops: list[dict[str, Any]] = []
        self.last_reward_components: list[dict[str, Any]] = []
        self.last_memory_diff = _empty_memory_diff()
        self.valid_zero_reward_action_counts_this_session: dict[str, int] = {}
        self.rewarded_memory_ops_this_session: set[str] = set()

    def reset(self, seed: int | None = None, data_idx: int = 0):
        del seed
        self._close_native_session()
        self.data_idx = int(data_idx)
        self.bundle = self.bundles[self.data_idx % len(self.bundles)]
        if len(self.bundle.questions) != 6 or len(self.bundle.target_asins) != 6:
            raise ValueError(f"Bundle {self.bundle.task_id!r} is not a six-session chain.")
        self.episode_counter += 1
        self.current_session_index = 0
        self.step_count = 0
        self.spent_cents = 0
        self.status = "active"
        self.done = False
        self.long_term_memory = {}
        self.memory_id_counter = 0
        self.active_context = []
        self.session_trace = []
        self.purchase_ledger = []
        self.last_tool_ops = []
        self.last_reward_components = []
        self.last_memory_diff = _empty_memory_diff()
        self._reset_session_reward_state()
        self.native_page = self.backend.open_session(
            self._new_native_session_token(),
            self.bundle.questions[0],
        )
        return self.render_observation(), self.build_info()

    def step(self, action: str):
        if self.done:
            return self.render_terminal_observation("Episode is already done."), 0.0, True, False, self.build_info()

        self.step_count += 1
        self.last_tool_ops = []
        self.last_reward_components = []
        self.last_memory_diff = _empty_memory_diff()
        action_text = action.strip() if isinstance(action, str) else repr(action)
        parsed: ParsedAction | None = None
        action_executed = False
        try:
            parsed = parse_mixed_action(action)
            if parsed.is_native:
                observation, reward, done = self._step_native(parsed)
            else:
                observation, reward, done = self._step_memory(parsed)
            action_executed = True
        except InvalidNativeAction as exc:
            message = f"Invalid action: {exc}"
            self._append_trace(action_text, message)
            observation = self.render_observation(message)
            reward = INVALID_ACTION_PENALTY
            done = False
            attempted_op = parsed.op if parsed is not None else infer_raw_action_op(action_text)
            self.last_reward_components = [
                {
                    "name": "invalid_action",
                    "value": reward,
                    "op": attempted_op,
                    "step": self.step_count,
                    "raw_action": action_text,
                    "error": str(exc),
                }
            ]
        except NativeWebShopInfrastructureError as exc:
            self.status = "infra_error"
            self.done = True
            self.last_tool_ops = [
                {
                    "op": "INFRA_ERROR",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "sample_excluded": True,
                    "step": self.step_count,
                }
            ]
            observation = self.render_terminal_observation(
                "The shopping environment encountered an infrastructure error."
            )
            reward = 0.0
            done = True
            attempted_op = parsed.op if parsed is not None else infer_raw_action_op(action_text)
            self.last_reward_components = [
                {
                    "name": "infrastructure_error_excluded",
                    "value": reward,
                    "op": attempted_op,
                    "step": self.step_count,
                    "error_type": type(exc).__name__,
                }
            ]

        if not self.last_reward_components:
            self.last_reward_components = build_reward_components(
                raw_action=action_text,
                reward=reward,
                step=self.step_count,
                tool_ops=self.last_tool_ops,
            )
        if action_executed and parsed is not None:
            reward = self._apply_session_action_shaping(
                parsed=parsed,
                raw_action=action_text,
                reward=reward,
                done=done,
            )

        self.done = bool(done)
        return observation, float(reward), self.done, False, self.build_info()

    def close(self) -> None:
        self._close_native_session()
        self.status = "closed"

    def render_observation(self, prefix: str | None = None) -> str:
        self._require_bundle()
        page = self._require_page()
        sections: list[str] = []
        if prefix:
            sections.append(prefix.strip())
        sections.extend(
            [
                f"Task family: bundled_shopping\nProgress: {self.current_session_index}/6",
                page.observation.strip(),
                _render_native_actions(page),
                _render_context(self.active_context, self.session_trace),
                _memory_action_contract(),
            ]
        )
        return "\n\n".join(section for section in sections if section)

    def render_terminal_observation(self, message: str) -> str:
        return "\n\n".join(
            [
                message.strip(),
                f"Task family: bundled_shopping\nProgress: {self.current_session_index}/6",
            ]
        )

    def build_info(self) -> dict[str, Any]:
        bundle = self._require_bundle()
        phase_count = len(bundle.questions)
        subtask_count = len(bundle.target_asins)
        return {
            "task_id": bundle.task_id,
            "task_family": "bundled_shopping",
            "split": bundle.split,
            "source": "memoryarena_original_webshop",
            "surface": self.surface,
            "progress_score": self.current_session_index / float(phase_count),
            "episode_success": self.status == "success",
            "status": self.status,
            "current_subtask_index": self.current_session_index,
            "phase_count": phase_count,
            "subtask_count": subtask_count,
            "tool_ops": list(self.last_tool_ops),
            "reward_components": [dict(item) for item in self.last_reward_components],
            "memory_ops": [item for item in self.last_tool_ops if item.get("op") in MEMORY_TOOL_OPS],
            "memory_state_diff": self.last_memory_diff,
            "purchase_history": list(self.purchase_ledger),
            "session_trace": list(self.session_trace),
            "reward_contract": self.reward_contract(),
            "sample_excluded": self.status == "infra_error",
        }

    def reward_contract(self) -> dict[str, Any]:
        return dict(self._reward_contract)

    def _step_native(self, parsed: ParsedAction) -> tuple[str, float, bool]:
        if self.native_session_token is None:
            raise NativeWebShopInfrastructureError("Native session is not open.")
        try:
            page = self.backend.step(self.native_session_token, parsed.raw_action)
        except Exception as exc:
            raise NativeWebShopInfrastructureError(
                f"Native WebShop step failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.native_page = page
        if page.purchase is None:
            tool_op = {
                "op": parsed.op,
                "raw_action": parsed.raw_action,
                "step": self.step_count,
            }
            if parsed.op == "SEARCH":
                tool_op["result_count"] = sum(
                    1 for value in page.clickables
                    if self.backend.has_product(str(value))
                )
            self.last_tool_ops = [tool_op]
            self._append_trace(parsed.raw_action, page.observation)
            return self.render_observation(), 0.0, False

        if parsed.raw_action.lower() != "click[buy now]":
            raise NativeWebShopInfrastructureError("Native backend committed a purchase for a non-purchase action.")
        return self._commit_purchase(parsed.raw_action, page)

    def _apply_session_action_shaping(
        self,
        *,
        parsed: ParsedAction,
        raw_action: str,
        reward: float,
        done: bool,
    ) -> float:
        if done or float(reward) != 0.0:
            return float(reward)

        occurrence = self.valid_zero_reward_action_counts_this_session.get(raw_action, 0) + 1
        self.valid_zero_reward_action_counts_this_session[raw_action] = occurrence

        component_name: str | None = None
        component_value = 0.0
        if parsed.op == "ADD" and "ADD" not in self.rewarded_memory_ops_this_session:
            self.rewarded_memory_ops_this_session.add("ADD")
            component_name = "memory_add_first_valid_this_session"
            component_value = self.first_valid_add_reward
        elif (
            parsed.op == "RETRIEVE"
            and self.current_session_index >= 1
            and "RETRIEVE" not in self.rewarded_memory_ops_this_session
        ):
            self.rewarded_memory_ops_this_session.add("RETRIEVE")
            component_name = "memory_retrieve_first_valid_later_session"
            component_value = self.first_valid_later_session_retrieve_reward
        elif occurrence >= 2:
            component_name = "exact_repeated_valid_zero_reward_action"
            component_value = EXACT_REPEAT_ACTION_PENALTY

        if component_name is None:
            return float(reward)
        self.last_reward_components.append(
            {
                "name": component_name,
                "value": component_value,
                "op": parsed.op,
                "step": self.step_count,
                "raw_action": raw_action,
                "session_index": self.current_session_index,
                "occurrence": occurrence,
            }
        )
        return float(reward) + component_value

    def _commit_purchase(self, raw_action: str, page: NativePage) -> tuple[str, float, bool]:
        bundle = self._require_bundle()
        purchase = page.purchase
        if purchase is None:
            raise NativeWebShopInfrastructureError("Purchase event is missing structured state.")
        expected_asin = bundle.target_asins[self.current_session_index].upper()
        actual_asin = purchase.asin.upper()
        new_spent_cents = self.spent_cents + purchase.price_cents
        budget_ok = new_spent_cents <= bundle.budget_cents
        purchase_correct = actual_asin == expected_asin and budget_ok
        final_purchase = purchase_correct and self.current_session_index == 5
        event = {
            "op": "BUY",
            "raw_action": raw_action,
            "actual_asin": actual_asin,
            "actual_price_cents": purchase.price_cents,
            "selected_options": dict(purchase.selected_options),
            "committed": True,
            "purchase_correct": purchase_correct,
            "budget_ok": budget_ok,
            "session_advanced": purchase_correct,
            "terminal": (not purchase_correct) or final_purchase,
            "step": self.step_count,
            "session_index": self.current_session_index,
        }
        self.purchase_ledger.append(dict(event))
        self.last_tool_ops = [event]

        if not purchase_correct:
            self.status = "failed_purchase"
            self._close_native_session()
            return (
                self.render_terminal_observation("The shopping episode has ended."),
                WRONG_BUY_TERMINAL_FAILURE,
                True,
            )

        self.spent_cents = new_spent_cents
        self.current_session_index += 1
        self._close_native_session()
        self.active_context = []
        self.session_trace = []
        self._reset_session_reward_state()
        if final_purchase:
            self.status = "success"
            self.native_page = page
            return self.render_terminal_observation("The bundled shopping task is complete."), 2.0, True

        self.native_page = self.backend.open_session(
            self._new_native_session_token(),
            bundle.questions[self.current_session_index],
        )
        return self.render_observation("Purchase recorded. The next shopping session is ready."), 1.0, False

    def _step_memory(self, parsed: ParsedAction) -> tuple[str, float, bool]:
        payload = parsed.payload or {}
        handlers = {
            "ADD": self._memory_add,
            "UPDATE": self._memory_update,
            "DELETE": self._memory_delete,
            "RETRIEVE": self._memory_retrieve,
            "SUMMARY": self._memory_summary,
            "FILTER": self._memory_filter,
        }
        message, event = handlers[parsed.op](payload)
        event["step"] = self.step_count
        self.last_tool_ops = [event]
        self._append_trace(parsed.raw_action, message)
        return self.render_observation(message), 0.0, False

    def _memory_add(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _require_exact_fields(payload, required={"key", "value"})
        key = _require_text(payload, "key")
        value = _require_text(payload, "value")
        memory_id = f"mem_{self.memory_id_counter:04d}"
        self.memory_id_counter += 1
        entry = MemoryEntry(memory_id, key, value, self.step_count, self.step_count)
        self.long_term_memory[memory_id] = entry
        self.last_memory_diff["added"].append(_memory_dict(entry))
        return f"Stored memory [{memory_id}] {key}: {value}", {"op": "ADD", "memory_id": memory_id, "key": key}

    def _memory_update(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _require_exact_fields(payload, required={"memory_id", "value"}, optional={"key"})
        entry = self._require_memory(_require_text(payload, "memory_id"))
        before = _memory_dict(entry)
        if "key" in payload:
            entry.key = _require_text(payload, "key")
        entry.value = _require_text(payload, "value")
        entry.updated_step = self.step_count
        self.last_memory_diff["updated"].append({"before": before, "after": _memory_dict(entry)})
        self._refresh_active_entry(entry)
        return f"Updated memory [{entry.memory_id}] {entry.key}: {entry.value}", {"op": "UPDATE", "memory_id": entry.memory_id, "key": entry.key}

    def _memory_delete(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _require_exact_fields(payload, required={"memory_id"})
        entry = self._require_memory(_require_text(payload, "memory_id"))
        del self.long_term_memory[entry.memory_id]
        self.active_context = [
            item for item in self.active_context if not item.startswith(f"[{entry.memory_id}] ")
        ]
        self.last_memory_diff["deleted"].append(_memory_dict(entry))
        return f"Deleted memory {entry.memory_id}.", {"op": "DELETE", "memory_id": entry.memory_id, "key": entry.key}

    def _memory_retrieve(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _require_exact_fields(payload, required={"query"}, optional={"top_k"})
        query = _require_text(payload, "query")
        top_k = payload.get("top_k", 3)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise InvalidNativeAction("RETRIEVE top_k must be an integer from 1 to 20.")
        ranked = rank_memory_entries_bm25(query, list(self.long_term_memory.values()), top_k=top_k)
        entries = [entry for entry, score in ranked if score > 0]
        for entry in entries:
            entry.access_count += 1
        self.active_context = [_render_memory(entry) for entry in entries]
        if entries:
            message = "Retrieved memories:\n" + "\n".join(self.active_context)
        elif self.long_term_memory:
            message = "No relevant memory retrieved."
        else:
            message = "No relevant memory retrieved. Long-term memory is empty."
        return message, {
            "op": "RETRIEVE",
            "query": query,
            "top_k": top_k,
            "retrieved_memory_ids": [entry.memory_id for entry in entries],
            "retrieved_count": len(entries),
        }

    def _memory_summary(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _require_exact_fields(payload, required={"text", "source_ids"})
        text = _require_text(payload, "text")
        source_ids = _require_string_list(payload, "source_ids")
        if not source_ids:
            raise InvalidNativeAction("SUMMARY source_ids must not be empty.")
        self._resolve_context_ids(source_ids)
        self.active_context = [f"Summary: {text}"]
        return "Active context replaced by the policy-authored summary.", {
            "op": "SUMMARY",
            "source_ids": source_ids,
        }

    def _memory_filter(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _require_exact_fields(payload, required=set(), optional={"keep_ids", "drop_ids", "scope"})
        keep_ids = payload.get("keep_ids")
        drop_ids = payload.get("drop_ids")
        if (keep_ids is None) == (drop_ids is None):
            raise InvalidNativeAction("FILTER expects exactly one of keep_ids or drop_ids.")
        scope = payload.get("scope", "active")
        if scope not in {"active", "session", "all"}:
            raise InvalidNativeAction("FILTER scope must be active, session, or all.")
        selected = _require_string_list(payload, "keep_ids" if keep_ids is not None else "drop_ids")
        self._resolve_context_ids(selected)
        keep_mode = keep_ids is not None
        removed = 0
        if scope in {"active", "all"}:
            self.active_context, count = _filter_indexed(self.active_context, "C", selected, keep_mode)
            removed += count
        if scope in {"session", "all"}:
            self.session_trace, count = _filter_indexed(self.session_trace, "S", selected, keep_mode)
            removed += count
        return f"Filtered visible context; removed {removed} item(s).", {
            "op": "FILTER",
            "scope": scope,
            "keep_ids": selected if keep_mode else [],
            "drop_ids": [] if keep_mode else selected,
            "removed": removed,
        }

    def _resolve_context_ids(self, context_ids: Sequence[str]) -> list[str]:
        values: list[str] = []
        for context_id in context_ids:
            match = re.fullmatch(r"([CS])(\d+)", context_id)
            if match is None:
                raise InvalidNativeAction(f"Invalid context id {context_id!r}.")
            items = self.active_context if match.group(1) == "C" else self.session_trace
            index = int(match.group(2))
            if index >= len(items):
                raise InvalidNativeAction(f"Unknown context id {context_id!r}.")
            values.append(items[index])
        return values

    def _refresh_active_entry(self, entry: MemoryEntry) -> None:
        prefix = f"[{entry.memory_id}] "
        self.active_context = [
            _render_memory(entry) if item.startswith(prefix) else item
            for item in self.active_context
        ]

    def _require_memory(self, memory_id: str) -> MemoryEntry:
        try:
            return self.long_term_memory[memory_id]
        except KeyError as exc:
            raise InvalidNativeAction(f"Unknown memory_id {memory_id!r}.") from exc

    def _append_trace(self, action: str, result: str) -> None:
        self.session_trace.append(f"Action: {action}\nResult: {result}")

    def _reset_session_reward_state(self) -> None:
        self.valid_zero_reward_action_counts_this_session = {}
        self.rewarded_memory_ops_this_session = set()

    def _new_native_session_token(self) -> str:
        token = f"amg_{self.env_uid}_{self.episode_counter}_{self.current_session_index}"
        self.native_session_token = token
        return token

    def _close_native_session(self) -> None:
        token = self.native_session_token
        self.native_session_token = None
        if token is not None:
            self.backend.close_session(token)

    def _require_bundle(self) -> MemoryArenaBundle:
        if self.bundle is None:
            raise RuntimeError("Environment must be reset before use.")
        return self.bundle

    def _require_page(self) -> NativePage:
        if self.native_page is None:
            raise RuntimeError("Native WebShop page is unavailable.")
        return self.native_page


def parse_mixed_action(action: str) -> ParsedAction:
    if not isinstance(action, str):
        raise InvalidNativeAction(f"Action must be a string, got {type(action).__name__}.")
    text = action.strip()
    native_match = NATIVE_ACTION_RE.fullmatch(text)
    if native_match is not None:
        argument = native_match.group(2).strip()
        if not argument:
            raise InvalidNativeAction(f"{native_match.group(1)} argument must be non-empty.")
        raw_action = f"{native_match.group(1)}[{argument}]"
        return ParsedAction(op=native_match.group(1).upper(), raw_action=raw_action)

    memory_match = MEMORY_ACTION_RE.fullmatch(text)
    if memory_match is None:
        raise InvalidNativeAction(
            "Expected one native search[...] / click[...] action or one uppercase memory-tool JSON action."
        )
    try:
        payload = json.loads(memory_match.group(2))
    except json.JSONDecodeError as exc:
        raise InvalidNativeAction(f"Memory-tool payload must be valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise InvalidNativeAction("Memory-tool payload must be a JSON object.")
    return ParsedAction(op=memory_match.group(1), raw_action=text, payload=payload)


def _render_native_actions(page: NativePage) -> str:
    lines = ["Native WebShop actions currently available:"]
    if page.has_search_bar:
        lines.append("- search[keywords]")
    lines.extend(f"- click[{value}]" for value in page.clickables)
    return "\n".join(lines)


def _render_context(active: Sequence[str], trace: Sequence[str]) -> str:
    lines = ["Current-session trace:"]
    lines.extend(f"- S{index}: {item}" for index, item in enumerate(trace))
    if not trace:
        lines.append("<empty>")
    lines.append("Active retrieved/summary context:")
    lines.extend(f"- C{index}: {item}" for index, item in enumerate(active))
    if not active:
        lines.append("<empty>")
    return "\n".join(lines)


def _memory_action_contract() -> str:
    return "\n".join(
        [
            "Memory actions:",
            'ADD {"key": "...", "value": "..."}',
            'UPDATE {"memory_id": "mem_0000", "value": "..."}',
            'DELETE {"memory_id": "mem_0000"}',
            'RETRIEVE {"query": "...", "top_k": 3}',
            'SUMMARY {"text": "...", "source_ids": ["S0", "C0"]}',
            'FILTER {"keep_ids": ["C0"], "scope": "active"}',
        ]
    )


def _require_exact_fields(
    payload: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    fields = set(payload)
    missing = required - fields
    extra = fields - required - optional
    if missing:
        raise InvalidNativeAction(f"Missing field(s): {', '.join(sorted(missing))}.")
    if extra:
        raise InvalidNativeAction(f"Unexpected field(s): {', '.join(sorted(extra))}.")


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidNativeAction(f"Field {key!r} must be a non-empty string.")
    return value


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InvalidNativeAction(f"Field {key!r} must be a list of strings.")
    if len(value) != len(set(value)):
        raise InvalidNativeAction(f"Field {key!r} must not contain duplicate ids.")
    return value


def _memory_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "memory_id": entry.memory_id,
        "key": entry.key,
        "value": entry.value,
        "created_step": entry.created_step,
        "updated_step": entry.updated_step,
        "access_count": entry.access_count,
    }


def _render_memory(entry: MemoryEntry) -> str:
    return f"[{entry.memory_id}] {entry.key}: {entry.value}"


def _empty_memory_diff() -> dict[str, list[dict[str, Any]]]:
    return {"added": [], "updated": [], "deleted": []}


def _filter_indexed(
    items: Sequence[str],
    prefix: str,
    selected_ids: Sequence[str],
    keep_mode: bool,
) -> tuple[list[str], int]:
    selected = set(selected_ids)
    kept = [
        item
        for index, item in enumerate(items)
        if ((f"{prefix}{index}" in selected) == keep_mode)
    ]
    return kept, len(items) - len(kept)
