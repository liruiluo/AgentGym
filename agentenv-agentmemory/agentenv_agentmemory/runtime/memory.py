from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..memory_state import MemoryEntry, rank_memory_entries_bm25
MEMORY_TOOL_OPS = ("ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER")
_MEMORY_ACTION_RE = re.compile(
    r"\A(" + "|".join(MEMORY_TOOL_OPS) + r")\s+(\{.*\})\Z",
    flags=re.DOTALL,
)


class MemoryActionError(ValueError):
    pass


@dataclass(frozen=True)
class MemoryRewardPolicy:
    first_add: float = 0.0
    first_later_phase_retrieve: float = 0.0
    exact_repeat: float = 0.0


@dataclass(frozen=True)
class MemoryActionResult:
    message: str
    reward: float
    op: str
    tool_op: dict[str, Any]
    reward_components: tuple[dict[str, Any], ...]
    state_diff: dict[str, list[Any]]


@dataclass
class MemoryToolRuntime:
    """Policy-authored LTM plus phase-local visible context."""

    reward_policy: MemoryRewardPolicy = field(default_factory=MemoryRewardPolicy)
    long_term_memory: dict[str, MemoryEntry] = field(default_factory=dict)
    active_context: list[str] = field(default_factory=list)
    session_trace: list[str] = field(default_factory=list)
    memory_id_counter: int = 0
    action_counts_this_phase: dict[str, int] = field(default_factory=dict)
    rewarded_ops_this_phase: set[str] = field(default_factory=set)

    def reset_episode(self) -> None:
        self.long_term_memory.clear()
        self.active_context.clear()
        self.session_trace.clear()
        self.memory_id_counter = 0
        self._reset_phase_reward_state()

    def advance_phase(self) -> None:
        self.active_context.clear()
        self.session_trace.clear()
        self._reset_phase_reward_state()

    def append_trace(self, action: str, result: str) -> None:
        self.session_trace.append(f"Action: {action}\nResult: {result}")

    def apply(
        self,
        action: str,
        *,
        env_step: int,
        phase_index: int,
    ) -> MemoryActionResult | None:
        parsed = parse_memory_action(action)
        if parsed is None:
            return None
        op, payload = parsed
        state_diff = _empty_state_diff()
        handler = {
            "ADD": self._add,
            "UPDATE": self._update,
            "DELETE": self._delete,
            "RETRIEVE": self._retrieve,
            "SUMMARY": self._summary,
            "FILTER": self._filter,
        }[op]
        message, event = handler(payload, env_step, state_diff)
        event.update({"op": op, "step": env_step})
        base_component = {
            "name": f"memory_{op.lower()}_transition",
            "value": 0.0,
            "op": op,
            "step": env_step,
        }
        shaped_reward, shaping = self.shape_valid_zero_reward_action(
            action,
            op=op,
            env_step=env_step,
            phase_index=phase_index,
        )
        components = [base_component]
        if shaping is not None:
            components.append(shaping)
        self.append_trace(action, message)
        return MemoryActionResult(
            message=message,
            reward=shaped_reward,
            op=op,
            tool_op=event,
            reward_components=tuple(components),
            state_diff=state_diff,
        )

    def shape_valid_zero_reward_action(
        self,
        action: str,
        *,
        op: str,
        env_step: int,
        phase_index: int,
    ) -> tuple[float, dict[str, Any] | None]:
        occurrence = self.action_counts_this_phase.get(action, 0) + 1
        self.action_counts_this_phase[action] = occurrence
        name = None
        value = 0.0
        if op == "ADD" and op not in self.rewarded_ops_this_phase:
            self.rewarded_ops_this_phase.add(op)
            name = "memory_add_first_valid_this_phase"
            value = self.reward_policy.first_add
        elif (
            op == "RETRIEVE"
            and phase_index >= 1
            and op not in self.rewarded_ops_this_phase
        ):
            self.rewarded_ops_this_phase.add(op)
            name = "memory_retrieve_first_valid_later_phase"
            value = self.reward_policy.first_later_phase_retrieve
        elif occurrence >= 2:
            name = "exact_repeated_valid_zero_reward_action"
            value = self.reward_policy.exact_repeat
        if name is None or value == 0.0:
            return 0.0, None
        return value, {
            "name": name,
            "value": value,
            "op": op,
            "step": env_step,
            "phase_index": phase_index,
            "occurrence": occurrence,
            "submitted_action": action,
        }

    def render_context(self) -> str:
        lines = ["Current-phase trace:"]
        lines.extend(f"- S{index}: {item}" for index, item in enumerate(self.session_trace))
        if not self.session_trace:
            lines.append("<empty>")
        lines.append("Active retrieved/summary context:")
        lines.extend(f"- C{index}: {item}" for index, item in enumerate(self.active_context))
        if not self.active_context:
            lines.append("<empty>")
        return "\n".join(lines)

    def _add(self, payload, env_step, diff):
        _require_fields(payload, required={"key", "value"})
        key = _require_text(payload, "key")
        value = _require_text(payload, "value")
        memory_id = f"mem_{self.memory_id_counter:04d}"
        self.memory_id_counter += 1
        entry = MemoryEntry(memory_id, key, value, env_step, env_step)
        self.long_term_memory[memory_id] = entry
        diff["added"].append(_memory_dict(entry))
        return f"Stored memory [{memory_id}] {key}: {value}", {
            "memory_id": memory_id,
            "key": key,
        }

    def _update(self, payload, env_step, diff):
        _require_fields(payload, required={"memory_id", "value"}, optional={"key"})
        entry = self._require_memory(_require_text(payload, "memory_id"))
        before = _memory_dict(entry)
        if "key" in payload:
            entry.key = _require_text(payload, "key")
        entry.value = _require_text(payload, "value")
        entry.updated_step = env_step
        after = _memory_dict(entry)
        diff["updated"].append({"before": before, "after": after})
        self._refresh_active_entry(entry)
        return f"Updated memory [{entry.memory_id}] {entry.key}: {entry.value}", {
            "memory_id": entry.memory_id,
            "key": entry.key,
        }

    def _delete(self, payload, env_step, diff):
        del env_step
        _require_fields(payload, required={"memory_id"})
        entry = self._require_memory(_require_text(payload, "memory_id"))
        del self.long_term_memory[entry.memory_id]
        self.active_context = [
            item
            for item in self.active_context
            if not item.startswith(f"[{entry.memory_id}] ")
        ]
        diff["deleted"].append(_memory_dict(entry))
        return f"Deleted memory {entry.memory_id}.", {
            "memory_id": entry.memory_id,
            "key": entry.key,
        }

    def _retrieve(self, payload, env_step, diff):
        del env_step, diff
        _require_fields(payload, required={"query"}, optional={"top_k"})
        query = _require_text(payload, "query")
        top_k = payload.get("top_k", 3)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise MemoryActionError("RETRIEVE top_k must be an integer from 1 to 20")
        ranked = rank_memory_entries_bm25(
            query,
            list(self.long_term_memory.values()),
            top_k=top_k,
        )
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
            "query": query,
            "top_k": top_k,
            "retrieved_memory_ids": [entry.memory_id for entry in entries],
            "retrieved_count": len(entries),
        }

    def _summary(self, payload, env_step, diff):
        del env_step, diff
        _require_fields(payload, required={"text", "source_ids"})
        text = _require_text(payload, "text")
        source_ids = _require_string_list(payload, "source_ids")
        if not source_ids:
            raise MemoryActionError("SUMMARY source_ids must not be empty")
        self._resolve_context_ids(source_ids)
        self.active_context = [f"Summary: {text}"]
        return "Active context replaced by the policy-authored summary.", {
            "source_ids": source_ids,
        }

    def _filter(self, payload, env_step, diff):
        del env_step, diff
        _require_fields(
            payload,
            required=set(),
            optional={"keep_ids", "drop_ids", "scope"},
        )
        keep_ids = payload.get("keep_ids")
        drop_ids = payload.get("drop_ids")
        if (keep_ids is None) == (drop_ids is None):
            raise MemoryActionError("FILTER expects exactly one of keep_ids or drop_ids")
        scope = payload.get("scope", "active")
        if scope not in {"active", "session", "all"}:
            raise MemoryActionError("FILTER scope must be active, session, or all")
        field_name = "keep_ids" if keep_ids is not None else "drop_ids"
        selected = _require_string_list(payload, field_name)
        self._resolve_context_ids(selected)
        keep_mode = keep_ids is not None
        removed = 0
        if scope in {"active", "all"}:
            self.active_context, count = _filter_indexed(
                self.active_context,
                "C",
                selected,
                keep_mode,
            )
            removed += count
        if scope in {"session", "all"}:
            self.session_trace, count = _filter_indexed(
                self.session_trace,
                "S",
                selected,
                keep_mode,
            )
            removed += count
        return f"Filtered visible context; removed {removed} item(s).", {
            "scope": scope,
            "keep_ids": selected if keep_mode else [],
            "drop_ids": [] if keep_mode else selected,
            "removed": removed,
        }

    def _resolve_context_ids(self, context_ids: Sequence[str]) -> list[str]:
        values = []
        for context_id in context_ids:
            match = re.fullmatch(r"([CS])(\d+)", context_id)
            if match is None:
                raise MemoryActionError(f"Invalid context id {context_id!r}")
            items = self.active_context if match.group(1) == "C" else self.session_trace
            index = int(match.group(2))
            if index >= len(items):
                raise MemoryActionError(f"Unknown context id {context_id!r}")
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
            raise MemoryActionError(f"Unknown memory_id {memory_id!r}") from exc

    def _reset_phase_reward_state(self) -> None:
        self.action_counts_this_phase.clear()
        self.rewarded_ops_this_phase.clear()


def parse_memory_action(action: str) -> tuple[str, dict[str, Any]] | None:
    text = action.strip()
    match = _MEMORY_ACTION_RE.fullmatch(text)
    if match is None:
        prefix = text.split(None, 1)[0] if text else ""
        if prefix in MEMORY_TOOL_OPS:
            raise MemoryActionError(
                "Memory action must use an uppercase operation followed by one JSON object"
            )
        return None
    try:
        payload = json.loads(match.group(2))
    except json.JSONDecodeError as exc:
        raise MemoryActionError(f"Memory payload must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MemoryActionError("Memory payload must be a JSON object")
    return match.group(1), payload


def _empty_state_diff() -> dict[str, list[Any]]:
    return {"added": [], "updated": [], "deleted": []}


def _require_fields(payload, *, required, optional=None):
    optional = optional or set()
    fields = set(payload)
    missing = required - fields
    extra = fields - required - optional
    if missing:
        raise MemoryActionError(f"Missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise MemoryActionError(f"Unexpected field(s): {', '.join(sorted(extra))}")


def _require_text(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MemoryActionError(f"{key} must be a non-empty string")
    return value.strip()


def _require_string_list(payload, key):
    value = payload.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise MemoryActionError(f"{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _memory_dict(entry):
    return {
        "memory_id": entry.memory_id,
        "key": entry.key,
        "value": entry.value,
        "created_step": entry.created_step,
        "updated_step": entry.updated_step,
        "access_count": entry.access_count,
    }


def _render_memory(entry):
    return f"[{entry.memory_id}] {entry.key}: {entry.value}"


def _filter_indexed(items, prefix, selected, keep_mode):
    selected_set = set(selected)
    kept = []
    for index, item in enumerate(items):
        included = f"{prefix}{index}" in selected_set
        if included == keep_mode:
            kept.append(item)
    return kept, len(items) - len(kept)
