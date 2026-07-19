from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog_search import search_sqlite_catalog
from .memory_state import MemoryEntry, rank_memory_entries_bm25, tokenize, tokenize_terms


MEMORY_TOOL_OPS = {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    attributes: dict[str, Any]

    def render(self) -> str:
        attrs = ", ".join(f"{key}={render_attr_value(value)}" for key, value in self.attributes.items())
        return f"- {self.product_id}: {self.title} ({attrs})"


@dataclass(frozen=True)
class ShoppingSubtask:
    instruction: str
    candidate_products: tuple[Product, ...]
    target_product_id: str


@dataclass(frozen=True)
class InitialMemorySpec:
    key: str
    value: str
    product_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShoppingTask:
    task_id: str
    title: str
    subtasks: tuple[ShoppingSubtask, ...]
    split: str = "train"
    source: str = "handcrafted_v0"
    difficulty: str = "smoke"
    memory_dependency: str = "cross_session_product_attribute"
    start_subtask_index: int = 0
    initial_purchase_product_ids: tuple[str, ...] = ()
    initial_memories: tuple[InitialMemorySpec, ...] = ()
    curriculum_flags: frozenset[str] = field(default_factory=frozenset)


@dataclass
class StepInfo:
    task_id: str = ""
    task_family: str = "bundled_shopping"
    split: str = "train"
    source: str = "handcrafted_v0"
    difficulty: str = "smoke"
    memory_dependency: str = "cross_session_product_attribute"
    progress_score: float = 0.0
    episode_success: bool = False
    tool_ops: list[dict[str, Any]] = field(default_factory=list)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    memory_state_diff: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    compatibility_violations: list[dict[str, Any]] = field(default_factory=list)
    purchase_history: list[dict[str, Any]] = field(default_factory=list)
    current_subtask_index: int = 0
    session_trace: list[str] = field(default_factory=list)
    reward_components: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "split": self.split,
            "source": self.source,
            "difficulty": self.difficulty,
            "memory_dependency": self.memory_dependency,
            "progress_score": self.progress_score,
            "episode_success": self.episode_success,
            "tool_ops": self.tool_ops,
            "memory_ops": self.memory_ops,
            "memory_state_diff": self.memory_state_diff,
            "compatibility_violations": self.compatibility_violations,
            "purchase_history": self.purchase_history,
            "current_subtask_index": self.current_subtask_index,
            "session_trace": self.session_trace,
            "reward_components": self.reward_components,
        }


class InvalidAction(ValueError):
    pass


class AgentMemoryEnv:
    """Minimal memory-dependent bundled shopping environment.

    The environment keeps long-term memory hidden unless the policy explicitly
    calls RETRIEVE. It also exposes an automatic current-session trace as STM
    and clears that trace when a committed BUY advances to the next shopping
    session. Current v0 tasks are small handcrafted smoke items; they are
    placeholders for converted MemoryArena/WebShop-style data.
    """

    action_pattern = re.compile(r"^\s*([A-Za-z_]+)\s*(.*)\s*$", re.DOTALL)

    def __init__(
        self,
        tasks: list[ShoppingTask] | None = None,
        *,
        data_path: str | Path | None = None,
        split: str | None = None,
        split_dir: str | Path | None = None,
        catalog_index_path: str | Path | None = None,
        buy_semantics: str | None = None,
    ) -> None:
        self.tasks = tasks or load_task_dataset(data_path=data_path, split=split, split_dir=split_dir)
        resolved_catalog_index = catalog_index_path or os.environ.get("AGENTMEMORY_CATALOG_INDEX_PATH")
        self.catalog_index_path = Path(resolved_catalog_index) if resolved_catalog_index else None
        self.memory_shaping = parse_memory_shaping_mode(os.environ.get("AGENTMEMORY_MEMORY_SHAPING", "off"))
        self.buy_semantics = parse_buy_semantics(
            buy_semantics or os.environ.get("AGENTMEMORY_BUY_SEMANTICS", "terminate")
        )
        self.task: ShoppingTask | None = None
        self.data_idx = 0
        self.step_count = 0
        self.current_subtask_index = 0
        self.long_term_memory: dict[str, MemoryEntry] = {}
        self.short_term_context: list[str] = []
        self.session_trace: list[str] = []
        self.purchase_history: list[dict[str, Any]] = []
        self.bundle_state: dict[str, Any] = {}
        self.memory_product_refs: dict[str, set[str]] = {}
        self.retrieved_memory_ids_this_session: set[str] = set()
        self.repeated_active_retrieve_count_this_session = 0
        self.retrieve_query_counts_this_session: Counter[str] = Counter()
        self.empty_retrieve_count_this_session = 0
        self.rewarded_retrieve_subtasks: set[int] = set()
        self.searched_this_session = False
        self.search_query_counts_this_session: Counter[str] = Counter()
        self.no_result_search_count_this_session = 0
        self.rewarded_search_subtasks: set[int] = set()
        self.rejected_product_counts_this_session: Counter[str] = Counter()
        self.last_reward_components: list[dict[str, Any]] = []
        self.memory_id_counter = 0
        self.done = False
        self.last_info = StepInfo()

    def reset(self, seed: int | None = None, data_idx: int = 0):
        del seed  # Deterministic smoke tasks; data_idx selects the item.
        self.data_idx = data_idx
        self.task = self.tasks[data_idx % len(self.tasks)]
        self.step_count = 0
        self.current_subtask_index = 0
        self.long_term_memory = {}
        self.short_term_context = []
        self.session_trace = []
        self.purchase_history = []
        self.bundle_state = {}
        self.memory_product_refs = {}
        self.retrieved_memory_ids_this_session = set()
        self.repeated_active_retrieve_count_this_session = 0
        self.retrieve_query_counts_this_session = Counter()
        self.empty_retrieve_count_this_session = 0
        self.rewarded_retrieve_subtasks = set()
        self.searched_this_session = False
        self.search_query_counts_this_session = Counter()
        self.no_result_search_count_this_session = 0
        self.rewarded_search_subtasks = set()
        self.rejected_product_counts_this_session = Counter()
        self.last_reward_components = []
        self.memory_id_counter = 0
        self.apply_initial_task_state()
        self.done = False
        self.last_info = self.build_info(memory_state_diff=empty_memory_diff())
        return self.render_observation(), self.last_info.as_dict()

    def step(self, action: str):
        if self.done:
            return self.render_observation("Episode is already done."), 0.0, True, False, self.last_info.as_dict()

        self.step_count += 1
        self.last_reward_components = []
        old_subtask_index = self.current_subtask_index
        action_text = action.strip() if isinstance(action, str) else repr(action)
        op: str | None = None
        no_progress_feedback: str | None = None
        try:
            op, payload = self.parse_action(action)
            observation, reward, done, memory_diff, violations, tool_op = self.dispatch_action(op, payload)
        except InvalidAction as exc:
            invalid_message = str(exc)
            observation = self.render_observation(f"Invalid action: {invalid_message}")
            reward = -0.1
            if self.memory_shaping == "chain_v1" and self.invalid_context_reference_message(invalid_message):
                reward = -0.35
                self.last_reward_components.append(
                    {
                        "name": "invalid_memory_context_id_reference",
                        "value": reward,
                        "message": invalid_message,
                    }
                )
            done = False
            memory_diff = empty_memory_diff()
            violations = []
            tool_op = None

        if self.current_subtask_index == old_subtask_index and not done:
            if (
                self.memory_shaping == "chain_v1"
                and self.subtask_requires_prior_memory(old_subtask_index)
                and self.retrieved_memory_ids_this_session
                and op == "ANSWER"
            ):
                # Once the needed prior memory is already active in a dependent
                # shopping session, answering without completing the bundle is
                # a no-progress stall. Do not tax valid memory/context actions
                # here: ADD/UPDATE/DELETE/SUMMARY/FILTER may still be useful
                # memory refinement or context control, and broad negative
                # shaping would teach the policy to avoid memory tools. Spam is
                # handled by explicit duplicate/repeated no-op detectors and
                # functional failures such as dependent BUY without retrieval.
                stall_penalty = -0.05
                reward += stall_penalty
                self.last_reward_components.append(
                    {
                        "name": "dependent_memory_ready_answer_no_progress",
                        "value": stall_penalty,
                        "op": op,
                        "subtask_index": old_subtask_index,
                        "memory_ids": sorted(self.retrieved_memory_ids_this_session),
                    }
                )
                no_progress_feedback = f"No-progress feedback: {op} did not advance the session."
            result_text = extract_last_tool_result(observation)
            if no_progress_feedback:
                result_text = result_text.rstrip() + "\n" + no_progress_feedback
            self.session_trace.append(render_session_trace_entry(self.step_count, action_text, result_text))
            observation = self.render_observation(result_text)

        self.done = done
        tool_ops = [tool_op] if tool_op is not None else []
        self.last_info = self.build_info(
            memory_state_diff=memory_diff,
            compatibility_violations=violations,
            tool_ops=tool_ops,
        )
        return observation, reward, done, False, self.last_info.as_dict()

    def dispatch_action(self, op: str, payload: dict[str, Any]):
        if op == "ADD":
            return self.action_add(payload)
        if op == "UPDATE":
            return self.action_update(payload)
        if op == "DELETE":
            return self.action_delete(payload)
        if op == "RETRIEVE":
            return self.action_retrieve(payload)
        if op == "SUMMARY":
            return self.action_summary(payload)
        if op == "FILTER":
            return self.action_filter(payload)
        if op == "SEARCH":
            return self.action_search(payload)
        if op == "BUY":
            return self.action_buy(payload)
        if op == "ANSWER":
            return self.action_answer(payload)
        raise InvalidAction(f"Unsupported action '{op}'.")

    @staticmethod
    def invalid_context_reference_message(message: str) -> bool:
        """Return whether an invalid action misused visible memory/context ids.

        The current Qwen3 AgentMemory policy often copies `C0` or raw
        `mem_0000` into source-session BUY/UPDATE actions even when no context
        id is visible. This is a contract error that prevents the episode from
        reaching the cross-session memory task. Penalize this explicit invalid
        close-loop error more strongly than generic JSON typos, while keeping
        valid memory-tool calls untaxed.
        """

        return (
            "Context id" in message
            or "Unknown context id" in message
            or "Expected S0/S1 or C0/C1" in message
            or "Unknown memory_id 'C" in message
        )

    def action_add(self, payload: dict[str, Any]):
        key = require_str(payload, "key")
        value = require_str(payload, "value")
        referenced_product_ids = self.referenced_current_product_ids(f"{key} {value}")
        new_referenced_product_ids = [
            product_id for product_id in referenced_product_ids if not self.memory_product_refs.get(product_id)
        ]
        duplicate_referenced_product_ids = [
            product_id for product_id in referenced_product_ids if self.memory_product_refs.get(product_id)
        ]
        memory_id = f"mem_{self.memory_id_counter:04d}"
        self.memory_id_counter += 1
        entry = MemoryEntry(
            memory_id=memory_id,
            key=key,
            value=value,
            created_step=self.step_count,
            updated_step=self.step_count,
        )
        self.long_term_memory[memory_id] = entry
        self.link_memory_product_refs(memory_id, referenced_product_ids)
        reward = 0.0
        reward_event: dict[str, Any] | None = None
        if self.memory_shaping == "chain_v1":
            if duplicate_referenced_product_ids:
                # Re-ADDing the same visible product is no longer useful
                # memory formation; it is a no-op loop that prevents the
                # source-session BUY needed to make memory functional.
                # This is a duplicate/no-op penalty, not a blanket memory-tool
                # cost: the first valid ADD remains positive.
                reward -= 0.08
                reward_event = {
                    "name": "memory_add_duplicate_visible_product_reference",
                    "value": -0.08,
                    "product_ids": sorted(duplicate_referenced_product_ids),
                }
            elif new_referenced_product_ids:
                reward_value = 0.05
                reward += reward_value
                reward_event = {
                    "name": "memory_add_references_visible_product",
                    "value": reward_value,
                    "product_ids": sorted(new_referenced_product_ids),
                }
            else:
                # Do not make "calling memory tools" itself unattractive.
                # A policy-authored memory may be useful even when it does not
                # mention a visible product id verbatim; spam/no-op memory use
                # is handled by invalid/no-op diagnostics and downstream
                # missing-chain penalties, not by a blanket ADD cost.
                reward_event = {
                    "name": "memory_add_no_visible_product_reference",
                    "value": 0.0,
                }
        if reward_event is not None:
            self.last_reward_components.append(reward_event)
        diff = empty_memory_diff()
        diff["added"].append(memory_entry_dict(entry))
        memory_op = {
            "op": "ADD",
            "memory_id": memory_id,
            "key": key,
            "step": self.step_count,
            "referenced_product_ids": sorted(referenced_product_ids),
        }
        message = f"Stored memory {entry.render()}"
        return self.render_observation(message), reward, False, diff, [], memory_op

    def action_update(self, payload: dict[str, Any]):
        entry = self.find_memory_entry(payload)
        old_entry = memory_entry_dict(entry)
        old_product_ids = self.product_ids_for_memory(entry.memory_id)
        if "key" in payload:
            entry.key = require_str(payload, "key")
        entry.value = require_str(payload, "value")
        entry.updated_step = self.step_count
        referenced_product_ids = self.referenced_accessible_product_ids(
            f"{entry.key} {entry.value}",
            extra_product_ids=old_product_ids,
        )
        self.unlink_memory_product_refs(entry.memory_id)
        self.link_memory_product_refs(entry.memory_id, referenced_product_ids)
        self.refresh_active_memory_context(entry)
        diff = empty_memory_diff()
        diff["updated"].append({"before": old_entry, "after": memory_entry_dict(entry)})
        memory_op = {
            "op": "UPDATE",
            "memory_id": entry.memory_id,
            "key": entry.key,
            "step": self.step_count,
            "referenced_product_ids": sorted(referenced_product_ids),
        }
        message = f"Updated memory {entry.render()}"
        return self.render_observation(message), 0.0, False, diff, [], memory_op

    def action_delete(self, payload: dict[str, Any]):
        entry = self.find_memory_entry(payload)
        memory_id = entry.memory_id
        del self.long_term_memory[entry.memory_id]
        self.unlink_memory_product_refs(memory_id)
        self.retrieved_memory_ids_this_session.discard(memory_id)
        self.short_term_context = [
            item for item in self.short_term_context if not item.startswith(f"[{memory_id}] ")
        ]
        diff = empty_memory_diff()
        diff["deleted"].append(memory_entry_dict(entry))
        memory_op = {"op": "DELETE", "memory_id": entry.memory_id, "key": entry.key, "step": self.step_count}
        message = f"Deleted memory {entry.memory_id}."
        return self.render_observation(message), 0.0, False, diff, [], memory_op

    def action_retrieve(self, payload: dict[str, Any]):
        query = require_str(payload, "query")
        top_k = int(payload.get("top_k", 3))
        normalized_query = " ".join(query.lower().split())
        self.retrieve_query_counts_this_session[normalized_query] += 1
        ranked = rank_memory_entries_bm25(query, list(self.long_term_memory.values()), top_k=top_k)
        retrieved = [entry for entry, score in ranked if score > 0.0]
        retrieved_ids = {entry.memory_id for entry in retrieved}
        already_active_retrieve = bool(retrieved_ids) and retrieved_ids <= self.retrieved_memory_ids_this_session
        if already_active_retrieve:
            self.repeated_active_retrieve_count_this_session += 1
        else:
            self.repeated_active_retrieve_count_this_session = 0
        for entry in retrieved:
            entry.access_count += 1
        if retrieved:
            self.retrieved_memory_ids_this_session.update(retrieved_ids)
            self.short_term_context = [entry.render() for entry in retrieved]
            message = "Retrieved memories:\n" + "\n".join(self.short_term_context)
        else:
            self.short_term_context = []
            self.empty_retrieve_count_this_session += 1
            if not self.long_term_memory:
                message = "No relevant memory retrieved. Long-term memory is empty."
            else:
                message = "No relevant memory retrieved."
        reward = 0.0
        if self.memory_shaping == "chain_v1" and self.current_subtask_requires_prior_memory():
            if retrieved:
                if self.current_subtask_index not in self.rewarded_retrieve_subtasks:
                    reward += 0.06
                    self.rewarded_retrieve_subtasks.add(self.current_subtask_index)
                    self.last_reward_components.append(
                        {
                            "name": "memory_retrieve_nonempty_before_dependent_buy",
                            "value": 0.06,
                            "memory_ids": [entry.memory_id for entry in retrieved],
                            "subtask_index": self.current_subtask_index,
                        }
                    )
                elif already_active_retrieve:
                    # Repeated non-empty retrieval of memory that is already
                    # active in the same dependent session is a no-op loop, not
                    # useful memory use. Penalize only this explicit spam case;
                    # additional non-empty RETRIEVE calls that bring in new
                    # memory remain neutral so the policy does not learn to
                    # avoid memory tools altogether.
                    repeat_penalty = -min(0.03 * max(1, self.repeated_active_retrieve_count_this_session), 0.12)
                    reward += repeat_penalty
                    self.last_reward_components.append(
                        {
                            "name": "memory_retrieve_nonempty_repeat_same_session",
                            "value": repeat_penalty,
                            "memory_ids": [entry.memory_id for entry in retrieved],
                            "subtask_index": self.current_subtask_index,
                            "repeat_count": self.repeated_active_retrieve_count_this_session,
                        }
                    )
                else:
                    self.last_reward_components.append(
                        {
                            "name": "memory_retrieve_additional_nonempty_dependent_context",
                            "value": 0.0,
                            "memory_ids": [entry.memory_id for entry in retrieved],
                            "subtask_index": self.current_subtask_index,
                        }
                    )
            else:
                # Empty retrieval is diagnostic feedback, not a direct tool-use
                # penalty. The negative pressure belongs at the functional
                # failure point (e.g. a dependent BUY without retrieved memory)
                # so the policy does not learn to avoid RETRIEVE entirely.
                self.last_reward_components.append(
                    {"name": "memory_retrieve_empty_in_dependent_session", "value": 0.0}
                )
                if (
                    self.long_term_memory
                    and self.retrieve_query_counts_this_session[normalized_query] > 1
                ):
                    repeat_penalty = -0.04
                    reward += repeat_penalty
                    self.last_reward_components.append(
                        {
                            "name": "memory_retrieve_empty_repeat_same_query_noop",
                            "value": repeat_penalty,
                            "query": query,
                            "repeat_count": self.retrieve_query_counts_this_session[normalized_query],
                        }
                    )
        elif (
            self.memory_shaping == "chain_v1"
            and retrieved
            and self.subtask_can_provide_memory_for_later(self.current_subtask_index)
            and self.source_session_has_stored_current_product()
        ):
            # Retrieving the just-written source memory in the same source
            # session does not create a functional cross-session chain. Do not
            # penalize the first check, but do penalize exact repeated checks as
            # no-op/spam so the policy learns to close the source session with
            # BUY instead of farming same-session RETRIEVE.
            if already_active_retrieve:
                repeat_penalty = -min(0.05 * max(1, self.repeated_active_retrieve_count_this_session), 0.15)
                reward += repeat_penalty
                self.last_reward_components.append(
                    {
                        "name": "memory_retrieve_source_same_session_repeat_noop",
                        "value": repeat_penalty,
                        "memory_ids": [entry.memory_id for entry in retrieved],
                        "subtask_index": self.current_subtask_index,
                        "repeat_count": self.repeated_active_retrieve_count_this_session,
                    }
                )
            else:
                self.last_reward_components.append(
                    {
                        "name": "memory_retrieve_source_same_session_check",
                        "value": 0.0,
                        "memory_ids": [entry.memory_id for entry in retrieved],
                        "subtask_index": self.current_subtask_index,
                    }
                )
        memory_op = {
            "op": "RETRIEVE",
            "query": query,
            "top_k": top_k,
            "step": self.step_count,
            "retrieved_memory_ids": [entry.memory_id for entry in retrieved],
            "retrieved_count": len(retrieved),
        }
        return self.render_observation(message), reward, False, empty_memory_diff(), [], memory_op

    def action_summary(self, payload: dict[str, Any]):
        max_chars = clamp_int(payload.get("max_chars", 500), min_value=80, max_value=2000)
        span = str(payload.get("span", "session")).lower()
        source_ids = optional_str_list(payload, "source_ids")
        if source_ids is not None:
            validate_context_ids(
                source_ids,
                active_count=len(self.short_term_context),
                session_count=len(self.session_trace),
                scope="all",
            )
        if "text" in payload:
            text = require_str(payload, "text")
            summary_source = "policy_text"
        elif source_ids is not None:
            if not source_ids:
                raise InvalidAction("SUMMARY source_ids must not be empty.")
            text = "\n".join(context_items_by_ids(source_ids, self.short_term_context, self.session_trace))
            summary_source = "source_ids"
        else:
            if span not in {"session", "active", "all"}:
                raise InvalidAction("SUMMARY span must be one of: session, active, all.")
            source_items: list[str] = []
            if span in {"session", "all"}:
                source_items.extend(self.session_trace)
            if span in {"active", "all"}:
                source_items.extend(self.short_term_context)
            if not source_items:
                raise InvalidAction("SUMMARY needs text or non-empty selected context.")
            text = "\n".join(source_items)
            summary_source = span

        summary = compact_context_text(text, max_chars=max_chars)
        if not summary:
            raise InvalidAction("SUMMARY text/context must be non-empty.")
        self.short_term_context = [f"Summary ({summary_source}): {summary}"]
        tool_op = {
            "op": "SUMMARY",
            "source": summary_source,
            "span": span,
            "source_ids": source_ids or [],
            "step": self.step_count,
        }
        return self.render_observation("Active retrieved/summary context replaced by summary."), 0.0, False, empty_memory_diff(), [], tool_op

    def action_filter(self, payload: dict[str, Any]):
        scope = str(payload.get("scope", "active")).lower()
        if scope not in {"active", "session", "all"}:
            raise InvalidAction("FILTER scope must be one of: active, session, all.")

        keep_ids = optional_str_list(payload, "keep_ids")
        drop_ids = optional_str_list(payload, "drop_ids")
        has_query = "query" in payload
        mode_count = sum(item is not None for item in [keep_ids, drop_ids]) + int(has_query)
        if mode_count != 1:
            raise InvalidAction("FILTER expects exactly one of: keep_ids, drop_ids, query.")

        query = None
        removed = 0
        if keep_ids is not None or drop_ids is not None:
            ids = keep_ids if keep_ids is not None else drop_ids
            if ids is None:
                raise InvalidAction("FILTER keep_ids/drop_ids are missing.")
            if not ids:
                raise InvalidAction("FILTER keep_ids/drop_ids must not be empty.")
            validate_context_ids(
                ids,
                active_count=len(self.short_term_context),
                session_count=len(self.session_trace),
                scope=scope,
            )
            keep_mode = keep_ids is not None
            if scope in {"active", "all"}:
                kept, removed_active = filter_items_by_ids(
                    self.short_term_context,
                    prefix="C",
                    selected_ids=set(ids),
                    keep_mode=keep_mode,
                )
                self.short_term_context = kept
                removed += removed_active
            if scope in {"session", "all"}:
                kept, removed_session = filter_items_by_ids(
                    self.session_trace,
                    prefix="S",
                    selected_ids=set(ids),
                    keep_mode=keep_mode,
                )
                self.session_trace = kept
                removed += removed_session
        else:
            query = require_str(payload, "query")
            query_tokens = tokenize(query)
            if not query_tokens:
                raise InvalidAction("FILTER query must contain at least one alphanumeric token.")
            if scope in {"active", "all"}:
                kept, removed_active = filter_items_by_tokens(self.short_term_context, query_tokens)
                self.short_term_context = kept
                removed += removed_active
            if scope in {"session", "all"}:
                kept, removed_session = filter_items_by_tokens(self.session_trace, query_tokens)
                self.session_trace = kept
                removed += removed_session
        tool_op = {
            "op": "FILTER",
            "query": query,
            "keep_ids": keep_ids or [],
            "drop_ids": drop_ids or [],
            "scope": scope,
            "removed": removed,
            "step": self.step_count,
        }
        return self.render_observation(f"Filtered {removed} context items from {scope} scope."), 0.0, False, empty_memory_diff(), [], tool_op

    def action_search(self, payload: dict[str, Any]):
        if self.catalog_index_path is None:
            raise InvalidAction("SEARCH requires AGENTMEMORY_CATALOG_INDEX_PATH or catalog_index_path.")
        query = require_str(payload, "query")
        top_k = max(1, min(int(payload.get("top_k", 3)), 5))
        normalized_query = " ".join(query.lower().split())
        self.search_query_counts_this_session[normalized_query] += 1
        results = search_sqlite_catalog(self.catalog_index_path, query, top_k=top_k)
        if results:
            message = "Product search results:\n" + "\n".join(result.render() for result in results)
        else:
            self.no_result_search_count_this_session += 1
            message = "Product search returned no results."
        self.searched_this_session = True
        reward = 0.0
        if (
            self.memory_shaping == "chain_v1"
            and results
            and self.current_subtask_index not in self.rewarded_search_subtasks
            and (self.task_is_memoryarena() or self.current_subtask_requires_prior_memory())
        ):
            reward += 0.03
            self.rewarded_search_subtasks.add(self.current_subtask_index)
            self.last_reward_components.append(
                {
                    "name": "catalog_search_evidence_before_purchase",
                    "value": 0.03,
                    "subtask_index": self.current_subtask_index,
                    "query": query,
                }
            )
        if self.memory_shaping == "chain_v1":
            if self.search_query_counts_this_session[normalized_query] > 1:
                reward -= 0.04
                self.last_reward_components.append(
                    {
                        "name": "catalog_search_repeated_same_query_noop",
                        "value": -0.04,
                        "query": query,
                        "count": self.search_query_counts_this_session[normalized_query],
                    }
                )
            if not results:
                no_result_penalty = -0.03
                if self.no_result_search_count_this_session >= 2:
                    no_result_penalty = -0.07
                reward += no_result_penalty
                self.last_reward_components.append(
                    {
                        "name": "catalog_search_no_results_noop",
                        "value": no_result_penalty,
                        "query": query,
                        "no_result_count": self.no_result_search_count_this_session,
                    }
                )
            if (
                self.subtask_can_provide_memory_for_later(self.current_subtask_index)
                and self.source_session_has_stored_current_product()
            ):
                reward -= 0.08
                stored_product_ids = sorted(
                    product.product_id
                    for product in self.current_subtask().candidate_products
                    if self.memory_product_refs.get(product.product_id)
                )
                self.last_reward_components.append(
                    {
                        "name": "source_search_after_memory_ready_noop",
                        "value": -0.08,
                        "stored_product_ids": stored_product_ids,
                    }
                )
        tool_op = {"op": "SEARCH", "tool_family": "catalog", "query": query, "top_k": top_k, "step": self.step_count}
        return self.render_observation(message), reward, False, empty_memory_diff(), [], tool_op

    def action_buy(self, payload: dict[str, Any]):
        if set(payload) != {"product_id"}:
            raise InvalidAction("BUY accepts exactly one field: product_id.")
        product_id = require_str(payload, "product_id")
        subtask = self.current_subtask()
        subtask_index = self.current_subtask_index
        products = {product.product_id: product for product in subtask.candidate_products}
        if product_id not in products:
            raise InvalidAction(f"Unknown product_id '{product_id}' for current subtask.")

        product = products[product_id]
        visible_retrieved_memory_ids = sorted(self.retrieved_memory_ids_this_session)
        retrieved_memory_relevant = bool(visible_retrieved_memory_ids) and (
            self.retrieved_memory_references_relevant_prior(subtask_index)
        )
        compatible, reason = self.is_compatible_purchase(product)
        if not compatible and self.buy_semantics == "retry":
            repeat_rejected_buy = self.rejected_product_counts_this_session[product_id] > 0
            self.rejected_product_counts_this_session[product_id] += 1
            violation = {"product_id": product_id, "reason": reason, "step": self.step_count}
            reward = -0.5
            message = f"Purchase rejected: {self.render_purchase_rejection_reason(product, reason)}"
            if self.memory_shaping == "chain_v1" and repeat_rejected_buy:
                reward = -0.6
                self.last_reward_components.append(
                    {
                        "name": "buy_repeats_rejected_product_noop",
                        "value": -0.10,
                        "product_id": product_id,
                        "attempt_count": self.rejected_product_counts_this_session[product_id],
                    }
                )
            return self.render_observation(message), reward, False, empty_memory_diff(), [violation], None

        progress_reward = 1.0 if compatible else -0.5
        shaping_bonus = 0.0
        if compatible:
            shaping_bonus = self.buy_memory_shaping_bonus(
                product,
                subtask_index=subtask_index,
            )
        else:
            self.last_reward_components.append(
                {
                    "name": "buy_committed_incorrect",
                    "value": progress_reward,
                    "product_id": product_id,
                }
            )
        self.record_purchase(product, purchase_correct=compatible)
        session_advanced = compatible or self.buy_semantics == "continue"
        if session_advanced:
            self.current_subtask_index += 1
        done = (
            (not compatible and self.buy_semantics == "terminate")
            or self.current_subtask_index >= len(self.require_task().subtasks)
        )
        final_bonus = 1.0 if done and self.purchase_history_is_successful() else 0.0
        self.short_term_context = []
        self.session_trace = []
        self.retrieved_memory_ids_this_session = set()
        self.repeated_active_retrieve_count_this_session = 0
        self.retrieve_query_counts_this_session = Counter()
        self.empty_retrieve_count_this_session = 0
        self.searched_this_session = False
        self.search_query_counts_this_session = Counter()
        self.no_result_search_count_this_session = 0
        self.rejected_product_counts_this_session = Counter()
        if done and compatible:
            message = "All bundled shopping subtasks are complete and compatible."
        elif done:
            message = "Shopping episode is complete."
        elif compatible:
            message = "Purchase accepted. A new shopping session starts."
        else:
            message = "Purchase recorded. A new shopping session starts."
        tool_op = {
            "op": "BUY",
            "product_id": product_id,
            "step": self.step_count,
            "committed": True,
            "purchase_correct": compatible,
            "outcome": "correct" if compatible else "incorrect",
            "session_advanced": session_advanced,
            "terminal": done,
            "memory_shaping_bonus": shaping_bonus,
        }
        if visible_retrieved_memory_ids:
            tool_op["visible_retrieved_memory_ids"] = visible_retrieved_memory_ids
            tool_op["visible_retrieved_source_product_ids"] = sorted(
                self.memory_ids_source_product_ids(visible_retrieved_memory_ids)
            )
            tool_op["retrieved_memory_relevant_prior"] = retrieved_memory_relevant
        violations = []
        if not compatible:
            violations.append({"product_id": product_id, "reason": reason, "step": self.step_count})
        return (
            self.render_observation(message),
            progress_reward + final_bonus + shaping_bonus,
            done,
            empty_memory_diff(),
            violations,
            tool_op,
        )

    def action_answer(self, payload: dict[str, Any]):
        text = require_str(payload, "text")
        required_ids = {subtask.target_product_id for subtask in self.require_task().subtasks}
        purchased_ids = {item["product_id"] for item in self.purchase_history}
        if required_ids <= purchased_ids and all(product_id in text for product_id in required_ids):
            self.done = True
            return self.render_observation("Answer accepted."), 1.0, True, empty_memory_diff(), [], None
        message = "Answer recorded, but the bundle is not complete yet."
        reward = 0.0
        if self.memory_shaping == "chain_v1":
            # Do not punish memory tools for being used. The current recurrent
            # failure mode is the opposite: after ADD/RETRIEVE the policy often
            # stalls by emitting safe-looking ANSWER text instead of closing the
            # shopping session with BUY. Penalize this non-progress outcome
            # directly, while leaving valid memory operations untaxed.
            reward = -0.12
            reward_event: dict[str, Any] = {
                "name": "premature_answer_before_bundle_complete",
                "value": reward,
                "subtask_index": self.current_subtask_index,
            }
            if self.current_subtask_requires_prior_memory() and self.retrieved_memory_ids_this_session:
                reward = -0.20
                reward_event = {
                    "name": "dependent_memory_ready_answer_instead_of_buy",
                    "value": reward,
                    "subtask_index": self.current_subtask_index,
                    "memory_ids": sorted(self.retrieved_memory_ids_this_session),
                }
            elif self.subtask_can_provide_memory_for_later(self.current_subtask_index) and self.source_session_has_stored_current_product():
                reward = -0.20
                reward_event = {
                    "name": "source_memory_ready_answer_instead_of_buy",
                    "value": reward,
                    "subtask_index": self.current_subtask_index,
                    "stored_product_ids": sorted(
                        product.product_id
                        for product in self.current_subtask().candidate_products
                        if self.memory_product_refs.get(product.product_id)
                    ),
                }
            self.last_reward_components.append(reward_event)
        return self.render_observation(message), reward, False, empty_memory_diff(), [], None

    def parse_action(self, action: str) -> tuple[str, dict[str, Any]]:
        if not isinstance(action, str):
            raise InvalidAction(f"Action must be a string, got {type(action).__name__}.")
        match = self.action_pattern.match(action.strip())
        if not match:
            raise InvalidAction("Expected '<TOOL> {json payload}'.")
        op = match.group(1).upper()
        payload_text = match.group(2).strip()
        if not payload_text:
            return op, {}
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise InvalidAction(f"Payload must be valid JSON: {exc.msg}.") from exc
        if not isinstance(payload, dict):
            raise InvalidAction("Payload must be a JSON object.")
        return op, payload

    def find_memory_entry(self, payload: dict[str, Any]) -> MemoryEntry:
        memory_id = payload.get("memory_id")
        key = payload.get("key")
        if memory_id is not None:
            if memory_id not in self.long_term_memory:
                raise InvalidAction(f"Unknown memory_id '{memory_id}'.")
            return self.long_term_memory[str(memory_id)]
        if key is not None:
            matches = [entry for entry in self.long_term_memory.values() if entry.key == key]
            if not matches:
                raise InvalidAction(f"Unknown memory key '{key}'.")
            if len(matches) > 1:
                raise InvalidAction(f"Memory key '{key}' is ambiguous; use memory_id.")
            return matches[0]
        raise InvalidAction("Expected memory_id or key.")

    def is_compatible_purchase(self, product: Product) -> tuple[bool, str]:
        subtask = self.current_subtask()
        if product.product_id == subtask.target_product_id and not has_compatibility_constraints(product):
            return True, "target product without external compatibility constraints"
        attrs = product.attributes
        if "compatible_tv_min" in attrs:
            tv_size = self.bundle_state.get("tv_size_in")
            if tv_size is None:
                return False, "no previously purchased TV size exists in hidden bundle state"
            min_size = attrs["compatible_tv_min"]
            max_size = attrs["compatible_tv_max"]
            if not min_size <= tv_size <= max_size:
                return False, f"TV size {tv_size} is outside supported range {min_size}-{max_size}"
        if "compatible_monitor_min" in attrs:
            monitor_size = self.bundle_state.get("monitor_size_in")
            if monitor_size is None:
                return False, "no previously purchased monitor size exists in hidden bundle state"
            min_size = attrs["compatible_monitor_min"]
            max_size = attrs["compatible_monitor_max"]
            if not min_size <= monitor_size <= max_size:
                return False, f"monitor size {monitor_size} is outside supported range {min_size}-{max_size}"
        if "max_weight_kg" in attrs:
            mounted_weight = self.bundle_state.get("tv_weight_kg", self.bundle_state.get("monitor_weight_kg"))
            if mounted_weight is None:
                return False, "no previously purchased mounted-device weight exists in hidden bundle state"
            max_weight = attrs["max_weight_kg"]
            if mounted_weight > max_weight:
                return False, f"mounted-device weight {mounted_weight}kg exceeds supported weight {max_weight}kg"
        if "supported_vesa" in attrs:
            tv_vesa = self.bundle_state.get("vesa")
            if tv_vesa is None:
                return False, "no previously purchased TV VESA mount pattern exists in hidden bundle state"
            if tv_vesa not in attrs["supported_vesa"]:
                return False, f"TV VESA {tv_vesa} is not supported by {render_attr_value(attrs['supported_vesa'])}"
        if "compatible_laptop_min" in attrs:
            laptop_size = self.bundle_state.get("laptop_size_in")
            if laptop_size is None:
                return False, "no previously purchased laptop size exists in hidden bundle state"
            min_size = attrs["compatible_laptop_min"]
            max_size = attrs["compatible_laptop_max"]
            if not min_size <= laptop_size <= max_size:
                return False, f"laptop size {laptop_size} is outside supported range {min_size}-{max_size}"
        if "required_port" in attrs:
            available_ports = self.bundle_state.get("laptop_ports", self.bundle_state.get("monitor_ports"))
            if available_ports is None:
                return False, "no previously purchased device port type exists in hidden bundle state"
            if attrs["required_port"] not in str(available_ports).split(","):
                return False, f"required port {attrs['required_port']} does not match device ports {available_ports}"
        return product.product_id == subtask.target_product_id or has_compatibility_constraints(product), "compatible"

    def record_purchase(
        self,
        product: Product,
        *,
        subtask_index: int | None = None,
        purchase_correct: bool | None = None,
    ) -> None:
        resolved_subtask_index = self.current_subtask_index if subtask_index is None else subtask_index
        if purchase_correct is None:
            purchase_correct = (
                product.product_id == self.require_task().subtasks[resolved_subtask_index].target_product_id
            )
        purchase = {
            "step": self.step_count,
            "subtask_index": resolved_subtask_index,
            "product_id": product.product_id,
            "title": product.title,
            "attributes": dict(product.attributes),
            "purchase_correct": purchase_correct,
        }
        self.purchase_history.append(purchase)
        for key, value in product.attributes.items():
            if key in {
                "tv_size_in",
                "tv_weight_kg",
                "vesa",
                "laptop_size_in",
                "laptop_ports",
                "monitor_size_in",
                "monitor_weight_kg",
                "monitor_ports",
            }:
                self.bundle_state[key] = value

    def purchase_history_is_successful(self) -> bool:
        return bool(self.purchase_history) and all(
            item.get("purchase_correct", False) for item in self.purchase_history
        )

    def current_subtask_requires_prior_memory(self) -> bool:
        return self.subtask_requires_prior_memory(self.current_subtask_index)

    def task_uses_cross_session_memory(self) -> bool:
        task = self.require_task()
        marker = f"{task.source} {task.memory_dependency}".lower()
        return "cross_session" in marker or "previous" in marker or "memoryarena" in marker

    def task_is_memoryarena(self) -> bool:
        task = self.require_task()
        return "memoryarena" in f"{task.source} {task.task_id} {task.title}".lower()

    def subtask_requires_prior_memory(self, subtask_index: int) -> bool:
        task = self.require_task()
        if not 0 <= subtask_index < len(task.subtasks):
            return False
        subtask = task.subtasks[subtask_index]
        instruction = subtask.instruction.lower()
        if any(has_dependency_constraints(product) for product in subtask.candidate_products):
            return True
        if subtask_index > 0 and self.task_uses_cross_session_memory():
            return True
        return "previous" in instruction or "prior" in instruction or "must be compatible" in instruction

    def subtask_can_provide_memory_for_later(self, subtask_index: int) -> bool:
        task = self.require_task()
        return 0 <= subtask_index < len(task.subtasks) - 1 and self.task_uses_cross_session_memory()

    def source_session_has_stored_current_product(self) -> bool:
        return any(
            self.memory_ids_for_product(product.product_id) for product in self.current_subtask().candidate_products
        )

    def render_purchase_rejection_reason(self, product: Product, reason: str) -> str:
        if reason == "compatible" and not has_compatibility_constraints(product):
            return "this product was not accepted by the current instruction verifier."
        return reason

    def referenced_current_product_ids(self, text: str) -> set[str]:
        return self.referenced_product_ids(
            text,
            [product for product in self.current_subtask().candidate_products],
        )

    def referenced_accessible_product_ids(
        self,
        text: str,
        *,
        extra_product_ids: set[str] | None = None,
    ) -> set[str]:
        product_by_id = self.task_products_by_id()
        accessible_ids = {
            product.product_id for product in self.current_subtask().candidate_products
        }
        accessible_ids.update(item["product_id"] for item in self.purchase_history)
        accessible_ids.update(extra_product_ids or set())
        return self.referenced_product_ids(
            text,
            [product_by_id[product_id] for product_id in sorted(accessible_ids) if product_id in product_by_id],
        )

    def referenced_product_ids(self, text: str, products: list[Product]) -> set[str]:
        normalized = text.lower()
        referenced: set[str] = set()
        for product in products:
            signals = 0
            if product.product_id.lower() in normalized:
                signals += 2
            if product.title.lower() in normalized:
                signals += 2
            for key, value in product.attributes.items():
                rendered_key = self.normalize_compatibility_phrase(key)
                rendered_value = self.normalize_compatibility_phrase(str(render_attr_value(value)))
                key_seen = self.phrase_in_text(rendered_key, text)
                value_seen = self.phrase_in_text(rendered_value, text)
                key_value_seen = bool(
                    rendered_key
                    and rendered_value
                    and re.search(
                        rf"(?<![a-z0-9]){re.escape(rendered_key)}\s*{re.escape(rendered_value)}(?![a-z0-9])",
                        self.normalize_compatibility_phrase(text),
                    )
                )
                if key_seen and value_seen:
                    signals += 2
                elif key_value_seen:
                    signals += 2
                elif len(rendered_value) > 1 and value_seen:
                    signals += 1
            if signals >= 2:
                referenced.add(product.product_id)
        return referenced

    def task_products_by_id(self) -> dict[str, Product]:
        task = self.require_task()
        return {
            product.product_id: product
            for subtask in task.subtasks
            for product in subtask.candidate_products
        }

    def product_subtask_index(self, product_id: str) -> int:
        task = self.require_task()
        for subtask_index, subtask in enumerate(task.subtasks):
            if any(product.product_id == product_id for product in subtask.candidate_products):
                return subtask_index
        raise ValueError(f"Unknown product_id in task '{task.task_id}': {product_id}.")

    def apply_initial_task_state(self) -> None:
        task = self.require_task()
        if not 0 <= task.start_subtask_index < len(task.subtasks):
            raise ValueError(
                f"Task '{task.task_id}' has invalid start_subtask_index={task.start_subtask_index}; "
                f"expected 0..{len(task.subtasks) - 1}."
            )
        product_by_id = self.task_products_by_id()
        for product_id in task.initial_purchase_product_ids:
            if product_id not in product_by_id:
                raise ValueError(f"Initial purchase product_id '{product_id}' is not in task '{task.task_id}'.")
            self.record_purchase(
                product_by_id[product_id],
                subtask_index=self.product_subtask_index(product_id),
            )
        for memory in task.initial_memories:
            unknown_product_ids = [product_id for product_id in memory.product_ids if product_id not in product_by_id]
            if unknown_product_ids:
                raise ValueError(
                    f"Initial memory for task '{task.task_id}' references unknown product_ids: "
                    f"{unknown_product_ids}."
                )
            memory_id = f"mem_{self.memory_id_counter:04d}"
            self.memory_id_counter += 1
            entry = MemoryEntry(
                memory_id=memory_id,
                key=memory.key,
                value=memory.value,
                created_step=0,
                updated_step=0,
            )
            self.long_term_memory[memory_id] = entry
            self.link_memory_product_refs(memory_id, set(memory.product_ids))
        self.current_subtask_index = task.start_subtask_index

    def product_ids_for_memory(self, memory_id: str) -> set[str]:
        return {
            product_id
            for product_id, memory_ids in self.memory_product_refs.items()
            if memory_id in memory_ids and memory_id in self.long_term_memory
        }

    def memory_ids_for_product(self, product_id: str) -> set[str]:
        return {
            memory_id
            for memory_id in self.memory_product_refs.get(product_id, set())
            if memory_id in self.long_term_memory
        }

    def visible_memory_ids(self) -> set[str]:
        visible_text = "\n".join([*self.short_term_context, *self.session_trace])
        return {
            memory_id
            for memory_id in re.findall(r"\bmem_\d{4}\b", visible_text)
            if memory_id in self.long_term_memory
        }

    def link_memory_product_refs(self, memory_id: str, product_ids: set[str]) -> None:
        for product_id in product_ids:
            self.memory_product_refs.setdefault(product_id, set()).add(memory_id)

    def unlink_memory_product_refs(self, memory_id: str) -> None:
        for product_id in list(self.memory_product_refs):
            memory_ids = self.memory_product_refs[product_id]
            memory_ids.discard(memory_id)
            if not memory_ids:
                del self.memory_product_refs[product_id]

    def refresh_active_memory_context(self, entry: MemoryEntry) -> None:
        prefix = f"[{entry.memory_id}] "
        self.short_term_context = [
            entry.render() if item.startswith(prefix) else item for item in self.short_term_context
        ]

    def relevant_prior_product_ids_for_subtask(self, subtask_index: int) -> set[str]:
        if subtask_index <= 0 or not self.purchase_history:
            return set()
        task = self.require_task()
        if self.task_has_only_source_option_attributes():
            # MemoryArena bundled-shopping subtasks express dependencies as a
            # sequence of product categories: cake -> frosting -> coloring ->
            # sprinkles ... . In the frozen data the only stable candidate
            # attribute is source_option, so the most reliable deterministic
            # precision check is whether evidence refers to the latest accepted
            # prior product, not an older stale source.
            return {self.purchase_history[-1]["product_id"]}

        if not 0 <= subtask_index < len(task.subtasks):
            return set()
        subtask = task.subtasks[subtask_index]
        needs: set[str] = set()
        for product in subtask.candidate_products:
            attrs = product.attributes
            if any(key in attrs for key in ("compatible_tv_min", "compatible_tv_max", "supported_vesa")):
                needs.update({"tv_size_in", "tv_weight_kg", "vesa"})
            if any(key in attrs for key in ("compatible_monitor_min", "compatible_monitor_max")):
                needs.update({"monitor_size_in", "monitor_weight_kg", "vesa"})
            if "compatible_laptop_min" in attrs or "compatible_laptop_max" in attrs:
                needs.add("laptop_size_in")
            if "required_port" in attrs:
                needs.update({"laptop_ports", "monitor_ports"})
            if "max_weight_kg" in attrs and not needs:
                needs.update({"tv_weight_kg", "monitor_weight_kg"})
        if not needs:
            return {item["product_id"] for item in self.purchase_history}
        relevant = {
            item["product_id"]
            for item in self.purchase_history
            if set(item.get("attributes", {})) & needs
        }
        return relevant or {item["product_id"] for item in self.purchase_history}

    def task_has_only_source_option_attributes(self) -> bool:
        task = self.require_task()
        return all(
            set(product.attributes) <= {"source_option"}
            for subtask in task.subtasks
            for product in subtask.candidate_products
        )

    def memory_ids_source_product_ids(self, memory_ids: set[str] | list[str]) -> set[str]:
        return {
            product_id
            for memory_id in memory_ids
            for product_id in self.product_ids_for_memory(memory_id)
        }

    def retrieved_memory_references_relevant_prior(self, subtask_index: int) -> bool:
        relevant_prior_product_ids = self.relevant_prior_product_ids_for_subtask(subtask_index)
        if not relevant_prior_product_ids:
            return bool(self.retrieved_memory_ids_this_session)
        return bool(
            self.memory_ids_source_product_ids(self.retrieved_memory_ids_this_session)
            & relevant_prior_product_ids
        )

    @staticmethod
    def normalize_compatibility_phrase(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        return re.sub(r"\s+", " ", normalized)

    @classmethod
    def phrase_in_text(cls, phrase: str, text: str) -> bool:
        normalized_phrase = cls.normalize_compatibility_phrase(phrase)
        if not normalized_phrase:
            return False
        normalized_text = cls.normalize_compatibility_phrase(text)
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_text))

    def buy_memory_shaping_bonus(
        self,
        product: Product,
        *,
        subtask_index: int | None = None,
    ) -> float:
        if self.memory_shaping != "chain_v1":
            return 0.0

        if subtask_index is None:
            subtask_index = self.current_subtask_index
        bonus = 0.0
        if has_memory_source_attributes(product) or self.subtask_can_provide_memory_for_later(subtask_index):
            memory_ids = self.memory_ids_for_product(product.product_id)
            if memory_ids:
                bonus += 0.10
                self.last_reward_components.append(
                    {
                        "name": "memory_written_before_source_purchase",
                        "value": 0.10,
                        "product_id": product.product_id,
                        "memory_ids": sorted(memory_ids),
                    }
                )

        if (
            (has_dependency_constraints(product) or self.subtask_requires_prior_memory(subtask_index))
            and self.retrieved_memory_ids_this_session
            and self.retrieved_memory_references_relevant_prior(subtask_index)
        ):
            bonus += 0.10
            self.last_reward_components.append(
                {
                    "name": "relevant_memory_visible_before_dependent_purchase",
                    "value": 0.10,
                    "product_id": product.product_id,
                    "memory_ids": sorted(self.retrieved_memory_ids_this_session),
                    "relevant_prior_product_ids": sorted(
                        self.relevant_prior_product_ids_for_subtask(subtask_index)
                    ),
                }
            )
        return bonus

    def render_observation(self, prefix: str | None = None) -> str:
        task = self.require_task()
        lines = [
            "Task family: bundled_shopping",
            f"Task: {task.title}",
            f"Progress: {self.current_subtask_index}/{len(task.subtasks)}",
        ]
        if prefix:
            lines.append(f"Last tool/result: {prefix}")
        if self.current_subtask_index < len(task.subtasks):
            subtask = self.current_subtask()
            lines.extend(
                [
                    "",
                    f"Current shopping session {self.current_subtask_index + 1}/{len(task.subtasks)}:",
                    subtask.instruction,
                    "",
                    "Visible candidate products:",
                ]
            )
            lines.extend(product.render() for product in subtask.candidate_products)
        if self.session_trace:
            lines.extend(["", "Current session short-term history:"])
            lines.extend(f"- S{index}: {indent_trace_item(item)}" for index, item in enumerate(self.session_trace))
        else:
            lines.extend(["", "Current session short-term history: <empty>"])
        if self.short_term_context:
            lines.extend(["", "Active retrieved/summary context:"])
            lines.extend(f"- C{index}: {item}" for index, item in enumerate(self.short_term_context))
        else:
            lines.extend(["", "Active retrieved/summary context: <empty>"])
        available_actions = [
            'ADD {"key": "...", "value": "..."}',
            'UPDATE {"memory_id": "mem_0000", "value": "..."}',
            'DELETE {"memory_id": "mem_0000"}',
            'RETRIEVE {"query": "...", "top_k": 3}',
            'SUMMARY {"text": "...", "source_ids": ["S0", "C0"]}',
            'FILTER {"keep_ids": ["C0"], "scope": "active"}',
            'BUY {"product_id": "..."}',
            'ANSWER {"text": "..."}',
        ]
        if self.catalog_index_path is not None:
            available_actions.insert(-2, 'SEARCH {"query": "...", "top_k": 3}')
        lines.extend(["", "Available actions now:", *available_actions])
        lines.extend(
            [
                "ADD stores the policy-authored key and value verbatim in hidden long-term memory.",
                "RETRIEVE matches its query against text previously stored with ADD and exposes matches as C0, C1, and so on.",
                "SUMMARY and FILTER operate only on visible S*/C* context items.",
                "BUY accepts exactly one product_id from the visible candidates.",
                "Current-session S* history clears when a committed BUY advances to a new shopping session; long-term memory remains hidden until RETRIEVE.",
            ]
        )
        if self.catalog_index_path is not None:
            lines.append(
                "SEARCH matches its query against catalog product titles and returns title, price_usd, "
                "average_rating, total_reviews, and match_score."
            )
        if self.buy_semantics == "terminate":
            lines.append("Under terminate semantics, a BUY that fails verification ends the episode without exposing the verifier reason.")
        return "\n".join(lines)

    def build_info(
        self,
        memory_state_diff: dict[str, list[dict[str, Any]]],
        compatibility_violations: list[dict[str, Any]] | None = None,
        tool_ops: list[dict[str, Any]] | None = None,
    ) -> StepInfo:
        task = self.require_task()
        normalized_tool_ops = tool_ops or []
        return StepInfo(
            task_id=task.task_id,
            split=task.split,
            source=task.source,
            difficulty=task.difficulty,
            memory_dependency=task.memory_dependency,
            progress_score=self.current_subtask_index / len(task.subtasks),
            episode_success=(
                self.done
                and self.current_subtask_index >= len(task.subtasks)
                and self.purchase_history_is_successful()
            ),
            tool_ops=normalized_tool_ops,
            memory_ops=[item for item in normalized_tool_ops if item.get("op") in MEMORY_TOOL_OPS],
            memory_state_diff=memory_state_diff,
            compatibility_violations=compatibility_violations or [],
            purchase_history=list(self.purchase_history),
            current_subtask_index=self.current_subtask_index,
            session_trace=list(self.session_trace),
            reward_components=list(self.last_reward_components),
        )

    def current_subtask(self) -> ShoppingSubtask:
        task = self.require_task()
        if self.current_subtask_index >= len(task.subtasks):
            return task.subtasks[-1]
        return task.subtasks[self.current_subtask_index]

    def require_task(self) -> ShoppingTask:
        if self.task is None:
            raise RuntimeError("Environment must be reset before use.")
        return self.task

    def close(self) -> None:
        self.long_term_memory.clear()
        self.short_term_context.clear()
        self.session_trace.clear()
        self.purchase_history.clear()
        self.bundle_state.clear()


def extract_last_tool_result(observation: str) -> str:
    lines = observation.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("Last tool/result: "):
            continue
        result_lines = [line.removeprefix("Last tool/result: ")]
        for follow in lines[index + 1:]:
            if not follow.strip():
                break
            result_lines.append(follow)
        return "\n".join(result_lines).strip()
    return ""


def render_session_trace_entry(step_count: int, action_text: str, result_text: str) -> str:
    lines = [f"Step {step_count}", f"Action: {action_text}"]
    if result_text:
        lines.append(f"Result: {result_text}")
    return "\n".join(lines)


def indent_trace_item(item: str) -> str:
    return item.replace("\n", "\n  ")


def build_default_tasks() -> list[ShoppingTask]:
    return load_task_dataset()


def load_task_dataset(
    data_path: str | Path | None = None,
    split: str | None = None,
    split_dir: str | Path | None = None,
) -> list[ShoppingTask]:
    resolved_data_path = Path(data_path or os.environ.get("AGENTMEMORY_DATA_PATH", default_smoke_data_path()))
    resolved_split = split if split is not None else os.environ.get("AGENTMEMORY_SPLIT")
    resolved_split_dir = split_dir or os.environ.get("AGENTMEMORY_SPLIT_DIR")

    if resolved_data_path.exists():
        tasks = load_tasks_from_jsonl(resolved_data_path)
    else:
        tasks = [build_tv_bundle_task(), build_laptop_bundle_task()]
    if resolved_split:
        return select_tasks_by_split(tasks, resolved_split, split_dir=resolved_split_dir)
    return tasks


def default_smoke_data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "bundled_shopping_smoke.jsonl"


def default_split_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "splits"


def load_split_task_ids(split: str, split_dir: str | Path | None = None) -> list[str]:
    root = Path(split_dir) if split_dir is not None else default_split_dir()
    path = root / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing AgentMemoryGym split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def select_tasks_by_split(
    tasks: list[ShoppingTask],
    split: str,
    split_dir: str | Path | None = None,
) -> list[ShoppingTask]:
    task_by_id = {task.task_id: task for task in tasks}
    split_task_ids = load_split_task_ids(split, split_dir=split_dir)
    missing = [task_id for task_id in split_task_ids if task_id not in task_by_id]
    if missing:
        raise ValueError(f"Split '{split}' references unknown AgentMemoryGym task ids: {missing}.")
    selected = [task_by_id[task_id] for task_id in split_task_ids]
    mismatched = [task.task_id for task in selected if task.split != split]
    if mismatched:
        raise ValueError(f"Split '{split}' contains tasks whose record split differs: {mismatched}.")
    return selected


def load_tasks_from_jsonl(path: str | Path) -> list[ShoppingTask]:
    tasks: list[ShoppingTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            tasks.append(task_from_record(record, source_path=str(path), line_number=line_number))
    if not tasks:
        raise ValueError(f"No AgentMemoryGym tasks found in {path}.")
    return tasks


def task_from_record(record: dict[str, Any], *, source_path: str, line_number: int) -> ShoppingTask:
    for key in ["task_id", "title", "subtasks"]:
        if key not in record:
            raise ValueError(f"{source_path}:{line_number} missing required field '{key}'.")
    subtasks = tuple(subtask_from_record(item, source_path=source_path, line_number=line_number) for item in record["subtasks"])
    if not subtasks:
        raise ValueError(f"{source_path}:{line_number} must contain at least one subtask.")
    return ShoppingTask(
        task_id=str(record["task_id"]),
        title=str(record["title"]),
        subtasks=subtasks,
        split=str(record.get("split", "train")),
        source=str(record.get("source", "jsonl")),
        difficulty=str(record.get("difficulty", "unknown")),
        memory_dependency=str(record.get("memory_dependency", "cross_session_product_attribute")),
        start_subtask_index=parse_optional_int(record, "start_subtask_index", default=0),
        initial_purchase_product_ids=parse_optional_str_tuple(record, "initial_purchase_product_ids"),
        initial_memories=tuple(
            initial_memory_from_record(item, source_path=source_path, line_number=line_number)
            for item in parse_optional_list(record, "initial_memories")
        ),
        curriculum_flags=frozenset(parse_optional_str_tuple(record, "curriculum_flags")),
    )


def initial_memory_from_record(
    record: dict[str, Any],
    *,
    source_path: str,
    line_number: int,
) -> InitialMemorySpec:
    if not isinstance(record, dict):
        raise ValueError(f"{source_path}:{line_number} initial_memories entries must be objects.")
    for key in ["key", "value"]:
        if key not in record or not isinstance(record[key], str):
            raise ValueError(f"{source_path}:{line_number} initial memory missing string field '{key}'.")
    return InitialMemorySpec(
        key=record["key"],
        value=record["value"],
        product_ids=parse_optional_str_tuple(record, "product_ids"),
    )


def parse_optional_list(record: dict[str, Any], key: str) -> list[Any]:
    value = record.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"Optional field '{key}' must be a list.")
    return value


def parse_optional_str_tuple(record: dict[str, Any], key: str) -> tuple[str, ...]:
    value = parse_optional_list(record, key)
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Optional field '{key}' must be a list of strings.")
    return tuple(value)


def parse_optional_int(record: dict[str, Any], key: str, *, default: int) -> int:
    value = record.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Optional field '{key}' must be an integer.")
    return value


def subtask_from_record(record: dict[str, Any], *, source_path: str, line_number: int) -> ShoppingSubtask:
    for key in ["instruction", "candidate_products", "target_product_id"]:
        if key not in record:
            raise ValueError(f"{source_path}:{line_number} subtask missing required field '{key}'.")
    products = tuple(product_from_record(item, source_path=source_path, line_number=line_number) for item in record["candidate_products"])
    if not products:
        raise ValueError(f"{source_path}:{line_number} subtask must contain candidate_products.")
    product_ids = {product.product_id for product in products}
    target_product_id = str(record["target_product_id"])
    if target_product_id not in product_ids:
        raise ValueError(
            f"{source_path}:{line_number} target_product_id '{target_product_id}' "
            f"is not in candidate product ids {sorted(product_ids)}."
        )
    return ShoppingSubtask(
        instruction=str(record["instruction"]),
        candidate_products=products,
        target_product_id=target_product_id,
    )


def product_from_record(record: dict[str, Any], *, source_path: str, line_number: int) -> Product:
    for key in ["product_id", "title", "attributes"]:
        if key not in record:
            raise ValueError(f"{source_path}:{line_number} product missing required field '{key}'.")
    attributes = record["attributes"]
    if not isinstance(attributes, dict):
        raise ValueError(f"{source_path}:{line_number} product attributes must be an object.")
    return Product(
        product_id=str(record["product_id"]),
        title=str(record["title"]),
        attributes=attributes,
    )


def build_tv_bundle_task() -> ShoppingTask:
    tv_a = Product("tv_a", "Orion 4K TV", {"tv_size_in": 65, "tv_weight_kg": 24, "vesa": "300x300"})
    tv_b = Product("tv_b", "Nebula 4K TV", {"tv_size_in": 75, "tv_weight_kg": 32, "vesa": "400x400"})
    mount_a = Product(
        "mount_a",
        "Tilt wall mount A",
        {"compatible_tv_min": 40, "compatible_tv_max": 60, "max_weight_kg": 28, "supported_vesa": ("200x200", "300x300")},
    )
    mount_b = Product(
        "mount_b",
        "Articulating wall mount B",
        {"compatible_tv_min": 65, "compatible_tv_max": 85, "max_weight_kg": 45, "supported_vesa": ("300x300", "400x400")},
    )
    console_a = Product("console_a", "Media console A", {"compatible_tv_min": 45, "compatible_tv_max": 65, "width_in": 58})
    console_b = Product("console_b", "Media console B", {"compatible_tv_min": 70, "compatible_tv_max": 85, "width_in": 78})
    return ShoppingTask(
        task_id="tv_bundle_75",
        title="Bundle a living-room TV setup across multiple shopping sessions",
        subtasks=(
            ShoppingSubtask(
                instruction="Buy the user's chosen larger-screen TV for a large living room. Remember product attributes that later accessories may need.",
                candidate_products=(tv_a, tv_b),
                target_product_id="tv_b",
            ),
            ShoppingSubtask(
                instruction="Buy a wall mount compatible with the previously purchased TV. The current request does not restate the TV size.",
                candidate_products=(mount_a, mount_b),
                target_product_id="mount_b",
            ),
            ShoppingSubtask(
                instruction="Buy a media console compatible with the same previously purchased TV.",
                candidate_products=(console_a, console_b),
                target_product_id="console_b",
            ),
        ),
        split="train",
        source="memoryarena_webshop_style_handcrafted_v0",
        difficulty="smoke_dependency_distance_2",
        memory_dependency="tv_size_weight_vesa_reused_across_sessions",
    )


def build_laptop_bundle_task() -> ShoppingTask:
    laptop_a = Product("laptop_a", "Atlas developer laptop", {"laptop_size_in": 14, "laptop_ports": "usb-c"})
    laptop_b = Product("laptop_b", "Forge mobile workstation", {"laptop_size_in": 16, "laptop_ports": "usb-c"})
    sleeve_a = Product("sleeve_a", "Slim sleeve A", {"compatible_laptop_min": 12, "compatible_laptop_max": 13})
    sleeve_b = Product("sleeve_b", "Protective sleeve B", {"compatible_laptop_min": 14, "compatible_laptop_max": 14})
    dock_a = Product("dock_a", "Dual-display dock A", {"required_port": "usb-c"})
    dock_b = Product("dock_b", "Legacy dock B", {"required_port": "barrel"})
    return ShoppingTask(
        task_id="laptop_bundle_14",
        title="Bundle laptop accessories across multiple shopping sessions",
        subtasks=(
            ShoppingSubtask(
                instruction="Buy the 14-inch laptop the user selected. Remember attributes needed for accessories.",
                candidate_products=(laptop_a, laptop_b),
                target_product_id="laptop_a",
            ),
            ShoppingSubtask(
                instruction="Buy a sleeve compatible with the previously purchased laptop size.",
                candidate_products=(sleeve_a, sleeve_b),
                target_product_id="sleeve_b",
            ),
            ShoppingSubtask(
                instruction="Buy the dock that matches the previously purchased laptop port type.",
                candidate_products=(dock_a, dock_b),
                target_product_id="dock_a",
            ),
        ),
        split="dev",
        source="memoryarena_webshop_style_handcrafted_v0",
        difficulty="smoke_dependency_distance_2",
        memory_dependency="laptop_size_port_reused_across_sessions",
    )


def require_str(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise InvalidAction(f"Missing required field '{key}'.")
    value = payload[key]
    if not isinstance(value, str):
        raise InvalidAction(f"Field '{key}' must be a string.")
    return value


def optional_str_list(payload: dict[str, Any], key: str) -> list[str] | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidAction(f"Field '{key}' must be a list of strings.")
    return value


def validate_context_ids(
    context_ids: list[str],
    *,
    active_count: int,
    session_count: int,
    scope: str,
) -> None:
    allowed_prefixes = {
        "active": {"C"},
        "session": {"S"},
        "all": {"C", "S"},
    }[scope]
    for context_id in context_ids:
        match = re.fullmatch(r"([CS])(\d+)", context_id)
        if match is None:
            raise InvalidAction(f"Unknown context id '{context_id}'. Expected S0/S1 or C0/C1.")
        prefix, raw_index = match.groups()
        if prefix not in allowed_prefixes:
            raise InvalidAction(f"Context id '{context_id}' is outside context scope '{scope}'.")
        index = int(raw_index)
        count = active_count if prefix == "C" else session_count
        if index >= count:
            raise InvalidAction(f"Context id '{context_id}' is not visible in the current observation.")


def context_items_by_ids(
    context_ids: list[str],
    active_items: list[str],
    session_items: list[str],
) -> list[str]:
    items: list[str] = []
    for context_id in context_ids:
        prefix = context_id[0]
        index = int(context_id[1:])
        source = active_items if prefix == "C" else session_items
        items.append(f"{context_id}: {source[index]}")
    return items


def filter_items_by_ids(
    items: list[str],
    *,
    prefix: str,
    selected_ids: set[str],
    keep_mode: bool,
) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    for index, item in enumerate(items):
        context_id = f"{prefix}{index}"
        selected = context_id in selected_ids
        should_keep = selected if keep_mode else not selected
        if should_keep:
            kept.append(item)
        else:
            removed += 1
    return kept, removed


def empty_memory_diff() -> dict[str, list[dict[str, Any]]]:
    return {"added": [], "updated": [], "deleted": []}


def memory_entry_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "memory_id": entry.memory_id,
        "key": entry.key,
        "value": entry.value,
        "created_step": entry.created_step,
        "updated_step": entry.updated_step,
        "access_count": entry.access_count,
    }


def render_attr_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def clamp_int(value: Any, *, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidAction(f"Expected integer value, got {value!r}.") from exc
    return max(min_value, min(parsed, max_value))


def compact_context_text(text: str, *, max_chars: int) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."


def filter_items_by_tokens(items: list[str], query_tokens: set[str]) -> tuple[list[str], int]:
    kept = [item for item in items if tokenize(item) & query_tokens]
    return kept, len(items) - len(kept)


def parse_memory_shaping_mode(value: str | None) -> str:
    mode = (value or "off").strip().lower()
    if mode in {"", "0", "false", "no", "none", "off"}:
        return "off"
    if mode in {"chain", "chain_v1", "memory_chain_v1"}:
        return "chain_v1"
    raise ValueError(
        "AGENTMEMORY_MEMORY_SHAPING must be one of: off, chain_v1 "
        f"(got {value!r})."
    )


def parse_buy_semantics(value: str | None) -> str:
    mode = (value or "terminate").strip().lower()
    if mode in {"terminate", "terminate_on_wrong", "formal"}:
        return "terminate"
    if mode in {"continue", "commit_continue", "benchmark_replay"}:
        return "continue"
    if mode in {"retry", "verifier_retry", "curriculum"}:
        return "retry"
    raise ValueError(
        "AGENTMEMORY_BUY_SEMANTICS must be one of: terminate, continue, retry "
        f"(got {value!r})."
    )


def has_compatibility_constraints(product: Product) -> bool:
    return any(key.startswith("compatible_") or key == "required_port" for key in product.attributes)


def has_memory_source_attributes(product: Product) -> bool:
    return any(
        key
        in {
            "tv_size_in",
            "tv_weight_kg",
            "vesa",
            "laptop_size_in",
            "laptop_ports",
            "monitor_size_in",
            "monitor_weight_kg",
            "monitor_ports",
        }
        for key in product.attributes
    )


def has_dependency_constraints(product: Product) -> bool:
    return any(
        key.startswith("compatible_") or key in {"required_port", "supported_vesa", "max_weight_kg"}
        for key in product.attributes
    )
