"""Task-neutral AgeMem-style memory tools for CAMG environment clients.

This adapter keeps the native environment wrapper authoritative for task
lifecycle, reward, grading, and mandatory CAMG continuation checkpoints.  It
adds six policy-authored memory tools as ordinary sampled actions without
changing the shared rollout or PPO path.  The store is created on reset and is
never shared across episodes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NamedTuple

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_REPLACE,
    TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)

AGEMEM_ADAPTER_SCHEMA = "camg_agemem_style_adapter_v1"
AGEMEM_ACTION_SCHEMA = "camg_agemem_style_action_v1"
AGEMEM_PROMPT_MARKER = "[CAMG_AGEMEM_STYLE_TOOLS_V1]"
AGEMEM_TOOL_NAMES = (
    "Add_memory",
    "Update_memory",
    "Delete_memory",
    "Retrieve_memory",
    "Summary_context",
    "Filter_context",
)

_TOOL_OPEN = "<agemem_tool_call>"
_TOOL_CLOSE = "</agemem_tool_call>"
_TOOL_RE = re.compile(
    rf"\A{re.escape(_TOOL_OPEN)}\s*(.*?)\s*{re.escape(_TOOL_CLOSE)}\Z",
    re.DOTALL,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)
_ALLOWED_CONTINUATION_PATH = ".agent_memory/CONTINUATION.md"
_PATH_TERMINATORS = frozenset(" \t\r\n'\"`;,:|&<>()[]{}>")


class AgeMemInvocation(NamedTuple):
    name: str
    arguments: dict[str, Any]


class AgeMemActionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgeMemAdapterConfig:
    max_memories: int = 128
    max_content_bytes: int = 8192
    max_metadata_bytes: int = 2048
    max_retrieval_k: int = 5
    max_summary_bytes: int = 8192
    max_observation_bytes: int = 24576
    filter_similarity_threshold: float = 0.6
    invalid_action_reward: float = 0.0
    forbid_additional_policy_memory_files: bool = True

    def __post_init__(self) -> None:
        for field in (
            "max_memories",
            "max_content_bytes",
            "max_metadata_bytes",
            "max_retrieval_k",
            "max_summary_bytes",
            "max_observation_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"AgeMem {field} must be a positive integer")
        if self.max_observation_bytes < self.max_content_bytes:
            raise ValueError(
                "AgeMem max_observation_bytes must be at least max_content_bytes"
            )
        threshold = self.filter_similarity_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) <= 1.0
        ):
            raise ValueError(
                "AgeMem filter_similarity_threshold must be finite in (0, 1]"
            )
        reward = self.invalid_action_reward
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or float(reward) > 0.0
        ):
            raise ValueError(
                "AgeMem invalid_action_reward must be finite and non-positive"
            )
        if type(self.forbid_additional_policy_memory_files) is not bool:
            raise TypeError(
                "AgeMem forbid_additional_policy_memory_files must be boolean"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AgeMemAdapterConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("AgeMem adapter config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in value if str(key) not in allowed)
        if unknown:
            raise ValueError("unknown AgeMem adapter config fields: " + ", ".join(unknown))
        return cls(**{str(key): item for key, item in value.items()})


@dataclass
class _MemoryEntry:
    memory_id: str
    content: str
    metadata: dict[str, Any]
    memory_type: str | None
    created_at_action: int
    updated_at_action: int

    def public(self, *, score: float | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "memory_id": self.memory_id,
            "content": self.content,
            "metadata": deepcopy(self.metadata),
            "memory_type": self.memory_type,
            "created_at_action": self.created_at_action,
            "updated_at_action": self.updated_at_action,
        }
        if score is not None:
            value["score"] = round(float(score), 6)
        return value


def parse_agemem_action(action: str) -> AgeMemInvocation | None:
    """Parse one exact AgeMem action; return ``None`` for native actions."""

    if not isinstance(action, str):
        raise TypeError("policy action must be text")
    mentions_adapter = _TOOL_OPEN in action or _TOOL_CLOSE in action
    if not mentions_adapter:
        return None
    match = _TOOL_RE.fullmatch(action.strip())
    if match is None:
        raise ValueError(
            "AgeMem action must contain only one exact agemem_tool_call block"
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"AgeMem tool payload is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("AgeMem action must contain exactly one tool invocation")
    call = payload[0]
    if not isinstance(call, Mapping) or set(call) != {"name", "arguments"}:
        raise ValueError("AgeMem invocation must contain only name and arguments")
    name = call.get("name")
    arguments = call.get("arguments")
    if name not in AGEMEM_TOOL_NAMES:
        raise ValueError(f"unsupported AgeMem tool: {name!r}")
    if not isinstance(arguments, Mapping):
        raise ValueError("AgeMem tool arguments must be an object")
    return AgeMemInvocation(str(name), {str(k): deepcopy(v) for k, v in arguments.items()})


class AgeMemEnvClientAdapter(BaseEnvClient):
    """Add six episode-private AgeMem-style tools to any CAMG env client."""

    def __init__(
        self,
        native_client: BaseEnvClient,
        config: AgeMemAdapterConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(native_client, BaseEnvClient):
            raise TypeError("AgeMem adapter requires a BaseEnvClient")
        super().__init__(native_client.action_format)
        self.native_client = native_client
        self.config = (
            config
            if isinstance(config, AgeMemAdapterConfig)
            else AgeMemAdapterConfig.from_mapping(config)
        )
        self._memories: dict[str, _MemoryEntry] = {}
        self._retrieved_ids: set[str] = set()
        self._next_memory_sequence = 1
        self._memory_action_count = 0
        self._current_policy_context: list[dict[str, str]] | None = None
        self._native_control_selected = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_client, name)

    def __len__(self) -> int:
        return len(self.native_client)

    @property
    def memory_size(self) -> int:
        return len(self._memories)

    @property
    def sample_excluded(self) -> bool:
        return bool(getattr(self.native_client, "sample_excluded", False))

    def observe(self) -> str:
        return str(self.native_client.observe())

    def policy_framing(self) -> list[dict[str, str]]:
        method = getattr(self.native_client, "policy_framing", None)
        native = method() if callable(method) else None
        return self._inject_prompt(_copy_messages(native or ()))

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        native_messages = self._strip_prompt(_copy_messages(messages))
        normalized = self.native_client.normalize_initial_policy_context(native_messages)
        return self._inject_prompt(_copy_messages(normalized))

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        adapted = _copy_messages(messages)
        self._current_policy_context = deepcopy(adapted)
        self.native_client.bind_policy_context(
            self._strip_prompt(adapted),
            initial=initial,
        )

    def policy_turn_candidate(self) -> str | None:
        self._native_control_selected = False
        return self.native_client.policy_turn_candidate()

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        selected = self.native_client.prepare_policy_turn(pressure)
        self._native_control_selected = selected is not None
        return selected

    def step(self, action: str) -> StepOutput:
        if not isinstance(action, str):
            raise TypeError("AgeMem-adapted policy action must be text")
        if self._native_control_selected:
            self._native_control_selected = False
            return self._wrap_native_output(self.native_client.step(action))

        try:
            invocation = parse_agemem_action(action)
        except ValueError as exc:
            return self._error_step(
                action,
                code="memory_action_parse_error",
                message=str(exc),
            )
        if invocation is not None:
            return self._execute_memory_action(action, invocation)
        if (
            self.config.forbid_additional_policy_memory_files
            and _touches_forbidden_policy_memory_path(action)
        ):
            return self._error_step(
                action,
                code="filesystem_memory_namespace_disabled",
                message=(
                    "AgeMem-style baseline permits the mandatory "
                    f"{_ALLOWED_CONTINUATION_PATH} path only; use the six AgeMem tools "
                    "for voluntary memory"
                ),
            )
        return self._wrap_native_output(self.native_client.step(action))

    def reset(self, idx: int = 0) -> Any:
        # Clear first: a native reset may partially advance and then raise.  The
        # adapter must fail closed rather than retain an earlier episode's LTM
        # if a caller retries the same client instance.
        self._memories.clear()
        self._retrieved_ids.clear()
        self._next_memory_sequence = 1
        self._memory_action_count = 0
        self._current_policy_context = None
        self._native_control_selected = False
        return self.native_client.reset(idx)

    def finalize_policy_horizon(self) -> StepOutput | None:
        output = self.native_client.finalize_policy_horizon()
        return None if output is None else self._wrap_native_output(output)

    def close(self) -> Any:
        return self.native_client.close()

    def _execute_memory_action(
        self, raw_action: str, invocation: AgeMemInvocation
    ) -> StepOutput:
        before = self.memory_size
        self._memory_action_count += 1
        try:
            payload, transition = self._dispatch(invocation)
        except AgeMemActionError as exc:
            return self._error_step(
                raw_action,
                code=exc.code,
                message=str(exc),
                operation=invocation.name,
                increment=False,
            )
        payload = {
            "schema": AGEMEM_ACTION_SCHEMA,
            "ok": True,
            "operation": invocation.name,
            **payload,
            "memory_size_before": before,
            "memory_size_after": self.memory_size,
            "memory_action_index": self._memory_action_count,
        }
        return self._build_memory_step(
            raw_action=raw_action,
            payload=payload,
            transition=transition,
            accepted=True,
        )

    def _dispatch(
        self, invocation: AgeMemInvocation
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        handlers = {
            "Add_memory": self._add_memory,
            "Update_memory": self._update_memory,
            "Delete_memory": self._delete_memory,
            "Retrieve_memory": self._retrieve_memory,
            "Summary_context": self._summary_context,
            "Filter_context": self._filter_context,
        }
        return handlers[invocation.name](invocation.arguments)

    def _add_memory(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_keys(arguments, required={"content"}, optional={"metadata", "memory_type"})
        if self.memory_size >= self.config.max_memories:
            raise AgeMemActionError("memory_store_full", "AgeMem store is full")
        content = self._bounded_text(arguments["content"], field="content")
        metadata = self._metadata(arguments.get("metadata", {}))
        memory_type = arguments.get("memory_type")
        if memory_type is not None:
            memory_type = self._bounded_text(
                memory_type,
                field="memory_type",
                max_bytes=min(256, self.config.max_content_bytes),
            )
        memory_id = f"m{self._next_memory_sequence:06d}"
        self._next_memory_sequence += 1
        entry = _MemoryEntry(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            memory_type=memory_type,
            created_at_action=self._memory_action_count,
            updated_at_action=self._memory_action_count,
        )
        self._memories[memory_id] = entry
        return (
            {"status": "added", "memory_id": memory_id},
            build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND),
        )

    def _update_memory(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_keys(
            arguments,
            required={"memory_id", "content"},
            optional={"metadata"},
        )
        memory_id = self._memory_id(arguments["memory_id"])
        self._require_retrieved(memory_id)
        entry = self._require_memory(memory_id)
        content = self._bounded_text(arguments["content"], field="content")
        metadata = (
            self._metadata(arguments["metadata"])
            if "metadata" in arguments
            else deepcopy(entry.metadata)
        )
        entry.content = content
        entry.metadata = metadata
        entry.updated_at_action = self._memory_action_count
        return (
            {"status": "updated", "memory_id": memory_id},
            build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND),
        )

    def _delete_memory(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_keys(
            arguments,
            required={"memory_id", "confirmation"},
            optional=set(),
        )
        memory_id = self._memory_id(arguments["memory_id"])
        self._require_retrieved(memory_id)
        self._require_memory(memory_id)
        if arguments["confirmation"] is not True:
            raise AgeMemActionError(
                "delete_confirmation_required",
                "Delete_memory requires confirmation=true",
            )
        del self._memories[memory_id]
        self._retrieved_ids.discard(memory_id)
        return (
            {"status": "deleted", "memory_id": memory_id},
            build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND),
        )

    def _retrieve_memory(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_keys(
            arguments,
            required={"query"},
            optional={"top_k", "metadata_filter"},
        )
        query = self._bounded_text(arguments["query"], field="query")
        raw_top_k = arguments.get("top_k", 3)
        if type(raw_top_k) is not int:
            raise AgeMemActionError("invalid_top_k", "top_k must be an integer")
        top_k = raw_top_k
        if top_k <= 0 or top_k > self.config.max_retrieval_k:
            raise AgeMemActionError(
                "invalid_top_k",
                f"top_k must be within 1..{self.config.max_retrieval_k}",
            )
        metadata_filter = self._metadata(arguments.get("metadata_filter", {}))
        query_tokens = _tokens(query)
        ranked: list[tuple[float, int, _MemoryEntry]] = []
        for entry in self._memories.values():
            if any(entry.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            score = _cosine_overlap(query_tokens, _tokens(entry.content))
            sequence = int(entry.memory_id[1:])
            ranked.append((score, sequence, entry))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked[:top_k]
        retrieved_memory_ids = [entry.memory_id for _, _, entry in selected]
        self._retrieved_ids.update(retrieved_memory_ids)
        return (
            {
                "status": "retrieved",
                "query": query,
                "memories": [entry.public(score=score) for score, _, entry in selected],
                "retrieved_memory_ids": retrieved_memory_ids,
                "retrieved_memory_count": len(retrieved_memory_ids),
            },
            build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND),
        )

    def _summary_context(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_keys(
            arguments,
            required={"span", "summary"},
            optional=set(),
        )
        summary = self._bounded_text(
            arguments["summary"],
            field="summary",
            max_bytes=self.config.max_summary_bytes,
        )
        messages = self._bound_context()
        non_system_indices = [
            index for index, message in enumerate(messages) if message["role"] != "system"
        ]
        span = arguments["span"]
        if span == "all":
            selected = non_system_indices
        else:
            if isinstance(span, bool):
                raise AgeMemActionError("invalid_span", "span must be 'all' or a positive integer string")
            text = str(span)
            if not text.isdigit() or int(text) <= 0:
                raise AgeMemActionError("invalid_span", "span must be 'all' or a positive integer string")
            selected = non_system_indices[-int(text) :]
        if not selected:
            raise AgeMemActionError("empty_summary_span", "Summary_context selected no messages")
        selected_set = set(selected)
        replacement = [
            deepcopy(message)
            for index, message in enumerate(messages)
            if index not in selected_set
        ]
        replacement.append(
            {
                "role": "user",
                "content": (
                    "[AgeMem Summary_context result]\n"
                    f"Policy-authored summary of {len(selected)} message(s):\n{summary}"
                ),
            }
        )
        return (
            {"status": "summarized", "summarized_message_count": len(selected)},
            build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=self._inject_prompt(replacement),
            ),
        )

    def _filter_context(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_keys(arguments, required={"criteria"}, optional=set())
        criteria = self._bounded_text(arguments["criteria"], field="criteria")
        criteria_tokens = _tokens(criteria)
        if not criteria_tokens:
            raise AgeMemActionError("empty_filter_criteria", "criteria has no searchable tokens")
        messages = self._bound_context()
        removed: list[int] = []
        replacement: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if message["role"] == "system":
                replacement.append(deepcopy(message))
                continue
            score = _criteria_overlap(criteria_tokens, _tokens(message["content"]))
            if score >= self.config.filter_similarity_threshold:
                removed.append(index)
            else:
                replacement.append(deepcopy(message))
        result_text = (
            "[AgeMem Filter_context result]\n"
            f"Removed {len(removed)} message(s) matching criteria: {criteria}"
        )
        if removed:
            replacement.append({"role": "user", "content": result_text})
            transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=self._inject_prompt(replacement),
            )
        else:
            transition = build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND)
        return (
            {
                "status": "filtered",
                "removed_message_count": len(removed),
                "removed_message_indices": removed,
            },
            transition,
        )

    def _error_step(
        self,
        raw_action: str,
        *,
        code: str,
        message: str,
        operation: str | None = None,
        increment: bool = True,
    ) -> StepOutput:
        before = self.memory_size
        if increment:
            self._memory_action_count += 1
        payload = {
            "schema": AGEMEM_ACTION_SCHEMA,
            "ok": False,
            "operation": operation,
            "error_code": code,
            "error": message,
            "memory_size_before": before,
            "memory_size_after": self.memory_size,
            "memory_action_index": self._memory_action_count,
        }
        return self._build_memory_step(
            raw_action=raw_action,
            payload=payload,
            transition=build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND),
            accepted=False,
        )

    def _build_memory_step(
        self,
        *,
        raw_action: str,
        payload: Mapping[str, Any],
        transition: Mapping[str, Any],
        accepted: bool,
    ) -> StepOutput:
        observation = "[AgeMem tool result]\n" + _bounded_json(
            payload,
            max_bytes=self.config.max_observation_bytes,
        )
        evidence = {
            "schema": AGEMEM_ADAPTER_SCHEMA,
            "event": "memory_tool_action",
            "operation": payload.get("operation"),
            "accepted": accepted,
            "memory_action_index": self._memory_action_count,
            "memory_size_before": payload.get("memory_size_before"),
            "memory_size_after": payload.get("memory_size_after"),
            "context_operation": transition.get("operation"),
            "episode_private": True,
            "hidden_model_calls": 0,
        }
        for field in (
            "memory_id",
            "retrieved_memory_ids",
            "retrieved_memory_count",
            "summarized_message_count",
            "removed_message_count",
            "error_code",
        ):
            if field in payload:
                evidence[field] = deepcopy(payload[field])
        return StepOutput(
            state=observation,
            reward=float(self.config.invalid_action_reward) if not accepted else 0.0,
            done=False,
            info=build_task_neutral_transition_info(
                action_submission={
                    "schema": AGEMEM_ACTION_SCHEMA,
                    "raw_policy_output": raw_action,
                    "submitted_action": raw_action,
                    "parser_status": "agemem_adapter",
                    "accepted": accepted,
                    "operation": payload.get("operation"),
                    "error_code": payload.get("error_code"),
                },
                context_transition=transition,
                wrapper_evidence={"agemem_adapter": evidence},
            ),
        )

    def _wrap_native_output(self, output: StepOutput) -> StepOutput:
        if not isinstance(output, StepOutput):
            raise TypeError("native client step must return StepOutput")
        info = deepcopy(dict(output.info)) if isinstance(output.info, Mapping) else {}
        transition = info.get("context_transition")
        if isinstance(transition, Mapping):
            transition = deepcopy(dict(transition))
            if (
                transition.get("schema") == TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA
                and transition.get("operation") == CONTEXT_OPERATION_REPLACE
            ):
                transition["messages"] = self._inject_prompt(
                    _copy_messages(transition.get("messages", ()))
                )
            info["context_transition"] = transition
        evidence = info.get("wrapper_evidence")
        evidence = deepcopy(dict(evidence)) if isinstance(evidence, Mapping) else {}
        evidence["agemem_adapter"] = {
            "schema": AGEMEM_ADAPTER_SCHEMA,
            "event": "native_action_passthrough",
            "memory_action_count": self._memory_action_count,
            "memory_size_after": self.memory_size,
            "episode_private": True,
            "hidden_model_calls": 0,
        }
        info["wrapper_evidence"] = evidence
        reward = output.reward
        if reward is None:
            if not bool(output.done) or not self.sample_excluded:
                raise RuntimeError(
                    "native client returned a null reward without a terminal "
                    "sample-exclusion receipt"
                )
        else:
            reward = float(reward)
        return StepOutput(
            state=str(output.state),
            reward=reward,
            done=bool(output.done),
            info=info,
        )

    def _inject_prompt(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        normalized = _copy_messages(messages)
        for message in normalized:
            if message["role"] != "system":
                continue
            if AGEMEM_PROMPT_MARKER not in message["content"]:
                message["content"] = message["content"] + "\n\n" + _agemem_prompt()
            return normalized
        return [{"role": "system", "content": _agemem_prompt()}, *normalized]

    def _strip_prompt(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        normalized = _copy_messages(messages)
        prompt = _agemem_prompt()
        stripped: list[dict[str, str]] = []
        for message in normalized:
            if message["role"] == "system":
                content = message["content"]
                if content == prompt:
                    continue
                suffix = "\n\n" + prompt
                if content.endswith(suffix):
                    message["content"] = content[: -len(suffix)]
            stripped.append(message)
        return stripped

    def _bound_context(self) -> list[dict[str, str]]:
        if not self._current_policy_context:
            raise AgeMemActionError(
                "policy_context_unbound",
                "AgeMem context tool requires a bound policy context",
            )
        return deepcopy(self._current_policy_context)

    def _require_memory(self, memory_id: str) -> _MemoryEntry:
        try:
            return self._memories[memory_id]
        except KeyError as exc:
            raise AgeMemActionError(
                "memory_id_not_found", f"unknown memory_id {memory_id!r}"
            ) from exc

    def _require_retrieved(self, memory_id: str) -> None:
        if memory_id not in self._retrieved_ids:
            raise AgeMemActionError(
                "memory_id_not_retrieved",
                "Update_memory/Delete_memory require an id returned by Retrieve_memory",
            )

    @staticmethod
    def _memory_id(value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"m[0-9]{6}", value):
            raise AgeMemActionError("invalid_memory_id", "memory_id is invalid")
        return value

    def _bounded_text(
        self,
        value: Any,
        *,
        field: str,
        max_bytes: int | None = None,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AgeMemActionError(f"invalid_{field}", f"{field} must be non-empty text")
        limit = self.config.max_content_bytes if max_bytes is None else max_bytes
        if len(value.encode("utf-8")) > limit:
            raise AgeMemActionError(
                f"{field}_too_large", f"{field} exceeds {limit} UTF-8 bytes"
            )
        return value.strip()

    def _metadata(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise AgeMemActionError("invalid_metadata", "metadata must be an object")
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise AgeMemActionError("invalid_metadata", "metadata must be JSON-safe") from exc
        if len(encoded) > self.config.max_metadata_bytes:
            raise AgeMemActionError(
                "metadata_too_large",
                f"metadata exceeds {self.config.max_metadata_bytes} UTF-8 bytes",
            )
        return json.loads(encoded.decode("utf-8"))


def _agemem_prompt() -> str:
    return f"""{AGEMEM_PROMPT_MARKER}
This matched AgeMem-style baseline adds six episode-private memory tools. A memory tool is one ordinary policy action and consumes the same response/action budget as a native task action. Memory persists across context replacement inside this episode and is cleared at reset. It is never shared across episodes. No memory tool receives a positive reward and no hidden model is called.

To call one memory tool, output exactly one JSON invocation and no prose:
<agemem_tool_call>[{{\"name\":\"TOOL_NAME\",\"arguments\":{{...}}}}]</agemem_tool_call>
Exactly one invocation is allowed per policy response.

LTM tools:
- Add_memory: required content; optional metadata object and memory_type.
- Retrieve_memory: required query; optional top_k (1-5) and metadata_filter object. It returns memory_id values.
- Update_memory: required previously retrieved memory_id and replacement content; optional metadata.
- Delete_memory: required previously retrieved memory_id and confirmation=true.
STM tools:
- Summary_context: required span (\"all\" or a positive integer string) and policy-authored summary. The selected non-system messages are replaced by the supplied summary; no hidden summarizer is used.
- Filter_context: required criteria. Messages that deterministically match the removal criteria are deleted from active context.

Native task tools and final-submission syntax remain unchanged. Do not combine a memory tool with a native action. The mandatory .agent_memory/CONTINUATION.md checkpoint may still be requested by the CAMG environment; follow that request exactly. For voluntary memory, use these six tools rather than other .agent_memory files."""


def _copy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("policy messages must be a sequence")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"invalid policy message at index {index}")
        normalized.append({"role": str(role), "content": content})
    return normalized


def _require_keys(
    arguments: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
) -> None:
    keys = set(arguments)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise AgeMemActionError("missing_arguments", "missing arguments: " + ", ".join(missing))
    if unknown:
        raise AgeMemActionError("unknown_arguments", "unknown arguments: " + ", ".join(unknown))


def _tokens(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(value)}


def _cosine_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _criteria_overlap(criteria: set[str], message: set[str]) -> float:
    if not criteria:
        return 0.0
    return len(criteria & message) / len(criteria)


def _touches_forbidden_policy_memory_path(action: str) -> bool:
    """Allow only exact references to the mandatory continuation path.

    This is a protocol guard, not a substitute for the native sandbox.  In
    particular, suffix tricks such as ``CONTINUATION.md/../notes`` and
    ``CONTINUATION.md.bak`` must not turn the one allowed path into a namespace
    prefix.  Any other literal reference to ``.agent_memory`` fails closed.
    """

    cursor = 0
    marker = ".agent_memory"
    while True:
        start = action.find(marker, cursor)
        if start < 0:
            return False
        if not action.startswith(_ALLOWED_CONTINUATION_PATH, start):
            return True
        end = start + len(_ALLOWED_CONTINUATION_PATH)
        if end < len(action) and action[end] not in _PATH_TERMINATORS:
            return True
        cursor = end


def _bounded_json(value: Mapping[str, Any], *, max_bytes: int) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    raw = encoded.encode("utf-8")
    if len(raw) <= max_bytes:
        return encoded
    # A retrieval response can be large even though every stored item is bounded.
    # Preserve valid JSON and evidence rather than byte-slicing the serialized text.
    compact = dict(value)
    memories = compact.get("memories")
    if isinstance(memories, list):
        compact["memories"] = [
            {
                **dict(item),
                "content": _truncate_utf8(str(item.get("content", "")), 512),
            }
            for item in memories
            if isinstance(item, Mapping)
        ]
        compact["observation_truncated"] = True
    encoded = json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > max_bytes:
        raise RuntimeError("AgeMem result exceeds its observation byte budget")
    return encoded


def _truncate_utf8(value: str, max_bytes: int) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    return raw[:max_bytes].decode("utf-8", errors="ignore") + "…"
