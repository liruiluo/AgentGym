"""Task-neutral policy-authored context compaction for matched baselines.

The environment wrapper owns the trigger and returns a normal
``replace_messages`` receipt.  The shared policy runner still samples every
summary response, records its tokens/log probabilities, and dispatches it
through ``env.step``.  This module never calls a native environment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)


CONTEXT_MEMORY_MODE_FILESYSTEM = "filesystem"
CONTEXT_MEMORY_MODE_COMPACTIONRL = "compactionrl"
CONTEXT_MEMORY_MODES = frozenset(
    {CONTEXT_MEMORY_MODE_FILESYSTEM, CONTEXT_MEMORY_MODE_COMPACTIONRL}
)

COMPACTIONRL_RECEIPT_SCHEMA = "agentmemory_compactionrl_receipt_v1"
COMPACTIONRL_RECENT_STEPS = 2
COMPACTIONRL_SUMMARY_MAX_BYTES = 8 * 1024
COMPACTIONRL_REQUEST_TOKEN_SLACK = 32

def build_compactionrl_request(summary_max_bytes: int, *, retry: bool = False) -> str:
    if (
        isinstance(summary_max_bytes, bool)
        or not isinstance(summary_max_bytes, int)
        or summary_max_bytes <= 0
    ):
        raise ValueError("CompactionRL summary byte limit must be a positive integer")
    if retry:
        return (
            "The previous context-compaction response was empty, exceeded the byte "
            "limit, or did not fit the reconstructed prompt, so it was not accepted. "
            "Retry now. Output only one non-empty plain-text continuation summary of "
            f"at most {summary_max_bytes} UTF-8 bytes; do not emit a tool call, task "
            "action, final answer, Markdown fence, or extra wrapper."
        )
    return (
        "The conversation is nearing its context limit. This is the one explicit "
        "exception to the normal task-action output format. Output only a plain-text "
        "continuation summary; do not emit a tool call, task action, final answer, "
        "Markdown fence, or extra wrapper. Preserve the original goal, completed "
        "actions, important observations and evidence, unresolved errors, current "
        "state, and plausible next steps. Your summary is sampled from the same "
        "trainable policy, is not sent to the native environment, and is retained "
        f"verbatim up to {summary_max_bytes} UTF-8 bytes."
    )


COMPACTIONRL_REQUEST = build_compactionrl_request(COMPACTIONRL_SUMMARY_MAX_BYTES)
COMPACTIONRL_RETRY_REQUEST = build_compactionrl_request(
    COMPACTIONRL_SUMMARY_MAX_BYTES,
    retry=True,
)

COMPACTIONRL_POLICY_SUFFIX = (
    "\n\n# CompactionRL baseline override\n"
    "This run uses policy-authored direct context summaries instead of a "
    "filesystem continuation checkpoint. The normal action-only rule has one "
    "explicit exception: when the final user message asks for context "
    "compaction, output only the requested plain-text continuation summary. "
    "That summary is not a native environment action. On every other turn, "
    "follow the task's normal action grammar exactly. This baseline does not "
    "provide a voluntary `.agent_memory/**` memory mechanism: do not create, "
    "read, update, or rely on that namespace. Ordinary task, source, data, and "
    "artifact files remain available through the normal task tools."
)

COMPACTIONRL_RESUME_PREFIX = (
    "Earlier interaction was compacted without changing the native environment "
    "or workspace. Continue the same task from the policy-authored summary and "
    "the retained recent action/observation pairs below.\n\n"
    "Policy-authored continuation summary:\n"
)


def normalize_context_memory_mode(value: Any) -> str:
    mode = str(value or CONTEXT_MEMORY_MODE_FILESYSTEM).strip().lower()
    if mode not in CONTEXT_MEMORY_MODES:
        raise ValueError(
            f"context_memory_mode must be one of {sorted(CONTEXT_MEMORY_MODES)}, "
            f"got {value!r}"
        )
    return mode


def compactionrl_policy_prompt(base_prompt: str, *, mode: str) -> str:
    """Return the effective system prompt without changing the default mode."""

    normalized_mode = normalize_context_memory_mode(mode)
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("policy system prompt must be non-empty text")
    if normalized_mode == CONTEXT_MEMORY_MODE_FILESYSTEM:
        return base_prompt
    return base_prompt.rstrip() + COMPACTIONRL_POLICY_SUFFIX


def _copy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": str(role), "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return normalized


def _message_digest(messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        [dict(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recent_action_observation_pairs(
    messages: Sequence[Mapping[str, str]],
    *,
    limit: int,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return the latest complete assistant/user pairs without splitting a step."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("recent compaction steps must be a non-negative integer")
    normalized = _copy_messages(messages)
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    cursor = len(normalized) - 1
    while cursor > 0 and len(pairs) < limit:
        observation = normalized[cursor]
        action = normalized[cursor - 1]
        if observation["role"] == "user" and action["role"] == "assistant":
            pairs.append((deepcopy(action), deepcopy(observation)))
            cursor -= 2
            continue
        cursor -= 1
    pairs.reverse()
    return pairs


def _trigger_projection_tokens(
    pressure: PolicyContextPressure,
    *,
    control_request: str,
) -> int:
    # UTF-8 bytes are a conservative upper bound for request token count.  The
    # fixed slack covers chat-template role markers without assuming a tokenizer.
    request_upper_bound = (
        len(control_request.encode("utf-8")) + COMPACTIONRL_REQUEST_TOKEN_SLACK
    )
    return (
        pressure.action_prompt_tokens
        + pressure.max_response_tokens
        + pressure.max_observation_tokens
        + pressure.action_observation_envelope_tokens
        + request_upper_bound
    )


@dataclass(frozen=True)
class CompactionRLCompletion:
    step_output: StepOutput
    context_replaced: bool
    retained_recent_steps: int


class CompactionRLController:
    """State machine shared by every CAMG environment wrapper.

    The controller is intentionally ignorant of environment names, parsers,
    rewards, sessions, and endpoint protocols.
    """

    def __init__(
        self,
        *,
        mode: str = CONTEXT_MEMORY_MODE_FILESYSTEM,
        recent_steps: int = COMPACTIONRL_RECENT_STEPS,
        summary_max_bytes: int = COMPACTIONRL_SUMMARY_MAX_BYTES,
    ) -> None:
        self.mode = normalize_context_memory_mode(mode)
        if (
            isinstance(recent_steps, bool)
            or not isinstance(recent_steps, int)
            or recent_steps < 0
        ):
            raise ValueError("compaction_recent_steps must be a non-negative integer")
        if (
            isinstance(summary_max_bytes, bool)
            or not isinstance(summary_max_bytes, int)
            or summary_max_bytes <= 0
        ):
            raise ValueError("compaction_summary_max_bytes must be a positive integer")
        self.recent_steps = recent_steps
        self.summary_max_bytes = summary_max_bytes
        self._prompt_counter: Callable[[Sequence[Mapping[str, str]]], int] | None = None
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.mode == CONTEXT_MEMORY_MODE_COMPACTIONRL

    @property
    def selected(self) -> bool:
        return self._selected

    def reset(self) -> None:
        self._immutable_framing: list[dict[str, str]] | None = None
        self._current_context: list[dict[str, str]] | None = None
        self._pre_request_context: list[dict[str, str]] | None = None
        self._prompt_capacity: int | None = None
        self._selected = False
        self._retry_pending = False

    def bind_prompt_counter(
        self,
        counter: Callable[[Sequence[Mapping[str, str]]], int],
    ) -> None:
        if not callable(counter):
            raise TypeError("policy prompt counter must be callable")
        self._prompt_counter = counter

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        immutable_framing: Sequence[Mapping[str, str]],
        initial: bool = False,
    ) -> None:
        if not self.enabled:
            return
        normalized = _copy_messages(messages)
        framing = _copy_messages(immutable_framing)
        if initial:
            self._immutable_framing = framing
        elif self._immutable_framing is None:
            raise RuntimeError("CompactionRL context was not initialized")
        elif self._immutable_framing != framing:
            raise RuntimeError("CompactionRL immutable policy framing drifted")
        self._current_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if not self.enabled or self._immutable_framing is None:
            return None
        return build_compactionrl_request(
            self.summary_max_bytes,
            retry=self._retry_pending,
        )

    def prepare_policy_turn(
        self,
        pressure: PolicyContextPressure | None,
    ) -> str | None:
        self._selected = False
        if not self.enabled:
            return None
        if pressure is None:
            raise RuntimeError("CompactionRL requires task-neutral token pressure")
        if self._immutable_framing is None or self._current_context is None:
            raise RuntimeError("CompactionRL policy context is not bound")
        request = self.policy_turn_candidate()
        assert request is not None
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "context reached the prompt cap before a trainable CompactionRL "
                "summary could be sampled"
            )
        if (
            not self._retry_pending
            and _trigger_projection_tokens(pressure, control_request=request) < capacity
        ):
            return None
        if not self._retry_pending:
            self._pre_request_context = deepcopy(self._current_context)
        elif self._pre_request_context is None:
            raise RuntimeError("CompactionRL retry lost its pre-request context")
        self._prompt_capacity = capacity
        self._selected = True
        return request

    def complete(
        self,
        summary: str,
        *,
        native_call_count: int,
        context_epoch: int,
        session_epoch: int,
        policy_step_count: int,
        workspace_continuity_id: str | int | None,
    ) -> CompactionRLCompletion:
        if not self.enabled or not self._selected:
            raise RuntimeError("CompactionRL summary completed without selection")
        if not isinstance(summary, str):
            raise TypeError("CompactionRL summary must be text")
        if self._pre_request_context is None or self._immutable_framing is None:
            raise RuntimeError("CompactionRL summary lost its bound context")
        if self._prompt_capacity is None:
            raise RuntimeError("CompactionRL summary lost its prompt capacity")

        summary_bytes = len(summary.encode("utf-8"))
        valid = bool(summary.strip()) and summary_bytes <= self.summary_max_bytes
        failure_reason = None
        retained_steps = 0
        post_prompt_tokens: int | None = None
        next_control_prompt_tokens: int | None = None
        pre_context = deepcopy(self._pre_request_context)
        if valid:
            pairs = _recent_action_observation_pairs(
                pre_context,
                limit=self.recent_steps,
            )
            next_control_request = build_compactionrl_request(
                self.summary_max_bytes,
            )
            replacement = None
            for keep in range(len(pairs), -1, -1):
                candidate = deepcopy(self._immutable_framing)
                candidate.append(
                    {
                        "role": "user",
                        "content": COMPACTIONRL_RESUME_PREFIX + summary,
                    }
                )
                for action, observation in pairs[-keep:] if keep else ():
                    candidate.extend((deepcopy(action), deepcopy(observation)))
                candidate_tokens = (
                    None
                    if self._prompt_counter is None
                    else int(self._prompt_counter(candidate))
                )
                candidate_with_next_control = candidate + [
                    {
                        "role": "user",
                        "content": next_control_request,
                    }
                ]
                candidate_next_control_tokens = (
                    None
                    if self._prompt_counter is None
                    else int(self._prompt_counter(candidate_with_next_control))
                )
                if candidate_tokens is None or (
                    candidate_tokens <= self._prompt_capacity
                    and candidate_next_control_tokens <= self._prompt_capacity
                ):
                    replacement = candidate
                    retained_steps = keep
                    post_prompt_tokens = candidate_tokens
                    next_control_prompt_tokens = candidate_next_control_tokens
                    break
            if replacement is None:
                valid = False
                failure_reason = "summary_prompt_overflow"
                replacement = pre_context
                self._retry_pending = True
                retained_steps = 0
                if self._prompt_counter is not None:
                    post_prompt_tokens = int(self._prompt_counter(replacement))
            else:
                self._retry_pending = False
                self._pre_request_context = None
        else:
            failure_reason = "empty_summary" if not summary.strip() else "summary_too_large"
            replacement = pre_context
            self._retry_pending = True
            if self._prompt_counter is not None:
                post_prompt_tokens = int(self._prompt_counter(replacement))

        self._selected = False
        context_after = context_epoch + (1 if valid else 0)
        policy_after = policy_step_count + 1
        context_transition = build_task_neutral_context_transition(
            CONTEXT_OPERATION_REPLACE,
            messages=replacement,
        )
        evidence = {
            "schema": COMPACTIONRL_RECEIPT_SCHEMA,
            "event": "context_compaction",
            "context_memory_mode": CONTEXT_MEMORY_MODE_COMPACTIONRL,
            "workspace_continuity_id": (
                None
                if workspace_continuity_id is None
                else str(workspace_continuity_id)
            ),
            "native_environment_call_count": 0,
            "summary_sent_to_native_environment": False,
            "summary_valid": valid,
            "summary_failure_reason": failure_reason,
            "summary_byte_count": summary_bytes,
            "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            "summary_max_bytes": self.summary_max_bytes,
            "summary_specific_reward": False,
            "requested_recent_steps": self.recent_steps,
            "retained_recent_steps": retained_steps,
            "pre_context_message_count": len(pre_context),
            "pre_context_sha256": _message_digest(pre_context),
            "post_context_message_count": len(replacement),
            "post_context_sha256": _message_digest(replacement),
            "post_prompt_token_count": post_prompt_tokens,
            "next_control_prompt_token_count": next_control_prompt_tokens,
            "prompt_capacity": self._prompt_capacity,
            "context_replaced": valid,
            "retry_pending": self._retry_pending,
            "policy_memory_namespace_enabled": False,
        }
        state = (
            "Context compaction accepted; continue from the reconstructed prompt."
            if valid
            else (
                f"Context compaction rejected ({failure_reason}); retry with one "
                "non-empty bounded plain-text summary."
            )
        )
        step_output = StepOutput(
            state=state,
            reward=0.0,
            done=False,
            info=build_task_neutral_transition_info(
                env_info={},
                action_submission={
                    "raw_policy_output": summary,
                    "submitted_action": None,
                    "parser_status": "compactionrl_summary_not_dispatched",
                },
                native_step_before=native_call_count,
                native_step_after=native_call_count,
                native_call_count_before=native_call_count,
                native_call_count_after=native_call_count,
                context_epoch_before=context_epoch,
                context_epoch_after=context_after,
                session_epoch_before=session_epoch,
                session_epoch_after=session_epoch,
                policy_step_before=policy_step_count,
                policy_step_after=policy_after,
                context_transition=context_transition,
                wrapper_evidence=evidence,
            ),
        )
        return CompactionRLCompletion(
            step_output=step_output,
            context_replaced=valid,
            retained_recent_steps=retained_steps,
        )


def configure_compactionrl_controller(
    owner: Any,
    *,
    mode: str = CONTEXT_MEMORY_MODE_FILESYSTEM,
    recent_steps: int = COMPACTIONRL_RECENT_STEPS,
    summary_max_bytes: int = COMPACTIONRL_SUMMARY_MAX_BYTES,
) -> CompactionRLController:
    """Attach one validated controller to an environment-client instance."""

    controller = CompactionRLController(
        mode=mode,
        recent_steps=recent_steps,
        summary_max_bytes=summary_max_bytes,
    )
    owner.context_memory_mode = controller.mode
    owner._context_compactor = controller
    return controller


def context_compaction_controller(owner: Any) -> CompactionRLController:
    """Return an attached controller, preserving legacy ``__new__`` fixtures."""

    controller = getattr(owner, "_context_compactor", None)
    if controller is None:
        controller = configure_compactionrl_controller(
            owner,
            mode=getattr(owner, "context_memory_mode", CONTEXT_MEMORY_MODE_FILESYSTEM),
        )
    if not isinstance(controller, CompactionRLController):
        raise TypeError("environment _context_compactor has the wrong type")
    return controller
