from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, TypedDict, List


TASK_NEUTRAL_TRANSITION_INFO_SCHEMA = "agentmemory_task_neutral_transition_v1"
TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA = (
    "agentmemory_task_neutral_context_transition_v1"
)
CONTEXT_OPERATION_APPEND = "append_observation"
CONTEXT_OPERATION_PRESERVE = "preserve"
CONTEXT_OPERATION_REPLACE = "replace_messages"
CONTEXT_OPERATION_RETRY_CONTROL = "retry_control"
CONTEXT_OPERATIONS = frozenset(
    {
        CONTEXT_OPERATION_APPEND,
        CONTEXT_OPERATION_PRESERVE,
        CONTEXT_OPERATION_REPLACE,
        CONTEXT_OPERATION_RETRY_CONTROL,
    }
)
POLICY_CONTINUATION_MARKER = "Continue the same task in the unchanged workspace."


@dataclass(frozen=True)
class PolicyContextPressure:
    """Task-neutral token accounting supplied to an environment wrapper.

    ``candidate_prompt_tokens`` is the exact prompt length after appending the
    wrapper's candidate control request.  It may be shorter than
    ``action_prompt_tokens`` when a chat template normalizes generation-only
    history during a full rerender.  The wrapper owns the decision to use that
    request; the policy runner only measures token counts.
    """

    action_prompt_tokens: int
    candidate_prompt_tokens: int
    max_prompt_tokens: int
    max_model_tokens: int
    max_response_tokens: int
    max_observation_tokens: int
    action_observation_envelope_tokens: int = 0

    def __post_init__(self) -> None:
        positive = {
            "action_prompt_tokens": self.action_prompt_tokens,
            "candidate_prompt_tokens": self.candidate_prompt_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_model_tokens": self.max_model_tokens,
            "max_response_tokens": self.max_response_tokens,
            "max_observation_tokens": self.max_observation_tokens,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        envelope = self.action_observation_envelope_tokens
        if isinstance(envelope, bool) or not isinstance(envelope, int) or envelope < 0:
            raise ValueError(
                "action_observation_envelope_tokens must be a non-negative integer"
            )

    @property
    def projected_next_prompt_tokens_without_control(self) -> int:
        """Conservative next-turn prompt size if the wrapper does not compact."""

        return (
            self.action_prompt_tokens
            + self.max_response_tokens
            + self.max_observation_tokens
            + self.action_observation_envelope_tokens
        )

    @property
    def effective_prompt_capacity(self) -> int:
        capacity = min(
            self.max_prompt_tokens,
            self.max_model_tokens - self.max_response_tokens,
        )
        if capacity <= 0:
            raise ValueError("effective prompt capacity must be positive")
        return capacity


def build_task_neutral_context_transition(
    operation: str,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the only context mutation envelope understood by shared rollout."""

    if operation not in CONTEXT_OPERATIONS:
        raise ValueError(f"unsupported context transition operation: {operation!r}")
    normalized_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages or ()):
        if not isinstance(message, Mapping):
            raise TypeError(f"context message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"context message {index} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"context message {index} content must be text")
        normalized_messages.append({"role": role, "content": content})
    if operation == CONTEXT_OPERATION_REPLACE and not normalized_messages:
        raise ValueError("replace_messages requires at least one context message")
    if operation != CONTEXT_OPERATION_REPLACE and normalized_messages:
        raise ValueError(f"{operation} must not carry replacement messages")
    return {
        "schema": TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
        "operation": operation,
        "messages": normalized_messages,
    }


def build_task_neutral_transition_info(
    *,
    env_info: Mapping[str, Any] | None = None,
    action_submission: Mapping[str, Any] | None = None,
    native_step_before: int | None = None,
    native_step_after: int | None = None,
    native_call_count_before: int | None = None,
    native_call_count_after: int | None = None,
    context_epoch_before: int | None = None,
    context_epoch_after: int | None = None,
    session_epoch_before: int | None = None,
    session_epoch_after: int | None = None,
    policy_step_before: int | None = None,
    policy_step_after: int | None = None,
    context_transition: Mapping[str, Any] | None = None,
    wrapper_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable envelope shared rollout code may consume.

    The envelope intentionally keeps domain evidence opaque under ``env_info``
    and ``wrapper_evidence``.  A wrapper owns the meaning of those fields; the
    policy runner only preserves them alongside sampled tokens and rewards.
    ``None`` is a valid value for environments that do not expose a counter.
    """

    transition = (
        build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND)
        if context_transition is None
        else dict(context_transition)
    )
    if transition.get("schema") != TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA:
        raise ValueError("context transition has an unsupported schema")
    canonical_transition = build_task_neutral_context_transition(
        str(transition.get("operation")),
        messages=transition.get("messages"),
    )
    return {
        "schema": TASK_NEUTRAL_TRANSITION_INFO_SCHEMA,
        "env_info": dict(env_info or {}),
        "action_submission": (
            None if action_submission is None else dict(action_submission)
        ),
        "native_step_before": native_step_before,
        "native_step_after": native_step_after,
        "native_call_count_before": native_call_count_before,
        "native_call_count_after": native_call_count_after,
        "context_epoch_before": context_epoch_before,
        "context_epoch_after": context_epoch_after,
        "session_epoch_before": session_epoch_before,
        "session_epoch_after": session_epoch_after,
        "policy_step_before": policy_step_before,
        "policy_step_after": policy_step_after,
        "context_transition": canonical_transition,
        "wrapper_evidence": dict(wrapper_evidence or {}),
    }

ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)

APIConversationMessage = TypedDict(
    "APIConversationMessage", {"role": str, "content": str, "reasoning_content": Optional[str]}
)

TokenizedConversationOutput = TypedDict(
    "TokenizedConversationOutput",
    {
        "text": str,
        "input_ids": Sequence[int],
        "action_mask": Sequence[int],
    },
)


class ActionFormat(Enum):
    REACT = "react"
    FUNCTION_CALLING = "function_calling"
    CODE_AS_ACTION = "code_as_action"


class InferenceEngine(Enum):
    DEFAULT = "default"
    VLLM = "vllm"


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool
    # Wrapper-owned, task-neutral transition evidence.  Existing environments
    # may leave it empty; long-horizon wrappers use it to preserve lifecycle
    # boundaries without teaching the shared rollout about domain semantics.
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float
    text: str
    seq_ids: list[int]
    attention_mask: list[int]
    action_mask: list[int]


@dataclass
class APIExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float


@dataclass
class ActionWithTought:
    thought: str
    action: str


@dataclass
class EvaluationOutput:
    experiences: list[ExperienceOutput]
    score: float
    success: float


@dataclass
class Function():
    name: str
    arguments: str


@dataclass
class ChatCompletionMessageToolCall():
    # tool_call id
    id: str

    # extracted tool calls
    function: Function
