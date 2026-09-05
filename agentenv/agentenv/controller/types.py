from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, TypedDict, List


TASK_NEUTRAL_TRANSITION_INFO_SCHEMA = "agentmemory_task_neutral_transition_v1"
TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA = (
    "agentmemory_task_neutral_context_transition_v1"
)
TASK_NEUTRAL_ACTION_BUDGET_SCHEMA = "agentmemory_task_neutral_action_budget_v1"
CONTEXT_OPERATION_APPEND = "append_observation"
CONTEXT_OPERATION_PRESERVE = "preserve"
CONTEXT_OPERATION_REPLACE = "replace_messages"
CONTEXT_OPERATIONS = frozenset(
    {
        CONTEXT_OPERATION_APPEND,
        CONTEXT_OPERATION_PRESERVE,
        CONTEXT_OPERATION_REPLACE,
    }
)
POLICY_CONTINUATION_MARKER = "Continue the same task in the unchanged workspace."


@dataclass(frozen=True)
class PolicyActionBudget:
    """Runner-owned global action budget before one sampled policy action."""

    maximum_steps: int
    consumed_steps: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, int)
            or self.maximum_steps <= 0
        ):
            raise ValueError("maximum_steps must be a positive integer")
        if (
            isinstance(self.consumed_steps, bool)
            or not isinstance(self.consumed_steps, int)
            or self.consumed_steps < 0
            or self.consumed_steps >= self.maximum_steps
        ):
            raise ValueError(
                "consumed_steps must be a non-negative integer below maximum_steps"
            )

    @property
    def remaining_steps(self) -> int:
        return self.maximum_steps - self.consumed_steps


def build_task_neutral_action_budget_receipt(
    budget: PolicyActionBudget,
    *,
    auxiliary_steps: int = 0,
    required_auxiliary_steps: int | None = None,
    atomic_operation_blocked: bool = False,
) -> dict[str, Any]:
    """Describe the online cost of one policy action and wrapper-owned work.

    A sampled policy action always costs one step.  Wrapper-owned high-level
    operations may add steps, but an atomic group is either charged in full or
    not executed.  A blocked group terminates the episode without consuming a
    partial auxiliary operation.
    """

    if not isinstance(budget, PolicyActionBudget):
        raise TypeError("budget must be a PolicyActionBudget")
    for name, value in (("auxiliary_steps", auxiliary_steps),):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    required = (
        auxiliary_steps
        if required_auxiliary_steps is None
        else required_auxiliary_steps
    )
    if isinstance(required, bool) or not isinstance(required, int) or required < 0:
        raise ValueError("required_auxiliary_steps must be a non-negative integer")
    if atomic_operation_blocked:
        if auxiliary_steps != 0 or required <= 0:
            raise ValueError(
                "a blocked atomic operation must consume zero of a positive auxiliary requirement"
            )
        if budget.remaining_steps - 1 >= required:
            raise ValueError(
                "atomic operation was marked blocked despite sufficient budget"
            )
    elif required != auxiliary_steps:
        raise ValueError(
            "an unblocked atomic operation must consume every required auxiliary step"
        )
    consumed_after = budget.consumed_steps + 1 + auxiliary_steps
    if consumed_after > budget.maximum_steps:
        raise ValueError("action-budget receipt exceeds maximum_steps")
    return {
        "schema": TASK_NEUTRAL_ACTION_BUDGET_SCHEMA,
        "maximum_steps": budget.maximum_steps,
        "consumed_steps_before": budget.consumed_steps,
        "policy_action_steps": 1,
        "auxiliary_steps": auxiliary_steps,
        "required_auxiliary_steps": required,
        "consumed_steps_after": consumed_after,
        "remaining_steps_after": budget.maximum_steps - consumed_after,
        "atomic_operation_blocked": bool(atomic_operation_blocked),
        "terminate_after_action": bool(atomic_operation_blocked),
    }


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
    action_budget: Mapping[str, Any] | None = None,
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
    result = {
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
    if action_budget is not None:
        result["action_budget"] = dict(action_budget)
    return result

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
