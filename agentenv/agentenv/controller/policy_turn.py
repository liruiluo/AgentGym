from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .env import BaseEnvClient
from .types import (
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_PRESERVE,
    CONTEXT_OPERATION_REPLACE,
    CONTEXT_OPERATION_RETRY_CONTROL,
    TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
    PolicyContextPressure,
    StepOutput,
)


PolicyMessage = dict[str, str]


@dataclass(frozen=True)
class PreparedPolicyTurn:
    messages: tuple[PolicyMessage, ...]
    prompt_token_count: int
    control_request: str | None
    pre_sampling_terminal: StepOutput | None = None


def bind_initial_policy_context(
    client: BaseEnvClient,
    messages: Sequence[Mapping[str, str]],
) -> list[PolicyMessage]:
    prepared = client.normalize_initial_policy_context(deepcopy(list(messages)))
    normalized = _normalize_messages(prepared)
    client.bind_policy_context(deepcopy(normalized), initial=True)
    return normalized


def prepare_policy_turn(
    client: BaseEnvClient,
    messages: Sequence[Mapping[str, str]],
    *,
    count_prompt_tokens: Callable[[Sequence[Mapping[str, str]]], int],
    max_prompt_tokens: int,
    max_model_tokens: int,
    max_response_tokens: int,
    max_observation_tokens: int | None,
    action_observation_envelope_tokens: int = 0,
) -> PreparedPolicyTurn:
    """Prepare one task-neutral policy row through a wrapper-owned decision."""

    action_messages = _normalize_messages(messages)
    client.bind_policy_context(deepcopy(action_messages), initial=False)
    action_prompt_tokens = int(count_prompt_tokens(action_messages))
    effective_prompt_capacity = min(
        int(max_prompt_tokens), int(max_model_tokens) - int(max_response_tokens)
    )
    if effective_prompt_capacity <= 0:
        raise ValueError("effective prompt capacity must be positive")
    if action_prompt_tokens > effective_prompt_capacity:
        if max_observation_tokens is None:
            raise RuntimeError(
                "prompt-capacity finalization requires an explicit "
                "observation-token budget"
            )
        pressure = PolicyContextPressure(
            action_prompt_tokens=action_prompt_tokens,
            candidate_prompt_tokens=action_prompt_tokens,
            max_prompt_tokens=int(max_prompt_tokens),
            max_model_tokens=int(max_model_tokens),
            max_response_tokens=int(max_response_tokens),
            max_observation_tokens=int(max_observation_tokens),
            action_observation_envelope_tokens=int(
                action_observation_envelope_tokens
            ),
        )
        capacity_finalizer = getattr(
            client, "finalize_policy_prompt_capacity", None
        )
        if callable(capacity_finalizer):
            terminal = capacity_finalizer(pressure)
        else:
            horizon_finalizer = getattr(client, "finalize_policy_horizon", None)
            terminal = horizon_finalizer() if callable(horizon_finalizer) else None
        if terminal is None:
            raise RuntimeError(
                "policy prompt exceeded capacity and the wrapper did not provide "
                "a terminal finalization"
            )
        if not isinstance(terminal, StepOutput):
            raise TypeError(
                "prompt-capacity finalization must return StepOutput or None"
            )
        if not terminal.done:
            raise RuntimeError(
                "prompt-capacity finalization must terminate the episode"
            )
        return PreparedPolicyTurn(
            messages=tuple(action_messages),
            prompt_token_count=action_prompt_tokens,
            control_request=None,
            pre_sampling_terminal=terminal,
        )

    candidate = client.policy_turn_candidate()
    if candidate is None:
        return PreparedPolicyTurn(
            messages=tuple(action_messages),
            prompt_token_count=action_prompt_tokens,
            control_request=None,
        )
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("policy_turn_candidate must return nonempty text or None")
    candidate_messages = action_messages + [{"role": "user", "content": candidate}]
    candidate_prompt_tokens = int(count_prompt_tokens(candidate_messages))
    if max_observation_tokens is None:
        raise RuntimeError(
            "a wrapper control turn requires an explicit observation-token budget"
        )
    pressure = PolicyContextPressure(
        action_prompt_tokens=action_prompt_tokens,
        candidate_prompt_tokens=candidate_prompt_tokens,
        max_prompt_tokens=int(max_prompt_tokens),
        max_model_tokens=int(max_model_tokens),
        max_response_tokens=int(max_response_tokens),
        max_observation_tokens=int(max_observation_tokens),
        action_observation_envelope_tokens=int(action_observation_envelope_tokens),
    )
    selected = client.prepare_policy_turn(pressure)
    if selected is None:
        return PreparedPolicyTurn(
            messages=tuple(action_messages),
            prompt_token_count=action_prompt_tokens,
            control_request=None,
        )
    if selected != candidate:
        raise ValueError("wrapper selected a control request other than its candidate")
    client.bind_policy_context(deepcopy(candidate_messages), initial=False)
    return PreparedPolicyTurn(
        messages=tuple(candidate_messages),
        prompt_token_count=candidate_prompt_tokens,
        control_request=selected,
    )


def complete_policy_turn(
    client: BaseEnvClient,
    prepared: PreparedPolicyTurn,
    policy_output: str,
) -> tuple[StepOutput, list[PolicyMessage]]:
    """Dispatch every sampled output through ``env.step`` and apply its receipt."""

    if prepared.pre_sampling_terminal is not None:
        raise ValueError(
            "a pre-sampling terminal turn cannot dispatch a policy output"
        )
    if not isinstance(policy_output, str):
        raise TypeError("policy output must be text")
    step_output = client.step(policy_output)
    messages = [dict(message) for message in prepared.messages]
    messages.append({"role": "assistant", "content": policy_output})
    transition = step_output.info.get("context_transition", {})
    if not transition:
        operation = CONTEXT_OPERATION_APPEND
        replacement: list[PolicyMessage] = []
    else:
        if transition.get("schema") != TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA:
            raise ValueError("wrapper returned an unsupported context transition")
        operation = transition.get("operation")
        transition_messages = transition.get("messages", ())
        if operation == CONTEXT_OPERATION_REPLACE:
            replacement = _normalize_messages(transition_messages)
        else:
            if transition_messages:
                raise ValueError(
                    f"{operation} receipt must not carry replacement messages"
                )
            replacement = []

    if operation == CONTEXT_OPERATION_APPEND:
        messages.append({"role": "user", "content": str(step_output.state)})
        return step_output, messages
    if operation == CONTEXT_OPERATION_PRESERVE:
        return step_output, messages
    if operation == CONTEXT_OPERATION_RETRY_CONTROL:
        if prepared.control_request is None:
            raise ValueError("retry_control requires a prepared control request")
        if len(prepared.messages) < 2:
            raise ValueError("retry_control has no pre-control policy context")
        control_message = prepared.messages[-1]
        if control_message != {
            "role": "user",
            "content": prepared.control_request,
        }:
            raise ValueError("prepared control request is not the final message")
        return step_output, [
            dict(message) for message in prepared.messages[:-1]
        ]
    if operation == CONTEXT_OPERATION_REPLACE:
        if not replacement:
            raise ValueError("replace_messages receipt has no messages")
        return step_output, replacement
    raise ValueError(f"wrapper returned an unsupported context operation: {operation!r}")


def _normalize_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[PolicyMessage]:
    normalized: list[PolicyMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return normalized
