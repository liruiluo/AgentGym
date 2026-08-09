from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    ConversationMessage,
    POLICY_CONTINUATION_MARKER,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)


SWE_CONTEXT_COMPACTION_REQUEST = (
    "The conversation is nearing its context limit. Write the continuation "
    "state you want to retain after the earlier interaction is removed. Your "
    "response will be preserved verbatim and will not be sent to the "
    "environment. Include only information you choose to carry forward."
)

SWE_POLICY_SYSTEM_PROMPT = (
    "You are a coding agent in one persistent /testbed repository. Inspect, edit, "
    "and test until the issue is fixed. Think privately. Every policy turn is exactly "
    "one action.\n\n"
    "# Exact tool syntax\n"
    "For inspection, editing, or tests, output one line in exactly this shape:\n"
    'shell_command {"command":"find . -maxdepth 2 -type f | head -80",'
    '"workdir":".","timeout_ms":120000}\n'
    "Replace the command value with the command you need. The command field is required; "
    "workdir and timeout_ms are optional.\n"
    "For a patch, output exactly this shape:\n"
    "apply_patch\n"
    "*** Begin Patch\n"
    "*** Update File: relative/path.py\n"
    "@@\n"
    "-old text\n"
    "+new text\n"
    "*** End Patch\n"
    "Replace the path and hunk with text from the file you inspected. apply_patch is "
    "optional; use shell_command when an exact patch is uncertain.\n\n"
    "# Output contract\n"
    "Start at byte zero with shell_command or apply_patch. Output only that one action: "
    "no XML tags, explanation, label, Markdown fence, or <think> tag. After an "
    "observation, emit the next action directly; do not describe what you plan to do. "
    "A shell command can edit the persistent workspace. Do not repeat a successful "
    "inspection or edit. Do not submit plain text until at least one source path has "
    "changed and the relevant tests have run; then a plain final response may summarize "
    "the result. Prose before or after a tool action is a parser error and nothing runs. "
    "This workspace intentionally has no .git directory."
)


class SwesmithEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": SWE_POLICY_SYSTEM_PROMPT,
            }
        ),
        ConversationMessage(
            {"from": "gpt", "loss": False, "value": "Understood."}
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int | None = None,
        *args,
        timeout: int = 900,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        metadata = self._request("GET", "metadata")
        task_count = int(metadata["task_count"])
        if data_len is not None and int(data_len) > task_count:
            raise ValueError(
                f"SWE-smith data_len {data_len} exceeds server task_count {task_count}"
            )
        self.data_len = task_count if data_len is None else int(data_len)
        created = self._request("POST", "create", json={})
        self.env_id = int(created["id"])
        self.info = created
        self.metadata = metadata
        self._reset_policy_transition_state()

    def _reset_policy_transition_state(self) -> None:
        self._policy_step_count = 0
        self._native_call_count = 0
        self._context_epoch = 0
        self._session_epoch = 0
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": SWE_POLICY_SYSTEM_PROMPT}]

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = _copy_policy_messages(messages)
        observation = str(self.observe())
        if (
            not normalized
            or normalized[-1]["role"] != "user"
            or normalized[-1]["content"] != observation
        ):
            raise ValueError(
                "SWE-smith initial policy context must end with the current observation"
            )
        return self.policy_framing() + [
            {"role": "user", "content": observation}
        ]

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = _copy_policy_messages(messages)
        if initial:
            expected = self.policy_framing() + [
                {"role": "user", "content": str(self.observe())}
            ]
            if normalized != expected:
                raise ValueError(
                    "SWE-smith initial policy context differs from its system framing"
                )
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if not self._policy_context_bound:
            return None
        return SWE_CONTEXT_COMPACTION_REQUEST

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        self._selected_policy_control = None
        if not self._policy_context_bound:
            return None
        if pressure is None:
            raise RuntimeError(
                "SWE-smith context compaction requires task-neutral token pressure"
            )
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "SWE-smith context reached the prompt cap before a trainable "
                "compaction could be sampled"
            )
        request_tokens = (
            pressure.candidate_prompt_tokens - pressure.action_prompt_tokens
        )
        if request_tokens <= 0:
            raise RuntimeError(
                "SWE-smith compaction request must extend the action prompt"
            )
        projected_next_request = (
            pressure.action_prompt_tokens
            + pressure.max_response_tokens
            + pressure.max_observation_tokens
            + pressure.action_observation_envelope_tokens
            + request_tokens
        )
        if projected_next_request < capacity:
            return None
        self._selected_policy_control = "context_compaction"
        return SWE_CONTEXT_COMPACTION_REQUEST

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return str(self.info["observation"])

    def step(self, action: str) -> StepOutput:
        if self._selected_policy_control == "context_compaction":
            return self._complete_context_compaction(action)
        return self._step_native_policy_action(action)

    def _step_native_policy_action(self, action: str) -> StepOutput:
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        session_before = self._session_epoch
        response = self._request(
            "POST",
            "step",
            json={"id": self.env_id, "action": action},
        )
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        response_env_info = response.get("info", {})
        after_step = response_env_info.get("step")
        if after_step is not None and int(after_step) != self._native_call_count:
            raise RuntimeError(
                "SWE-smith native step counter drifted from wrapper dispatches"
            )
        return StepOutput(
            state=str(response["observation"]),
            reward=float(response["reward"]),
            done=bool(response["done"]),
            info=build_task_neutral_transition_info(
                env_info=response_env_info,
                action_submission={"raw_policy_output": action},
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=session_before,
                session_epoch_after=self._session_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "native_action",
                    "workspace_continuity_id": self.env_id,
                },
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        framing = self._immutable_policy_context
        if framing is None:
            raise RuntimeError("SWE-smith compaction lost its immutable task framing")
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        session_before = self._session_epoch
        replacement = deepcopy(framing)
        replacement.extend(
            [
                {"role": "assistant", "content": str(action)},
                {"role": "user", "content": POLICY_CONTINUATION_MARKER},
            ]
        )
        self._policy_step_count += 1
        self._context_epoch += 1
        self._selected_policy_control = None
        return StepOutput(
            state=str(self.info.get("observation", "")),
            reward=0.0,
            done=False,
            info=build_task_neutral_transition_info(
                env_info=self.info.get("info", {}),
                action_submission={
                    "raw_policy_output": action,
                    "submitted_action": None,
                    "parser_status": "policy_context_compaction",
                },
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=session_before,
                session_epoch_after=self._session_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE,
                    messages=replacement,
                ),
                wrapper_evidence={
                    "event": "context_compaction",
                    "workspace_continuity_id": self.env_id,
                },
            ),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._request(
            "POST",
            "reset",
            json={"id": self.env_id, "data_idx": idx},
        )
        self.info = response
        self._reset_policy_transition_state()
        return response

    def detail(self, *, private_token: str | None = None) -> dict[str, Any]:
        headers = (
            {} if private_token is None else {"X-SWESMITH-Detail-Token": private_token}
        )
        return self._request(
            "GET", "detail", params={"id": self.env_id}, headers=headers
        )

    def finalize_horizon(self) -> StepOutput:
        response = self._request("POST", "horizon", json={"id": self.env_id})
        self.info = response
        response_env_info = response.get("info", {})
        return StepOutput(
            state=str(response["observation"]),
            reward=float(response["reward"]),
            done=bool(response["done"]),
            info=build_task_neutral_transition_info(
                env_info=response_env_info,
                action_submission={"control_action": "horizon"},
                native_step_before=self._native_call_count,
                native_step_after=self._native_call_count,
                native_call_count_before=self._native_call_count,
                native_call_count_after=self._native_call_count,
                context_epoch_before=self._context_epoch,
                context_epoch_after=self._context_epoch,
                session_epoch_before=self._session_epoch,
                session_epoch_after=self._session_epoch,
                policy_step_before=self._policy_step_count,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "horizon_finalization",
                    "workspace_continuity_id": self.env_id,
                },
            ),
        )

    def finalize_policy_horizon(self) -> StepOutput:
        """Expose horizon grading through the task-neutral wrapper contract."""

        return self.finalize_horizon()

    def close(self) -> dict[str, Any]:
        return self._request("POST", "close", json={"id": self.env_id})

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.env_server_base}/{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"SWE-smith {method} /{path} failed: "
                f"status={response.status_code} body={response.text[-1000:]}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"SWE-smith {method} /{path} returned a non-object response"
            )
        return value


def _copy_policy_messages(
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
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return normalized


class SwesmithTask(BaseTask):
    env_client_cls = SwesmithEnvClient
    env_name = "SWE-smith"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
