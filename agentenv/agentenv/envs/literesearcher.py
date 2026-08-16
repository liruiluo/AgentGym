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


LITERESEARCHER_CONTEXT_COMPACTION_REQUEST = (
    "The research conversation is nearing its context limit. Write the "
    "continuation state you want to retain after the earlier interaction is "
    "removed. Your response will be preserved verbatim and will not call a "
    "research or workspace tool. Include the unresolved question, useful "
    "evidence, and paths to any workspace notes you want to consult later."
)


class LiteResearcherEnvClient(BaseEnvClient):
    """Task-neutral LiteResearcher client with policy-authored compaction."""

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are a deep-research agent working on one continuous "
                    "question with an empty private workspace that persists for "
                    "the episode and survives context compaction. A visit returns one "
                    "bounded page; follow next_page with the same URL and goal when "
                    "more evidence is needed. Preserve useful source URLs, evidence, "
                    "and unresolved work in ordinary workspace files before a long "
                    "context is removed, then inspect those files later.\n\n"
                    "Emit exactly one action per turn with no surrounding prose. A "
                    "complete search response can be exactly:\n"
                    "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":["
                    "\"query one\",\"query two\"]}}</tool_call>\n"
                    "A complete visit response can be exactly:\n"
                    "<tool_call>{\"name\":\"visit\",\"arguments\":{\"url\":"
                    "\"https://source.example/page\",\"goal\":\"specific evidence\","
                    "\"page\":1}}</tool_call>\n"
                    "Tool-call JSON must use ASCII punctuation and double-quoted keys "
                    "and strings. Close every array and object before </tool_call>; a "
                    "search call ends with ]}} immediately before that tag. Never use "
                    "a markdown code fence.\n"
                    "Inspect workspace files with exactly this form:\n"
                    "shell_command {\"command\":\"rg -n . notes.md\",\"workdir\":\".\","
                    "\"timeout_ms\":10000}\n"
                    "Create or update them with apply_patch followed on the next line "
                    "by one *** Begin Patch ... *** End Patch patch. Workspace actions "
                    "have zero task reward. Submit the final answer as "
                    "<answer>...</answer>."
                ),
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
        if metadata.get("domain_id") != "literesearcher":
            raise RuntimeError("LiteResearcher endpoint reports the wrong domain")
        if (
            metadata.get("compaction_contract")
            != "task_neutral_client_replace_messages_v1"
        ):
            raise RuntimeError("LiteResearcher endpoint reports the wrong compaction contract")
        task_count = int(metadata["task_count"])
        if data_len is not None and int(data_len) > task_count:
            raise ValueError(
                f"LiteResearcher data_len {data_len} exceeds task_count {task_count}"
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
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return str(self.info["observation"])

    @property
    def sample_excluded(self) -> bool:
        return bool(self.info.get("info", {}).get("sample_excluded", False))

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = _copy_policy_messages(messages)
        if initial:
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if not self._policy_context_bound:
            return None
        return LITERESEARCHER_CONTEXT_COMPACTION_REQUEST

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        self._selected_policy_control = None
        if not self._policy_context_bound:
            return None
        if pressure is None:
            raise RuntimeError(
                "LiteResearcher compaction requires task-neutral token pressure"
            )
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "LiteResearcher context reached the prompt cap before a trainable "
                "compaction could be sampled"
            )
        request_tokens = (
            pressure.candidate_prompt_tokens - pressure.action_prompt_tokens
        )
        if request_tokens <= 0:
            raise RuntimeError("LiteResearcher compaction request must extend the prompt")
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
        return LITERESEARCHER_CONTEXT_COMPACTION_REQUEST

    def step(self, action: str) -> StepOutput:
        if self._selected_policy_control == "context_compaction":
            return self._complete_context_compaction(action)
        return self._step_native_policy_action(action)

    def _step_native_policy_action(self, action: str) -> StepOutput:
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        response = self._request(
            "POST",
            "step",
            json={"id": self.env_id, "action": action},
        )
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        response_info = response.get("info", {})
        action_submission = response_info.get("action_submission")
        if not isinstance(action_submission, Mapping):
            action_submission = {"raw_policy_output": action}
        return StepOutput(
            state=str(response["observation"]),
            reward=float(response["reward"]),
            done=bool(response["done"]),
            info=build_task_neutral_transition_info(
                env_info=response_info,
                action_submission=action_submission,
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "native_action",
                    "server_wrapper_evidence": response_info.get(
                        "wrapper_evidence", {}
                    ),
                },
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        framing = self._immutable_policy_context
        if framing is None:
            raise RuntimeError("LiteResearcher compaction lost its task framing")
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
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
                    "raw_policy_output": str(action),
                    "submitted_action": None,
                    "parser_status": "policy_context_compaction",
                },
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE,
                    messages=replacement,
                ),
                wrapper_evidence={
                    "event": "context_compaction",
                    "workspace_continuity_id": self.env_id,
                    "native_environment_call_count": 0,
                },
            ),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._request(
            "POST", "reset", json={"id": self.env_id, "data_idx": idx}
        )
        self.info = response
        self._reset_policy_transition_state()
        return response

    def finalize_policy_horizon(self) -> StepOutput:
        return StepOutput(
            state="LiteResearcher policy-turn budget exhausted without an accepted answer.",
            reward=0.0,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={
                    **dict(self.info.get("info", {})),
                    "status": "max_policy_steps_exhausted",
                    "episode_success": False,
                    "sample_excluded": False,
                },
                action_submission={"control_action": "horizon"},
                native_step_before=self._native_call_count,
                native_step_after=self._native_call_count,
                native_call_count_before=self._native_call_count,
                native_call_count_after=self._native_call_count,
                context_epoch_before=self._context_epoch,
                context_epoch_after=self._context_epoch,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=self._policy_step_count,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "horizon_finalization",
                    "outcome": "max_rounds",
                },
            ),
        )

    def close(self) -> bool:
        value = self._request_json("POST", "close", json={"id": self.env_id})
        if value is not True:
            raise requests.RequestException(
                "LiteResearcher POST /close did not return true"
            )
        return True

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        value = self._request_json(method, path, **kwargs)
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"LiteResearcher {method} /{path} returned a non-object response"
            )
        return value

    def _request_json(self, method: str, path: str, **kwargs) -> Any:
        response = requests.request(
            method,
            f"{self.env_server_base}/{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"LiteResearcher {method} /{path} failed: "
                f"status={response.status_code} body={response.text[-1000:]}"
            )
        return response.json()


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


class LiteResearcherTask(BaseTask):
    env_client_cls = LiteResearcherEnvClient
    env_name = "LiteResearcher"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
