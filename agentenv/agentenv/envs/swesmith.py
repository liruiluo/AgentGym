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
    "environment. Include only information you choose to carry forward. Keep "
    "this response short: retain the immediate objective, decisive current "
    "state, next action, and the path/search key of any durable notes you "
    "already wrote. Do not claim that this response executed a shell command "
    "or changed a file."
)

ACTOR_CREDIT_SCHEMA = "task_neutral_actor_credit_v1"
ACTION_PROGRESS_SCHEMA = "swesmith_action_progress_v1"
SWE_MEMORY_CONTRACT = "policy_compaction_plus_optional_durable_filesystem_v1"
_POSITIVE_ACTOR_CREDIT_BASES = {
    "shell_executed",
    "workspace_changed",
    "terminal_submission",
    "policy_context_compaction",
}
_INELIGIBLE_ACTOR_CREDIT_BASES = {
    "parser_rejected",
    "executor_rejected",
    "no_workspace_change",
    "zero_progress_repeat",
}


def _validate_actor_credit_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("SWE-smith response is missing its actor-credit receipt")
    expected_keys = {"schema", "positive_eligible", "basis"}
    if set(value) != expected_keys:
        raise RuntimeError(
            "SWE-smith actor-credit receipt has unexpected fields: "
            f"{sorted(set(value) - expected_keys)}"
        )
    if value.get("schema") != ACTOR_CREDIT_SCHEMA:
        raise RuntimeError("SWE-smith actor-credit receipt schema drifted")
    positive_eligible = value.get("positive_eligible")
    if type(positive_eligible) is not bool:
        raise RuntimeError(
            "SWE-smith actor-credit positive_eligible must be boolean"
        )
    basis = value.get("basis")
    allowed_bases = (
        _POSITIVE_ACTOR_CREDIT_BASES
        if positive_eligible
        else _INELIGIBLE_ACTOR_CREDIT_BASES
    )
    if basis not in allowed_bases:
        raise RuntimeError(
            "SWE-smith actor-credit basis disagrees with positive_eligible: "
            f"{basis!r}"
        )
    return {
        "schema": ACTOR_CREDIT_SCHEMA,
        "positive_eligible": positive_eligible,
        "basis": str(basis),
    }


def _validate_action_progress_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("SWE-smith shell response is missing action progress")
    expected_keys = {
        "schema",
        "action_fingerprint",
        "result_fingerprint",
        "workspace_changed",
    }
    if set(value) != expected_keys:
        raise RuntimeError(
            "SWE-smith action-progress receipt has unexpected fields: "
            f"{sorted(set(value) - expected_keys)}"
        )
    if value.get("schema") != ACTION_PROGRESS_SCHEMA:
        raise RuntimeError("SWE-smith action-progress receipt schema drifted")
    normalized: dict[str, Any] = {"schema": ACTION_PROGRESS_SCHEMA}
    for key in ("action_fingerprint", "result_fingerprint"):
        fingerprint = value.get(key)
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise RuntimeError(
                f"SWE-smith action-progress {key} must be lowercase SHA256"
            )
        normalized[key] = fingerprint
    workspace_changed = value.get("workspace_changed")
    if type(workspace_changed) is not bool:
        raise RuntimeError(
            "SWE-smith action-progress workspace_changed must be boolean"
        )
    normalized["workspace_changed"] = workspace_changed
    return normalized

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
    "optional; use shell_command when an exact patch is uncertain. The path after "
    "*** Update File must be relative to /testbed, for example src/module.py, never "
    "/testbed/src/module.py. For an existing file, make a bounded local edit after "
    "inspecting the exact target lines. Do not use cat > or a here-document to rewrite "
    "an existing source file. If a patch fails, re-inspect the small target region and "
    "retry a bounded patch or short shell edit; do not replace the whole file.\n\n"
    "# Durable debugging notes\n"
    "This is a long-running debugging task. The persistent workspace can hold ordinary "
    "files that you create and maintain with the same shell_command and apply_patch "
    "actions. For a debugging path that may span context compactions, maintain a "
    "concise evidence ledger incrementally instead of waiting for the compaction "
    "request. Update it at meaningful state changes: a hypothesis is introduced or "
    "ruled out, a command or test changes the evidence, a root cause or partial fix is "
    "verified, a regression is found, or the next branch is chosen. Record high-volume "
    "state that may outlive the visible "
    "conversation: hypotheses, commands or tests already tried, exact observations, "
    "failed approaches and why they failed, successful partial results, and unresolved "
    "next checks. Choose a safe relative path and organization; no filename or note "
    "format is prescribed, and a short task may not need notes at all. Keep notes "
    "separate from source code when practical.\n"
    "Before context compaction, make sure important detailed evidence needed later is "
    "already in an ordinary file. The compaction response is only a short "
    "working-state handoff: include the locator of notes you already wrote, not the "
    "whole debugging history. That response is not executed by the environment. After "
    "compaction, rediscover and read the notes with normal commands such as find, rg, "
    "grep, sed, head, tail, or cat before taking an action that depends on them. The "
    "harness will not create, list, summarize, or restore note contents for you.\n"
    "Illustrative pattern only (do not copy its content as a task answer): one turn may "
    "append a concise entry to a relative debugging file, a later turn may search or "
    "read that file, and a subsequent patch may use the recovered evidence. For example, "
    "a first action could be `shell_command {\"command\":\"mkdir -p .agent_memory && "
    "printf '%s\\n' 'hypothesis: parser state is stale' 'evidence: test output ...' "
    ">> .agent_memory/debugging.md\",\"workdir\":\".\"}`, followed in a later "
    "action by `shell_command {\"command\":\"rg -n 'hypothesis|evidence|next check' "
    ".agent_memory\",\"workdir\":\".\"}`. This is only a syntax illustration: "
    "do not assume this filename, content, or timing is useful for the current issue. "
    "Writing or reading a note has no separate reward; the native task result is the "
    "objective.\n\n"
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
        memory_contract = metadata.get("memory_contract")
        if memory_contract != SWE_MEMORY_CONTRACT:
            raise RuntimeError(
                "SWE-smith endpoint memory contract mismatch: "
                f"expected {SWE_MEMORY_CONTRACT!r}, got {memory_contract!r}"
            )
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
        self._zero_progress_shell_receipts: set[tuple[str, str]] = set()

    def _classify_shell_actor_credit(
        self,
        actor_credit: Mapping[str, Any],
        action_progress: Mapping[str, Any],
    ) -> dict[str, Any]:
        if action_progress["workspace_changed"]:
            self._zero_progress_shell_receipts.clear()
            return dict(actor_credit)
        key = (
            str(action_progress["action_fingerprint"]),
            str(action_progress["result_fingerprint"]),
        )
        if key in self._zero_progress_shell_receipts:
            return {
                "schema": ACTOR_CREDIT_SCHEMA,
                "positive_eligible": False,
                "basis": "zero_progress_repeat",
            }
        self._zero_progress_shell_receipts.add(key)
        return dict(actor_credit)

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
        # Compare the exact candidate prompt itself.  A full chat-template
        # rerender may normalize generation-only history and therefore be shorter
        # than the preserved Continuous Token runtime; token lengths are not
        # required to be monotonic across those two representations.
        projected_next_request = (
            pressure.candidate_prompt_tokens
            + pressure.max_response_tokens
            + pressure.max_observation_tokens
            + pressure.action_observation_envelope_tokens
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
        actor_credit = _validate_actor_credit_receipt(
            response_env_info.get("actor_credit")
            if isinstance(response_env_info, Mapping)
            else None
        )
        action_progress = None
        if actor_credit["basis"] == "shell_executed":
            action_progress = _validate_action_progress_receipt(
                response_env_info.get("action_progress")
            )
            actor_credit = self._classify_shell_actor_credit(
                actor_credit,
                action_progress,
            )
        elif response_env_info.get("action_progress") is not None:
            raise RuntimeError(
                "SWE-smith action progress appeared on a non-executed shell action"
            )
        if actor_credit["basis"] == "workspace_changed":
            self._zero_progress_shell_receipts.clear()
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
                    "actor_credit": actor_credit,
                    "action_progress": action_progress,
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
        self._zero_progress_shell_receipts.clear()
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
                    "actor_credit": {
                        "schema": ACTOR_CREDIT_SCHEMA,
                        "positive_eligible": True,
                        "basis": "policy_context_compaction",
                    },
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
