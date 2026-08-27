from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    CONTEXT_OPERATION_RETRY_CONTROL,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from .filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER,
    FILESYSTEM_CHECKPOINT_MAX_BYTES,
    FILESYSTEM_CHECKPOINT_PATH,
    FILESYSTEM_CHECKPOINT_REQUEST,
    checkpoint_bounded_retry_trigger_tokens,
    filesystem_checkpoint_failure_reason,
    filesystem_checkpoint_write_succeeded,
    normalize_filesystem_checkpoint_receipt,
)


SWE_CONTEXT_COMPACTION_REQUEST = (
    FILESYSTEM_CHECKPOINT_REQUEST
    + " For this coding task, preserve the issue objective, decisive inspection "
    "and test evidence, changed source paths, unresolved failure, and the next "
    "concrete edit or test."
)
SWE_POLICY_CONTINUATION_MARKER = FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER


ACTOR_CREDIT_SCHEMA = "task_neutral_actor_credit_v1"
ACTION_PROGRESS_SCHEMA = "swesmith_action_progress_v1"
SWE_MEMORY_CONTRACT = "policy_filesystem_checkpoint_then_client_replace_v2"
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
    "and test until the issue is fixed. Your response channel is an action parser, "
    "not a chat channel. Reason silently. Every policy turn is exactly one action, "
    "starting at byte zero. Never prefix an action with narration such as 'Let me' "
    "or 'I found'.\n\n"
    "# Exact tool syntax\n"
    "For inspection, editing, or tests, output one line in exactly this shape:\n"
    'shell_command {"command":"find . -maxdepth 2 -type f | head -80",'
    '"workdir":".","timeout_ms":120000}\n'
    "Replace the command value with the command you need. The command field is required; "
    "workdir and timeout_ms are optional. workdir is relative to /testbed; use `.` for "
    "the repository root, never `/testbed` or `./testbed`.\n"
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
    "Before context compaction, keep detailed evidence that may be needed later in "
    "ordinary task files. At an explicit context-boundary request, use one normal "
    "shell_command or apply_patch action to overwrite "
    "`.agent_memory/CONTINUATION.md` with a short working-state snapshot and locators "
    "for those longer-lived files. That action is executed normally and consumes the "
    "same policy-action budget. Old messages are removed only after the exact write is "
    "verified. After replacement, read the checkpoint with a normal command, then read "
    "any detailed notes it points to before acting on them. The checkpoint does not "
    "replace source files, test artifacts, or voluntary debugging notes.\n"
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
    "inspection or edit. Never submit a plain-text final response. After at least one "
    "non-generated source path has changed and the relevant tests have run, submit with "
    "exactly `shell_command {\"command\":\"echo "
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\",\"workdir\":\".\"}`. "
    "Prose before or after a tool action is a parser error and nothing runs. This "
    "workspace intentionally has no .git directory."
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
        invalid_action_reward: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            isinstance(invalid_action_reward, bool)
            or not isinstance(invalid_action_reward, (int, float))
            or not math.isfinite(float(invalid_action_reward))
            or float(invalid_action_reward) > 0.0
        ):
            raise ValueError(
                "SWE-smith invalid_action_reward must be finite and non-positive"
            )
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        self.invalid_action_reward = float(invalid_action_reward)
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
        self._checkpoint_retry_pending = False
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
        # Decide from the no-control append path, whose base is the preserved
        # Continuous Token runtime.  The freshly rendered control candidate may
        # legitimately be shorter after generation-only history normalization,
        # so using it here can miss an imminent overflow on the ordinary path.
        if (
            not self._checkpoint_retry_pending
            and checkpoint_bounded_retry_trigger_tokens(pressure) < capacity
        ):
            return None
        self._selected_policy_control = "context_compaction"
        return SWE_CONTEXT_COMPACTION_REQUEST

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return str(self.info["observation"])

    @property
    def sample_excluded(self) -> bool:
        info = self.info.get("info", {})
        if not isinstance(info, Mapping):
            return False
        value = info.get("sample_excluded", False)
        if type(value) is not bool:
            raise RuntimeError("SWE-smith sample_excluded must be boolean")
        return value

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
        if actor_credit["basis"] in {"shell_executed", "terminal_submission"}:
            action_progress = _validate_action_progress_receipt(
                response_env_info.get("action_progress")
            )
            if actor_credit["basis"] == "shell_executed":
                actor_credit = self._classify_shell_actor_credit(
                    actor_credit,
                    action_progress,
                )
        elif response_env_info.get("action_progress") is not None:
            raise RuntimeError(
                "SWE-smith action progress appeared without shell execution"
            )
        if actor_credit["basis"] == "workspace_changed":
            self._zero_progress_shell_receipts.clear()
        native_reward = float(response["reward"])
        sample_excluded = response_env_info.get("sample_excluded", False)
        if type(sample_excluded) is not bool:
            raise RuntimeError("SWE-smith sample_excluded must be boolean")
        if sample_excluded and (
            not bool(response.get("done")) or native_reward != 0.0
        ):
            raise RuntimeError(
                "SWE-smith excluded samples must be terminal with zero reward"
            )
        reward = native_reward
        reward_overlay = None
        if (
            actor_credit["basis"] in {"parser_rejected", "executor_rejected"}
            and self.invalid_action_reward != 0.0
        ):
            reward = native_reward + self.invalid_action_reward
            reward_overlay = {
                "schema": "swesmith_invalid_action_reward_v1",
                "basis": actor_credit["basis"],
                "native_reward": native_reward,
                "penalty": self.invalid_action_reward,
                "final_reward": reward,
            }
        after_step = response_env_info.get("step")
        if after_step is not None and int(after_step) != self._native_call_count:
            raise RuntimeError(
                "SWE-smith native step counter drifted from wrapper dispatches"
            )
        wrapper_evidence = {
            "event": "native_action",
            "workspace_continuity_id": self.env_id,
            "actor_credit": actor_credit,
            "action_progress": action_progress,
        }
        if reward_overlay is not None:
            wrapper_evidence["reward_overlay"] = reward_overlay
        return StepOutput(
            state=str(response["observation"]),
            reward=reward,
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
                wrapper_evidence=wrapper_evidence,
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        # The checkpoint is an ordinary sampled policy action. Execute it first;
        # only an attested, bounded write to the exact continuation path may
        # authorize the wrapper-owned context replacement.
        native_output = self._step_native_policy_action(action)
        self._selected_policy_control = None
        info = dict(native_output.info)
        env_info = info.get("env_info", {})
        receipt_value = (
            env_info.get("filesystem_checkpoint")
            if isinstance(env_info, Mapping)
            else None
        )
        checkpoint_receipt = normalize_filesystem_checkpoint_receipt(receipt_value)
        persisted = filesystem_checkpoint_write_succeeded(checkpoint_receipt)
        self._checkpoint_retry_pending = bool(not persisted and not native_output.done)

        context_transition = None
        if persisted and not native_output.done:
            framing = self._immutable_policy_context
            if framing is None:
                raise RuntimeError("SWE-smith compaction lost its immutable task framing")
            replacement = deepcopy(framing)
            # The successor prompt contains only immutable task framing plus a
            # fixed locator. The sampled write and native observation remain in
            # the rollout ledger, never as a shortcut around the later file read.
            replacement.append(
                {"role": "user", "content": SWE_POLICY_CONTINUATION_MARKER}
            )
            self._context_epoch += 1
            self._zero_progress_shell_receipts.clear()
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=replacement,
            )
        elif not native_output.done:
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_RETRY_CONTROL
            )

        native_wrapper = info.get("wrapper_evidence", {})
        actor_credit = (
            native_wrapper.get("actor_credit")
            if isinstance(native_wrapper, Mapping)
            else None
        )
        return StepOutput(
            state=native_output.state,
            reward=native_output.reward,
            done=native_output.done,
            info=build_task_neutral_transition_info(
                env_info=env_info if isinstance(env_info, Mapping) else {},
                action_submission=info.get(
                    "action_submission", {"raw_policy_output": action}
                ),
                native_step_before=info.get("native_step_before"),
                native_step_after=info.get("native_step_after"),
                native_call_count_before=info.get("native_call_count_before"),
                native_call_count_after=info.get("native_call_count_after"),
                context_epoch_before=info.get("context_epoch_before"),
                context_epoch_after=self._context_epoch,
                session_epoch_before=info.get("session_epoch_before"),
                session_epoch_after=info.get("session_epoch_after"),
                policy_step_before=info.get("policy_step_before"),
                policy_step_after=info.get("policy_step_after"),
                context_transition=context_transition,
                wrapper_evidence={
                    "event": "context_compaction",
                    "workspace_continuity_id": self.env_id,
                    "native_environment_call_count": 1,
                    "actor_credit": actor_credit,
                    "continuation_path": FILESYSTEM_CHECKPOINT_PATH,
                    "continuation_max_bytes": FILESYSTEM_CHECKPOINT_MAX_BYTES,
                    "continuation_persisted": persisted,
                    "checkpoint_receipt": checkpoint_receipt,
                    "checkpoint_failure_reason": (
                        filesystem_checkpoint_failure_reason(checkpoint_receipt)
                    ),
                    "context_replaced": bool(persisted and not native_output.done),
                    "retry_pending": self._checkpoint_retry_pending,
                    "retry_context_restored": bool(
                        not persisted and not native_output.done
                    ),
                    "sampled_policy_output_preserved_in_ledger": True,
                    "native_observation_preserved_in_ledger": True,
                    "replacement_contains_policy_output": False,
                    "replacement_contains_native_observation": False,
                    "native_wrapper_evidence": (
                        dict(native_wrapper)
                        if isinstance(native_wrapper, Mapping)
                        else {}
                    ),
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
