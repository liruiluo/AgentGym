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
    checkpoint_retry_trigger_tokens,
    filesystem_checkpoint_failure_reason,
    filesystem_checkpoint_write_succeeded,
    normalize_filesystem_checkpoint_receipt,
)


SWE_CONTEXT_COMPACTION_REQUEST = (
    FILESYSTEM_CHECKPOINT_REQUEST
    + " The reserved `.agent_memory` parent directory already exists; write "
    "the fixed file directly without creating, removing, or replacing that "
    "directory."
    + " For this coding task, preserve the issue objective, decisive inspection "
    "and test evidence, changed source paths, unresolved failure, and the next "
    "concrete edit or test."
)
SWE_POLICY_CONTINUATION_MARKER = FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER


ACTOR_CREDIT_SCHEMA = "task_neutral_actor_credit_v1"
ACTION_PROGRESS_SCHEMA = "swesmith_action_progress_v1"
SWE_MEMORY_CONTRACT = "policy_filesystem_checkpoint_then_client_replace_v3"
SWE_HORIZON_CONTRACT = "unified_policy_step_terminal_failure_minus0p01_v3"
SWE_CHECKPOINT_MAX_ATTEMPTS = 2
SWE_CHECKPOINT_MAX_CYCLES_PER_CONTEXT = 2
SWE_CHECKPOINT_MIN_POST_READ_TASK_TURNS = 4
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
    "Use shell_command for inspection, editing, or tests through a delimiter-light "
    "action in this exact shape:\n"
    "shell_command\n"
    "find . -maxdepth 2 -type f | head -80\n"
    "The first line is the literal shell_command header. Everything after its first "
    "newline is the raw shell command executed from /testbed. There is no JSON, XML, "
    "workdir field, timeout field, or closing delimiter. To work below the repository "
    "root, use a relative cd inside the command.\n"
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
    "state that may outlive the visible conversation: hypotheses, commands or tests "
    "already tried, exact observations, failed approaches and why they failed, "
    "successful partial results, and unresolved next checks. Choose a safe relative "
    "path and organization; no filename or note format is prescribed, and a short task "
    "may not need notes at all. Keep notes separate from source code when practical.\n"
    "Before context compaction, keep detailed evidence that may be needed later in "
    "ordinary task files. At an explicit context-boundary request, use one normal "
    "shell_command or apply_patch action to overwrite "
    "`.agent_memory/CONTINUATION.md` with a short working-state snapshot and locators "
    "for those longer-lived files. That action is executed normally and consumes the "
    "same policy-action budget. Old messages are removed only after the exact write is "
    "verified. After replacement, read the checkpoint with a normal command, then read "
    "any detailed notes it points to before acting on them. The checkpoint does not "
    "replace source files, test artifacts, or voluntary debugging notes.\n"
    "Illustrative pattern only (do not copy its content as a task answer): one action "
    "can place `mkdir -p .agent_memory && printf '%s\n' 'hypothesis: parser state is "
    "stale' 'evidence: test output ...' >> .agent_memory/debugging.md` after the "
    "shell_command header; a later shell action can run `rg -n "
    "'hypothesis|evidence|next check' .agent_memory`. This is only a syntax "
    "illustration: do not assume this filename, content, or timing is useful for the "
    "current issue. Writing or reading a note has no separate reward; the native task "
    "result is the objective.\n\n"
    "# Output contract\n"
    "Start at byte zero with the line shell_command or apply_patch. Output only that "
    "one action: no JSON wrapper, XML tags, explanation, label, Markdown fence, or "
    "<think> tag. After an observation, emit the next action directly; do not describe "
    "what you plan to do. A shell command can edit the persistent workspace. Do not "
    "repeat a successful inspection or edit. Never submit a plain-text final response. "
    "After at least one non-generated source path has changed and the relevant tests "
    "have run, submit with exactly:\n"
    "shell_command\n"
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
    "Prose before an action is a parser error and nothing runs. This workspace "
    "intentionally has no .git directory."
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
        checkpoint_contract_penalty: float = 0.0,
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
        if (
            isinstance(checkpoint_contract_penalty, bool)
            or not isinstance(checkpoint_contract_penalty, (int, float))
            or not math.isfinite(float(checkpoint_contract_penalty))
            or float(checkpoint_contract_penalty) > 0.0
        ):
            raise ValueError(
                "SWE-smith checkpoint_contract_penalty must be finite and "
                "non-positive"
            )
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        self.invalid_action_reward = float(invalid_action_reward)
        self.checkpoint_contract_penalty = float(checkpoint_contract_penalty)
        metadata = self._request("GET", "metadata")
        memory_contract = metadata.get("memory_contract")
        if memory_contract != SWE_MEMORY_CONTRACT:
            raise RuntimeError(
                "SWE-smith endpoint memory contract mismatch: "
                f"expected {SWE_MEMORY_CONTRACT!r}, got {memory_contract!r}"
            )
        horizon_contract = metadata.get("horizon_contract")
        if horizon_contract != SWE_HORIZON_CONTRACT:
            raise RuntimeError(
                "SWE-smith endpoint horizon contract mismatch: "
                f"expected {SWE_HORIZON_CONTRACT!r}, got {horizon_contract!r}"
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
        self._checkpoint_attempt_count = 0
        self._checkpoint_retry_exhausted = False
        self._checkpoint_cycle_index = 0
        self._checkpoint_cycle_attempt_limit = SWE_CHECKPOINT_MAX_ATTEMPTS
        self._checkpoint_total_attempt_count = 0
        self._checkpoint_ordinary_turn_required = False
        self._checkpoint_capacity_terminal_after_ordinary = False
        self._checkpoint_capacity_terminal_reason: str | None = None
        self._selected_checkpoint_terminal_on_failure = False
        self._selected_checkpoint_terminal_on_executed_failure = False
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

    def _configured_max_policy_turns(self) -> int:
        value = self.metadata.get(
            "configured_max_policy_turns", self.metadata.get("max_steps")
        )
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                "SWE-smith metadata is missing a positive policy-turn limit"
            )
        return value

    def _remaining_policy_turns(self) -> int:
        return max(0, self._configured_max_policy_turns() - self._policy_step_count)

    def _checkpoint_turns_required_before_attempt(self) -> int:
        attempts_left = (
            self._checkpoint_cycle_attempt_limit - self._checkpoint_attempt_count
        )
        return (
            attempts_left
            + 1  # one explicit checkpoint read after replacement
            + SWE_CHECKPOINT_MIN_POST_READ_TASK_TURNS
        )

    def _checkpoint_action_budget_available(self) -> bool:
        return (
            not self._checkpoint_retry_exhausted
            and not self._checkpoint_ordinary_turn_required
            and self._checkpoint_attempt_count
            < self._checkpoint_cycle_attempt_limit
            and self._remaining_policy_turns()
            >= self._checkpoint_turns_required_before_attempt()
        )

    def _fresh_checkpoint_cycle_turns_required(
        self, *, attempt_limit: int = SWE_CHECKPOINT_MAX_ATTEMPTS
    ) -> int:
        return (
            attempt_limit
            + 1  # one explicit checkpoint read after replacement
            + SWE_CHECKPOINT_MIN_POST_READ_TASK_TURNS
        )

    def _planned_fresh_checkpoint_attempt_limit(self) -> int:
        remaining = self._remaining_policy_turns()
        if remaining >= self._fresh_checkpoint_cycle_turns_required():
            return SWE_CHECKPOINT_MAX_ATTEMPTS
        if remaining >= self._fresh_checkpoint_cycle_turns_required(
            attempt_limit=1
        ):
            return 1
        return 0

    def _start_checkpoint_cycle(self, *, attempt_limit: int) -> None:
        if attempt_limit not in {1, SWE_CHECKPOINT_MAX_ATTEMPTS}:
            raise RuntimeError("SWE-smith checkpoint attempt limit is invalid")
        if self._checkpoint_ordinary_turn_required:
            raise RuntimeError(
                "SWE-smith checkpoint cycle rearmed before an ordinary task turn"
            )
        if self._checkpoint_cycle_index >= SWE_CHECKPOINT_MAX_CYCLES_PER_CONTEXT:
            raise RuntimeError(
                "SWE-smith checkpoint cycle exceeded its per-context bound"
            )
        self._checkpoint_cycle_index += 1
        self._checkpoint_cycle_attempt_limit = attempt_limit
        self._checkpoint_attempt_count = 0
        self._checkpoint_retry_exhausted = False
        self._checkpoint_retry_pending = False

    def _checkpoint_request(
        self,
        *,
        attempt: int | None = None,
        attempt_limit: int | None = None,
    ) -> str:
        if attempt is None:
            attempt = self._checkpoint_attempt_count + 1
        if attempt_limit is None:
            attempt_limit = self._checkpoint_cycle_attempt_limit
        remaining_after_success = self._remaining_policy_turns() - 1
        return (
            SWE_CONTEXT_COMPACTION_REQUEST
            + f" This is checkpoint attempt {attempt}/{attempt_limit}."
            + f" If it succeeds, exactly {remaining_after_success} policy actions "
            "will remain, beginning with the required checkpoint read."
        )

    def policy_turn_candidate(self) -> str | None:
        if not self._policy_context_bound or self._remaining_policy_turns() <= 0:
            return None
        if self._checkpoint_retry_pending:
            return self._checkpoint_request()
        attempt_limit = self._planned_fresh_checkpoint_attempt_limit() or 1
        return self._checkpoint_request(
            attempt=1, attempt_limit=attempt_limit
        )

    def _arm_capacity_terminal_after_ordinary(self, reason: str) -> None:
        self._checkpoint_capacity_terminal_after_ordinary = True
        self._checkpoint_capacity_terminal_reason = reason

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        self._selected_policy_control = None
        self._selected_checkpoint_terminal_on_failure = False
        self._selected_checkpoint_terminal_on_executed_failure = False
        self._checkpoint_capacity_terminal_after_ordinary = False
        self._checkpoint_capacity_terminal_reason = None
        candidate = self.policy_turn_candidate()
        if candidate is None:
            return None
        if pressure is None:
            raise RuntimeError(
                "SWE-smith context compaction requires task-neutral token pressure"
            )
        capacity = pressure.effective_prompt_capacity
        if pressure.action_prompt_tokens > capacity:
            raise RuntimeError(
                "SWE-smith context reached the prompt cap before a trainable "
                "compaction could be sampled"
            )
        if pressure.candidate_prompt_tokens > capacity:
            self._arm_capacity_terminal_after_ordinary(
                "checkpoint_request_does_not_fit"
            )
            return None

        one_growth = checkpoint_bounded_retry_trigger_tokens(pressure)
        two_growth = checkpoint_retry_trigger_tokens(pressure)

        if self._checkpoint_retry_pending:
            self._selected_policy_control = "context_compaction"
            self._selected_checkpoint_terminal_on_executed_failure = bool(
                one_growth > capacity
            )
            return candidate

        if self._checkpoint_ordinary_turn_required:
            if pressure.projected_next_prompt_tokens_without_control > capacity:
                self._arm_capacity_terminal_after_ordinary(
                    "ordinary_progress_would_exceed_prompt_capacity"
                )
            return None

        if self._checkpoint_cycle_index >= SWE_CHECKPOINT_MAX_CYCLES_PER_CONTEXT:
            if pressure.projected_next_prompt_tokens_without_control > capacity:
                self._arm_capacity_terminal_after_ordinary(
                    "checkpoint_cycle_limit_exhausted"
                )
            return None

        attempt_limit = self._planned_fresh_checkpoint_attempt_limit()
        if attempt_limit == 0:
            if pressure.projected_next_prompt_tokens_without_control <= capacity:
                return None
            # A checkpoint write is useful only if at least one later sampled
            # action can read it after replacement. On the final policy turn,
            # preserve the normal task/submission action instead and let the
            # native horizon close that real transition.
            if self._remaining_policy_turns() <= 1:
                self._arm_capacity_terminal_after_ordinary(
                    "checkpoint_read_turn_unavailable"
                )
                return None
            self._start_checkpoint_cycle(attempt_limit=1)
            self._selected_policy_control = "context_compaction"
            self._selected_checkpoint_terminal_on_failure = True
            return candidate

        if two_growth < capacity:
            return None

        self._start_checkpoint_cycle(attempt_limit=attempt_limit)
        self._selected_policy_control = "context_compaction"
        if attempt_limit == 1:
            self._selected_checkpoint_terminal_on_failure = True
        elif two_growth > capacity:
            self._selected_checkpoint_terminal_on_executed_failure = True
        return candidate

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
        capacity_terminal = self._checkpoint_capacity_terminal_after_ordinary
        capacity_reason = self._checkpoint_capacity_terminal_reason
        self._checkpoint_capacity_terminal_after_ordinary = False
        self._checkpoint_capacity_terminal_reason = None
        output = self._step_native_policy_action(action)
        if output.done:
            return output
        if capacity_terminal:
            return self._finalize_checkpoint_capacity(
                output,
                reason=capacity_reason or "checkpoint_capacity_exhausted",
            )
        if self._checkpoint_ordinary_turn_required:
            info = output.info if isinstance(output.info, Mapping) else {}
            native_env_info = info.get("env_info", {})
            native_actor_credit = (
                native_env_info.get("actor_credit", {})
                if isinstance(native_env_info, Mapping)
                else {}
            )
            if not isinstance(native_actor_credit, Mapping) or not (
                native_actor_credit.get("basis")
            ):
                wrapper_evidence = info.get("wrapper_evidence", {})
                native_actor_credit = (
                    wrapper_evidence.get("actor_credit", {})
                    if isinstance(wrapper_evidence, Mapping)
                    else {}
                )
            executed_ordinary_action = bool(
                isinstance(native_actor_credit, Mapping)
                and native_actor_credit.get("basis")
                in {"shell_executed", "workspace_changed", "no_workspace_change"}
            )
            if executed_ordinary_action:
                self._checkpoint_ordinary_turn_required = False
                self._checkpoint_retry_exhausted = False
                self._checkpoint_retry_pending = False
                self._checkpoint_attempt_count = 0
                self._checkpoint_cycle_attempt_limit = SWE_CHECKPOINT_MAX_ATTEMPTS
        return output

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

    def _finalize_checkpoint_capacity(
        self,
        native_output: StepOutput,
        *,
        reason: str,
        wrapper_evidence: Mapping[str, Any] | None = None,
    ) -> StepOutput:
        """Close an unsafe continuation through the native failure endpoint.

        The sampled action and its native receipt stay in the PPO ledger. The
        attested horizon contract supplies a terminal failure without grading;
        no advantage or return is edited by the wrapper.
        """

        response = self._request("POST", "horizon", json={"id": self.env_id})
        self.info = response
        if not bool(response.get("done")):
            raise RuntimeError(
                "SWE-smith checkpoint-capacity finalization was not terminal"
            )
        horizon_info = response.get("info", {})
        if not isinstance(horizon_info, Mapping):
            raise RuntimeError(
                "SWE-smith checkpoint-capacity finalization info is invalid"
            )
        sample_excluded = horizon_info.get("sample_excluded", False)
        if type(sample_excluded) is not bool:
            raise RuntimeError(
                "SWE-smith checkpoint-capacity sample_excluded must be boolean"
            )
        horizon_reward = float(response["reward"])
        # This terminal receipt belongs to the sampled action that immediately
        # preceded it. Preserve any negative reward already assigned to that
        # action, then apply the endpoint's natural horizon failure. In
        # particular, do not charge the checkpoint-contract penalty to an
        # ordinary task action merely because no further prompt would fit.
        native_action_reward = float(native_output.reward)
        reward = min(native_action_reward, horizon_reward)

        native_info = dict(native_output.info)
        native_wrapper = native_info.get("wrapper_evidence", {})
        # The horizon endpoint is a wrapper-owned terminal transition for the
        # immediately preceding sampled action. Preserve that action's standard
        # top-level credit/progress receipts instead of hiding them only under
        # native_wrapper_evidence. A richer control wrapper may override those
        # fields, while the capacity-terminal fields below remain authoritative.
        evidence = (
            dict(native_wrapper) if isinstance(native_wrapper, Mapping) else {}
        )
        evidence.update(dict(wrapper_evidence or {}))
        evidence.update(
            {
                "event": "checkpoint_capacity_termination",
                "workspace_continuity_id": self.env_id,
                "capacity_termination_reason": reason,
                "checkpoint_cycle_index": self._checkpoint_cycle_index,
                "checkpoint_max_cycles_per_context": (
                    SWE_CHECKPOINT_MAX_CYCLES_PER_CONTEXT
                ),
                "checkpoint_total_attempt_count": (
                    self._checkpoint_total_attempt_count
                ),
                "native_action_reward": float(native_output.reward),
                "native_action_done": bool(native_output.done),
                "native_action_state": str(native_output.state),
                "native_wrapper_evidence": (
                    dict(native_wrapper)
                    if isinstance(native_wrapper, Mapping)
                    else {}
                ),
                "horizon_reward": horizon_reward,
                "final_reward": reward,
                "reward_source": "native_horizon_failure_transition",
                "advantage_modified": False,
            }
        )
        if reward != horizon_reward:
            evidence["native_action_reward_preserved"] = True
        self._checkpoint_capacity_terminal_after_ordinary = False
        self._checkpoint_capacity_terminal_reason = None
        self._checkpoint_retry_pending = False
        self._checkpoint_retry_exhausted = True
        self._checkpoint_ordinary_turn_required = False
        self._selected_checkpoint_terminal_on_failure = False
        self._selected_checkpoint_terminal_on_executed_failure = False
        return StepOutput(
            state=str(response["observation"]),
            reward=reward,
            done=True,
            info=build_task_neutral_transition_info(
                env_info=dict(horizon_info),
                action_submission=native_info.get(
                    "action_submission", {"raw_policy_output": None}
                ),
                native_step_before=native_info.get("native_step_before"),
                native_step_after=native_info.get("native_step_after"),
                native_call_count_before=native_info.get(
                    "native_call_count_before"
                ),
                native_call_count_after=native_info.get(
                    "native_call_count_after"
                ),
                context_epoch_before=native_info.get("context_epoch_before"),
                context_epoch_after=self._context_epoch,
                session_epoch_before=native_info.get("session_epoch_before"),
                session_epoch_after=self._session_epoch,
                policy_step_before=native_info.get("policy_step_before"),
                policy_step_after=native_info.get("policy_step_after"),
                wrapper_evidence=evidence,
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        # The checkpoint is an ordinary sampled policy action. Execute it first;
        # only an attested, bounded write to the exact continuation path may
        # authorize the wrapper-owned context replacement.
        terminal_on_failure = self._selected_checkpoint_terminal_on_failure
        terminal_on_executed_failure = (
            self._selected_checkpoint_terminal_on_executed_failure
        )
        native_output = self._step_native_policy_action(action)
        self._selected_policy_control = None
        self._selected_checkpoint_terminal_on_failure = False
        self._selected_checkpoint_terminal_on_executed_failure = False
        info = dict(native_output.info)
        env_info = info.get("env_info", {})
        receipt_value = (
            env_info.get("filesystem_checkpoint")
            if isinstance(env_info, Mapping)
            else None
        )
        checkpoint_receipt = normalize_filesystem_checkpoint_receipt(receipt_value)
        persisted = filesystem_checkpoint_write_succeeded(checkpoint_receipt)
        checkpoint_attempt_number = self._checkpoint_attempt_count + 1
        self._checkpoint_attempt_count = checkpoint_attempt_number
        self._checkpoint_total_attempt_count += 1
        if self._checkpoint_cycle_index == 0:
            # Direct test/debug dispatches may bypass prepare_policy_turn. Keep
            # their receipts well-formed without weakening production guards.
            self._checkpoint_cycle_index = 1
        checkpoint_cycle_index = self._checkpoint_cycle_index
        checkpoint_attempt_limit = self._checkpoint_cycle_attempt_limit

        native_wrapper = info.get("wrapper_evidence", {})
        actor_credit = (
            native_wrapper.get("actor_credit")
            if isinstance(native_wrapper, Mapping)
            else None
        )
        safe_retry_context_restore = bool(
            not persisted
            and not native_output.done
            and isinstance(actor_credit, Mapping)
            and actor_credit.get("basis") in {"parser_rejected", "executor_rejected"}
            and checkpoint_receipt is not None
            and not checkpoint_receipt["action_completed"]
            and not checkpoint_receipt["changed"]
        )

        context_transition = None
        retry_feedback_preserved = False
        retry_context_restored = False
        capacity_termination_reason = None
        if persisted and not native_output.done:
            framing = self._immutable_policy_context
            if framing is None:
                raise RuntimeError("SWE-smith compaction lost its immutable task framing")
            remaining_turns = self._remaining_policy_turns()
            replacement = deepcopy(framing)
            # The successor prompt contains only immutable task framing plus a
            # fixed locator and an exact action-budget receipt. The sampled write
            # and native observation remain in the rollout ledger, never as a
            # shortcut around the later file read.
            replacement.append(
                {
                    "role": "user",
                    "content": (
                        SWE_POLICY_CONTINUATION_MARKER
                        + f" Exactly {remaining_turns} policy actions remain in this "
                        "episode. Read the checkpoint first, then execute its saved "
                        "next concrete action; do not restart broad repository "
                        "inspection unless the checkpoint identifies missing evidence."
                    ),
                }
            )
            self._context_epoch += 1
            self._zero_progress_shell_receipts.clear()
            self._checkpoint_retry_pending = False
            self._checkpoint_attempt_count = 0
            self._checkpoint_retry_exhausted = False
            self._checkpoint_cycle_index = 0
            self._checkpoint_cycle_attempt_limit = SWE_CHECKPOINT_MAX_ATTEMPTS
            self._checkpoint_ordinary_turn_required = False
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=replacement,
            )
        elif not native_output.done:
            if terminal_on_failure:
                capacity_termination_reason = (
                    "checkpoint_failure_would_exceed_prompt_capacity"
                )
            elif terminal_on_executed_failure and not safe_retry_context_restore:
                capacity_termination_reason = (
                    "checkpoint_failure_would_exceed_prompt_capacity"
                )
            else:
                # A rejected, unexecuted control action has no state transition
                # to preserve. Restore the exact pre-control task context so its
                # malformed output cannot consume the remaining prompt budget.
                # Executed actions append feedback because arbitrary workspace
                # side effects cannot be rolled back safely.
                if safe_retry_context_restore:
                    retry_context_restored = True
                    context_transition = build_task_neutral_context_transition(
                        CONTEXT_OPERATION_RETRY_CONTROL
                    )
                else:
                    retry_feedback_preserved = True
                self._checkpoint_retry_pending = bool(
                    self._checkpoint_attempt_count < checkpoint_attempt_limit
                    and self._remaining_policy_turns()
                    >= self._checkpoint_turns_required_before_attempt()
                )
                self._checkpoint_retry_exhausted = not (
                    self._checkpoint_retry_pending
                )
                self._checkpoint_ordinary_turn_required = (
                    self._checkpoint_retry_exhausted
                )
        else:
            self._checkpoint_retry_pending = False
            self._checkpoint_retry_exhausted = True
            self._checkpoint_ordinary_turn_required = False

        native_sample_excluded = (
            env_info.get("sample_excluded", False)
            if isinstance(env_info, Mapping)
            else False
        )
        if type(native_sample_excluded) is not bool:
            raise RuntimeError("SWE-smith checkpoint sample_excluded must be boolean")
        native_episode_success = (
            env_info.get("episode_success", False)
            if isinstance(env_info, Mapping)
            else False
        )
        if type(native_episode_success) is not bool:
            raise RuntimeError("SWE-smith checkpoint episode_success must be boolean")
        native_step = env_info.get("step") if isinstance(env_info, Mapping) else None
        native_max_steps_terminal = bool(
            not persisted
            and native_output.done
            and not native_sample_excluded
            and not native_episode_success
            and isinstance(actor_credit, Mapping)
            and actor_credit.get("basis") != "terminal_submission"
            and isinstance(native_step, int)
            and not isinstance(native_step, bool)
            and native_step >= self._configured_max_policy_turns()
        )

        reward = float(native_output.reward)
        checkpoint_reward_overlay = None
        if (
            not persisted
            and (not native_output.done or native_max_steps_terminal)
            and self.checkpoint_contract_penalty != 0.0
        ):
            # The contract violation is an ordinary environment transition.
            # Apply the configured negative reward as a ceiling so an existing
            # parser/executor penalty for the same action is never added twice.
            # A checkpoint failure coincident with the native max-step terminal
            # is still the same failed sampled action; successful submissions
            # and infrastructure-excluded samples retain their native reward.
            reward_before = reward
            reward = min(reward_before, self.checkpoint_contract_penalty)
            applied_delta = reward - reward_before
            checkpoint_reward_overlay = {
                "schema": "swesmith_checkpoint_contract_reward_v1",
                "basis": "checkpoint_contract_unsatisfied",
                "reward_before": reward_before,
                "configured_penalty": self.checkpoint_contract_penalty,
                "applied_delta": applied_delta,
                "final_reward": reward,
                "deduplicated": applied_delta == 0.0,
            }
        wrapper_evidence = {
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
            "checkpoint_attempt_count": checkpoint_attempt_number,
            "checkpoint_cycle_attempt_limit": checkpoint_attempt_limit,
            "checkpoint_max_attempts": SWE_CHECKPOINT_MAX_ATTEMPTS,
            "checkpoint_cycle_index": checkpoint_cycle_index,
            "checkpoint_max_cycles_per_context": (
                SWE_CHECKPOINT_MAX_CYCLES_PER_CONTEXT
            ),
            "checkpoint_total_attempt_count": self._checkpoint_total_attempt_count,
            "remaining_policy_turns": self._remaining_policy_turns(),
            "retry_pending": self._checkpoint_retry_pending,
            "retry_exhausted": self._checkpoint_retry_exhausted,
            "ordinary_turn_required": self._checkpoint_ordinary_turn_required,
            "retry_context_restored": retry_context_restored,
            "retry_feedback_preserved": retry_feedback_preserved,
            "terminal_on_failure": terminal_on_failure,
            "terminal_on_executed_failure": terminal_on_executed_failure,
            "native_max_steps_terminal": native_max_steps_terminal,
            "sampled_policy_output_preserved_in_ledger": True,
            "native_observation_preserved_in_ledger": True,
            "replacement_contains_policy_output": False,
            "replacement_contains_native_observation": False,
            "native_wrapper_evidence": (
                dict(native_wrapper) if isinstance(native_wrapper, Mapping) else {}
            ),
        }
        if checkpoint_reward_overlay is not None:
            wrapper_evidence["reward_overlay"] = checkpoint_reward_overlay
        if capacity_termination_reason is not None:
            wrapper_evidence[
                "capacity_termination_reason"
            ] = capacity_termination_reason
            return self._finalize_checkpoint_capacity(
                StepOutput(
                    state=native_output.state,
                    reward=reward,
                    done=False,
                    info=build_task_neutral_transition_info(
                        env_info=(
                            env_info if isinstance(env_info, Mapping) else {}
                        ),
                        action_submission=info.get(
                            "action_submission", {"raw_policy_output": action}
                        ),
                        native_step_before=info.get("native_step_before"),
                        native_step_after=info.get("native_step_after"),
                        native_call_count_before=info.get(
                            "native_call_count_before"
                        ),
                        native_call_count_after=info.get(
                            "native_call_count_after"
                        ),
                        context_epoch_before=info.get("context_epoch_before"),
                        context_epoch_after=self._context_epoch,
                        session_epoch_before=info.get("session_epoch_before"),
                        session_epoch_after=info.get("session_epoch_after"),
                        policy_step_before=info.get("policy_step_before"),
                        policy_step_after=info.get("policy_step_after"),
                        wrapper_evidence=wrapper_evidence,
                    ),
                ),
                reason=capacity_termination_reason,
                wrapper_evidence=wrapper_evidence,
            )
        return StepOutput(
            state=native_output.state,
            reward=reward,
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
                wrapper_evidence=wrapper_evidence,
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

    def finalize_policy_prompt_capacity(
        self, pressure: PolicyContextPressure
    ) -> StepOutput:
        """Terminate an already-over-cap prompt through the native horizon path."""

        capacity = pressure.effective_prompt_capacity
        if pressure.action_prompt_tokens <= capacity:
            raise ValueError(
                "SWE-smith prompt-capacity finalization requires an over-cap prompt"
            )
        output = self.finalize_horizon()
        info = dict(output.info)
        wrapper_evidence = info.get("wrapper_evidence", {})
        evidence = (
            dict(wrapper_evidence)
            if isinstance(wrapper_evidence, Mapping)
            else {}
        )
        evidence.update(
            {
                "event": "prompt_capacity_finalization",
                "workspace_continuity_id": self.env_id,
                "capacity_termination_reason": "action_prompt_exceeds_capacity",
                "action_prompt_tokens": pressure.action_prompt_tokens,
                "effective_prompt_capacity": capacity,
                "grader_called": False,
                "advantage_modified": False,
            }
        )
        info["wrapper_evidence"] = evidence
        return StepOutput(
            state=output.state,
            reward=output.reward,
            done=output.done,
            info=info,
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
