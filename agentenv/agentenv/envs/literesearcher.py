from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)


# The route-level forecast must cover the largest policy-visible observation
# seen with the frozen LiteResearcher service and Qwen3.5 tokenizer: the
# maximum r43 next-prompt growth was 10,652 tokens, comprising a 52-token
# response plus a 10,600-token observation-and-template delta. Keep a bounded
# margin so compaction is sampled before the
# next native action can push the prompt past the 30,720-token PPO width.
LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE = 12_288
LITERESEARCHER_CONTINUATION_PATH = ".agent_memory/CONTINUATION.md"
LITERESEARCHER_CONTINUATION_MAX_BYTES = 8192
LITERESEARCHER_RESEARCH_NOTE_PATH = ".agent_memory/research.md"
LITERESEARCHER_WORKSPACE_ACTION_ENVELOPE_CONTRACT = (
    "literesearcher_tool_call_workspace_v1"
)


def _render_workspace_tool_call(command: str) -> str:
    return "<tool_call>" + json.dumps(
        {
            "name": "shell_command",
            "arguments": {
                "command": command,
                "workdir": ".",
                "timeout_ms": 10000,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "</tool_call>"


LITERESEARCHER_RESEARCH_NOTE_WRITE_EXAMPLE = _render_workspace_tool_call(
    "mkdir -p .agent_memory && cat > "
    ".agent_memory/research.md <<'AMG_RESEARCH'\n"
    "question: ...\n"
    "evidence_with_urls: ...\n"
    "failed_attempts: ...\n"
    "next_step: ...\n"
    "AMG_RESEARCH\n"
)
LITERESEARCHER_RESEARCH_NOTE_READ_ACTION = _render_workspace_tool_call(
    f"cat {LITERESEARCHER_RESEARCH_NOTE_PATH}"
)
LITERESEARCHER_COMPACTION_CONTRACT = "task_neutral_filesystem_checkpoint_v2"
LITERESEARCHER_MIN_ACTIONS_FOR_CHECKPOINT_READ_ANSWER = 3
_RECEIPT_SCHEMA = "agentmemory_continuation_checkpoint_v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORKSPACE_ACTION_MARKER_RE = re.compile(
    r"(?m)^(shell_command(?= \{)|apply_patch(?=\r?$))"
)
_MAX_WORKSPACE_REASONING_PREFIX_CHARS = 2048


LITERESEARCHER_CONTEXT_COMPACTION_EXAMPLE = _render_workspace_tool_call(
    "mkdir -p .agent_memory && cat > "
    ".agent_memory/CONTINUATION.md <<'AMG_CHECKPOINT'\n"
    "question: ...\n"
    "evidence: ...\n"
    "failed_attempts: ...\n"
    "other_files: ...\n"
    "next_action: ...\n"
    "AMG_CHECKPOINT\n"
)


LITERESEARCHER_CONTEXT_COMPACTION_REQUEST = (
    "CHECKPOINT WRITE PHASE (before context reset). The research conversation "
    "is nearing its context limit. Emit exactly one normal executable shell_command "
    "action that overwrites "
    f"{LITERESEARCHER_CONTINUATION_PATH} with a concise continuation checkpoint. "
    "The resulting file must be nonempty and at most "
    f"{LITERESEARCHER_CONTINUATION_MAX_BYTES} bytes. Preserve the unresolved "
    "question, evidence with exact source URLs, failed attempts, other useful file "
    "paths, and the next action. Keep the mkdir -p .agent_memory && prefix so the "
    "write also works in an otherwise empty workspace. This is a write-only phase. "
    f"Do not read {LITERESEARCHER_CONTINUATION_PATH} or .agent_memory/research.md; "
    "do not Search, Visit, or answer. Your response must be one complete "
    "<tool_call> JSON envelope whose name is shell_command. Do not emit raw "
    "shell_command syntax, Markdown backticks, a code fence, or prose. Copy the "
    "complete executable action below, replacing only the checkpoint field "
    "values:\n"
    f"{LITERESEARCHER_CONTEXT_COMPACTION_EXAMPLE}"
)


LITERESEARCHER_CONTINUATION_READ_ACTION = _render_workspace_tool_call(
    f"cat {LITERESEARCHER_CONTINUATION_PATH}"
)


LITERESEARCHER_POLICY_CONTINUATION_MARKER = (
    "CHECKPOINT READ PHASE (after context reset). The earlier interaction was "
    "removed after a verified policy-authored checkpoint was written to "
    f"{LITERESEARCHER_CONTINUATION_PATH}. Before any search, visit, or answer, "
    "read it with one normal shell_command inside the same <tool_call> JSON "
    "envelope used by Search and Visit. Do not emit raw shell_command syntax, "
    "Markdown backticks, a code fence, or prose. "
    "Continue from the file output instead of reconstructing omitted history. "
    "The complete executable action is the final line below:\n"
    f"{LITERESEARCHER_CONTINUATION_READ_ACTION}"
)



def _checkpoint_write_retry_request(rejection_reason: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", rejection_reason) is None:
        rejection_reason = "checkpoint_rejected"
    return (
        "CHECKPOINT WRITE RETRY (before context reset). The previous checkpoint "
        f"action was rejected with reason {rejection_reason}. Correct that "
        "specific error; the rejected action itself was removed. "
        f"{LITERESEARCHER_CONTEXT_COMPACTION_REQUEST}"
    )


class LiteResearcherEnvClient(BaseEnvClient):
    """Task-neutral LiteResearcher client with policy-authored compaction."""

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are a deep-research agent answering one continuous question. "
                    "Your empty private workspace persists for the episode. Search and "
                    "Visit, and shell_command all use one <tool_call> JSON envelope; "
                    "use exactly one action per turn. Literal forms: "
                    "<tool_call>{\"name\":\"search\","
                    "\"arguments\":{\"query\":[\"climate policy\"]}}</tool_call> "
                    "or <tool_call>{\"name\":\"visit\",\"arguments\":{"
                    "\"url\":\"https://literesearcher.local/page/00001\","
                    "\"goal\":\"extract evidence\",\"page\":1}}</tool_call>. "
                    "Keep both closing braces. A Visit returns one bounded page; follow "
                    "next_page with the same URL and goal when needed. A workspace "
                    "turn uses name shell_command and the arguments object "
                    "{command, workdir, timeout_ms} inside that same envelope, with no "
                    "prose, Markdown fence, or second tool. "
                    f"{LITERESEARCHER_RESEARCH_NOTE_PATH} is optional. Write or refresh "
                    "it only when evidence with source URLs, failed attempts, or a plan "
                    "must survive several later actions or a future context checkpoint. "
                    "Do not write it after every useful Visit. If the evidence already "
                    "supports the final answer, answer directly instead of staging that "
                    "answer in a file. To create or replace the note, copy this action "
                    "and replace only the field values:\n"
                    f"{LITERESEARCHER_RESEARCH_NOTE_WRITE_EXAMPLE}\n"
                    "To read it, emit only:\n"
                    f"{LITERESEARCHER_RESEARCH_NOTE_READ_ACTION}\n"
                    "The workspace shell is networkless; use Search and Visit for "
                    "external evidence. Submit the final answer as <answer>...</answer>. "
                    "At an explicit context-checkpoint request, follow its separate "
                    f"executable write to {LITERESEARCHER_CONTINUATION_PATH} exactly. "
                    "That required checkpoint is distinct from the optional research "
                    "note and is not free-form continuation text."
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
                "LiteResearcher invalid_action_reward must be finite and non-positive"
            )
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        self.invalid_action_reward = float(invalid_action_reward)
        metadata = self._request("GET", "metadata")
        if metadata.get("domain_id") != "literesearcher":
            raise RuntimeError("LiteResearcher endpoint reports the wrong domain")
        if metadata.get("compaction_contract") != LITERESEARCHER_COMPACTION_CONTRACT:
            raise RuntimeError("LiteResearcher endpoint reports the wrong compaction contract")
        if metadata.get("continuation_checkpoint_receipt_schema") != _RECEIPT_SCHEMA:
            raise RuntimeError("LiteResearcher endpoint reports the wrong checkpoint receipt schema")
        if (
            metadata.get("workspace_action_envelope_contract")
            != LITERESEARCHER_WORKSPACE_ACTION_ENVELOPE_CONTRACT
        ):
            raise RuntimeError(
                "LiteResearcher endpoint reports the wrong workspace action envelope"
            )
        if metadata.get("workspace_action_envelope_tools") != ["shell_command"]:
            raise RuntimeError(
                "LiteResearcher endpoint reports the wrong enveloped workspace tools"
            )
        if metadata.get("raw_workspace_action_compatibility") is not True:
            raise RuntimeError(
                "LiteResearcher endpoint removed raw workspace action compatibility"
            )
        workspace_memory_reward = metadata.get("workspace_memory_reward")
        if (
            isinstance(workspace_memory_reward, bool)
            or not isinstance(workspace_memory_reward, (int, float))
            or float(workspace_memory_reward) != 0.0
        ):
            raise RuntimeError("LiteResearcher endpoint changes workspace memory reward")
        if metadata.get("compaction_calls_endpoint_step") is not True:
            raise RuntimeError("LiteResearcher endpoint does not charge checkpoint writes")
        if metadata.get("compaction_calls_research_backend") is not False:
            raise RuntimeError("LiteResearcher checkpoint unexpectedly calls research backend")
        if metadata.get("continuation_checkpoint_path") != LITERESEARCHER_CONTINUATION_PATH:
            raise RuntimeError("LiteResearcher endpoint reports the wrong checkpoint path")
        if (
            metadata.get("continuation_checkpoint_max_bytes")
            != LITERESEARCHER_CONTINUATION_MAX_BYTES
        ):
            raise RuntimeError("LiteResearcher endpoint reports the wrong checkpoint limit")
        max_policy_steps = metadata.get("max_policy_steps")
        if type(max_policy_steps) is not int or max_policy_steps <= 0:
            raise RuntimeError("LiteResearcher endpoint reports invalid max_policy_steps")
        self.max_policy_steps = max_policy_steps
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
        self._checkpoint_retry_context: list[dict[str, str]] | None = None
        self._checkpoint_retry_reason: str | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return str(self.info["observation"])

    @property
    def sample_excluded(self) -> bool:
        return bool(self.info.get("info", {}).get("sample_excluded", False))

    def policy_framing(self) -> list[dict[str, str]]:
        """Expose the exact immutable prompt used by this wrapper."""

        return [
            {
                "role": "user" if message["from"] == "human" else "assistant",
                "content": str(message["value"]),
            }
            for message in self.conversation_start
        ]

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

    def _checkpoint_control_request(self) -> str:
        if self._checkpoint_retry_reason is None:
            return LITERESEARCHER_CONTEXT_COMPACTION_REQUEST
        return _checkpoint_write_retry_request(self._checkpoint_retry_reason)

    def policy_turn_candidate(self) -> str | None:
        if not self._policy_context_bound:
            return None
        return self._checkpoint_control_request()

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
        if (
            pressure.max_observation_tokens
            < LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE
        ):
            raise RuntimeError(
                "LiteResearcher route observation-token envelope is too small: "
                f"configured={pressure.max_observation_tokens} "
                f"minimum={LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE}"
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
        # A rejected checkpoint remains pending independently of a fresh token-
        # pressure estimate.  Re-rendering the restored message list can change
        # its token count slightly; recomputing the threshold here could
        # otherwise dispatch a normal research action before the checkpoint is
        # completed.
        if self._checkpoint_retry_context is not None:
            if self._current_policy_context != self._checkpoint_retry_context:
                raise RuntimeError(
                    "LiteResearcher checkpoint retry context changed before retry"
                )
            self._selected_policy_control = "context_compaction"
            return self._checkpoint_control_request()
        # Decide from the no-control append path.  Continuous Token chat
        # normalization may make the rendered control candidate shorter than the
        # ordinary action prompt, so candidate-minus-action is not a valid size
        # or safety invariant.
        if pressure.projected_next_prompt_tokens_without_control < capacity:
            return None
        remaining_actions = self.max_policy_steps - self._policy_step_count
        if remaining_actions < LITERESEARCHER_MIN_ACTIONS_FOR_CHECKPOINT_READ_ANSWER:
            # A forced write with fewer than write/read/answer actions remaining
            # creates an unavoidable checkpoint dead end. The current prompt is
            # still within capacity, so leave the final actions to the policy.
            return None
        if self._current_policy_context is None:
            raise RuntimeError(
                "LiteResearcher compaction lost its pre-attempt policy context"
            )
        self._checkpoint_retry_context = deepcopy(self._current_policy_context)
        self._checkpoint_retry_reason = None
        self._selected_policy_control = "context_compaction"
        return self._checkpoint_control_request()

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
        if not isinstance(response_info, Mapping):
            raise RuntimeError("LiteResearcher endpoint returned non-object info")
        action_submission = response_info.get("action_submission")
        if not isinstance(action_submission, Mapping):
            action_submission = {"raw_policy_output": action}
        server_wrapper_evidence = response_info.get("wrapper_evidence", {})
        if not isinstance(server_wrapper_evidence, Mapping):
            server_wrapper_evidence = {}
        native_reward = float(response["reward"])
        reward = native_reward
        reward_overlay = None
        invalid_action = (
            response_info.get("status") == "invalid_action"
            or server_wrapper_evidence.get("invalid_action") is True
        )
        if invalid_action:
            reward, reward_overlay = self._invalid_action_reward_overlay(
                native_reward=native_reward,
                done=bool(response["done"]),
                sample_excluded=bool(response_info.get("sample_excluded", False)),
            )
        wrapper_evidence = {
            "event": "native_action",
            "server_wrapper_evidence": dict(server_wrapper_evidence),
        }
        if reward_overlay is not None:
            wrapper_evidence["reward_overlay"] = reward_overlay
        return StepOutput(
            state=str(response["observation"]),
            reward=reward,
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
                wrapper_evidence=wrapper_evidence,
            ),
        )

    def _invalid_action_reward_overlay(
        self,
        *,
        native_reward: float,
        done: bool,
        sample_excluded: bool,
    ) -> tuple[float, dict[str, Any] | None]:
        if self.invalid_action_reward == 0.0:
            return native_reward, None
        if sample_excluded:
            raise RuntimeError(
                "LiteResearcher invalid action cannot also be sample_excluded"
            )
        if native_reward != 0.0:
            raise RuntimeError(
                "LiteResearcher invalid-action overlay requires zero native reward"
            )
        reward = native_reward + self.invalid_action_reward
        return reward, {
            "schema": "literesearcher_invalid_action_reward_v1",
            "native_reward": native_reward,
            "penalty": self.invalid_action_reward,
            "total_reward": reward,
            "terminal": done,
        }

    def _checkpoint_rejection_lifecycle(
        self,
    ) -> tuple[bool, dict[str, Any] | None, bool, int]:
        remaining_actions = self.max_policy_steps - self._policy_step_count
        if remaining_actions < LITERESEARCHER_MIN_ACTIONS_FOR_CHECKPOINT_READ_ANSWER:
            self._checkpoint_retry_context = None
            self._checkpoint_retry_reason = None
            return True, None, False, remaining_actions
        retry_context = self._checkpoint_retry_context
        if retry_context is None:
            raise RuntimeError(
                "LiteResearcher checkpoint retry lost its pre-attempt research context"
            )
        return (
            False,
            build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=deepcopy(retry_context),
            ),
            True,
            remaining_actions,
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        framing = self._immutable_policy_context
        if framing is None:
            raise RuntimeError("LiteResearcher compaction lost its task framing")
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        if not _is_workspace_action_candidate(action):
            self._policy_step_count += 1
            self._selected_policy_control = None
            (
                done,
                context_transition,
                retry_context_restored,
                remaining_actions,
            ) = self._checkpoint_rejection_lifecycle()
            self._checkpoint_retry_reason = (
                None if done else "workspace_action_required"
            )
            state = (
                "Continuation checkpoint was not accepted (workspace action required). "
                "Use exactly one <tool_call> JSON envelope whose name is "
                "shell_command to overwrite "
                f"{LITERESEARCHER_CONTINUATION_PATH}; keep mkdir -p .agent_memory && "
                "and add no search, visit, answer, Markdown backtick, code fence, "
                "raw shell_command, or standalone prose. "
            )
            if done:
                state += (
                    "The episode ended because too few policy actions remain for a "
                    "checkpoint write, required read, and answer."
                )
            else:
                state += (
                    "The failed attempt was removed from the policy context; the "
                    "pre-attempt research context will be retried unchanged."
                )
            reward, reward_overlay = self._invalid_action_reward_overlay(
                native_reward=0.0,
                done=done,
                sample_excluded=False,
            )
            env_info = dict(self.info.get("info", {}))
            event = "forced_checkpoint_rejected"
            if done:
                event = "forced_checkpoint_retry_budget_exhausted"
                env_info.update(
                    {
                        "status": "checkpoint_retry_budget_exhausted",
                        "episode_success": False,
                        "sample_excluded": False,
                    }
                )
            wrapper_evidence = {
                "event": event,
                "workspace_continuity_id": self.env_id,
                "checkpoint_rejection_reason": "workspace_action_required",
                "endpoint_step_dispatched": False,
                "native_environment_call_count": 0,
                "retry_context_restored": retry_context_restored,
                "checkpoint_retry_remaining_actions": remaining_actions,
            }
            if reward_overlay is not None:
                wrapper_evidence["reward_overlay"] = reward_overlay
            return StepOutput(
                state=state,
                reward=reward,
                done=done,
                info=build_task_neutral_transition_info(
                    env_info=env_info,
                    action_submission={
                        "raw_policy_output": str(action),
                        "submitted_action": None,
                        "parser_status": (
                            "forced_checkpoint_requires_workspace_action"
                        ),
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
                    context_transition=context_transition,
                    wrapper_evidence=wrapper_evidence,
                ),
            )

        response = self._request(
            "POST",
            "step",
            json={"id": self.env_id, "action": action},
        )
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        self._selected_policy_control = None

        response_info = response.get("info", {})
        if not isinstance(response_info, Mapping):
            raise RuntimeError("LiteResearcher endpoint returned non-object info")
        response_info = dict(response_info)
        action_submission = response_info.get("action_submission")
        if not isinstance(action_submission, Mapping):
            action_submission = {"raw_policy_output": action}
        server_evidence = response_info.get("wrapper_evidence")
        if not isinstance(server_evidence, Mapping):
            server_evidence = {}
        receipt = server_evidence.get("continuation_checkpoint")
        checkpoint_valid, rejection_reason = _validate_checkpoint_receipt(receipt)
        if checkpoint_valid and (
            action_submission.get("kind") != "workspace"
            or action_submission.get("op") != receipt.get("action_kind")
        ):
            checkpoint_valid = False
            rejection_reason = "checkpoint_action_submission_mismatch"
        if (
            isinstance(receipt, Mapping)
            and receipt.get("valid") is True
            and server_evidence.get("native_environment_call_count") != 0
        ):
            raise RuntimeError(
                "LiteResearcher checkpoint workspace action called the research backend"
            )
        checkpoint_reward = response.get("reward")
        if (
            isinstance(checkpoint_reward, bool)
            or not isinstance(checkpoint_reward, (int, float))
            or float(checkpoint_reward) != 0.0
        ):
            raise RuntimeError("LiteResearcher checkpoint workspace action changed reward")
        done = bool(response["done"])
        sample_excluded = bool(response_info.get("sample_excluded", False))
        if sample_excluded and not done:
            raise RuntimeError(
                "LiteResearcher excluded checkpoint sample must be terminal"
            )
        context_transition = None
        retry_context_restored = False
        remaining_actions = self.max_policy_steps - self._policy_step_count
        state = str(response["observation"])
        event = "forced_checkpoint_rejected"
        if checkpoint_valid and not done:
            self._checkpoint_retry_context = None
            self._checkpoint_retry_reason = None
            replacement = deepcopy(framing)
            replacement.append(
                {
                    "role": "user",
                    "content": LITERESEARCHER_POLICY_CONTINUATION_MARKER,
                }
            )
            self._context_epoch += 1
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=replacement,
            )
            event = "forced_checkpoint_write"
        elif done:
            self._checkpoint_retry_context = None
            self._checkpoint_retry_reason = None
            event = "forced_checkpoint_terminal"
        else:
            (
                done,
                context_transition,
                retry_context_restored,
                remaining_actions,
            ) = self._checkpoint_rejection_lifecycle()
            self._checkpoint_retry_reason = (
                None if done else str(rejection_reason or "checkpoint_rejected")
            )
            state = (
                "Continuation checkpoint was not accepted "
                f"({rejection_reason}). Use exactly one <tool_call> JSON envelope "
                "whose name is shell_command to "
                f"overwrite {LITERESEARCHER_CONTINUATION_PATH} with "
                f"1-{LITERESEARCHER_CONTINUATION_MAX_BYTES} bytes; keep "
                "mkdir -p .agent_memory && and add no raw shell_command, Markdown "
                "backtick, or standalone prose. "
            )
            if done:
                event = "forced_checkpoint_retry_budget_exhausted"
                state += (
                    "The episode ended because too few policy actions remain for a "
                    "checkpoint write, required read, and answer."
                )
                response_info.update(
                    {
                        "status": "checkpoint_retry_budget_exhausted",
                        "episode_success": False,
                        "sample_excluded": False,
                    }
                )
            else:
                state += (
                    "The failed attempt was removed from the policy context; the "
                    "pre-attempt research context will be retried unchanged."
                )

        reward = float(checkpoint_reward)
        reward_overlay = None
        if not checkpoint_valid and not sample_excluded:
            reward, reward_overlay = self._invalid_action_reward_overlay(
                native_reward=reward,
                done=done,
                sample_excluded=False,
            )
        wrapper_evidence = {
            "event": event,
            "workspace_continuity_id": self.env_id,
            "continuation_checkpoint": (
                dict(receipt) if isinstance(receipt, Mapping) else None
            ),
            "checkpoint_rejection_reason": rejection_reason,
            "server_wrapper_evidence": dict(server_evidence),
            "endpoint_step_dispatched": True,
            "native_environment_call_count": 0,
            "retry_context_restored": retry_context_restored,
            "checkpoint_retry_remaining_actions": remaining_actions,
        }
        if reward_overlay is not None:
            wrapper_evidence["reward_overlay"] = reward_overlay
        return StepOutput(
            state=state,
            reward=reward,
            done=done,
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
                context_transition=context_transition,
                wrapper_evidence=wrapper_evidence,
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
        # Horizon finalization is terminal. Clear pending control state so a
        # controller audit cannot mistake a rejected checkpoint for a live retry.
        self._checkpoint_retry_context = None
        self._checkpoint_retry_reason = None
        self._selected_policy_control = None
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


def _is_workspace_action_candidate(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("shell_command") or value.startswith("apply_patch\n"):
        return True
    match = _WORKSPACE_ACTION_MARKER_RE.search(value)
    if match is not None:
        prefix = value[: match.start(1)]
        return (
            "```" not in prefix
            and len(prefix) <= _MAX_WORKSPACE_REASONING_PREFIX_CHARS
        )
    envelope = re.fullmatch(
        r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*",
        value,
        re.IGNORECASE | re.DOTALL,
    )
    if envelope is None:
        return False
    try:
        payload = json.loads(envelope.group(1))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, Mapping) or set(payload) != {"name", "arguments"}:
        return False
    arguments = payload.get("arguments")
    return (
        payload.get("name") == "shell_command"
        and isinstance(arguments, Mapping)
        and isinstance(arguments.get("command"), str)
    )


def _validate_checkpoint_receipt(value: Any) -> tuple[bool, str | None]:
    if not isinstance(value, Mapping):
        return False, "missing_receipt"
    if value.get("schema") != _RECEIPT_SCHEMA:
        return False, "invalid_receipt_schema"
    if value.get("path") != LITERESEARCHER_CONTINUATION_PATH:
        return False, "wrong_checkpoint_path"
    if value.get("valid") is not True:
        reason = value.get("rejection_reason")
        return False, reason if isinstance(reason, str) and reason else "invalid_checkpoint"
    if value.get("action_kind") not in {"SHELL_COMMAND", "APPLY_PATCH"}:
        return False, "wrong_checkpoint_action_kind"
    if value.get("action_execution_succeeded") is not True:
        return False, "checkpoint_action_failed"
    change_kind = value.get("change_kind")
    before_digest = value.get("before_sha256")
    size = value.get("bytes")
    digest = value.get("sha256")
    if change_kind not in {"added", "modified"}:
        return False, "invalid_change_kind"
    if change_kind == "added":
        before_valid = before_digest is None
    else:
        before_valid = (
            isinstance(before_digest, str)
            and _SHA256_RE.fullmatch(before_digest) is not None
            and before_digest != digest
        )
    if (
        value.get("changed_in_action") is not True
        or value.get("content_changed") is not True
        or value.get("nonempty") is not True
        or value.get("within_size_limit") is not True
        or type(size) is not int
        or not 1 <= size <= LITERESEARCHER_CONTINUATION_MAX_BYTES
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or not before_valid
        or value.get("rejection_reason") is not None
    ):
        return False, "inconsistent_valid_receipt"
    return True, None


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
