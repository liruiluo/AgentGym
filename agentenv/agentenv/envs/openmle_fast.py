from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

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

OPENMLE_FAST_POLICY_SYSTEM_PROMPT = """You are solving one OpenMLE-fast task in an isolated /workspace with exactly 30 total policy actions.
This interface is a plain-text action protocol, not a native tool-calling API. Start every response at byte zero with exactly one action. Output no reasoning, explanation, Markdown fence, XML/tool_call tag, native tool wrapper, action-number prefix, or bare JSON before or after it. Put reflection that must survive context replacement into a workspace file through a valid action.

The only valid action forms are:
shell_command {"command":"cat TASK.md","workdir":".","timeout_ms":20000}
apply_patch
*** Begin Patch
*** Add File: script.py
+print("ok")
*** End Patch
submit

For shell_command, emit the literal prefix `shell_command ` followed by one valid JSON object on one line. `command` must be a non-empty JSON string, optional `workdir` must be exactly ".", and optional integer `timeout_ms` must be between 1 and 20000. The command already runs from /workspace; use workspace-relative paths.
For creating or replacing `train.py`, prefer one shell_command with `printf`: `shell_command {"command":"printf '%s\\n' 'print(1)' > train.py","workdir":".","timeout_ms":20000}`. Keep its JSON on one line; do not place a heredoc or raw multiline program inside it. Never use `python -c`, `python3 -c`, a heredoc, or `bash -c`.
For apply_patch, use only workspace-relative paths. Never use `/workspace/` in an apply_patch file path. Use the raw form above with `*** Begin Patch`, `*** Add File:` or `*** Update File:`, and `*** End Patch`; never wrap it in JSON or use `Create File`. Every added file line starts with `+`.
Never emit two actions or prose with an action. A parser error still consumes an action; after one, emit only one corrected valid action.

Dependencies are already installed. No network access is available. Never run `pip`, `pip3`, `conda`, `apt`, `ssh`, `curl`, `wget`, or `chmod`, and do not inspect package versions. For shell work prefer `cat`, `head`, `tail`, `ls`, `grep`, `cut`, `sort`, `wc`, `mkdir`, `printf`, and `python train.py`.

Use the first turns efficiently. A useful first response is exactly this one line:
shell_command {"command":"cat TASK.md; head -3 data/train.csv; head -3 data/test.csv; head -3 data/sample_submission.csv","workdir":".","timeout_ms":20000}
On a later turn, create workspace-relative `train.py` with one shell_command and `printf`; put each Python source line in one shell-quoted `printf` argument and redirect only that bounded command to `train.py`. The script must load public labelled training data, use only public labelled training data to make one deterministic local validation split (a holdout or small cross-validation), print one explicit measured metric line in the form `validation_<metric>=<finite_value>`, fit a bounded candidate with deterministic seeds and `n_jobs=1` where supported, and write workspace-relative `submission.csv` in the required schema. On the following turn, run `python train.py` with one shell_command. Do not spend separate early turns on more row previews, dataset summaries, package checks, or training-set-only metrics.

Work as an iterative ML engineer. Training-set metrics do not count as local validation. The printed validation value must come from executed code, not a placeholder. Keep each run within the managed 15000 ms runtime; do not use broad sweeps, nested parallelism, or large grid searches. After a traceback or measured validation result, modify `train.py` and run it again. The environment exposes no repeatable private score or free validation oracle.

Do not write the continuation note before the first measured validation unless an explicit context-compaction request requires it. At that request, write `.agent_memory/OPENMLE_CONTINUATION.md` exactly once: `shell_command {"command":"mkdir -p .agent_memory && printf '%s\\n' 'objective: ...' 'measured_validation_or_failure: ...' 'conclusion: ...' 'code_path: train.py' 'next_action: ...' > .agent_memory/OPENMLE_CONTINUATION.md","workdir":".","timeout_ms":20000}`. Fill the fields with the last metric or exact failure and a `next_action` that changes `train.py` before rerunning; `python train.py` alone is not enough. When compaction finishes, after a continuation marker, read it exactly once and perform that edit; do not re-inspect the task or repeat the note without new evidence.

Reading, editing, running, writing or reading memory, compaction, and submit all consume the same 30-action budget. Every observation reports completed and remaining actions. TASK.md and data are read-only. When a measured local validation and `submission.csv` exist, iterate only while enough actions remain and submit no later than action 27. `submit` grades against the protected private data exactly once; the first submit is terminal, and there is no automatic submission at the action limit; action 30 ends in failure if it is not submit.
If an observation reports a parser error, respond next with only a corrected action in one of the exact forms above. Never describe the correction.
"""
OPENMLE_FAST_POLICY_PROMPT_SHA256 = hashlib.sha256(
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()
OPENMLE_CONTEXT_COMPACTION_REQUEST = (
    "The conversation is nearing its context limit. Take exactly one normal "
    "OpenMLE action now and use this action to create or update the workspace-relative "
    ".agent_memory/OPENMLE_CONTINUATION.md exactly once. Record the immediate "
    "objective, last measured validation metric or exact failure, conclusion, "
    "`code_path: train.py`, and make `next_action` one concrete `train.py` code improvement "
    "to perform before rerunning; a bare run is not enough. If validation has "
    "not completed, write `validation: not measured yet` and the exact blocker. "
    "Use a valid relative apply_patch or the prompt's `mkdir -p ... && printf` "
    "form. Do not inspect data, run code, or submit instead. This is one charged "
    "action; emit only one of the three valid raw action forms."
)
OPENMLE_POLICY_CONTINUATION_MARKER = (
    "Earlier conversation was removed; the workspace is unchanged. If the "
    "budget line says only one action remains and submission.csv exists, submit "
    "now. Otherwise read .agent_memory/OPENMLE_CONTINUATION.md exactly once with "
    "one normal shell_command, then immediately execute its `next_action`: modify "
    "`train.py` once before running it again. Do not inspect the task or schema again, "
    "and do not reread or rewrite the note before "
    "the next edit/run produces a new measured validation or exact failure. If "
    "the post-compaction rerun prints a finite validation metric and submission.csv "
    "exists, the next action is `submit`; do not rewrite the note or edit again, and "
    "do not start a third iteration. If that rerun fails, update the note once and "
    "continue while preserving one final action. All operations consume the shared "
    "30-action budget."
)

_OPENMLE_CONTINUATION_PATH = ".agent_memory/OPENMLE_CONTINUATION.md"
_EXPECTED_BOUNDARIES = {
    "actions": "openmle_fast_three_tool_v1",
    "workspace": "openmle_fast_public_workspace_v1",
    "executor": "openmle_fast_executor_v1",
    "grader": "openmle_fast_authenticated_private_ipc_v1",
    "observation": "openmle_fast_bounded_observation_v1",
    "horizon": "openmle_fast_global_30_action_v1",
    "cleanup": "openmle_fast_owned_resource_cleanup_v1",
    "audit": "openmle_fast_append_only_episode_audit_v1",
}
_FROZEN_RESOURCE_LIMITS = {
    "max_policy_actions": 30,
    "cpu_vcpus": 2,
    "memory_bytes": 4 * 1024**3,
    "swap_bytes": 0,
    "workspace_bytes": 2 * 1024**3,
    "tmp_bytes": 256 * 1024**2,
    "max_processes": 64,
    "max_open_files": 256,
    "max_files": 100_000,
    "max_file_bytes": 256 * 1024**2,
    "max_submission_bytes": 64 * 1024**2,
    "shell_wall_ms": 20_000,
    "managed_runtime_per_action_ms": 15_000,
    "managed_runtime_per_episode_ms": 120_000,
    "episode_wall_ms": 180_000,
    "grader_cpu_vcpus": 1,
    "grader_memory_bytes": 2 * 1024**3,
    "grader_max_processes": 32,
    "grader_worker_wall_ms": 4_000,
    "grader_total_wall_ms": 5_000,
    "grader_max_concurrent_requests": 8,
    "grader_input_bytes": 64 * 1024**2,
    "raw_output_bytes": 8 * 1024**2,
    "observation_bytes": 64 * 1024,
    "observation_head_bytes": 32 * 1024,
    "observation_tail_bytes": 32 * 1024,
}
_OPENMLE_RELEASE_REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"

_ALLOWED_MANIFEST_ROLES = frozenset({"gate_only", "train_pool", "heldout"})
_CLOSE_MAX_ATTEMPTS = 3
_GRADE_SCHEMA = "openmle_fast_grade_response_v1"
_GRADE_CONTRACT_VERSION = "openmle_fast_v1"
_PUBLIC_GRADE_FIELDS = frozenset(
    {
        "schema",
        "contract_version",
        "request_id",
        "episode_id",
        "task_id",
        "submission_sha256",
        "submission_valid",
        "native_score",
        "higher_is_better",
        "normalized_reward",
        "improved_over_baseline",
        "runtime_success",
        "terminal_reason",
        "classification",
        "audit_digest",
    }
)


class OpenMLEFastEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Understood."}),
    )

    def __init__(
        self,
        env_server_base: str,
        *,
        expected_manifest_sha256: str | None = None,
        expected_release_revision: str | None = None,
        expected_outer_commit: str | None = None,
        expected_inner_commit: str | None = None,
        expected_role: str | None = None,
        expected_executor_runtime_digest: str | None = None,
        expected_materializer_sha256: str | None = None,
        expected_actions_sha256: str | None = None,
        expected_max_observation_tokens: int | None = None,
        data_len: int | None = None,
        allow_ineligible_test_backend: bool = False,
        timeout: float = 200.0,
        timeout_margin_seconds: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.env_server_base = env_server_base.rstrip("/")
        if not self.env_server_base:
            raise ValueError("OpenMLE-fast endpoint must not be empty")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("OpenMLE-fast client timeout must be positive")
        if (
            isinstance(timeout_margin_seconds, bool)
            or not isinstance(timeout_margin_seconds, (int, float))
            or not math.isfinite(float(timeout_margin_seconds))
            or timeout_margin_seconds <= 0
        ):
            raise ValueError("OpenMLE-fast timeout margin must be positive")
        if type(allow_ineligible_test_backend) is not bool:
            raise TypeError("OpenMLE-fast test-backend opt-in must be Boolean")
        self.timeout = float(timeout)
        metadata = self._request("GET", "metadata")
        expected_manifest_sha256 = _resolve_expected_text(
            expected_manifest_sha256,
            "OPENMLE_FAST_TASK_MANIFEST_SHA256",
            None,
        )
        expected_release_revision = _resolve_expected_text(
            expected_release_revision,
            "OPENMLE_FAST_RELEASE_REVISION",
            _OPENMLE_RELEASE_REVISION,
        )
        expected_outer_commit = _resolve_expected_text(
            expected_outer_commit,
            "OPENMLE_FAST_RUNTIME_OUTER_COMMIT",
            None,
        )
        expected_inner_commit = _resolve_expected_text(
            expected_inner_commit,
            "OPENMLE_FAST_RUNTIME_INNER_COMMIT",
            None,
        )
        expected_role = _resolve_expected_text(
            expected_role,
            "OPENMLE_FAST_MANIFEST_ROLE",
            None,
        )
        expected_executor_runtime_digest = _resolve_expected_text(
            expected_executor_runtime_digest,
            "OPENMLE_FAST_EXECUTOR_RUNTIME_DIGEST",
            None,
        )
        expected_materializer_sha256 = _resolve_expected_text(
            expected_materializer_sha256,
            "OPENMLE_FAST_MATERIALIZER_SHA256",
            None,
        )
        expected_actions_sha256 = _resolve_expected_text(
            expected_actions_sha256,
            "OPENMLE_FAST_ACTIONS_SHA256",
            None,
        )
        expected_max_observation_tokens = _resolve_expected_integer(
            expected_max_observation_tokens,
            "OPENMLE_FAST_MAX_OBSERVATION_TOKENS",
            None,
        )
        _validate_expected_configuration(
            manifest_sha256=expected_manifest_sha256,
            release_revision=expected_release_revision,
            outer_commit=expected_outer_commit,
            inner_commit=expected_inner_commit,
            role=expected_role,
            executor_runtime_digest=expected_executor_runtime_digest,
            materializer_sha256=expected_materializer_sha256,
            actions_sha256=expected_actions_sha256,
            max_observation_tokens=expected_max_observation_tokens,
        )
        _attest_metadata(
            metadata,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_release_revision=expected_release_revision,
            expected_outer_commit=expected_outer_commit,
            expected_inner_commit=expected_inner_commit,
            expected_role=expected_role,
            expected_executor_runtime_digest=expected_executor_runtime_digest,
            expected_materializer_sha256=expected_materializer_sha256,
            expected_actions_sha256=expected_actions_sha256,
            expected_max_observation_tokens=expected_max_observation_tokens,
            allow_ineligible_test_backend=allow_ineligible_test_backend,
        )
        resource_limits = metadata["resource_limits"]
        longest_path_seconds = int(resource_limits["episode_wall_ms"]) / 1000.0
        minimum_timeout = longest_path_seconds + float(timeout_margin_seconds)
        if self.timeout <= minimum_timeout:
            raise ValueError(
                "OpenMLE-fast client timeout must exceed the longest step path "
                "plus its fixed margin"
            )
        task_count = metadata["task_count"]
        if data_len is not None and (
            type(data_len) is not int or data_len <= 0 or data_len > task_count
        ):
            raise ValueError(
                f"OpenMLE-fast data_len {data_len!r} exceeds server task_count {task_count}"
            )
        self.data_len = task_count if data_len is None else data_len
        self.metadata = metadata
        created = self._request("POST", "create", json={})
        if set(created) != {"id", "observation", "info"}:
            raise RuntimeError("OpenMLE-fast create response schema drifted")
        env_id = created["id"]
        if type(env_id) is not int or env_id < 0:
            raise RuntimeError("OpenMLE-fast create response has an invalid slot id")
        if not isinstance(created["observation"], str) or not isinstance(
            created["info"], Mapping
        ):
            raise RuntimeError(  # noqa: TRY004 - remote schema drift
                "OpenMLE-fast create response fields are invalid"
            )
        self.env_id = env_id
        self.info = created
        self._episode_identity: dict[str, Any] | None = None
        self._reset_transition_state()

    def _reset_transition_state(self) -> None:
        self._policy_step_count = 0
        self._native_call_count = 0
        self._context_epoch = 0
        self._session_epoch = 0
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None

    def __len__(self) -> int:
        return self.data_len

    @property
    def sample_excluded(self) -> bool:
        info = self.info.get("info")
        return bool(isinstance(info, Mapping) and info.get("truncated") is True)

    def observe(self) -> str:
        observation = self.info.get("observation")
        if not isinstance(observation, str):
            raise RuntimeError(  # noqa: TRY004 - remote schema drift
                "OpenMLE-fast observation is not text"
            )
        return observation

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": OPENMLE_FAST_POLICY_SYSTEM_PROMPT}]

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        normalized = _copy_messages(messages)
        observation = self.observe()
        if (
            not normalized
            or normalized[-1]["role"] != "user"
            or normalized[-1]["content"] != observation
        ):
            raise ValueError(
                "OpenMLE-fast initial policy context must end with its observation"
            )
        return self.policy_framing() + [{"role": "user", "content": observation}]

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = _copy_messages(messages)
        if initial:
            expected = self.policy_framing() + [
                {"role": "user", "content": self.observe()}
            ]
            if normalized != expected:
                raise ValueError(
                    "OpenMLE-fast initial policy context differs from its framing"
                )
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if not self._policy_context_bound:
            return None
        return OPENMLE_CONTEXT_COMPACTION_REQUEST

    def prepare_policy_turn(self, pressure: PolicyContextPressure | None) -> str | None:
        self._selected_policy_control = None
        if not self._policy_context_bound:
            return None
        if pressure is None:
            raise RuntimeError(
                "OpenMLE-fast context compaction requires token pressure"
            )
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "OpenMLE-fast context reached the prompt cap before a trainable "
                "persistence action could be sampled"
            )
        # Decide from the no-control append path, whose base is the preserved
        # Continuous Token runtime.  The freshly rendered control candidate may
        # legitimately be shorter after generation-only history normalization,
        # so using it here can miss an imminent overflow on the ordinary path.
        if pressure.projected_next_prompt_tokens_without_control < capacity:
            return None
        self._selected_policy_control = "context_compaction"
        return OPENMLE_CONTEXT_COMPACTION_REQUEST

    def step(self, action: str) -> StepOutput:
        if not isinstance(action, str):
            raise TypeError("OpenMLE-fast action must be raw policy text")
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        session_before = self._session_epoch
        context_control_selected = self._selected_policy_control == "context_compaction"
        response = self._request(
            "POST", "step", json={"id": self.env_id, "action": action}
        )
        state, reward, done, env_info = _validate_step_response(
            response,
            metadata=self.metadata,
            expected_action_count=native_before + 1,
            expected_action_delta=1,
            expected_episode_identity=self._episode_identity,
        )
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        self._selected_policy_control = None
        action_submission = _action_submission(
            action=action,
            done=done,
            env_info=env_info,
        )
        context_transition = None
        wrapper_evidence = {
            "event": "native_action",
            "workspace_continuity_id": self.env_id,
            "action_contract": _EXPECTED_BOUNDARIES["actions"],
        }
        if context_control_selected and not done:
            framing = self._immutable_policy_context
            if framing is None:
                raise RuntimeError(
                    "OpenMLE-fast compaction lost its immutable task framing"
                )
            replacement = deepcopy(framing)
            replacement.extend(
                [
                    {"role": "assistant", "content": action},
                    {
                        "role": "user",
                        "content": f"{state}\n\n{OPENMLE_POLICY_CONTINUATION_MARKER}",
                    },
                ]
            )
            self._context_epoch += 1
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=replacement,
            )
            continuation_persisted = _continuation_write_succeeded(env_info)
            wrapper_evidence = {
                "event": "context_compaction",
                "workspace_continuity_id": self.env_id,
                "action_contract": _EXPECTED_BOUNDARIES["actions"],
                "native_action_kind": env_info["action_kind"],
                "native_action_status": env_info["action_status"],
                "continuation_path": _OPENMLE_CONTINUATION_PATH,
                "continuation_persisted": continuation_persisted,
                "preserved_policy_output": True,
                "preserved_native_observation": True,
            }
        return StepOutput(
            state=state,
            reward=reward,
            done=done,
            info=build_task_neutral_transition_info(
                env_info=env_info,
                action_submission=action_submission,
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
                context_transition=context_transition,
                wrapper_evidence=wrapper_evidence,
            ),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        if type(idx) is not int or idx < 0 or idx >= self.data_len:
            raise ValueError(
                "OpenMLE-fast reset index is outside the configured data range"
            )
        response = self._request(
            "POST", "reset", json={"id": self.env_id, "data_idx": idx}
        )
        _, _, _, env_info = _validate_step_response(
            response,
            metadata=self.metadata,
            expected_action_count=None,
            expected_action_delta=0,
            expected_action_kind="reset",
            expected_data_idx=idx,
        )
        if not env_info["truncated"] and env_info["counters"]["action_count"] != 0:
            raise RuntimeError("OpenMLE-fast reset did not clear the action ledger")
        self.info = response
        self._reset_transition_state()
        self._episode_identity = _receipt_identity(env_info)
        if self.sample_excluded:
            raise RuntimeError("OpenMLE-fast reset was truncated and must be resampled")
        return response

    def finalize_policy_horizon(self) -> StepOutput:
        response = self._request("POST", "horizon", json={"id": self.env_id})
        state, reward, done, env_info = _validate_step_response(
            response,
            metadata=self.metadata,
            expected_action_count=self._native_call_count,
            expected_action_delta=0,
            expected_action_kind="policy_horizon",
            require_terminal=True,
            expected_episode_identity=self._episode_identity,
        )
        self.info = response
        return StepOutput(
            state=state,
            reward=reward,
            done=done,
            info=build_task_neutral_transition_info(
                env_info=env_info,
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

    def close(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        response: dict[str, Any] | None = None
        for _attempt in range(_CLOSE_MAX_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            response = self._request(
                "POST",
                "close",
                json={"id": self.env_id},
                request_timeout=remaining,
            )
            _validate_cleanup_receipt(response)
            if response["closed"] or response["already_closed"]:
                return response
            if not response["retryable"]:
                break
        raise RuntimeError("OpenMLE-fast cleanup did not complete")

    def _request(
        self,
        method: str,
        path: str,
        *,
        request_timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        timeout = self.timeout if request_timeout is None else request_timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OpenMLE-fast request timeout must be finite and positive")
        response = requests.request(
            method,
            f"{self.env_server_base}/{path}",
            timeout=timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"OpenMLE-fast {method} /{path} failed with status "
                f"{response.status_code}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"OpenMLE-fast {method} /{path} returned a non-object response"
            )
        return value


def _continuation_write_succeeded(env_info: Mapping[str, Any]) -> bool:
    if (
        env_info.get("action_kind") not in {"apply_patch", "shell_command"}
        or env_info.get("action_status") != "completed"
    ):
        return False
    execution = env_info.get("execution")
    if not isinstance(execution, Mapping):
        return False
    changed_paths = execution.get("changed_paths")
    if (
        not isinstance(changed_paths, Sequence)
        or isinstance(changed_paths, (str, bytes))
    ):
        return False
    return _OPENMLE_CONTINUATION_PATH in changed_paths


def _action_submission(
    *,
    action: str,
    done: bool,
    env_info: Mapping[str, Any],
) -> dict[str, Any]:
    submission: dict[str, Any] = {"raw_policy_output": action}
    grade = env_info.get("grade")
    if grade is None:
        return submission
    if not done or not isinstance(grade, Mapping) or set(grade) != _PUBLIC_GRADE_FIELDS:
        raise RuntimeError("OpenMLE-fast terminal grade receipt schema drifted")
    request_id = grade.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("OpenMLE-fast grade request identity is invalid")
    if grade.get("episode_id") != env_info.get("episode_id"):
        raise RuntimeError("OpenMLE-fast grade episode identity drifted")
    if grade.get("task_id") != env_info.get("task_id"):
        raise RuntimeError("OpenMLE-fast grade task identity drifted")
    submission_sha256 = _require_sha256(
        grade.get("submission_sha256"), "grade submission_sha256"
    )
    submission.update(
        {
            "request_id": request_id,
            "episode_id": grade["episode_id"],
            "submission_sha256": submission_sha256,
        }
    )
    return submission


def _validate_cleanup_receipt(response: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "closed",
        "already_closed",
        "workspace_removed",
        "retryable",
        "failure_class",
        "cleanup_contract",
    }
    if (
        set(response) != required
        or response["schema"] != "openmle_fast_cleanup_receipt_v1"
        or response["cleanup_contract"] != _EXPECTED_BOUNDARIES["cleanup"]
        or any(
            type(response[key]) is not bool
            for key in (
                "closed",
                "already_closed",
                "workspace_removed",
                "retryable",
            )
        )
        or (
            response["failure_class"] is not None
            and not isinstance(response["failure_class"], str)
        )
    ):
        raise RuntimeError("OpenMLE-fast cleanup response schema drifted")
    if response["closed"] and (
        response["already_closed"]
        or response["retryable"]
        or response["failure_class"] is not None
    ):
        raise RuntimeError("OpenMLE-fast cleanup response is inconsistent")
    if response["already_closed"] and (
        response["closed"]
        or response["retryable"]
        or response["failure_class"] is not None
    ):
        raise RuntimeError("OpenMLE-fast cleanup response is inconsistent")
    if response["retryable"] and (
        response["closed"]
        or response["already_closed"]
        or not isinstance(response["failure_class"], str)
    ):
        raise RuntimeError("OpenMLE-fast cleanup response is inconsistent")
    if not (response["closed"] or response["already_closed"] or response["retryable"]):
        raise RuntimeError("OpenMLE-fast cleanup response is inconsistent")


class OpenMLEFastTask(BaseTask):
    env_client_cls = OpenMLEFastEnvClient
    env_name = "OpenMLE-fast"


def _resolve_expected_text(
    explicit: Any,
    environment_name: str,
    fallback: Any,
) -> str:
    environment_value = os.environ.get(environment_name)
    value = (
        explicit
        if explicit is not None
        else environment_value
        if environment_value is not None and environment_value.strip()
        else fallback
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"OpenMLE-fast expected {environment_name} is missing")
    return value.strip()


def _resolve_expected_integer(
    explicit: Any,
    environment_name: str,
    fallback: Any,
) -> int:
    environment_value = os.environ.get(environment_name)
    value = explicit
    if value is None and environment_value is not None and environment_value.strip():
        try:
            value = int(environment_value)
        except ValueError as exc:
            raise ValueError(
                f"OpenMLE-fast expected {environment_name} must be an integer"
            ) from exc
    if value is None:
        value = fallback
    if type(value) is not int or value <= 0:
        raise ValueError(f"OpenMLE-fast expected {environment_name} is invalid")
    return value


_RECEIPT_IDENTITY_FIELDS = (
    "episode_id",
    "data_idx",
    "task_id",
    "source_family",
    "public_tree_sha256",
    "archive_sha256",
    "package_identity_sha256",
    "task_spec_sha256",
    "grader_binding_sha256",
)


def _receipt_identity(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: info.get(key) for key in _RECEIPT_IDENTITY_FIELDS}


def _validate_expected_configuration(
    *,
    manifest_sha256: str,
    release_revision: str,
    outer_commit: str,
    inner_commit: str,
    role: str,
    executor_runtime_digest: str,
    materializer_sha256: str,
    actions_sha256: str,
    max_observation_tokens: int,
) -> None:
    for value, label in (
        (manifest_sha256, "manifest_sha256"),
        (materializer_sha256, "materializer_sha256"),
        (actions_sha256, "actions_sha256"),
    ):
        try:
            _require_sha256(value, label)
        except RuntimeError as exc:
            raise ValueError(f"OpenMLE-fast expected {label} is invalid") from exc
    for value, label in (
        (release_revision, "release_revision"),
        (outer_commit, "outer_commit"),
        (inner_commit, "inner_commit"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"OpenMLE-fast {label} must be a full Git revision")
    if role not in _ALLOWED_MANIFEST_ROLES:
        raise ValueError("OpenMLE-fast expected role is not executable")
    if (
        not isinstance(executor_runtime_digest, str)
        or not executor_runtime_digest.startswith("sha256:")
        or len(executor_runtime_digest) != 71
        or any(
            character not in "0123456789abcdef"
            for character in executor_runtime_digest[7:]
        )
    ):
        raise ValueError("OpenMLE-fast executor runtime digest is invalid")
    if type(max_observation_tokens) is not int or max_observation_tokens <= 0:
        raise ValueError("OpenMLE-fast observation token cap must be positive")


def _attest_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_release_revision: str,
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_role: str,
    expected_executor_runtime_digest: str,
    expected_materializer_sha256: str,
    expected_actions_sha256: str,
    expected_max_observation_tokens: int,
    allow_ineligible_test_backend: bool,
) -> None:
    expected = {
        "schema": "openmle_fast_public_metadata_v1",
        "domain_id": "openmle_fast",
        "contract_version": "openmle_fast_v1",
        "openmle_tasks_revision": expected_release_revision,
        "task_manifest_sha256": expected_manifest_sha256,
        "policy_prompt_sha256": OPENMLE_FAST_POLICY_PROMPT_SHA256,
        "max_policy_actions": 30,
        "observation_max_bytes": 64 * 1024,
        "max_observation_tokens": expected_max_observation_tokens,
        "executor_runtime_digest": expected_executor_runtime_digest,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"OpenMLE-fast endpoint metadata mismatch for {key}")
    if (
        metadata.get("release_revision") != expected_release_revision
        or metadata.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise RuntimeError("OpenMLE-fast endpoint metadata aliases drifted")
    for key in ("panel_id", "task_id_list_sha256", "compact_panel_sha256"):
        value = metadata.get(key)
        if key == "panel_id":
            if not isinstance(value, str) or not value:
                raise RuntimeError("OpenMLE-fast endpoint panel identity is invalid")
        else:
            _require_sha256(value, key)
    if metadata.get("role") != expected_role:
        raise RuntimeError("OpenMLE-fast endpoint role mismatch")
    task_count = metadata.get("task_count")
    if type(task_count) is not int or task_count <= 0:
        raise RuntimeError("OpenMLE-fast endpoint task_count is invalid")
    source = metadata.get("runtime_source")
    if (
        not isinstance(source, Mapping)
        or source.get("outer_commit") != expected_outer_commit
        or source.get("inner_commit") != expected_inner_commit
    ):
        raise RuntimeError("OpenMLE-fast endpoint runtime source mismatch")
    boundaries = metadata.get("boundary_contracts")
    if boundaries != _EXPECTED_BOUNDARIES:
        raise RuntimeError("OpenMLE-fast endpoint boundary contracts drifted")
    contracts = metadata.get("contracts")
    expected_contracts = {
        "action": _EXPECTED_BOUNDARIES["actions"],
        "observation": _EXPECTED_BOUNDARIES["observation"],
        "horizon": _EXPECTED_BOUNDARIES["horizon"],
        "workspace": _EXPECTED_BOUNDARIES["workspace"],
        "executor": _EXPECTED_BOUNDARIES["executor"],
        "grader_boundary": _EXPECTED_BOUNDARIES["grader"],
        "cleanup": _EXPECTED_BOUNDARIES["cleanup"],
    }
    if contracts != expected_contracts:
        raise RuntimeError("OpenMLE-fast endpoint verifier contracts drifted")
    resource_limits = metadata.get("resource_limits")
    if resource_limits != _FROZEN_RESOURCE_LIMITS:
        raise RuntimeError("OpenMLE-fast endpoint resource limits drifted")
    limits = metadata.get("limits")
    if (
        not isinstance(limits, Mapping)
        or limits.get("max_policy_actions") != 30
        or isinstance(limits.get("max_request_wall_seconds"), bool)
        or not isinstance(limits.get("max_request_wall_seconds"), (int, float))
        or limits["max_request_wall_seconds"] <= 0
    ):
        raise RuntimeError("OpenMLE-fast endpoint verifier limits drifted")
    implementation = metadata.get("implementation_digests")
    if implementation != {
        "materializer_sha256": expected_materializer_sha256,
        "actions_sha256": expected_actions_sha256,
    }:
        raise RuntimeError("OpenMLE-fast endpoint implementation digests drifted")
    coverage = metadata.get("executor_coverage")
    if not isinstance(coverage, Mapping):
        raise RuntimeError(  # noqa: TRY004 - remote schema drift
            "OpenMLE-fast endpoint executor coverage is missing"
        )
    if allow_ineligible_test_backend:
        if type(coverage.get("formal_eligible")) is not bool:
            raise RuntimeError("OpenMLE-fast endpoint eligibility is invalid")
    elif (
        coverage.get("formal_eligible") is not True
        or coverage.get("backend_contract")
        != "openmle_fast_linux_cgroup_namespace_runner_v1"
        or coverage.get("execution_counter_coverage") != "complete"
        or coverage.get("fit_counter_coverage") != "partial"
    ):
        raise RuntimeError("OpenMLE-fast endpoint is not formal-runtime eligible")
    if type(metadata.get("audit_enabled")) is not bool or (
        not allow_ineligible_test_backend and metadata["audit_enabled"] is not True
    ):
        raise RuntimeError("OpenMLE-fast endpoint audit contract is inactive")
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise RuntimeError("OpenMLE-fast endpoint active counts are invalid")


def _validate_step_response(
    response: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    expected_action_count: int | None,
    expected_action_delta: int,
    expected_action_kind: str | None = None,
    require_terminal: bool = False,
    expected_data_idx: int | None = None,
    expected_episode_identity: Mapping[str, Any] | None = None,
) -> tuple[str, float | None, bool, Mapping[str, Any]]:
    if set(response) != {
        "observation",
        "state",
        "reward",
        "done",
        "truncated",
        "info",
    }:
        raise RuntimeError("OpenMLE-fast step response schema drifted")
    observation = response["observation"]
    if (
        not isinstance(observation, str)
        or response["state"] != observation
        or len(observation.encode("utf-8")) > metadata["observation_max_bytes"]
        or len(observation.encode("utf-8")) > metadata["max_observation_tokens"]
    ):
        raise RuntimeError("OpenMLE-fast response observation is invalid")
    done = response["done"]
    if type(done) is not bool or (require_terminal and done is not True):
        raise RuntimeError("OpenMLE-fast response terminal flag is invalid")
    reward = _nullable_reward(response["reward"])
    info = response["info"]
    required_info = {
        "schema",
        "episode_id",
        "data_idx",
        "task_id",
        "source_family",
        "public_tree_sha256",
        "manifest_sha256",
        "task_manifest_sha256",
        "release_revision",
        "manifest_role",
        "archive_sha256",
        "package_identity_sha256",
        "task_spec_sha256",
        "grader_binding_sha256",
        "runtime_source",
        "executor_runtime_digest",
        "implementation_digests",
        "boundary_contracts",
        "action_kind",
        "action_status",
        "terminal",
        "truncated",
        "terminal_reason",
        "runtime_success",
        "episode_success",
        "counters",
        "counter_delta",
        "fit_counter_coverage",
        "execution",
        "sandbox_freeze",
        "sandbox_teardown",
        "grade",
        "audit_digest",
        "unaudited_evidence_sha256",
    }
    if not isinstance(info, Mapping) or set(info) != required_info:
        raise RuntimeError("OpenMLE-fast environment receipt schema drifted")
    if (
        info["schema"] != "openmle_fast_episode_v1"
        or not isinstance(info["episode_id"], str)
        or not info["episode_id"]
        or info["manifest_sha256"] != metadata["manifest_sha256"]
        or info["task_manifest_sha256"] != metadata["task_manifest_sha256"]
        or info["release_revision"] != metadata["release_revision"]
        or info["manifest_role"] != metadata["role"]
        or info["runtime_source"] != metadata["runtime_source"]
        or info["executor_runtime_digest"] != metadata["executor_runtime_digest"]
        or info["implementation_digests"] != metadata["implementation_digests"]
        or info["boundary_contracts"] != metadata["boundary_contracts"]
    ):
        raise RuntimeError("OpenMLE-fast environment receipt provenance drifted")
    if (
        (type(info["data_idx"]) is not int or info["data_idx"] < 0)
        and info["data_idx"] is not None
    ) or any(
        not isinstance(info[key], str) and info[key] is not None
        for key in ("task_id", "source_family")
    ):
        raise RuntimeError("OpenMLE-fast task identity fields are invalid")
    if (
        expected_data_idx is not None
        and not info["truncated"]
        and info["data_idx"] != expected_data_idx
    ):
        raise RuntimeError("OpenMLE-fast reset returned the wrong data_idx")
    if expected_episode_identity is not None and _receipt_identity(info) != dict(
        expected_episode_identity
    ):
        raise RuntimeError("OpenMLE-fast episode/task identity drifted")
    if not isinstance(info["action_kind"], str) or not isinstance(
        info["action_status"], str
    ):
        raise RuntimeError(  # noqa: TRY004 - remote schema drift
            "OpenMLE-fast action receipt fields are invalid"
        )
    if expected_action_kind is not None and info["action_kind"] != expected_action_kind:
        raise RuntimeError("OpenMLE-fast action kind drifted")
    for key in (
        "public_tree_sha256",
        "archive_sha256",
        "package_identity_sha256",
        "task_spec_sha256",
        "grader_binding_sha256",
    ):
        if info[key] is not None:
            _require_sha256(info[key], key)
    if type(info["terminal"]) is not bool or info["terminal"] != done:
        raise RuntimeError("OpenMLE-fast receipt terminal flag drifted")
    if type(info["truncated"]) is not bool or (info["truncated"] and not done):
        raise RuntimeError("OpenMLE-fast receipt truncation flag is invalid")
    if (
        type(response["truncated"]) is not bool
        or response["truncated"] != info["truncated"]
    ):
        raise RuntimeError("OpenMLE-fast response truncation flag drifted")
    if any(
        type(info[key]) is not bool for key in ("runtime_success", "episode_success")
    ):
        raise RuntimeError("OpenMLE-fast receipt success flags are invalid")
    terminal_reason = info["terminal_reason"]
    if (done and not isinstance(terminal_reason, str)) or (
        not done and terminal_reason is not None
    ):
        raise RuntimeError("OpenMLE-fast terminal reason is invalid")
    if not done and reward != 0.0:
        raise RuntimeError("OpenMLE-fast intermediate reward must be zero")
    if info["truncated"] and reward is not None:
        raise RuntimeError("OpenMLE-fast truncation must carry null reward")
    if done and not info["truncated"] and (reward is None or not -1.0 <= reward <= 1.0):
        raise RuntimeError("OpenMLE-fast terminal reward is invalid")
    _validate_counters(info["counters"], expected_action_count)
    _validate_counters(info["counter_delta"], None)
    if info["counter_delta"]["action_count"] != expected_action_delta:
        raise RuntimeError("OpenMLE-fast action counter delta drifted")
    for key, delta in info["counter_delta"].items():
        if delta > info["counters"][key]:
            raise RuntimeError("OpenMLE-fast counter delta exceeds cumulative value")
    for key in ("execution", "sandbox_freeze", "sandbox_teardown", "grade"):
        if info[key] is not None and not isinstance(info[key], Mapping):
            raise RuntimeError(f"OpenMLE-fast {key} receipt is invalid")
    if info["grade"] is not None:
        _validate_public_grade_receipt(
            info["grade"],
            reward=reward,
            done=done,
            info=info,
        )
    audit_digest = info["audit_digest"]
    unaudited = info["unaudited_evidence_sha256"]
    if audit_digest is not None:
        _require_sha256(audit_digest, "audit_digest")
    elif metadata["audit_enabled"] and not info["truncated"]:
        raise RuntimeError("OpenMLE-fast audited response omitted its audit digest")
    if unaudited is not None:
        _require_sha256(unaudited, "unaudited_evidence_sha256")
        if not info["truncated"]:
            raise RuntimeError("unaudited evidence requires truncation")
    return observation, reward, done, info


def _validate_public_grade_receipt(
    receipt: Mapping[str, Any],
    *,
    reward: float | None,
    done: bool,
    info: Mapping[str, Any],
) -> None:
    if set(receipt) != _PUBLIC_GRADE_FIELDS:
        raise RuntimeError("OpenMLE-fast terminal grade receipt schema drifted")
    if (
        receipt["schema"] != _GRADE_SCHEMA
        or receipt["contract_version"] != _GRADE_CONTRACT_VERSION
        or done is not True
        or info["action_kind"] != "submit"
    ):
        raise RuntimeError("OpenMLE-fast terminal grade contract drifted")
    request_id = receipt["request_id"]
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("OpenMLE-fast grade request identity is invalid")
    if (
        receipt["episode_id"] != info["episode_id"]
        or receipt["task_id"] != info["task_id"]
    ):
        raise RuntimeError("OpenMLE-fast grade task identity drifted")
    _require_sha256(receipt["submission_sha256"], "grade submission_sha256")
    _require_sha256(receipt["audit_digest"], "grade audit_digest")
    for field in (
        "submission_valid",
        "higher_is_better",
        "improved_over_baseline",
        "runtime_success",
    ):
        if type(receipt[field]) is not bool:
            raise RuntimeError(f"OpenMLE-fast grade {field} is not Boolean")
    native_score = _nullable_finite_number(receipt["native_score"], "native_score")
    normalized_reward = _nullable_finite_number(
        receipt["normalized_reward"], "normalized_reward"
    )
    if normalized_reward is not None and not -1.0 <= normalized_reward <= 1.0:
        raise RuntimeError("OpenMLE-fast grade reward is outside [-1, 1]")
    classification = receipt["classification"]
    terminal_reason = receipt["terminal_reason"]
    submission_valid = receipt["submission_valid"]
    improved = receipt["improved_over_baseline"]
    runtime_success = receipt["runtime_success"]
    truncated = info["truncated"]
    if classification == "graded":
        consistent = (
            submission_valid
            and native_score is not None
            and normalized_reward is not None
            and runtime_success
            and terminal_reason == "graded_submission"
            and truncated is False
            and reward == normalized_reward
        )
    elif classification == "invalid_submission":
        consistent = (
            not submission_valid
            and native_score is None
            and normalized_reward == -1.0
            and not improved
            and not runtime_success
            and terminal_reason == "invalid_submission"
            and truncated is False
            and reward == -1.0
        )
    elif classification == "infrastructure_fault":
        consistent = (
            not submission_valid
            and native_score is None
            and normalized_reward is None
            and not improved
            and not runtime_success
            and terminal_reason == "grader_infrastructure_fault"
            and truncated is True
            and reward is None
        )
    else:
        raise RuntimeError("OpenMLE-fast grade classification is invalid")
    if not consistent or (improved and not submission_valid):
        raise RuntimeError("OpenMLE-fast terminal grade receipt is inconsistent")
    if (
        info["terminal_reason"] != terminal_reason
        or info["runtime_success"] != runtime_success
        or info["episode_success"] != (submission_valid and improved)
        or info["counters"]["grading_count"] != 1
        or info["counter_delta"]["grading_count"] != 1
    ):
        raise RuntimeError("OpenMLE-fast grade/environment receipt drifted")


def _validate_counters(value: Any, expected_action_count: int | None) -> None:
    names = {
        "action_count",
        "execution_action_count",
        "execution_attempt_count",
        "execution_completed_count",
        "nested_subprocess_count",
        "fit_count",
        "grading_count",
        "managed_runtime_wall_seconds",
        "raw_output_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != names:
        raise RuntimeError("OpenMLE-fast counters schema drifted")
    for name in names - {"managed_runtime_wall_seconds"}:
        if type(value[name]) is not int or value[name] < 0:
            raise RuntimeError("OpenMLE-fast integer counter is invalid")
    wall = value["managed_runtime_wall_seconds"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        raise RuntimeError(  # noqa: TRY004 - remote schema drift
            "OpenMLE-fast runtime counter is invalid"
        )
    if not math.isfinite(float(wall)) or wall < 0:
        raise RuntimeError("OpenMLE-fast runtime counter is not finite")
    if (
        expected_action_count is not None
        and value["action_count"] != expected_action_count
    ):
        raise RuntimeError("OpenMLE-fast action ledger drifted")
    if value["action_count"] > 30 or value["grading_count"] > 1:
        raise RuntimeError("OpenMLE-fast bounded counter exceeded its cap")
    if value["execution_completed_count"] > value["execution_attempt_count"]:
        raise RuntimeError("OpenMLE-fast completed count exceeds attempts")


def _nullable_finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"OpenMLE-fast grade {label} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"OpenMLE-fast grade {label} is not finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise RuntimeError(f"OpenMLE-fast {label} is not a SHA256 digest")
    return value.lower()


def _nullable_reward(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(  # noqa: TRY004 - remote contract drift is a runtime fault
            "OpenMLE-fast response reward is invalid"
        )
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("OpenMLE-fast response reward must be finite")
    return result


def _copy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"policy message {index} is invalid")
        normalized.append({"role": role, "content": content})
    return normalized
