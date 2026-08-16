from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    POLICY_CONTINUATION_MARKER,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)

UPSTREAM_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
SPLIT_SHA256 = "590270f007fa96b4060f59f3861500159c73ca50f7f30ff6bd38303c236c799b"
LITE_COMPETITION_IDS = (
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
)
SUBMISSION_PATH = "/home/submission/submission.csv"
MODE_NATIVE = "native"
MODE_AMG_COMPACTION_ONLY = "amg_compaction_only"
MODE_AMG_MEMORY = "amg_memory"
MODES = (MODE_NATIVE, MODE_AMG_COMPACTION_ONLY, MODE_AMG_MEMORY)
COMPACTION_MODES = (MODE_AMG_COMPACTION_ONLY, MODE_AMG_MEMORY)
METADATA_SCHEMA = "mlebench_lite_metadata_v2"
COMPACTION_RECEIPT_SCHEMA = "mlebench_lite_compaction_receipt_v2"
RESOURCE_CONTRACT_SCHEMA = "mlebench_lite_resource_contract_v2"
STEP_RESPONSE_SLACK_MS = 30_000
RESOURCE_USAGE_KEYS = (
    "execution_time_ms",
    "cpu_time_ms",
    "writable_bytes",
    "writable_inodes",
    "processes_started",
)
_RESOURCE_CONTRACT_NUMERIC_FIELDS = (
    "max_actions",
    "max_submission_bytes",
    "max_shell_timeout_ms",
    "max_visible_output_bytes",
    "episode_timeout_ms",
    "max_total_execution_ms",
    "cpu_limit_cores",
    "memory_limit_bytes",
    "pids_limit",
    "writable_bytes_limit",
    "writable_inodes_limit",
    "gpu_count",
    "max_step_response_ms",
)
_RESOURCE_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        *_RESOURCE_CONTRACT_NUMERIC_FIELDS,
        "submission_path",
        "network_disabled",
        "read_only_public_data",
        "process_scope",
        "cgroup_required",
        "isolated_process_group_required",
    }
)

BASE_POLICY_PROMPT = (
    "You are an ML coding agent in one isolated competition workspace. Every turn "
    "must be exactly one action. Use `inspect` with JSON fields path and optional "
    "offset/max_bytes; `edit` with path/content; `shell` with command and optional "
    "timeout_ms; or the exact word `submit`. Public data is read-only at /home/data. "
    "Work under /home/workspace and write the final CSV only to "
    "/home/submission/submission.csv. Do not add prose or Markdown around actions."
)
MEMORY_POLICY_ADDITION = (
    " This arm also mounts a separate empty per-task durable-note store at "
    "/run/amg_memory. Use the existing edit and inspect actions with absolute paths "
    "under /run/amg_memory, or the existing shell action, to write and read notes. "
    "This store is separate from /home/workspace and is never part of the submission. "
    "Before context compaction, keep detailed durable evidence there and return only "
    "a short handoff with the note path and immediate next action."
)
COMPACTION_REQUEST = (
    "The context is nearing its limit. Return a short continuation handoff only. "
    "Preserve the immediate objective, decisive state, next action, and paths of "
    "ordinary workspace notes you already wrote. This response consumes one action "
    "and does not execute a tool."
)


Requester = Callable[..., Any]


class MLEBenchLiteEnvClient(BaseEnvClient):
    def __init__(
        self,
        env_server_base: str,
        *,
        mode: str,
        expected_public_manifest_sha256: str,
        expected_runner_sha256: str,
        expected_runtime_digest: str,
        expected_max_actions: int = 30,
        expected_max_submission_bytes: int = 100_000_000,
        expected_max_shell_timeout_ms: int = 3_600_000,
        expected_episode_timeout_ms: int = 86_400_000,
        expected_max_total_execution_ms: int = 72_000_000,
        expected_cpu_limit_cores: int = 36,
        expected_memory_limit_bytes: int = 440_000_000_000,
        expected_pids_limit: int = 4096,
        expected_writable_bytes_limit: int = 500_000_000_000,
        expected_writable_inodes_limit: int = 2_000_000,
        expected_gpu_count: int = 1,
        data_len: int | None = None,
        timeout: float | None = None,
        requester: Requester | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in MODES:
            raise ValueError("unsupported MLE-bench Lite mode")
        for label, value in (
            ("public manifest SHA256", expected_public_manifest_sha256),
            ("runner SHA256", expected_runner_sha256),
            ("runtime digest", expected_runtime_digest),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{label} must be lowercase SHA256")
        for label, value in (
            ("expected_max_actions", expected_max_actions),
            ("expected_max_submission_bytes", expected_max_submission_bytes),
            ("expected_max_shell_timeout_ms", expected_max_shell_timeout_ms),
            ("expected_episode_timeout_ms", expected_episode_timeout_ms),
            (
                "expected_max_total_execution_ms",
                expected_max_total_execution_ms,
            ),
            ("expected_cpu_limit_cores", expected_cpu_limit_cores),
            ("expected_memory_limit_bytes", expected_memory_limit_bytes),
            ("expected_pids_limit", expected_pids_limit),
            ("expected_writable_bytes_limit", expected_writable_bytes_limit),
            ("expected_writable_inodes_limit", expected_writable_inodes_limit),
            ("expected_gpu_count", expected_gpu_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if expected_max_shell_timeout_ms > expected_episode_timeout_ms:
            raise ValueError("shell timeout exceeds the episode deadline")
        if expected_max_total_execution_ms > expected_episode_timeout_ms:
            raise ValueError("execution budget exceeds the episode deadline")
        if expected_max_submission_bytes > expected_writable_bytes_limit:
            raise ValueError("submission exceeds the writable-byte budget")
        max_step_response_ms = expected_episode_timeout_ms + STEP_RESPONSE_SLACK_MS
        if timeout is None:
            timeout = max_step_response_ms / 1000.0 + 30.0
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) * 1000.0 <= max_step_response_ms
        ):
            raise ValueError("HTTP timeout must exceed max_step_response_ms")
        self.env_server_base = env_server_base.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self._requester = requests.request if requester is None else requester
        resource_contract = _resource_contract(
            max_actions=expected_max_actions,
            max_submission_bytes=expected_max_submission_bytes,
            max_shell_timeout_ms=expected_max_shell_timeout_ms,
            episode_timeout_ms=expected_episode_timeout_ms,
            max_total_execution_ms=expected_max_total_execution_ms,
            cpu_limit_cores=expected_cpu_limit_cores,
            memory_limit_bytes=expected_memory_limit_bytes,
            pids_limit=expected_pids_limit,
            writable_bytes_limit=expected_writable_bytes_limit,
            writable_inodes_limit=expected_writable_inodes_limit,
            gpu_count=expected_gpu_count,
        )
        self._expected_metadata = {
            "schema": METADATA_SCHEMA,
            "upstream_commit": UPSTREAM_COMMIT,
            "split_sha256": SPLIT_SHA256,
            "competition_ids": list(LITE_COMPETITION_IDS),
            "task_count": 22,
            "public_manifest_sha256": expected_public_manifest_sha256,
            "runner_sha256": expected_runner_sha256,
            "runtime_digest": expected_runtime_digest,
            "submission_path": SUBMISSION_PATH,
            "modes": list(MODES),
            "resource_contract": resource_contract,
            "resource_contract_sha256": _resource_contract_sha256(resource_contract),
        }
        metadata = self._request("GET", "metadata")
        try:
            received_contract = _validate_resource_contract(
                metadata.get("resource_contract")
            )
            received_contract_sha256 = _resource_contract_sha256(received_contract)
        except ValueError as exc:
            raise RuntimeError("MLE-bench Lite server metadata drifted") from exc
        if metadata.get(
            "resource_contract_sha256"
        ) != received_contract_sha256 or not _strict_equal(
            metadata, self._expected_metadata
        ):
            raise RuntimeError("MLE-bench Lite server metadata drifted")
        if data_len is not None and (
            isinstance(data_len, bool)
            or not isinstance(data_len, int)
            or data_len != 22
        ):
            raise ValueError("data_len must be absent or exactly 22")
        self.data_len = 22
        created = self._request("POST", "create", json={"mode": mode})
        if (
            set(created) != {"id", "capability_token"}
            or type(created["id"]) is not int
            or not _is_capability_token(created["capability_token"])
        ):
            raise RuntimeError("MLE-bench Lite create response drifted")
        self.env_id = created["id"]
        self._capability_token = created["capability_token"]
        self.info: dict[str, Any] = {}
        self._reset_policy_state()

    def _reset_policy_state(self) -> None:
        self._policy_step_count = 0
        self._native_call_count = 0
        self._server_action_count = 0
        self._context_epoch = 0
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None
        self._competition_id: str | None = None
        self._last_counters = _zero_counters()
        self._pending_action_id: str | None = None
        self._pending_action_payload: str | None = None

    def __len__(self) -> int:
        return self.data_len

    def policy_framing(self) -> list[dict[str, str]]:
        prompt = BASE_POLICY_PROMPT
        if self.mode == MODE_AMG_MEMORY:
            prompt += MEMORY_POLICY_ADDITION
        return [{"role": "system", "content": prompt}]

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = _copy_messages(messages)
        observation = self.observe()
        if normalized[-1] != {"role": "user", "content": observation}:
            raise ValueError(
                "MLE-bench Lite initial context must end with its observation"
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
                raise ValueError("MLE-bench Lite immutable framing drifted")
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if self.mode not in COMPACTION_MODES or not self._policy_context_bound:
            return None
        return COMPACTION_REQUEST

    def prepare_policy_turn(self, pressure: PolicyContextPressure | None) -> str | None:
        self._selected_policy_control = None
        if self.mode not in COMPACTION_MODES or not self._policy_context_bound:
            return None
        if pressure is None:
            raise RuntimeError("MLE-bench Lite compaction requires token pressure")
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError("context cap reached before compaction could be sampled")
        request_tokens = (
            pressure.candidate_prompt_tokens - pressure.action_prompt_tokens
        )
        if request_tokens <= 0:
            raise RuntimeError("compaction request must extend the action prompt")
        projected = (
            pressure.action_prompt_tokens
            + pressure.max_response_tokens
            + pressure.max_observation_tokens
            + pressure.action_observation_envelope_tokens
            + request_tokens
        )
        if projected < capacity:
            return None
        self._selected_policy_control = "compaction"
        return COMPACTION_REQUEST

    def observe(self) -> str:
        return str(self.info.get("observation", ""))

    def reset(self, idx: int = 0) -> dict[str, Any]:
        if (
            isinstance(idx, bool)
            or not isinstance(idx, int)
            or not 0 <= idx < self.data_len
        ):
            raise IndexError("MLE-bench Lite data index is out of range")
        response = self._request(
            "POST",
            "reset",
            json={
                "id": self.env_id,
                "capability_token": self._capability_token,
                "data_idx": idx,
            },
        )
        self._reset_policy_state()
        self._validate_step_response(response, expected_before=None)
        if response["done"] or response["info"]["counters"]["action_count"] != 0:
            raise RuntimeError("MLE-bench Lite reset counters drifted")
        self._competition_id = LITE_COMPETITION_IDS[idx]
        self.info = response
        return response

    def step(self, action: str) -> StepOutput:
        if self._selected_policy_control == "compaction":
            return self._complete_compaction(action)
        return self._native_step(action)

    def _native_step(self, action: str) -> StepOutput:
        before_server = self._server_action_count
        before_counters = dict(self._last_counters)
        policy_before = self._policy_step_count
        native_before = self._native_call_count
        context_before = self._context_epoch
        response = self._request_action(
            {
                "action": action,
                "expected_action_count": before_server,
            }
        )
        self._validate_step_response(response, expected_before=before_counters)
        self._validate_terminal_receipt(response)
        self._server_action_count = response["info"]["counters"]["action_count"]
        self._last_counters = dict(response["info"]["counters"])
        self._policy_step_count += 1
        self._native_call_count += 1
        self.info = response
        self._complete_pending_action()
        return StepOutput(
            state=response["observation"],
            reward=float(response["reward"]),
            done=response["done"],
            info=build_task_neutral_transition_info(
                env_info=response["info"],
                action_submission={"raw_policy_output": action},
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "native_action",
                    "server_action_count": self._server_action_count,
                },
            ),
        )

    def _complete_compaction(self, summary: str) -> StepOutput:
        framing = self._immutable_policy_context
        if framing is None:
            raise RuntimeError("MLE-bench Lite compaction lost immutable framing")
        before_server = self._server_action_count
        before_counters = dict(self._last_counters)
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        response = self._request_action(
            {
                "action": summary,
                "control": "compaction",
                "expected_action_count": before_server,
            }
        )
        self._validate_step_response(response, expected_before=before_counters)
        expected_delta = _zero_counters()
        expected_delta["action_count"] = 1
        precontrol_infrastructure_terminal = (
            response["done"]
            and response["info"].get("action_kind") == "infrastructure_terminal"
            and response["info"].get("terminal_reason") == "infrastructure_failure"
            and "control_receipt" not in response["info"]
        )
        if precontrol_infrastructure_terminal:
            if set(response["info"]) != {
                "action_kind",
                "terminal_reason",
                "counters",
                "counter_delta",
            }:
                raise RuntimeError("pre-control infrastructure terminal fields drifted")
            if response["info"]["counter_delta"] != expected_delta:
                raise RuntimeError(
                    "pre-control infrastructure terminal counter drifted"
                )
            self._server_action_count = response["info"]["counters"]["action_count"]
            self._last_counters = dict(response["info"]["counters"])
            self._policy_step_count += 1
            self._selected_policy_control = None
            self.info = response
            self._complete_pending_action()
            return StepOutput(
                state=response["observation"],
                reward=float(response["reward"]),
                done=True,
                info=build_task_neutral_transition_info(
                    env_info=response["info"],
                    action_submission={
                        "raw_policy_output": summary,
                        "parser_status": "precontrol_infrastructure_terminal",
                    },
                    native_step_before=self._native_call_count,
                    native_step_after=self._native_call_count,
                    native_call_count_before=self._native_call_count,
                    native_call_count_after=self._native_call_count,
                    context_epoch_before=context_before,
                    context_epoch_after=self._context_epoch,
                    policy_step_before=policy_before,
                    policy_step_after=self._policy_step_count,
                    wrapper_evidence={
                        "event": "precontrol_infrastructure_terminal",
                        "server_action_count": self._server_action_count,
                    },
                ),
            )
        try:
            accepted = len(summary.encode("utf-8")) <= 8192
        except UnicodeEncodeError:
            accepted = False
        expected_receipt = {
            "schema": COMPACTION_RECEIPT_SCHEMA,
            "action_count_before": before_server,
            "action_count_after": before_server + 1,
            "counter_delta": expected_delta,
            "accepted": accepted,
        }
        if response["info"].get("counter_delta") != expected_delta:
            raise RuntimeError("compaction counter delta drifted")
        if not _strict_equal(response["info"].get("control_receipt"), expected_receipt):
            raise RuntimeError("compaction server receipt drifted")
        if "terminal_receipt" in response["info"]:
            raise RuntimeError("compaction must not create a submission receipt")
        self._server_action_count = response["info"]["counters"]["action_count"]
        self._last_counters = dict(response["info"]["counters"])
        self._policy_step_count += 1
        self._selected_policy_control = None
        self.info = response
        self._complete_pending_action()
        if response["done"]:
            return StepOutput(
                state=response["observation"],
                reward=float(response["reward"]),
                done=True,
                info=build_task_neutral_transition_info(
                    env_info=response["info"],
                    action_submission={
                        "raw_policy_output": summary,
                        "parser_status": "budget_terminal_compaction",
                    },
                    native_step_before=self._native_call_count,
                    native_step_after=self._native_call_count,
                    native_call_count_before=self._native_call_count,
                    native_call_count_after=self._native_call_count,
                    context_epoch_before=context_before,
                    context_epoch_after=self._context_epoch,
                    policy_step_before=policy_before,
                    policy_step_after=self._policy_step_count,
                    wrapper_evidence={
                        "event": "budget_terminal_compaction",
                        "server_action_count": self._server_action_count,
                    },
                ),
            )
        if not accepted:
            return StepOutput(
                state=response["observation"],
                reward=float(response["reward"]),
                done=False,
                info=build_task_neutral_transition_info(
                    env_info=response["info"],
                    action_submission={
                        "raw_policy_output": summary,
                        "parser_status": "control_rejected",
                    },
                    native_step_before=self._native_call_count,
                    native_step_after=self._native_call_count,
                    native_call_count_before=self._native_call_count,
                    native_call_count_after=self._native_call_count,
                    context_epoch_before=context_before,
                    context_epoch_after=self._context_epoch,
                    policy_step_before=policy_before,
                    policy_step_after=self._policy_step_count,
                    wrapper_evidence={
                        "event": "control_rejected",
                        "server_action_count": self._server_action_count,
                    },
                ),
            )
        replacement = deepcopy(framing)
        replacement.extend(
            [
                {"role": "assistant", "content": str(summary)},
                {"role": "user", "content": POLICY_CONTINUATION_MARKER},
            ]
        )
        self._context_epoch += 1
        return StepOutput(
            state=response["observation"],
            reward=float(response["reward"]),
            done=False,
            info=build_task_neutral_transition_info(
                env_info=response["info"],
                action_submission={
                    "raw_policy_output": summary,
                    "parser_status": "policy_context_compaction",
                },
                native_step_before=self._native_call_count,
                native_step_after=self._native_call_count,
                native_call_count_before=self._native_call_count,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE,
                    messages=replacement,
                ),
                wrapper_evidence={
                    "event": "context_compaction",
                    "server_action_count": self._server_action_count,
                },
            ),
        )

    def close(self) -> dict[str, Any]:
        response = self._request(
            "POST",
            "close",
            json={
                "id": self.env_id,
                "capability_token": self._capability_token,
            },
        )
        if not _strict_equal(response, {"closed": True}):
            raise RuntimeError("MLE-bench Lite close response drifted")
        return response

    def _validate_step_response(
        self,
        response: Mapping[str, Any],
        *,
        expected_before: Mapping[str, int] | None,
    ) -> None:
        if set(response) != {"observation", "reward", "done", "info"}:
            raise RuntimeError("MLE-bench Lite step response fields drifted")
        if (
            not isinstance(response["observation"], str)
            or type(response["done"]) is not bool
            or isinstance(response["reward"], bool)
            or not isinstance(response["reward"], (int, float))
            or not math.isfinite(float(response["reward"]))
            or float(response["reward"]) != 0.0
            or not isinstance(response["info"], Mapping)
        ):
            raise RuntimeError("MLE-bench Lite step response types drifted")
        counters = response["info"].get("counters")
        expected_counter_keys = {
            "action_count",
            "native_action_count",
            "execution_count",
            "grading_count",
            *RESOURCE_USAGE_KEYS,
        }
        if not isinstance(counters, Mapping) or set(counters) != expected_counter_keys:
            raise RuntimeError("MLE-bench Lite counter fields drifted")
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise RuntimeError("MLE-bench Lite counters are invalid")
        if counters["grading_count"] != 0:
            raise RuntimeError("MLE-bench Lite adapter must not invoke grading")
        allowed_info = {
            "action_kind",
            "counters",
            "counter_delta",
            "control_receipt",
            "external_memory_operation",
            "terminal_reason",
            "terminal_receipt",
        }
        if set(response["info"]) - allowed_info:
            raise RuntimeError("MLE-bench Lite step info fields drifted")
        memory_operation = response["info"].get("external_memory_operation")
        if memory_operation is not None and (
            self.mode != MODE_AMG_MEMORY
            or memory_operation not in {"read", "write", "read_write"}
        ):
            raise RuntimeError("MLE-bench Lite memory-operation receipt drifted")
        if _contains_forbidden_result(response["info"]):
            raise RuntimeError("MLE-bench Lite response exposed forbidden result data")
        if expected_before is None:
            if set(response["info"]) != {"counters"}:
                raise RuntimeError("MLE-bench Lite reset info fields drifted")
            if dict(counters) != _zero_counters():
                raise RuntimeError("MLE-bench Lite reset counters drifted")
            return
        delta = response["info"].get("counter_delta")
        if not isinstance(delta, Mapping) or set(delta) != expected_counter_keys:
            raise RuntimeError("MLE-bench Lite counter delta fields drifted")
        if (
            delta["action_count"] != 1
            or delta["grading_count"] != 0
            or any(type(value) is not int or value < 0 for value in delta.values())
        ):
            raise RuntimeError("MLE-bench Lite counter delta is invalid")
        expected_after = {
            key: expected_before[key] + delta[key] for key in expected_counter_keys
        }
        if dict(counters) != expected_after:
            raise RuntimeError("MLE-bench Lite cumulative counters drifted")

    def _validate_terminal_receipt(self, response: Mapping[str, Any]) -> None:
        receipt = response["info"].get("terminal_receipt")
        is_submission_handoff = (
            response["info"].get("terminal_reason") == "submission_handoff"
        )
        if receipt is None:
            if is_submission_handoff:
                raise RuntimeError("MLE-bench Lite submission receipt is missing")
            return
        if (
            not response["done"]
            or not is_submission_handoff
            or response["info"].get("action_kind") != "submit"
            or set(receipt)
            != {
                "competition_id",
                "submission_path",
                "submission_sha256",
            }
        ):
            raise RuntimeError("MLE-bench Lite terminal receipt fields drifted")
        if (
            receipt["competition_id"] != self._competition_id
            or receipt["submission_path"] != SUBMISSION_PATH
            or not _is_sha256(receipt["submission_sha256"])
        ):
            raise RuntimeError("MLE-bench Lite terminal receipt binding drifted")

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = self._requester(
            method,
            f"{self.env_server_base}/{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"MLE-bench Lite {method} /{path} failed with status "
                f"{response.status_code}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"MLE-bench Lite {method} /{path} returned a non-object"
            )
        return value

    def _request_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        canonical_payload = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":")
        )
        if self._pending_action_id is None:
            self._pending_action_id = str(uuid.uuid4())
            self._pending_action_payload = canonical_payload
        elif self._pending_action_payload != canonical_payload:
            raise RuntimeError(
                "a pending MLE-bench Lite action must be replayed exactly"
            )
        return self._request(
            "POST",
            "step",
            json={
                "id": self.env_id,
                "capability_token": self._capability_token,
                "action_id": self._pending_action_id,
                **dict(payload),
            },
        )

    def _complete_pending_action(self) -> None:
        self._pending_action_id = None
        self._pending_action_payload = None


def _copy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"policy message {index} is invalid")
        result.append({"role": role, "content": content})
    if not result:
        raise ValueError("policy context must not be empty")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resource_contract(
    *,
    max_actions: int,
    max_submission_bytes: int,
    max_shell_timeout_ms: int,
    episode_timeout_ms: int = 86_400_000,
    max_total_execution_ms: int = 72_000_000,
    cpu_limit_cores: int = 36,
    memory_limit_bytes: int = 440_000_000_000,
    pids_limit: int = 4096,
    writable_bytes_limit: int = 500_000_000_000,
    writable_inodes_limit: int = 2_000_000,
    gpu_count: int = 1,
) -> dict[str, Any]:
    return _validate_resource_contract(
        {
            "schema": RESOURCE_CONTRACT_SCHEMA,
            "max_actions": max_actions,
            "max_submission_bytes": max_submission_bytes,
            "max_shell_timeout_ms": max_shell_timeout_ms,
            "max_visible_output_bytes": 65_536,
            "episode_timeout_ms": episode_timeout_ms,
            "max_total_execution_ms": max_total_execution_ms,
            "cpu_limit_cores": cpu_limit_cores,
            "memory_limit_bytes": memory_limit_bytes,
            "pids_limit": pids_limit,
            "writable_bytes_limit": writable_bytes_limit,
            "writable_inodes_limit": writable_inodes_limit,
            "gpu_count": gpu_count,
            "max_step_response_ms": episode_timeout_ms + STEP_RESPONSE_SLACK_MS,
            "submission_path": SUBMISSION_PATH,
            "network_disabled": True,
            "read_only_public_data": True,
            "process_scope": "episode_cgroup_descendants",
            "cgroup_required": True,
            "isolated_process_group_required": True,
        }
    )


def _validate_resource_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_CONTRACT_FIELDS:
        raise ValueError("resource contract fields drifted")
    contract = dict(value)
    if type(contract["schema"]) is not str or (
        contract["schema"] != RESOURCE_CONTRACT_SCHEMA
    ):
        raise ValueError("resource contract schema drifted")
    for label in _RESOURCE_CONTRACT_NUMERIC_FIELDS:
        item = contract[label]
        if type(item) is not int or item <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if contract["max_shell_timeout_ms"] > contract["episode_timeout_ms"]:
        raise ValueError("max_shell_timeout_ms exceeds the episode deadline")
    if contract["max_total_execution_ms"] > contract["episode_timeout_ms"]:
        raise ValueError("max_total_execution_ms exceeds the episode deadline")
    if contract["max_submission_bytes"] > contract["writable_bytes_limit"]:
        raise ValueError("submission limit exceeds writable-byte budget")
    if contract["max_step_response_ms"] != (
        contract["episode_timeout_ms"] + STEP_RESPONSE_SLACK_MS
    ):
        raise ValueError("max_step_response_ms is not canonically derived")
    submission_path = contract["submission_path"]
    if (
        type(submission_path) is not str
        or not submission_path.startswith("/")
        or "\x00" in submission_path
    ):
        raise ValueError("submission_path must be absolute")
    for label in (
        "network_disabled",
        "read_only_public_data",
        "cgroup_required",
        "isolated_process_group_required",
    ):
        if type(contract[label]) is not bool or contract[label] is not True:
            raise ValueError("resource contract isolation fields drifted")
    if (
        type(contract["process_scope"]) is not str
        or contract["process_scope"] != "episode_cgroup_descendants"
    ):
        raise ValueError("resource contract isolation fields drifted")
    return contract


def _resource_contract_sha256(
    contract: Mapping[str, Any] | None = None,
    **legacy_limits: int,
) -> str:
    if contract is None:
        contract = _resource_contract(**legacy_limits)
    elif legacy_limits:
        raise ValueError("resource contract input is ambiguous")
    payload = json.dumps(
        _validate_resource_contract(contract),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _zero_counters() -> dict[str, int]:
    return {
        "action_count": 0,
        "native_action_count": 0,
        "execution_count": 0,
        "grading_count": 0,
        **{key: 0 for key in RESOURCE_USAGE_KEYS},
    }


def _is_capability_token(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contains_forbidden_result(value: Any) -> bool:
    forbidden = ("score", "grader", "private", "leaderboard", "valid")
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or _contains_forbidden_result(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_result(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in forbidden)
    return False


class MLEBenchLiteTask(BaseTask):
    env_client_cls = MLEBenchLiteEnvClient
    env_name = "MLE-bench Lite"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
