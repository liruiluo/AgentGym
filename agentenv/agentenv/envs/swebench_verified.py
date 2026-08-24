from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    POLICY_CONTINUATION_MARKER,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from agentenv_swebench_verified.environment import (
    EPISODE_SCHEMA,
    HORIZON_CONTRACT,
    NO_SUBMISSION_PREDICTION_CONTRACT,
    OBSERVATION_CONTRACT,
    RUN_CAPABILITY_CONTRACT,
    SUBMISSION_CONTRACT,
    require_run_capability,
)
from agentenv_swebench_verified.exporter import (
    MAX_MODEL_PATCH_BYTES,
    PATCH_EXPORT_CONTRACT,
    PREDICTION_SCHEMA_FIELDS,
)
from agentenv_swebench_verified.images import (
    IMAGE_MANIFEST_CONTRACT,
    PRODUCTION_IMAGE_PINS,
)
from agentenv_swebench_verified.protocol import (
    ARMS,
    DATASET_REPOSITORY,
    EVALUATION_MAX_POLICY_TURNS,
    FORBIDDEN_POLICY_FIELDS,
    MODEL_LABELS,
    POLICY_FIELDS,
    PRODUCTION_DATASET_PINS,
    require_arm,
    require_sha256,
)
from agentenv_swebench_verified.sandbox import SANDBOX_CONTRACT
from agentenv_swebench_verified.testspec import TESTSPEC_BINDING_CONTRACT


SBV_CONTEXT_COMPACTION_REQUEST = (
    "The conversation is nearing its context limit. Write the continuation "
    "state you want to retain after the earlier interaction is removed. Your "
    "response will be preserved verbatim and will not be sent to the "
    "environment. Include only information you choose to carry forward. Keep "
    "this response short: retain the immediate objective, decisive current "
    "state, next action, and the path/search key of any durable notes you "
    "already wrote. Do not claim that this response executed a shell command "
    "or changed a file."
)

SBV_BASE_POLICY_SYSTEM_PROMPT = (
    "You are a coding agent in one persistent /testbed repository. Repair the issue "
    "by inspecting, editing, and testing the repository. Every policy turn is exactly "
    "one action. Do not emit visible reasoning.\n\n"
    "For a shell action, start at byte zero and output exactly one line such as:\n"
    'shell_command {"command":"find . -maxdepth 2 -type f | head -80",'
    '"workdir":".","timeout_ms":120000}\n'
    "The command field is required; workdir and timeout_ms are optional. For a patch, "
    "start with the literal line apply_patch followed by one complete "
    "*** Begin Patch ... *** End Patch payload. Paths are relative to /testbed. "
    "Output only the action, with no XML, explanation, label, or Markdown fence. "
    "The workspace has no .git directory and persists for the whole task. After a "
    "source path changed and relevant tests ran, submit with exactly:\n"
    'shell_command {"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",'
    '"workdir":"."}\n'
    "The successful command must print that sentinel as its first stdout line. Plain "
    "text is invalid. Reaching the 30-turn limit without the sentinel ends the episode "
    "without exporting the workspace. Official SWE-bench grading is external; no "
    "hidden scorer is available in this environment."
)

SBV_MEMORY_ADDENDUM = (
    "\n\n# AMG durable task memory and context compaction\n"
    "This arm mounts a separate clean per-task durable-note store at "
    "/run/amg_memory. Use the existing shell_command action to read and write files "
    "there, for example `printf '%s' '...' > /run/amg_memory/debugging.md` and "
    "`cat /run/amg_memory/debugging.md`. This path is outside /testbed and never "
    "enters the submitted repository patch. Keep high-value debugging evidence there "
    "when it may need to "
    "survive context compaction: hypotheses, commands/tests already tried, exact "
    "observations, failed approaches, verified partial fixes, and next checks. Notes "
    "persist across policy-authored context compactions, start empty on the next task, "
    "and are available only in this arm. "
    "A compaction response is only a short working-state handoff; it does not execute "
    "a tool and has no separate task reward. After compaction, read the note again "
    "before relying on it."
)
SBV_COMPACTION_ARMS = frozenset({"amg_compaction_only", "amg_memory"})


def _validate_step_response(
    response: Mapping[str, Any],
    *,
    expected_done: bool | None = None,
) -> tuple[str, float, bool, Mapping[str, Any]]:
    required_fields = {"observation", "reward", "done", "info"}
    if set(response) not in (required_fields, required_fields | {"state"}):
        raise RuntimeError("Verified step response fields drifted")
    observation = response["observation"]
    reward = response["reward"]
    done = response["done"]
    info = response["info"]
    if (
        not isinstance(observation, str)
        or isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or float(reward) != 0.0
        or type(done) is not bool
        or not isinstance(info, Mapping)
    ):
        raise RuntimeError("Verified step response types drifted")
    if "state" in response and (
        not isinstance(response["state"], str)
        or response["state"] != observation
    ):
        raise RuntimeError("Verified step response state drifted")
    if expected_done is not None and done is not expected_done:
        raise RuntimeError("Verified step terminal state drifted")
    return observation, float(reward), done, info


class SwebenchVerifiedEnvClient(BaseEnvClient):
    def __init__(
        self,
        env_server_base: str,
        *,
        arm: str,
        run_id: str,
        run_capability: str,
        image_manifest_sha256: str,
        data_len: int | None = None,
        timeout: int = 900,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.env_server_base = env_server_base.rstrip("/")
        self.arm = require_arm(arm)
        self.image_manifest_sha256 = require_sha256(
            image_manifest_sha256,
            "image_manifest_sha256",
        )
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be non-empty text")
        self.run_id = run_id
        self._run_capability = require_run_capability(run_capability)
        self.timeout = timeout
        self.metadata = self._request("GET", "metadata")
        self._validate_metadata(self.metadata)
        task_count = int(self.metadata["task_count"])
        if data_len is not None and not 0 < int(data_len) <= task_count:
            raise ValueError(
                "SWE-bench Verified data_len must fit the available task panel"
            )
        self.data_len = task_count if data_len is None else int(data_len)
        self.system_prompt = SBV_BASE_POLICY_SYSTEM_PROMPT + (
            SBV_MEMORY_ADDENDUM if self.arm == "amg_memory" else ""
        )
        self.conversation_start = (
            ConversationMessage(
                {"from": "human", "loss": None, "value": self.system_prompt}
            ),
            ConversationMessage(
                {"from": "gpt", "loss": False, "value": "Understood."}
            ),
        )
        created = self._request(
            "POST",
            "create",
            json={"arm": self.arm, "run_id": self.run_id},
            headers=self._bearer_headers(self._run_capability),
        )
        self.env_id = int(created["id"])
        capability = created.get("capability")
        if not isinstance(capability, str) or not capability:
            raise RuntimeError("Verified endpoint returned no slot capability")
        self._slot_capability = capability
        self.info = dict(created)
        self.info.pop("capability")
        self._reset_policy_state()

    def _validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        exact = {
            "schema": EPISODE_SCHEMA,
            "task_count": 500,
            "full_benchmark_task_count": 500,
            "evaluation_max_policy_turns": EVALUATION_MAX_POLICY_TURNS,
            "max_native_actions": EVALUATION_MAX_POLICY_TURNS,
            "compaction_consumes_policy_turn": True,
            "compaction_consumes_native_call": False,
            "submission_contract": SUBMISSION_CONTRACT,
            "horizon_contract": HORIZON_CONTRACT,
            "no_submission_prediction_contract": (
                NO_SUBMISSION_PREDICTION_CONTRACT
            ),
            "run_capability_contract": RUN_CAPABILITY_CONTRACT,
            "reward_contract": "external_official_grading_only",
            "tool_contract": "codex_shell_command_apply_patch_v1",
            "tool_serialization": "qwen35_native_single_function_v1",
            "observation_contract": OBSERVATION_CONTRACT,
            "patch_export_contract": PATCH_EXPORT_CONTRACT,
            "max_model_patch_bytes": MAX_MODEL_PATCH_BYTES,
            "testspec_contract": TESTSPEC_BINDING_CONTRACT,
            "sandbox_contract": SANDBOX_CONTRACT,
            "official_grading_inside_adapter": False,
        }
        for key, expected in exact.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Verified endpoint metadata mismatch for {key}: "
                    f"expected {expected!r}, got {metadata.get(key)!r}"
                )
        resource_contract = {
            "max_observation_bytes": 6144,
            "stdout_bytes": 3072,
            "stderr_bytes": 3072,
            "default_timeout_ms": 120_000,
            "max_timeout_ms": 120_000,
            "thinking_enabled": False,
            "reasoning_enabled": False,
        }
        for key, expected in resource_contract.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Verified endpoint metadata mismatch for {key}: "
                    f"expected {expected!r}, got {metadata.get(key)!r}"
                )
        if metadata.get("supported_arms") != list(ARMS):
            raise RuntimeError("Verified endpoint arm contract drifted")
        if metadata.get("model_labels") != MODEL_LABELS:
            raise RuntimeError("Verified endpoint model-label contract drifted")
        if metadata.get("policy_visible_fields") != list(POLICY_FIELDS):
            raise RuntimeError("Verified endpoint policy projection drifted")
        if metadata.get("denied_grader_fields") != sorted(
            FORBIDDEN_POLICY_FIELDS
        ):
            raise RuntimeError("Verified endpoint grader-field denial drifted")
        for key in ("max_observation_bytes", "max_observation_tokens"):
            value = metadata.get(key)
            if type(value) is not int or value <= 0:
                raise RuntimeError(f"Verified endpoint {key} must be positive")
        if metadata["max_observation_bytes"] >= metadata["max_observation_tokens"]:
            raise RuntimeError("Verified endpoint observation budget drifted")
        dataset = metadata.get("dataset")
        if not isinstance(dataset, Mapping):
            raise RuntimeError("Verified endpoint has no dataset provenance")
        dataset_expected = {
            "repository": DATASET_REPOSITORY,
            "revision": PRODUCTION_DATASET_PINS.revision,
            "split": PRODUCTION_DATASET_PINS.split,
            "row_count": PRODUCTION_DATASET_PINS.row_count,
            "canonical_jsonl_sha256": (
                PRODUCTION_DATASET_PINS.canonical_jsonl_sha256
            ),
            "id_ledger_sha256": PRODUCTION_DATASET_PINS.id_ledger_sha256,
        }
        for key, expected in dataset_expected.items():
            if dataset.get(key) != expected:
                raise RuntimeError(f"Verified endpoint dataset {key} drifted")

        image_manifest = metadata.get("image_manifest")
        if not isinstance(image_manifest, Mapping):
            raise RuntimeError("Verified endpoint has no image manifest identity")
        image_expected = {
            "contract": IMAGE_MANIFEST_CONTRACT,
            "tag_count": PRODUCTION_IMAGE_PINS.tag_count,
            "tag_ledger_sha256": PRODUCTION_IMAGE_PINS.tag_ledger_sha256,
            "manifest_sha256": self.image_manifest_sha256,
        }
        for key, expected in image_expected.items():
            if image_manifest.get(key) != expected:
                raise RuntimeError(f"Verified endpoint image manifest {key} drifted")
        unique_digest_count = image_manifest.get("unique_digest_count")
        if (
            type(unique_digest_count) is not int
            or not 1 <= unique_digest_count <= PRODUCTION_IMAGE_PINS.tag_count
        ):
            raise RuntimeError(
                "Verified endpoint image manifest digest count drifted"
            )

        prediction = metadata.get("prediction_contract")
        if not isinstance(prediction, Mapping):
            raise RuntimeError("Verified endpoint has no prediction contract")
        prediction_expected = {
            "schema_fields": list(PREDICTION_SCHEMA_FIELDS),
            "task_count": PRODUCTION_DATASET_PINS.row_count,
            "instance_id_ledger_sha256": (
                PRODUCTION_DATASET_PINS.id_ledger_sha256
            ),
            "model_labels": MODEL_LABELS,
        }
        for key, expected in prediction_expected.items():
            if prediction.get(key) != expected:
                raise RuntimeError(f"Verified endpoint prediction {key} drifted")

    def _reset_policy_state(self) -> None:
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

    def observe(self) -> str:
        observation = self.info.get("observation")
        if not isinstance(observation, str):
            raise RuntimeError("Verified observation must be text")
        return observation

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}]

    def _compaction_enabled(self) -> bool:
        return self.arm in SBV_COMPACTION_ARMS

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        normalized = copy_policy_messages(messages)
        observation = self.observe()
        if (
            not normalized
            or normalized[-1]["role"] != "user"
            or normalized[-1]["content"] != observation
        ):
            raise ValueError(
                "Verified initial policy context must end with the observation"
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
        normalized = copy_policy_messages(messages)
        if initial:
            expected = self.policy_framing() + [
                {"role": "user", "content": self.observe()}
            ]
            if normalized != expected:
                raise ValueError("Verified initial policy framing changed")
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if not self._compaction_enabled() or not self._policy_context_bound:
            return None
        return SBV_CONTEXT_COMPACTION_REQUEST

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        self._selected_policy_control = None
        if not self._compaction_enabled() or not self._policy_context_bound:
            return None
        if pressure is None:
            raise RuntimeError(
                "AMG context compaction requires task-neutral token pressure"
            )
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "context reached the prompt cap before compaction could be sampled"
            )
        request_tokens = (
            pressure.candidate_prompt_tokens - pressure.action_prompt_tokens
        )
        if request_tokens <= 0:
            raise RuntimeError("compaction request must extend the action prompt")
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
        return SBV_CONTEXT_COMPACTION_REQUEST

    def step(self, action: str) -> StepOutput:
        if self._policy_step_count >= EVALUATION_MAX_POLICY_TURNS:
            raise RuntimeError(
                "the unified 30-turn policy budget is exhausted; finalize horizon"
            )
        if self._selected_policy_control == "context_compaction":
            return self._complete_context_compaction(action)
        return self._step_native_action(action)

    def _step_native_action(self, action: str) -> StepOutput:
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        response = self._request(
            "POST",
            "step",
            json=self._slot_transport(action=action),
            headers=self._bearer_headers(self._slot_capability),
        )
        state, reward, done, env_info = _validate_step_response(response)
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        server_step = env_info.get("step")
        if server_step is not None and (
            type(server_step) is not int
            or server_step != self._native_call_count
        ):
            raise RuntimeError("Verified native action counter drifted")
        return StepOutput(
            state=state,
            reward=reward,
            done=done,
            info=build_task_neutral_transition_info(
                env_info=env_info,
                action_submission={"raw_policy_output": action},
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=self._session_epoch,
                session_epoch_after=self._session_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "native_action",
                    "workspace_continuity_id": self.env_id,
                    "arm": self.arm,
                },
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        framing = self._immutable_policy_context
        if framing is None or not self._compaction_enabled():
            raise RuntimeError("AMG compaction lost its immutable task framing")
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
        prior_env_info = self.info.get("info", {})
        if not isinstance(prior_env_info, Mapping):
            raise RuntimeError("Verified step info must be an object")
        env_info = dict(prior_env_info)
        env_info.pop("external_memory_operation", None)
        return StepOutput(
            state=self.observe(),
            reward=0.0,
            done=False,
            info=build_task_neutral_transition_info(
                env_info=env_info,
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
                session_epoch_before=self._session_epoch,
                session_epoch_after=self._session_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE, messages=replacement
                ),
                wrapper_evidence={
                    "event": "context_compaction",
                    "workspace_continuity_id": self.env_id,
                    "arm": self.arm,
                },
            ),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._request(
            "POST",
            "reset",
            json=self._slot_transport(data_idx=idx),
            headers=self._bearer_headers(self._slot_capability),
        )
        _validate_step_response(response, expected_done=False)
        self.info = response
        self._reset_policy_state()
        return response

    def finalize_policy_horizon(self) -> StepOutput:
        response = self._request(
            "POST",
            "horizon",
            json=self._slot_transport(),
            headers=self._bearer_headers(self._slot_capability),
        )
        state, reward, done, env_info = _validate_step_response(
            response,
            expected_done=True,
        )
        self.info = response
        return StepOutput(
            state=state,
            reward=reward,
            done=done,
            info=build_task_neutral_transition_info(
                env_info=env_info,
                action_submission={"control_action": "unified_policy_horizon"},
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
                    "event": "horizon_no_submission",
                    "workspace_continuity_id": self.env_id,
                    "arm": self.arm,
                },
            ),
        )

    def record_no_submission(self) -> dict[str, str]:
        row = self._request(
            "POST",
            "no-submission",
            json=self._slot_transport(),
            headers=self._bearer_headers(self._slot_capability),
        )
        return self._validate_prediction_row(row)

    def prediction(self) -> dict[str, str]:
        row = self._request(
            "GET",
            "prediction",
            params=self._slot_transport(),
            headers=self._bearer_headers(self._slot_capability),
        )
        return self._validate_prediction_row(row)

    def _validate_prediction_row(self, row: Mapping[str, Any]) -> dict[str, str]:
        expected = {"instance_id", "model_name_or_path", "model_patch"}
        if (
            set(row) != expected
            or row.get("model_name_or_path") != MODEL_LABELS[self.arm]
        ):
            raise RuntimeError("Verified prediction row contract drifted")
        if not all(isinstance(row.get(key), str) for key in expected):
            raise RuntimeError("Verified prediction row values must be text")
        return dict(row)

    def assemble_predictions(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "predictions/assemble",
            json=self._slot_transport(arm=self.arm, run_id=self.run_id),
            headers=self._bearer_headers(self._slot_capability),
        )

    def close(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "close",
            json=self._slot_transport(),
            headers=self._bearer_headers(self._slot_capability),
        )

    def _slot_transport(self, **fields: Any) -> dict[str, Any]:
        return {
            "id": self.env_id,
            **fields,
        }

    @staticmethod
    def _bearer_headers(capability: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {capability}"}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.env_server_base}/{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"SWE-bench Verified {method} /{path} failed with "
                f"status {response.status_code}: {response.text[-1000:]}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"SWE-bench Verified {method} /{path} returned a non-object"
            )
        return value


def copy_policy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has invalid role")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return normalized


class SwebenchVerifiedTask(BaseTask):
    env_client_cls = SwebenchVerifiedEnvClient
    env_name = "SWE-bench Verified"
