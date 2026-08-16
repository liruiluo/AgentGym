from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import requests

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    POLICY_CONTINUATION_MARKER,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)

GAIA_TEXT_PROTOCOL_ID = (
    "gaia_text_2023_validation_no_attachment@682dd723ee1e1697e00360edccf2366dc8418dd9"
)
GAIA_TEXT_DATASET_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"
GAIA_TEXT_MANIFEST_SHA256 = (
    "06f6da09978555c39f70f2794499012a1d07eb391e01a0f3d498957b09a1fda7"
)
GAIA_TEXT_TASK_IDS_SHA256 = (
    "57e76233b8b12d8d9ea18639d1d52616449cf521559cd9d103c76ff399a842ad"
)
GAIA_TEXT_ARMS = frozenset({"native", "amg_memory"})
GAIA_TEXT_PAIRED_RUNTIME_SCHEMA = "gaia_text_paired_runtime_contract_v1"

GAIA_TEXT_DOMAIN_PROMPT = (
    "You are a research agent answering one GAIA-Text question. Search with "
    '<tool_call>{"name":"search","arguments":{"query":["..."]}}</tool_call>; '
    "visit an opaque result URL with "
    '<tool_call>{"name":"visit","arguments":{"url":"...","goal":"...",'
    '"page":1}}</tool_call>. A visit returns one bounded page; follow next_page '
    "with the same URL and goal when more evidence is needed. Submit exactly the "
    "final answer as <answer>...</answer>. Emit exactly one action per policy turn."
)
GAIA_TEXT_MEMORY_AFFORDANCE = (
    "\n\nThis arm also provides an empty private filesystem workspace that lasts only "
    "for the current task. Use shell_command followed by one JSON object such as "
    'shell_command {"command":"...","workdir":"."}, or apply_patch followed on '
    "the next line by one patch. A workspace action consumes an ordinary policy turn "
    "and has no separate reward. The workspace persists if the client asks for a "
    "policy-authored context compaction. Keep detailed evidence in ordinary files; a "
    "compaction response is only a short continuation state and is not a tool action."
)
GAIA_TEXT_CONTEXT_COMPACTION_REQUEST = (
    "The research conversation is nearing its context limit. Write the short "
    "continuation state to retain after earlier messages are removed. This response "
    "will be preserved verbatim and will not call search, visit, or workspace tools. "
    "Include the unresolved question, useful evidence, and paths to workspace notes "
    "that should be read later."
)

_PRODUCTION_EXPECTATION = {
    "protocol_id": GAIA_TEXT_PROTOCOL_ID,
    "dataset_revision": GAIA_TEXT_DATASET_REVISION,
    "split": "validation",
    "task_count": 127,
    "level_counts": {"1": 42, "2": 66, "3": 19},
    "manifest_sha256": GAIA_TEXT_MANIFEST_SHA256,
    "task_ids_sha256": GAIA_TEXT_TASK_IDS_SHA256,
}
_PAIRED_RUNTIME_KEYS = frozenset(
    {
        "schema",
        "protocol_id",
        "dataset_revision",
        "split",
        "task_count",
        "level_counts",
        "manifest_sha256",
        "task_ids_sha256",
        "questions_sha256",
        "backend_contract",
        "backend_asset_sha256",
        "visit_page_chars",
        "max_policy_steps",
        "domain_action_contract",
        "answer_extraction_contract",
        "reward_contract",
        "submission_contract",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PAIRED_STATIC_CONTRACTS = {
    "domain_action_contract": "shared_search_visit_answer_v1",
    "answer_extraction_contract": "single_trimmed_answer_tag_v1",
    "reward_contract": "external_official_scoring_zero_online_reward_v1",
    "submission_contract": "gaia_task_id_model_answer_jsonl_v1",
}


def _validate_step_response(
    response: Mapping[str, Any],
    *,
    expected_done: bool | None = None,
) -> tuple[str, float, bool, Mapping[str, Any]]:
    if set(response) != {"observation", "reward", "done", "info"}:
        raise RuntimeError("GAIA-Text step response fields drifted")
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
        raise RuntimeError("GAIA-Text step response types drifted")
    if expected_done is not None and done is not expected_done:
        raise RuntimeError("GAIA-Text step terminal state drifted")
    return observation, float(reward), done, info


class GaiaTextEnvClient(BaseEnvClient):
    """AgentGym client for both arms of the thin GAIA-Text adapter."""

    def __init__(
        self,
        env_server_base: str,
        data_len: int | None = None,
        *args,
        arm: str,
        timeout: int = 900,
        expected_protocol: Mapping[str, Any] | None = None,
        expected_paired_runtime_sha256: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if arm not in GAIA_TEXT_ARMS:
            raise ValueError(f"GAIA-Text arm must be one of {sorted(GAIA_TEXT_ARMS)}")
        self.arm = arm
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        metadata = self._request("GET", "metadata")
        expectation = (
            _PRODUCTION_EXPECTATION if expected_protocol is None else expected_protocol
        )
        paired_contract, paired_digest = self._validate_metadata(
            metadata,
            expectation,
            expected_paired_runtime_sha256=expected_paired_runtime_sha256,
        )
        task_count = int(metadata["task_count"])
        if data_len is not None and (
            isinstance(data_len, bool)
            or not isinstance(data_len, int)
            or data_len <= 0
            or data_len > task_count
        ):
            raise ValueError(
                f"GAIA-Text data_len must be within 1..{task_count}, got {data_len!r}"
            )
        self.data_len = task_count if data_len is None else data_len
        self.metadata = metadata
        self.paired_runtime_contract = paired_contract
        self.paired_runtime_contract_sha256 = paired_digest
        self.max_policy_steps = int(metadata["max_policy_steps"])
        self._system_prompt = GAIA_TEXT_DOMAIN_PROMPT + (
            GAIA_TEXT_MEMORY_AFFORDANCE if arm == "amg_memory" else ""
        )
        self.conversation_start = (
            ConversationMessage(
                {"from": "human", "loss": None, "value": self._system_prompt}
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Understood."}),
        )
        created = self._request("POST", "create", json={})
        self.env_id = int(created["id"])
        self.info = created
        self.compaction_request = GAIA_TEXT_CONTEXT_COMPACTION_REQUEST
        self._reset_policy_transition_state()

    def _validate_metadata(
        self,
        metadata: Mapping[str, Any],
        expected_protocol: Mapping[str, Any],
        *,
        expected_paired_runtime_sha256: str | None,
    ) -> tuple[dict[str, Any], str]:
        if metadata.get("domain_id") != "gaia_text":
            raise RuntimeError("GAIA-Text endpoint reports the wrong domain")
        if metadata.get("arm") != self.arm:
            raise RuntimeError(
                f"GAIA-Text endpoint arm mismatch: expected {self.arm!r}, "
                f"got {metadata.get('arm')!r}"
            )
        for key in (
            "protocol_id",
            "dataset_revision",
            "split",
            "task_count",
            "level_counts",
            "manifest_sha256",
            "task_ids_sha256",
        ):
            if key not in expected_protocol:
                raise ValueError(f"expected_protocol is missing {key}")
            if metadata.get(key) != expected_protocol[key]:
                raise RuntimeError(
                    f"GAIA-Text endpoint {key} mismatch: expected "
                    f"{expected_protocol[key]!r}, got {metadata.get(key)!r}"
                )
        if (
            "questions_sha256" in expected_protocol
            and metadata.get("questions_sha256")
            != expected_protocol["questions_sha256"]
        ):
            raise RuntimeError("GAIA-Text endpoint questions_sha256 mismatch")
        memory = self.arm == "amg_memory"
        expected_compaction = (
            "task_neutral_client_replace_messages_v1" if memory else "disabled"
        )
        if metadata.get("compaction_contract") != expected_compaction:
            raise RuntimeError("GAIA-Text endpoint compaction contract mismatch")
        if metadata.get("workspace_available") is not memory:
            raise RuntimeError("GAIA-Text endpoint workspace availability mismatch")
        expected_workspace = (
            "codex_shell_command_apply_patch_v1" if memory else "disabled"
        )
        if metadata.get("workspace_contract") != expected_workspace:
            raise RuntimeError("GAIA-Text endpoint workspace contract mismatch")
        if metadata.get("domain_action_contract") != "shared_search_visit_answer_v1":
            raise RuntimeError("GAIA-Text endpoint domain-action contract mismatch")
        if (
            metadata.get("reward_contract")
            != "external_official_scoring_zero_online_reward_v1"
        ):
            raise RuntimeError("GAIA-Text endpoint reward contract mismatch")
        if metadata.get("submission_contract") != "gaia_task_id_model_answer_jsonl_v1":
            raise RuntimeError("GAIA-Text endpoint submission contract mismatch")
        if (
            metadata.get("compaction_calls_server") is not False
            or metadata.get("compaction_calls_backend") is not False
        ):
            raise RuntimeError(
                "GAIA-Text endpoint must keep compaction client-owned and backend-free"
            )
        paired_contract, paired_digest = _validate_paired_runtime_metadata(metadata)
        if expected_paired_runtime_sha256 is not None:
            if (
                not isinstance(expected_paired_runtime_sha256, str)
                or _SHA256_RE.fullmatch(expected_paired_runtime_sha256) is None
            ):
                raise ValueError(
                    "expected_paired_runtime_sha256 must be a lowercase SHA-256 digest"
                )
            if paired_digest != expected_paired_runtime_sha256:
                raise RuntimeError(
                    "GAIA-Text paired-runtime digest mismatch: "
                    f"expected {expected_paired_runtime_sha256}, got {paired_digest}"
                )
        return paired_contract, paired_digest

    def _reset_policy_transition_state(self) -> None:
        self._policy_step_count = 0
        self._native_call_count = 0
        self._context_epoch = 0
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None

    @property
    def context_epoch(self) -> int:
        return self._context_epoch

    @property
    def sample_excluded(self) -> bool:
        return bool(self.info.get("info", {}).get("sample_excluded", False))

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        observation = self.info.get("observation")
        if not isinstance(observation, str):
            raise RuntimeError("GAIA-Text observation must be text")
        return observation

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self._system_prompt}]

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = _copy_policy_messages(messages)
        observation = self.observe()
        if (
            not normalized
            or normalized[-1]["role"] != "user"
            or normalized[-1]["content"] != observation
        ):
            raise ValueError(
                "GAIA-Text initial policy context must end with the current observation"
            )
        return self.policy_framing() + [{"role": "user", "content": observation}]

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = _copy_policy_messages(messages)
        if initial:
            expected = self.policy_framing() + [
                {"role": "user", "content": self.observe()}
            ]
            if normalized != expected:
                raise ValueError(
                    "GAIA-Text initial policy context differs from its arm framing"
                )
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if self.arm != "amg_memory" or not self._policy_context_bound:
            return None
        return self.compaction_request

    def prepare_policy_turn(self, pressure: PolicyContextPressure | None) -> str | None:
        self._selected_policy_control = None
        if self.arm != "amg_memory" or not self._policy_context_bound:
            return None
        if pressure is None:
            raise RuntimeError(
                "GAIA-Text memory compaction requires task-neutral token pressure"
            )
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "GAIA-Text context reached the prompt cap before a trainable "
                "compaction could be sampled"
            )
        request_tokens = (
            pressure.candidate_prompt_tokens - pressure.action_prompt_tokens
        )
        if request_tokens <= 0:
            raise RuntimeError("GAIA-Text compaction request must extend the prompt")
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
        return self.compaction_request

    def step(self, action: str) -> StepOutput:
        if self._selected_policy_control == "context_compaction":
            return self._complete_context_compaction(action)
        return self._step_server_action(action)

    def _step_server_action(self, action: str) -> StepOutput:
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        response = self._request(
            "POST", "step", json={"id": self.env_id, "action": action}
        )
        state, reward, done, response_info = _validate_step_response(response)
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        server_step = response_info.get("step")
        if server_step is not None and (
            type(server_step) is not int
            or server_step != self._native_call_count
        ):
            raise RuntimeError(
                "GAIA-Text native step counter drifted from client calls"
            )
        action_submission = response_info.get("action_submission")
        if not isinstance(action_submission, Mapping):
            action_submission = {"raw_policy_output": action}
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
                wrapper_evidence={
                    "event": "native_action",
                    "server_wrapper_evidence": dict(
                        response_info.get("wrapper_evidence", {})
                    ),
                },
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        if self.arm != "amg_memory":
            raise RuntimeError("native GAIA-Text cannot execute context compaction")
        framing = self._immutable_policy_context
        if framing is None:
            raise RuntimeError("GAIA-Text compaction lost its immutable task framing")
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
            state=self.observe(),
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
                    "native_environment_call_count": 0,
                },
            ),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._request(
            "POST", "reset", json={"id": self.env_id, "data_idx": idx}
        )
        _validate_step_response(response, expected_done=False)
        self.info = response
        self._reset_policy_transition_state()
        return response

    def finalize_policy_horizon(self) -> StepOutput:
        response = self._request("POST", "horizon", json={"id": self.env_id})
        state, reward, done, response_info = _validate_step_response(
            response,
            expected_done=True,
        )
        self.info = response
        return StepOutput(
            state=state,
            reward=reward,
            done=done,
            info=build_task_neutral_transition_info(
                env_info=response_info,
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
                wrapper_evidence={"event": "horizon_finalization"},
            ),
        )

    def close(self) -> bool:
        value = self._request_json("POST", "close", json={"id": self.env_id})
        if value is not True:
            raise requests.RequestException("GAIA-Text POST /close did not return true")
        return True

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        value = self._request_json(method, path, **kwargs)
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"GAIA-Text {method} /{path} returned a non-object response"
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
                f"GAIA-Text {method} /{path} failed: "
                f"status={response.status_code} body={response.text[-1000:]}"
            )
        return response.json()


def _validate_paired_runtime_metadata(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    raw_contract = metadata.get("paired_runtime_contract")
    if not isinstance(raw_contract, Mapping) or set(raw_contract) != set(
        _PAIRED_RUNTIME_KEYS
    ):
        raise RuntimeError("GAIA-Text paired-runtime contract schema mismatch")
    contract = deepcopy(dict(raw_contract))
    if contract["schema"] != GAIA_TEXT_PAIRED_RUNTIME_SCHEMA:
        raise RuntimeError("GAIA-Text paired-runtime contract version mismatch")

    for key in (
        "protocol_id",
        "dataset_revision",
        "split",
        "task_count",
        "level_counts",
        "manifest_sha256",
        "task_ids_sha256",
        "questions_sha256",
    ):
        if contract[key] != metadata.get(key):
            raise RuntimeError(
                f"GAIA-Text paired-runtime {key} disagrees with public metadata"
            )
    for key in (
        "manifest_sha256",
        "task_ids_sha256",
        "questions_sha256",
        "backend_asset_sha256",
    ):
        if (
            not isinstance(contract[key], str)
            or _SHA256_RE.fullmatch(contract[key]) is None
        ):
            raise RuntimeError(f"GAIA-Text paired-runtime {key} is not a SHA-256")

    backend = metadata.get("backend")
    if not isinstance(backend, Mapping):
        raise TypeError("GAIA-Text endpoint backend metadata must be an object")
    backend_expectations = {
        "backend_contract": backend.get("backend_contract"),
        "backend_asset_sha256": backend.get("asset_sha256"),
        "visit_page_chars": backend.get("page_chars"),
    }
    for key, expected in backend_expectations.items():
        if contract[key] != expected:
            raise RuntimeError(
                f"GAIA-Text paired-runtime {key} disagrees with backend metadata"
            )
    for key in ("task_count", "visit_page_chars", "max_policy_steps"):
        if type(contract[key]) is not int or contract[key] <= 0:
            raise RuntimeError(
                f"GAIA-Text paired-runtime {key} must be a positive integer"
            )
    if contract["max_policy_steps"] != metadata.get("max_policy_steps"):
        raise RuntimeError(
            "GAIA-Text paired-runtime max_policy_steps disagrees with metadata"
        )
    for key, expected in _PAIRED_STATIC_CONTRACTS.items():
        if contract[key] != expected or metadata.get(key) != expected:
            raise RuntimeError(f"GAIA-Text paired-runtime {key} mismatch")

    observed_digest = metadata.get("paired_runtime_contract_sha256")
    if (
        not isinstance(observed_digest, str)
        or _SHA256_RE.fullmatch(observed_digest) is None
    ):
        raise RuntimeError(
            "GAIA-Text paired-runtime canonical digest must be a lowercase SHA-256"
        )
    try:
        canonical = json.dumps(
            contract,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "GAIA-Text paired-runtime contract is not canonical JSON data"
        ) from exc
    computed_digest = hashlib.sha256(canonical).hexdigest()
    if observed_digest != computed_digest:
        raise RuntimeError(
            "GAIA-Text paired-runtime canonical digest does not match its contract"
        )
    return contract, observed_digest


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
