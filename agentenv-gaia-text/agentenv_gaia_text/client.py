from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from agentenv.envs import gaia_text as _shared

from .contracts import EvaluationArm

GAIA_TEXT_ARMS = frozenset(arm.value for arm in EvaluationArm)
GAIA_TEXT_DOMAIN_PROMPT = _shared.GAIA_TEXT_DOMAIN_PROMPT
GAIA_TEXT_MEMORY_AFFORDANCE = _shared.GAIA_TEXT_MEMORY_AFFORDANCE
GAIA_TEXT_CONTEXT_COMPACTION_REQUEST = (
    "The research conversation is nearing its context limit. Write the short "
    "continuation state to retain after earlier messages are removed. This response "
    "will be preserved verbatim and will not call environment tools. Include the "
    "unresolved question and useful evidence needed to continue."
)
GAIA_TEXT_POLICY_CONTINUATION_MARKER = (
    "Continue the same task from this retained state."
)

_COMPACTION_ONLY = EvaluationArm.AMG_COMPACTION_ONLY.value
_MEMORY = EvaluationArm.AMG_MEMORY.value


class GaiaTextEnvClient(_shared.GaiaTextEnvClient):
    """Triad client with one task-neutral path for both compacting arms."""

    def __init__(
        self,
        env_server_base: str,
        data_len: int | None = None,
        *args: Any,
        arm: str,
        timeout: int = 900,
        expected_protocol: Mapping[str, Any] | None = None,
        expected_paired_runtime_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        BaseEnvClient.__init__(self, *args, **kwargs)
        if arm not in GAIA_TEXT_ARMS:
            raise ValueError(f"GAIA-Text arm must be one of {sorted(GAIA_TEXT_ARMS)}")
        self.arm = arm
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        metadata = self._request("GET", "metadata")
        expectation = (
            _shared._PRODUCTION_EXPECTATION
            if expected_protocol is None
            else expected_protocol
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
            GAIA_TEXT_MEMORY_AFFORDANCE if arm == _MEMORY else ""
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
        self._validate_capability_metadata(metadata)
        if self.arm != _COMPACTION_ONLY:
            return super()._validate_metadata(
                metadata,
                expected_protocol,
                expected_paired_runtime_sha256=expected_paired_runtime_sha256,
            )

        proxy = deepcopy(dict(metadata))
        proxy["arm"] = _MEMORY
        proxy["workspace_available"] = True
        proxy["workspace_contract"] = "codex_shell_command_apply_patch_v1"
        return self._run_memory_compaction_path(
            super()._validate_metadata,
            proxy,
            expected_protocol,
            expected_paired_runtime_sha256=expected_paired_runtime_sha256,
        )

    def _validate_capability_metadata(self, metadata: Mapping[str, Any]) -> None:
        if metadata.get("arm") != self.arm:
            raise RuntimeError(
                f"GAIA-Text endpoint arm mismatch: expected {self.arm!r}, "
                f"got {metadata.get('arm')!r}"
            )
        compaction_available = self.arm != EvaluationArm.NATIVE.value
        memory_available = self.arm == _MEMORY
        expected_compaction_contract = (
            "task_neutral_client_replace_messages_v1"
            if compaction_available
            else "disabled"
        )
        expected_workspace_contract = (
            "codex_shell_command_apply_patch_v1" if memory_available else "disabled"
        )
        expected_workspace_lifetime = (
            "clean_per_task" if memory_available else "none"
        )
        if metadata.get("compaction_available") is not compaction_available:
            raise RuntimeError("GAIA-Text endpoint compaction availability mismatch")
        if metadata.get("compaction_contract") != expected_compaction_contract:
            raise RuntimeError("GAIA-Text endpoint compaction contract mismatch")
        if metadata.get("workspace_available") is not memory_available:
            raise RuntimeError("GAIA-Text endpoint workspace availability mismatch")
        if metadata.get("workspace_contract") != expected_workspace_contract:
            raise RuntimeError("GAIA-Text endpoint workspace contract mismatch")
        if metadata.get("workspace_lifetime") != expected_workspace_lifetime:
            raise RuntimeError("GAIA-Text endpoint workspace lifetime mismatch")
        workspace_runtime = metadata.get("workspace_runtime")
        if memory_available:
            if not isinstance(workspace_runtime, Mapping):
                raise RuntimeError("GAIA-Text endpoint workspace runtime mismatch")
        elif workspace_runtime != {}:
            raise RuntimeError("GAIA-Text memory-disabled endpoint exposed runtime state")
        if not memory_available and metadata.get("active_workspace_count") != 0:
            raise RuntimeError("GAIA-Text memory-disabled endpoint has workspace state")

    def policy_turn_candidate(self) -> str | None:
        return self._run_memory_compaction_path(super().policy_turn_candidate)

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        return self._run_memory_compaction_path(
            super().prepare_policy_turn,
            pressure,
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        if self.arm == EvaluationArm.NATIVE.value:
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
                {"role": "user", "content": GAIA_TEXT_POLICY_CONTINUATION_MARKER},
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

    def _run_memory_compaction_path(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.arm != _COMPACTION_ONLY:
            return callback(*args, **kwargs)
        arm = self.arm
        self.arm = _MEMORY
        try:
            return callback(*args, **kwargs)
        finally:
            self.arm = arm


__all__ = [
    "GAIA_TEXT_ARMS",
    "GAIA_TEXT_CONTEXT_COMPACTION_REQUEST",
    "GAIA_TEXT_DOMAIN_PROMPT",
    "GAIA_TEXT_MEMORY_AFFORDANCE",
    "GAIA_TEXT_POLICY_CONTINUATION_MARKER",
    "GaiaTextEnvClient",
]
