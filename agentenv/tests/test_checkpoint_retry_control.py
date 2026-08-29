from __future__ import annotations

import hashlib
import unittest
from unittest.mock import Mock

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.policy_turn import (
    PreparedPolicyTurn,
    complete_policy_turn,
    prepare_policy_turn,
)
from agentenv.controller.types import (
    ActionFormat,
    CONTEXT_OPERATION_REPLACE,
    CONTEXT_OPERATION_RETRY_CONTROL,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_PATH,
    build_filesystem_checkpoint_receipt,
    checkpoint_bounded_retry_trigger_tokens,
)
from agentenv.envs.swesmith import SwesmithEnvClient


class RetryControlClient(BaseEnvClient):
    control_request = "Write the bounded checkpoint now."

    def __init__(self) -> None:
        super().__init__(action_format=ActionFormat.REACT)
        self.outputs: list[str] = []

    def __len__(self) -> int:
        return 1

    def observe(self) -> str:
        return "task"

    def reset(self, idx: int = 0) -> None:
        del idx

    def step(self, action: str) -> StepOutput:
        self.outputs.append(action)
        return StepOutput(
            state="large failed checkpoint observation " * 2_000,
            reward=-0.01,
            done=False,
            info=build_task_neutral_transition_info(
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_RETRY_CONTROL
                ),
                wrapper_evidence={
                    "sampled_policy_output_preserved_in_ledger": True,
                    "native_observation_preserved_in_ledger": True,
                },
            ),
        )

    def policy_turn_candidate(self) -> str | None:
        return self.control_request

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        if pressure is None:
            raise AssertionError("pressure required")
        return self.control_request


def count_words(messages) -> int:
    return sum(len(message["content"].split()) + 2 for message in messages)


class BoundedCheckpointRetryTests(unittest.TestCase):
    def test_trigger_reserves_one_ordinary_turn_before_next_control(self) -> None:
        pressure = PolicyContextPressure(
            action_prompt_tokens=14_000,
            candidate_prompt_tokens=14_200,
            max_prompt_tokens=24_576,
            max_model_tokens=32_768,
            max_response_tokens=1_024,
            max_observation_tokens=8_192,
            action_observation_envelope_tokens=10,
        )
        self.assertEqual(
            checkpoint_bounded_retry_trigger_tokens(pressure),
            23_426,
        )

    def test_repeated_failed_control_turns_restore_exact_base_context(self) -> None:
        client = RetryControlClient()
        base = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task evidence " * 5_000},
        ]

        retry_context = base
        prompt_token_counts = []
        for attempt in range(24):
            prepared = prepare_policy_turn(
                client,
                retry_context,
                count_prompt_tokens=count_words,
                max_prompt_tokens=24_576,
                max_model_tokens=32_768,
                max_response_tokens=1_024,
                max_observation_tokens=8_192,
                action_observation_envelope_tokens=10,
            )
            self.assertEqual(prepared.control_request, client.control_request)
            prompt_token_counts.append(prepared.prompt_token_count)
            output, retry_context = complete_policy_turn(
                client, prepared, f"malformed checkpoint action {attempt}"
            )
            self.assertEqual(output.reward, -0.01)
            self.assertFalse(output.done)
            self.assertEqual(retry_context, base)

        self.assertEqual(len(set(prompt_token_counts)), 1)
        self.assertEqual(len(client.outputs), 24)

    def test_retry_control_cannot_rewind_an_ordinary_policy_turn(self) -> None:
        client = RetryControlClient()
        prepared = PreparedPolicyTurn(
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "ordinary task observation"},
            ),
            prompt_token_count=7,
            control_request=None,
        )

        with self.assertRaisesRegex(
            ValueError, "retry_control requires a prepared control request"
        ):
            complete_policy_turn(client, prepared, "ordinary action")

    def test_swesmith_failed_write_preserves_feedback_for_bounded_retry(self) -> None:
        client = object.__new__(SwesmithEnvClient)
        client.env_id = 202
        client._selected_policy_control = "context_compaction"
        client._checkpoint_retry_pending = False
        client._checkpoint_attempt_count = 0
        client._checkpoint_retry_exhausted = False
        client.checkpoint_contract_penalty = -0.01
        client._policy_step_count = 0
        client._native_call_count = 0
        client._session_epoch = 0
        client.metadata = {"configured_max_policy_turns": 30, "max_steps": 30}
        client._context_epoch = 0
        client._immutable_policy_context = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        failed_receipt = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=True,
            workspace_diff={"added": [], "modified": [], "deleted": []},
            workspace_snapshot={"files": []},
        )
        native_output = StepOutput(
            state="large shell observation",
            reward=0.0,
            done=False,
            info=build_task_neutral_transition_info(
                env_info={"filesystem_checkpoint": failed_receipt},
                action_submission={"kind": "shell_command"},
                native_step_before=0,
                native_step_after=1,
                native_call_count_before=0,
                native_call_count_after=1,
                context_epoch_before=0,
                context_epoch_after=0,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=0,
                policy_step_after=1,
                wrapper_evidence={
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": True,
                        "basis": "shell_executed",
                    }
                },
            ),
        )
        client._step_native_policy_action = Mock(return_value=native_output)

        output = client._complete_context_compaction(
            'shell_command {"command":"pwd","workdir":"."}'
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertTrue(output.info["wrapper_evidence"]["retry_pending"])
        self.assertTrue(output.info["wrapper_evidence"]["retry_feedback_preserved"])
        self.assertFalse(output.info["wrapper_evidence"]["retry_context_restored"])
        self.assertFalse(output.info["wrapper_evidence"]["context_replaced"])
        self.assertEqual(output.reward, -0.01)
        self.assertFalse(output.done)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"],
            {
                "schema": "swesmith_checkpoint_contract_reward_v1",
                "basis": "checkpoint_contract_unsatisfied",
                "reward_before": 0.0,
                "configured_penalty": -0.01,
                "applied_delta": -0.01,
                "final_reward": -0.01,
                "deduplicated": False,
            },
        )
        self.assertEqual(
            output.info["wrapper_evidence"]["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": True,
                "basis": "shell_executed",
            },
        )
        self.assertEqual(client._context_epoch, 0)

    def test_swesmith_failed_checkpoint_does_not_double_existing_penalty(self) -> None:
        client = object.__new__(SwesmithEnvClient)
        client.env_id = 203
        client._selected_policy_control = "context_compaction"
        client._checkpoint_retry_pending = False
        client._checkpoint_attempt_count = 0
        client._checkpoint_retry_exhausted = False
        client.checkpoint_contract_penalty = -0.01
        client._policy_step_count = 0
        client._native_call_count = 0
        client._session_epoch = 0
        client.metadata = {"configured_max_policy_turns": 30, "max_steps": 30}
        client._context_epoch = 0
        client._immutable_policy_context = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        failed_receipt = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=False,
            workspace_diff={"added": [], "modified": [], "deleted": []},
            workspace_snapshot={"files": []},
        )
        native_output = StepOutput(
            state="parser rejected",
            reward=-0.01,
            done=False,
            info=build_task_neutral_transition_info(
                env_info={"filesystem_checkpoint": failed_receipt},
                action_submission={"kind": "parser_error"},
                native_step_before=0,
                native_step_after=1,
                native_call_count_before=0,
                native_call_count_after=1,
                context_epoch_before=0,
                context_epoch_after=0,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=0,
                policy_step_after=1,
                wrapper_evidence={
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": False,
                        "basis": "parser_rejected",
                    },
                    "reward_overlay": {
                        "schema": "swesmith_invalid_action_reward_v1",
                        "basis": "parser_rejected",
                        "native_reward": 0.0,
                        "penalty": -0.01,
                        "final_reward": -0.01,
                    },
                },
            ),
        )
        client._step_native_policy_action = Mock(return_value=native_output)

        output = client._complete_context_compaction("malformed action")

        self.assertEqual(output.reward, -0.01)
        self.assertFalse(output.done)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"]["applied_delta"],
            0.0,
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["reward_overlay"]["deduplicated"]
        )
        self.assertEqual(
            output.info["context_transition"]["operation"],
            CONTEXT_OPERATION_RETRY_CONTROL,
        )
        self.assertTrue(output.info["wrapper_evidence"]["retry_context_restored"])
        self.assertFalse(
            output.info["wrapper_evidence"]["retry_feedback_preserved"]
        )

    def test_swesmith_terminal_submission_keeps_native_reward(self) -> None:
        client = object.__new__(SwesmithEnvClient)
        client.env_id = 204
        client._selected_policy_control = "context_compaction"
        client._checkpoint_retry_pending = False
        client._checkpoint_attempt_count = 0
        client._checkpoint_retry_exhausted = False
        client.checkpoint_contract_penalty = -0.01
        client._policy_step_count = 0
        client._native_call_count = 0
        client._session_epoch = 0
        client.metadata = {"configured_max_policy_turns": 30, "max_steps": 30}
        client._context_epoch = 0
        client._immutable_policy_context = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        native_output = StepOutput(
            state="submitted",
            reward=1.0,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={"filesystem_checkpoint": None},
                action_submission={"kind": "terminal_submission"},
                native_step_before=0,
                native_step_after=1,
                native_call_count_before=0,
                native_call_count_after=1,
                context_epoch_before=0,
                context_epoch_after=0,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=0,
                policy_step_after=1,
                wrapper_evidence={
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": True,
                        "basis": "terminal_submission",
                    }
                },
            ),
        )
        client._step_native_policy_action = Mock(return_value=native_output)

        output = client._complete_context_compaction("submit")

        self.assertEqual(output.reward, 1.0)
        self.assertTrue(output.done)
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])

    def test_swesmith_replaces_only_after_attested_bounded_write(self) -> None:
        client = object.__new__(SwesmithEnvClient)
        client.env_id = 202
        client._selected_policy_control = "context_compaction"
        client._checkpoint_retry_pending = True
        client._checkpoint_attempt_count = 1
        client._checkpoint_retry_exhausted = False
        client.checkpoint_contract_penalty = -0.01
        client._policy_step_count = 1
        client._native_call_count = 1
        client._session_epoch = 0
        client.metadata = {"configured_max_policy_turns": 30, "max_steps": 30}
        client._context_epoch = 0
        client._zero_progress_shell_receipts = set()
        client._immutable_policy_context = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        payload = b"objective: fix parser\nnext: run tests\n"
        receipt = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=True,
            workspace_diff={
                "added": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
                "modified": [],
                "deleted": [],
            },
            workspace_snapshot={
                "files": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            },
        )
        native_output = StepOutput(
            state="checkpoint written",
            reward=0.0,
            done=False,
            info=build_task_neutral_transition_info(
                env_info={"filesystem_checkpoint": receipt},
                action_submission={"kind": "shell_command"},
                context_epoch_before=0,
                context_epoch_after=0,
                wrapper_evidence={
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": True,
                        "basis": "workspace_changed",
                    }
                },
            ),
        )
        client._step_native_policy_action = Mock(return_value=native_output)

        output = client._complete_context_compaction(
            'shell_command {"command":"write checkpoint","workdir":"."}'
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            CONTEXT_OPERATION_REPLACE,
        )
        self.assertFalse(output.info["wrapper_evidence"]["retry_pending"])
        self.assertTrue(output.info["wrapper_evidence"]["context_replaced"])
        self.assertEqual(output.reward, 0.0)
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])
        self.assertEqual(client._context_epoch, 1)


if __name__ == "__main__":
    unittest.main()
