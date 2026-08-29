from __future__ import annotations

import hashlib
import unittest
from unittest.mock import Mock

from agentenv.controller.policy_turn import PreparedPolicyTurn, complete_policy_turn
from agentenv.controller.types import (
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_REPLACE,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_transition_info,
)
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_PATH,
    build_filesystem_checkpoint_receipt,
)
from agentenv.envs.swesmith import (
    SWE_CHECKPOINT_MAX_ATTEMPTS,
    SWE_CHECKPOINT_MIN_POST_READ_TASK_TURNS,
    SwesmithEnvClient,
)


def pressure(*, action_tokens: int = 14_500) -> PolicyContextPressure:
    return PolicyContextPressure(
        action_prompt_tokens=action_tokens,
        candidate_prompt_tokens=action_tokens + 200,
        max_prompt_tokens=24_576,
        max_model_tokens=32_768,
        max_response_tokens=1_024,
        max_observation_tokens=4_096,
        action_observation_envelope_tokens=10,
    )


def client(*, policy_steps: int = 0) -> SwesmithEnvClient:
    value = object.__new__(SwesmithEnvClient)
    value.env_id = 202
    value.metadata = {"configured_max_policy_turns": 30, "max_steps": 30}
    value._selected_policy_control = None
    value._checkpoint_retry_pending = False
    value._checkpoint_attempt_count = 0
    value._checkpoint_retry_exhausted = False
    value.checkpoint_contract_penalty = 0.0
    value._policy_step_count = policy_steps
    value._native_call_count = policy_steps
    value._context_epoch = 0
    value._session_epoch = 0
    value._policy_context_bound = True
    value._current_policy_context = None
    value._zero_progress_shell_receipts = set()
    value._immutable_policy_context = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    return value


def failed_output(*, workspace_changed: bool, state: str = "write failed") -> StepOutput:
    receipt = build_filesystem_checkpoint_receipt(
        action_kind="shell_command",
        action_completed=True,
        workspace_diff={"added": [], "modified": [], "deleted": []},
        workspace_snapshot={"files": []},
    )
    return StepOutput(
        state=state,
        reward=0.0,
        done=False,
        info=build_task_neutral_transition_info(
            env_info={"filesystem_checkpoint": receipt},
            action_submission={"kind": "shell_command"},
            native_step_before=10,
            native_step_after=11,
            native_call_count_before=10,
            native_call_count_after=11,
            context_epoch_before=0,
            context_epoch_after=0,
            session_epoch_before=0,
            session_epoch_after=0,
            policy_step_before=10,
            policy_step_after=11,
            wrapper_evidence={
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": workspace_changed,
                    "basis": "workspace_changed" if workspace_changed else "shell_executed",
                },
                "action_progress": {
                    "schema": "swesmith_action_progress_v1",
                    "action_fingerprint": "a" * 64,
                    "result_fingerprint": "b" * 64,
                    "workspace_changed": workspace_changed,
                },
            },
        ),
    )


def successful_output() -> StepOutput:
    payload = b"objective: fix parser\nnext: edit source\n"
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
    return StepOutput(
        state="checkpoint written",
        reward=0.0,
        done=False,
        info=build_task_neutral_transition_info(
            env_info={"filesystem_checkpoint": receipt},
            action_submission={"kind": "shell_command"},
            native_step_before=23,
            native_step_after=24,
            native_call_count_before=23,
            native_call_count_after=24,
            context_epoch_before=0,
            context_epoch_after=0,
            session_epoch_before=0,
            session_epoch_after=0,
            policy_step_before=23,
            policy_step_after=24,
            wrapper_evidence={
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": True,
                    "basis": "workspace_changed",
                }
            },
        ),
    )


class SwesmithCompactionLifecycleTests(unittest.TestCase):
    def test_failed_control_preserves_action_and_observation_even_after_workspace_change(self) -> None:
        value = client(policy_steps=10)
        value._selected_policy_control = "context_compaction"

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return failed_output(workspace_changed=True, state="checkpoint path missing")

        value._step_native_policy_action = Mock(side_effect=execute)
        control = value.policy_turn_candidate()
        self.assertIsNotNone(control)
        prepared = PreparedPolicyTurn(
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task evidence"},
                {"role": "user", "content": control},
            ),
            prompt_token_count=100,
            control_request=control,
        )
        output, messages = complete_policy_turn(
            value,
            prepared,
            'shell_command {"command":"touch partial.txt","workdir":"."}',
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            CONTEXT_OPERATION_APPEND,
        )
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertIn("touch partial.txt", messages[-2]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "checkpoint path missing"})
        self.assertTrue(output.info["wrapper_evidence"]["retry_feedback_preserved"])

    def test_retry_is_attempt_bounded_instead_of_repeating_to_horizon(self) -> None:
        value = client(policy_steps=10)
        for expected_attempt in range(1, SWE_CHECKPOINT_MAX_ATTEMPTS + 1):
            candidate = value.policy_turn_candidate()
            self.assertIsNotNone(candidate)
            self.assertIn(
                f"checkpoint attempt {expected_attempt}/{SWE_CHECKPOINT_MAX_ATTEMPTS}",
                candidate,
            )
            selected = value.prepare_policy_turn(pressure())
            self.assertEqual(selected, candidate)
            value._selected_policy_control = "context_compaction"

            def execute(_action: str) -> StepOutput:
                value._policy_step_count += 1
                value._native_call_count += 1
                return failed_output(workspace_changed=False)

            value._step_native_policy_action = Mock(side_effect=execute)
            output = value._complete_context_compaction(
                'shell_command {"command":"pwd","workdir":"."}'
            )
            self.assertEqual(
                output.info["context_transition"]["operation"],
                CONTEXT_OPERATION_APPEND,
            )

        self.assertTrue(value._checkpoint_retry_exhausted)
        candidate = value.policy_turn_candidate()
        self.assertIsNotNone(candidate)
        self.assertIn(
            f"checkpoint attempt 1/{SWE_CHECKPOINT_MAX_ATTEMPTS}",
            candidate,
        )
        # A completed retry cycle must not immediately become an unbounded
        # third control turn while one ordinary turn still fits.
        self.assertIsNone(value.prepare_policy_turn(pressure()))

    def test_exhausted_cycle_rearms_after_one_safe_ordinary_action(self) -> None:
        value = client(policy_steps=14)
        value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
        value._checkpoint_retry_exhausted = True

        self.assertIsNone(value.prepare_policy_turn(pressure()))

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return failed_output(workspace_changed=True, state="ordinary progress")

        value._step_native_policy_action = Mock(side_effect=execute)
        output = value.step(
            'shell_command {"command":"pwd","workdir":"."}'
        )

        self.assertFalse(output.done)
        self.assertEqual(value._checkpoint_attempt_count, 0)
        self.assertFalse(value._checkpoint_retry_exhausted)
        self.assertTrue(value._checkpoint_retry_pending)
        candidate = value.prepare_policy_turn(pressure())
        self.assertIsNotNone(candidate)
        self.assertIn(
            f"checkpoint attempt 1/{SWE_CHECKPOINT_MAX_ATTEMPTS}",
            candidate,
        )

    def test_exhausted_cycle_rearms_before_one_more_action_would_overflow(self) -> None:
        value = client(policy_steps=14)
        value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
        value._checkpoint_retry_exhausted = True

        candidate = value.prepare_policy_turn(pressure(action_tokens=19_300))

        self.assertIsNotNone(candidate)
        self.assertIn(
            f"checkpoint attempt 1/{SWE_CHECKPOINT_MAX_ATTEMPTS}",
            candidate,
        )
        self.assertEqual(value._checkpoint_attempt_count, 0)
        self.assertFalse(value._checkpoint_retry_exhausted)
        self.assertTrue(value._checkpoint_retry_pending)

    def test_exhausted_cycle_stays_closed_without_fresh_cycle_budget(self) -> None:
        value = client(policy_steps=39)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
        value._checkpoint_retry_exhausted = True

        self.assertIsNone(value.policy_turn_candidate())
        self.assertIsNone(value.prepare_policy_turn(pressure(action_tokens=19_300)))

    def test_late_checkpoint_is_not_requested_without_recovery_budget(self) -> None:
        # Six turns remain. A fresh checkpoint cycle needs two possible writes,
        # one read, and the configured number of task actions.
        value = client(policy_steps=24)
        self.assertEqual(SWE_CHECKPOINT_MIN_POST_READ_TASK_TURNS, 4)
        self.assertIsNone(value.policy_turn_candidate())
        self.assertIsNone(value.prepare_policy_turn(pressure(action_tokens=20_000)))

    def test_successor_states_exact_remaining_turns_and_continues_saved_next_action(self) -> None:
        value = client(policy_steps=23)
        value._selected_policy_control = "context_compaction"

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return successful_output()

        value._step_native_policy_action = Mock(side_effect=execute)
        output = value._complete_context_compaction(
            'shell_command {"command":"write checkpoint","workdir":"."}'
        )
        self.assertEqual(
            output.info["context_transition"]["operation"],
            CONTEXT_OPERATION_REPLACE,
        )
        marker = output.info["context_transition"]["messages"][-1]["content"]
        self.assertIn("Exactly 6 policy actions remain", marker)
        self.assertIn("execute its saved next concrete action", marker)
        self.assertIn("do not restart broad repository inspection", marker)


if __name__ == "__main__":
    unittest.main()
