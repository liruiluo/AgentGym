from __future__ import annotations

import hashlib
import unittest
from unittest.mock import Mock

from agentenv.controller.policy_turn import (
    PreparedPolicyTurn,
    complete_policy_turn,
    prepare_policy_turn,
)
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
        max_response_tokens=2_048,
        max_observation_tokens=6_145,
        action_observation_envelope_tokens=9,
    )


def client(*, policy_steps: int = 0) -> SwesmithEnvClient:
    value = object.__new__(SwesmithEnvClient)
    value.env_id = 202
    value.metadata = {"configured_max_policy_turns": 30, "max_steps": 30}
    value._selected_policy_control = None
    value._checkpoint_retry_pending = False
    value._checkpoint_attempt_count = 0
    value._checkpoint_retry_exhausted = False
    value._checkpoint_cycle_index = 0
    value._checkpoint_cycle_attempt_limit = SWE_CHECKPOINT_MAX_ATTEMPTS
    value._checkpoint_total_attempt_count = 0
    value._checkpoint_ordinary_turn_required = False
    value._checkpoint_capacity_terminal_after_ordinary = False
    value._checkpoint_capacity_terminal_reason = None
    value._selected_checkpoint_terminal_on_failure = False
    value._selected_checkpoint_terminal_on_executed_failure = False
    value.checkpoint_contract_penalty = -0.01
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


def rejected_output(
    *,
    state: str = "invalid checkpoint action",
    basis: str = "parser_rejected",
) -> StepOutput:
    if basis not in {"parser_rejected", "executor_rejected"}:
        raise ValueError(f"unsupported rejection basis: {basis}")
    receipt = build_filesystem_checkpoint_receipt(
        action_kind="parser_error" if basis == "parser_rejected" else "shell_command",
        action_completed=False,
        workspace_diff={"added": [], "modified": [], "deleted": []},
        workspace_snapshot={"files": []},
    )
    return StepOutput(
        state=state,
        reward=-0.01,
        done=False,
        info=build_task_neutral_transition_info(
            env_info={"filesystem_checkpoint": receipt},
            action_submission={
                "kind": "parser_error" if basis == "parser_rejected" else "shell_command"
            },
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
                    "positive_eligible": False,
                    "basis": basis,
                }
            },
        ),
    )


def horizon_response(*, reward: float = -0.01) -> dict:
    return {
        "observation": "Checkpoint capacity exhausted; episode closed.",
        "reward": reward,
        "done": True,
        "info": {
            "status": "max_policy_steps_exhausted",
            "sample_excluded": False,
        },
    }


def terminal_failed_checkpoint_output(
    *,
    reward: float = -0.01,
    sample_excluded: bool = False,
) -> StepOutput:
    receipt = build_filesystem_checkpoint_receipt(
        action_kind="parser_error",
        action_completed=False,
        workspace_diff={"added": [], "modified": [], "deleted": []},
        workspace_snapshot={"files": []},
    )
    return StepOutput(
        state="Episode ended without a successful official submission.",
        reward=reward,
        done=True,
        info=build_task_neutral_transition_info(
            env_info={
                "step": 45,
                "terminal": True,
                "sample_excluded": sample_excluded,
                "episode_success": False,
                "filesystem_checkpoint": receipt,
            },
            action_submission={"kind": "parser_error"},
            native_step_before=44,
            native_step_after=45,
            native_call_count_before=44,
            native_call_count_after=45,
            context_epoch_before=0,
            context_epoch_after=0,
            session_epoch_before=0,
            session_epoch_after=0,
            policy_step_before=44,
            policy_step_after=45,
            wrapper_evidence={
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": False,
                    "basis": "parser_rejected",
                }
            },
        ),
    )


def terminal_submission_output(*, reward: float = 1.0) -> StepOutput:
    return StepOutput(
        state="submitted",
        reward=reward,
        done=True,
        info=build_task_neutral_transition_info(
            env_info={
                "step": 45,
                "terminal": True,
                "sample_excluded": False,
                "episode_success": reward > 0.0,
                "filesystem_checkpoint": None,
            },
            action_submission={"kind": "terminal_submission"},
            native_step_before=44,
            native_step_after=45,
            native_call_count_before=44,
            native_call_count_after=45,
            context_epoch_before=0,
            context_epoch_after=0,
            session_epoch_before=0,
            session_epoch_after=0,
            policy_step_before=44,
            policy_step_after=45,
            wrapper_evidence={
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": True,
                    "basis": "terminal_submission",
                }
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
                return rejected_output()

            value._step_native_policy_action = Mock(side_effect=execute)
            output = value._complete_context_compaction(
                'shell_command {"command":"pwd","workdir":"."}'
            )
            self.assertEqual(
                output.info["context_transition"]["operation"],
                "retry_control",
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
        value._checkpoint_ordinary_turn_required = True

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
        self.assertFalse(value._checkpoint_retry_pending)
        self.assertFalse(value._checkpoint_ordinary_turn_required)
        candidate = value.prepare_policy_turn(pressure())
        self.assertIsNotNone(candidate)
        self.assertIn(
            f"checkpoint attempt 1/{SWE_CHECKPOINT_MAX_ATTEMPTS}",
            candidate,
        )

    def test_rejected_ordinary_action_does_not_rearm_checkpoint_cycle(self) -> None:
        for basis in ("parser_rejected", "executor_rejected"):
            with self.subTest(basis=basis):
                value = client(policy_steps=14)
                value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
                value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
                value._checkpoint_retry_exhausted = True
                value._checkpoint_ordinary_turn_required = True
                self.assertIsNone(value.prepare_policy_turn(pressure()))

                def reject(_action: str, *, rejection_basis: str = basis) -> StepOutput:
                    value._policy_step_count += 1
                    value._native_call_count += 1
                    return rejected_output(basis=rejection_basis)

                value._step_native_policy_action = Mock(side_effect=reject)
                output = value.step("not an executable action")

                self.assertFalse(output.done)
                self.assertTrue(value._checkpoint_retry_exhausted)
                self.assertTrue(value._checkpoint_ordinary_turn_required)
                self.assertFalse(value._checkpoint_retry_pending)
                self.assertEqual(
                    output.info["wrapper_evidence"]["actor_credit"]["basis"],
                    basis,
                )
                self.assertIsNone(value.prepare_policy_turn(pressure()))
                self.assertEqual(value._checkpoint_cycle_index, 0)

    def test_executed_shell_rearms_even_when_repeat_credit_is_ineligible(self) -> None:
        value = client(policy_steps=14)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
        value._checkpoint_retry_exhausted = True
        value._checkpoint_ordinary_turn_required = True
        native = failed_output(workspace_changed=False, state="ordinary shell output")
        info = dict(native.info)
        env_info = dict(info["env_info"])
        env_info["actor_credit"] = {
            "schema": "task_neutral_actor_credit_v1",
            "positive_eligible": True,
            "basis": "shell_executed",
        }
        info["env_info"] = env_info
        wrapper_evidence = dict(info["wrapper_evidence"])
        wrapper_evidence["actor_credit"] = {
            "schema": "task_neutral_actor_credit_v1",
            "positive_eligible": False,
            "basis": "zero_progress_repeat",
        }
        info["wrapper_evidence"] = wrapper_evidence
        native = StepOutput(
            state=native.state, reward=native.reward, done=native.done, info=info
        )

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return native

        value._step_native_policy_action = Mock(side_effect=execute)
        output = value.step(
            'shell_command {"command":"pwd","workdir":"."}'
        )

        self.assertFalse(output.done)
        self.assertFalse(value._checkpoint_retry_exhausted)
        self.assertFalse(value._checkpoint_ordinary_turn_required)
        self.assertEqual(
            output.info["wrapper_evidence"]["actor_credit"]["basis"],
            "zero_progress_repeat",
        )

    def test_exhausted_cycle_rearms_before_one_more_action_would_overflow(self) -> None:
        value = client(policy_steps=14)
        value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
        value._checkpoint_retry_exhausted = True
        value._checkpoint_ordinary_turn_required = True

        candidate = value.prepare_policy_turn(pressure(action_tokens=19_500))

        self.assertIsNone(candidate)
        self.assertTrue(value._checkpoint_retry_exhausted)
        self.assertTrue(value._checkpoint_capacity_terminal_after_ordinary)
        self.assertEqual(
            value._checkpoint_capacity_terminal_reason,
            "ordinary_progress_would_exceed_prompt_capacity",
        )

    def test_exhausted_cycle_stays_closed_without_fresh_cycle_budget(self) -> None:
        value = client(policy_steps=39)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value._checkpoint_attempt_count = SWE_CHECKPOINT_MAX_ATTEMPTS
        value._checkpoint_retry_exhausted = True
        value._checkpoint_ordinary_turn_required = True

        self.assertIsNotNone(value.policy_turn_candidate())
        self.assertIsNone(value.prepare_policy_turn(pressure(action_tokens=19_300)))

    def test_late_checkpoint_is_not_requested_without_recovery_budget(self) -> None:
        # Six turns remain. A fresh checkpoint cycle needs two possible writes,
        # one read, and the configured number of task actions.
        value = client(policy_steps=24)
        self.assertEqual(SWE_CHECKPOINT_MIN_POST_READ_TASK_TURNS, 4)
        self.assertIsNotNone(value.policy_turn_candidate())
        self.assertIsNotNone(value.prepare_policy_turn(pressure(action_tokens=20_000)))


    def test_rejected_cycle_requires_an_ordinary_turn_before_rearm(self) -> None:
        value = client(policy_steps=10)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        high_pressure = pressure(action_tokens=14_000)

        for expected_attempt in range(1, SWE_CHECKPOINT_MAX_ATTEMPTS + 1):
            candidate = value.prepare_policy_turn(high_pressure)
            self.assertIsNotNone(candidate)
            self.assertIn(
                f"checkpoint attempt {expected_attempt}/{SWE_CHECKPOINT_MAX_ATTEMPTS}",
                candidate,
            )
            value._step_native_policy_action = Mock(
                side_effect=lambda _action: (
                    setattr(value, "_policy_step_count", value._policy_step_count + 1)
                    or setattr(value, "_native_call_count", value._native_call_count + 1)
                    or rejected_output()
                )
            )
            output = value._complete_context_compaction("not an action")
            self.assertFalse(output.done)

        self.assertTrue(value._checkpoint_retry_exhausted)
        self.assertTrue(value._checkpoint_ordinary_turn_required)
        self.assertIsNone(value.prepare_policy_turn(high_pressure))
        self.assertFalse(value._checkpoint_retry_pending)

        value._step_native_policy_action = Mock(
            side_effect=lambda _action: (
                setattr(value, "_policy_step_count", value._policy_step_count + 1)
                or setattr(value, "_native_call_count", value._native_call_count + 1)
                or failed_output(workspace_changed=True, state="ordinary progress")
            )
        )
        ordinary = value.step('shell_command {"command":"pwd","workdir":"."}')
        self.assertFalse(ordinary.done)
        self.assertFalse(value._checkpoint_retry_exhausted)
        self.assertFalse(value._checkpoint_ordinary_turn_required)
        self.assertFalse(value._checkpoint_retry_pending)

        candidate = value.prepare_policy_turn(high_pressure)
        self.assertIsNotNone(candidate)
        self.assertIn("checkpoint attempt 1/2", candidate)
        self.assertEqual(value._checkpoint_cycle_index, 2)

    def test_executed_first_failure_keeps_attempt_two_sampleable_then_closes(self) -> None:
        value = client(policy_steps=10)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        # Use the exact AG launch envelope (2,048 response + 6,145
        # observation + 9 message overhead). At the exact two-growth boundary,
        # an executed first failure still leaves attempt two sampleable.
        first_pressure = pressure(action_tokens=8_172)
        self.assertIsNotNone(value.prepare_policy_turn(first_pressure))

        value._step_native_policy_action = Mock(
            side_effect=lambda _action: (
                setattr(value, "_policy_step_count", value._policy_step_count + 1)
                or setattr(value, "_native_call_count", value._native_call_count + 1)
                or failed_output(workspace_changed=True, state="x" * 4_096)
            )
        )
        first = value._complete_context_compaction(
            'shell_command {"command":"touch wrong.txt","workdir":"."}'
        )
        self.assertFalse(first.done)
        self.assertTrue(value._checkpoint_retry_pending)

        second_pressure = pressure(action_tokens=16_574)
        second = value.prepare_policy_turn(second_pressure)
        self.assertIsNotNone(second)
        self.assertIn("checkpoint attempt 2/2", second)
        self.assertFalse(value._selected_checkpoint_terminal_on_failure)
        self.assertTrue(
            value._selected_checkpoint_terminal_on_executed_failure
        )
        self.assertLessEqual(second_pressure.candidate_prompt_tokens, 24_576)
        self.assertGreater(
            second_pressure.candidate_prompt_tokens
            + second_pressure.max_response_tokens
            + second_pressure.max_observation_tokens
            + second_pressure.action_observation_envelope_tokens,
            24_576,
        )
        value._request = Mock(return_value=horizon_response())
        second_output = value._complete_context_compaction(
            'shell_command {"command":"touch another-wrong.txt","workdir":"."}'
        )
        self.assertTrue(second_output.done)
        self.assertEqual(second_output.reward, -0.01)
        value._request.assert_called_once_with(
            "POST", "horizon", json={"id": value.env_id}
        )

    def test_high_pressure_executed_first_failure_closes_before_overflow(self) -> None:
        value = client(policy_steps=10)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value.checkpoint_contract_penalty = -0.1
        selected = value.prepare_policy_turn(pressure(action_tokens=19_300))

        self.assertIsNotNone(selected)
        self.assertTrue(value._selected_checkpoint_terminal_on_executed_failure)
        value._step_native_policy_action = Mock(
            side_effect=lambda _action: (
                setattr(value, "_policy_step_count", value._policy_step_count + 1)
                or setattr(value, "_native_call_count", value._native_call_count + 1)
                or failed_output(workspace_changed=True, state="x" * 6_144)
            )
        )
        value._request = Mock(return_value=horizon_response())
        output = value._complete_context_compaction(
            'shell_command {"command":"touch wrong.txt","workdir":"."}'
        )

        self.assertTrue(output.done)
        self.assertEqual(output.reward, -0.1)
        self.assertEqual(
            output.info["wrapper_evidence"]["capacity_termination_reason"],
            "checkpoint_failure_would_exceed_prompt_capacity",
        )

    def test_six_turn_tail_uses_one_shot_checkpoint_and_closes_on_failure(self) -> None:
        value = client(policy_steps=39)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        selected = value.prepare_policy_turn(pressure(action_tokens=19_300))

        self.assertIsNotNone(selected)
        self.assertIn("checkpoint attempt 1/1", selected)
        self.assertTrue(value._selected_checkpoint_terminal_on_failure)
        self.assertEqual(value._checkpoint_cycle_index, 1)

        value._step_native_policy_action = Mock(
            side_effect=lambda _action: (
                setattr(value, "_policy_step_count", value._policy_step_count + 1)
                or setattr(value, "_native_call_count", value._native_call_count + 1)
                or failed_output(workspace_changed=True)
            )
        )
        value._request = Mock(return_value=horizon_response())
        output = value._complete_context_compaction(
            'shell_command {"command":"touch wrong.txt","workdir":"."}'
        )

        self.assertTrue(output.done)
        self.assertEqual(output.reward, -0.01)
        self.assertEqual(
            output.info["wrapper_evidence"]["capacity_termination_reason"],
            "checkpoint_failure_would_exceed_prompt_capacity",
        )
        value._request.assert_called_once_with(
            "POST", "horizon", json={"id": value.env_id}
        )

    def test_capacity_terminal_after_ordinary_keeps_natural_horizon_reward(self) -> None:
        value = client(policy_steps=39)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value.checkpoint_contract_penalty = -0.1
        selected = value.prepare_policy_turn(pressure(action_tokens=24_400))

        self.assertIsNone(selected)
        value._step_native_policy_action = Mock(
            side_effect=lambda _action: (
                setattr(value, "_policy_step_count", value._policy_step_count + 1)
                or setattr(value, "_native_call_count", value._native_call_count + 1)
                or failed_output(workspace_changed=True, state="final observation")
            )
        )
        value._request = Mock(return_value=horizon_response())
        output = value.step('shell_command {"command":"pytest","workdir":"."}')

        self.assertTrue(output.done)
        self.assertEqual(output.reward, -0.01)
        evidence = output.info["wrapper_evidence"]
        self.assertEqual(evidence["native_action_reward"], 0.0)
        self.assertEqual(evidence["horizon_reward"], -0.01)
        self.assertNotIn("reward_overlay", evidence)
        self.assertEqual(
            evidence["actor_credit"],
            evidence["native_wrapper_evidence"]["actor_credit"],
        )
        self.assertEqual(
            evidence["action_progress"],
            evidence["native_wrapper_evidence"]["action_progress"],
        )

    def test_final_turn_never_selects_checkpoint_without_read_budget(self) -> None:
        value = client(policy_steps=44)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}

        selected = value.prepare_policy_turn(pressure(action_tokens=17_000))

        self.assertIsNone(selected)
        self.assertIsNone(value._selected_policy_control)
        self.assertTrue(value._checkpoint_capacity_terminal_after_ordinary)
        self.assertEqual(
            value._checkpoint_capacity_terminal_reason,
            "checkpoint_read_turn_unavailable",
        )
        self.assertEqual(value._checkpoint_cycle_index, 0)

    def test_final_turn_capacity_pressure_preserves_terminal_submission(self) -> None:
        value = client(policy_steps=44)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        selected = value.prepare_policy_turn(pressure(action_tokens=17_000))
        self.assertIsNone(selected)

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return terminal_submission_output()

        value._step_native_policy_action = Mock(side_effect=execute)
        value._request = Mock()
        output = value.step("submit")

        self.assertTrue(output.done)
        self.assertEqual(output.reward, 1.0)
        value._request.assert_not_called()

    def test_terminal_max_steps_failed_checkpoint_gets_contract_ceiling(self) -> None:
        value = client(policy_steps=44)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value.checkpoint_contract_penalty = -0.1
        value._selected_policy_control = "context_compaction"
        value._checkpoint_cycle_index = 1
        value._checkpoint_cycle_attempt_limit = 1

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return terminal_failed_checkpoint_output()

        value._step_native_policy_action = Mock(side_effect=execute)
        output = value._complete_context_compaction("malformed checkpoint")

        self.assertTrue(output.done)
        self.assertEqual(output.reward, -0.1)
        evidence = output.info["wrapper_evidence"]
        self.assertEqual(
            evidence["reward_overlay"]["basis"],
            "checkpoint_contract_unsatisfied",
        )
        self.assertEqual(evidence["reward_overlay"]["reward_before"], -0.01)
        self.assertEqual(evidence["reward_overlay"]["final_reward"], -0.1)
        self.assertTrue(evidence["native_max_steps_terminal"])

    def test_sample_excluded_terminal_checkpoint_preserves_zero_reward(self) -> None:
        value = client(policy_steps=44)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value.checkpoint_contract_penalty = -0.1
        value._selected_policy_control = "context_compaction"
        value._checkpoint_cycle_index = 1
        value._checkpoint_cycle_attempt_limit = 1

        def execute(_action: str) -> StepOutput:
            value._policy_step_count += 1
            value._native_call_count += 1
            return terminal_failed_checkpoint_output(
                reward=0.0, sample_excluded=True
            )

        value._step_native_policy_action = Mock(side_effect=execute)
        output = value._complete_context_compaction("malformed checkpoint")

        self.assertTrue(output.done)
        self.assertEqual(output.reward, 0.0)
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])
        self.assertFalse(output.info["wrapper_evidence"]["native_max_steps_terminal"])

    def test_already_over_capacity_finalizes_before_sampling(self) -> None:
        value = client(policy_steps=39)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value._request = Mock(return_value=horizon_response())
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "oversized task history"},
        ]

        prepared = prepare_policy_turn(
            value,
            messages,
            count_prompt_tokens=lambda _messages: 24_600,
            max_prompt_tokens=24_576,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=6_145,
            action_observation_envelope_tokens=9,
        )

        self.assertIsNone(prepared.control_request)
        self.assertIsNotNone(prepared.pre_sampling_terminal)
        terminal = prepared.pre_sampling_terminal
        assert terminal is not None
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -0.01)
        evidence = terminal.info["wrapper_evidence"]
        self.assertEqual(evidence["event"], "prompt_capacity_finalization")
        self.assertEqual(evidence["action_prompt_tokens"], 24_600)
        self.assertEqual(evidence["effective_prompt_capacity"], 24_576)
        self.assertEqual(value._policy_step_count, 39)
        self.assertEqual(value._native_call_count, 39)
        value._request.assert_called_once_with(
            "POST", "horizon", json={"id": value.env_id}
        )

    def test_candidate_over_capacity_allows_only_one_final_ordinary_action(self) -> None:
        value = client(policy_steps=39)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        selected = value.prepare_policy_turn(pressure(action_tokens=24_400))

        self.assertIsNone(selected)
        self.assertTrue(value._checkpoint_capacity_terminal_after_ordinary)
        value._step_native_policy_action = Mock(
            side_effect=lambda _action: (
                setattr(value, "_policy_step_count", value._policy_step_count + 1)
                or setattr(value, "_native_call_count", value._native_call_count + 1)
                or failed_output(workspace_changed=True, state="final observation")
            )
        )
        value._request = Mock(return_value=horizon_response())
        output = value.step('shell_command {"command":"pytest","workdir":"."}')

        self.assertTrue(output.done)
        self.assertEqual(
            output.info["wrapper_evidence"]["capacity_termination_reason"],
            "checkpoint_request_does_not_fit",
        )

    def test_failed_cycles_are_bounded_per_context_epoch(self) -> None:
        value = client(policy_steps=20)
        value.metadata = {"configured_max_policy_turns": 45, "max_steps": 45}
        value._checkpoint_cycle_index = 2

        selected = value.prepare_policy_turn(pressure(action_tokens=19_500))

        self.assertIsNone(selected)
        self.assertTrue(value._checkpoint_capacity_terminal_after_ordinary)

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
