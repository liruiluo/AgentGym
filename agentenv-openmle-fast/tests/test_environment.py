from __future__ import annotations

import hashlib
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentenv_openmle_fast.audit import OpenMLEFastAuditSink
from agentenv_openmle_fast.dataset import OpenMLEFastDataset
from agentenv_openmle_fast.deadline import MonotonicDeadline
from agentenv_openmle_fast.environment import (
    OpenMLEFastEpisodeManager,
    _bound_text,
    _read_submission,
)
from agentenv_openmle_fast.executor import (
    LocalCPUExecutionBackend,
    OpenMLEFastExecutor,
    OpenMLEFastResourceLimits,
)
from agentenv_openmle_fast.grader_client import PrivateGraderClient
from agentenv_openmle_fast.grader_protocol import GradeResult
from agentenv_openmle_fast.materializer import OpenMLEFastWorkspaceMaterializer
from agentenv_openmle_fast.private_grader import PrivateGraderService
from agentenv_openmle_fast.private_grader_runner import (
    LocalCPUPrivateGraderBackend,
    PrivateGraderLimits,
)
from tests.support import (
    PRIVATE_RUNTIME_DIGEST,
    RELEASE_REVISION,
    FakeWorkspaceMountBackend,
    GraderServiceThread,
    create_fixture,
)


class _FaultingGrader:
    def grade(self, **_kwargs):
        raise RuntimeError("secret grader path /private/answer.csv")


class _CountingGrader:
    def __init__(self) -> None:
        self.calls = 0

    def grade(self, **_kwargs):
        self.calls += 1
        raise AssertionError("expired episode reached the private grader")


class _PoisonedRewardGrader:
    def grade(self, **kwargs):
        submission_sha256 = hashlib.sha256(kwargs["submission"]).hexdigest()
        return GradeResult(
            request_id=kwargs["request_id"],
            episode_id=kwargs["episode_id"],
            task_id=kwargs["task_id"],
            grader_binding_sha256=kwargs["grader_binding_sha256"],
            package_identity_sha256=kwargs["package_identity_sha256"],
            baseline_score=kwargs["baseline_score"],
            ideal_score=kwargs["ideal_score"],
            submission_sha256=submission_sha256,
            submission_valid=True,
            native_score=0.0,
            higher_is_better=kwargs["higher_is_better"],
            normalized_reward=-0.5,
            improved_over_baseline=True,
            runtime_success=True,
            terminal_reason="graded_submission",
            classification="graded",
            audit_digest="d" * 64,
        )


class _DeadlineRecordingGrader:
    def __init__(self) -> None:
        self.deadline: MonotonicDeadline | None = None

    def grade(self, *, deadline, **kwargs):
        self.deadline = deadline
        return GradeResult(
            request_id=kwargs["request_id"],
            episode_id=kwargs["episode_id"],
            task_id=kwargs["task_id"],
            grader_binding_sha256=kwargs["grader_binding_sha256"],
            package_identity_sha256=kwargs["package_identity_sha256"],
            baseline_score=kwargs["baseline_score"],
            ideal_score=kwargs["ideal_score"],
            submission_sha256=hashlib.sha256(kwargs["submission"]).hexdigest(),
            submission_valid=False,
            native_score=None,
            higher_is_better=kwargs["higher_is_better"],
            normalized_reward=-1.0,
            improved_over_baseline=False,
            runtime_success=False,
            terminal_reason="invalid_submission",
            classification="invalid_submission",
            audit_digest="e" * 64,
        )


class _FailSecondAudit:
    def __init__(self) -> None:
        self.calls = 0

    def emit(self, **_kwargs):
        self.calls += 1
        if self.calls == 2:
            raise OSError("audit disk unavailable")
        return "a" * 64


class OpenMLEFastEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="openmle-env-test-")
        self.root = Path(self.temporary.name)
        self.fixture = create_fixture(self.root)
        self.dataset = OpenMLEFastDataset(
            manifest_path=Path(self.fixture["manifest"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_manifest_sha256=str(self.fixture["manifest_sha256"]),
            expected_release_revision=RELEASE_REVISION,
            expected_role="gate_only",
        )
        self.socket_path = self.root / "grader.sock"
        self.service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            audit_root=Path(self.fixture["audit_root"]),
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=LocalCPUPrivateGraderBackend(PrivateGraderLimits.frozen_v1()),
        )
        self.thread = GraderServiceThread(self.service)
        self.thread.__enter__()
        self.limits = OpenMLEFastResourceLimits.frozen_v1()
        self.manager = self.build_manager(
            PrivateGraderClient(
                endpoint=self.socket_path,
                credential_path=Path(self.fixture["credential"]),
                timeout_seconds=5.0,
            )
        )

    def tearDown(self) -> None:
        self.thread.__exit__(None, None, None)
        self.temporary.cleanup()

    def build_manager(self, grader, *, audit_sink=None):
        return OpenMLEFastEpisodeManager(
            dataset=self.dataset,
            materializer=OpenMLEFastWorkspaceMaterializer(
                Path(self.fixture["episodes_root"]),
                runner_workspace_parent=Path(self.fixture["episodes_root"]),
                workspace_bytes=2 * 1024**3,
                max_files=100_000,
                mount_backend=FakeWorkspaceMountBackend(),
            ),
            executor_factory=lambda: OpenMLEFastExecutor(
                limits=self.limits,
                backend=LocalCPUExecutionBackend(self.limits),
            ),
            grader_client=grader,
            limits=self.limits,
            runtime_metadata={
                "runtime_source": {
                    "outer_commit": "1" * 40,
                    "inner_commit": "2" * 40,
                },
                "executor_runtime_digest": "sha256:" + "3" * 64,
            },
            audit_sink=audit_sink,
        )

    def reset(self, manager=None):
        manager = manager or self.manager
        slot = manager.create()
        step = manager.reset(slot, 0)
        self.assertFalse(step.done)
        return manager, slot, step

    def test_observations_report_the_shared_action_budget_after_reset_and_step(
        self,
    ) -> None:
        manager, slot, reset = self.reset()
        self.assertIn(
            "[OpenMLE action budget: action 0 completed; 30 actions remain.]",
            reset.observation,
        )
        first = manager.step(slot, "not a tool")
        self.assertIn(
            "[OpenMLE action budget: action 1 completed; 29 actions remain.]",
            first.observation,
        )
        self.assertLessEqual(
            len(first.observation.encode("utf-8")),
            self.limits.observation_bytes,
        )

    def test_protected_patch_attempt_is_terminal_minus_one(self) -> None:
        manager, slot, _ = self.reset()
        terminal = manager.step(
            slot,
            "apply_patch\n*** Begin Patch\n*** Delete File: TASK.md\n*** End Patch",
        )
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertEqual(
            terminal.info["terminal_reason"],
            "immutable_public_tree_mutation_attempt",
        )
        self.assertEqual(terminal.info["counters"]["execution_attempt_count"], 0)

    def test_submit_is_terminal_and_grades_exactly_once(self) -> None:
        manager, slot, _ = self.reset()
        patch = manager.step(
            slot,
            "apply_patch\n*** Begin Patch\n*** Add File: submission.csv\n"
            "+id,target\n+3,1\n+4,2\n*** End Patch",
        )
        self.assertEqual(patch.reward, 0.0)
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 1.0)
        self.assertEqual(terminal.info["counters"]["action_count"], 2)
        self.assertEqual(terminal.info["counters"]["grading_count"], 1)
        with self.assertRaises(RuntimeError):
            manager.step(slot, "submit")

    def test_action_30_non_submit_executes_then_terminates_minus_one(self) -> None:
        manager, slot, _ = self.reset()
        for index in range(29):
            step = manager.step(slot, "not a tool")
            self.assertFalse(step.done, index)
            self.assertEqual(step.reward, 0.0)
        terminal = manager.step(
            slot,
            'shell_command {"command":"printf final > final.txt"}',
        )
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertEqual(terminal.info["terminal_reason"], "action_budget_exhausted")
        self.assertEqual(terminal.info["counters"]["action_count"], 30)
        workspace = manager._testing_policy_root(slot)
        self.assertEqual((workspace / "final.txt").read_text(), "final")

    def test_expired_episode_is_charged_then_rejected_before_submit(self) -> None:
        grader = _CountingGrader()
        manager = self.build_manager(grader)
        manager, slot, _ = self.reset(manager)
        episode = manager._slot(slot).episode
        assert episode is not None
        episode.started_monotonic = (
            time.monotonic() - self.limits.episode_wall_ms / 1000.0 - 1.0
        )
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertEqual(terminal.info["terminal_reason"], "episode_wall_limit")
        self.assertEqual(terminal.info["counters"]["action_count"], 1)
        self.assertEqual(terminal.info["counters"]["grading_count"], 0)
        self.assertEqual(grader.calls, 0)

    def test_shell_is_capped_by_remaining_episode_deadline(self) -> None:
        original_limits = self.limits
        self.limits = replace(original_limits, episode_wall_ms=200)
        try:
            manager = self.build_manager(_FaultingGrader())
            manager, slot, _ = self.reset(manager)
            started = time.monotonic()
            terminal = manager.step(
                slot,
                "shell_command "
                '{"command":"python3 -c \'import time; time.sleep(2)\'",'
                '"timeout_ms":20000}',
            )
            elapsed = time.monotonic() - started
        finally:
            self.limits = original_limits
        self.assertLess(elapsed, 0.8)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertIn(
            terminal.info["terminal_reason"],
            {"episode_wall_limit", "wall_timeout"},
        )
        self.assertEqual(terminal.info["counters"]["execution_action_count"], 1)
        self.assertEqual(terminal.info["counters"]["execution_attempt_count"], 1)
        self.assertEqual(terminal.info["counters"]["execution_completed_count"], 0)

    def test_submit_receives_only_the_remaining_episode_deadline(self) -> None:
        grader = _DeadlineRecordingGrader()
        manager = self.build_manager(grader)
        manager, slot, _ = self.reset(manager)
        episode = manager._slot(slot).episode
        assert episode is not None
        episode.started_monotonic = (
            time.monotonic() - self.limits.episode_wall_ms / 1000.0 + 0.5
        )
        expected_expiry = (
            episode.started_monotonic + self.limits.episode_wall_ms / 1000.0
        )
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.info["counters"]["grading_count"], 1)
        self.assertIsNotNone(grader.deadline)
        assert grader.deadline is not None
        self.assertEqual(grader.deadline.expires_at, expected_expiry)

    def test_compound_python_counters_are_monotone(self) -> None:
        manager, slot, _ = self.reset()
        first = manager.step(
            slot,
            'shell_command {"command":"python3 -c \\"print(1)\\"; '
            'python3 -c \\"print(2)\\""}',
        )
        counters = first.info["counters"]
        self.assertEqual(counters["execution_action_count"], 1)
        self.assertEqual(counters["execution_attempt_count"], 2)
        self.assertEqual(counters["execution_completed_count"], 2)
        second = manager.step(slot, "malformed")
        self.assertEqual(second.info["counters"]["action_count"], 2)
        self.assertEqual(second.info["counters"]["execution_attempt_count"], 2)

    def test_missing_submission_is_policy_failure(self) -> None:
        manager, slot, _ = self.reset()
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertEqual(terminal.info["counters"]["grading_count"], 1)

    def test_grader_fault_is_sanitized_null_reward_truncation(self) -> None:
        manager = self.build_manager(_FaultingGrader())
        manager, slot, _ = self.reset(manager)
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertIsNone(terminal.reward)
        self.assertTrue(terminal.info["truncated"])
        self.assertEqual(
            terminal.info["terminal_reason"], "grader_infrastructure_fault"
        )
        self.assertNotIn("private", terminal.observation.lower())
        self.assertNotIn("answer", terminal.observation.lower())

    def test_poisoned_private_reward_is_recomputed_and_rejected(self) -> None:
        manager = self.build_manager(_PoisonedRewardGrader())
        manager, slot, _ = self.reset(manager)
        manager.step(
            slot,
            "apply_patch\n*** Begin Patch\n*** Add File: submission.csv\n"
            "+id,target\n+3,1\n+4,2\n*** End Patch",
        )
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertTrue(terminal.info["truncated"])
        self.assertIsNone(terminal.reward)
        self.assertEqual(terminal.info["terminal_reason"], "grader_binding_fault")

    def test_horizon_does_not_add_an_action_or_grade(self) -> None:
        manager, slot, _ = self.reset()
        manager.step(slot, "malformed")
        terminal = manager.finalize_horizon(slot)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertEqual(terminal.info["counters"]["action_count"], 1)
        self.assertEqual(terminal.info["counters"]["grading_count"], 0)

    def test_horizon_honors_the_absolute_episode_deadline(self) -> None:
        manager, slot, _ = self.reset()
        episode = manager._slot(slot).episode
        assert episode is not None
        episode.started_monotonic = (
            time.monotonic() - self.limits.episode_wall_ms / 1000.0 - 0.1
        )
        terminal = manager.finalize_horizon(slot)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)
        self.assertEqual(terminal.info["terminal_reason"], "episode_wall_limit")
        self.assertEqual(terminal.info["counters"]["action_count"], 0)

    def test_remaining_cumulative_managed_budget_caps_next_python_action(self) -> None:
        manager, slot, _ = self.reset()
        episode = manager._slot(slot).episode
        assert episode is not None
        episode.counters.managed_runtime_wall_seconds = 119.95
        started = time.monotonic()
        terminal = manager.step(
            slot,
            'shell_command {"command":"python3 -c \\"import time; time.sleep(2)\\""}',
        )
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, -1.0)

    def test_submission_symlink_hardlink_and_fifo_fail_without_escape(self) -> None:
        outside = self.root / "outside.csv"
        outside.write_text("id,target\n3,1\n4,2\n", encoding="utf-8")
        cases = ("symlink", "hardlink", "fifo")
        for case in cases:
            with self.subTest(case=case):
                manager, slot, _ = self.reset()
                submission = manager._testing_policy_root(slot) / "submission.csv"
                if case == "symlink":
                    submission.symlink_to(outside)
                elif case == "hardlink":
                    os.link(outside, submission)
                else:
                    os.mkfifo(submission)
                terminal = manager.step(slot, "submit")
                self.assertTrue(terminal.done)
                self.assertEqual(terminal.reward, -1.0)
                self.assertEqual(terminal.info["counters"]["grading_count"], 1)
                self.assertEqual(
                    outside.read_text(encoding="utf-8"),
                    "id,target\n3,1\n4,2\n",
                )
                manager.close(slot)

    def test_close_is_idempotent_and_cleans_workspace(self) -> None:
        manager, slot, _ = self.reset()
        root = manager._testing_policy_root(slot).parent
        first = manager.close(slot)
        second = manager.close(slot)
        self.assertTrue(first["closed"])
        self.assertTrue(second["already_closed"])
        self.assertFalse(root.exists())

    def test_observation_byte_and_token_caps_are_hard_bounds(self) -> None:
        bounded = _bound_text("😀" * 100_000, self.limits, 1_024)
        self.assertLessEqual(len(bounded.encode("utf-8")), 1_024)

    def test_submit_freezes_workspace_before_beneath_only_read(self) -> None:
        manager, slot, _ = self.reset()
        manager.step(
            slot,
            "apply_patch\n*** Begin Patch\n*** Add File: submission.csv\n"
            "+id,target\n+3,1\n+4,2\n*** End Patch",
        )
        from agentenv_openmle_fast import environment as environment_module

        original = environment_module._read_submission

        def assert_frozen(root, maximum, *, deadline):
            self.assertEqual(root.stat().st_mode & 0o777, 0o555)
            return original(root, maximum, deadline=deadline)

        with patch.object(environment_module, "_read_submission", assert_frozen):
            terminal = manager.step(slot, "submit")
        self.assertEqual(terminal.reward, 1.0)
        self.assertTrue(terminal.info["sandbox_freeze"]["processes_reaped"])

    def test_cleanup_failure_retains_handle_for_retry(self) -> None:
        manager, slot, _ = self.reset()
        workspace = manager._testing_policy_root(slot).parent
        original = manager.materializer.close
        calls = 0

        def fail_once(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected cleanup fault")
            return original(value)

        manager.materializer.close = fail_once
        first = manager.close(slot)
        self.assertFalse(first["closed"])
        self.assertTrue(first["retryable"])
        self.assertTrue(workspace.exists())
        second = manager.close(slot)
        self.assertTrue(second["closed"])
        self.assertFalse(workspace.exists())

    def test_audit_failure_becomes_null_reward_truncation(self) -> None:
        sink = _FailSecondAudit()
        manager = self.build_manager(
            PrivateGraderClient(
                endpoint=self.socket_path,
                credential_path=Path(self.fixture["credential"]),
                timeout_seconds=5.0,
            ),
            audit_sink=sink,
        )
        manager, slot, reset = self.reset(manager)
        self.assertEqual(reset.info["audit_digest"], "a" * 64)
        terminal = manager.step(slot, "malformed")
        self.assertTrue(terminal.done)
        self.assertTrue(terminal.info["truncated"])
        self.assertIsNone(terminal.reward)
        self.assertEqual(terminal.info["terminal_reason"], "audit_infrastructure_fault")
        self.assertEqual(len(terminal.info["unaudited_evidence_sha256"]), 64)

    def test_receipts_carry_join_provenance_and_audit_digest(self) -> None:
        sink = OpenMLEFastAuditSink(self.root / "public-audit")
        manager = self.build_manager(
            PrivateGraderClient(
                endpoint=self.socket_path,
                credential_path=Path(self.fixture["credential"]),
                timeout_seconds=5.0,
            ),
            audit_sink=sink,
        )
        _, _slot, reset = self.reset(manager)
        for key in (
            "manifest_sha256",
            "release_revision",
            "archive_sha256",
            "package_identity_sha256",
            "task_spec_sha256",
            "grader_binding_sha256",
            "runtime_source",
            "executor_runtime_digest",
            "boundary_contracts",
            "audit_digest",
        ):
            self.assertIn(key, reset.info)
        self.assertEqual(len(reset.info["audit_digest"]), 64)

    def test_reset_constructor_and_cleanup_fault_retains_workspace_handle(self) -> None:
        manager = self.build_manager(_FaultingGrader())
        slot = manager.create()
        original_close = manager.materializer.close
        manager.executor_factory = lambda: (_ for _ in ()).throw(
            RuntimeError("injected executor construction fault")
        )
        manager.materializer.close = lambda _workspace: (_ for _ in ()).throw(
            OSError("injected reset cleanup fault")
        )
        step = manager.reset(slot, 0)
        self.assertTrue(step.done)
        self.assertTrue(step.info["truncated"])
        retained = manager._slot(slot).episode
        assert retained is not None
        self.assertIsNotNone(retained.workspace)
        manager.materializer.close = original_close
        receipt = manager.close(slot)
        self.assertTrue(receipt["closed"])
        self.assertTrue(receipt["workspace_removed"])

    def test_fully_sparse_submission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="openmle-sparse-submission-") as raw:
            root = Path(raw)
            submission = root / "submission.csv"
            with submission.open("wb") as handle:
                handle.truncate(1024 * 1024)
            self.assertEqual(submission.stat().st_blocks, 0)
            with self.assertRaisesRegex(ValueError, "sparse"):
                _read_submission(
                    root,
                    2 * 1024 * 1024,
                    deadline=MonotonicDeadline.after_ms(1_000),
                )


if __name__ == "__main__":
    unittest.main()
