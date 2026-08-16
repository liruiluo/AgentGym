from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_mlebench_lite.actions import parse_policy_action
from agentenv_mlebench_lite.dataset import load_lite_dataset
from agentenv_mlebench_lite.environment import MLEBenchLiteEpisodeManager
from agentenv_mlebench_lite.executor import MLEBenchLiteExecutorError, SandboxExecutor
from agentenv_mlebench_lite.identity import (
    LITE_COMPETITION_IDS,
    UPSTREAM_COMMIT,
    load_official_lite_identity,
)
from agentenv_mlebench_lite.resources import zero_resource_usage
from agentenv_mlebench_lite.workspace import (
    MODE_AMG_COMPACTION_ONLY,
    MODE_AMG_MEMORY,
    MODE_NATIVE,
    MODES,
    MLEBenchLiteWorkspaceError,
    WorkspaceManager,
)

from tests.support import (
    FAKE_RUNNER_SHA256,
    FAKE_RUNTIME_DIGEST,
    RecordingFormalBackend,
    write_fixture,
)


class MLEBenchLiteActionsEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mlebench-lite-env-")
        self.root = Path(self.temporary.name)
        self.fixture = write_fixture(self.root)
        identity = load_official_lite_identity(
            self.fixture["upstream_root"],
            commit_resolver=lambda _root: UPSTREAM_COMMIT,
        )
        self.dataset = load_lite_dataset(
            identity=identity,
            manifest_path=self.fixture["manifest_path"],
            expected_manifest_sha256=self.fixture["manifest_sha256"],
            data_root=self.fixture["data_root"],
        )
        self.backends: list[RecordingFormalBackend] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_manager(
        self,
        max_actions: int = 20,
        max_submission_bytes: int = 10_000,
    ) -> MLEBenchLiteEpisodeManager:
        def executor_factory():
            backend = RecordingFormalBackend()
            self.backends.append(backend)
            return SandboxExecutor(
                backend,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            )

        return MLEBenchLiteEpisodeManager(
            dataset=self.dataset,
            workspace_manager=WorkspaceManager(self.fixture["episodes_root"]),
            executor_factory=executor_factory,
            max_actions=max_actions,
            max_submission_bytes=max_submission_bytes,
            runner_sha256=FAKE_RUNNER_SHA256,
            runtime_digest=FAKE_RUNTIME_DIGEST,
        )

    def build_manager_with_backends(
        self,
        backends: list[RecordingFormalBackend],
        **kwargs,
    ) -> MLEBenchLiteEpisodeManager:
        queue = list(backends)

        def executor_factory():
            backend = queue.pop(0)
            self.backends.append(backend)
            return SandboxExecutor(
                backend,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            )

        return MLEBenchLiteEpisodeManager(
            dataset=self.dataset,
            workspace_manager=WorkspaceManager(self.fixture["episodes_root"]),
            executor_factory=executor_factory,
            runner_sha256=FAKE_RUNNER_SHA256,
            runtime_digest=FAKE_RUNTIME_DIGEST,
            **kwargs,
        )

    def reset(self, manager, mode=MODE_AMG_MEMORY, data_idx=0):
        slot = manager.create(mode=mode)
        step = manager.reset(slot, data_idx)
        self.assertFalse(step.done)
        return slot, step

    def test_exact_parser_supports_all_public_action_kinds(self) -> None:
        cases = {
            'inspect {"path":"/home/data/train.csv","offset":0,"max_bytes":64}': "inspect",
            'edit {"path":"/home/workspace/train.py","content":"print(1)\\n"}': "edit",
            'shell {"command":"python /home/workspace/train.py","timeout_ms":1000}': "shell",
            "submit": "submit",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_policy_action(raw).kind, expected)
        for raw in (
            "",
            "submit now",
            "```submit```",
            "inspect {}",
            "unknown {}",
            'memory_read {"path":"notes.md"}',
            'memory_write {"path":"notes.md","content":"x"}',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_policy_action(raw).kind, "parser_error")
        self.assertEqual(
            parse_policy_action(
                'edit {"path":"/home/workspace/x","content":"\\ud800"}'
            ).kind,
            "parser_error",
        )

    def test_write_compaction_then_later_read_uses_one_persistent_workspace(
        self,
    ) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager)
        written = manager.step(
            slot,
            'edit {"path":"/run/amg_memory/notes.md",'
            '"content":"hypothesis: normalize labels\\n"}',
        )
        self.assertEqual(written.info["counters"]["action_count"], 1)
        self.assertEqual(
            written.info["external_memory_operation"],
            "write",
        )
        compacted = manager.step(
            slot,
            "durable note key notes.md; next inspect labels",
            control="compaction",
        )
        self.assertEqual(compacted.info["counters"]["action_count"], 2)
        self.assertEqual(
            compacted.info["control_receipt"]["counter_delta"],
            {
                "action_count": 1,
                "native_action_count": 0,
                "execution_count": 0,
                "grading_count": 0,
                **zero_resource_usage(),
            },
        )
        read = manager.step(
            slot,
            'inspect {"path":"/run/amg_memory/notes.md",'
            '"offset":0,"max_bytes":1024}',
        )
        self.assertIn("normalize labels", read.observation)
        self.assertEqual(read.info["counters"]["action_count"], 3)
        self.assertEqual(read.info["external_memory_operation"], "read")

    def test_external_memory_mount_is_separate_from_task_workspace(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_AMG_MEMORY)
        workspace = manager._testing_workspace(slot)
        self.assertIsNotNone(workspace.memory_root)
        self.assertNotEqual(workspace.memory_root, workspace.workspace_root)

        written = manager.step(
            slot,
            'edit {"path":"/run/amg_memory/notes.md","content":"private clue"}',
        )
        self.assertEqual(written.info["action_kind"], "edit")
        self.assertEqual(written.info["external_memory_operation"], "write")
        indirect = manager.step(
            slot,
            'shell {"command":"cat /home/workspace/.agent_*/notes.md 2>/dev/null || printf unavailable","timeout_ms":1000}',
        )
        self.assertNotIn("external_memory_operation", indirect.info)
        self.assertNotIn("private clue", indirect.observation)
        read = manager.step(
            slot,
            'inspect {"path":"/run/amg_memory/notes.md","offset":0,"max_bytes":1024}',
        )
        self.assertEqual(read.info["action_kind"], "inspect")
        self.assertEqual(read.info["external_memory_operation"], "read")
        self.assertIn("private clue", read.observation)

        for mode in (MODE_NATIVE, MODE_AMG_COMPACTION_ONLY):
            with self.subTest(mode=mode):
                disabled_slot, _ = self.reset(manager, mode)
                disabled = manager._testing_workspace(disabled_slot)
                self.assertIsNone(disabled.memory_root)
                denied = manager.step(
                    disabled_slot,
                    'edit {"path":"/run/amg_memory/notes.md","content":"forbidden"}',
                )
                self.assertEqual(denied.info["action_kind"], "edit")
                self.assertEqual(denied.observation, "Path is unavailable.")
                self.assertNotIn("external_memory_operation", denied.info)

    def test_failed_memory_inspect_and_edit_emit_no_access_receipt(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_AMG_MEMORY)

        missing = manager.step(
            slot,
            'inspect {"path":"/run/amg_memory/missing","max_bytes":64}',
        )
        self.assertEqual(missing.observation, "Path is unavailable.")
        self.assertNotIn("external_memory_operation", missing.info)

        denied = manager.step(
            slot,
            'edit {"path":"/run/amg_memory","content":"not a file"}',
        )
        self.assertEqual(denied.observation, "Path is unavailable.")
        self.assertNotIn("external_memory_operation", denied.info)

        with patch(
            "agentenv_mlebench_lite.workspace.os.fsync",
            side_effect=OSError(errno.ENOSPC, "injected no space"),
        ):
            failed = manager.step(
                slot,
                'edit {"path":"/run/amg_memory/note","content":"x"}',
            )
        self.assertTrue(failed.done)
        self.assertEqual(failed.info["terminal_reason"], "infrastructure_failure")
        self.assertNotIn("external_memory_operation", failed.info)

    def test_shell_memory_receipt_comes_only_from_sandbox_access_evidence(
        self,
    ) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_AMG_MEMORY)
        backend = self.backends[-1]

        mentioned = manager.step(
            slot,
            'shell {"command":"printf /run/amg_memory","timeout_ms":1000}',
        )
        self.assertNotIn("external_memory_operation", mentioned.info)

        commands = (
            "cat /run/amg_memory/notes.md",
            'root=/run/amg_memory; cat "$root/notes.md"',
            "cat /run/amg_memory/*.md",
            "find /run/amg_memory -maxdepth 1 -type f -print",
        )
        for command in commands:
            with self.subTest(command=command):
                backend.next_external_memory_access = "read"
                step = manager.step(
                    slot,
                    "shell "
                    + json.dumps({"command": command, "timeout_ms": 1000}),
                )
                self.assertEqual(step.info["external_memory_operation"], "read")

        backend.next_external_memory_access = "read_write"
        both = manager.step(
            slot,
            'shell {"command":"cat /run/amg_memory/in >> '
            '/run/amg_memory/out","timeout_ms":1000}',
        )
        self.assertEqual(both.info["external_memory_operation"], "read_write")

        ordinary = manager.step(
            slot,
            'edit {"path":"/home/workspace/note.txt",'
            '"content":"mention /run/amg_memory only"}',
        )
        self.assertNotIn("external_memory_operation", ordinary.info)

    def test_compaction_only_accepts_compaction_but_denies_memory_residue(
        self,
    ) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_AMG_COMPACTION_ONLY)
        workspace = manager._testing_workspace(slot)
        self.assertIsNone(workspace.memory_root)

        written = manager.step(
            slot,
            'edit {"path":"/home/workspace/notes.md","content":"durable state"}',
        )
        self.assertEqual(written.observation, "Edit completed.")
        compacted = manager.step(slot, "objective fixed", control="compaction")
        self.assertEqual(compacted.info["action_kind"], "compaction")
        self.assertTrue(compacted.info["control_receipt"]["accepted"])
        self.assertEqual(compacted.info["counter_delta"]["action_count"], 1)
        survived = manager.step(
            slot,
            'inspect {"path":"/home/workspace/notes.md"}',
        )
        self.assertEqual(survived.observation, "durable state")

        for raw in (
            'inspect {"path":"/run/amg_memory/note"}',
            'edit {"path":"/run/amg_memory/note","content":"x"}',
        ):
            denied = manager.step(slot, raw)
            self.assertEqual(denied.observation, "Path is unavailable.")
            self.assertNotIn("external_memory_operation", denied.info)
        self.assertIsNone(workspace.memory_root)

    def test_compacting_arms_have_identical_server_compaction_receipts(self) -> None:
        manager = self.build_manager()
        results = []
        for mode in (MODE_AMG_COMPACTION_ONLY, MODE_AMG_MEMORY):
            slot, _ = self.reset(manager, mode)
            results.append(manager.step(slot, "same handoff", control="compaction"))
        self.assertEqual(results[0].observation, results[1].observation)
        self.assertEqual(results[0].info, results[1].info)

    def test_every_action_type_including_compaction_uses_one_budget(self) -> None:
        manager = self.build_manager(max_actions=6)
        slot, _ = self.reset(manager)
        actions = [
            ("not an action", None),
            ('inspect {"path":"/home/data/train.csv"}', None),
            (
                'edit {"path":"/run/amg_memory/note","content":"durable"}',
                None,
            ),
            ('shell {"command":"python -V","timeout_ms":1000}', None),
            ("short handoff", "compaction"),
            (
                'edit {"path":"/home/submission/submission.csv","content":"id,target\\n1,0\\n"}',
                None,
            ),
        ]
        for expected_count, (raw, control) in enumerate(actions, start=1):
            step = manager.step(slot, raw, control=control)
            self.assertEqual(step.info["counters"]["action_count"], expected_count)
        self.assertTrue(step.done)
        self.assertEqual(step.info["terminal_reason"], "action_budget_exhausted")

    def test_budget_terminal_compaction_is_counted_server_side(self) -> None:
        manager = self.build_manager(max_actions=1)
        slot, _ = self.reset(manager)
        workspace = manager._testing_workspace(slot)
        terminal = manager.step(slot, "handoff", control="compaction")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.info["counters"]["action_count"], 1)
        receipt = terminal.info["control_receipt"]
        self.assertEqual(receipt["action_count_before"], 0)
        self.assertEqual(receipt["action_count_after"], 1)
        self.assertEqual(
            receipt["counter_delta"],
            {
                "action_count": 1,
                "native_action_count": 0,
                "execution_count": 0,
                "grading_count": 0,
                **zero_resource_usage(),
            },
        )
        self.assertEqual(terminal.info["terminal_reason"], "action_budget_exhausted")
        self.assertEqual(len(self.backends[0].torn_down), 1)
        self.assertFalse(workspace.episode_root.exists())

    def test_charged_compaction_rejection_has_a_sequence_receipt(self) -> None:
        manager = self.build_manager(max_actions=3)
        slot, _ = self.reset(manager)
        rejected = manager.step(
            slot,
            "x" * 9000,
            control="compaction",
            expected_action_count=0,
        )
        self.assertFalse(rejected.done)
        self.assertEqual(rejected.info["counters"]["action_count"], 1)
        self.assertEqual(
            rejected.info["control_receipt"],
            {
                "schema": "mlebench_lite_compaction_receipt_v2",
                "action_count_before": 0,
                "action_count_after": 1,
                "counter_delta": rejected.info["counter_delta"],
                "accepted": False,
            },
        )

    def test_failed_preflight_attempts_supervised_cleanup(self) -> None:
        backend = RecordingFormalBackend()
        backend.attestation_override = {"invalid": True}
        manager = self.build_manager_with_backends([backend])
        slot = manager.create(mode=MODE_NATIVE)
        with self.assertRaises(MLEBenchLiteExecutorError):
            manager.reset(slot, 0)
        self.assertEqual(len(backend.torn_down), 1)
        self.assertEqual(list(self.fixture["episodes_root"].iterdir()), [])

    def test_executor_factory_and_contract_fail_before_workspace_creation(self) -> None:
        workspace_manager = WorkspaceManager(self.fixture["episodes_root"])

        def broken_factory():
            raise OSError("injected factory failure")

        manager = MLEBenchLiteEpisodeManager(
            dataset=self.dataset,
            workspace_manager=workspace_manager,
            executor_factory=broken_factory,
            runner_sha256=FAKE_RUNNER_SHA256,
            runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        slot = manager.create(mode=MODE_NATIVE)
        with (
            patch.object(
                workspace_manager, "create", wraps=workspace_manager.create
            ) as create,
            self.assertRaises(OSError),
        ):
            manager.reset(slot, 0)
        create.assert_not_called()
        self.assertEqual(list(self.fixture["episodes_root"].iterdir()), [])

        backend = RecordingFormalBackend()
        mismatch = SandboxExecutor(
            backend,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            expected_resource_contract_sha256="f" * 64,
        )
        mismatch_manager = MLEBenchLiteEpisodeManager(
            dataset=self.dataset,
            workspace_manager=workspace_manager,
            executor_factory=lambda: mismatch,
            runner_sha256=FAKE_RUNNER_SHA256,
            runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        mismatch_slot = mismatch_manager.create(mode=MODE_NATIVE)
        with self.assertRaises(MLEBenchLiteExecutorError):
            mismatch_manager.reset(mismatch_slot, 0)
        self.assertEqual(list(self.fixture["episodes_root"].iterdir()), [])

    def test_reset_keeps_provisional_episode_when_workspace_remove_needs_retry(
        self,
    ) -> None:
        failed = RecordingFormalBackend()
        failed.attestation_override = {"invalid": True}
        healthy = RecordingFormalBackend()
        manager = self.build_manager_with_backends([failed, healthy])
        slot = manager.create(mode=MODE_NATIVE)
        original_remove = manager.workspace_manager.remove
        attempts = 0

        def retry_remove(workspace):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise MLEBenchLiteWorkspaceError("injected remove failure")
            return original_remove(workspace)

        with patch.object(
            manager.workspace_manager, "remove", side_effect=retry_remove
        ):
            with self.assertRaises(MLEBenchLiteWorkspaceError):
                manager.reset(slot, 0)
            provisional = manager._testing_workspace(slot)
            self.assertTrue(provisional.episode_root.exists())
            reset = manager.reset(slot, 1)
        self.assertFalse(reset.done)
        self.assertFalse(provisional.episode_root.exists())
        self.assertEqual(len(failed.torn_down), 1)

    def test_failed_cleanup_blocks_steps_but_reset_retries_cleanup(self) -> None:
        class RetryCleanupBackend(RecordingFormalBackend):
            def __init__(self):
                super().__init__()
                self.attestation_override = {"invalid": True}
                self.teardown_attempts = 0

            def teardown(self, **kwargs):
                self.teardown_attempts += 1
                if self.teardown_attempts == 1:
                    raise OSError("injected cleanup failure")
                self.attestation_override = None
                return super().teardown(**kwargs)

        failed = RetryCleanupBackend()
        healthy = RecordingFormalBackend()
        manager = self.build_manager_with_backends([failed, healthy])
        slot = manager.create(mode=MODE_NATIVE)
        with self.assertRaises(MLEBenchLiteExecutorError):
            manager.reset(slot, 0)
        with self.assertRaises(RuntimeError):
            manager.step(slot, "submit")
        reset = manager.reset(slot, 0)
        self.assertFalse(reset.done)
        self.assertEqual(failed.teardown_attempts, 2)

    def test_failed_create_rollback_is_retained_for_reset_and_close_retry(
        self,
    ) -> None:
        from agentenv_mlebench_lite import workspace as workspace_module

        for retry in ("reset", "close"):
            with self.subTest(retry=retry):
                create_failures = 0

                def fail_first_create(stage: str, _path: Path) -> None:
                    nonlocal create_failures
                    if stage == "submission_created" and create_failures == 0:
                        create_failures += 1
                        raise OSError("injected create failure")

                episodes = self.root / f"pending-create-{retry}-episodes"
                handoffs = self.root / f"pending-create-{retry}-handoffs"
                workspace_manager = WorkspaceManager(
                    episodes,
                    handoffs,
                    stage_hook=fail_first_create,
                )
                manager = MLEBenchLiteEpisodeManager(
                    dataset=self.dataset,
                    workspace_manager=workspace_manager,
                    executor_factory=lambda: SandboxExecutor(
                        RecordingFormalBackend(),
                        expected_runner_sha256=FAKE_RUNNER_SHA256,
                        expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                    ),
                    runner_sha256=FAKE_RUNNER_SHA256,
                    runtime_digest=FAKE_RUNTIME_DIGEST,
                )
                slot = manager.create(mode=MODE_NATIVE)
                original_remove_tree = workspace_module._remove_private_tree
                rollback_failures = 0

                def fail_first_rollback(path, original=original_remove_tree):
                    nonlocal rollback_failures
                    if (
                        Path(path).name.startswith(".creating-")
                        and rollback_failures == 0
                    ):
                        rollback_failures += 1
                        raise OSError(errno.EIO, "injected rollback failure")
                    return original(path)

                with (
                    patch(
                        "agentenv_mlebench_lite.workspace._remove_private_tree",
                        side_effect=fail_first_rollback,
                    ),
                    self.assertRaises(MLEBenchLiteWorkspaceError),
                ):
                    manager.reset(slot, 0)
                self.assertEqual(len(list(episodes.iterdir())), 1)
                with self.assertRaises(RuntimeError):
                    manager.step(slot, "submit")

                if retry == "reset":
                    reset = manager.reset(slot, 1)
                    self.assertFalse(reset.done)
                    manager.close(slot)
                else:
                    manager.close(slot)
                    with self.assertRaises(KeyError):
                        manager.capability_token(slot)
                self.assertEqual(list(episodes.iterdir()), [])
                self.assertEqual(list(handoffs.iterdir()), [])

    def test_close_rejects_a_stale_concurrent_reset_before_it_can_create(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_NATIVE)
        reset_has_slot = threading.Event()
        release_reset = threading.Event()
        reset_errors: list[BaseException] = []
        original_slot = manager._slot

        def pause_reset_after_slot_lookup(slot_id, capability_token=None):
            value = original_slot(slot_id, capability_token)
            if threading.current_thread().name == "stale-reset":
                reset_has_slot.set()
                if not release_reset.wait(timeout=5):
                    raise TimeoutError("reset barrier timed out")
            return value

        def stale_reset() -> None:
            try:
                manager.reset(slot, 1)
            except BaseException as exc:  # noqa: BLE001 - assert worker outcome
                reset_errors.append(exc)

        worker = threading.Thread(target=stale_reset, name="stale-reset")
        with patch.object(manager, "_slot", side_effect=pause_reset_after_slot_lookup):
            worker.start()
            self.assertTrue(reset_has_slot.wait(timeout=5))
            try:
                manager.close(slot)
            finally:
                release_reset.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(reset_errors), 1)
        self.assertIsInstance(reset_errors[0], KeyError)
        self.assertEqual(list(manager.workspace_manager.episodes_root.iterdir()), [])
        with self.assertRaises(KeyError):
            manager.capability_token(slot)

    def test_execution_infrastructure_fault_terminalizes_and_cleans_once(self) -> None:
        class BrokenExecutionBackend(RecordingFormalBackend):
            def execute(self, **_kwargs):
                raise OSError("runner transport disappeared")

        backend = BrokenExecutionBackend()
        manager = self.build_manager_with_backends([backend])
        slot, _ = self.reset(manager, MODE_NATIVE)
        workspace = manager._testing_workspace(slot)
        terminal = manager.step(
            slot,
            'shell {"command":"python -V","timeout_ms":1000}',
        )
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 0.0)
        self.assertEqual(terminal.observation, "Episode terminated.")
        self.assertEqual(terminal.info["terminal_reason"], "infrastructure_failure")
        self.assertNotIn("runner", repr(terminal.info).lower())
        self.assertEqual(len(backend.torn_down), 1)
        self.assertFalse(workspace.episode_root.exists())

    def test_storage_faults_terminalize_but_policy_unavailability_does_not(
        self,
    ) -> None:
        cases = ("inspect_eio", "edit_enospc", "submission_edquot")
        for kind in cases:
            with self.subTest(kind=kind):
                manager = self.build_manager()
                slot, _ = self.reset(manager, MODE_NATIVE)
                workspace = manager._testing_workspace(slot)
                if kind == "inspect_eio":
                    workspace.workspace_root.joinpath("value.txt").write_text(
                        "value", encoding="utf-8"
                    )
                    target = "agentenv_mlebench_lite.workspace.os.read"
                    action = 'inspect {"path":"/home/workspace/value.txt"}'
                    error = OSError(errno.EIO, "injected I/O failure")
                elif kind == "edit_enospc":
                    target = "agentenv_mlebench_lite.workspace.os.fsync"
                    action = 'edit {"path":"/home/workspace/value.txt","content":"x"}'
                    error = OSError(errno.ENOSPC, "injected no space")
                else:
                    workspace.submission_path.write_bytes(b"id,target\n1,0\n")
                    target = "agentenv_mlebench_lite.environment.os.read"
                    action = "submit"
                    error = OSError(errno.EDQUOT, "injected quota failure")
                with patch(target, side_effect=error):
                    terminal = manager.step(slot, action)
                self.assertTrue(terminal.done)
                self.assertEqual(terminal.reward, 0.0)
                self.assertEqual(terminal.observation, "Episode terminated.")
                self.assertEqual(
                    terminal.info["terminal_reason"], "infrastructure_failure"
                )
                self.assertNotIn("injected", repr(terminal.info))

        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_NATIVE)
        directory = manager._testing_workspace(slot).workspace_root / "directory"
        directory.mkdir()
        unavailable = manager.step(slot, 'inspect {"path":"/home/workspace/directory"}')
        self.assertFalse(unavailable.done)
        self.assertEqual(unavailable.observation, "Path is unavailable.")

    def test_submission_lstat_eio_is_an_infrastructure_terminal(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_NATIVE)
        workspace = manager._testing_workspace(slot)
        workspace.submission_path.write_bytes(b"id,target\n1,0\n")
        original_lstat = Path.lstat

        def fail_submission_lstat(path):
            if Path(path) == workspace.submission_path:
                raise OSError(errno.EIO, "injected lstat failure")
            return original_lstat(path)

        with patch.object(Path, "lstat", fail_submission_lstat):
            terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.info["terminal_reason"], "infrastructure_failure")

    def test_normal_nonzero_execution_remains_nonterminal_with_status_header(
        self,
    ) -> None:
        class NonzeroBackend(RecordingFormalBackend):
            def execute(self, *, workspace, command, timeout_ms, operation_id):
                return self._result(
                    workspace,
                    command,
                    timeout_ms,
                    operation_id=operation_id,
                    returncode=7,
                    stderr="ordinary program failure",
                )

        manager = self.build_manager_with_backends([NonzeroBackend()])
        slot, _ = self.reset(manager, MODE_NATIVE)
        result = manager.step(
            slot,
            'shell {"command":"false","timeout_ms":1000}',
        )
        self.assertFalse(result.done)
        self.assertTrue(
            result.observation.startswith(
                "[execution returncode=7 timed_out=false truncated=false]"
            )
        )

    def test_submit_receipt_is_allowlisted_and_contains_no_grader_details(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager)
        manager.step(
            slot,
            'edit {"path":"/home/submission/submission.csv",'
            '"content":"id,target\\n1,0\\n"}',
        )
        workspace = manager._testing_workspace(slot)
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 0.0)
        receipt = terminal.info["terminal_receipt"]
        self.assertEqual(
            set(receipt),
            {"competition_id", "submission_path", "submission_sha256"},
        )
        self.assertEqual(receipt["submission_path"], "/home/submission/submission.csv")
        public = repr(
            {"observation": terminal.observation, "info": terminal.info}
        ).lower()
        for forbidden in (
            "score",
            "valid",
            "grader",
            "private",
            str(self.fixture["data_root"]).lower(),
        ):
            self.assertNotIn(forbidden, public)
        self.assertEqual(len(self.backends[0].frozen), 1)
        self.assertEqual(len(self.backends[0].torn_down), 1)
        self.assertFalse(workspace.episode_root.exists())
        host_submission = manager.host_submission_path(slot)
        self.assertTrue(host_submission.is_file())
        self.assertEqual(host_submission.read_bytes(), b"id,target\n1,0\n")
        handoff_directory = host_submission.parent
        manifest = json.loads((handoff_directory / "handoff.json").read_text())
        self.assertEqual(
            set(manifest),
            {
                "schema",
                "episode_id",
                "mode",
                "competition_id",
                "submission_file",
                "submission_sha256",
                "runner_sha256",
                "runtime_digest",
                "resource_contract_sha256",
                "freeze_receipt",
                "teardown_receipt",
            },
        )
        self.assertEqual(manifest["mode"], MODE_AMG_MEMORY)
        self.assertEqual(stat.S_IMODE(handoff_directory.stat().st_mode), 0o500)
        self.assertEqual(stat.S_IMODE(host_submission.stat().st_mode), 0o400)
        self.assertEqual(
            stat.S_IMODE((handoff_directory / "handoff.json").stat().st_mode),
            0o400,
        )
        self.assertNotIn(str(host_submission), repr(terminal.info))
        with self.assertRaises(RuntimeError):
            manager.step(slot, "submit")

        host_submission.chmod(0o600)
        host_submission.write_bytes(b"id,target\n1,1\n")
        host_submission.chmod(0o400)
        with self.assertRaises(RuntimeError):
            manager.host_submission_path(slot)

    def test_sealed_grader_handoff_contract_is_equal_across_triad(self) -> None:
        manager = self.build_manager()
        terminal_receipts = []
        invariant_manifests = []
        receipt_shapes = []
        expected_manifest_keys = {
            "schema",
            "episode_id",
            "mode",
            "competition_id",
            "submission_file",
            "submission_sha256",
            "runner_sha256",
            "runtime_digest",
            "resource_contract_sha256",
            "freeze_receipt",
            "teardown_receipt",
        }
        for mode in MODES:
            slot, _ = self.reset(manager, mode)
            workspace = manager._testing_workspace(slot)
            manager.step(
                slot,
                'edit {"path":"/home/submission/submission.csv",'
                '"content":"id,target\\n1,0\\n"}',
            )
            terminal = manager.step(slot, "submit")
            self.assertTrue(terminal.done)
            self.assertFalse(workspace.episode_root.exists())
            terminal_receipts.append(terminal.info["terminal_receipt"])

            host_submission = manager.host_submission_path(slot)
            manifest = json.loads(
                (host_submission.parent / "handoff.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(manifest), expected_manifest_keys)
            self.assertEqual(manifest["mode"], mode)
            self.assertEqual(host_submission.read_bytes(), b"id,target\n1,0\n")
            self.assertEqual(stat.S_IMODE(host_submission.stat().st_mode), 0o400)
            self.assertEqual(stat.S_IMODE(host_submission.parent.stat().st_mode), 0o500)
            invariant_manifests.append(
                {
                    key: manifest[key]
                    for key in (
                        "schema",
                        "competition_id",
                        "submission_file",
                        "submission_sha256",
                        "runner_sha256",
                        "runtime_digest",
                        "resource_contract_sha256",
                    )
                }
            )
            receipt_shapes.append(
                (
                    set(manifest["freeze_receipt"]),
                    set(manifest["teardown_receipt"]),
                )
            )

        self.assertTrue(
            all(receipt == terminal_receipts[0] for receipt in terminal_receipts)
        )
        self.assertTrue(
            all(manifest == invariant_manifests[0] for manifest in invariant_manifests)
        )
        self.assertTrue(all(shape == receipt_shapes[0] for shape in receipt_shapes))

    def test_host_handoff_manifest_rejects_nested_type_confusion(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_NATIVE)
        manager.step(
            slot,
            'edit {"path":"/home/submission/submission.csv",'
            '"content":"id,target\\n1,0\\n"}',
        )
        terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        handoff = manager.host_submission_path(slot).parent
        manifest_path = handoff / "handoff.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["freeze_receipt"]["processes_reaped"] = 1
        manifest_path.chmod(0o600)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        manifest_path.chmod(0o400)
        with self.assertRaises(RuntimeError):
            manager.host_submission_path(slot)

    def test_handoff_publish_failure_is_generic_terminal_and_rolls_back(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager)
        manager.step(
            slot,
            'edit {"path":"/home/submission/submission.csv",'
            '"content":"id,target\\n1,0\\n"}',
        )
        workspace = manager._testing_workspace(slot)
        with patch.object(
            manager.workspace_manager,
            "publish_submission",
            side_effect=OSError("injected protected-store failure"),
        ):
            terminal = manager.step(slot, "submit")
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 0.0)
        self.assertEqual(terminal.observation, "Episode terminated.")
        self.assertEqual(terminal.info["terminal_reason"], "infrastructure_failure")
        self.assertNotIn("terminal_receipt", terminal.info)
        self.assertFalse(workspace.episode_root.exists())
        self.assertEqual(list(manager.workspace_manager.handoff_root.iterdir()), [])

    def test_staging_rollback_failure_is_retained_for_reset_and_close_retry(
        self,
    ) -> None:
        for retry in ("reset", "close"):
            with self.subTest(retry=retry):
                manager = self.build_manager()
                slot, _ = self.reset(manager, MODE_NATIVE)
                workspace = manager._testing_workspace(slot)
                workspace.submission_path.write_bytes(b"id,target\n1,0\n")
                from agentenv_mlebench_lite import workspace as workspace_module

                original_remove_tree = workspace_module._remove_private_tree

                def fail_staging_remove(path, original=original_remove_tree):
                    if Path(path).name.startswith(".staging-"):
                        raise OSError(errno.EIO, "injected rollback failure")
                    return original(path)

                with (
                    patch(
                        "agentenv_mlebench_lite.workspace._write_new_file",
                        side_effect=OSError(errno.ENOSPC, "injected staging ENOSPC"),
                    ),
                    patch(
                        "agentenv_mlebench_lite.workspace._remove_private_tree",
                        side_effect=fail_staging_remove,
                    ),
                ):
                    terminal = manager.step(slot, "submit")
                self.assertTrue(terminal.done)
                self.assertEqual(
                    terminal.info["terminal_reason"], "infrastructure_failure"
                )
                staging = list(
                    manager.workspace_manager.handoff_root.glob(".staging-*")
                )
                self.assertEqual(len(staging), 1)
                if retry == "reset":
                    reset = manager.reset(slot, 1)
                    self.assertFalse(reset.done)
                    manager.close(slot)
                else:
                    manager.close(slot)
                self.assertEqual(
                    list(manager.workspace_manager.handoff_root.iterdir()), []
                )

    def test_submission_rejects_symlink_hardlink_fifo_and_oversize(self) -> None:
        cases = ("symlink", "hardlink", "fifo", "oversize")
        for kind in cases:
            with self.subTest(kind=kind):
                manager = self.build_manager(max_submission_bytes=8)
                slot, _ = self.reset(manager)
                workspace = manager._testing_workspace(slot)
                submission = workspace.submission_path
                source = workspace.workspace_root / "source.csv"
                source.write_bytes(b"id,x\n1,0\n")
                if kind == "symlink":
                    submission.symlink_to(source)
                elif kind == "hardlink":
                    os.link(source, submission)
                elif kind == "fifo":
                    os.mkfifo(submission)
                else:
                    submission.write_bytes(b"123456789")
                rejected = manager.step(slot, "submit")
                self.assertFalse(rejected.done)
                self.assertEqual(rejected.observation, "Submission is unavailable.")
                self.assertNotIn("terminal_receipt", rejected.info)
                self.assertEqual(len(self.backends[-1].frozen), 0)

    def test_submission_rejects_missing_empty_and_malformed_csv_before_freeze(
        self,
    ) -> None:
        for kind in ("missing", "empty", "malformed", "late_malformed", "late_width"):
            with self.subTest(kind=kind):
                manager = self.build_manager()
                slot, _ = self.reset(manager)
                submission = manager._testing_workspace(slot).submission_path
                if kind == "empty":
                    submission.write_bytes(b"")
                elif kind == "malformed":
                    submission.write_bytes(b'id,target\n1,"unterminated\n')
                elif kind == "late_malformed":
                    submission.write_bytes(b'id,target\n1,0\n2,"unterminated\n')
                elif kind == "late_width":
                    submission.write_bytes(b"id,target\n1,0\n2,0,extra\n")
                rejected = manager.step(slot, "submit")
                self.assertFalse(rejected.done)
                self.assertEqual(rejected.observation, "Submission is unavailable.")
                self.assertNotIn("terminal_receipt", rejected.info)
                self.assertEqual(len(self.backends[-1].frozen), 0)

    def test_private_and_host_probes_are_denied_without_path_disclosure(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager)
        for expected_count, path in enumerate(
            ("/private/data/answer.csv", "/host/etc/passwd", "/etc/passwd"),
            start=1,
        ):
            step = manager.step(slot, f'inspect {{"path":"{path}"}}')
            self.assertFalse(step.done)
            self.assertEqual(step.info["counters"]["action_count"], expected_count)
            self.assertEqual(step.observation, "Path is unavailable.")

        shell = manager.step(
            slot,
            'shell {"command":"test -e /private/data/answer.csv","timeout_ms":1000}',
        )
        self.assertNotIn(str(self.fixture["data_root"]), shell.observation)
        self.assertNotIn("answer.csv", shell.observation)

    def test_native_has_no_compaction_or_memory_namespace_even_when_probed(
        self,
    ) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager, MODE_NATIVE)
        compact = manager.step(slot, "attempted summary", control="compaction")
        self.assertEqual(compact.observation, "Action is unavailable.")
        self.assertEqual(compact.info["counters"]["action_count"], 1)
        note = manager.step(
            slot,
            'edit {"path":"/run/amg_memory/note","content":"should fail"}',
        )
        self.assertEqual(note.observation, "Path is unavailable.")
        self.assertNotIn("external_memory_operation", note.info)
        shell = manager.step(
            slot,
            'shell {"command":"mkdir -p /home/workspace/ordinary_notes",'
            '"timeout_ms":1000}',
        )
        self.assertNotIn("external_memory_operation", shell.info)
        self.assertIsNone(manager._testing_workspace(slot).memory_root)

    def test_triad_metadata_pins_modes_ids_runtime_resources_and_budgets(self) -> None:
        manager = self.build_manager()
        metadata = manager.metadata()
        self.assertEqual(
            MODES,
            (MODE_NATIVE, MODE_AMG_COMPACTION_ONLY, MODE_AMG_MEMORY),
        )
        self.assertEqual(metadata["modes"], list(MODES))
        self.assertEqual(metadata["competition_ids"], list(LITE_COMPETITION_IDS))
        self.assertEqual(metadata["task_count"], 22)
        self.assertEqual(metadata["runner_sha256"], FAKE_RUNNER_SHA256)
        self.assertEqual(metadata["runtime_digest"], FAKE_RUNTIME_DIGEST)
        self.assertEqual(metadata["resource_contract"], manager.resource_contract)
        self.assertEqual(metadata["resource_contract"]["max_actions"], 20)

    def test_reset_and_task_boundaries_isolate_submission_and_memory(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager)
        manager.step(
            slot,
            'edit {"path":"/run/amg_memory/note","content":"old"}',
        )
        manager.step(
            slot,
            'edit {"path":"/home/submission/submission.csv","content":"old"}',
        )
        first_workspace = manager._testing_workspace(slot)
        manager.reset(slot, 0)
        self.assertFalse(first_workspace.episode_root.exists())
        self.assertEqual(len(self.backends[0].torn_down), 1)
        missing_note = manager.step(
            slot,
            'inspect {"path":"/run/amg_memory/note"}',
        )
        missing_submission = manager.step(
            slot,
            'inspect {"path":"/home/submission/submission.csv"}',
        )
        self.assertEqual(missing_note.observation, "Path is unavailable.")
        self.assertEqual(missing_submission.observation, "Path is unavailable.")

        manager.reset(slot, 1)
        crossing = manager.step(
            slot,
            f'inspect {{"path":"{self.dataset[0].public_root}/train.csv"}}',
        )
        self.assertEqual(crossing.observation, "Path is unavailable.")

    def test_close_tears_down_and_removes_active_workspace(self) -> None:
        manager = self.build_manager()
        slot, _ = self.reset(manager)
        workspace = manager._testing_workspace(slot)
        manager.close(slot)
        self.assertFalse(workspace.episode_root.exists())
        self.assertEqual(len(self.backends[0].torn_down), 1)
        with self.assertRaises(KeyError):
            manager.reset(slot, 0)

    def test_native_tool_dispatch_is_identical_across_triad(self) -> None:
        manager = self.build_manager()
        raw = 'shell {"command":"python -V","timeout_ms":1234}'
        steps = []
        for mode in MODES:
            slot, _ = self.reset(manager, mode)
            steps.append(manager.step(slot, raw))
        self.assertTrue(all(step.observation == steps[0].observation for step in steps))
        executions = [backend.executed[0] for backend in self.backends]
        self.assertTrue(
            all(item["command"] == executions[0]["command"] for item in executions)
        )
        self.assertTrue(
            all(
                item["timeout_ms"] == executions[0]["timeout_ms"] for item in executions
            )
        )


if __name__ == "__main__":
    unittest.main()
