from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentenv_mlebench_lite.dataset import load_lite_dataset
from agentenv_mlebench_lite.executor import (
    ExternalSandboxRunnerBackend,
    MLEBenchLiteExecutorError,
    SandboxExecutor,
)
from agentenv_mlebench_lite.identity import UPSTREAM_COMMIT, load_official_lite_identity
from agentenv_mlebench_lite.resources import (
    build_resource_contract,
    resource_contract_sha256,
    validate_resource_contract,
)
from agentenv_mlebench_lite.workspace import (
    MODE_AMG_COMPACTION_ONLY,
    MODE_AMG_MEMORY,
    MODE_NATIVE,
    MLEBenchLiteWorkspaceError,
    WorkspaceManager,
)

from tests.support import (
    FAKE_RUNNER_SHA256,
    FAKE_RUNTIME_DIGEST,
    RecordingFormalBackend,
    UnsafeLocalBackend,
    sandbox_attestation,
    sha256_bytes,
    write_fixture,
)


class MLEBenchLiteWorkspaceExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mlebench-lite-work-")
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
        self.manager = WorkspaceManager(self.fixture["episodes_root"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reset_workspaces_are_unique_and_public_data_is_not_copied(self) -> None:
        record = self.dataset[0]
        first = self.manager.create(record, MODE_AMG_MEMORY)
        second = self.manager.create(record, MODE_AMG_MEMORY)
        native = self.manager.create(record, MODE_NATIVE)
        compact_only = self.manager.create(record, MODE_AMG_COMPACTION_ONLY)

        self.assertNotEqual(first.episode_root, second.episode_root)
        self.assertIsNotNone(first.memory_root)
        self.assertTrue(first.memory_root.is_dir())
        self.assertNotEqual(first.memory_root, first.workspace_root)
        self.assertIsNone(native.memory_root)
        self.assertIsNone(compact_only.memory_root)
        self.assertEqual(list(first.submission_root.iterdir()), [])
        self.assertFalse((first.workspace_root / "train.csv").exists())
        self.assertEqual(first.public_root, record.public_root)

    def test_existing_episode_and_handoff_roots_must_be_owner_only(self) -> None:
        for name in ("unsafe-episodes", "unsafe-handoffs"):
            path = self.root / name
            path.mkdir(mode=0o700)
            path.chmod(0o770)
        with self.assertRaises(MLEBenchLiteWorkspaceError):
            WorkspaceManager(
                self.root / "unsafe-episodes", self.root / "unsafe-handoffs"
            )

    def test_staged_workspace_fault_rolls_back_every_partial_directory(self) -> None:
        stages: list[str] = []

        def fail_after_submission(stage: str, _path: Path) -> None:
            stages.append(stage)
            if stage == "submission_created":
                raise OSError("injected staged create failure")

        episodes = self.root / "transactional-episodes"
        handoffs = self.root / "transactional-handoffs"
        manager = WorkspaceManager(
            episodes,
            handoffs,
            stage_hook=fail_after_submission,
        )
        with self.assertRaises(MLEBenchLiteWorkspaceError):
            manager.create(self.dataset[0], MODE_AMG_MEMORY)
        self.assertIn("submission_created", stages)
        self.assertEqual(list(episodes.iterdir()), [])
        self.assertEqual(list(handoffs.iterdir()), [])

    def test_publish_failure_after_rename_leaves_no_partial_handoff(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)
        payload = b"id,target\n1,0\n"
        staging = self.manager.stage_submission(
            workspace,
            payload,
            sha256_bytes(payload),
        )
        final = self.manager.handoff_root / workspace.episode_id
        original_chmod = os.chmod

        def fail_final_seal(path, mode, *args, **kwargs):
            if Path(path) == final and mode == 0o500:
                raise OSError("injected final seal failure")
            return original_chmod(path, mode, *args, **kwargs)

        with (
            patch(
                "agentenv_mlebench_lite.workspace.os.chmod",
                side_effect=fail_final_seal,
            ),
            self.assertRaises(MLEBenchLiteWorkspaceError),
        ):
            self.manager.publish_submission(staging, {"schema": "test"})
        self.assertFalse(staging.directory.exists())
        self.assertFalse(final.exists())
        self.assertEqual(list(self.manager.handoff_root.iterdir()), [])

    def test_virtual_path_escape_host_private_and_cross_task_are_denied(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_AMG_MEMORY)
        denied = (
            "/private/data/answer.csv",
            "/host/etc/passwd",
            "/etc/passwd",
            "/home/workspace/../../private/data/answer.csv",
            str(self.dataset[1].public_root / "train.csv"),
            str(
                self.fixture["data_root"]
                / self.dataset[0].competition_id
                / "prepared"
                / "private"
            ),
        )
        for path in denied:
            with self.subTest(path=path), self.assertRaises(MLEBenchLiteWorkspaceError):
                workspace.resolve_policy_path(path, write=False)

    def test_public_is_read_only_native_memory_namespace_absent_and_symlink_denied(
        self,
    ) -> None:
        memory = self.manager.create(self.dataset[0], MODE_AMG_MEMORY)
        native = self.manager.create(self.dataset[0], MODE_NATIVE)
        compact_only = self.manager.create(self.dataset[0], MODE_AMG_COMPACTION_ONLY)
        with self.assertRaises(MLEBenchLiteWorkspaceError):
            memory.resolve_policy_path("/home/data/train.csv", write=True)
        with self.assertRaises(MLEBenchLiteWorkspaceError):
            native.resolve_policy_path("/run/amg_memory/notes.md", write=True)
        with self.assertRaises(MLEBenchLiteWorkspaceError):
            compact_only.resolve_policy_path(
                "/run/amg_memory/notes.md", write=True
            )

        for workspace in (native, compact_only):
            attestation = sandbox_attestation(workspace)
            self.assertEqual(
                attestation["external_memory_isolation"]["sandbox_access"],
                "none",
            )

        link = memory.workspace_root / "escape"
        link.symlink_to(
            self.fixture["data_root"]
            / self.dataset[0].competition_id
            / "prepared"
            / "private"
        )
        with self.assertRaises(MLEBenchLiteWorkspaceError):
            memory.resolve_policy_path("/home/workspace/escape/answer.csv", write=False)

    def test_dirfd_write_rejects_a_parent_swapped_to_symlink_mid_walk(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_AMG_MEMORY)
        race = workspace.workspace_root / "race"
        race.mkdir(mode=0o700)
        outside = self.root / "outside-race-target"
        outside.mkdir(mode=0o700)
        original_open = os.open
        swapped = False

        def racing_open(path, flags, *args, dir_fd=None, **kwargs):
            nonlocal swapped
            if path == "race" and dir_fd is not None and not swapped:
                swapped = True
                race.rename(workspace.workspace_root / "race-original")
                race.symlink_to(outside, target_is_directory=True)
            return original_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

        with (
            patch("agentenv_mlebench_lite.workspace.os.open", side_effect=racing_open),
            self.assertRaises(MLEBenchLiteWorkspaceError),
        ):
            workspace.atomic_write_policy_file(
                "/home/workspace/race/escaped.txt", b"must not escape"
            )
        self.assertTrue(swapped)
        self.assertFalse((outside / "escaped.txt").exists())

    def test_formal_executor_requires_exact_mount_attestation(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_AMG_MEMORY)
        backend = RecordingFormalBackend()
        executor = SandboxExecutor(
            backend,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        executor.preflight(workspace)
        result = executor.run(workspace, "python train.py", timeout_ms=1234)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(backend.executed[0]["command"], "python train.py")

        attestation = backend.attest(workspace)
        mounts = {mount["target"]: mount for mount in attestation["mounts"]}
        self.assertTrue(mounts["/home/data"]["read_only"])
        self.assertFalse(mounts["/home/workspace"]["read_only"])
        self.assertFalse(mounts["/home/submission"]["read_only"])
        self.assertFalse(mounts["/run/amg_memory"]["read_only"])
        self.assertEqual(
            mounts["/run/amg_memory"]["source"], str(workspace.memory_root)
        )
        self.assertNotIn("/private", mounts)
        self.assertNotIn("/host", mounts)
        self.assertEqual(
            attestation["external_memory_isolation"],
            {
                "sandbox_access": "read_write_mount_v1",
                "native_tool_surface": "inspect_edit_shell_v1",
                "private_root_state": "allocated",
            },
        )

    def test_full_resource_contract_is_bound_to_requests_and_attestation(self) -> None:
        contract = build_resource_contract(
            max_actions=7,
            max_submission_bytes=4096,
            max_shell_timeout_ms=5000,
            max_visible_output_bytes=65_536,
            submission_path="/home/submission/submission.csv",
            episode_timeout_ms=20_000,
            max_total_execution_ms=15_000,
            cpu_limit_cores=2,
            memory_limit_bytes=1_000_000,
            pids_limit=32,
            writable_bytes_limit=8192,
            writable_inodes_limit=64,
            gpu_count=1,
        )
        contract_sha256 = resource_contract_sha256(contract)
        workspace = self.manager.create(
            self.dataset[0],
            MODE_NATIVE,
            resource_contract=contract,
            resource_contract_sha256=contract_sha256,
        )
        backend = RecordingFormalBackend()
        executor = SandboxExecutor(
            backend,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            expected_resource_contract_sha256=contract_sha256,
        )
        executor.preflight(workspace)
        self.assertEqual(backend.attested[0].resource_contract, contract)
        self.assertEqual(backend.attest(workspace)["resource_contract"], contract)

        tampered = replace(
            workspace,
            resource_contract={**contract, "max_actions": 8},
        )
        with self.assertRaises(MLEBenchLiteExecutorError):
            executor.preflight(tampered)

        missing = RecordingFormalBackend()
        missing.attestation_override = missing.attest(workspace)
        missing.attestation_override.pop("resource_contract")
        with self.assertRaises(MLEBenchLiteExecutorError):
            SandboxExecutor(
                missing,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                expected_resource_contract_sha256=contract_sha256,
            ).preflight(workspace)

    def test_resource_contract_validation_is_exact_and_hashes_only_valid_values(
        self,
    ) -> None:
        contract = build_resource_contract(
            max_actions=3,
            max_submission_bytes=100,
            max_shell_timeout_ms=1000,
            max_visible_output_bytes=200,
            submission_path="/home/submission/submission.csv",
            episode_timeout_ms=2000,
            max_total_execution_ms=1500,
            cpu_limit_cores=1,
            memory_limit_bytes=1000,
            pids_limit=2,
            writable_bytes_limit=200,
            writable_inodes_limit=3,
            gpu_count=1,
        )
        self.assertEqual(validate_resource_contract(contract), contract)
        variants = (
            {key: value for key, value in contract.items() if key != "max_actions"},
            {**contract, "unexpected": 1},
            {**contract, "max_actions": True},
            {**contract, "max_step_response_ms": contract["max_step_response_ms"] - 1},
        )
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_contract_sha256(value)

    def test_incomplete_attestation_and_host_subprocess_backend_fail_closed(
        self,
    ) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)
        backend = RecordingFormalBackend()
        backend.attestation_override = backend.attest(workspace)
        backend.attestation_override["network_disabled"] = False
        with self.assertRaises(MLEBenchLiteExecutorError):
            SandboxExecutor(
                backend,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            ).preflight(workspace)

        missing = RecordingFormalBackend()
        missing.attestation_override = missing.attest(workspace)
        missing.attestation_override.pop("mount_namespace")
        with self.assertRaises(MLEBenchLiteExecutorError):
            SandboxExecutor(
                missing,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            ).preflight(workspace)

        extra = RecordingFormalBackend()
        extra.attestation_override = extra.attest(workspace)
        extra.attestation_override["unexpected"] = True
        with self.assertRaises(MLEBenchLiteExecutorError):
            SandboxExecutor(
                extra,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            ).preflight(workspace)

        with self.assertRaises(MLEBenchLiteExecutorError):
            SandboxExecutor(
                UnsafeLocalBackend(),
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            ).preflight(workspace)

    def test_lifecycle_evidence_rejects_nested_numeric_type_confusion(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)

        class TypeConfusedBackend(RecordingFormalBackend):
            def __init__(self, phase: str) -> None:
                super().__init__()
                self.phase = phase

            def attest(self, workspace):
                value = super().attest(workspace)
                if self.phase == "preflight":
                    value["execution_scope"]["cgroup_enforced"] = 1
                return value

            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                if self.phase == "execution":
                    result.receipt["containment"]["cgroup_enforced"] = 1
                return result

            def freeze_and_reap(self, **kwargs):
                receipt = super().freeze_and_reap(**kwargs)
                if self.phase == "freeze":
                    receipt["resource_cumulative"]["execution_time_ms"] = 0.0
                return receipt

            def teardown(self, **kwargs):
                receipt = super().teardown(**kwargs)
                if self.phase == "teardown":
                    receipt["resource_cumulative"]["cpu_time_ms"] = False
                return receipt

        for phase in ("preflight", "execution", "freeze", "teardown"):
            with self.subTest(phase=phase):
                executor = SandboxExecutor(
                    TypeConfusedBackend(phase),
                    expected_runner_sha256=FAKE_RUNNER_SHA256,
                    expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                )
                if phase == "preflight":
                    with self.assertRaises(MLEBenchLiteExecutorError):
                        executor.preflight(workspace)
                    continue
                executor.preflight(workspace)
                with self.assertRaises(MLEBenchLiteExecutorError):
                    if phase == "execution":
                        executor.run(workspace, "python -V", timeout_ms=1000)
                    elif phase == "freeze":
                        executor.freeze_and_reap(workspace)
                    else:
                        executor.teardown(workspace)

    def test_execution_receipt_is_strictly_allowlisted(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)

        class ExtraReceiptBackend(RecordingFormalBackend):
            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                assert result.receipt is not None
                result.receipt["unexpected"] = True
                return result

        executor = SandboxExecutor(
            ExtraReceiptBackend(),
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        executor.preflight(workspace)
        with self.assertRaises(MLEBenchLiteExecutorError):
            executor.run(workspace, "python -V", timeout_ms=1000)

    def test_external_memory_access_receipt_is_capability_bound(self) -> None:
        memory = self.manager.create(self.dataset[0], MODE_AMG_MEMORY)
        backend = RecordingFormalBackend()
        backend.next_external_memory_access = "read_write"
        executor = SandboxExecutor(
            backend,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        executor.preflight(memory)
        result = executor.run(memory, "fixture", timeout_ms=1000)
        self.assertEqual(
            result.receipt["external_memory_access"],
            {
                "schema": "amg_external_memory_access_v1",
                "operation": "read_write",
            },
        )

        for mode, operation in (
            (MODE_NATIVE, "read"),
            (MODE_AMG_COMPACTION_ONLY, "write"),
            (MODE_AMG_MEMORY, "delete"),
        ):
            with self.subTest(mode=mode, operation=operation):
                workspace = self.manager.create(self.dataset[0], mode)
                bad_backend = RecordingFormalBackend()
                bad_backend.next_external_memory_access = operation
                bad = SandboxExecutor(
                    bad_backend,
                    expected_runner_sha256=FAKE_RUNNER_SHA256,
                    expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                )
                bad.preflight(workspace)
                with self.assertRaises(MLEBenchLiteExecutorError):
                    bad.run(workspace, "fixture", timeout_ms=1000)

    def test_timed_out_execution_must_reap_every_descendant(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)

        class TimeoutBackend(RecordingFormalBackend):
            def execute(self, *, workspace, command, timeout_ms, operation_id):
                return self._result(
                    workspace,
                    command,
                    timeout_ms,
                    operation_id=operation_id,
                    returncode=124,
                    timed_out=True,
                )

        healthy = SandboxExecutor(
            TimeoutBackend(),
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        healthy.preflight(workspace)
        result = healthy.run(workspace, "sleep 10", timeout_ms=1)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.receipt["containment"]["descendant_process_count"], 0)

        class LeakyTimeoutBackend(TimeoutBackend):
            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                assert result.receipt is not None
                result.receipt["containment"]["descendant_process_count"] = 1
                return result

        leaky = SandboxExecutor(
            LeakyTimeoutBackend(),
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        leaky.preflight(workspace)
        with self.assertRaises(MLEBenchLiteExecutorError):
            leaky.run(workspace, "sleep 10", timeout_ms=1)

    def test_teardown_uses_cached_preflight_and_retry_stable_operation_id(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)

        class RetryBackend(RecordingFormalBackend):
            def __init__(self):
                super().__init__()
                self.teardown_attempts: list[str] = []

            def teardown(self, *, workspace, operation_id):
                self.teardown_attempts.append(operation_id)
                if len(self.teardown_attempts) == 1:
                    raise OSError("lost teardown response")
                return super().teardown(
                    workspace=workspace,
                    operation_id=operation_id,
                )

        backend = RetryBackend()
        executor = SandboxExecutor(
            backend,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        executor.preflight(workspace)
        backend.attestation_override = {"invalid": True}
        with self.assertRaises(MLEBenchLiteExecutorError):
            executor.teardown(workspace)
        executor.teardown(workspace)
        self.assertEqual(len(backend.attested), 1)
        self.assertEqual(len(set(backend.teardown_attempts)), 1)

    def test_runner_permissions_and_replacement_are_checked_for_each_call(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)
        runner = self.root / "mutable-runner"
        runner.write_bytes(b"#!/bin/sh\nexit 1\n")
        runner.chmod(0o500)
        digest = sha256_bytes(runner.read_bytes())
        backend = ExternalSandboxRunnerBackend(
            runner,
            expected_runner_sha256=digest,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            expected_runner_uid=os.geteuid(),
        )
        runner.chmod(0o700)
        with self.assertRaises(MLEBenchLiteExecutorError):
            backend.attest(workspace)
        runner.chmod(0o500)
        replacement = self.root / "replacement"
        replacement.write_bytes(b"#!/bin/sh\nexit 2\n")
        replacement.chmod(0o500)
        os.replace(replacement, runner)
        with self.assertRaises(MLEBenchLiteExecutorError):
            backend.attest(workspace)

    def test_external_runner_receives_a_minimal_nonsecret_environment(self) -> None:
        workspace = self.manager.create(self.dataset[0], MODE_NATIVE)
        runner = self.root / "runner"
        runner.write_bytes(b"#!/bin/sh\nexit 1\n")
        runner.chmod(0o500)
        runner_sha256 = sha256_bytes(runner.read_bytes())
        backend = ExternalSandboxRunnerBackend(
            runner,
            expected_runner_sha256=runner_sha256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            expected_runner_uid=os.geteuid(),
        )
        completed = subprocess.CompletedProcess(
            args=[str(runner), "attest"],
            returncode=0,
            stdout=json.dumps(sandbox_attestation(workspace)).encode("utf-8"),
            stderr=b"",
        )
        with patch(
            "agentenv_mlebench_lite.executor.subprocess.run",
            return_value=completed,
        ) as run:
            backend.attest(workspace)
        environment = run.call_args.kwargs["env"]
        self.assertEqual(
            environment,
            {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        self.assertNotIn("KAGGLE_CONFIG_DIR", environment)
        self.assertTrue(run.call_args.kwargs["close_fds"])
        self.assertEqual(len(run.call_args.kwargs["pass_fds"]), 1)
        self.assertIn("/fd/", run.call_args.args[0][0])
        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(
            request["resource_contract"], dict(workspace.resource_contract)
        )
        self.assertEqual(
            request["resource_contract_sha256"],
            workspace.resource_contract_sha256,
        )
        self.assertNotIn("external_memory_root", request)

        memory = self.manager.create(self.dataset[0], MODE_AMG_MEMORY)
        memory_completed = subprocess.CompletedProcess(
            args=[str(runner), "attest"],
            returncode=0,
            stdout=json.dumps(sandbox_attestation(memory)).encode("utf-8"),
            stderr=b"",
        )
        with patch(
            "agentenv_mlebench_lite.executor.subprocess.run",
            return_value=memory_completed,
        ) as memory_run:
            backend.attest(memory)
        memory_request = json.loads(memory_run.call_args.kwargs["input"])
        self.assertEqual(
            memory_request["external_memory_root"], str(memory.memory_root)
        )

    def test_runner_parent_must_be_owned_and_not_group_or_world_writable(self) -> None:
        parent = self.root / "unsafe-runner-parent"
        parent.mkdir(mode=0o700)
        runner = parent / "runner"
        runner.write_bytes(b"#!/bin/sh\nexit 1\n")
        runner.chmod(0o500)
        parent.chmod(0o770)
        with self.assertRaises(MLEBenchLiteExecutorError):
            ExternalSandboxRunnerBackend(
                runner,
                expected_runner_sha256=sha256_bytes(runner.read_bytes()),
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                expected_runner_uid=os.geteuid(),
            )


if __name__ == "__main__":
    unittest.main()
