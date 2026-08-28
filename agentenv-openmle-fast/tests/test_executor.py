from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentenv_openmle_fast.actions import parse_policy_action
from agentenv_openmle_fast.executor import (
    EXTERNAL_RUNNER_COMPLETION_GRACE_MS,
    EXTERNAL_RUNNER_PROCESS_GRACE_MS,
    EXTERNAL_RUNNER_CONTRACT,
    FIT_HOOK_CONTRACT,
    BackendExecution,
    ExternalSandboxRunnerBackend,
    LocalCPUExecutionBackend,
    OpenMLEFastExecutor,
    OpenMLEFastExecutorError,
    OpenMLEFastResourceLimits,
    _visible_text,
)


class _RecordingBackend:
    def __init__(self, limits: OpenMLEFastResourceLimits) -> None:
        self.limits = limits
        self.timeout_ms: int | None = None
        self.managed_runtime_budget_ms: int | None = None

    @property
    def metadata(self):
        return {
            "contract": "recording-test-backend",
            "formal_eligible": False,
            "resource_limits": self.limits.as_dict(),
        }

    def run(
        self,
        _workspace,
        *,
        command: str,
        timeout_ms: int,
        managed_runtime_budget_ms: int,
    ) -> BackendExecution:
        del command
        self.timeout_ms = timeout_ms
        self.managed_runtime_budget_ms = managed_runtime_budget_ms
        return BackendExecution(
            stdout=b"ok\n",
            stderr=b"",
            exit_code=0,
            timed_out=False,
            wall_seconds=0.0,
            managed_runtime_wall_seconds=0.0,
            cpu_seconds=0.0,
            peak_rss_bytes=0,
            bytes_read=0,
            bytes_written=0,
            process_peak=1,
            execution_attempt_delta=0,
            execution_completed_delta=0,
            nested_subprocess_delta=0,
            fit_delta=0,
            fit_counter_coverage="partial",
        )


class _InvariantAndInfrastructureFaultBackend(_RecordingBackend):
    def run(
        self,
        workspace,
        *,
        command: str,
        timeout_ms: int,
        managed_runtime_budget_ms: int,
    ) -> BackendExecution:
        result = super().run(
            workspace,
            command=command,
            timeout_ms=timeout_ms,
            managed_runtime_budget_ms=managed_runtime_budget_ms,
        )
        (Path(workspace) / "policy-created-link").symlink_to("missing-target")
        return replace(
            result,
            failure_class="runner_protocol_fault",
            infrastructure_fault=True,
        )


class OpenMLEFastExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="openmle-executor-test-")
        self.workspace = Path(self.temporary.name)
        self.limits = OpenMLEFastResourceLimits.frozen_v1()
        self.backend = LocalCPUExecutionBackend(self.limits)
        self.executor = OpenMLEFastExecutor(
            limits=self.limits,
            backend=self.backend,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_shell_receipt_is_bounded_and_honest_about_local_coverage(self) -> None:
        action = parse_policy_action(
            "shell_command "
            '{"command":"python3 -c \\"print(\'one\')\\"; '
            'python3 -c \\"print(\'two\')\\"","timeout_ms":10000}'
        )
        receipt = self.executor.execute(self.workspace, action)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.exit_code, 0)
        self.assertIn("one", receipt.stdout)
        self.assertIn("two", receipt.stdout)
        self.assertEqual(receipt.execution_action_delta, 1)
        self.assertEqual(receipt.execution_attempt_delta, 2)
        self.assertEqual(receipt.execution_completed_delta, 2)
        self.assertGreaterEqual(receipt.nested_subprocess_delta, 2)
        self.assertEqual(receipt.fit_counter_coverage, "partial")
        self.assertFalse(self.backend.metadata["formal_eligible"])
        self.assertEqual(len(receipt.output_sha256), 64)
        self.assertEqual(len(receipt.tree_sha256_after), 64)

    def test_patch_receipt_has_no_execution_delta(self) -> None:
        action = parse_policy_action(
            "apply_patch\n*** Begin Patch\n*** Add File: solution.py\n"
            "+print('ok')\n*** End Patch"
        )
        receipt = self.executor.execute(self.workspace, action)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.changed_paths, ("solution.py",))
        self.assertEqual(receipt.execution_action_delta, 0)
        self.assertEqual(receipt.execution_attempt_delta, 0)

    def test_protected_patch_is_terminal_policy_violation_without_execution(self) -> None:
        (self.workspace / "TASK.md").write_text("contract\n", encoding="utf-8")
        action = parse_policy_action(
            "apply_patch\n*** Begin Patch\n*** Delete File: TASK.md\n*** End Patch"
        )
        receipt = self.executor.execute(self.workspace, action)
        self.assertEqual(receipt.status, "policy_violation")
        self.assertEqual(
            receipt.failure_class, "immutable_public_tree_mutation_attempt"
        )
        self.assertTrue(receipt.policy_terminal)
        self.assertEqual(receipt.changed_paths, ())
        self.assertEqual(receipt.execution_action_delta, 0)
        self.assertEqual(receipt.execution_attempt_delta, 0)
        self.assertTrue((self.workspace / "TASK.md").is_file())

    def test_workspace_invariant_takes_precedence_over_backend_fault(self) -> None:
        backend = _InvariantAndInfrastructureFaultBackend(self.limits)
        executor = OpenMLEFastExecutor(limits=self.limits, backend=backend)
        action = parse_policy_action(
            'shell_command {"command":"ln -s missing-target policy-created-link"}'
        )

        receipt = executor.execute(self.workspace, action)

        self.assertEqual(receipt.status, "policy_violation")
        self.assertEqual(receipt.failure_class, "workspace_invariant_violation")
        self.assertTrue(receipt.policy_terminal)
        self.assertFalse(receipt.infrastructure_fault)
        self.assertEqual(tuple(self.workspace.iterdir()), ())

    def test_timeout_is_a_policy_resource_violation(self) -> None:
        action = parse_policy_action(
            'shell_command {"command":"python3 -c \\"import time; time.sleep(2)\\"",'
            '"timeout_ms":50}'
        )
        receipt = self.executor.execute(self.workspace, action)
        self.assertEqual(receipt.status, "policy_violation")
        self.assertTrue(receipt.timed_out)
        self.assertEqual(receipt.failure_class, "wall_timeout")

    def test_shell_deadline_includes_both_snapshots_and_receipt_work(self) -> None:
        limits = replace(
            self.limits,
            shell_wall_ms=1_000,
            managed_runtime_per_action_ms=800,
        )
        backend = _RecordingBackend(limits)
        executor = OpenMLEFastExecutor(limits=limits, backend=backend)
        action = parse_policy_action(
            'shell_command {"command":"true","timeout_ms":1000}'
        )
        from agentenv_openmle_fast import executor as executor_module

        original_snapshot = executor_module._snapshot_tree

        def delayed_snapshot(root, frozen_limits, *, deadline):
            time.sleep(0.05)
            return original_snapshot(root, frozen_limits, deadline=deadline)

        with patch.object(executor_module, "_snapshot_tree", delayed_snapshot):
            receipt = executor.execute(self.workspace, action)
        self.assertIsNotNone(backend.timeout_ms)
        self.assertLess(backend.timeout_ms, 950)
        self.assertGreaterEqual(receipt.wall_seconds, 0.09)

    def test_external_runner_reserves_bounded_receipt_cleanup_grace(self) -> None:
        backend = object.__new__(ExternalSandboxRunnerBackend)
        backend.runner_path = self.workspace / "runner"
        backend.limits = self.limits
        backend.expected_runtime_digest = "sha256:" + "1" * 64
        timeout_ms = EXTERNAL_RUNNER_COMPLETION_GRACE_MS + 1_000
        with patch(
            "agentenv_openmle_fast.executor.subprocess.run",
            side_effect=subprocess.TimeoutExpired("runner", 1.0),
        ) as run:
            result = backend.run(
                self.workspace,
                command="true",
                timeout_ms=timeout_ms,
                managed_runtime_budget_ms=15_000,
            )
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            (timeout_ms + EXTERNAL_RUNNER_PROCESS_GRACE_MS) / 1_000.0,
        )
        self.assertFalse(result.timed_out)
        self.assertTrue(result.infrastructure_fault)
        self.assertEqual(result.failure_class, "runner_protocol_fault")

        self.assertGreater(
            EXTERNAL_RUNNER_PROCESS_GRACE_MS,
            EXTERNAL_RUNNER_COMPLETION_GRACE_MS,
        )

    def test_external_runner_treats_too_short_timeout_as_policy_timeout(self) -> None:
        backend = object.__new__(ExternalSandboxRunnerBackend)
        backend.runner_path = self.workspace / "runner"
        backend.limits = self.limits
        backend.expected_runtime_digest = "sha256:" + "1" * 64
        with patch("agentenv_openmle_fast.executor.subprocess.run") as run:
            result = backend.run(
                self.workspace,
                command="cat TASK.md",
                timeout_ms=20,
                managed_runtime_budget_ms=15_000,
            )
        run.assert_not_called()
        self.assertTrue(result.timed_out)
        self.assertEqual(result.failure_class, "wall_timeout")
        self.assertFalse(result.infrastructure_fault)
        self.assertEqual(result.execution_attempt_delta, 0)

    def test_external_runner_process_uses_utf8_locale_for_policy_commands(self) -> None:
        backend = object.__new__(ExternalSandboxRunnerBackend)
        backend.runner_path = self.workspace / "runner"
        backend.limits = self.limits
        backend.expected_runtime_digest = "sha256:" + "1" * 64
        command = "printf 'unicode: 中文 ° µ\\n'"
        timeout_ms = EXTERNAL_RUNNER_COMPLETION_GRACE_MS + 1_000
        with patch(
            "agentenv_openmle_fast.executor.subprocess.run",
            side_effect=subprocess.TimeoutExpired("runner", 1.0),
        ) as run:
            backend.run(
                self.workspace,
                command=command,
                timeout_ms=timeout_ms,
                managed_runtime_budget_ms=15_000,
            )
        self.assertEqual(
            run.call_args.kwargs["env"],
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
        )
        request = json.loads(run.call_args.kwargs["input"].decode("utf-8"))
        self.assertEqual(request["command"], command)

    def test_lifecycle_timeout_kills_the_owned_runner_process_group(self) -> None:
        backend = object.__new__(ExternalSandboxRunnerBackend)
        backend.runner_path = self.workspace / "runner"
        backend.limits = self.limits
        backend.expected_runtime_digest = "sha256:" + "1" * 64
        with (
            patch("agentenv_openmle_fast.executor.subprocess.Popen") as popen,
            patch("agentenv_openmle_fast.executor._kill_process_group") as kill_group,
        ):
            popen.return_value.pid = 43210
            popen.return_value.communicate.side_effect = (
                subprocess.TimeoutExpired("runner", 2.0),
                (b"", b""),
            )
            popen.return_value.poll.return_value = 0
            receipt = backend._lifecycle(
                "freeze",
                self.workspace,
                timeout_ms=EXTERNAL_RUNNER_COMPLETION_GRACE_MS + 1_000,
            )
        self.assertFalse(receipt.success)
        self.assertEqual(receipt.failure_class, "runner_lifecycle_fault")
        kill_group.assert_called_with(43210)
        request = json.loads(popen.return_value.communicate.call_args_list[0].args[0])
        self.assertEqual(request["timeout_ms"], 1_000)

    def test_lifecycle_caps_episode_deadline_to_runner_request_limit(self) -> None:
        backend = object.__new__(ExternalSandboxRunnerBackend)
        backend.runner_path = self.workspace / "runner"
        backend.limits = self.limits
        backend.expected_runtime_digest = "sha256:" + "1" * 64
        response = {
            "schema": "openmle_fast_backend_lifecycle_v1",
            "operation": "freeze",
            "success": True,
            "processes_reaped": True,
            "workspace_read_only": True,
            "cgroup_empty": True,
            "failure_class": None,
        }
        with (
            patch("agentenv_openmle_fast.executor.subprocess.Popen") as popen,
            patch(
                "agentenv_openmle_fast.executor._process_group_exists",
                return_value=False,
            ),
        ):
            popen.return_value.pid = 43210
            popen.return_value.communicate.return_value = (
                json.dumps(response).encode(),
                b"",
            )
            popen.return_value.returncode = 0
            popen.return_value.poll.return_value = 0
            receipt = backend._lifecycle(
                "freeze",
                self.workspace,
                timeout_ms=self.limits.episode_wall_ms,
            )
        self.assertTrue(receipt.success)
        request = json.loads(popen.return_value.communicate.call_args.args[0])
        self.assertEqual(request["timeout_ms"], self.limits.shell_wall_ms)
        self.assertEqual(
            popen.return_value.communicate.call_args.kwargs["timeout"],
            (
                self.limits.shell_wall_ms + EXTERNAL_RUNNER_COMPLETION_GRACE_MS
            )
            / 1_000.0,
        )

    def test_remaining_managed_runtime_is_passed_separately_to_backend(self) -> None:
        backend = _RecordingBackend(self.limits)
        executor = OpenMLEFastExecutor(limits=self.limits, backend=backend)
        action = parse_policy_action(
            'shell_command {"command":"python solution.py","timeout_ms":20000}'
        )
        executor.execute(
            self.workspace,
            action,
            managed_runtime_budget_ms=321,
        )
        self.assertGreater(backend.timeout_ms, 321)
        self.assertEqual(backend.managed_runtime_budget_ms, 321)

    def test_local_diagnostic_caps_path_qualified_python_entrypoints(self) -> None:
        for command in ("/usr/bin/python -V", "./venv/bin/python script.py"):
            with self.subTest(command=command):
                backend = LocalCPUExecutionBackend(self.limits)
                with (
                    patch("agentenv_openmle_fast.executor.subprocess.Popen") as process,
                    patch("agentenv_openmle_fast.executor._kill_process_group"),
                    patch(
                        "agentenv_openmle_fast.executor._process_group_exists",
                        return_value=False,
                    ),
                ):
                    process.return_value.pid = 12345
                    process.return_value.communicate.side_effect = (
                        subprocess.TimeoutExpired(command, 0.321),
                        (b"", b""),
                    )
                    process.return_value.returncode = -9
                    backend.run(
                        self.workspace,
                        command=command,
                        timeout_ms=20_000,
                        managed_runtime_budget_ms=321,
                    )
                self.assertEqual(
                    process.return_value.communicate.call_args_list[0].kwargs[
                        "timeout"
                    ],
                    0.321,
                )

    def test_modified_existing_file_is_reported_as_changed(self) -> None:
        target = self.workspace / "solution.py"
        target.write_text("before\n", encoding="utf-8")
        action = parse_policy_action(
            'shell_command {"command":"printf after > solution.py"}'
        )
        receipt = self.executor.execute(self.workspace, action)
        self.assertEqual(receipt.status, "completed")
        self.assertIn("solution.py", receipt.changed_paths)
        self.assertNotEqual(receipt.tree_sha256_before, receipt.tree_sha256_after)

    def test_visible_output_reserves_marker_and_utf8_replacement_bytes(self) -> None:
        visible, truncated = _visible_text(b"\xff" * 100_000, 65_536)
        self.assertTrue(truncated)
        self.assertLessEqual(len(visible.encode("utf-8")), 65_536)

    def test_exact_runner_attestation_rejects_partial_network_claims(self) -> None:
        runner = self.workspace / "runner"
        runner.write_bytes(b"runner")
        runner.chmod(0o700)
        metadata = {
            "contract": EXTERNAL_RUNNER_CONTRACT,
            "runtime_digest": "sha256:" + "1" * 64,
            "resource_limits": self.limits.as_dict(),
            "formal_eligible": True,
            "network_namespace": True,
            "pid_namespace": True,
            "ipc_namespace": True,
            "mount_namespace": True,
            "cgroup_v2_cpu": True,
            "cgroup_v2_memory": True,
            "cgroup_v2_pids": True,
            "no_new_privs": True,
            "seccomp": True,
            "read_only_rootfs": True,
            "workspace_noexec": True,
            "instrumented_python_only": True,
            "execution_counter_coverage": "complete",
            "fit_counter_coverage": "partial",
        }
        completed = subprocess.CompletedProcess(
            args=[str(runner), "metadata"],
            returncode=0,
            stdout=json.dumps(metadata).encode(),
            stderr=b"",
        )
        with (
            patch(
                "agentenv_openmle_fast.executor.subprocess.run",
                return_value=completed,
            ),
            self.assertRaises(OpenMLEFastExecutorError),
        ):
            ExternalSandboxRunnerBackend(
                runner_path=runner,
                expected_runner_sha256=hashlib.sha256(b"runner").hexdigest(),
                expected_runtime_digest="sha256:" + "1" * 64,
                expected_artifact_lock_sha256="a" * 64,
                limits=self.limits,
            )

    def test_exact_runner_accepts_truthful_v1_and_pins_artifact_lock(self) -> None:
        runner = self.workspace / "runner-v1"
        runner.write_bytes(b"runner-v1")
        runner.chmod(0o700)
        artifact_lock = "a" * 64
        true_fields = (
            "formal_eligible",
            "network_namespace",
            "network_no_egress",
            "dns_disabled",
            "metadata_service_blocked",
            "external_unix_sockets_blocked",
            "pid_namespace",
            "ipc_namespace",
            "mount_namespace",
            "workspace_quota",
            "tmpfs_limit",
            "file_size_limit",
            "open_files_limit",
            "fresh_unprivileged_uid_gid",
            "capabilities_dropped",
            "no_new_privs",
            "seccomp",
            "read_only_rootfs",
            "workspace_noexec",
            "instrumented_python_only",
            "cumulative_managed_runtime_budget",
            "isolated_proc",
            "minimal_devices",
            "gpu_devices_absent",
            "core_dumps_disabled",
            "mount_denied",
            "ptrace_denied",
            "setuid_denied",
            "user_namespace_creation_denied",
            "ebpf_denied",
            "raw_sockets_denied",
            "kernel_module_denied",
            "container_engine_absent",
            "background_process_detection",
            "descendant_kill_reap",
            "parent_death_cleanup_watchdog",
            "freeze_reap",
            "read_only_workspace_freeze",
            "teardown_cgroup_empty",
            "teardown_mount_empty",
            "idempotent_teardown",
        )
        metadata = {field: True for field in true_fields}
        metadata.update(
            {
                "contract": EXTERNAL_RUNNER_CONTRACT,
                "runtime_digest": "sha256:" + "1" * 64,
                "resource_limits": self.limits.as_dict(),
                "cgroup_version": "v1",
                "cgroup_controller_attestation": {"version": "v1"},
                "cgroup_v1_cpu": True,
                "cgroup_v1_memory": True,
                "cgroup_v1_pids": True,
                "cgroup_v2_cpu": False,
                "cgroup_v2_memory": False,
                "cgroup_v2_pids": False,
                "active_verification": {
                    "admission_stamp_valid": True,
                    "all_checks_pass": True,
                },
                "artifact_identity": {
                    "artifact_lock_sha256": artifact_lock,
                    "artifact_lock_expected_sha256": artifact_lock,
                },
                "execution_counter_coverage": "complete",
                "fit_counter_coverage": "partial",
                "fit_hook_contract": FIT_HOOK_CONTRACT,
                "fit_hook_digest": hashlib.sha256(
                    FIT_HOOK_CONTRACT.encode()
                ).hexdigest(),
                "workspace_parent": str(self.workspace),
                "workspace_storage_contract": ("owned_dedicated_tmpfs_at_most_2g_v1"),
            }
        )

        def completed(value):
            return subprocess.CompletedProcess(
                args=[str(runner), "metadata"],
                returncode=0,
                stdout=json.dumps(value).encode(),
                stderr=b"",
            )

        with patch(
            "agentenv_openmle_fast.executor.subprocess.run",
            return_value=completed(metadata),
        ):
            backend = ExternalSandboxRunnerBackend(
                runner_path=runner,
                expected_runner_sha256=hashlib.sha256(b"runner-v1").hexdigest(),
                expected_runtime_digest="sha256:" + "1" * 64,
                expected_artifact_lock_sha256=artifact_lock,
                limits=self.limits,
            )
        self.assertEqual(backend.metadata["cgroup_version"], "v1")

        drifted = copy.deepcopy(metadata)
        drifted["artifact_identity"]["artifact_lock_sha256"] = "b" * 64
        with (
            patch(
                "agentenv_openmle_fast.executor.subprocess.run",
                return_value=completed(drifted),
            ),
            self.assertRaisesRegex(OpenMLEFastExecutorError, "attestation"),
        ):
            ExternalSandboxRunnerBackend(
                runner_path=runner,
                expected_runner_sha256=hashlib.sha256(b"runner-v1").hexdigest(),
                expected_runtime_digest="sha256:" + "1" * 64,
                expected_artifact_lock_sha256=artifact_lock,
                limits=self.limits,
            )


if __name__ == "__main__":
    unittest.main()
