from __future__ import annotations

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
    EXTERNAL_RUNNER_CONTRACT,
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
            '{"command":"python -c \\"print(\'one\')\\"; '
            'python -c \\"print(\'two\')\\"","timeout_ms":10000}'
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

    def test_timeout_is_a_policy_resource_violation(self) -> None:
        action = parse_policy_action(
            'shell_command {"command":"python -c \\"import time; time.sleep(2)\\"",'
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
        with patch(
            "agentenv_openmle_fast.executor.subprocess.run",
            side_effect=subprocess.TimeoutExpired("runner", 1.0),
        ) as run:
            result = backend.run(
                self.workspace,
                command="true",
                timeout_ms=1_000,
                managed_runtime_budget_ms=15_000,
            )
        self.assertEqual(run.call_args.kwargs["timeout"], 2.0)
        self.assertFalse(result.timed_out)
        self.assertTrue(result.infrastructure_fault)
        self.assertEqual(result.failure_class, "runner_protocol_fault")

    def test_lifecycle_timeout_kills_the_owned_runner_process_group(self) -> None:
        backend = object.__new__(ExternalSandboxRunnerBackend)
        backend.runner_path = self.workspace / "runner"
        backend.limits = self.limits
        backend.expected_runtime_digest = "sha256:" + "1" * 64
        with (
            patch("agentenv_openmle_fast.executor.subprocess.Popen") as popen,
            patch(
                "agentenv_openmle_fast.executor._kill_process_group"
            ) as kill_group,
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
                timeout_ms=2_000,
            )
        self.assertFalse(receipt.success)
        self.assertEqual(receipt.failure_class, "runner_lifecycle_fault")
        kill_group.assert_called_with(43210)
        request = json.loads(popen.return_value.communicate.call_args_list[0].args[0])
        self.assertEqual(request["timeout_ms"], 1_000)

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
                    patch(
                        "agentenv_openmle_fast.executor.subprocess.Popen"
                    ) as process,
                    patch(
                        "agentenv_openmle_fast.executor._kill_process_group"
                    ),
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
                limits=self.limits,
            )


if __name__ == "__main__":
    unittest.main()
