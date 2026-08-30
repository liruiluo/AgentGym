from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import (
    ExecutableFingerprint,
    ShellExecutionResult,
    ShellSandboxError,
    ShellSandboxLimits,
)
from agentenv_swesmith.dataset import SwesmithDataset
from agentenv_swesmith.audit import AUDIT_SCHEMA, SwesmithEpisodeAuditSink
from agentenv_swesmith.environment import SwesmithEpisodeManager
from agentenv_swesmith.actions import ParsedPolicyAction, parse_policy_action
from agentenv_swesmith.environment import (
    _changed_paths_observation,
    _shell_action_progress,
    _shell_observation,
    _stable_json_sha256,
)
from agentenv_swesmith.grader import SwesmithGradeResult
from agentenv_swesmith.profile import SwesmithProfileBinding
from agentenv_swesmith.sandbox import LinuxNamespaceEpisodeSandbox
from agentenv_swesmith.workspace import SwesmithWorkspaceMaterializer


def limits() -> ShellSandboxLimits:
    return ShellSandboxLimits(
        workspace_bytes=1024 * 1024,
        workspace_inodes=4096,
        max_files=3072,
        max_directories=1024,
        max_file_bytes=256 * 1024,
        max_path_chars=512,
        default_timeout_ms=10_000,
        max_timeout_ms=30_000,
        cpu_seconds=10,
        address_space_bytes=1024 * 1024 * 1024,
        max_processes=32,
        max_open_files=128,
        stdout_bytes=64 * 1024,
        stderr_bytes=64 * 1024,
        tmp_bytes=1024 * 1024,
        tmp_inodes=512,
    )


class Lease:
    def __init__(self) -> None:
        self.closed = False

    def __exit__(self, *_exc: object) -> None:
        self.closed = True


class LocalSandbox(LinuxNamespaceEpisodeSandbox):
    def __init__(self) -> None:
        self.lease = Lease()
        super().__init__(
            limits=limits(),
            rg_binary=Path("/unused/rg"),
            expected_rg_sha256="0" * 64,
            rg_sha256="0" * 64,
            rg_version="test",
            rg_fingerprint=ExecutableFingerprint(0, 0, 0, 0, 0, 0),
            binaries={},
            uid_lease_context=self.lease,
            # The production policy identity must never be root.  The remote
            # test container itself runs as root, so use a deterministic
            # unprivileged UID for the host-side chown contract.
            model_uid=(1000 if os.getuid() == 0 else os.getuid()),
        )

    @property
    def model_gid(self) -> int:
        return 1000 if os.getgid() == 0 else os.getgid()

    def _run_namespace(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
        stdout_limit_bytes: int | None = None,
        stderr_limit_bytes: int | None = None,
    ) -> ShellExecutionResult:
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=workspace_root / workdir,
            capture_output=True,
            timeout=timeout_ms / 1000,
        )
        return ShellExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            elapsed_ms=1,
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
            termination_reason=None,
            sandbox_contract="test",
            model_uid=self.model_uid,
        )


class OutputPipeLeakSandbox(LocalSandbox):
    def _run_namespace(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
        stdout_limit_bytes: int | None = None,
        stderr_limit_bytes: int | None = None,
    ) -> ShellExecutionResult:
        raise ShellSandboxError(
            "shell sandbox left an output pipe open after process cleanup"
        )


class Resolver:
    def resolve(self, instance):
        return SwesmithProfileBinding(
            repo=instance["repo"],
            image="swebench/test",
            f2p_test_paths=("tests/test_fix.py",),
            p2p_test_paths=("tests/test_keep.py",),
            f2p_command="true",
            full_command="true",
            log_parser=lambda _: {},
            get_eval_tests_report=lambda *_: {},
            get_resolution_status=lambda _: "FULL",
            full_resolution_status="FULL",
        )


class Grader:
    timeout_ms = 10_000

    def __init__(self) -> None:
        self.calls = 0

    def grade(self, *, instance, profile, workspace, sandbox):
        self.calls += 1
        resolved = (workspace.policy_root / "src/value.py").read_text() == "fixed\n"
        return SwesmithGradeResult(
            reward=1.0 if resolved else 0.0,
            resolution_status="FULL" if resolved else "PARTIAL",
            report={},
            restored_test_paths=(),
            f2p_run=None,
            full_run=None,
        )


class InfrastructureFailingGrader:
    timeout_ms = 10_000

    def __init__(self) -> None:
        self.calls = 0

    def grade(self, *, instance, profile, workspace, sandbox):
        self.calls += 1
        return SwesmithGradeResult(
            reward=0.0,
            resolution_status=None,
            report={},
            restored_test_paths=(),
            f2p_run=None,
            full_run=None,
            error="RuntimeError: grader backend unavailable",
        )


class FailingAuditSink:
    def write(self, **_kwargs) -> None:
        raise OSError("audit storage unavailable")


class SwesmithEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirrors = self.root / "mirrors"
        self.episodes = self.root / "episodes"
        self.audits = self.root / "audits"
        self.mirrors.mkdir()
        self.mirror = self.mirrors / "owner__repo.12345678"
        self.mirror.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.write("src/value.py", "good\n")
        self.write("tests/test_fix.py", "def test_fix(): pass\n")
        self.write("tests/test_keep.py", "def test_keep(): pass\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "pristine")
        self.write("src/value.py", "bug\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "bug")
        (self.mirror / "tests/test_fix.py").unlink()
        self.git("add", "-u")
        self.git("commit", "-q", "-m", "remove hidden test")
        self.instance_id = "owner__repo.12345678.combine_file__opaque"
        self.git("branch", self.instance_id)

        row = {
            "instance_id": self.instance_id,
            "repo": "swesmith/owner__repo.12345678",
            "problem_statement": "Fix the public value without reading hidden material.",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
            "patch": "SECRET_GOLD_PATCH",
        }
        shard = self.root / "data.jsonl"
        shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "swesmith_jsonl_manifest_v1",
            "dataset_id": "unit",
            "upstream": {
                "repository": "SWE-bench/SWE-smith",
                "revision": "e" * 40,
            },
            "role": "plumbing",
            "selection": {"mode": "all_usable"},
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "physical_rows": 1,
                    "usable_rows": 1,
                }
            ],
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.grader = Grader()
        self.manager = SwesmithEpisodeManager(
            dataset=SwesmithDataset(manifest_path),
            materializer=SwesmithWorkspaceMaterializer(
                mirrors_root=self.mirrors,
                episodes_root=self.episodes,
            ),
            profile_resolver=Resolver(),
            sandbox_factory=lambda _record, _profile: LocalSandbox(),
            grader=self.grader,
            audit_sink=SwesmithEpisodeAuditSink(self.audits),
            max_steps=8,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.mirror), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def write(self, relative: str, content: str) -> None:
        path = self.mirror / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_continuous_workspace_patch_final_and_private_evidence(self) -> None:
        self.assertEqual(
            {
                "active_slot_count": 0,
                "active_environment_count": 0,
                "active_workspace_count": 0,
            },
            {
                key: self.manager.metadata()[key]
                for key in (
                    "active_slot_count",
                    "active_environment_count",
                    "active_workspace_count",
                )
            },
        )
        slot = self.manager.create()
        self.assertEqual(self.manager.metadata()["active_slot_count"], 1)
        self.assertEqual(self.manager.metadata()["active_environment_count"], 0)
        self.assertEqual(self.manager.metadata()["max_steps"], 8)
        self.assertEqual(self.manager.metadata()["configured_max_policy_turns"], 8)
        self.assertEqual(
            self.manager.metadata()["observation_contract"],
            "bounded_combined_shell_output_v1",
        )
        self.assertEqual(self.manager.metadata()["max_observation_bytes"], 6144)
        self.assertEqual(
            self.manager.metadata()["memory_contract"],
            "policy_filesystem_checkpoint_then_client_replace_v4",
        )
        self.assertEqual(self.manager.metadata()["training_max_policy_turns"], 75)
        self.assertEqual(
            self.manager.metadata()["upstream_reference_max_policy_turns"],
            250,
        )
        self.assertEqual(
            self.manager.metadata()["upstream_agent_repository"],
            "SWE-agent/mini-swe-agent",
        )
        self.assertEqual(
            self.manager.metadata()["upstream_agent_revision"],
            "a83fcae82d2a08f0ee0c688f9d137b3566c097f8",
        )
        self.assertEqual(
            self.manager.metadata()["reward_contract"],
            "submission_success1_wrong0_recoverable_invalid_minus0p01_v2",
        )
        self.assertEqual(
            self.manager.metadata()["submission_contract"],
            "upstream_shell_output_sentinel_source_change_required_v2",
        )
        self.assertEqual(
            self.manager.metadata()["horizon_contract"],
            "unified_policy_step_terminal_failure_minus0p01_v3",
        )
        reset = self.manager.reset(slot, 0)
        self.assertIs(reset.info["episode_success"], False)
        self.assertEqual(self.manager.metadata()["active_environment_count"], 1)
        self.assertIn("Fix the public value", reset.observation)
        self.assertIn("Use shell_command", reset.observation)
        self.assertIn('shell_command {"command":"ls","workdir":"."}', reset.observation)
        self.assertIn("without XML", reset.observation)
        self.assertIn("literal line apply_patch", reset.observation)
        self.assertIn("*** Begin Patch ... *** End Patch", reset.observation)
        self.assertIn("Keep edits localized", reset.observation)
        self.assertIn("never paste or rewrite an entire existing file", reset.observation)
        self.assertIn("stay below the response limit", reset.observation)
        self.assertIn("has no .git directory", reset.observation)
        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", reset.observation)
        self.assertIn("at most 8 total policy turns", reset.observation)
        self.assertIn("context compactions consume this same budget", reset.observation)
        self.assertIn("do not wait for the horizon", reset.observation)
        self.assertIn("reward -0.01", reset.observation)
        self.assertEqual(self.manager.metadata()["invalid_action_reward"], -0.01)
        self.assertFalse(self.manager.metadata()["invalid_action_terminal"])
        self.assertEqual(
            self.manager.metadata()["penalized_nonterminal_conditions"],
            ["parser_rejected", "executor_rejected"],
        )
        self.assertEqual(
            self.manager.metadata()["penalized_terminal_conditions"],
            ["max_steps", "policy_turn_horizon"],
        )
        self.assertEqual(self.manager.metadata()["failed_submission_reward"], 0.0)
        self.assertEqual(self.manager.metadata()["horizon_failure_reward"], -0.01)
        self.assertFalse(
            self.manager.metadata()["valid_shell_nonzero_exit_is_terminal"]
        )
        self.assertEqual(
            self.manager.metadata()["grader_infrastructure_failure"],
            "sample_excluded",
        )
        self.assertEqual(
            self.manager.metadata()["grader_execution_contract"],
            "single_full_official_command_v1",
        )
        self.assertEqual(self.manager.metadata()["grader_phase_count"], 1)
        self.assertEqual(self.manager.metadata()["grader_timeout_ms"], 10_000)
        self.assertNotIn("SECRET_GOLD_PATCH", reset.observation)
        self.assertNotIn(self.instance_id, reset.observation)

        created = self.manager.step(
            slot,
            'shell_command {"command":"printf persistent > notes.txt"}',
        )
        self.assertFalse(created.done)
        self.assertIs(created.info["episode_success"], False)
        read = self.manager.step(
            slot,
            'shell_command {"command":"cat notes.txt"}',
        )
        self.assertIn("persistent", read.observation)

        patched = self.manager.step(
            slot,
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-bug\n"
            "+fixed\n"
            "*** End Patch",
        )
        self.assertIn("apply_patch succeeded", patched.observation)
        final = self.manager.step(
            slot,
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}',
        )
        self.assertTrue(final.done)
        self.assertEqual(final.reward, 1.0)
        self.assertIs(final.info["episode_success"], True)
        self.assertEqual(final.info["action_kind"], "final")
        self.assertEqual(
            final.info["actor_credit"]["basis"], "terminal_submission"
        )
        self.assertEqual(self.grader.calls, 1)

        detail = self.manager.detail(slot)
        self.assertEqual(detail["data_idx"], 0)
        self.assertEqual(detail["slot_id"], slot)
        self.assertEqual(detail["instance_id"], self.instance_id)
        self.assertEqual(
            Path(detail["workspace"]["episode_root"]).parent,
            self.episodes.resolve(),
        )
        self.assertEqual(
            Path(detail["workspace"]["policy_root"]).parent,
            Path(detail["workspace"]["episode_root"]),
        )
        self.assertEqual(
            detail["workspace"]["model_uid"],
            1000 if os.getuid() == 0 else os.getuid(),
        )
        self.assertEqual(
            detail["workspace"]["model_gid"],
            1000 if os.getgid() == 0 else os.getgid(),
        )
        self.assertEqual(detail["evidence"][2]["result"]["stdout"], "persistent")
        closed = self.manager.close(slot)
        self.assertTrue(closed["closed"])
        self.assertEqual(self.manager.metadata()["active_slot_count"], 0)
        self.assertEqual(self.manager.metadata()["active_environment_count"], 0)
        self.assertEqual(self.manager.metadata()["active_workspace_count"], 0)
        self.assertEqual(list(self.episodes.iterdir()), [])
        audits = list(self.audits.glob("episode-*.json"))
        self.assertEqual(len(audits), 1)
        audit = json.loads(audits[0].read_text(encoding="utf-8"))
        self.assertEqual(audit["schema"], AUDIT_SCHEMA)
        self.assertEqual(audit["close_reason"], "client_close")
        self.assertEqual(audit["slot_id"], slot)
        self.assertEqual(audit["data_idx"], 0)
        self.assertEqual(audit["instance_id"], self.instance_id)
        self.assertEqual(audit["grade"]["reward"], 1.0)
        self.assertEqual(audit["evidence"][2]["result"]["stdout"], "persistent")
        self.assertIn("Fix the public value", audit["initial_observation"])
        self.assertEqual(audits[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.audits.stat().st_mode & 0o777, 0o700)
        self.assertEqual(list(self.audits.glob(".*.tmp")), [])

    def test_raw_checkpoint_block_requires_wrapper_control_state(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        raw = (
            "shell_command\n"
            "cat > .agent_memory/CONTINUATION.md <<'AGENT_MEMORY_EOF'\n"
            "objective: fix the public value\n"
            "evidence: src/value.py is still buggy\n"
            "next: inspect and edit src/value.py\n"
            "AGENT_MEMORY_EOF"
        )

        ordinary = self.manager.step(slot, raw)
        self.assertEqual(ordinary.info["action_kind"], "parser_error")
        self.assertFalse(ordinary.info["filesystem_checkpoint"]["exists"])
        self.assertNotIn("policy_control", ordinary.info)

        controlled = self.manager.step(
            slot,
            raw,
            policy_control={
                "schema": "task_neutral_policy_control_v1",
                "kind": "context_compaction",
            },
        )
        self.assertEqual(controlled.info["action_kind"], "shell_command")
        self.assertTrue(controlled.info["filesystem_checkpoint"]["changed"])
        self.assertEqual(
            controlled.info["policy_control"],
            {
                "schema": "task_neutral_policy_control_v1",
                "kind": "context_compaction",
            },
        )
        self.manager.close(slot)

    def test_filesystem_checkpoint_receipt_requires_current_action_change(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        action = (
            'shell_command {"command":"mkdir -p .agent_memory && '
            'printf checkpoint > .agent_memory/CONTINUATION.md","workdir":"."}'
        )
        written = self.manager.step(slot, action)
        self.assertFalse(written.done)
        receipt = written.info["filesystem_checkpoint"]
        self.assertEqual(
            receipt["schema"], "agentmemory_filesystem_checkpoint_receipt_v1"
        )
        self.assertEqual(receipt["path"], ".agent_memory/CONTINUATION.md")
        self.assertEqual(receipt["action_kind"], "shell_command")
        self.assertTrue(receipt["action_completed"])
        self.assertTrue(receipt["changed"])
        self.assertTrue(receipt["exists"])
        self.assertTrue(receipt["regular_file"])
        self.assertEqual(receipt["size_bytes"], len(b"checkpoint"))
        self.assertEqual(len(receipt["sha256"]), 64)

        unchanged = self.manager.step(slot, action)
        self.assertFalse(unchanged.done)
        unchanged_receipt = unchanged.info["filesystem_checkpoint"]
        self.assertTrue(unchanged_receipt["action_completed"])
        self.assertFalse(unchanged_receipt["changed"])
        self.assertTrue(unchanged_receipt["exists"])

    def test_empty_oversized_or_failed_checkpoint_write_is_not_valid(self) -> None:
        cases = (
            (
                'mkdir -p .agent_memory && : > .agent_memory/CONTINUATION.md',
                0,
                True,
            ),
            (
                'mkdir -p .agent_memory && head -c 8193 /dev/zero > '
                '.agent_memory/CONTINUATION.md',
                8193,
                True,
            ),
            (
                'mkdir -p .agent_memory && printf partial > '
                '.agent_memory/CONTINUATION.md; exit 1',
                len(b"partial"),
                False,
            ),
        )
        for command, expected_size, completed in cases:
            with self.subTest(command=command):
                slot = self.manager.create()
                self.manager.reset(slot, 0)
                result = self.manager.step(
                    slot,
                    'shell_command ' + json.dumps(
                        {"command": command, "workdir": "."},
                        separators=(",", ":"),
                    ),
                )
                receipt = result.info["filesystem_checkpoint"]
                self.assertEqual(receipt["size_bytes"], expected_size)
                self.assertEqual(receipt["action_completed"], completed)
                self.assertTrue(receipt["changed"])
                self.assertTrue(receipt["exists"])
                self.assertTrue(receipt["regular_file"])
                if expected_size > 8192:
                    self.assertIsNone(receipt["sha256"])
                self.manager.close(slot)

    def test_unrelated_workspace_write_does_not_authorize_checkpoint(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        result = self.manager.step(
            slot,
            'shell_command {"command":"printf note > notes.txt","workdir":"."}',
        )
        receipt = result.info["filesystem_checkpoint"]
        self.assertTrue(receipt["action_completed"])
        self.assertFalse(receipt["changed"])
        self.assertFalse(receipt["exists"])
        self.assertIsNone(receipt["size_bytes"])
        self.assertIsNone(receipt["sha256"])

    def test_no_source_change_submission_is_terminal_once_without_grading(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        generated = self.manager.step(
            slot,
            'shell_command {"command":"mkdir -p src/__pycache__; '
            'printf cache > src/__pycache__/noise.pyc","workdir":"."}',
        )
        self.assertFalse(generated.done)

        result = self.manager.step(
            slot,
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}',
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.info["sample_excluded"])
        self.assertEqual(self.grader.calls, 0)
        detail = self.manager.detail(slot)
        event = detail["evidence"][-1]
        self.assertEqual(event["termination_reason"], "submission_without_source_change")
        self.assertEqual(event["submission"]["source_changed_paths"], [])
        with self.assertRaisesRegex(RuntimeError, "already terminal"):
            self.manager.step(slot, 'shell_command {"command":"pwd","workdir":"."}')
        self.manager.close(slot)

    def test_checkpoint_only_submission_is_not_a_source_change(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        checkpoint = self.manager.step(
            slot,
            'shell_command {"command":"printf checkpoint > '
            '.agent_memory/CONTINUATION.md","workdir":"."}',
        )
        self.assertFalse(checkpoint.done)
        self.assertTrue(checkpoint.info["filesystem_checkpoint"]["changed"])

        result = self.manager.step(
            slot,
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}',
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(self.grader.calls, 0)
        detail = self.manager.detail(slot)
        event = detail["evidence"][-1]
        self.assertEqual(event["termination_reason"], "submission_without_source_change")
        self.assertEqual(event["submission"]["source_changed_paths"], [])
        self.manager.close(slot)

    def test_changed_but_unresolved_submission_gets_zero_terminal_reward(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        self.manager.step(
            slot,
            'shell_command {"command":"printf still-wrong > src/value.py","workdir":"."}',
        )

        result = self.manager.step(
            slot,
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}',
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.info["sample_excluded"])
        self.assertEqual(self.grader.calls, 1)
        detail = self.manager.detail(slot)
        event = detail["evidence"][-1]
        self.assertEqual(event["termination_reason"], "grader_unresolved")
        self.assertEqual(event["terminal_grade"]["native_reward"], 0.0)
        self.manager.close(slot)

    def test_grader_infrastructure_failure_is_excluded_not_penalized(self) -> None:
        grader = InfrastructureFailingGrader()
        manager = SwesmithEpisodeManager(
            dataset=self.manager.dataset,
            materializer=self.manager.materializer,
            profile_resolver=Resolver(),
            sandbox_factory=lambda _record, _profile: LocalSandbox(),
            grader=grader,
            max_steps=8,
        )
        slot = manager.create()
        manager.reset(slot, 0)
        manager.step(
            slot,
            'shell_command {"command":"printf changed > src/value.py","workdir":"."}',
        )

        result = manager.step(
            slot,
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}',
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertTrue(result.info["sample_excluded"])
        self.assertEqual(
            result.info["sample_exclusion_reason"],
            "grader_infrastructure_failure",
        )
        self.assertEqual(grader.calls, 1)
        manager.close(slot)

    def test_submission_sentinel_must_be_the_first_stdout_line(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        result = self.manager.step(
            slot,
            'shell_command {"command":"printf \'prefix\\n'
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n\'","workdir":"."}',
        )

        self.assertFalse(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.info["action_kind"], "shell_command")
        self.assertEqual(self.grader.calls, 0)
        self.assertNotIn("Submission accepted", result.observation)
        self.manager.close(slot)

    def test_submission_sentinel_requires_a_zero_exit_status(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        result = self.manager.step(
            slot,
            'shell_command {"command":"printf \'COMPLETE_TASK_AND_SUBMIT_'
            'FINAL_OUTPUT\\n\'; exit 7","workdir":"."}',
        )

        self.assertFalse(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.info["action_kind"], "shell_command")
        self.assertEqual(self.grader.calls, 0)
        self.assertIn("exit_code=7", result.observation)
        self.manager.close(slot)

    def test_parser_error_repeats_exact_action_shapes(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        result = self.manager.step(slot, "shell_command pwd")

        self.assertFalse(result.done)
        self.assertEqual(result.reward, -0.01)
        self.assertFalse(result.info["sample_excluded"])
        self.assertEqual(
            result.info["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "parser_rejected",
            },
        )
        self.assertIn('shell_command {"command":"pwd","workdir":"."}', result.observation)
        self.assertIn("literal line apply_patch", result.observation)
        self.assertIn("one complete *** Begin Patch", result.observation)
        self.assertIn("no XML tags", result.observation)
        self.assertIn("surrounding text", result.observation)
        recovered = self.manager.step(
            slot, 'shell_command {"command":"pwd","workdir":"."}'
        )
        self.assertFalse(recovered.done)
        self.assertEqual(recovered.reward, 0.0)
        self.assertEqual(
            recovered.info["actor_credit"]["basis"], "shell_executed"
        )
        self.manager.close(slot)

    def test_output_pipe_cleanup_failure_is_an_executor_rejection(self) -> None:
        manager = SwesmithEpisodeManager(
            dataset=self.manager.dataset,
            materializer=self.manager.materializer,
            profile_resolver=Resolver(),
            sandbox_factory=lambda _record, _profile: OutputPipeLeakSandbox(),
            grader=self.grader,
            max_steps=8,
        )
        slot = manager.create()
        manager.reset(slot, 0)

        result = manager.step(
            slot,
            'shell_command {"command":"sleep 30 &","workdir":"."}',
        )

        self.assertFalse(result.done)
        self.assertEqual(result.reward, -0.01)
        self.assertFalse(result.info["sample_excluded"])
        self.assertEqual(result.info["action_kind"], "shell_command")
        self.assertEqual(
            result.info["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "executor_rejected",
            },
        )
        self.assertIn(
            "shell_command failed: shell sandbox left an output pipe open",
            result.observation,
        )
        manager.close(slot)

    def test_stable_json_sha256_accepts_lone_surrogate(self) -> None:
        digest = _stable_json_sha256({"command": "\udcff"})

        self.assertEqual(
            digest,
            "e4049663273d87ec3e803d43db4d495f22fbc23bae7b455170aeb0fd0c483bc9",
        )
        self.assertEqual(digest, _stable_json_sha256({"command": "\udcff"}))

    def test_stable_json_sha256_preserves_ascii_digest(self) -> None:
        self.assertEqual(
            _stable_json_sha256({"command": "printf ok", "workdir": "."}),
            "ce54a7b3a1f4907ea937c40a52d805131ea3c1f485678f3a2f7ef177019e0411",
        )

    def test_shell_action_progress_accepts_surrogate_payloads(self) -> None:
        action = parse_policy_action(
            'shell_command {"command":"printf \\udcff","workdir":"."}'
        )
        self.assertIsInstance(action, ParsedPolicyAction)
        self.assertEqual(action.arguments, {"command": "printf \udcff", "workdir": "."})
        progress = _shell_action_progress(
            action,
            result={
                "exit_code": 0,
                "timed_out": False,
                "stdout": "\udcff",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
                "termination_reason": None,
                "workspace_diff": {
                    "before_tree_sha256": "a" * 64,
                    "after_tree_sha256": "a" * 64,
                    "changed_paths": [],
                },
            },
        )

        self.assertEqual(progress["schema"], "swesmith_action_progress_v1")
        self.assertRegex(progress["action_fingerprint"], r"\A[0-9a-f]{64}\Z")
        self.assertRegex(progress["result_fingerprint"], r"\A[0-9a-f]{64}\Z")
        self.assertFalse(progress["workspace_changed"])

    def test_oversized_shell_output_is_bounded_with_a_visible_marker(self) -> None:
        observation = _shell_observation(
            exit_code=0,
            elapsed_ms=1,
            timed_out=False,
            stdout="A" * 10_000,
            stderr="B" * 10_000,
            stdout_truncated=True,
            stderr_truncated=True,
            changed_paths=(),
            max_observation_bytes=128,
        )

        self.assertIn("visible_output_truncated=true", observation)
        self.assertIn("[stdout truncated: visible output budget reached]", observation)
        self.assertIn("[stderr truncated: visible output budget reached]", observation)
        self.assertNotIn("A" * 10_000, observation)
        self.assertNotIn("B" * 10_000, observation)

    def test_changed_paths_summary_is_bounded_and_digest_backed(self) -> None:
        paths = tuple(f"generated/{index:04d}/" + "x" * 240 for index in range(3230))
        summary = _changed_paths_observation(paths, max_bytes=768)

        expected_digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()
        self.assertLessEqual(len(summary.encode()), 768)
        self.assertIn("3230 paths", summary)
        self.assertIn(f"sha256={expected_digest}", summary)
        self.assertIn("omitted=3228", summary)
        self.assertNotIn(paths[100], summary)

        tiny_summary = _changed_paths_observation(paths, max_bytes=128)
        self.assertLessEqual(len(tiny_summary.encode()), 128)
        self.assertIn("3230 paths", tiny_summary)
        self.assertIn(f"sha256={expected_digest}", tiny_summary)

    def test_shell_observation_does_not_expand_with_changed_path_count(self) -> None:
        paths = tuple(f"changed/{index:04d}/" + "y" * 240 for index in range(3230))
        observation = _shell_observation(
            exit_code=0,
            elapsed_ms=1,
            timed_out=False,
            stdout="ok",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            changed_paths=paths,
            max_observation_bytes=6144,
        )

        self.assertLess(len(observation.encode()), 8000)
        self.assertIn("3230 paths", observation)
        self.assertNotIn(paths[100], observation)

    def test_bounded_reasoning_prefix_executes_exactly_one_shell_action(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        result = self.manager.step(
            slot,
            "I found the bug. Let me record it.\n\n"
            'shell_command {"command":"printf changed > notes.txt"}',
        )

        self.assertFalse(result.done)
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.info["sample_excluded"])
        self.assertEqual(result.info["action_kind"], "shell_command")
        self.assertTrue(result.info["actor_credit"]["positive_eligible"])
        self.assertEqual(self.grader.calls, 0)
        detail = self.manager.detail(slot)
        self.assertEqual(
            Path(detail["workspace"]["policy_root"], "notes.txt").read_text(),
            "changed",
        )
        action = detail["evidence"][-1]["action"]
        self.assertEqual(action["thought"], "I found the bug. Let me record it.")
        self.manager.close(slot)

    def test_native_shell_and_patch_use_the_existing_execution_paths(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        shell = self.manager.step(
            slot,
            "<tool_call>\n"
            "<function=shell_command>\n"
            "<parameter=command>\n"
            "printf native > notes.txt\n"
            "</parameter>\n"
            "<parameter=workdir>\n"
            ".\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
        )
        self.assertEqual(shell.info["action_kind"], "shell_command")
        self.assertEqual(shell.info["actor_credit"]["basis"], "shell_executed")
        self.assertTrue(shell.info["actor_credit"]["positive_eligible"])
        self.assertIn("notes.txt", shell.observation)

        patched = self.manager.step(
            slot,
            "<tool_call>\n"
            "<function=apply_patch>\n"
            "<parameter=patch>\n"
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-bug\n"
            "+fixed\n"
            "*** End Patch\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
        )
        self.assertEqual(patched.info["action_kind"], "apply_patch")
        self.assertEqual(
            patched.info["actor_credit"]["basis"], "workspace_changed"
        )
        self.assertTrue(patched.info["actor_credit"]["positive_eligible"])
        self.assertIn("apply_patch succeeded", patched.observation)
        self.manager.close(slot)

    def test_shell_progress_fingerprint_ignores_runtime_and_tracks_mutation(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        direct = self.manager.step(
            slot,
            'shell_command {"command":"cat src/value.py","workdir":"."}',
        )
        native = self.manager.step(
            slot,
            "<tool_call>\n"
            "<function=shell_command>\n"
            "<parameter=command>\n"
            "cat src/value.py\n"
            "</parameter>\n"
            "<parameter=workdir>\n"
            ".\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
        )

        self.assertEqual(
            direct.info["action_progress"],
            native.info["action_progress"],
        )
        self.assertEqual(
            set(direct.info["action_progress"]),
            {
                "schema",
                "action_fingerprint",
                "result_fingerprint",
                "workspace_changed",
            },
        )
        self.assertFalse(direct.info["action_progress"]["workspace_changed"])

        mutation = self.manager.step(
            slot,
            'shell_command {"command":"printf changed > notes.txt","workdir":"."}',
        )
        self.assertTrue(mutation.info["action_progress"]["workspace_changed"])
        self.manager.close(slot)

    def test_rejected_patch_is_not_eligible_for_positive_actor_credit(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        result = self.manager.step(
            slot,
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-line that does not exist\n"
            "+fixed\n"
            "*** End Patch",
        )

        self.assertFalse(result.done)
        self.assertEqual(result.reward, -0.01)
        self.assertFalse(result.info["sample_excluded"])
        self.assertIn("apply_patch failed", result.observation)
        self.assertEqual(
            result.info["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "executor_rejected",
            },
        )
        recovered = self.manager.step(
            slot, 'shell_command {"command":"pwd","workdir":"."}'
        )
        self.assertFalse(recovered.done)
        self.assertEqual(recovered.reward, 0.0)
        self.assertEqual(
            recovered.info["actor_credit"]["basis"], "shell_executed"
        )
        self.manager.close(slot)

    def test_noop_patch_is_not_eligible_for_positive_actor_credit(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)

        result = self.manager.step(
            slot,
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-bug\n"
            "+bug\n"
            "*** End Patch",
        )

        self.assertFalse(result.done)
        self.assertIn("apply_patch succeeded", result.observation)
        self.assertEqual(
            result.info["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "no_workspace_change",
            },
        )
        detail = self.manager.detail(slot)
        self.assertEqual(detail["evidence"][-1]["result"]["workspace_diff"]["changed_paths"], [])
        self.manager.close(slot)

    def test_reset_replaces_previous_episode_with_pristine_workspace(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        self.manager.step(
            slot,
            'shell_command {"command":"printf stale > notes.txt"}',
        )
        self.manager.reset(slot, 0)
        result = self.manager.step(
            slot,
            'shell_command {"command":"test ! -e notes.txt"}',
        )
        self.assertIn("exit_code=0", result.observation)
        self.manager.close(slot)

    def test_unfinished_client_close_persists_private_audit(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        self.manager.step(slot, 'shell_command {"command":"printf unfinished"}')
        self.manager.close(slot)

        audit_path = next(self.audits.glob("episode-*.json"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["close_reason"], "client_close")
        self.assertFalse(audit["done"])
        self.assertEqual(audit["reward"], 0.0)
        self.assertIsNone(audit["grade"])
        self.assertEqual(audit["evidence"][1]["result"]["stdout"], "unfinished")

    def test_policy_turn_horizon_fails_without_grading_workspace(self) -> None:
        slot = self.manager.create()
        self.manager.reset(slot, 0)
        self.manager.step(
            slot,
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-bug\n"
            "+fixed\n"
            "*** End Patch",
        )

        horizon = self.manager.finalize_horizon(slot)
        self.assertTrue(horizon.done)
        self.assertEqual(horizon.reward, -0.01)
        self.assertIs(horizon.info["episode_success"], False)
        self.assertEqual(self.grader.calls, 0)
        detail = self.manager.detail(slot)
        self.assertEqual(detail["step_count"], 1)
        self.assertIsNone(detail["grade"])
        self.assertEqual(
            detail["evidence"][-1]["action"]["kind"],
            "policy_turn_horizon",
        )
        self.assertEqual(detail["evidence"][-1]["event"], "horizon_exhaustion")
        self.assertEqual(
            detail["evidence"][-1]["termination_reason"],
            "policy_turn_horizon",
        )
        with self.assertRaisesRegex(RuntimeError, "already terminal"):
            self.manager.finalize_horizon(slot)
        self.manager.close(slot)

    def test_native_step_limit_fails_without_implicit_submission(self) -> None:
        manager = SwesmithEpisodeManager(
            dataset=self.manager.dataset,
            materializer=self.manager.materializer,
            profile_resolver=Resolver(),
            sandbox_factory=lambda _record, _profile: LocalSandbox(),
            grader=self.grader,
            max_steps=1,
        )
        slot = manager.create()
        manager.reset(slot, 0)

        result = manager.step(
            slot,
            'shell_command {"command":"printf fixed > src/value.py","workdir":"."}',
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, -0.01)
        self.assertFalse(result.info["sample_excluded"])
        self.assertIs(result.info["episode_success"], False)
        self.assertEqual(self.grader.calls, 0)
        detail = manager.detail(slot)
        self.assertIsNone(detail["grade"])
        self.assertEqual(detail["evidence"][-1]["event"], "horizon_exhaustion")
        self.assertEqual(detail["evidence"][-1]["termination_reason"], "max_steps")
        manager.close(slot)

    def test_concurrent_episode_closes_write_unique_atomic_audits(self) -> None:
        slots = [self.manager.create(), self.manager.create()]
        for slot in slots:
            self.manager.reset(slot, 0)

        errors: list[BaseException] = []

        def close(slot: int) -> None:
            try:
                self.manager.close(slot)
            except BaseException as exc:  # pragma: no cover - asserted below.
                errors.append(exc)

        threads = [threading.Thread(target=close, args=(slot,)) for slot in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        audits = list(self.audits.glob("episode-*.json"))
        self.assertEqual(len(audits), 2)
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in audits]
        self.assertEqual(len({document["audit_id"] for document in documents}), 2)
        self.assertTrue(all(document["grade"] is None for document in documents))
        self.assertEqual(list(self.audits.glob(".*.tmp")), [])

    def test_audit_failure_still_removes_private_episode_workspace(self) -> None:
        manager = SwesmithEpisodeManager(
            dataset=self.manager.dataset,
            materializer=self.manager.materializer,
            profile_resolver=Resolver(),
            sandbox_factory=lambda _record, _profile: LocalSandbox(),
            grader=self.grader,
            audit_sink=FailingAuditSink(),
            max_steps=8,
        )
        slot = manager.create()
        manager.reset(slot, 0)

        with self.assertRaisesRegex(OSError, "audit storage unavailable"):
            manager.close(slot)
        self.assertEqual(list(self.episodes.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
