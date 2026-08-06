from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agentenv_agentmemory.workspace_sandbox import ShellExecutionResult
from agentenv_swesmith.grader import SwesmithHiddenGrader
from agentenv_swesmith.profile import SwesmithProfileBinding
from agentenv_swesmith.workspace import HiddenTestFile, SwesmithWorkspace


def parse_status(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = fields[1]
    return result


def eval_report(status: dict[str, str], gold: dict[str, list[str]]) -> dict:
    def group(name: str) -> dict[str, list[str]]:
        success = [case for case in gold[name] if status.get(case) == "PASSED"]
        failure = [case for case in gold[name] if status.get(case) != "PASSED"]
        return {"success": success, "failure": failure}

    return {
        "FAIL_TO_PASS": group("FAIL_TO_PASS"),
        "PASS_TO_PASS": group("PASS_TO_PASS"),
    }


def resolution(report: dict) -> str:
    failures = [
        *report["FAIL_TO_PASS"]["failure"],
        *report["PASS_TO_PASS"]["failure"],
    ]
    return "FULL" if not failures else "PARTIAL"


class FakeSandbox:
    def __init__(self, policy_root: Path, outputs: list[dict]) -> None:
        self.policy_root = policy_root
        self.outputs = list(outputs)
        self.refresh_count = 0
        self.test_contents: list[str] = []

    def refresh_after_host_mutation(self):
        self.refresh_count += 1
        return SimpleNamespace(changed_paths=("tests/test_fix.py",))

    def run(self, *, command: str, workdir: str, timeout_ms: int):
        self.test_contents.append(
            (self.policy_root / "tests/test_fix.py").read_text(encoding="utf-8")
        )
        spec = self.outputs.pop(0)
        if spec.get("mutate_test"):
            (self.policy_root / "tests/test_fix.py").write_text(
                "mutated by test run\n", encoding="utf-8"
            )
        result = ShellExecutionResult(
            stdout=spec.get("stdout", "").encode(),
            stderr=spec.get("stderr", "").encode(),
            exit_code=spec.get("exit_code", 0),
            elapsed_ms=spec.get("elapsed_ms", 1),
            timed_out=spec.get("timed_out", False),
            stdout_truncated=spec.get("stdout_truncated", False),
            stderr_truncated=spec.get("stderr_truncated", False),
            termination_reason=None,
            sandbox_contract="test",
            model_uid=1000,
        )
        return SimpleNamespace(result=result)


class HiddenGraderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.policy = root / "workspace"
        self.hidden = root / "private" / "pristine-tests"
        (self.policy / "tests").mkdir(parents=True)
        (self.hidden / "tests").mkdir(parents=True)
        self.pristine = "def test_fix(): pass\n"
        (self.policy / "tests/test_fix.py").write_text(
            "policy tampering\n", encoding="utf-8"
        )
        (self.hidden / "tests/test_fix.py").write_text(
            self.pristine, encoding="utf-8"
        )
        hidden_file = HiddenTestFile(
            path="tests/test_fix.py",
            sha256=hashlib.sha256(self.pristine.encode()).hexdigest(),
            mode=0o644,
        )
        self.workspace = SwesmithWorkspace(
            episode_root=root,
            policy_root=self.policy,
            hidden_tests_root=self.hidden,
            mirror_root=root / "mirror",
            instance_id="owner__repo.deadbeef.task",
            bug_commit="a" * 40,
            pristine_commit="b" * 40,
            hidden_tests=(hidden_file,),
        )
        self.instance = {
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
        }
        self.profile = SwesmithProfileBinding(
            repo="swesmith/owner__repo.deadbeef",
            image="swebench/example",
            f2p_test_paths=("tests/test_fix.py",),
            p2p_test_paths=("tests/test_keep.py",),
            f2p_command="pytest tests/test_fix.py",
            full_command="pytest tests/test_fix.py tests/test_keep.py",
            log_parser=parse_status,
            get_eval_tests_report=eval_report,
            get_resolution_status=resolution,
            full_resolution_status="FULL",
        )
        self.grader = SwesmithHiddenGrader(timeout_ms=10_000)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_f2p_failure_stops_before_p2p_and_restores_hidden_test(self) -> None:
        sandbox = FakeSandbox(
            self.policy,
            [{"stdout": "tests/test_fix.py::test_fix FAILED\n"}],
        )
        result = self.grader.grade(
            instance=self.instance,
            profile=self.profile,
            workspace=self.workspace,
            sandbox=sandbox,
        )
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.resolution_status, "PARTIAL")
        self.assertIsNone(result.full_run)
        self.assertEqual(sandbox.refresh_count, 1)
        self.assertEqual(sandbox.test_contents, [self.pristine])

    def test_full_reward_requires_healthy_f2p_and_full_runs(self) -> None:
        sandbox = FakeSandbox(
            self.policy,
            [
                {
                    "stdout": "tests/test_fix.py::test_fix PASSED\n",
                    "mutate_test": True,
                },
                {
                    "stdout": (
                        "tests/test_fix.py::test_fix PASSED\n"
                        "tests/test_keep.py::test_keep PASSED\n"
                    )
                },
            ],
        )
        result = self.grader.grade(
            instance=self.instance,
            profile=self.profile,
            workspace=self.workspace,
            sandbox=sandbox,
        )
        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.resolved)
        self.assertEqual(sandbox.refresh_count, 2)
        self.assertEqual(sandbox.test_contents, [self.pristine, self.pristine])
        self.assertEqual(
            result.restored_test_paths,
            ("tests/test_fix.py", "tests/test_fix.py"),
        )

    def test_truncated_or_nonzero_full_run_fails_closed(self) -> None:
        for override in (
            {"stdout_truncated": True},
            {"exit_code": 1},
        ):
            with self.subTest(override=override):
                (self.policy / "tests/test_fix.py").write_text(
                    "tampered again\n", encoding="utf-8"
                )
                full = {
                    "stdout": (
                        "tests/test_fix.py::test_fix PASSED\n"
                        "tests/test_keep.py::test_keep PASSED\n"
                    ),
                    **override,
                }
                sandbox = FakeSandbox(
                    self.policy,
                    [
                        {"stdout": "tests/test_fix.py::test_fix PASSED\n"},
                        full,
                    ],
                )
                result = self.grader.grade(
                    instance=self.instance,
                    profile=self.profile,
                    workspace=self.workspace,
                    sandbox=sandbox,
                )
                self.assertEqual(result.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
