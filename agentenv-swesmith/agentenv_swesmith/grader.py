from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from agentenv_agentmemory.workspace_patch import WorkspacePatchError

from .profile import SwesmithProfileBinding
from .sandbox import LinuxNamespaceEpisodeSandbox, SwesmithShellExecution
from .workspace import SwesmithWorkspace, restore_hidden_tests


class SwesmithGraderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestRunEvidence:
    role: str
    command: str
    exit_code: int
    timed_out: bool
    output_truncated: bool
    elapsed_ms: int
    stdout: str
    stderr: str
    status_map: Mapping[str, str]
    status_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "command": self.command,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "elapsed_ms": self.elapsed_ms,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "status_map": dict(self.status_map),
            "status_source": self.status_source,
        }


@dataclass(frozen=True)
class SwesmithGradeResult:
    reward: float
    resolution_status: str | None
    report: Mapping[str, Any]
    restored_test_paths: tuple[str, ...]
    f2p_run: TestRunEvidence | None
    full_run: TestRunEvidence | None
    error: str | None = None

    @property
    def resolved(self) -> bool:
        return self.reward == 1.0

    def as_private_dict(self) -> dict[str, Any]:
        return {
            "schema": "swesmith_grade_evidence_v1",
            "reward": self.reward,
            "resolution_status": self.resolution_status,
            "report": _jsonable(self.report),
            "restored_test_paths": list(self.restored_test_paths),
            "f2p_run": None if self.f2p_run is None else self.f2p_run.as_dict(),
            "full_run": None if self.full_run is None else self.full_run.as_dict(),
            "error": self.error,
        }


class SwesmithHiddenGrader:
    """Run the official profile tests behind the policy/private boundary."""

    def __init__(
        self,
        *,
        timeout_ms: int,
        output_joiner: str = "\n",
        private_output_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("grader timeout_ms must be a positive integer")
        if type(private_output_bytes) is not int or private_output_bytes <= 0:
            raise ValueError("private_output_bytes must be a positive integer")
        self.timeout_ms = timeout_ms
        self.output_joiner = output_joiner
        self.private_output_bytes = private_output_bytes

    def grade(
        self,
        *,
        instance: Mapping[str, Any],
        profile: SwesmithProfileBinding,
        workspace: SwesmithWorkspace,
        sandbox: LinuxNamespaceEpisodeSandbox,
    ) -> SwesmithGradeResult:
        """Grade with the one full official command used by SWE-smith.

        ``profile.full_command`` already contains both FAIL_TO_PASS and
        PASS_TO_PASS selections.  Running an F2P command first and then the full
        command duplicates the failing tests and can make one terminal ``/step``
        exceed the client timeout even though each trusted grader phase remains
        within its own limit.
        """

        restored: list[str] = []
        full_run: TestRunEvidence | None = None
        try:
            restored.extend(restore_hidden_tests(workspace))
            sandbox.refresh_after_host_mutation()
            full_run = self._run_tests(
                role="P2P/full",
                command=profile.full_command,
                profile=profile,
                instance=instance,
                sandbox=sandbox,
            )
            gold = _gold_results(instance)
            report = _eval_report(profile, full_run.status_map, gold)
            status = _resolution(profile, report)
            reward = (
                1.0
                if _healthy(full_run) and status == profile.full_resolution_status
                else 0.0
            )
            return SwesmithGradeResult(
                reward=reward,
                resolution_status=status,
                report=report,
                restored_test_paths=tuple(restored),
                f2p_run=None,
                full_run=full_run,
            )
        except Exception as exc:
            return SwesmithGradeResult(
                reward=0.0,
                resolution_status=None,
                report={},
                restored_test_paths=tuple(restored),
                f2p_run=None,
                full_run=full_run,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_tests(
        self,
        *,
        role: str,
        command: str,
        profile: SwesmithProfileBinding,
        instance: Mapping[str, Any],
        sandbox: LinuxNamespaceEpisodeSandbox,
    ) -> TestRunEvidence:
        execution: SwesmithShellExecution = sandbox.run_trusted(
            command=command,
            workdir=".",
            timeout_ms=self.timeout_ms,
            stdout_limit_bytes=self.private_output_bytes,
            stderr_limit_bytes=self.private_output_bytes,
        )
        stdout = execution.result.stdout.decode("utf-8", errors="replace")
        stderr = execution.result.stderr.decode("utf-8", errors="replace")
        combined = stdout + self.output_joiner + stderr
        output_truncated = (
            execution.result.stdout_truncated or execution.result.stderr_truncated
        )
        if (
            output_truncated
            and execution.result.exit_code == 0
            and not execution.result.timed_out
        ):
            # A successful, attested test command is stronger than its bounded
            # human-readable stream: pytest's exit 0 means every test selected
            # by this official command passed.  Reconstruct only the declared
            # private test IDs; never infer a pass from a nonzero/timeout run.
            status_map = _declared_pass_status(role, instance)
            status_source = "exit_zero_declared_tests"
        elif output_truncated:
            status_map = {}
            status_source = "unavailable_truncated_output"
        else:
            status_map = dict(profile.log_parser(combined))
            status_source = "profile_log_parser"
        return TestRunEvidence(
            role=role,
            command=command,
            exit_code=execution.result.exit_code,
            timed_out=execution.result.timed_out,
            output_truncated=output_truncated,
            elapsed_ms=execution.result.elapsed_ms,
            stdout=stdout,
            stderr=stderr,
            status_map=status_map,
            status_source=status_source,
        )


def _gold_results(instance: Mapping[str, Any]) -> dict[str, list[str]]:
    f2p = instance.get("FAIL_TO_PASS")
    p2p = instance.get("PASS_TO_PASS")
    if not isinstance(f2p, list) or not isinstance(p2p, list):
        raise SwesmithGraderError("instance lacks declared FAIL_TO_PASS/PASS_TO_PASS lists")
    return {
        "FAIL_TO_PASS": [str(value) for value in f2p],
        "PASS_TO_PASS": [str(value) for value in p2p],
    }


def _declared_f2p_passed(
    status_map: Mapping[str, str],
    gold: Mapping[str, Any],
    profile: SwesmithProfileBinding,
) -> bool:
    report = _eval_report(profile, status_map, gold)
    failures = report.get("FAIL_TO_PASS", {})
    return isinstance(failures, Mapping) and not failures.get("failure", [])


def _declared_pass_status(
    role: str, instance: Mapping[str, Any]
) -> dict[str, str]:
    gold = _gold_results(instance)
    if role == "F2P":
        paths = gold["FAIL_TO_PASS"]
    elif role == "P2P/full":
        paths = [*gold["FAIL_TO_PASS"], *gold["PASS_TO_PASS"]]
    else:
        raise SwesmithGraderError(f"unknown test phase: {role!r}")
    return {path: "PASSED" for path in paths}


def _eval_report(
    profile: SwesmithProfileBinding,
    status_map: Mapping[str, str],
    gold: Mapping[str, Any],
) -> Mapping[str, Any]:
    return profile.get_eval_tests_report(dict(status_map), dict(gold))


def _resolution(profile: SwesmithProfileBinding, report: Mapping[str, Any]) -> str | None:
    try:
        return str(profile.get_resolution_status(report))
    except Exception:
        return None


def _healthy(run: TestRunEvidence | None) -> bool:
    return bool(
        run is not None
        and not run.timed_out
        and run.status_source in {"profile_log_parser", "exit_zero_declared_tests"}
    )


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return repr(value)
