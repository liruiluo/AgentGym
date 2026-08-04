from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import ShellExecutionResult


class InProcessTestShellSandbox:
    """Execute only test-authored commands in a temporary test workspace.

    Formal environments never construct this class. It exists so macOS unit
    tests can exercise the workspace transaction and audit layers without
    pretending to provide the Linux namespace security boundary.
    """

    @property
    def metadata(self):
        return {
            "contract": "test_only_inprocess_shell_v1",
            "formal_eligible": False,
        }

    def run(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> ShellExecutionResult:
        cwd = workspace_root if workdir == "." else workspace_root / workdir
        started = time.monotonic()
        try:
            completed = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_ms / 1000.0,
                check=False,
                env={
                    "HOME": str(workspace_root),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                },
            )
        except subprocess.TimeoutExpired as exc:
            return ShellExecutionResult(
                stdout=exc.stdout or b"",
                stderr=exc.stderr or b"",
                exit_code=124,
                elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
                timed_out=True,
                stdout_truncated=False,
                stderr_truncated=False,
                termination_reason="wall_timeout",
                sandbox_contract="test_only_inprocess_shell_v1",
            )
        return ShellExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
            termination_reason=None,
            sandbox_contract="test_only_inprocess_shell_v1",
        )
