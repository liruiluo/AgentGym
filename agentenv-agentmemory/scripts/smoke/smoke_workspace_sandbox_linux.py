#!/usr/bin/env python3
"""Exercise the formal Codex shell sandbox on a real rootful Linux host."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from agentenv_agentmemory.persistent_workspace import WorkspaceLimits
from agentenv_agentmemory.workspace_sandbox import (
    LinuxNamespaceShellSandbox,
    ShellExecutionResult,
    ShellSandboxError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-rg-binary", required=True, type=Path)
    parser.add_argument("--workspace-rg-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limits = WorkspaceLimits()
    sandbox = LinuxNamespaceShellSandbox.from_environment(
        limits=limits.shell_limits(),
        rg_binary=args.workspace_rg_binary,
        expected_rg_sha256=args.workspace_rg_sha256,
    )
    cases: list[dict[str, object]] = []
    try:
        LinuxNamespaceShellSandbox.from_environment(
            limits=limits.shell_limits(),
            rg_binary=args.workspace_rg_binary,
            expected_rg_sha256="0" * 64,
            run_preflight=False,
        )
    except ShellSandboxError as exc:
        _require("does not match" in str(exc), "wrong ripgrep pin failure reason")
        cases.append({"name": "reject_wrong_rg_sha256", "rejected": True})
    else:
        raise AssertionError("wrong ripgrep SHA256 was accepted")
    with tempfile.TemporaryDirectory(prefix="agentmemory-linux-gate-") as raw:
        parent = Path(raw)

        def fresh(name: str) -> Path:
            root = parent / name
            root.mkdir(mode=0o700)
            return root

        def run(
            name: str,
            command: str,
            *,
            timeout_ms: int = 5000,
            root: Path | None = None,
        ) -> tuple[Path, ShellExecutionResult]:
            root = fresh(name) if root is None else root
            result = sandbox.run(
                root,
                command=command,
                workdir=".",
                timeout_ms=timeout_ms,
            )
            cases.append(
                {
                    "name": name,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "stdout_bytes": len(result.stdout),
                    "stderr_bytes": len(result.stderr),
                    "stdout_truncated": result.stdout_truncated,
                    "stderr_truncated": result.stderr_truncated,
                    "termination_reason": result.termination_reason,
                    "model_uid": result.model_uid,
                }
            )
            return root, result

        drift_rg = parent / "rg-drift-probe"
        shutil.copy2(args.workspace_rg_binary, drift_rg)
        drift_sandbox = LinuxNamespaceShellSandbox.from_environment(
            limits=limits.shell_limits(),
            rg_binary=drift_rg,
            expected_rg_sha256=args.workspace_rg_sha256,
            run_preflight=False,
        )
        with drift_rg.open("ab") as handle:
            handle.write(b"\n")
        try:
            drift_sandbox.run(
                fresh("reject_rg_drift"),
                command="printf should-not-run",
                workdir=".",
                timeout_ms=5000,
            )
        except ShellSandboxError as exc:
            _require("changed after startup" in str(exc), "ripgrep drift failure reason")
            cases.append({"name": "reject_rg_drift_after_startup", "rejected": True})
        else:
            raise AssertionError("ripgrep drift after startup was accepted")

        host_secret = parent / "host-secret.txt"
        secret_value = "HOST_SECRET_" + uuid.uuid4().hex
        host_secret.write_text(secret_value, encoding="utf-8")
        quoted_host_secret = shlex.quote(str(host_secret))
        _, result = run(
            "host_escape",
            "test ! -e "
            + quoted_host_secret
            + " && test ! -e /proc/1/root"
            + quoted_host_secret
            + " && printf isolated",
        )
        _require(result.exit_code == 0 and result.stdout == b"isolated", "host escape")
        _require(secret_value.encode() not in result.stdout + result.stderr, "host secret leak")

        readonly_marker = Path("/usr/bin/agentmemory-write-probe-" + uuid.uuid4().hex)
        _, result = run(
            "readonly_system_roots",
            "if touch "
            + shlex.quote(str(readonly_marker))
            + " 2>/dev/null; then exit 91; fi; "
            "awk '$5 == \"/usr\" || index($5, \"/usr/\") == 1 {print $5, $6}' "
            "/proc/self/mountinfo",
        )
        mount_lines = result.stdout.decode("utf-8").splitlines()
        _require(result.exit_code == 0, "read-only system root")
        _require(len(mount_lines) == 1 and mount_lines[0].startswith("/usr ro"), "non-recursive /usr bind")
        _require(not readonly_marker.exists(), "host /usr mutation")

        _, result = run(
            "privilege",
            "printf 'uid=%s\\n' \"$(id -u)\"; "
            "awk '/^(CapInh|CapPrm|CapEff|CapBnd|CapAmb|NoNewPrivs):/ {print}' "
            "/proc/self/status; "
            "if mknod /workspace/device c 1 3 2>/dev/null; then exit 92; fi; "
            "mkdir /tmp/mount-probe; "
            "if mount -t tmpfs none /tmp/mount-probe 2>/dev/null; then exit 94; fi; "
            "if unshare --mount true 2>/dev/null; then exit 95; fi",
        )
        privilege = result.stdout.decode("utf-8")
        _require(result.exit_code == 0, "privilege command")
        _require(f"uid={result.model_uid}" in privilege, "ephemeral uid")
        _require("CapInh:\t0000000000000000" in privilege, "inheritable capabilities")
        _require("CapPrm:\t0000000000000000" in privilege, "permitted capabilities")
        _require("CapEff:\t0000000000000000" in privilege, "effective capabilities")
        _require("CapBnd:\t0000000000000000" in privilege, "bounding capabilities")
        _require("CapAmb:\t0000000000000000" in privilege, "ambient capabilities")
        _require("NoNewPrivs:\t1" in privilege, "no_new_privileges")

        os.environ["AGENTMEMORY_SANDBOX_HOST_SECRET"] = secret_value
        _, result = run(
            "network_and_environment",
            "test -z \"${AGENTMEMORY_SANDBOX_HOST_SECRET-}\"; "
            "test \"$(awk 'NR > 1 {count++} END {print count + 0}' /proc/net/route)\" -eq 0; "
            "if exec 3<>/dev/tcp/1.1.1.1/80; then exit 93; fi; printf blocked",
        )
        _require(result.exit_code == 0 and result.stdout == b"blocked", "network namespace")

        persistence_root = fresh("persistence")
        _, write_result = run(
            "persistence_write",
            "mkdir -p .agent_memory && printf 'finish=black\\n' > .agent_memory/MEMORY.md",
            root=persistence_root,
        )
        _, read_result = run(
            "persistence_read",
            "rg -n --fixed-strings finish .agent_memory/MEMORY.md",
            root=persistence_root,
        )
        _require(write_result.exit_code == 0, "workspace write")
        _require(read_result.exit_code == 0 and b"finish=black" in read_result.stdout, "workspace persistence and rg")

        _, result = run(
            "output_limits",
            "i=0; while [ $i -lt 20000 ]; do printf x; printf y >&2; i=$((i+1)); done",
        )
        _require(result.exit_code == 0, "bounded output exit")
        _require(len(result.stdout) == limits.stdout_bytes, "stdout hard bound")
        _require(len(result.stderr) == limits.stderr_bytes, "stderr hard bound")
        _require(result.stdout_truncated and result.stderr_truncated, "output truncation flags")

        for name, command in (
            ("reject_symlink", "ln -s /usr/bin/bash bad"),
            ("reject_hardlink", "printf x > source && ln source bad"),
            ("reject_fifo", "mkfifo bad"),
            (
                "reject_socket",
                "python3 -c 'import socket; s=socket.socket(socket.AF_UNIX); "
                "s.bind(\"bad\"); s.close()'",
            ),
            ("reject_file_count", "for i in $(seq 1 65); do : > f$i; done"),
            ("reject_directory_count", "for i in $(seq 1 65); do mkdir d$i; done"),
        ):
            root = fresh(name)
            try:
                sandbox.run(root, command=command, workdir=".", timeout_ms=5000)
            except ShellSandboxError:
                pass
            else:
                raise AssertionError(f"{name} was not rejected")
            _require(not any(root.iterdir()), f"{name} mutated committed workspace")
            cases.append({"name": name, "rejected": True})

        large_root, result = run(
            "file_size_limit",
            "dd if=/dev/zero of=large bs=1024 count=65 status=none",
        )
        _require(result.exit_code != 0, "file-size limit exit")
        if (large_root / "large").exists():
            _require((large_root / "large").stat().st_size <= limits.max_file_bytes, "file-size hard cap")

        storage_root, result = run(
            "total_storage_limit",
            "for i in $(seq 1 9); do "
            "dd if=/dev/zero of=file-$i bs=1024 count=60 status=none || exit $?; "
            "done",
        )
        committed_bytes = sum(
            path.stat().st_size for path in storage_root.iterdir() if path.is_file()
        )
        _require(result.exit_code != 0, "total-storage limit exit")
        _require(committed_bytes <= limits.max_total_bytes, "total-storage hard cap")

        marker = "agentmemory-detached-" + uuid.uuid4().hex
        _, result = run(
            "timeout_cleanup",
            "setsid bash -c 'while :; do sleep 1; done' "
            + shlex.quote(marker)
            + " & while :; do sleep 1; done",
            timeout_ms=250,
        )
        _require(result.timed_out and result.exit_code == 124, "wall timeout")
        time.sleep(0.2)
        _require(not _host_process_contains(marker), "detached descendant cleanup")

        concurrent_roots = [fresh(f"concurrent-{index}") for index in range(8)]

        def concurrent_run(index: int) -> ShellExecutionResult:
            return sandbox.run(
                concurrent_roots[index],
                command="for i in 1 2 3; do sleep 0.2 & done; wait; printf ok",
                workdir=".",
                timeout_ms=5000,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            concurrent_results = list(executor.map(concurrent_run, range(8)))
        _require(all(item.exit_code == 0 and item.stdout == b"ok" for item in concurrent_results), "concurrent process limits")
        concurrent_uids = [item.model_uid for item in concurrent_results]
        _require(len(set(concurrent_uids)) == len(concurrent_uids), "unique concurrent UIDs")
        cases.append(
            {
                "name": "concurrent_nproc_isolation",
                "workers": len(concurrent_results),
                "model_uids": concurrent_uids,
            }
        )

    report = {
        "schema": "agentmemory_workspace_sandbox_linux_gate_v1",
        "status": "passed",
        "sandbox": dict(sandbox.metadata),
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "output_json": str(args.output_json)}))


def _host_process_contains(marker: str) -> bool:
    needle = marker.encode("utf-8")
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if needle in cmdline.read_bytes():
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(f"Linux sandbox gate failed: {label}")


if __name__ == "__main__":
    main()
