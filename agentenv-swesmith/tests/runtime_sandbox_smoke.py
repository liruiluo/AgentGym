#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import ShellSandboxLimits
from agentenv_swesmith.sandbox import (
    LinuxNamespaceEpisodeSandbox,
    SwesmithSandboxError,
)


def _limits() -> ShellSandboxLimits:
    return ShellSandboxLimits(
        workspace_bytes=8 * 1024 * 1024,
        workspace_inodes=4096,
        max_files=3072,
        max_directories=1024,
        max_file_bytes=4 * 1024 * 1024,
        max_path_chars=512,
        default_timeout_ms=10_000,
        max_timeout_ms=20_000,
        cpu_seconds=10,
        address_space_bytes=1024 * 1024 * 1024,
        max_processes=32,
        max_open_files=128,
        stdout_bytes=64 * 1024,
        stderr_bytes=64 * 1024,
        tmp_bytes=1024 * 1024,
        tmp_inodes=512,
    )


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        os.chown(current_path, uid, gid, follow_symlinks=False)
        for name in [*directories, *files]:
            path = current_path / name
            if not path.is_symlink():
                os.chown(path, uid, gid, follow_symlinks=False)


def _new_workspace(parent: Path, sandbox: LinuxNamespaceEpisodeSandbox) -> Path:
    root = Path(tempfile.mkdtemp(prefix="workspace-", dir=parent))
    (root / "src").mkdir()
    (root / "src/value.txt").write_text("initial\n", encoding="utf-8")
    (root / "value-link").symlink_to("src/value.txt")
    _chown_tree(root, sandbox.model_uid, sandbox.model_gid)
    sandbox.attach_workspace(root)
    return root


def _new_sandbox(args: argparse.Namespace) -> LinuxNamespaceEpisodeSandbox:
    return LinuxNamespaceEpisodeSandbox.from_environment(
        limits=_limits(),
        rg_binary=args.rg_binary,
        expected_rg_sha256=args.rg_sha256,
        oci_cache_root=args.oci_cache_root,
        repo_profile_image=args.repo_profile_image,
        repo_profile_digest=args.repo_profile_digest,
        lease_root=args.lease_root,
        lease_slots=32,
        run_preflight=True,
    )


def _assert_process_gone(pid_path: Path, label: str) -> None:
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise AssertionError(f"{label} did not record a host PID") from exc
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if Path(f"/proc/{pid}").exists():
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        raise AssertionError(f"{label} process survived cleanup: pid={pid} {cmdline!r}")


def _run_main_contract(args: argparse.Namespace, parent: Path) -> dict[str, object]:
    sandbox = _new_sandbox(args)
    try:
        root = _new_workspace(parent, sandbox)
        identity = sandbox.run(
            command=(
                f'test "$(id -u)" = "{sandbox.model_uid}" && '
                "test \"$(awk '/^CapEff:/ {print $2}' /proc/self/status)\" = "
                '"0000000000000000" && '
                "test \"$(awk '/^NoNewPrivs:/ {print $2}' /proc/self/status)\" = "
                '"1" && '
                "test \"$(awk 'END {print NR}' /proc/net/route)\" -le 1 && "
                "test -c /dev/null && test -d /dev/shm && test -r /proc/self/status && "
                "if test -r /run/out/status; then exit 43; fi && "
                "test \"$(command -v python)\" = "
                "/opt/miniconda3/envs/testbed/bin/python && "
                "python -c 'import pytest' && "
                "if touch /usr/SWESMITH_MUST_BE_READ_ONLY 2>/dev/null; then exit 41; fi && "
                "if touch /opt/SWESMITH_MUST_BE_READ_ONLY 2>/dev/null; then exit 42; fi && "
                "printf shm > /dev/shm/proof && test \"$(cat /dev/shm/proof)\" = shm && "
                "printf persisted > state.txt && cat value-link"
            ),
            workdir=".",
            timeout_ms=10_000,
        )
        assert identity.result.exit_code == 0, identity.result
        assert identity.result.stdout == b"initial\n", identity.result.stdout
        assert identity.workspace_diff.changed_paths == ("state.txt",)

        persistence = sandbox.run(
            command="test \"$(cat state.txt)\" = persisted && printf second >> state.txt",
            workdir=".",
            timeout_ms=10_000,
        )
        assert persistence.result.exit_code == 0, persistence.result
        assert persistence.workspace_diff.changed_paths == ("state.txt",)

        cleanup = sandbox.run(
            command=(
                "setsid bash -c 'trap \"\" TERM HUP; "
                "grep \"^NSpid:\" /proc/self/status | tr -s \"[:space:]\" \" \" | "
                "cut -d \" \" -f2 > normal-host-pid; "
                "while :; do sleep 1; done' swesmith-normal-background & "
                "for i in 1 2 3 4 5 6 7 8 9 10; do "
                "test -s normal-host-pid && break; sleep 0.05; done; "
                "test -s normal-host-pid"
            ),
            workdir=".",
            timeout_ms=10_000,
        )
        assert cleanup.result.exit_code == 0, cleanup.result
        _assert_process_gone(root / "normal-host-pid", "normal descendant")

        timed_out = sandbox.run(
            command=(
                "setsid bash -c 'trap \"\" TERM HUP; "
                "grep \"^NSpid:\" /proc/self/status | tr -s \"[:space:]\" \" \" | "
                "cut -d \" \" -f2 > timeout-host-pid; "
                "while :; do sleep 1; done' swesmith-timeout-background & "
                "for i in 1 2 3 4 5 6 7 8 9 10; do "
                "test -s timeout-host-pid && break; sleep 0.05; done; "
                "sleep 60"
            ),
            workdir=".",
            timeout_ms=750,
        )
        assert timed_out.result.timed_out
        assert timed_out.result.exit_code == 124
        assert timed_out.result.termination_reason == "wall_timeout"
        _assert_process_gone(root / "timeout-host-pid", "timed-out descendant")

        core_limit = sandbox.run(
            command=(
                "rm -f core core.*; "
                "(ulimit -c unlimited 2>/dev/null || true; "
                "/bin/bash -c 'kill -SEGV $$') >/dev/null 2>&1 || true; "
                "test -z \"$(find . -maxdepth 1 -type f -name 'core*' -print -quit)\" && "
                "printf CORE_DUMP_DISABLED_OK"
            ),
            workdir=".",
            timeout_ms=10_000,
        )
        assert core_limit.result.exit_code == 0, core_limit.result
        assert core_limit.result.stdout == b"CORE_DUMP_DISABLED_OK"

        tmp_limit = sandbox.run(
            command=(
                "if dd if=/dev/zero of=/tmp/too-large bs=262144 count=8 "
                "status=none 2>/dev/null; then exit 42; fi; printf TMP_LIMIT_OK"
            ),
            workdir=".",
            timeout_ms=10_000,
        )
        assert tmp_limit.result.exit_code == 0, tmp_limit.result
        assert tmp_limit.result.stdout == b"TMP_LIMIT_OK"
        assert not (root / "too-large").exists()
        return {
            "contract": sandbox.metadata["contract"],
            "oci_digest": sandbox.metadata["oci_rootfs"]["digest"],
            "model_uid": sandbox.model_uid,
            "initial_tree": identity.workspace_before.tree_sha256,
            "final_tree": tmp_limit.workspace_after.tree_sha256,
            "normal_cleanup": True,
            "timeout_cleanup": True,
            "network_routes": 0,
            "cap_eff": 0,
            "system_root_read_only": True,
            "core_dump_disabled": True,
            "tmp_limit": True,
        }
    finally:
        sandbox.close()


def _run_poison_contracts(args: argparse.Namespace, parent: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}

    symlink_sandbox = _new_sandbox(args)
    try:
        _new_workspace(parent, symlink_sandbox)
        try:
            symlink_sandbox.run(
                command="ln -s /etc/passwd escaped-link",
                workdir=".",
                timeout_ms=10_000,
            )
        except SwesmithSandboxError as exc:
            assert "absolute symlink" in str(exc), str(exc)
            assert symlink_sandbox.poisoned_reason is not None
            results["absolute_symlink_poisoned"] = True
        else:
            raise AssertionError("absolute workspace symlink did not poison the episode")
    finally:
        symlink_sandbox.close()

    quota_sandbox = _new_sandbox(args)
    try:
        _new_workspace(parent, quota_sandbox)
        command = "for n in 1 2 3; do dd if=/dev/zero of=q$n bs=1048576 count=3 status=none; done"
        try:
            quota_sandbox.run(command=command, workdir=".", timeout_ms=20_000)
        except SwesmithSandboxError as exc:
            assert "aggregate byte limit" in str(exc), str(exc)
            assert quota_sandbox.poisoned_reason is not None
            results["aggregate_quota_poisoned"] = True
        else:
            raise AssertionError("aggregate workspace overflow did not poison the episode")
    finally:
        quota_sandbox.close()

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rg-binary", type=Path, required=True)
    parser.add_argument("--rg-sha256", required=True)
    parser.add_argument("--oci-cache-root", type=Path, required=True)
    parser.add_argument("--repo-profile-image", required=True)
    parser.add_argument("--repo-profile-digest", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()

    actual = hashlib.sha256(args.rg_binary.read_bytes()).hexdigest()
    if actual != args.rg_sha256:
        raise SystemExit(f"ripgrep mismatch: expected {args.rg_sha256}, got {actual}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.lease_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=args.work_root) as raw:
        parent = Path(raw)
        evidence = _run_main_contract(args, parent)
        evidence.update(_run_poison_contracts(args, parent))
    print("SWESMITH_OCI_ROOTFS_RUNTIME_OK")
    for key in sorted(evidence):
        print(f"{key}={evidence[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
