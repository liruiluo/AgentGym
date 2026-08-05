from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import (
    LinuxNamespaceShellSandbox,
    ShellSandboxError,
    ShellSandboxLimits,
    _lease_ephemeral_model_uid,
    _collect_bounded_output,
    _normalize_sha256,
    _validate_staged_workspace,
    assert_executable_fingerprint,
    executable_fingerprint,
)


def limits(**overrides: int) -> ShellSandboxLimits:
    values = {
        "workspace_bytes": 4096,
        "workspace_inodes": 9,
        "max_files": 4,
        "max_directories": 4,
        "max_file_bytes": 1024,
        "max_path_chars": 80,
        "default_timeout_ms": 1000,
        "max_timeout_ms": 2000,
        "cpu_seconds": 1,
        "address_space_bytes": 128 * 1024 * 1024,
        "max_processes": 4,
        "max_open_files": 16,
        "stdout_bytes": 1024,
        "stderr_bytes": 1024,
        "tmp_bytes": 4096,
        "tmp_inodes": 16,
    }
    values.update(overrides)
    return ShellSandboxLimits(**values)


class BoundedOutputTests(unittest.TestCase):
    def test_stdout_and_stderr_are_streamed_with_hard_visible_bounds(self) -> None:
        process = subprocess.Popen(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                "i=0; while [ $i -lt 5000 ]; do printf x; printf y >&2; "
                "i=$((i+1)); done",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr, stdout_truncated, stderr_truncated, timed_out = (
            _collect_bounded_output(
                process,
                stdout_limit=257,
                stderr_limit=193,
                timeout_ms=2000,
            )
        )
        self.assertEqual(process.returncode, 0)
        self.assertFalse(timed_out)
        self.assertEqual(stdout, b"x" * 257)
        self.assertEqual(stderr, b"y" * 193)
        self.assertTrue(stdout_truncated)
        self.assertTrue(stderr_truncated)


class StagedWorkspaceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.limits = limits()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_rejected(self, expected: str) -> None:
        with self.assertRaisesRegex(ShellSandboxError, expected):
            _validate_staged_workspace(self.root, self.limits)

    def test_accepts_private_regular_tree(self) -> None:
        directory = self.root / "notes"
        directory.mkdir()
        (directory / "MEMORY.md").write_text("finish=black\n", encoding="utf-8")
        _validate_staged_workspace(self.root, self.limits)

    def test_rejects_symlink_hardlink_fifo_and_unix_socket(self) -> None:
        cases = ("symlink", "hardlink", "fifo", "socket")
        for case in cases:
            with self.subTest(case=case):
                for path in self.root.iterdir():
                    if path.is_dir() and not path.is_symlink():
                        for child in path.iterdir():
                            child.unlink()
                        path.rmdir()
                    else:
                        path.unlink()
                if case == "symlink":
                    (self.root / "bad").symlink_to("target")
                elif case == "hardlink":
                    source = self.root / "source"
                    source.write_text("x", encoding="utf-8")
                    os.link(source, self.root / "bad")
                elif case == "fifo":
                    os.mkfifo(self.root / "bad")
                else:
                    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        listener.bind(str(self.root / "bad"))
                    finally:
                        listener.close()
                self.assert_rejected("symlinks, hard links, or special files")

    def test_rejects_file_directory_byte_inode_and_path_limits(self) -> None:
        (self.root / "large").write_bytes(b"x" * 1025)
        self.assert_rejected("file larger")
        (self.root / "large").unlink()

        for index in range(5):
            (self.root / f"f{index}").write_text("x", encoding="utf-8")
        self.assert_rejected("file-count")
        for path in self.root.iterdir():
            path.unlink()

        for index in range(5):
            (self.root / f"d{index}").mkdir()
        self.assert_rejected("directory-count")
        for path in self.root.iterdir():
            path.rmdir()

        (self.root / ("p" * 81)).write_text("x", encoding="utf-8")
        self.assert_rejected("path longer")


class PinValidationTests(unittest.TestCase):
    def test_prepared_static_roots_are_visible_to_the_model_uid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rootfs = root / "rootfs"
            output = root / "output"
            rootfs.mkdir(mode=0o700)
            output.mkdir(mode=0o700)

            LinuxNamespaceShellSandbox._prepare_rootfs(  # type: ignore[arg-type]
                None,
                rootfs,
                output,
                model_uid=1_500_000_001,
            )

            self.assertEqual(rootfs.stat().st_mode & 0o777, 0o755)
            self.assertEqual((rootfs / "etc").stat().st_mode & 0o777, 0o755)
            self.assertEqual((rootfs / "tools").stat().st_mode & 0o777, 0o755)
            self.assertEqual((rootfs / "tools/rg").stat().st_mode & 0o777, 0o755)

    def test_sha256_pin_is_canonical_and_fail_closed(self) -> None:
        self.assertEqual(_normalize_sha256("A" * 64, "pin"), "a" * 64)
        for value in ("", "g" * 64, "a" * 63, None):
            with self.subTest(value=value), self.assertRaises(ShellSandboxError):
                _normalize_sha256(value, "pin")

    def test_stat_fingerprint_rejects_executable_drift_without_rehashing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            executable = Path(raw) / "rg"
            executable.write_bytes(b"first")
            executable.chmod(0o755)
            fingerprint = executable_fingerprint(executable)
            assert_executable_fingerprint(executable, fingerprint, "ripgrep")
            executable.write_bytes(b"second")
            with self.assertRaisesRegex(ShellSandboxError, "changed after startup"):
                assert_executable_fingerprint(executable, fingerprint, "ripgrep")

    def test_uid_lease_is_exclusive_for_overlapping_commands(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lease_root = Path(raw) / "leases"
            with _lease_ephemeral_model_uid(
                lease_root, slot_count=2
            ) as first_uid, _lease_ephemeral_model_uid(
                lease_root, slot_count=2
            ) as second_uid:
                self.assertNotEqual(first_uid, second_uid)
                with self.assertRaisesRegex(ShellSandboxError, "slots are in use"):
                    with _lease_ephemeral_model_uid(lease_root, slot_count=2):
                        self.fail("an occupied UID lease must not be reused")


if __name__ == "__main__":
    unittest.main()
