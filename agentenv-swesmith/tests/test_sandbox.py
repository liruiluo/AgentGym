from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import (
    ExecutableFingerprint,
    ShellExecutionResult,
    ShellSandboxLimits,
)
from agentenv_swesmith.sandbox import (
    LinuxNamespaceEpisodeSandbox,
    SwesmithSandboxError,
    _attest_oci_rootfs_identity,
    _normalize_workdir,
    diff_workspace_trees,
    load_oci_rootfs_identity,
    snapshot_workspace_tree,
)


def limits(**overrides: int) -> ShellSandboxLimits:
    values = {
        "workspace_bytes": 16 * 1024,
        "workspace_inodes": 32,
        "max_files": 16,
        "max_directories": 12,
        "max_file_bytes": 4096,
        "max_path_chars": 120,
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


class _LeaseContext:
    def __init__(self) -> None:
        self.closed = False

    def __exit__(self, *_exc: object) -> None:
        self.closed = True


class _FakeEpisodeSandbox(LinuxNamespaceEpisodeSandbox):
    def __init__(self, mutation=None) -> None:
        self.lease = _LeaseContext()
        self.mutation = mutation
        super().__init__(
            limits=limits(),
            rg_binary=Path("/unused/rg"),
            expected_rg_sha256="0" * 64,
            rg_sha256="0" * 64,
            rg_version="ripgrep test",
            rg_fingerprint=ExecutableFingerprint(0, 0, 0, 0, 0, 0),
            binaries={},
            uid_lease_context=self.lease,
            model_uid=os.getuid(),
        )

    @property
    def model_gid(self) -> int:
        # Local unit tests run as a normal macOS user whose primary GID differs
        # from its UID; the formal Linux lease uses the same high UID/GID.
        return os.getgid()

    def _run_namespace(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> ShellExecutionResult:
        if self.mutation is not None:
            self.mutation(workspace_root)
        return ShellExecutionResult(
            stdout=b"ok",
            stderr=b"",
            exit_code=0,
            elapsed_ms=1,
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
            termination_reason=None,
            sandbox_contract="test",
            model_uid=self.model_uid,
        )


def _write_fake_oci_cache(parent: Path) -> tuple[Path, str, str, Path]:
    cache_root = parent / "oci"
    cache_root.mkdir(mode=0o700)
    image = "swebench/swesmith.x86_64.example_1776_repo.deadbeef"
    config = {
        "architecture": "amd64",
        "os": "linux",
        "config": {"WorkingDir": "/testbed"},
    }
    config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "config": {
            "digest": f"sha256:{config_sha}",
            "size": len(config_bytes),
        },
        "layers": [],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    digest = f"sha256:{manifest_sha}"
    cache_dir = cache_root / f"sha256-{manifest_sha}"
    cache_dir.mkdir(mode=0o700)
    rootfs = cache_dir / "rootfs"
    rootfs.mkdir(mode=0o755)
    for relative in (
        "testbed",
        "tmp",
        "var/tmp",
        "dev/shm",
        "proc",
        "run",
        "bin",
        "usr/bin",
        "opt/miniconda3/bin",
        "opt/miniconda3/envs/testbed/bin",
    ):
        (rootfs / relative).mkdir(parents=True, exist_ok=True)
    source = Path(sys.executable).resolve()
    for relative in (
        "bin/bash",
        "usr/bin/setpriv",
        "usr/bin/prlimit",
        "usr/bin/env",
        "bin/sleep",
        "usr/bin/cut",
        "opt/miniconda3/bin/python3.12",
        "opt/miniconda3/envs/testbed/bin/python",
    ):
        destination = rootfs / relative
        shutil.copyfile(source, destination)
        destination.chmod(0o755)
    executable_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "schema": "swesmith_oci_rootfs_cache_v1",
        "resolved_digest": digest,
        "repo_profile_image": image,
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "rootfs": {
            "bytes": 1,
            "regular_files": 8,
            "bash_sha256": executable_sha,
            "python312_sha256": executable_sha,
        },
    }
    (cache_dir / "manifest.json").write_bytes(manifest_bytes)
    (cache_dir / "config.json").write_bytes(config_bytes)
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    (cache_dir / ".complete").write_text("complete\n", encoding="ascii")
    return cache_root, image, digest, cache_dir


class OciRootfsIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.parent = Path(self.temporary.name)
        self.cache_root, self.image, self.digest, self.cache_dir = _write_fake_oci_cache(
            self.parent
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def load(self):
        return load_oci_rootfs_identity(
            self.cache_root,
            expected_image=self.image,
            expected_digest=self.digest,
            expected_owner_uid=os.getuid(),
        )

    def test_loads_complete_digest_pinned_cache(self) -> None:
        identity = self.load()
        self.assertEqual(identity.digest, self.digest)
        self.assertEqual(identity.image, self.image)
        self.assertEqual(identity.working_dir, "/testbed")
        _attest_oci_rootfs_identity(identity)

    def test_incomplete_or_wrong_profile_cache_fails_closed(self) -> None:
        (self.cache_dir / ".complete").write_text("partial\n", encoding="ascii")
        with self.assertRaisesRegex(SwesmithSandboxError, "incomplete"):
            self.load()
        (self.cache_dir / ".complete").write_text("complete\n", encoding="ascii")
        with self.assertRaisesRegex(SwesmithSandboxError, "profile image"):
            load_oci_rootfs_identity(
                self.cache_root,
                expected_image="swebench/other",
                expected_digest=self.digest,
                expected_owner_uid=os.getuid(),
            )

    def test_manifest_or_runtime_mutation_is_rejected(self) -> None:
        identity = self.load()
        manifest = self.cache_dir / "manifest.json"
        original_manifest = manifest.read_bytes()
        manifest.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(SwesmithSandboxError, "manifest changed"):
            _attest_oci_rootfs_identity(identity)
        manifest.write_bytes(original_manifest)
        identity = self.load()
        runtime = identity.rootfs / "usr/bin/setpriv"
        runtime.write_bytes(runtime.read_bytes() + b"changed")
        with self.assertRaisesRegex(SwesmithSandboxError, "changed after startup"):
            _attest_oci_rootfs_identity(identity)

class WorkspaceTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "src/value.py").write_text("VALUE = 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_safe_relative_git_style_symlinks(self) -> None:
        (self.root / "value.py").symlink_to("src/value.py")
        (self.root / "src/alias.py").symlink_to("../src/value.py")
        (self.root / "dangling").symlink_to("src/missing.py")
        snapshot = snapshot_workspace_tree(self.root, limits())
        self.assertEqual(snapshot.regular_file_count, 1)
        self.assertEqual(snapshot.symlink_count, 3)
        self.assertEqual(snapshot.directory_count, 1)

    def test_rejects_absolute_and_escaping_symlinks(self) -> None:
        for name, target in (("absolute", "/etc/passwd"), ("escaping", "../secret")):
            with self.subTest(target=target):
                path = self.root / name
                path.symlink_to(target)
                with self.assertRaisesRegex(
                    SwesmithSandboxError, "absolute symlink|escapes"
                ):
                    snapshot_workspace_tree(self.root, limits())
                path.unlink()

    def test_rejects_hardlinks_and_special_files(self) -> None:
        source = self.root / "src/value.py"
        hardlink = self.root / "hardlink"
        os.link(source, hardlink)
        with self.assertRaisesRegex(SwesmithSandboxError, "hard-linked"):
            snapshot_workspace_tree(self.root, limits())
        hardlink.unlink()

        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(SwesmithSandboxError, "unsupported entry"):
            snapshot_workspace_tree(self.root, limits())
        fifo.unlink()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.root / "socket"))
            with self.assertRaisesRegex(SwesmithSandboxError, "unsupported entry"):
                snapshot_workspace_tree(self.root, limits())
        finally:
            listener.close()
            (self.root / "socket").unlink(missing_ok=True)

    def test_enforces_file_tree_and_path_quotas(self) -> None:
        with self.assertRaisesRegex(SwesmithSandboxError, "per-file"):
            snapshot_workspace_tree(self.root, limits(max_file_bytes=4))
        (self.root / "extra.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(SwesmithSandboxError, "file-count"):
            snapshot_workspace_tree(self.root, limits(max_files=1))
        with self.assertRaisesRegex(SwesmithSandboxError, "character limit"):
            snapshot_workspace_tree(self.root, limits(max_path_chars=4))
        with self.assertRaisesRegex(SwesmithSandboxError, "aggregate byte"):
            snapshot_workspace_tree(
                self.root,
                limits(workspace_bytes=10, max_file_bytes=10),
            )

    def test_rejects_workspace_root_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            actual = parent / "actual"
            actual.mkdir()
            alias = parent / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(SwesmithSandboxError, "real directory"):
                snapshot_workspace_tree(alias, limits())

    def test_diff_records_added_modified_deleted_and_mode_changes(self) -> None:
        before = snapshot_workspace_tree(self.root, limits())
        (self.root / "src/value.py").write_text("VALUE = 2\n", encoding="utf-8")
        (self.root / "src/value.py").chmod(0o755)
        (self.root / "new.txt").write_text("new\n", encoding="utf-8")
        after = snapshot_workspace_tree(self.root, limits(), previous=before)
        diff = diff_workspace_trees(before, after)
        self.assertEqual(diff.changed_paths, ("new.txt", "src/value.py"))
        self.assertEqual([item["path"] for item in diff.added], ["new.txt"])
        self.assertEqual([item["path"] for item in diff.modified], ["src/value.py"])

        (self.root / "new.txt").unlink()
        final = snapshot_workspace_tree(self.root, limits(), previous=after)
        deleted = diff_workspace_trees(after, final)
        self.assertEqual([item["path"] for item in deleted.deleted], ["new.txt"])


class WorkdirTests(unittest.TestCase):
    def test_accepts_root_and_normalized_relative_paths(self) -> None:
        self.assertEqual(_normalize_workdir("."), ".")
        self.assertEqual(_normalize_workdir("src/pkg"), "src/pkg")

    def test_rejects_absolute_parent_and_noncanonical_paths(self) -> None:
        for value in ("", "/tmp", "../src", "src/../pkg", "./src", "src//pkg"):
            with self.subTest(value=value), self.assertRaises(SwesmithSandboxError):
                _normalize_workdir(value)


class EpisodeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "value.txt").write_text("before\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_run_attests_command_diff_and_releases_episode_lease(self) -> None:
        def mutate(root: Path) -> None:
            (root / "value.txt").write_text("after\n", encoding="utf-8")
            (root / "new.txt").write_text("new\n", encoding="utf-8")

        sandbox = _FakeEpisodeSandbox(mutate)
        initial = sandbox.attach_workspace(self.root)
        execution = sandbox.run(command="test", workdir=".", timeout_ms=100)
        self.assertEqual(initial.tree_sha256, execution.workspace_before.tree_sha256)
        self.assertEqual(execution.workspace_diff.changed_paths, ("new.txt", "value.txt"))
        self.assertEqual(execution.result.stdout, b"ok")
        sandbox.close()
        self.assertTrue(sandbox.lease.closed)
        with self.assertRaisesRegex(SwesmithSandboxError, "closed"):
            sandbox.run(command="test", workdir=".", timeout_ms=100)

    def test_refresh_attests_trusted_host_patch_before_next_command(self) -> None:
        sandbox = _FakeEpisodeSandbox()
        sandbox.attach_workspace(self.root)
        (self.root / "value.txt").write_text("patched\n", encoding="utf-8")
        diff = sandbox.refresh_after_host_mutation()
        self.assertEqual(diff.changed_paths, ("value.txt",))
        execution = sandbox.run(command="test", workdir=".", timeout_ms=100)
        self.assertEqual(execution.workspace_diff.changed_paths, ())
        sandbox.close()

    def test_unattested_host_mutation_poisons_episode(self) -> None:
        sandbox = _FakeEpisodeSandbox()
        sandbox.attach_workspace(self.root)
        (self.root / "value.txt").write_text("outside\n", encoding="utf-8")
        with self.assertRaisesRegex(SwesmithSandboxError, "outside the attested"):
            sandbox.run(command="test", workdir=".", timeout_ms=100)
        self.assertIn("outside the attested", sandbox.poisoned_reason or "")
        with self.assertRaisesRegex(SwesmithSandboxError, "poisoned"):
            sandbox.refresh_after_host_mutation()
        sandbox.close()

    def test_invalid_command_mutation_poisons_episode(self) -> None:
        def mutate(root: Path) -> None:
            os.mkfifo(root / "fifo")

        sandbox = _FakeEpisodeSandbox(mutate)
        sandbox.attach_workspace(self.root)
        with self.assertRaisesRegex(SwesmithSandboxError, "unsupported entry"):
            sandbox.run(command="test", workdir=".", timeout_ms=100)
        self.assertIn("validation failed", sandbox.poisoned_reason or "")
        sandbox.close()


if __name__ == "__main__":
    unittest.main()
