from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentenv_swesmith.workspace import (
    SwesmithWorkspaceError,
    SwesmithWorkspaceMaterializer,
    _remove_episode_tree,
    restore_hidden_tests,
)


class SwesmithWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirrors = self.root / "mirrors"
        self.episodes = self.root / "episodes"
        self.mirrors.mkdir()
        self.mirror = self.mirrors / "owner__repo.12345678"
        self.mirror.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.write("src/value.py", "VALUE = 'good'\n")
        self.write("tests/test_fix.py", "def test_fix(): pass\n")
        self.write("tests/test_keep.py", "def test_keep(): pass\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "initial")
        self.write("src/value.py", "VALUE = 'bug'\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "bug")
        self.pristine_commit = self.git("rev-parse", "HEAD").strip()
        (self.mirror / "tests/test_fix.py").unlink()
        self.git("add", "-u")
        self.git("commit", "-q", "-m", "Remove F2P Tests")
        self.instance_id = "owner__repo.12345678.combine_file__opaque"
        self.git("branch", self.instance_id)
        self.bug_commit = self.git("rev-parse", "HEAD").strip()
        self.git("checkout", "-q", "main")
        self.instance = {
            "instance_id": self.instance_id,
            "repo": "swesmith/owner__repo.12345678",
        }
        self.materializer = SwesmithWorkspaceMaterializer(
            mirrors_root=self.mirrors,
            episodes_root=self.episodes,
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

    def test_policy_uses_branch_head_while_tests_are_private_from_parent(self) -> None:
        before_head = self.git("rev-parse", "HEAD").strip()
        before_status = self.git("status", "--porcelain")
        workspace = self.materializer.materialize(
            self.instance,
            test_paths=["tests/test_fix.py", "tests/test_keep.py"],
        )
        try:
            self.assertEqual(workspace.bug_commit, self.bug_commit)
            self.assertEqual(workspace.pristine_commit, self.pristine_commit)
            self.assertEqual(
                (workspace.policy_root / "src/value.py").read_text(),
                "VALUE = 'bug'\n",
            )
            self.assertFalse((workspace.policy_root / "tests/test_fix.py").exists())
            self.assertTrue((workspace.policy_root / "tests/test_keep.py").exists())
            self.assertFalse((workspace.policy_root / ".git").exists())
            checkpoint_parent = workspace.policy_root / ".agent_memory"
            self.assertTrue(checkpoint_parent.is_dir())
            self.assertFalse(checkpoint_parent.is_symlink())
            self.assertEqual(list(checkpoint_parent.iterdir()), [])
            self.assertTrue((workspace.hidden_tests_root / "tests/test_fix.py").is_file())

            (workspace.policy_root / "tests/test_keep.py").write_text(
                "tampered\n", encoding="utf-8"
            )
            restored = restore_hidden_tests(workspace)
            self.assertEqual(
                restored,
                ("tests/test_fix.py", "tests/test_keep.py"),
            )
            self.assertEqual(
                (workspace.policy_root / "tests/test_keep.py").read_text(),
                "def test_keep(): pass\n",
            )
        finally:
            self.materializer.close(workspace)
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), before_head)
        self.assertEqual(self.git("status", "--porcelain"), before_status)

    def test_missing_or_escaping_test_paths_fail_closed_and_clean_episode(self) -> None:
        for test_paths in (["../secret"], ["tests/missing.py"]):
            with self.subTest(test_paths=test_paths):
                with self.assertRaises(SwesmithWorkspaceError):
                    self.materializer.materialize(
                        self.instance,
                        test_paths=test_paths,
                    )
                self.assertEqual(list(self.episodes.iterdir()), [])

    def test_materializer_accepts_safe_parent_segments_in_git_symlinks(self) -> None:
        self.write("docs/creating-a-site.rst", "safe target\n")
        link = self.mirror / "docs/sphinx/creating-a-site.rst"
        link.parent.mkdir(parents=True)
        link.symlink_to("../creating-a-site.rst")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "add safe relative symlink")
        instance_id = "owner__repo.12345678.safe_symlink__opaque"
        self.git("branch", instance_id)

        workspace = self.materializer.materialize(
            {"instance_id": instance_id, "repo": self.instance["repo"]},
            test_paths=["tests/test_keep.py"],
        )
        try:
            exported = workspace.policy_root / "docs/sphinx/creating-a-site.rst"
            self.assertTrue(exported.is_symlink())
            self.assertEqual(os.readlink(exported), "../creating-a-site.rst")
            self.assertEqual(exported.read_text(encoding="utf-8"), "safe target\n")
        finally:
            self.materializer.close(workspace)

    def test_materializer_rejects_git_symlink_that_escapes_workspace(self) -> None:
        link = self.mirror / "escape"
        link.symlink_to("../outside")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "add escaping symlink")
        instance_id = "owner__repo.12345678.escape_symlink__opaque"
        self.git("branch", instance_id)

        with self.assertRaisesRegex(SwesmithWorkspaceError, "symlink target"):
            self.materializer.materialize(
                {"instance_id": instance_id, "repo": self.instance["repo"]},
                test_paths=["tests/test_fix.py", "tests/test_keep.py"],
            )
        self.assertEqual(list(self.episodes.iterdir()), [])

    def test_materializer_retries_one_failed_archive_from_an_empty_workspace(self) -> None:
        shim_dir, count_path = self._archive_failure_shim("once")
        environment = {
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "SWESMITH_REAL_GIT": str(shutil.which("git")),
            "SWESMITH_ARCHIVE_FAILURE_MODE": "once",
            "SWESMITH_ARCHIVE_ATTEMPT_FILE": str(count_path),
        }
        with mock.patch.dict(os.environ, environment), self.assertLogs(
            "agentenv_swesmith.workspace", level="WARNING"
        ) as captured:
            workspace = self.materializer.materialize(
                self.instance,
                test_paths=["tests/test_fix.py", "tests/test_keep.py"],
            )
        try:
            self.assertEqual(count_path.read_text(encoding="ascii"), "2")
            self.assertFalse(
                (workspace.policy_root / ".partial-from-first-attempt").exists()
            )
            self.assertEqual(
                (workspace.policy_root / "src/value.py").read_text(encoding="utf-8"),
                "VALUE = 'bug'\n",
            )
            warning = "\n".join(captured.output)
            self.assertIn("attempt=1/2", warning)
            self.assertIn("returncode=73", warning)
            self.assertIn("stderr='<empty>'", warning)
            self.assertIn(self.instance_id, warning)
        finally:
            self.materializer.close(workspace)

    def test_materializer_fails_closed_after_one_archive_retry(self) -> None:
        shim_dir, count_path = self._archive_failure_shim("always")
        environment = {
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "SWESMITH_REAL_GIT": str(shutil.which("git")),
            "SWESMITH_ARCHIVE_FAILURE_MODE": "always",
            "SWESMITH_ARCHIVE_ATTEMPT_FILE": str(count_path),
        }
        with mock.patch.dict(os.environ, environment), self.assertRaisesRegex(
            SwesmithWorkspaceError,
            rf"attempt=2/2.*returncode=74.*{self.instance_id}.*synthetic archive failure",
        ):
            self.materializer.materialize(
                self.instance,
                test_paths=["tests/test_fix.py", "tests/test_keep.py"],
            )
        self.assertEqual(count_path.read_text(encoding="ascii"), "2")
        self.assertEqual(list(self.episodes.iterdir()), [])

    def _archive_failure_shim(self, mode: str) -> tuple[Path, Path]:
        shim_dir = self.root / f"git-archive-shim-{mode}"
        shim_dir.mkdir()
        count_path = shim_dir / "archive-attempts"
        shim = shim_dir / "git"
        shim.write_text(
            r'''#!/usr/bin/env python3
import io
import os
import sys
import tarfile

arguments = sys.argv[1:]
if "archive" in arguments:
    count_path = os.environ["SWESMITH_ARCHIVE_ATTEMPT_FILE"]
    try:
        with open(count_path, "r", encoding="ascii") as handle:
            attempt = int(handle.read()) + 1
    except FileNotFoundError:
        attempt = 1
    with open(count_path, "w", encoding="ascii") as handle:
        handle.write(str(attempt))
    mode = os.environ["SWESMITH_ARCHIVE_FAILURE_MODE"]
    if mode == "always":
        sys.stderr.write("synthetic archive failure\n")
        raise SystemExit(74)
    if mode == "once" and attempt == 1:
        payload = b"must not survive retry\n"
        member = tarfile.TarInfo(".partial-from-first-attempt")
        member.size = len(payload)
        member.mode = 0o600
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            archive.addfile(member, io.BytesIO(payload))
        raise SystemExit(73)
real_git = os.environ["SWESMITH_REAL_GIT"]
os.execv(real_git, [real_git, *arguments])
''',
            encoding="utf-8",
        )
        shim.chmod(0o755)
        return shim_dir, count_path

    def test_materializer_scopes_every_git_call_to_the_exact_mirror(self) -> None:
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        shim_dir = self.root / "git-shim"
        shim_dir.mkdir()
        audit_path = self.root / "git-invocations.jsonl"
        shim = shim_dir / "git"
        shim.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["SWESMITH_GIT_AUDIT"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
real_git = os.environ["SWESMITH_REAL_GIT"]
os.execv(real_git, [real_git, *sys.argv[1:]])
""",
            encoding="utf-8",
        )
        shim.chmod(0o755)

        environment = {
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "SWESMITH_GIT_AUDIT": str(audit_path),
            "SWESMITH_REAL_GIT": str(real_git),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(self.root / "unrelated-repository"),
        }
        with mock.patch.dict(os.environ, environment):
            workspace = self.materializer.materialize(
                self.instance,
                test_paths=["tests/test_fix.py", "tests/test_keep.py"],
            )
        self.materializer.close(workspace)

        invocations = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
        ]
        expected_prefix = [
            "-c",
            f"safe.directory={self.mirror.resolve()}",
            "-C",
            str(self.mirror.resolve()),
        ]
        self.assertGreaterEqual(len(invocations), 8)
        self.assertTrue(
            all(arguments[:4] == expected_prefix for arguments in invocations),
            invocations,
        )


    def test_close_retries_transient_nonempty_directory_race(self) -> None:
        workspace = self.materializer.materialize(
            self.instance,
            test_paths=["tests/test_fix.py", "tests/test_keep.py"],
        )
        actual_rmtree = shutil.rmtree
        attempts = 0

        def transient_rmtree(path: Path, *args: object, **kwargs: object) -> None:
            nonlocal attempts
            if Path(path).resolve() == workspace.episode_root.resolve():
                attempts += 1
                if attempts == 1:
                    raise OSError(errno.ENOTEMPTY, "Directory not empty", path)
            actual_rmtree(path, *args, **kwargs)

        with mock.patch(
            "agentenv_swesmith.workspace.shutil.rmtree",
            side_effect=transient_rmtree,
        ):
            self.materializer.close(workspace)

        self.assertEqual(attempts, 2)
        self.assertFalse(workspace.episode_root.exists())

    def test_persistent_episode_cleanup_failure_stays_fail_closed(self) -> None:
        path = Path(tempfile.mkdtemp(prefix="swesmith-episode-cleanup-test-"))
        (path / ".nfs-placeholder").write_text("held", encoding="ascii")
        try:
            with mock.patch(
                "agentenv_swesmith.workspace.shutil.rmtree",
                side_effect=OSError(errno.ENOTEMPTY, "Directory not empty", path),
            ), self.assertRaisesRegex(
                SwesmithWorkspaceError,
                r"stayed busy.*\.nfs-placeholder",
            ):
                _remove_episode_tree(path, timeout_seconds=0.0)
        finally:
            shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
