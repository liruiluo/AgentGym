from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
