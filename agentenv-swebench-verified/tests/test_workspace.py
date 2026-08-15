from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from agentenv_swebench_verified.workspace import (
    VerifiedWorkspaceError,
    VerifiedWorkspaceMaterializer,
)


class VerifiedWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirrors = self.root / "mirrors"
        self.episodes = self.root / "episodes"
        self.mirrors.mkdir(mode=0o700)
        self.mirror = self.mirrors / "owner__repo"
        self.mirror.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.write("src/value.txt", "base\n")
        self.write("delete.txt", "delete me\n")
        self.write("hidden.txt", "must remain in the exact tree\n")
        self.write("src/template.txt", "$Format:%H$\n")
        self.write(
            ".gitattributes",
            "hidden.txt export-ignore\nsrc/template.txt export-subst\n",
        )
        self.git("add", ".")
        self.git("commit", "-q", "-m", "base")
        self.base_commit = self.git("rev-parse", "HEAD").strip()
        self.write("src/value.txt", "newer mirror head\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "later")
        self.row = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": self.base_commit,
            "problem_statement": "Repair the base tree",
        }
        self.materializer = VerifiedWorkspaceMaterializer(
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

    def test_materializes_exact_base_without_policy_git_metadata(self) -> None:
        workspace = self.materializer.materialize(
            self.row,
            model_uid=(1000 if os.getuid() == 0 else os.getuid()),
            model_gid=(1000 if os.getgid() == 0 else os.getgid()),
        )
        try:
            self.assertEqual(
                (workspace.policy_root / "src/value.txt").read_text(),
                "base\n",
            )
            self.assertEqual(workspace.base_commit, self.base_commit)
            self.assertEqual(
                (workspace.policy_root / "hidden.txt").read_text(),
                "must remain in the exact tree\n",
            )
            self.assertEqual(
                (workspace.policy_root / "src/template.txt").read_text(),
                "$Format:%H$\n",
            )
            self.assertFalse((workspace.policy_root / ".git").exists())
            self.assertFalse((workspace.policy_root / ".git").is_symlink())
            self.assertNotEqual(workspace.private_root, workspace.policy_root)
            self.assertNotIn(workspace.private_root, workspace.policy_root.parents)
            self.assertNotIn(workspace.git_dir, workspace.policy_root.parents)
        finally:
            self.materializer.close(workspace)
        self.assertEqual(list(self.episodes.iterdir()), [])

    def test_each_episode_gets_an_isolated_workspace(self) -> None:
        first = self.materializer.materialize(self.row)
        second = self.materializer.materialize(self.row)
        try:
            (first.policy_root / "src/value.txt").write_text("first\n")
            self.assertEqual(
                (second.policy_root / "src/value.txt").read_text(),
                "base\n",
            )
            self.assertNotEqual(first.episode_root, second.episode_root)
        finally:
            self.materializer.close(first)
            self.materializer.close(second)

    def test_materializes_safe_parent_relative_symlinks(self) -> None:
        self.write("target.txt", "safe target\n")
        link = self.mirror / "dir" / "link.txt"
        link.parent.mkdir()
        link.symlink_to("../target.txt")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "safe relative symlink")
        row = dict(self.row)
        row["base_commit"] = self.git("rev-parse", "HEAD").strip()

        workspace = self.materializer.materialize(row)
        try:
            materialized = workspace.policy_root / "dir" / "link.txt"
            self.assertTrue(materialized.is_symlink())
            self.assertEqual(os.readlink(materialized), "../target.txt")
            self.assertEqual(materialized.read_text(), "safe target\n")
        finally:
            self.materializer.close(workspace)

    def test_rejects_relative_symlinks_that_escape_the_workspace(self) -> None:
        link = self.mirror / "dir" / "escape"
        link.parent.mkdir()
        link.symlink_to("../../outside")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "escaping relative symlink")
        row = dict(self.row)
        row["base_commit"] = self.git("rev-parse", "HEAD").strip()

        with self.assertRaisesRegex(VerifiedWorkspaceError, "escapes"):
            self.materializer.materialize(row)

    def test_git_archive_warning_flood_is_bounded(self) -> None:
        self.write(
            ".gitattributes",
            "".join(f"!invalid-{index} attribute\n" for index in range(10_000)),
        )
        self.git("add", ".")
        self.git("commit", "-q", "-m", "warning-heavy attributes")
        row = dict(self.row)
        row["base_commit"] = self.git("rev-parse", "HEAD").strip()

        started = time.monotonic()
        with self.assertRaisesRegex(VerifiedWorkspaceError, "stderr"):
            self.materializer.materialize(row)
        self.assertLess(time.monotonic() - started, 10)

    def test_materialization_ignores_git_replacement_refs(self) -> None:
        later_commit = self.git("rev-parse", "HEAD").strip()
        self.git("replace", self.base_commit, later_commit)

        workspace = self.materializer.materialize(self.row)
        try:
            self.assertEqual(
                (workspace.policy_root / "src/value.txt").read_text(),
                "base\n",
            )
        finally:
            self.materializer.close(workspace)

    def test_rejects_a_missing_or_non_commit_base(self) -> None:
        bad = dict(self.row)
        bad["base_commit"] = "f" * 40
        with self.assertRaisesRegex(VerifiedWorkspaceError, "base_commit"):
            self.materializer.materialize(bad)


if __name__ == "__main__":
    unittest.main()
