from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from agentenv_openmle_fast.dataset import OpenMLEFastDataset
from agentenv_openmle_fast.materializer import OpenMLEFastWorkspaceMaterializer
from tests.support import PRIVATE_CANARY, RELEASE_REVISION, create_fixture


class OpenMLEFastMaterializerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="openmle-materializer-test-"
        )
        self.root = Path(self.temporary.name)
        self.fixture = create_fixture(self.root)
        self.dataset = OpenMLEFastDataset(
            manifest_path=Path(self.fixture["manifest"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_manifest_sha256=str(self.fixture["manifest_sha256"]),
            expected_release_revision=RELEASE_REVISION,
            expected_role="gate_only",
        )
        self.materializer = OpenMLEFastWorkspaceMaterializer(
            Path(self.fixture["episodes_root"])
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_materializes_only_task_and_public_data(self) -> None:
        workspace = self.materializer.materialize(self.dataset[0])
        relative_files = sorted(
            str(path.relative_to(workspace.policy_root))
            for path in workspace.policy_root.rglob("*")
            if path.is_file()
        )
        self.assertEqual(
            relative_files,
            [
                "TASK.md",
                "data/description.txt",
                "data/sample_submission.csv",
                "data/test.csv",
                "data/train.csv",
            ],
        )
        all_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in workspace.policy_root.rglob("*")
            if path.is_file()
        )
        self.assertNotIn(PRIVATE_CANARY, all_text)
        self.assertEqual(
            workspace.public_tree_sha256, self.dataset[0].public_tree_sha256
        )
        for path in workspace.policy_root.rglob("*"):
            if path.is_file():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        self.materializer.close(workspace)
        self.assertFalse(workspace.episode_root.exists())

    def test_episode_roots_are_unique_and_isolated(self) -> None:
        first = self.materializer.materialize(self.dataset[0])
        second = self.materializer.materialize(self.dataset[0])
        self.assertNotEqual(first.episode_root, second.episode_root)
        (first.policy_root / "notes.txt").write_text("first", encoding="utf-8")
        self.assertFalse((second.policy_root / "notes.txt").exists())
        self.materializer.close(first)
        self.materializer.close(second)


if __name__ == "__main__":
    unittest.main()
