from __future__ import annotations

import stat
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from agentenv_openmle_fast.dataset import OpenMLEFastDataset
from agentenv_openmle_fast.materializer import (
    LinuxTmpfsWorkspaceMountBackend,
    OpenMLEFastMaterializerError,
    OpenMLEFastWorkspaceMaterializer,
    WorkspaceMountIdentity,
)
from tests.support import (
    PRIVATE_CANARY,
    RELEASE_REVISION,
    FakeWorkspaceMountBackend,
    create_fixture,
)


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
        self.mount_backend = FakeWorkspaceMountBackend()
        self.materializer = OpenMLEFastWorkspaceMaterializer(
            Path(self.fixture["episodes_root"]),
            runner_workspace_parent=Path(self.fixture["episodes_root"]),
            workspace_bytes=2 * 1024**3,
            max_files=100_000,
            mount_backend=self.mount_backend,
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

    def test_mount_is_capped_and_adopted_mount_requires_runner_teardown(self) -> None:
        workspace = self.materializer.materialize(self.dataset[0])
        self.assertEqual(
            self.mount_backend.mount_calls,
            [(workspace.policy_root, 2 * 1024**3, 100_000)],
        )
        self.materializer.mark_adopted_by_runner(workspace)
        self.assertTrue(self.materializer.is_adopted_by_runner(workspace))
        with self.assertRaisesRegex(OpenMLEFastMaterializerError, "exact runner"):
            self.materializer.close(workspace)
        identity = self.mount_backend.inspect(workspace.policy_root)
        assert identity is not None
        self.mount_backend.unmount(
            workspace.policy_root,
            expected=identity,
            workspace_bytes=2 * 1024**3,
            max_files=100_000,
        )
        self.materializer.close(workspace)
        self.assertFalse(workspace.episode_root.exists())

    def test_rejects_episode_root_outside_attested_parent(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(OpenMLEFastMaterializerError, "attested"):
            OpenMLEFastWorkspaceMaterializer(
                outside,
                runner_workspace_parent=Path(self.fixture["episodes_root"]),
                workspace_bytes=2 * 1024**3,
                max_files=100_000,
                mount_backend=FakeWorkspaceMountBackend(),
            )

    def test_linux_backend_uses_hardened_capped_tmpfs_mount(self) -> None:
        backend = LinuxTmpfsWorkspaceMountBackend()
        mountpoint = Path(self.fixture["episodes_root"]) / "mountpoint"
        mountpoint.mkdir()
        identity = WorkspaceMountIdentity(
            mount_id="42",
            device=99,
            filesystem="tmpfs",
            options=("nodev", "noexec", "nosuid", "rw"),
            capacity_bytes=2 * 1024**3,
            inode_capacity=100_000,
        )
        with (
            patch(
                "agentenv_openmle_fast.materializer.Path.is_file",
                return_value=True,
            ),
            patch.object(backend, "inspect", side_effect=[None, identity]),
            patch("agentenv_openmle_fast.materializer.subprocess.run") as run,
        ):
            result = backend.mount(
                mountpoint, workspace_bytes=2 * 1024**3, max_files=100_000
            )
        self.assertEqual(result, identity)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["/usr/bin/mount", "-t", "tmpfs", "-o"])
        options = command[4]
        for required in (
            "size=2147483648",
            "nr_inodes=100000",
            "mode=0777",
            "noexec",
            "nosuid",
            "nodev",
        ):
            self.assertIn(required, options)

    def test_linux_backend_rolls_back_a_mount_that_fails_admission(self) -> None:
        backend = LinuxTmpfsWorkspaceMountBackend()
        mountpoint = Path(self.fixture["episodes_root"]) / "invalid-mountpoint"
        mountpoint.mkdir()
        invalid = WorkspaceMountIdentity(
            mount_id="43",
            device=100,
            filesystem="tmpfs",
            options=("nodev", "noexec", "nosuid", "rw"),
            capacity_bytes=3 * 1024**3,
            inode_capacity=100_000,
        )
        with (
            patch(
                "agentenv_openmle_fast.materializer.Path.is_file",
                return_value=True,
            ),
            patch.object(backend, "inspect", side_effect=[None, invalid, None]),
            patch("agentenv_openmle_fast.materializer.subprocess.run") as run,
            self.assertRaisesRegex(OpenMLEFastMaterializerError, "frozen cap"),
        ):
            backend.mount(
                mountpoint, workspace_bytes=2 * 1024**3, max_files=100_000
            )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][0], "/usr/bin/mount")
        self.assertEqual(
            run.call_args_list[1].args[0], ["/usr/bin/umount", str(mountpoint)]
        )

    def test_marker_write_failure_rolls_back_the_unrecorded_mount(self) -> None:
        with (
            patch(
                "agentenv_openmle_fast.materializer._replace_marker",
                side_effect=OpenMLEFastMaterializerError("marker write failed"),
            ),
            self.assertRaisesRegex(OpenMLEFastMaterializerError, "marker write failed"),
        ):
            self.materializer.materialize(self.dataset[0])
        self.assertEqual(len(self.mount_backend.unmount_calls), 1)
        self.assertEqual(self.mount_backend.mounts, {})
        self.assertEqual(tuple(Path(self.fixture["episodes_root"]).iterdir()), ())

    def test_cleanup_rejects_marker_caps_from_a_different_materializer(self) -> None:
        workspace = self.materializer.materialize(self.dataset[0])
        different_caps = OpenMLEFastWorkspaceMaterializer(
            Path(self.fixture["episodes_root"]),
            runner_workspace_parent=Path(self.fixture["episodes_root"]),
            workspace_bytes=1024**3,
            max_files=50_000,
            mount_backend=self.mount_backend,
        )
        with self.assertRaisesRegex(OpenMLEFastMaterializerError, "caps differ"):
            different_caps.close(workspace)
        self.materializer.close(workspace)


if __name__ == "__main__":
    unittest.main()
