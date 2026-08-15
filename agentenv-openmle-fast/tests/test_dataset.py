from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentenv_openmle_fast.dataset import (
    OpenMLEFastDataset,
    OpenMLEFastDatasetError,
)
from tests.support import RELEASE_REVISION, TASK_ID, create_fixture, sha256_file


class OpenMLEFastDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="openmle-dataset-test-")
        self.root = Path(self.temporary.name)
        self.fixture = create_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def load(self) -> OpenMLEFastDataset:
        return OpenMLEFastDataset(
            manifest_path=Path(self.fixture["manifest"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_manifest_sha256=str(self.fixture["manifest_sha256"]),
            expected_release_revision=RELEASE_REVISION,
            expected_role="gate_only",
        )

    def test_loads_exact_index_and_public_identity(self) -> None:
        dataset = self.load()
        self.assertEqual(len(dataset), 1)
        record = dataset[0]
        self.assertEqual(record.data_idx, 0)
        self.assertEqual(record.task_id, TASK_ID)
        self.assertEqual(record.public_task["metric_direction"], "lower")
        self.assertEqual(
            dataset.provenance.manifest_sha256, self.fixture["manifest_sha256"]
        )
        self.assertNotIn("private", json.dumps(record.public_metadata()))

    def test_rejects_bool_or_noncontiguous_data_idx(self) -> None:
        manifest_path = Path(self.fixture["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for invalid in (True, 2):
            manifest["records"][0]["data_idx"] = invalid
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(OpenMLEFastDatasetError),
            ):
                OpenMLEFastDataset(
                    manifest_path=manifest_path,
                    package_root=Path(self.fixture["package_root"]),
                    archive_root=Path(self.fixture["archive_root"]),
                    expected_manifest_sha256=sha256_file(manifest_path),
                    expected_release_revision=RELEASE_REVISION,
                    expected_role="gate_only",
                )

    def test_rejects_hash_release_role_and_public_tree_drift(self) -> None:
        with self.assertRaises(OpenMLEFastDatasetError):
            OpenMLEFastDataset(
                manifest_path=Path(self.fixture["manifest"]),
                package_root=Path(self.fixture["package_root"]),
                archive_root=Path(self.fixture["archive_root"]),
                expected_manifest_sha256="0" * 64,
                expected_release_revision=RELEASE_REVISION,
                expected_role="gate_only",
            )
        for keyword, value in (
            ("expected_release_revision", "1" * 40),
            ("expected_role", "train_pool"),
        ):
            arguments = {
                "manifest_path": Path(self.fixture["manifest"]),
                "package_root": Path(self.fixture["package_root"]),
                "archive_root": Path(self.fixture["archive_root"]),
                "expected_manifest_sha256": str(self.fixture["manifest_sha256"]),
                "expected_release_revision": RELEASE_REVISION,
                "expected_role": "gate_only",
            }
            arguments[keyword] = value
            with (
                self.subTest(keyword=keyword),
                self.assertRaises(OpenMLEFastDatasetError),
            ):
                OpenMLEFastDataset(**arguments)
        public_file = Path(self.fixture["package"]) / "data/train.csv"
        public_file.chmod(0o644)
        public_file.write_text("drift\n", encoding="utf-8")
        with self.assertRaises(OpenMLEFastDatasetError):
            self.load()

    def test_rejects_symlinked_public_input(self) -> None:
        public_file = Path(self.fixture["package"]) / "data/test.csv"
        target = self.root / "outside.csv"
        target.write_text("private\n", encoding="utf-8")
        public_file.unlink()
        public_file.symlink_to(target)
        with self.assertRaises(OpenMLEFastDatasetError):
            self.load()

    def test_rejects_duplicate_manifest_keys(self) -> None:
        manifest_path = Path(self.fixture["manifest"])
        original = manifest_path.read_text(encoding="utf-8").lstrip("{")
        manifest_path.write_text(
            '{"schema":"openmle_fast_public_manifest_v1",' + original,
            encoding="utf-8",
        )
        with self.assertRaises(OpenMLEFastDatasetError):
            OpenMLEFastDataset(
                manifest_path=manifest_path,
                package_root=Path(self.fixture["package_root"]),
                archive_root=Path(self.fixture["archive_root"]),
                expected_manifest_sha256=sha256_file(manifest_path),
                expected_release_revision=RELEASE_REVISION,
                expected_role="gate_only",
            )


if __name__ == "__main__":
    unittest.main()
