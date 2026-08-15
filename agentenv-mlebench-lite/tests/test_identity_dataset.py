from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agentenv_mlebench_lite.dataset import (
    MLEBenchLiteDatasetError,
    load_lite_dataset,
)
from agentenv_mlebench_lite.identity import (
    LITE_COMPETITION_IDS,
    SPLIT_SHA256,
    UPSTREAM_COMMIT,
    MLEBenchLiteIdentityError,
    load_official_lite_identity,
)

from tests.support import sha256_bytes, write_fixture


class MLEBenchLiteIdentityDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mlebench-lite-id-")
        self.root = Path(self.temporary.name)
        self.fixture = write_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def load_identity(self, commit: str = UPSTREAM_COMMIT):
        return load_official_lite_identity(
            self.fixture["upstream_root"],
            commit_resolver=lambda _root: commit,
        )

    def load_dataset(self):
        return load_lite_dataset(
            identity=self.load_identity(),
            manifest_path=self.fixture["manifest_path"],
            expected_manifest_sha256=self.fixture["manifest_sha256"],
            data_root=self.fixture["data_root"],
        )

    def test_exact_commit_hash_count_and_order_are_loaded_externally(self) -> None:
        identity = self.load_identity()
        self.assertEqual(identity.upstream_commit, UPSTREAM_COMMIT)
        self.assertEqual(identity.split_sha256, SPLIT_SHA256)
        self.assertEqual(identity.competition_ids, LITE_COMPETITION_IDS)
        self.assertEqual(len(identity.competition_ids), 22)
        self.assertEqual(len(set(identity.competition_ids)), 22)
        split_path = (
            self.fixture["upstream_root"] / "experiments" / "splits" / "low.txt"
        )
        self.assertFalse(split_path.read_bytes().endswith(b"\n"))

    def test_commit_hash_and_order_gates_fail_closed(self) -> None:
        with self.assertRaises(MLEBenchLiteIdentityError):
            self.load_identity("0" * 40)

        split_path = (
            self.fixture["upstream_root"] / "experiments" / "splits" / "low.txt"
        )
        ordered = split_path.read_text(encoding="utf-8").splitlines()
        ordered[0], ordered[1] = ordered[1], ordered[0]
        split_path.write_text("\n".join(ordered), encoding="utf-8")
        with self.assertRaises(MLEBenchLiteIdentityError):
            self.load_identity()

    def test_21_23_and_duplicate_membership_fail_closed(self) -> None:
        variants = {
            "21": list(LITE_COMPETITION_IDS[:-1]),
            "23": [*LITE_COMPETITION_IDS, "unexpected-competition"],
            "duplicate": [*LITE_COMPETITION_IDS[:-1], LITE_COMPETITION_IDS[0]],
        }
        for name, ids in variants.items():
            case_root = self.root / name
            fixture = write_fixture(case_root)
            split_path = fixture["upstream_root"] / "experiments" / "splits" / "low.txt"
            split_path.write_bytes("\n".join(ids).encode("utf-8"))
            with self.subTest(name=name), self.assertRaises(MLEBenchLiteIdentityError):
                load_official_lite_identity(
                    fixture["upstream_root"],
                    commit_resolver=lambda _root: UPSTREAM_COMMIT,
                )

    def test_manifest_is_hash_bound_and_returns_no_private_paths(self) -> None:
        dataset = self.load_dataset()
        self.assertEqual(
            tuple(record.competition_id for record in dataset), LITE_COMPETITION_IDS
        )
        self.assertFalse(hasattr(dataset[0], "private_root"))
        self.assertNotIn("/prepared/private", repr(dataset[0]).lower())

        payload = json.loads(self.fixture["manifest_path"].read_text(encoding="utf-8"))
        payload["tasks"][0]["public_tree_sha256"] = "0" * 64
        changed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.fixture["manifest_path"].write_bytes(changed)
        with self.assertRaises(MLEBenchLiteDatasetError):
            load_lite_dataset(
                identity=self.load_identity(),
                manifest_path=self.fixture["manifest_path"],
                expected_manifest_sha256=sha256_bytes(b"different expected manifest"),
                data_root=self.fixture["data_root"],
            )

    def test_official_public_private_siblings_and_symlinks_are_rejected(self) -> None:
        identity = self.load_identity()
        common = {
            "identity": identity,
            "manifest_path": self.fixture["manifest_path"],
            "expected_manifest_sha256": self.fixture["manifest_sha256"],
            "data_root": self.fixture["data_root"],
        }
        public_file = (
            self.fixture["data_root"]
            / LITE_COMPETITION_IDS[0]
            / "prepared"
            / "public"
            / "train.csv"
        )
        public_file.unlink()
        public_file.symlink_to(
            self.fixture["data_root"]
            / LITE_COMPETITION_IDS[0]
            / "prepared"
            / "private"
            / "answer.csv"
        )
        with self.assertRaises(MLEBenchLiteDatasetError):
            load_lite_dataset(**common)

    def test_task_crossing_manifest_path_is_rejected(self) -> None:
        payload = json.loads(self.fixture["manifest_path"].read_text(encoding="utf-8"))
        payload["tasks"][0]["public_relative_path"] = (
            f"{LITE_COMPETITION_IDS[1]}/prepared/public"
        )
        changed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.fixture["manifest_path"].write_bytes(changed)
        with self.assertRaises(MLEBenchLiteDatasetError):
            load_lite_dataset(
                identity=self.load_identity(),
                manifest_path=self.fixture["manifest_path"],
                expected_manifest_sha256=sha256_bytes(changed),
                data_root=self.fixture["data_root"],
            )

    def test_public_file_mutation_breaks_pinned_inventory(self) -> None:
        public_file = (
            self.fixture["data_root"]
            / LITE_COMPETITION_IDS[0]
            / "prepared"
            / "public"
            / "train.csv"
        )
        public_file.write_text("mutated public bytes\n", encoding="utf-8")
        with self.assertRaises(MLEBenchLiteDatasetError):
            self.load_dataset()

    def test_task_source_directory_symlink_is_rejected(self) -> None:
        task_root = self.fixture["data_root"] / LITE_COMPETITION_IDS[0]
        moved = self.root / "moved-task"
        task_root.rename(moved)
        task_root.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(MLEBenchLiteDatasetError):
            self.load_dataset()

    def test_public_hardlink_to_private_source_is_rejected(self) -> None:
        public_file = (
            self.fixture["data_root"]
            / LITE_COMPETITION_IDS[0]
            / "prepared"
            / "public"
            / "train.csv"
        )
        private_file = (
            self.fixture["data_root"]
            / LITE_COMPETITION_IDS[0]
            / "prepared"
            / "private"
            / "answer.csv"
        )
        public_file.unlink()
        os.link(private_file, public_file)
        with self.assertRaises(MLEBenchLiteDatasetError):
            self.load_dataset()

    def test_manifest_duplicate_keys_and_nonfinite_numbers_are_rejected(self) -> None:
        original = json.loads(self.fixture["manifest_path"].read_text(encoding="utf-8"))
        canonical = json.dumps(original, separators=(",", ":"))
        duplicate = (
            '{"schema":"mlebench_lite_public_manifest_v1",' + canonical[1:]
        ).encode("utf-8")
        nonfinite_value = json.loads(json.dumps(original))
        nonfinite_value["tasks"][0]["public_files"][0]["size"] = float("nan")
        nonfinite = json.dumps(nonfinite_value, allow_nan=True).encode("utf-8")
        for name, payload in (("duplicate", duplicate), ("nonfinite", nonfinite)):
            self.fixture["manifest_path"].write_bytes(payload)
            with self.subTest(name=name), self.assertRaises(MLEBenchLiteDatasetError):
                load_lite_dataset(
                    identity=self.load_identity(),
                    manifest_path=self.fixture["manifest_path"],
                    expected_manifest_sha256=sha256_bytes(payload),
                    data_root=self.fixture["data_root"],
                )

    def test_public_private_overlap_or_nesting_in_manifest_is_rejected(self) -> None:
        payload = json.loads(self.fixture["manifest_path"].read_text(encoding="utf-8"))
        public_relative = payload["tasks"][0]["public_relative_path"]
        for private_relative in (public_relative, f"{public_relative}/nested"):
            changed_payload = json.loads(json.dumps(payload))
            changed_payload["tasks"][0]["private_relative_path"] = private_relative
            changed = json.dumps(
                changed_payload, sort_keys=True, separators=(",", ":")
            ).encode()
            self.fixture["manifest_path"].write_bytes(changed)
            with (
                self.subTest(private_relative=private_relative),
                self.assertRaises(MLEBenchLiteDatasetError),
            ):
                load_lite_dataset(
                    identity=self.load_identity(),
                    manifest_path=self.fixture["manifest_path"],
                    expected_manifest_sha256=sha256_bytes(changed),
                    data_root=self.fixture["data_root"],
                )


if __name__ == "__main__":
    unittest.main()
