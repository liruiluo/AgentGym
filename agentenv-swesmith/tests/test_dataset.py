from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentenv_swesmith.dataset import SwesmithDataset, SwesmithDatasetError


REVISION = "e" * 40


def instance(name: str, problem: str) -> dict:
    return {
        "instance_id": name,
        "repo": "swesmith/owner__repo.12345678",
        "problem_statement": problem,
        "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
        "patch": "private gold material",
    }


class SwesmithDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(
        self,
        rows: list[dict],
        *,
        selection: dict | None = None,
        role: str = "plumbing",
    ) -> Path:
        shard = self.root / "train.jsonl"
        shard.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        usable = sum(bool(row["problem_statement"].strip()) for row in rows)
        manifest = {
            "schema_version": "swesmith_jsonl_manifest_v1",
            "dataset_id": "unit",
            "upstream": {
                "repository": "SWE-bench/SWE-smith",
                "revision": REVISION,
            },
            "role": role,
            "selection": selection or {"mode": "all_usable"},
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "physical_rows": len(rows),
                    "usable_rows": usable,
                }
            ],
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_filters_blank_rows_and_preserves_explicit_index_provenance(self) -> None:
        manifest = self.write_manifest(
            [
                instance("owner__repo.12345678.task_a", "fix A"),
                instance("owner__repo.12345678.task_blank", " \n"),
                instance("owner__repo.12345678.task_b", "fix B"),
            ]
        )
        dataset = SwesmithDataset(manifest)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.provenance.physical_rows, 3)
        self.assertEqual(dataset.provenance.blank_rows, 1)
        self.assertEqual(dataset[0].instance_id, "owner__repo.12345678.task_a")
        self.assertEqual(dataset[1].physical_index, 2)
        self.assertEqual(dataset[1].shard_line, 3)
        self.assertEqual(dataset[1].problem_statement, "fix B")

    def test_base_repository_uses_repo_revision_suffix_not_first_dot(self) -> None:
        row = instance("owner__repo.with.dot.12345678.task_a", "fix A")
        row["repo"] = "swesmith/owner__repo.with.dot.12345678"
        dataset = SwesmithDataset(self.write_manifest([row]))
        self.assertEqual(dataset[0].base_repository, "owner__repo.with.dot")

    def test_selection_is_frozen_by_file_hash_and_count(self) -> None:
        rows = [
            instance("owner__repo.12345678.task_a", "fix A"),
            instance("owner__repo.12345678.task_b", "fix B"),
        ]
        selection_path = self.root / "heldout.json"
        selection_path.write_text(json.dumps([rows[1]["instance_id"]]), encoding="utf-8")
        selection = {
            "mode": "instance_ids",
            "path": selection_path.name,
            "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "count": 1,
        }
        dataset = SwesmithDataset(self.write_manifest(rows, selection=selection))
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].instance_id, rows[1]["instance_id"])
        self.assertEqual(dataset.provenance.selection_mode, "instance_ids")

    def test_formal_heldout_role_is_accepted(self) -> None:
        dataset = SwesmithDataset(
            self.write_manifest(
                [instance("owner__repo.12345678.task_a", "fix")],
                role="formal_heldout",
            )
        )
        self.assertEqual(dataset.provenance.role, "formal_heldout")

    def test_opaque_instance_identity_is_never_accepted_as_data_idx(self) -> None:
        dataset = SwesmithDataset(
            self.write_manifest([instance("owner__repo.12345678.task_7", "fix")])
        )
        for invalid in (True, -1, "0", "task_7"):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, IndexError)):
                    dataset[invalid]

    def test_tampered_shard_and_duplicate_identity_fail_closed(self) -> None:
        manifest = self.write_manifest(
            [instance("owner__repo.12345678.task_a", "fix")]
        )
        (self.root / "train.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(SwesmithDatasetError, "field 'instance_id'"):
            SwesmithDataset(manifest)

        duplicate_manifest = self.write_manifest(
            [
                instance("owner__repo.12345678.task_a", "fix A"),
                instance("owner__repo.12345678.task_a", "fix B"),
            ]
        )
        with self.assertRaisesRegex(SwesmithDatasetError, "duplicate instance_id"):
            SwesmithDataset(duplicate_manifest)

    def test_upstream_revision_requires_a_full_git_commit(self) -> None:
        manifest = self.write_manifest(
            [instance("owner__repo.12345678.task_a", "fix")]
        )
        contents = json.loads(manifest.read_text(encoding="utf-8"))
        for invalid in ("main", "e" * 39, "e" * 64, "g" * 40):
            with self.subTest(invalid=invalid):
                contents["upstream"]["revision"] = invalid
                manifest.write_text(json.dumps(contents), encoding="utf-8")
                with self.assertRaisesRegex(SwesmithDatasetError, "Git commit"):
                    SwesmithDataset(manifest)


if __name__ == "__main__":
    unittest.main()
