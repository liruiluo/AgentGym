from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentenv_swebench_verified.dataset import (
    DATASET_MANIFEST_SCHEMA,
    VerifiedDataset,
    VerifiedDatasetError,
)
from agentenv_swebench_verified.protocol import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    POLICY_FIELDS,
    PRODUCTION_DATASET_PINS,
    FrozenDatasetPins,
)


def canonical_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def fixture_row(index: int) -> dict[str, object]:
    instance_id = f"owner__repo-{index:04d}"
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "base_commit": f"{index + 1:040x}",
        "problem_statement": f"Repair issue {index}",
        "version": "1.0",
        "patch": f"SECRET_GOLD_{index}",
        "test_patch": f"SECRET_TEST_{index}",
        "FAIL_TO_PASS": json.dumps([f"tests/test_fix.py::test_{index}"]),
        "PASS_TO_PASS": json.dumps(["tests/test_keep.py::test_keep"]),
        "hints_text": f"SECRET_HINT_{index}",
        "eval_script": f"SECRET_EVAL_{index}",
        "log_parser": f"SECRET_PARSER_{index}",
        "grader_logs": f"SECRET_LOG_{index}",
    }


class DatasetFixture:
    def __init__(self, root: Path, rows: list[dict[str, object]]) -> None:
        self.root = root
        self.rows = rows
        payload = canonical_bytes(rows)
        ids = [str(row["instance_id"]) for row in rows]
        ledger = "".join(f"{instance_id}\n" for instance_id in sorted(ids)).encode()
        self.pins = FrozenDatasetPins(
            repository=DATASET_REPOSITORY,
            revision=DATASET_REVISION,
            split="test",
            row_count=len(rows),
            canonical_jsonl_sha256=hashlib.sha256(payload).hexdigest(),
            id_ledger_sha256=hashlib.sha256(ledger).hexdigest(),
        )
        self.jsonl = root / "verified.jsonl"
        self.jsonl.write_bytes(payload)
        manifest = {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "dataset": {
                "repository": self.pins.repository,
                "revision": self.pins.revision,
                "split": self.pins.split,
            },
            "canonical_jsonl": {
                "path": self.jsonl.name,
                "sha256": self.pins.canonical_jsonl_sha256,
                "rows": self.pins.row_count,
                "id_ledger_sha256": self.pins.id_ledger_sha256,
            },
        }
        self.manifest = root / "manifest.json"
        self.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


class VerifiedDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_production_pins_match_the_audit(self) -> None:
        self.assertEqual(PRODUCTION_DATASET_PINS.repository, DATASET_REPOSITORY)
        self.assertEqual(
            PRODUCTION_DATASET_PINS.revision,
            "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        )
        self.assertEqual(PRODUCTION_DATASET_PINS.row_count, 500)
        self.assertEqual(
            PRODUCTION_DATASET_PINS.canonical_jsonl_sha256,
            "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb",
        )
        self.assertEqual(
            PRODUCTION_DATASET_PINS.id_ledger_sha256,
            "a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9",
        )

    def test_loads_all_500_rows_and_exposes_only_the_safe_projection(self) -> None:
        fixture = DatasetFixture(
            self.root,
            [fixture_row(index) for index in range(500)],
        )

        dataset = VerifiedDataset(fixture.manifest, pins=fixture.pins)

        self.assertEqual(len(dataset), 500)
        self.assertEqual(dataset.instance_ids[0], "owner__repo-0000")
        self.assertEqual(dataset.instance_ids[-1], "owner__repo-0499")
        record = dataset[17]
        self.assertEqual(tuple(record.policy_instance), POLICY_FIELDS)
        self.assertEqual(
            record.policy_instance,
            {
                "instance_id": "owner__repo-0017",
                "repo": "owner/repo",
                "base_commit": f"{18:040x}",
                "problem_statement": "Repair issue 17",
            },
        )
        serialized = json.dumps(record.policy_instance)
        for secret in (
            "SECRET_GOLD_17",
            "SECRET_TEST_17",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "SECRET_HINT_17",
            "SECRET_EVAL_17",
            "SECRET_PARSER_17",
            "SECRET_LOG_17",
        ):
            self.assertNotIn(secret, serialized)
        private = record.private_instance()
        self.assertEqual(private["patch"], "SECRET_GOLD_17")
        self.assertEqual(private["test_patch"], "SECRET_TEST_17")
        self.assertEqual(dataset.provenance.row_count, 500)
        self.assertEqual(
            dataset.provenance.canonical_jsonl_sha256,
            fixture.pins.canonical_jsonl_sha256,
        )

    def test_manifest_claims_must_equal_the_frozen_pins(self) -> None:
        fixture = DatasetFixture(self.root, [fixture_row(0)])
        manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
        manifest["dataset"]["revision"] = "f" * 40
        fixture.manifest.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(VerifiedDatasetError, "revision"):
            VerifiedDataset(fixture.manifest, pins=fixture.pins)

    def test_jsonl_bytes_and_manifest_hash_must_match(self) -> None:
        fixture = DatasetFixture(self.root, [fixture_row(0)])
        fixture.jsonl.write_bytes(fixture.jsonl.read_bytes() + b"\n")

        with self.assertRaisesRegex(VerifiedDatasetError, "SHA-256"):
            VerifiedDataset(fixture.manifest, pins=fixture.pins)

    def test_rows_must_be_sorted_and_unique(self) -> None:
        unsorted_fixture = DatasetFixture(
            self.root,
            [fixture_row(1), fixture_row(0)],
        )
        with self.assertRaisesRegex(VerifiedDatasetError, "sorted"):
            VerifiedDataset(unsorted_fixture.manifest, pins=unsorted_fixture.pins)

        duplicate_root = self.root / "duplicate"
        duplicate_root.mkdir()
        duplicate_fixture = DatasetFixture(
            duplicate_root,
            [fixture_row(0), fixture_row(0)],
        )
        with self.assertRaisesRegex(VerifiedDatasetError, "unique"):
            VerifiedDataset(duplicate_fixture.manifest, pins=duplicate_fixture.pins)

    def test_jsonl_must_use_the_canonical_serialization(self) -> None:
        fixture = DatasetFixture(self.root, [fixture_row(0)])
        noncanonical = (json.dumps(fixture.rows[0], indent=2) + "\n").encode()
        fixture.jsonl.write_bytes(noncanonical)
        manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
        manifest["canonical_jsonl"]["sha256"] = hashlib.sha256(
            noncanonical
        ).hexdigest()
        fixture.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        pins = FrozenDatasetPins(
            repository=fixture.pins.repository,
            revision=fixture.pins.revision,
            split=fixture.pins.split,
            row_count=1,
            canonical_jsonl_sha256=manifest["canonical_jsonl"]["sha256"],
            id_ledger_sha256=fixture.pins.id_ledger_sha256,
        )

        with self.assertRaisesRegex(VerifiedDatasetError, "canonical"):
            VerifiedDataset(fixture.manifest, pins=pins)

    def test_data_idx_must_stay_inside_the_canonical_order(self) -> None:
        fixture = DatasetFixture(self.root, [fixture_row(0)])
        dataset = VerifiedDataset(fixture.manifest, pins=fixture.pins)

        with self.assertRaisesRegex(IndexError, "outside"):
            dataset[-1]
        with self.assertRaisesRegex(IndexError, "outside"):
            dataset[1]


if __name__ == "__main__":
    unittest.main()
