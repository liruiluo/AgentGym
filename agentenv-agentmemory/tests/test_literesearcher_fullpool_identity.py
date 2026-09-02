from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.literesearcher.full_pool import load_full_pool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _targets_sha256(targets: list[str]) -> str:
    encoded = json.dumps(
        targets,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LiteResearcherFullPoolIdentityTests(unittest.TestCase):
    def _load(self, rows: list[dict[str, object]]):
        raw_rows: list[dict[str, object]] = []
        normalized_rows: list[dict[str, object]] = []
        for physical_row, source_row in enumerate(rows):
            question = f"Question {physical_row}?"
            targets = [f"Answer {physical_row}"]
            mask_url = f"https://example.test/{physical_row}"
            row = dict(source_row)
            row.update(
                {
                    "parquet_path": "train.parquet",
                    "physical_row": physical_row,
                    "question_sha256": _text_sha256(question),
                    "targets_sha256": _targets_sha256(targets),
                    "mask_url_sha256": _text_sha256(mask_url),
                    "reward_style": "llm",
                    "row_identity": f"{physical_row + 1:064x}",
                    "upstream_curriculum_stage": 1,
                }
            )
            normalized_rows.append(row)
            raw_rows.append(
                {
                    "question": question,
                    "data_source": "fixture",
                    "reward_model": {
                        "style": "llm",
                        "ground_truth": {"target": targets},
                    },
                    "extra_info": {"mask_url": mask_url},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            rows_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in normalized_rows
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "agentmemory_literesearcher_full_compatible_pool_v1",
                        "artifacts": {
                            "pool_rows.jsonl": {"sha256": _sha256(rows_path)}
                        },
                        "source_reports": [
                            {
                                "parquet_relative_path": "train.parquet",
                                "parquet_sha256": "unused-by-mock",
                                "physical_rows": len(raw_rows),
                            }
                        ],
                        "pool": {"contract_compatible_rows": len(raw_rows)},
                        "upstream": {
                            "dataset_revision": "fixture-revision",
                            "source_commit": "fixture-commit",
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            with patch(
                "agentenv_agentmemory.literesearcher.full_pool._read_raw_sources",
                return_value={"train.parquet": raw_rows},
            ):
                return load_full_pool(manifest_path, rows_path, root)

    def test_legacy_rows_use_contiguous_pool_index_as_source_identity(self) -> None:
        tasks = self._load([{"pool_index": 0}, {"pool_index": 1}])
        self.assertEqual([task.index for task in tasks.train], [0, 1])
        self.assertEqual(
            [task.source_pool_index for task in tasks.train],
            [0, 1],
        )

    def test_derived_rows_preserve_distinct_original_source_indices(self) -> None:
        tasks = self._load(
            [
                {"pool_index": 0, "source_pool_index": 17},
                {"pool_index": 1, "source_pool_index": 42},
            ]
        )
        self.assertEqual([task.index for task in tasks.train], [0, 1])
        self.assertEqual(
            [task.source_pool_index for task in tasks.train],
            [17, 42],
        )

    def test_runtime_indices_must_remain_contiguous(self) -> None:
        with self.assertRaisesRegex(ValueError, "row order is not contiguous"):
            self._load(
                [
                    {"pool_index": 0, "source_pool_index": 17},
                    {"pool_index": 2, "source_pool_index": 42},
                ]
            )

    def test_source_indices_must_be_unique_and_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "source pool index must be unique"):
            self._load(
                [
                    {"pool_index": 0, "source_pool_index": 17},
                    {"pool_index": 1, "source_pool_index": 17},
                ]
            )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            self._load([{"pool_index": 0, "source_pool_index": -1}])


if __name__ == "__main__":
    unittest.main()
