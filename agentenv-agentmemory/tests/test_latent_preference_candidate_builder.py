from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LatentPreferenceCandidateBuilderTests(unittest.TestCase):
    def test_builder_is_hash_pinned_and_byte_deterministic(self) -> None:
        records = [
            {
                "asin": "B000000001",
                "name": "BLACK Black black Silicone Phone Case Model One",
                "product_category": (
                    "Cell Phones & Accessories > Cases, Holsters & Sleeves > "
                    "Basic Cases"
                ),
            },
            {
                "asin": "B000000002",
                "name": "Gray Leather Phone Case Model Two",
                "product_category": (
                    "Cell Phones & Accessories > Cases, Holsters & Sleeves > "
                    "Basic Cases"
                ),
            },
            {
                "asin": "B000000003",
                "name": "Chocolate Classic Cake Mix 15 Oz",
                "product_category": (
                    "Grocery & Gourmet Food > Cooking & Baking > "
                    "Baking Mixes > Cakes"
                ),
            },
            {
                "asin": "B000000004",
                "name": "Organic Oat Cookies Family Pack",
                "product_category": (
                    "Grocery & Gourmet Food > Breads & Bakery > Cookies"
                ),
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "items.json"
            catalog.write_text(
                json.dumps(records, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            catalog_sha256 = hashlib.sha256(catalog.read_bytes()).hexdigest()
            first_candidates = root / "candidates-one.jsonl"
            first_report = root / "report-one.json"
            second_candidates = root / "candidates-two.jsonl"
            second_report = root / "report-two.json"
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "audits"
                / "build_latent_preference_candidates.py"
            )
            command = [
                sys.executable,
                str(script),
                "--items-file",
                str(catalog),
                "--catalog-sha256",
                catalog_sha256,
            ]
            first_environment = dict(os.environ)
            first_environment["PYTHONHASHSEED"] = "1"
            first = subprocess.run(
                [
                    *command,
                    "--output-candidates",
                    str(first_candidates),
                    "--output-report",
                    str(first_report),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=first_environment,
            )
            second_environment = dict(os.environ)
            second_environment["PYTHONHASHSEED"] = "777"
            subprocess.run(
                [
                    *command,
                    "--output-candidates",
                    str(second_candidates),
                    "--output-report",
                    str(second_report),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=second_environment,
            )

            self.assertEqual(
                first_candidates.read_bytes(),
                second_candidates.read_bytes(),
            )
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            rows = [
                json.loads(line)
                for line in first_candidates.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {row["asin"] for row in rows},
                {"B000000001", "B000000002", "B000000003", "B000000004"},
            )
            black = next(
                row
                for row in rows
                if row["asin"] == "B000000001" and row["axis"] == "color"
            )
            self.assertEqual(
                black["title_evidence"],
                ["BLACK", "Black", "black"],
            )
            edges = {(row["asin"], row["axis"]) for row in rows}
            self.assertIn(("B000000003", "flavor"), edges)
            self.assertNotIn(("B000000003", "dietary_profile"), edges)
            self.assertIn(("B000000004", "dietary_profile"), edges)
            self.assertNotIn(("B000000004", "flavor"), edges)
            report = json.loads(first_report.read_text(encoding="utf-8"))
            self.assertTrue(
                report["verification_scope"]["independent_axis_candidate_rows"]
            )
            self.assertIn(
                "human_review_required=false llm_judge_required=false",
                first.stdout,
            )

    def test_builder_rejects_catalog_hash_mismatch_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "items.json"
            catalog.write_text("[]", encoding="utf-8")
            candidates = root / "candidates.jsonl"
            report = root / "report.json"
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "audits"
                / "build_latent_preference_candidates.py"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--items-file",
                    str(catalog),
                    "--catalog-sha256",
                    "0" * 64,
                    "--output-candidates",
                    str(candidates),
                    "--output-report",
                    str(report),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("catalog SHA256 mismatch", result.stderr)
            self.assertFalse(candidates.exists())
            self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
