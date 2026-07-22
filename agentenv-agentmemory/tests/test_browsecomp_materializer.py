from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audits"
    / "materialize_browsecomp_corpus.py"
)


def load_materializer_module():
    spec = importlib.util.spec_from_file_location("browsecomp_materializer", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BrowseCompMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_materializer_module()

    def test_small_fixture_projects_only_openai_searcher_fields(self):
        rows = [
            {"docid": "D-1", "text": "first", "url": "https://one.invalid"},
            {"docid": "D-2", "text": "第二段", "url": "https://two.invalid"},
        ]
        self.assertEqual(
            list(self.module.materialize_rows(rows)),
            [
                {"docid": "D-1", "text": "first"},
                {"docid": "D-2", "text": "第二段"},
            ],
        )

    def test_small_fixture_rejects_unknown_column_and_duplicate_docid(self):
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            list(
                self.module.materialize_rows(
                    [{"docid": "D-1", "text": "x", "url": "u", "title": "x"}]
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate docid"):
            list(
                self.module.materialize_rows(
                    [
                        {"docid": "D-1", "text": "x", "url": "u"},
                        {"docid": "D-1", "text": "y", "url": "u2"},
                    ]
                )
            )

    def test_cli_help_does_not_require_pyarrow(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--input-glob", result.stdout)
        self.assertIn("--manifest", result.stdout)


if __name__ == "__main__":
    unittest.main()
