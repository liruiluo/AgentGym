from __future__ import annotations

import importlib.util
import hashlib
import subprocess
import sys
import tempfile
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

    def test_canonical_jsonl_encoding_has_stable_bytes(self):
        encoded = self.module.encode_projected_row({"text": "second", "docid": "D-1"})
        self.assertEqual(encoded, b'{"docid":"D-1","text":"second"}\n')
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "b8e62b36a38da3384b4c64bb871e5e12ad9ba3ad18287db602f639eb6e2a7096",
        )

    def test_source_attestation_hashes_actual_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "source.parquet"
            source.write_bytes(b"parquet fixture")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            evidence = self.module.attest_source_paths(
                (source.resolve(),),
                expected_source_sha256=(digest,),
            )
            self.assertEqual(evidence[0]["size_bytes"], len(b"parquet fixture"))
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "do not match"):
                self.module.attest_source_paths(
                    (source.resolve(),),
                    expected_source_sha256=(digest,),
                )

    def test_manifest_paths_are_relative_to_manifest_directory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "corpus-source" / "data.parquet"
            manifest = root / "corpus.manifest.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fixture")
            self.assertEqual(
                self.module.relative_manifest_path(source, manifest),
                "corpus-source/data.parquet",
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
