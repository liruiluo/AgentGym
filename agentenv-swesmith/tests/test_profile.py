from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentenv_swesmith.profile import (
    OfficialSwesmithProfileResolver,
    SwesmithProfileError,
    _normalize_paths,
)


class SwesmithProfileTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_relative_test_paths(self) -> None:
        self.assertEqual(
            _normalize_paths(
                ["tests/test_one.py", Path("tests/test_one.py"), "test/two.js"],
                "tests",
            ),
            ("tests/test_one.py", "test/two.js"),
        )

    def test_rejects_escaping_test_paths(self) -> None:
        for value in ("../secret", "/absolute", "tests/../secret"):
            with self.subTest(value=value):
                with self.assertRaises(SwesmithProfileError):
                    _normalize_paths([value], "tests")

    def test_source_revision_mismatch_fails_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            import subprocess

            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"],
                check=True,
            )
            (root / "README").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "README"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "init"],
                check=True,
            )
            resolver = OfficialSwesmithProfileResolver(
                source_root=root,
                expected_revision="f" * 40,
            )
            with self.assertRaisesRegex(SwesmithProfileError, "revision mismatch"):
                resolver.resolve({"instance_id": "x", "repo": "x"})


if __name__ == "__main__":
    unittest.main()
