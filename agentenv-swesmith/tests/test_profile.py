from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentenv_swesmith.profile import (
    OfficialSwesmithProfileResolver,
    SwesmithProfileError,
    _effective_full_command,
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

    def test_effective_full_command_deduplicates_in_first_seen_order(self) -> None:
        command, corrections = _effective_full_command(
            "pytest -vv tests/a.py tests/b.py tests/a.py tests/c.py",
            source_paths=(
                "tests/a.py",
                "tests/b.py",
                "tests/a.py",
                "tests/c.py",
            ),
            expected_paths=(
                "tests/a.py",
                "tests/b.py",
                "tests/a.py",
                "tests/c.py",
            ),
            image="example/python-image",
        )
        self.assertEqual(command, "pytest -vv tests/a.py tests/b.py tests/c.py")
        self.assertEqual(
            corrections,
            ("deduplicate_test_paths_preserve_first_occurrence_v1",),
        )

    def test_effective_full_command_uses_only_sybil_for_scrapy_rst(self) -> None:
        command, corrections = _effective_full_command(
            "pytest -vv tests/a.py docs/guide.rst tests/a.py",
            source_paths=("tests/a.py", "docs/guide.rst", "tests/a.py"),
            expected_paths=("tests/a.py", "docs/guide.rst", "tests/a.py"),
            image="swebench/swesmith.x86_64.scrapy_1776_scrapy.35212ec5",
        )
        self.assertEqual(
            command,
            "pytest -vv -p no:doctest tests/a.py docs/guide.rst",
        )
        self.assertEqual(
            corrections,
            (
                "deduplicate_test_paths_preserve_first_occurrence_v1",
                "scrapy_sybil_disable_builtin_doctest_v1",
            ),
        )

    def test_effective_full_command_rejects_path_suffix_mismatch(self) -> None:
        with self.assertRaisesRegex(SwesmithProfileError, "does not end"):
            _effective_full_command(
                "pytest -vv tests/other.py",
                source_paths=("tests/a.py",),
                expected_paths=("tests/a.py",),
                image="example/python-image",
            )

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
