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

    def test_resolver_preserves_official_full_suite_without_explicit_paths(
        self,
    ) -> None:
        full_command = (
            "source /opt/miniconda3/bin/activate; conda activate testbed; "
            "pytest --disable-warnings --color=no --tb=no --verbose"
        )

        class FullSuiteProfile:
            image_name = "swebench/swesmith.x86_64.feedparser_2713_cad965a3"
            log_parser = staticmethod(lambda log: {})

            def get_test_files(self, instance):
                return (
                    ["tests/test_well_formed.py"],
                    [
                        "tests/test_date_parsers.py",
                        "tests/test_well_formed.py",
                    ],
                )

            def get_test_cmd(self, instance, f2p_only=False):
                if f2p_only:
                    return (
                        f"{full_command} tests/test_well_formed.py",
                        ["tests/test_well_formed.py"],
                    )
                return full_command, []

        class Registry:
            @staticmethod
            def get_from_inst(instance):
                return FullSuiteProfile()

        resolver = OfficialSwesmithProfileResolver()
        resolver._loaded = True
        resolver._registry = Registry()
        resolver._get_eval_tests_report = lambda *args, **kwargs: {}
        resolver._get_resolution_status = lambda report: "FULL"
        resolver._full_resolution_status = "FULL"

        binding = resolver.resolve(
            {
                "instance_id": "kurtmckee__feedparser.cad965a3.combine_file__yet8s94g",
                "repo": "swesmith/kurtmckee__feedparser.cad965a3",
            }
        )

        self.assertEqual(binding.full_command, full_command)
        self.assertEqual(binding.source_full_command, full_command)
        self.assertEqual(binding.command_corrections, ())

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
