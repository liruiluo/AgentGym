from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_literesearcher_stage1.py"
)


def load_materializer_module():
    spec = importlib.util.spec_from_file_location(
        "literesearcher_stage1_materializer", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def read(self, _: int) -> bytes:
        return self.payload


class LiteResearcherMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_materializer_module()

    def candidate(self):
        row = {
            "question": "Which value is directly stated by the source?",
            "reward_model": {"ground_truth": {"target": ["42"]}},
            "extra_info": {"mask_url": "https://en.wikipedia.org/wiki/Answer)"},
        }
        return (
            7,
            row,
            "https://en.wikipedia.org/wiki/Answer",
            "removed_unbalanced_trailing_parenthesis",
        )

    def test_content_addressed_cache_removes_network_dependency(self) -> None:
        page_text = "The source directly states the value 42."
        digest = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = Path(tempdir)
            with mock.patch.object(
                self.module.urllib.request,
                "urlopen",
                return_value=FakeResponse(page_text.encode("utf-8")),
            ):
                first = self.module._fetch_candidate(
                    self.candidate(),
                    expected_sha256=digest,
                    cache_dir=cache_dir,
                    attempts=1,
                )
            self.assertTrue(first.ok)
            self.assertEqual(first.source, "network")

            with mock.patch.object(
                self.module.urllib.request,
                "urlopen",
                side_effect=AssertionError("cache hit must not use the network"),
            ):
                second = self.module._fetch_candidate(
                    self.candidate(),
                    expected_sha256=digest,
                    cache_dir=cache_dir,
                    attempts=1,
                )
            self.assertTrue(second.ok)
            self.assertEqual(second.source, "cache")

    def test_corrupt_cache_is_replaced_only_by_expected_source_bytes(self) -> None:
        page_text = "The source directly states the value 42."
        digest = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = Path(tempdir)
            cache_path = self.module._cache_path(cache_dir, digest)
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text("tampered", encoding="utf-8")
            with mock.patch.object(
                self.module.urllib.request,
                "urlopen",
                return_value=FakeResponse(page_text.encode("utf-8")),
            ):
                outcome = self.module._fetch_candidate(
                    self.candidate(),
                    expected_sha256=digest,
                    cache_dir=cache_dir,
                    attempts=1,
                )
            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.source, "network")
            self.assertEqual(cache_path.read_text(encoding="utf-8"), page_text)

    def test_network_failure_is_distinct_from_semantic_failure(self) -> None:
        with mock.patch.object(
            self.module.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            outcome = self.module._fetch_candidate(
                self.candidate(),
                expected_sha256="0" * 64,
                cache_dir=None,
                attempts=1,
            )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.category, "network_fetch_failed")

    def test_content_change_is_reported_before_anchor_validation(self) -> None:
        page_text = "This changed source no longer contains the expected fact."
        with mock.patch.object(
            self.module.urllib.request,
            "urlopen",
            return_value=FakeResponse(page_text.encode("utf-8")),
        ):
            outcome = self.module._fetch_candidate(
                self.candidate(),
                expected_sha256="0" * 64,
                cache_dir=None,
                attempts=1,
            )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.category, "content_hash_changed")

    def test_matching_source_without_target_is_semantic_failure(self) -> None:
        page_text = "The source discusses an unrelated value."
        digest = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        with mock.patch.object(
            self.module.urllib.request,
            "urlopen",
            return_value=FakeResponse(page_text.encode("utf-8")),
        ):
            outcome = self.module._fetch_candidate(
                self.candidate(),
                expected_sha256=digest,
                cache_dir=None,
                attempts=1,
            )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.category, "target_anchor_missing")


if __name__ == "__main__":
    unittest.main()
