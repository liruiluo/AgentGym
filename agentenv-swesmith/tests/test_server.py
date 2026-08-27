from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agentenv_swesmith.launch import (
    _integer,
    _limits_from_environment,
    _runtime_source_from_environment,
)
from agentenv_swesmith.privacy import private_detail_authorized


class SwesmithServerPrivacyTests(unittest.TestCase):
    def test_private_detail_is_disabled_without_a_server_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(private_detail_authorized("anything"))

    def test_private_detail_requires_exact_token(self) -> None:
        with patch.dict(os.environ, {"SWESMITH_DETAIL_TOKEN": "audit-secret"}, clear=True):
            self.assertFalse(private_detail_authorized(None))
            self.assertFalse(private_detail_authorized("audit-secret-wrong"))
            self.assertTrue(private_detail_authorized("audit-secret"))


class SwesmithRuntimeSourceTests(unittest.TestCase):
    def test_runtime_source_requires_both_commits(self) -> None:
        with patch.dict(
            os.environ,
            {"SWESMITH_RUNTIME_OUTER_COMMIT": "a" * 40},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "must be set together"):
                _runtime_source_from_environment()

    def test_runtime_source_binds_exact_commits(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SWESMITH_RUNTIME_OUTER_COMMIT": "a" * 40,
                "SWESMITH_RUNTIME_INNER_COMMIT": "b" * 40,
            },
            clear=True,
        ):
            self.assertEqual(
                _runtime_source_from_environment(),
                {
                    "runtime_source": {
                        "outer_commit": "a" * 40,
                        "inner_commit": "b" * 40,
                        "source_id": f"{'a' * 40}_{'b' * 40}",
                    }
                },
            )

    def test_stdout_capture_covers_full_filesystem_checkpoint(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            limits = _limits_from_environment(6144)
        self.assertEqual(limits.stdout_bytes, 8192)
        self.assertEqual(limits.stderr_bytes, 3072)

    def test_stdout_capture_cannot_undercut_checkpoint_bound(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SWESMITH_STDOUT_BYTES": "4096",
                "SWESMITH_STDERR_BYTES": "4096",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "filesystem checkpoint bound"):
                _limits_from_environment(6144)

    def test_policy_and_trusted_grader_timeouts_are_independent(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SWESMITH_DEFAULT_TIMEOUT_MS": "120000",
                "SWESMITH_MAX_TIMEOUT_MS": "120000",
                "SWESMITH_GRADER_TIMEOUT_MS": "600000",
            },
            clear=True,
        ):
            limits = _limits_from_environment()
            grader_timeout = _integer(
                "SWESMITH_GRADER_TIMEOUT_MS", limits.max_timeout_ms
            )
        self.assertEqual(limits.default_timeout_ms, 120_000)
        self.assertEqual(limits.max_timeout_ms, 120_000)
        self.assertEqual(grader_timeout, 600_000)


if __name__ == "__main__":
    unittest.main()
