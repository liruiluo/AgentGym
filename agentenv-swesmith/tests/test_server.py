from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agentenv_swesmith.launch import _runtime_source_from_environment
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


if __name__ == "__main__":
    unittest.main()
