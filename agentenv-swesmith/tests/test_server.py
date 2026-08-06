from __future__ import annotations

import os
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
