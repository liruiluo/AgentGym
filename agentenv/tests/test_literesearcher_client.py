from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.envs.literesearcher import LiteResearcherEnvClient


class LiteResearcherClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
        return client

    def test_policy_framing_exposes_normalized_conversation_start(self) -> None:
        framing = self._client().policy_framing()
        self.assertEqual(
            [message["role"] for message in framing], ["user", "assistant"]
        )
        self.assertIn("deep-research agent", framing[0]["content"])
        self.assertEqual(
            framing[1], {"role": "assistant", "content": "Understood."}
        )

    def test_close_accepts_server_boolean_true(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = True
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=response,
        ) as request:
            self.assertTrue(self._client().close())
        request.assert_called_once_with(
            "POST",
            "http://literesearcher.example/close",
            timeout=30,
            json={"id": 7},
        )

    def test_close_rejects_false_acknowledgement(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = False
        with (
            patch(
                "agentenv.envs.literesearcher.requests.request",
                return_value=response,
            ),
            self.assertRaisesRegex(requests.RequestException, "did not return true"),
        ):
            self._client().close()


if __name__ == "__main__":
    unittest.main()
