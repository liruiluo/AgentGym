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

    def test_prompt_uses_the_simple_string_search_contract(self) -> None:
        prompt = LiteResearcherEnvClient.conversation_start[0]["value"]
        self.assertIn('"query":"..."', prompt)
        self.assertNotIn('"query":["..."]', prompt)


if __name__ == "__main__":
    unittest.main()
