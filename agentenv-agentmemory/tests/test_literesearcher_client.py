from __future__ import annotations

import json
import re
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

    def test_prompt_contains_parseable_tool_and_workspace_examples(self) -> None:
        prompt = LiteResearcherEnvClient.conversation_start[0]["value"]
        tool_calls = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            prompt,
            flags=re.DOTALL,
        )
        self.assertEqual(len(tool_calls), 2)
        parsed = [json.loads(value) for value in tool_calls]
        self.assertEqual([item["name"] for item in parsed], ["search", "visit"])
        self.assertEqual(parsed[0]["arguments"]["query"], ["query one", "query two"])
        self.assertIn("]}}</tool_call>", prompt)
        self.assertIn('shell_command {"command":', prompt)
        self.assertIn("*** Begin Patch ... *** End Patch", prompt)
        self.assertIn("survives context compaction", prompt)


if __name__ == "__main__":
    unittest.main()
