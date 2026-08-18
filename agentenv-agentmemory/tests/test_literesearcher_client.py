from __future__ import annotations

import json
import re
import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.envs.literesearcher import (
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
)


class LiteResearcherClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
        return client

    def test_prompt_uses_complete_upstream_style_tool_json(self) -> None:
        tool_calls = re.findall(
            r"<tool_call>(.*?)</tool_call>",
            LITERESEARCHER_SYSTEM_PROMPT,
            flags=re.DOTALL,
        )
        self.assertEqual(len(tool_calls), 2)
        parsed = [json.loads(call) for call in tool_calls]
        self.assertEqual(parsed[0]["name"], "search")
        self.assertEqual(parsed[0]["arguments"]["query"], [
            "first query",
            "second query",
        ])
        self.assertEqual(parsed[1]["name"], "visit")
        self.assertEqual(parsed[1]["arguments"]["page"], 1)
        self.assertIn("Close every brace and bracket", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("Emit exactly one research tool", LITERESEARCHER_SYSTEM_PROMPT)

    def test_prompt_exposes_exact_workspace_action_grammar(self) -> None:
        self.assertIn(
            'shell_command {"command":"cat .agent_memory/research.md",'
            '"workdir":".","timeout_ms":10000}',
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "apply_patch\n*** Begin Patch\n*** Add File: "
            ".agent_memory/research.md",
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "including across context compaction",
            LITERESEARCHER_SYSTEM_PROMPT,
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
