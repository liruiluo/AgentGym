from __future__ import annotations

import re
import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.envs.literesearcher import (
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
)
from agentenv_agentmemory.literesearcher.wrapper import _parse_tool_call


class LiteResearcherClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
        client.info = {"observation": "Which source answers this question?"}
        return client

    def test_prompt_uses_qwen35_native_tool_format(self) -> None:
        self.assertIn("# Tools", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn('"name": "search"', LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn('"name": "visit"', LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=search>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=visit>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("MUST be a JSON array", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            "<answer>your evidence-backed answer</answer>",
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn("Never write\n<function=answer>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("On the first turn, call search", LITERESEARCHER_SYSTEM_PROMPT)

    def test_prompt_tool_examples_pass_the_production_parser(self) -> None:
        examples = re.findall(
            r"<tool_call>.*?</tool_call>",
            LITERESEARCHER_SYSTEM_PROMPT,
            flags=re.DOTALL,
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(
            _parse_tool_call(examples[0]),
            ("search", {"query": ["first search query", "second search query"]}),
        )
        self.assertEqual(
            _parse_tool_call(examples[1]),
            (
                "visit",
                {
                    "url": "URL_COPIED_VERBATIM_FROM_A_SEARCH_RESULT",
                    "goal": "specific evidence to find on that page",
                    "page": 1,
                },
            ),
        )

    def test_policy_framing_restores_the_system_role(self) -> None:
        client = self._client()
        self.assertEqual(
            client.policy_framing(),
            [{"role": "system", "content": LITERESEARCHER_SYSTEM_PROMPT}],
        )
        normalized = client.normalize_initial_policy_context(
            [
                {"role": "user", "content": "legacy prompt"},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": client.observe()},
            ]
        )
        self.assertEqual(
            normalized,
            [
                {"role": "system", "content": LITERESEARCHER_SYSTEM_PROMPT},
                {"role": "user", "content": client.observe()},
            ],
        )

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
            "persists across context compaction",
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
