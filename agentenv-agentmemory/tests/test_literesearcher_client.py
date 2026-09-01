from __future__ import annotations

import re
import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.envs.literesearcher import (
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
)
from agentenv_agentmemory.literesearcher.wrapper import (
    _canonical_workspace_action,
    _parse_tool_call,
)


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
        self.assertNotIn("<function=answer>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("On the first turn, call search", LITERESEARCHER_SYSTEM_PROMPT)

    def test_prompt_tool_examples_pass_the_production_parser(self) -> None:
        examples = re.findall(
            r"<tool_call>.*?</tool_call>",
            LITERESEARCHER_SYSTEM_PROMPT,
            flags=re.DOTALL,
        )
        self.assertEqual(len(examples), 4)
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
        shell_name, shell_arguments = _parse_tool_call(examples[2])
        self.assertEqual(shell_name, "shell_command")
        self.assertEqual(
            shell_arguments,
            {
                "command": "cat .agent_memory/research.md",
                "workdir": ".",
                "timeout_ms": 10000,
            },
        )
        self.assertEqual(
            _canonical_workspace_action(shell_name, shell_arguments),
            'shell_command {"command":"cat .agent_memory/research.md",'
            '"timeout_ms":10000,"workdir":"."}',
        )
        patch_name, patch_arguments = _parse_tool_call(examples[3])
        self.assertEqual(patch_name, "apply_patch")
        self.assertEqual(
            _canonical_workspace_action(patch_name, patch_arguments),
            "apply_patch\n*** Begin Patch\n*** Add File: "
            ".agent_memory/research.md\n+Question, evidence, source URLs, and "
            "next steps.\n*** End Patch",
        )

    def test_parser_accepts_observed_direct_patch_xml_without_broad_rewrite(self) -> None:
        raw = """<tool_call>
<function=apply_patch>
*** Begin Patch
*** Add File: .agent_memory/CONTINUATION.md
+objective: answer
+next: inspect source
*** End Patch
</function>
</tool_call>"""
        name, arguments = _parse_tool_call(raw)
        self.assertEqual(name, "apply_patch")
        self.assertEqual(
            _canonical_workspace_action(name, arguments),
            "apply_patch\n*** Begin Patch\n*** Add File: "
            ".agent_memory/CONTINUATION.md\n+objective: answer\n"
            "+next: inspect source\n*** End Patch",
        )

    def test_parser_normalizes_quoted_qwen_parameter_names(self) -> None:
        raw = """<tool_call>
<function=shell_command>
<parameter="command">
cat .agent_memory/CONTINUATION.md
</parameter>
<parameter="workdir">
.
</parameter>
<parameter="timeout_ms">
10000
</parameter>
</function>
</tool_call>"""
        name, arguments = _parse_tool_call(raw)
        self.assertEqual(name, "shell_command")
        self.assertEqual(
            arguments,
            {
                "command": "cat .agent_memory/CONTINUATION.md",
                "workdir": ".",
                "timeout_ms": 10000,
            },
        )

    def test_parser_rejects_incomplete_workspace_xml(self) -> None:
        raw = """<tool_call>
<function=apply_patch>
*** Begin Patch
*** Add File: .agent_memory/CONTINUATION.md
+unfinished
*** End Patch"""
        self.assertIsNone(_parse_tool_call(raw))

    def test_parser_rejects_fenced_workspace_actions(self) -> None:
        bare = 'shell_command {"command":"pwd"}'
        for fence in ("```analysis```\n", "~~~xml\n"):
            with self.subTest(fence=fence):
                self.assertIsNone(_parse_tool_call(fence + bare))

        xml = """<tool_call>
<function=shell_command>
<parameter=command>
pwd
</parameter>
</function>
</tool_call>"""
        for fence in ("```analysis```\n", "~~~xml\n"):
            with self.subTest(fence=fence):
                with self.assertRaisesRegex(
                    ValueError, "workspace tool_call must be the complete policy output"
                ):
                    _parse_tool_call(fence + xml)

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

    def test_prompt_exposes_one_qwen_xml_grammar_for_all_tools(self) -> None:
        self.assertIn('"name": "shell_command"', LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn('"name": "apply_patch"', LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=shell_command>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<parameter=command>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=apply_patch>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<parameter=patch>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            "never mix XML with a bare Codex-style action",
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
