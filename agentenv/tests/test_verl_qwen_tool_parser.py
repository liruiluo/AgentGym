from __future__ import annotations

import unittest

from agentenv.envs.verl_qwen_tool_parser import parse_single_qwen3_tool_call


TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
)


class VerlQwenToolParserAdapterTest(unittest.TestCase):
    def test_reuses_upstream_tolerance_for_parameter_written_as_function(self) -> None:
        parsed = parse_single_qwen3_tool_call(
            """<tool_call>
<function=shell_command>
<parameter=command>
cat .agent_memory/CONTINUATION.md
</parameter>
<function=workdir>
.
</function>
</tool_call>""",
            tool_schemas=TOOLS,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, "shell_command")
        self.assertEqual(
            dict(parsed.arguments),
            {"command": "cat .agent_memory/CONTINUATION.md"},
        )
        self.assertEqual(parsed.parser_name, "qwen3_coder")

    def test_keeps_outer_one_action_no_prose_contract_strict(self) -> None:
        call = """<tool_call>
<function=shell_command>
<parameter=command>pwd</parameter>
</function>
</tool_call>"""
        self.assertIsNone(
            parse_single_qwen3_tool_call("reasoning\n" + call, tool_schemas=TOOLS)
        )
        self.assertIsNone(
            parse_single_qwen3_tool_call(call + "\nextra", tool_schemas=TOOLS)
        )
        self.assertIsNone(
            parse_single_qwen3_tool_call(call + "\n" + call, tool_schemas=TOOLS)
        )

    def test_converts_integer_parameter_through_upstream_schema(self) -> None:
        parsed = parse_single_qwen3_tool_call(
            """<tool_call>
<function=shell_command>
<parameter=command>pwd</parameter>
<parameter=workdir>.</parameter>
<parameter=timeout_ms>20000</parameter>
</function>
</tool_call>""",
            tool_schemas=TOOLS,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.arguments["timeout_ms"], 20000)
        self.assertIsInstance(parsed.arguments["timeout_ms"], int)


if __name__ == "__main__":
    unittest.main()
