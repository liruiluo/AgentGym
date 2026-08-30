from __future__ import annotations

import unittest

from agentenv.envs.agentmemory import parse_qwen_workspace_action
from agentenv.envs.openmle_fast import (
    OPENMLE_CONTEXT_COMPACTION_REQUEST,
    OPENMLE_EXACT_CHECKPOINT_READ_ACTION,
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
    OPENMLE_POLICY_CONTINUATION_MARKER,
    OPENMLE_QWEN_XML_CHECKPOINT_GUIDANCE,
)


class OpenMLECheckpointPromptAlignmentTest(unittest.TestCase):
    def test_dynamic_request_repeats_qwen_xml_checkpoint_action(self) -> None:
        expected_action = """<tool_call>
<function=shell_command>
<parameter=command>
mkdir -p .agent_memory && printf '%s\n' 'objective: ...' 'measured_validation_or_failure: ...' 'conclusion: ...' 'code_path: train.py' 'next_action: ...' > .agent_memory/CONTINUATION.md
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
20000
</parameter>
</function>
</tool_call>"""
        self.assertIn(expected_action, OPENMLE_QWEN_XML_CHECKPOINT_GUIDANCE)
        self.assertIn(expected_action, OPENMLE_CONTEXT_COMPACTION_REQUEST)
        parsed = parse_qwen_workspace_action(expected_action)
        self.assertIsNotNone(parsed)
        action_name, arguments = parsed
        self.assertEqual(action_name, "shell_command")
        self.assertEqual(arguments["workdir"], ".")
        self.assertEqual(arguments["timeout_ms"], 20000)
        self.assertIn("> .agent_memory/CONTINUATION.md", arguments["command"])
        self.assertIn(
            "do not create, overwrite, edit, or run `train.py`",
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn(
            "Do not inspect data, run code, or submit instead",
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )

    def test_post_checkpoint_marker_requires_exact_qwen_read_only_turn(self) -> None:
        parsed = parse_qwen_workspace_action(OPENMLE_EXACT_CHECKPOINT_READ_ACTION)
        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed,
            (
                "shell_command",
                {
                    "command": "cat .agent_memory/CONTINUATION.md",
                    "workdir": ".",
                    "timeout_ms": 20000,
                },
            ),
        )
        self.assertIn(
            OPENMLE_EXACT_CHECKPOINT_READ_ACTION,
            OPENMLE_POLICY_CONTINUATION_MARKER,
        )
        self.assertIn("Do not overwrite", OPENMLE_POLICY_CONTINUATION_MARKER)
        self.assertIn("nothing else", OPENMLE_POLICY_CONTINUATION_MARKER)
        self.assertIn("After that exact read returns", OPENMLE_POLICY_CONTINUATION_MARKER)
        self.assertNotIn(
            "submit now. Otherwise, after reading",
            OPENMLE_POLICY_CONTINUATION_MARKER,
        )

    def test_prompt_requires_one_qwen_xml_call(self) -> None:
        self.assertIn("<function=shell_command>", OPENMLE_FAST_POLICY_SYSTEM_PROMPT)
        self.assertIn("<function=apply_patch>", OPENMLE_FAST_POLICY_SYSTEM_PROMPT)
        self.assertIn("<function=submit>", OPENMLE_FAST_POLICY_SYSTEM_PROMPT)
        self.assertIn("exactly one Qwen XML function call", OPENMLE_FAST_POLICY_SYSTEM_PROMPT)
        self.assertNotIn(
            "plain-text action protocol",
            OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
        )

    def test_dynamic_request_keeps_checkpoint_contract_local(self) -> None:
        self.assertIn(
            ".agent_memory/CONTINUATION.md",
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn("<tool_call>", OPENMLE_CONTEXT_COMPACTION_REQUEST)
        self.assertIn("Qwen", OPENMLE_CONTEXT_COMPACTION_REQUEST)


if __name__ == "__main__":
    unittest.main()
