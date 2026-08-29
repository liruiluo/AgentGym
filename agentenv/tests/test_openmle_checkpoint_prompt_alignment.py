from __future__ import annotations

import json
import unittest

from agentenv.envs.openmle_fast import (
    OPENMLE_BARE_CHECKPOINT_GUIDANCE,
    OPENMLE_CONTEXT_COMPACTION_REQUEST,
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
)


class OpenMLECheckpointPromptAlignmentTest(unittest.TestCase):
    def test_dynamic_request_repeats_one_line_bare_checkpoint_action(self) -> None:
        expected_action = (
            'shell_command {"command":"mkdir -p .agent_memory && printf '
            "'%s\\n' 'objective: ...' 'measured_validation_or_failure: ...' "
            "'conclusion: ...' 'code_path: train.py' 'next_action: ...' > "
            '.agent_memory/CONTINUATION.md","workdir":".",'
            '"timeout_ms":20000}'
        )
        self.assertIn(expected_action, OPENMLE_BARE_CHECKPOINT_GUIDANCE)
        self.assertIn(expected_action, OPENMLE_CONTEXT_COMPACTION_REQUEST)
        self.assertIn(expected_action, OPENMLE_FAST_POLICY_SYSTEM_PROMPT)
        self.assertNotIn("\n", expected_action)
        prefix, payload = expected_action.split(" ", 1)
        self.assertEqual(prefix, "shell_command")
        parsed = json.loads(payload)
        self.assertEqual(parsed["workdir"], ".")
        self.assertEqual(parsed["timeout_ms"], 20000)
        self.assertIn("> .agent_memory/CONTINUATION.md", parsed["command"])
        self.assertIn("one physical output line", OPENMLE_CONTEXT_COMPACTION_REQUEST)
        self.assertIn("do not put a raw newline", OPENMLE_CONTEXT_COMPACTION_REQUEST)
        self.assertIn(
            "do not create, overwrite, edit, or run `train.py`",
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn(
            "Do not inspect data, run code, or submit instead",
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )

    def test_dynamic_request_keeps_checkpoint_contract_local(self) -> None:
        self.assertIn(
            ".agent_memory/CONTINUATION.md",
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertNotIn("<tool_call>", OPENMLE_CONTEXT_COMPACTION_REQUEST)
        self.assertNotIn("Qwen XML", OPENMLE_CONTEXT_COMPACTION_REQUEST)


if __name__ == "__main__":
    unittest.main()
