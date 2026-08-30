from __future__ import annotations

from pathlib import Path
import unittest

from agentenv.envs.agentmemory import (
    PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
    build_filesystem_conversation_start,
)
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_BARE_CHECKPOINT_READ_ACTION,
    FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION,
)
from agentenv.envs.webshop_handoff import (
    WEBSHOP_CONTEXT_COMPACTION_REQUEST,
    WEBSHOP_POLICY_CONTINUATION_MARKER,
    WEBSHOP_SESSION_HANDOFF_REQUEST,
)
from agentenv.controller.types import ActionFormat


class WebShopCheckpointPromptAlignmentTest(unittest.TestCase):
    def test_react_prompt_and_dynamic_checkpoint_request_use_same_bare_format(self) -> None:
        conversation = build_filesystem_conversation_start(
            ActionFormat.REACT,
            surface=PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
        )
        system_prompt = conversation[0]["value"]
        for prompt in (
            system_prompt,
            WEBSHOP_SESSION_HANDOFF_REQUEST,
            WEBSHOP_CONTEXT_COMPACTION_REQUEST,
        ):
            self.assertIn("shell_command", prompt)
            self.assertIn(".agent_memory/CONTINUATION.md", prompt)
            self.assertNotIn("Qwen XML", prompt)
        self.assertIn("Use shell_command with one JSON object", system_prompt)
        for request in (
            WEBSHOP_SESSION_HANDOFF_REQUEST,
            WEBSHOP_CONTEXT_COMPACTION_REQUEST,
        ):
            self.assertIn(FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION, request)
            self.assertIn("do not change the command shape", request)
        self.assertIn(
            FILESYSTEM_BARE_CHECKPOINT_READ_ACTION,
            WEBSHOP_POLICY_CONTINUATION_MARKER,
        )
        self.assertIn("do not use search, click, or `rg`", WEBSHOP_POLICY_CONTINUATION_MARKER)
        self.assertIn("On the following action", WEBSHOP_POLICY_CONTINUATION_MARKER)

    def test_webshop_client_wires_exact_read_marker_after_replace_and_retry(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "agentenv"
            / "envs"
            / "agentmemory.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            source.count(
                "continuation_marker=WEBSHOP_POLICY_CONTINUATION_MARKER"
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
