from __future__ import annotations

import unittest

from agentenv.envs.agentmemory import (
    PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
    build_filesystem_conversation_start,
)
from agentenv.envs.webshop_handoff import (
    WEBSHOP_CONTEXT_COMPACTION_REQUEST,
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
        self.assertIn("canonical bare shell_command JSON form", WEBSHOP_SESSION_HANDOFF_REQUEST)


if __name__ == "__main__":
    unittest.main()
