from __future__ import annotations

from pathlib import Path
import unittest

from agentenv.envs.agentmemory import (
    PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
    build_filesystem_conversation_start,
    parse_env_action,
    parse_filesystem_env_action,
)
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_QWEN_CHECKPOINT_READ_ACTION,
    FILESYSTEM_QWEN_CHECKPOINT_WRITE_ACTION,
    FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE,
)
from agentenv.envs.webshop_handoff import (
    WEBSHOP_CONTEXT_COMPACTION_REQUEST,
    WEBSHOP_POLICY_CONTINUATION_MARKER,
    WEBSHOP_SESSION_HANDOFF_REQUEST,
)
from agentenv.controller.types import ActionFormat


class WebShopCheckpointPromptAlignmentTest(unittest.TestCase):
    def test_react_prompt_and_dynamic_checkpoint_request_use_same_qwen_xml_format(self) -> None:
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
            self.assertIn("Qwen XML", prompt)
            self.assertIn("<tool_call>", prompt)
        self.assertIn("never use bare search[...] or click[...] syntax", system_prompt)
        self.assertNotIn('<tool_call>\n{"name":', system_prompt)
        self.assertIn(
            FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE,
            system_prompt,
        )
        for request in (
            WEBSHOP_SESSION_HANDOFF_REQUEST,
            WEBSHOP_CONTEXT_COMPACTION_REQUEST,
        ):
            self.assertIn(FILESYSTEM_QWEN_CHECKPOINT_WRITE_ACTION, request)
            self.assertIn("do not change the command shape", request)
            self.assertIn(FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE, request)
        self.assertIn(
            FILESYSTEM_QWEN_CHECKPOINT_READ_ACTION,
            WEBSHOP_POLICY_CONTINUATION_MARKER,
        )
        self.assertIn("do not use search, click, or `rg`", WEBSHOP_POLICY_CONTINUATION_MARKER)
        self.assertIn("On the following action", WEBSHOP_POLICY_CONTINUATION_MARKER)

    def test_client_parsers_accept_balanced_brackets_in_exact_product_title(self) -> None:
        title = (
            '[2022] 12" Triple Portable Monitor for Laptop, FOPO FHD 1080P IPS '
            'Attachable Laptop Screen Extender, Triple Monitor for 13"-16" '
            'Notebook/Mac/Switch/Xbox One/Phone, Connect with USB-C/HDMI'
        )
        action = f"search[{title}]"
        for parser in (parse_env_action, parse_filesystem_env_action):
            with self.subTest(parser=parser.__name__):
                name, arguments = parser(action)
                self.assertEqual(name, "search")
                self.assertEqual(arguments, {"keywords": title})
        for parser in (parse_env_action, parse_filesystem_env_action):
            with self.subTest(parser=parser.__name__, malformed="multiple"):
                with self.assertRaises(ValueError):
                    parser("search[x] click[y]")

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
