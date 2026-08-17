from __future__ import annotations

import unittest

from agentenv_swesmith.actions import parse_policy_action


class SwesmithActionParserTests(unittest.TestCase):
    def test_parses_canonical_shell_command_and_defaults_workdir(self) -> None:
        parsed = parse_policy_action('shell_command {"command":"rg -n TODO ."}')
        self.assertEqual(parsed.kind, "shell_command")
        self.assertEqual(
            parsed.arguments,
            {"command": "rg -n TODO .", "workdir": "."},
        )
        self.assertFalse(parsed.terminates_episode)

    def test_preserves_explicit_shell_arguments_and_thinking(self) -> None:
        parsed = parse_policy_action(
            "<think>Inspect the failing test.</think>\n"
            'shell_command {"command":"pytest -q","workdir":"src","timeout_ms":120000}'
            "</s>"
        )
        self.assertEqual(parsed.kind, "shell_command")
        self.assertEqual(parsed.thought, "Inspect the failing test.")
        self.assertEqual(parsed.arguments["workdir"], "src")
        self.assertEqual(parsed.arguments["timeout_ms"], 120000)

    def test_parses_complete_apply_patch(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )
        parsed = parse_policy_action("apply_patch\n" + patch)
        self.assertEqual(parsed.kind, "apply_patch")
        self.assertEqual(parsed.patch, patch)
        self.assertFalse(parsed.terminates_episode)

    def test_parses_qwen_native_shell_command(self) -> None:
        parsed = parse_policy_action(
            "<tool_call>\n"
            "<function=shell_command>\n"
            "<parameter=command>\n"
            "python -m pytest tests/test_value.py -q\n"
            "</parameter>\n"
            "<parameter=workdir>\n"
            "/testbed\n"
            "</parameter>\n"
            "<parameter=timeout_ms>\n"
            "120000\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        self.assertEqual(parsed.kind, "shell_command")
        self.assertEqual(
            parsed.arguments,
            {
                "command": "python -m pytest tests/test_value.py -q",
                "workdir": "/testbed",
                "timeout_ms": 120000,
            },
        )

    def test_parses_qwen_native_apply_patch(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )
        parsed = parse_policy_action(
            "<tool_call>\n"
            "<function=apply_patch>\n"
            "<parameter=patch>\n"
            f"{patch}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        self.assertEqual(parsed.kind, "apply_patch")
        self.assertEqual(parsed.patch, patch)

    def test_upstream_submission_sentinel_remains_a_shell_action(self) -> None:
        parsed = parse_policy_action(
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}'
        )
        self.assertEqual(parsed.kind, "shell_command")
        self.assertFalse(parsed.terminates_episode)

    def test_plain_text_never_implicitly_submits(self) -> None:
        for output in (
            "final",
            "Final",
            "final done",
            "final\nsummary",
            "The change is complete; shell_command output now passes.",
        ):
            with self.subTest(output=output):
                parsed = parse_policy_action(output)
                self.assertEqual(parsed.kind, "parser_error")
                self.assertFalse(parsed.terminates_episode)

    def test_upstream_style_reasoning_prefix_preserves_one_canonical_action(self) -> None:
        attempts = (
            (
                "I found the bug. Let me inspect it.\n\n"
                'shell_command {"command":"sed -n \'1,20p\' src/value.py"}',
                "shell_command",
            ),
            (
                "I found the bug. Let me fix it.\n\n"
                "apply_patch\n*** Begin Patch\n*** Update File: src/value.py\n"
                "@@\n-old\n+new\n*** End Patch",
                "apply_patch",
            ),
        )
        for output, tool_hint in attempts:
            with self.subTest(output=output):
                parsed = parse_policy_action(output)
                self.assertEqual(parsed.kind, tool_hint)
                self.assertEqual(parsed.tool_hint, tool_hint)
                self.assertTrue(parsed.thought.startswith("I found the bug"))
                self.assertFalse(parsed.terminates_episode)

    def test_reasoning_prefix_still_rejects_multiple_or_trailing_actions(self) -> None:
        invalid = (
            "Inspect first.\n"
            'shell_command {"command":"pwd"}\n'
            'shell_command {"command":"ls"}',
            "Fix it.\napply_patch\n*** Begin Patch\n*** Update File: src/value.py\n"
            "@@\n-old\n+new\n*** End Patch\nDone.",
            "Explain an example shell_command {\"command\":\"pwd\"}.\n"
            'shell_command {"command":"ls"}',
            "Use a fence.\n```\n"
            'shell_command {"command":"pwd"}\n```',
        )
        for output in invalid:
            with self.subTest(output=output):
                parsed = parse_policy_action(output)
                self.assertEqual(parsed.kind, "parser_error")
                self.assertFalse(parsed.terminates_episode)

    def test_toolish_malformed_outputs_are_not_final_submissions(self) -> None:
        invalid = (
            "shell_command {bad json}",
            "shell_command []",
            'shell_command {"workdir":"."}',
            'shell_command {"command":"pwd","env":{}}',
            'shell_command {"command":"pwd","command":"ls"}',
            'shell_command {"command":"pwd","timeout_ms":true}',
            'shell_command {"command":"pwd","timeout_ms":0}',
            'Shell_Command {"command":"pwd"}',
            'Action: shell_command {"command":"pwd"}',
            '```json\nshell_command {"command":"pwd"}\n```',
            "apply_patch *** Begin Patch\n*** End Patch",
            "apply_patch\n*** Begin Patch\n*** End Patch",
            "apply_patch\n*** Begin Patch\n*** Add File: x\n+x",
            '{"function_name":"shell_command","arguments":{"command":"pwd"}}',
            "<tool_call>\n<function=shell_command>\n"
            "<parameter=command>\npwd\n</parameter>\n</tool_call>",
            "<tool_call>\n<function=shell_command>\n"
            "<parameter=command>\npwd\n</parameter>\n"
            "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=shell_command>\n"
            "<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=shell_command>\n"
            "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        )
        for output in invalid:
            with self.subTest(output=output):
                parsed = parse_policy_action(output)
                self.assertEqual(parsed.kind, "parser_error")
                self.assertFalse(parsed.terminates_episode)
                self.assertIsNotNone(parsed.error)

    def test_empty_or_broken_thinking_is_parser_error(self) -> None:
        for output in ("", "   ", "<think>unfinished", "done</think>"):
            with self.subTest(output=output):
                parsed = parse_policy_action(output)
                self.assertEqual(parsed.kind, "parser_error")
                self.assertFalse(parsed.terminates_episode)

    def test_multiple_thinking_blocks_are_rejected(self) -> None:
        parsed = parse_policy_action(
            "<think>one</think><think>two</think>final"
        )
        self.assertEqual(parsed.kind, "parser_error")

    def test_evidence_contains_raw_and_canonical_classification(self) -> None:
        raw = '<think>check</think>\nshell_command {"command":"pwd"}'
        evidence = parse_policy_action(raw).as_evidence()
        self.assertEqual(evidence["raw_output"], raw)
        self.assertEqual(evidence["kind"], "shell_command")
        self.assertEqual(evidence["thought"], "check")
        self.assertEqual(evidence["arguments"]["workdir"], ".")


if __name__ == "__main__":
    unittest.main()
