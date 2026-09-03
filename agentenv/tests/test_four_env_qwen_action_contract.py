from __future__ import annotations

import re
import unittest

from agentenv.controller.types import ActionFormat
from agentenv.envs.agentmemory import (
    PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
    build_filesystem_conversation_start,
    normalize_filesystem_webshop_policy_action,
)
from agentenv.envs.literesearcher import (
    LITERESEARCHER_SYSTEM_PROMPT,
    normalize_literesearcher_policy_action,
)
from agentenv.envs.openmle_fast import (
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
    _normalize_openmle_policy_action,
)
from agentenv.envs.swesmith import (
    SWE_POLICY_SYSTEM_PROMPT,
    normalize_swesmith_policy_action,
)
from agentenv.envs.verl_qwen_tool_parser import QWEN_INVALID_ACTION_SENTINEL


def qwen_call(name: str, **parameters: object) -> str:
    body = ["<tool_call>", f"<function={name}>"]
    for key, value in parameters.items():
        body.extend(
            (
                f"<parameter={key}>",
                str(value),
                "</parameter>",
            )
        )
    body.extend(("</function>", "</tool_call>"))
    return "\n".join(body)


class FourEnvironmentQwenActionContractTest(unittest.TestCase):
    def test_every_policy_prompt_teaches_native_qwen_xml_not_wrapped_json(self) -> None:
        webshop_prompt = build_filesystem_conversation_start(
            ActionFormat.REACT,
            surface=PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
        )[0]["value"]
        prompts = {
            "webshop": webshop_prompt,
            "swesmith": SWE_POLICY_SYSTEM_PROMPT,
            "literesearcher": LITERESEARCHER_SYSTEM_PROMPT,
            "openmle_fast": OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
        }
        for environment, prompt in prompts.items():
            with self.subTest(environment=environment):
                self.assertIn("<tool_call>", prompt)
                self.assertIn("<function=", prompt)
                self.assertIn("<parameter=", prompt)
                self.assertIsNone(re.search(r"<tool_call>\s*\{", prompt))

    def assert_valid_translation(
        self,
        normalizer,
        action: str,
        expected: str,
        **kwargs: object,
    ) -> None:
        submitted, evidence = normalizer(action, **kwargs)
        self.assertEqual(submitted, expected)
        self.assertEqual(evidence["tool_contract"], "qwen3_xml_single_call_v1")
        self.assertEqual(evidence["tool_parser"], "qwen3_coder")
        self.assertIs(evidence["tool_parser_normalized"], True)
        self.assertEqual(evidence["submitted_action"], expected)

    def assert_rejected(self, normalizer, action: str, **kwargs: object) -> None:
        submitted, evidence = normalizer(action, **kwargs)
        self.assertEqual(submitted, QWEN_INVALID_ACTION_SENTINEL)
        self.assertEqual(evidence["tool_contract"], "qwen3_xml_single_call_v1")
        self.assertEqual(evidence["tool_parser"], "qwen3_coder")
        self.assertIs(evidence["tool_parser_normalized"], False)
        self.assertTrue(evidence["tool_parser_error"])
        self.assertEqual(evidence["submitted_action"], QWEN_INVALID_ACTION_SENTINEL)

    def test_native_qwen_xml_translates_each_environment_action(self) -> None:
        self.assert_valid_translation(
            normalize_filesystem_webshop_policy_action,
            qwen_call("search", keywords="red mug"),
            "search[red mug]",
        )
        self.assert_valid_translation(
            normalize_filesystem_webshop_policy_action,
            qwen_call("click", item="Buy Now"),
            "click[Buy Now]",
        )
        self.assert_valid_translation(
            normalize_swesmith_policy_action,
            qwen_call(
                "shell_command",
                command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                workdir=".",
            ),
            'shell_command {"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",'
            '"workdir":"."}',
        )
        self.assert_valid_translation(
            normalize_literesearcher_policy_action,
            qwen_call("answer", answer="Evidence-backed answer."),
            "<answer>Evidence-backed answer.</answer>",
        )
        self.assert_valid_translation(
            normalize_literesearcher_policy_action,
            qwen_call(
                "visit",
                url="https://literesearcher.local/page/one?x=1&y=2",
                goal="find the source-backed evidence",
                page=1,
            ),
            "<tool_call>\n"
            "<function=visit>\n"
            "<parameter=url>\n"
            '"https://literesearcher.local/page/one?x=1&y=2"\n'
            "</parameter>\n"
            "<parameter=goal>\n"
            '"find the source-backed evidence"\n'
            "</parameter>\n"
            "<parameter=page>\n"
            "1\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
        )
        self.assert_valid_translation(
            _normalize_openmle_policy_action,
            qwen_call("submit"),
            "submit",
        )

    def test_webshop_balanced_bracket_title_round_trips_to_native_action(self) -> None:
        title = '[2022] 12" monitor [USB-C]'
        self.assert_valid_translation(
            normalize_filesystem_webshop_policy_action,
            qwen_call("click", item=title),
            f"click[{title}]",
        )

    def test_literesearcher_visit_json_looking_strings_remain_strings(self) -> None:
        submitted, evidence = normalize_literesearcher_policy_action(
            qwen_call("visit", url="123", goal="true")
        )
        self.assertEqual(
            submitted,
            "<tool_call>\n"
            "<function=visit>\n"
            "<parameter=url>\n"
            '"123"\n'
            "</parameter>\n"
            "<parameter=goal>\n"
            '"true"\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
        )
        self.assertIs(evidence["tool_parser_normalized"], True)

    def test_literesearcher_visit_rejects_silent_whitespace_normalization(self) -> None:
        self.assert_rejected(
            normalize_literesearcher_policy_action,
            qwen_call("visit", url=" 123", goal="true"),
        )

    def test_shared_filesystem_actions_use_same_policy_facing_envelope(self) -> None:
        shell = qwen_call(
            "shell_command",
            command="cat .agent_memory/CONTINUATION.md",
            workdir=".",
            timeout_ms=20000,
        )
        patch = qwen_call(
            "apply_patch",
            patch="*** Begin Patch\n*** Add File: note.txt\n+ok\n*** End Patch",
        )
        for normalizer in (
            normalize_filesystem_webshop_policy_action,
            normalize_swesmith_policy_action,
            normalize_literesearcher_policy_action,
            _normalize_openmle_policy_action,
        ):
            with self.subTest(environment=normalizer.__module__, action="shell"):
                submitted, evidence = normalizer(shell)
                self.assertTrue(submitted.startswith("shell_command "))
                self.assertIs(evidence["tool_parser_normalized"], True)
            with self.subTest(environment=normalizer.__module__, action="patch"):
                submitted, evidence = normalizer(patch)
                self.assertTrue(submitted.startswith("apply_patch\n*** Begin Patch"))
                self.assertIs(evidence["tool_parser_normalized"], True)

    def test_xml_wrapped_json_is_not_a_qwen_native_function_call(self) -> None:
        wrapped_json = (
            '<tool_call>{"name":"shell_command","arguments":'
            '{"command":"pwd"}}</tool_call>'
        )
        for normalizer in (
            normalize_filesystem_webshop_policy_action,
            normalize_swesmith_policy_action,
            normalize_literesearcher_policy_action,
            _normalize_openmle_policy_action,
        ):
            with self.subTest(environment=normalizer.__module__):
                self.assert_rejected(normalizer, wrapped_json)

    def test_bare_legacy_actions_are_rejected_at_every_policy_gateway(self) -> None:
        cases = (
            (normalize_filesystem_webshop_policy_action, "search[red mug]"),
            (
                normalize_swesmith_policy_action,
                'shell_command {"command":"pwd","workdir":"."}',
            ),
            (normalize_literesearcher_policy_action, "<answer>answer</answer>"),
            (_normalize_openmle_policy_action, "submit"),
        )
        for normalizer, action in cases:
            with self.subTest(environment=normalizer.__module__, action=action):
                self.assert_rejected(normalizer, action)

    def test_prose_think_and_multiple_calls_are_rejected_everywhere(self) -> None:
        call = qwen_call("shell_command", command="pwd", workdir=".")
        malformed = (
            "I will inspect first.\n" + call,
            "<think>inspect</think>\n" + call,
            call + "\n" + call,
        )
        for normalizer in (
            normalize_filesystem_webshop_policy_action,
            normalize_swesmith_policy_action,
            normalize_literesearcher_policy_action,
            _normalize_openmle_policy_action,
        ):
            for action in malformed:
                with self.subTest(environment=normalizer.__module__, action=action[:24]):
                    self.assert_rejected(normalizer, action)

    def test_openmle_rejects_out_of_contract_parameters(self) -> None:
        for action in (
            qwen_call("shell_command", command="pwd", workdir="/workspace"),
            qwen_call("shell_command", command="pwd", timeout_ms=20001),
            qwen_call("submit", reason="done"),
        ):
            with self.subTest(action=action):
                self.assert_rejected(_normalize_openmle_policy_action, action)


if __name__ == "__main__":
    unittest.main()
