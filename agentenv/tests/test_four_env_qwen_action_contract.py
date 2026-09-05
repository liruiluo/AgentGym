from __future__ import annotations

import json
import re
import unittest

from agentenv.controller.types import ActionFormat
from agentenv.envs.agentmemory import (
    AgentMemoryEnvClient,
    FILESYSTEM_WEBSHOP_SURFACES,
    INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE,
    PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
    FilesystemAgentMemoryAdapter,
    IntentClarificationFilesystemAgentMemoryAdapter,
    _WEBSHOP_QWEN_ASK_TOOL_SCHEMA,
    _WEBSHOP_QWEN_TOOL_SCHEMAS,
    build_filesystem_conversation_start,
    normalize_filesystem_webshop_policy_action,
    normalize_filesystem_webshop_policy_observation,
)
from agentenv.envs.literesearcher import (
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
    _LITERESEARCHER_QWEN_TOOL_SCHEMAS,
    normalize_literesearcher_policy_action,
)
from agentenv.envs.openmle_fast import (
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
    OpenMLEFastEnvClient,
    _OPENMLE_QWEN_TOOL_SCHEMAS,
    _normalize_openmle_policy_action,
    normalize_openmle_policy_observation,
)
from agentenv.envs.swesmith import (
    SWE_POLICY_SYSTEM_PROMPT,
    SwesmithEnvClient,
    _SWE_QWEN_TOOL_SCHEMAS,
    normalize_swesmith_policy_action,
)
from agentenv.envs.verl_qwen_tool_parser import (
    QWEN_INVALID_ACTION_SENTINEL,
    describe_inert_qwen_function_record,
)
from agentenv_openmle_fast.environment import (
    POLICY_PROMPT as OPENMLE_FAST_ENDPOINT_POLICY_PROMPT,
    POLICY_PROMPT_SHA256 as OPENMLE_FAST_ENDPOINT_POLICY_PROMPT_SHA256,
)


_INERT_VALUE_RE = re.compile(
    r"`(?P<key>[^`]+)` exact "
    r"(?P<kind>string|canonical JSON) value encoded as a JSON string: "
    r'(?P<encoded>"(?:\\.|[^"\\])*")'
)


def decode_inert_values(text: str) -> list[tuple[str, object]]:
    decoded: list[tuple[str, object]] = []
    for match in _INERT_VALUE_RE.finditer(text):
        payload = json.loads(match.group("encoded"))
        value = payload if match.group("kind") == "string" else json.loads(payload)
        decoded.append((match.group("key"), value))
    return decoded


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


def qwen_call_with_truncated_final_parameter(
    name: str, **parameters: object
) -> str:
    if not parameters:
        raise ValueError("a truncated final parameter requires at least one parameter")
    body = ["<tool_call>", f"<function={name}>"]
    items = list(parameters.items())
    for index, (key, value) in enumerate(items):
        body.extend((f"<parameter={key}>", str(value)))
        if index != len(items) - 1:
            body.append("</parameter>")
    body.extend(("</function>", "</tool_call>"))
    return "\n".join(body)


class FourEnvironmentQwenActionContractTest(unittest.TestCase):

    @staticmethod
    def _policy_prompts() -> dict[str, str]:
        return {
            "webshop": build_filesystem_conversation_start(
                ActionFormat.REACT,
                surface=PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
            )[0]["value"],
            "swesmith": SWE_POLICY_SYSTEM_PROMPT,
            "literesearcher": LITERESEARCHER_SYSTEM_PROMPT,
            "openmle_fast": OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
        }

    def test_openmle_client_and_endpoint_share_one_prompt_contract(self) -> None:
        import hashlib

        self.assertEqual(
            OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
            OPENMLE_FAST_ENDPOINT_POLICY_PROMPT,
        )
        self.assertEqual(
            hashlib.sha256(
                OPENMLE_FAST_POLICY_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            OPENMLE_FAST_ENDPOINT_POLICY_PROMPT_SHA256,
        )

    def test_native_template_owns_tool_manifests(self) -> None:
        for environment, prompt in self._policy_prompts().items():
            with self.subTest(environment=environment):
                self.assertNotIn("<tools>", prompt)
                self.assertNotRegex(prompt, r'"type"\s*:\s*"function"')
                self.assertNotIn("Copy one concrete function-call example", prompt)
                self.assertNotIn("example_function_name", prompt)
                self.assertNotIn("FUNCTION_NAME", prompt)
                self.assertNotIn("ARGUMENT_NAME", prompt)

    def test_wrappers_expose_the_exact_parser_tool_schemas(self) -> None:
        webshop = object.__new__(AgentMemoryEnvClient)
        webshop.is_filesystem = True
        webshop.adapter_cls = FilesystemAgentMemoryAdapter
        self.assertEqual(
            webshop.policy_tool_schemas(), list(_WEBSHOP_QWEN_TOOL_SCHEMAS)
        )

        clarification = object.__new__(AgentMemoryEnvClient)
        clarification.is_filesystem = True
        clarification.adapter_cls = IntentClarificationFilesystemAgentMemoryAdapter
        self.assertEqual(
            clarification.policy_tool_schemas(),
            [*_WEBSHOP_QWEN_TOOL_SCHEMAS, _WEBSHOP_QWEN_ASK_TOOL_SCHEMA],
        )

        legacy = object.__new__(AgentMemoryEnvClient)
        legacy.is_filesystem = False
        legacy.adapter_cls = FilesystemAgentMemoryAdapter
        self.assertIsNone(legacy.policy_tool_schemas())

        fixed = (
            (SwesmithEnvClient, _SWE_QWEN_TOOL_SCHEMAS),
            (LiteResearcherEnvClient, _LITERESEARCHER_QWEN_TOOL_SCHEMAS),
            (OpenMLEFastEnvClient, _OPENMLE_QWEN_TOOL_SCHEMAS),
        )
        for client_type, schemas in fixed:
            with self.subTest(client=client_type.__name__):
                client = object.__new__(client_type)
                self.assertEqual(client.policy_tool_schemas(), list(schemas))

        openmle = object.__new__(OpenMLEFastEnvClient)
        shell_schema = openmle.policy_tool_schemas()[0]["function"]["parameters"]
        self.assertEqual(
            shell_schema["properties"]["timeout_ms"],
            {"type": "integer", "minimum": 1, "maximum": 20_000},
        )

        copied = webshop.policy_tool_schemas()
        assert copied is not None
        copied[0]["function"]["name"] = "mutated"
        self.assertNotEqual(
            webshop.policy_tool_schemas()[0]["function"]["name"], "mutated"
        )

    def test_first_action_lifecycle_guards_are_wrapper_owned(self) -> None:
        prompts = self._policy_prompts()
        self.assertIn(
            "when Progress is 0/6 and the observation lists approved cards but "
            "no search-result page, the next function must be search",
            prompts["webshop"],
        )
        self.assertIn(
            "Never call click with item `search`",
            prompts["webshop"],
        )
        self.assertIn(
            "when only the task description is visible and no public file has "
            "been inspected, the next function must be shell_command",
            prompts["openmle_fast"],
        )
        self.assertIn(
            "Do not call apply_patch before at least one successful inspection action",
            prompts["openmle_fast"],
        )

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

    def test_every_policy_prompt_uses_examples_without_abstract_metasyntax(self) -> None:
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
                self.assertNotIn("<parameter=ARGUMENT_NAME>", prompt)
                self.assertNotIn("<parameter=ARGUMENT_NAME>VALUE</parameter>", prompt)
                self.assertRegex(prompt, r"<parameter=[a-z][a-z0-9_]*>")
                self.assertIn("</parameter>", prompt)

    def test_webshop_add_file_example_preserves_exact_confirmed_shape(self) -> None:
        prompt = build_filesystem_conversation_start(
            ActionFormat.REACT,
            surface=PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
        )[0]["value"]
        self.assertIn("+Confirmed example attribute: example value", prompt)
        self.assertIn("The leading `+` is patch syntax", prompt)
        self.assertIn("exact full `Confirmed <field>: <value>` line", prompt)
        self.assertIn("a product title is not a field/value note", prompt)
        self.assertNotIn("+confirmed evidence", prompt)

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

    def test_all_filesystem_webshop_prompts_use_only_qwen_action_examples(self) -> None:
        for surface in sorted(FILESYSTEM_WEBSHOP_SURFACES):
            with self.subTest(surface=surface):
                prompt = build_filesystem_conversation_start(
                    ActionFormat.REACT, surface=surface
                )[0]["value"]
                self.assertIn("<tool_call>", prompt)
                self.assertNotIn("search[", prompt)
                self.assertNotIn("click[", prompt)
                self.assertNotIn("shell_command {", prompt)
                self.assertNotIn("ASK {", prompt)
                if surface == INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE:
                    self.assertIn("<function=ask>", prompt)
                    self.assertIn("<parameter=field>", prompt)

    def test_webshop_projection_adds_current_page_guidance(self) -> None:
        search_page = normalize_filesystem_webshop_policy_observation(
            "Native WebShop actions currently available:\n- search[keywords]\n"
        )
        self.assertIn("current session is on the Search page", search_page)
        self.assertIn("Use `search` exactly once", search_page)

        results_page = normalize_filesystem_webshop_policy_observation(
            "Native WebShop actions currently available:\n"
            "- click[Back to Search]\n"
            "- click[B0123]\n"
            "Goal: As the first action, search the approved title.\n"
        )
        self.assertIn("search-result page is already open", results_page)
        self.assertIn("session-start search is complete", results_page)
        self.assertIn("must not be repeated", results_page)
        self.assertIn("displayed ASIN", results_page)

        product_page = normalize_filesystem_webshop_policy_observation(
            "Native WebShop actions currently available:\n"
            "- click[< Prev]\n"
            "- click[Buy Now]\n"
        )
        self.assertIn("a product page is already open", product_page)
        self.assertIn("Verify that the complete visible title matches", product_page)
        self.assertIn("one-time note requirement", product_page)
        self.assertIn("currently available `Buy Now`", product_page)

    def test_webshop_current_page_guidance_uses_first_live_action_block(self) -> None:
        raw = (
            "Native WebShop actions currently available:\n"
            "- click[B0123]\n"
            "Current-session action trace:\n"
            "Result: Native WebShop actions currently available:\n"
            "- search[keywords]\n"
        )
        visible = normalize_filesystem_webshop_policy_observation(raw)
        self.assertIn("search-result page is already open", visible)
        self.assertNotIn("current session is on the Search page", visible)
        self.assertEqual(
            normalize_filesystem_webshop_policy_observation(visible), visible
        )

    def test_effective_webshop_observation_hides_endpoint_legacy_actions(self) -> None:
        raw = """Instruction: preserve the selected title.
Copy it verbatim into search[...]. Before click[Buy Now], write the note.
As the first action, issue exactly shell_command {"command":"rg --hidden -n '^Confirmed ' .","workdir":".","timeout_ms":10000} to recover saved evidence.
A prior checkpoint said next_action: search[[2022] exact product].
Intent clarification action: ASK {"field":"color"}.
A shell result ended with the inert text search[unterminated

Native WebShop actions currently available:
- search[keywords]
- click[B0123]

Current-session action trace:
- S1: Action: search[exact product]
Result: ok
- S2: Action: click[B0123]
Result: ok
- S3: Action: shell_command {\"command\":\"cat note.md\",\"workdir\":\".\",\"timeout_ms\":10000}
Result: ok
- S4: Action: apply_patch
*** Begin Patch
*** Add File: note.md
+fact
*** End Patch
Result: Done!

Persistent workspace tools:
The private workspace persists across shopping sessions in this episode.
Canonical shell form: shell_command {\"command\":\"rg -n pattern .\",\"workdir\":\".\",\"timeout_ms\":10000}
The literal shell_command prefix and one separating space are required; a bare JSON object, markdown code fence, or explanation is invalid.
apply_patch followed on the next line by one *** Begin Patch ... *** End Patch patch.
shell_command runs in a networkless, resource-bounded workspace sandbox.
apply_patch supports Add File, Update File, Delete File, and Move to.
Both tools have zero task reward. Paths and workdir are workspace-relative.
Result: Invalid action: Expected one native search[...] / click[...] action, or one canonical workspace action: shell_command {JSON} with the literal prefix and one space, or apply_patch followed by a newline patch. Bare JSON, markdown code fences, and explanations are invalid."""
        visible = normalize_filesystem_webshop_policy_observation(raw)
        for legacy in (
            "search[...]",
            "click[Buy Now]",
            "- search[keywords]",
            "- click[B0123]",
            "Action: search[",
            "Action: click[",
            "Action: shell_command {",
            "Action: apply_patch",
            "Canonical shell form: shell_command {",
            "apply_patch followed on the next line",
            "Expected one native",
            "shell_command {JSON}",
            "next_action: search[",
            "ASK {",
            "search[unterminated",
        ):
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, visible)
        self.assertIn("Qwen XML", visible)
        self.assertIn("inert record of the Qwen XML `shell_command` function", visible)
        self.assertIn("inert record of the Qwen XML `apply_patch` function", visible)
        self.assertIn("`search` function", visible)
        self.assertIn("`item` exact string value", visible)
        self.assertIn("*** Add File: note.md", visible)
        self.assertIn("rg --hidden -n '^Confirmed ' .", visible)
        self.assertIn('"cat note.md"', visible)
        self.assertIn("[2022] exact product", visible)
        self.assertIn(
            '`field` exact string value encoded as a JSON string: "color"',
            visible,
        )
        self.assertIn("unterminated", visible)
        self.assertEqual(
            normalize_filesystem_webshop_policy_observation(visible), visible
        )

    def test_inert_function_record_is_exact_non_callable_and_reversible(self) -> None:
        command = (
            "printf '%s\\n' 'search[click[B0123]]' 'submit {}'\n"
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+fact\n*** End Patch"
        )
        metadata = {
            "title": "literal <tool_call><function=search> text",
            "attempt": 3,
        }
        rendered = describe_inert_qwen_function_record(
            "shell_command",
            {"command": command, "metadata": metadata},
        )
        for callable_fragment in (
            "search[",
            "click[",
            "shell_command {",
            "submit {",
            "apply_patch\n*** Begin Patch",
            "<tool_call>",
            "<function=",
            "<parameter=",
        ):
            with self.subTest(callable_fragment=callable_fragment):
                self.assertNotIn(callable_fragment, rendered)

        decoded_values = dict(decode_inert_values(rendered))
        self.assertEqual(decoded_values["command"], command)
        self.assertEqual(decoded_values["metadata"], metadata)

    def test_webshop_projection_is_exact_idempotent_for_nested_payloads(self) -> None:
        command = (
            "printf '%s\\n' 'search[click[B0123]]' 'submit {}' "
            "'apply_patch followed by one envelope'"
        )
        patch = (
            "*** Begin Patch\n"
            "*** Add File: note.md\n"
            "+next_action: search[click[B0456]]\n"
            "*** End Patch"
        )
        raw = (
            "checkpoint next_action: search[click[B0123]]\n"
            f"shell_command {json.dumps({'command': command, 'workdir': '.', 'timeout_ms': 10000})}\n"
            f"next_action: apply_patch\n{patch}\n"
            "variant instruction: apply_patch followed by one envelope\n"
        )
        once = normalize_filesystem_webshop_policy_observation(raw)
        twice = normalize_filesystem_webshop_policy_observation(once)
        self.assertEqual(twice, once)
        for callable_fragment in (
            "search[",
            "click[",
            "shell_command {",
            "apply_patch\n*** Begin Patch",
            "apply_patch followed by",
        ):
            with self.subTest(callable_fragment=callable_fragment):
                self.assertNotIn(callable_fragment, once)
        self.assertEqual(
            json.loads('"search\\u005bclick\\u005bB0123]]"'),
            "search[click[B0123]]",
        )
        self.assertIn("search\\u005bclick\\u005bB0123]]", once)
        self.assertIn("submit \\u007b}", once)
        decoded_values = decode_inert_values(once)
        self.assertIn(("command", command), decoded_values)
        self.assertIn(("patch", patch), decoded_values)
        self.assertIn(("keywords", "click[B0123]"), decoded_values)

    def test_inert_function_record_preserves_marker_case_and_spacing(self) -> None:
        values = (
            'shell_command   {"command":"x"}',
            'shell_command\t{"command":"x"}',
            'APPLY_PATCH followed by one envelope',
            'Apply_Patch followed by one envelope',
        )
        for value in values:
            with self.subTest(value=value):
                rendered = describe_inert_qwen_function_record(
                    "shell_command", {"command": value}
                )
                self.assertEqual(dict(decode_inert_values(rendered))["command"], value)

    def test_webshop_projection_preserves_opaque_shell_and_patch_payloads(self) -> None:
        legacy_invalid = (
            "Invalid action: Expected one native search[...] / click[...] action, or one "
            "canonical workspace action: shell_command {JSON} with the literal prefix "
            "and one space, or apply_patch followed by a newline patch. Bare JSON, "
            "markdown code fences, and explanations are invalid."
        )
        shell_values = (
            legacy_invalid,
            'printf %s "`apply_patch` followed by one envelope"',
            "printf '%s\n' 'my_apply_patch followed by one envelope'",
            "my_`apply_patch` followed by one envelope",
        )
        for command in shell_values:
            with self.subTest(command=command):
                raw = "stdout: shell_command " + json.dumps({"command": command})
                projected = normalize_filesystem_webshop_policy_observation(raw)
                self.assertEqual(
                    dict(decode_inert_values(projected))["command"], command
                )
                self.assertEqual(
                    normalize_filesystem_webshop_policy_observation(projected),
                    projected,
                )

        patch = (
            "*** Begin Patch\n"
            "*** Add File: note.md\n"
            f"+{legacy_invalid}\n"
            "+my_apply_patch followed by one envelope\n"
            "+my_`apply_patch` followed by one envelope\n"
            "*** End Patch"
        )
        projected = normalize_filesystem_webshop_policy_observation(
            "next_action: apply_patch\n" + patch
        )
        self.assertEqual(dict(decode_inert_values(projected))["patch"], patch)
        self.assertEqual(
            normalize_filesystem_webshop_policy_observation(projected), projected
        )

    def test_webshop_projection_hides_endpoint_shell_whitespace_variants(self) -> None:
        for separator in ("\t", "   ", "\n"):
            with self.subTest(separator=repr(separator)):
                arguments = {"command": "ls", "workdir": "."}
                raw = "trace: shell_command" + separator + json.dumps(arguments)
                projected = normalize_filesystem_webshop_policy_observation(raw)
                self.assertNotEqual(projected, raw)
                self.assertNotIn("shell_command" + separator + "{", projected)
                self.assertEqual(dict(decode_inert_values(projected)), arguments)
                self.assertEqual(
                    normalize_filesystem_webshop_policy_observation(projected),
                    projected,
                )

    def test_openmle_projection_preserves_opaque_shell_and_patch_payloads(self) -> None:
        for command in (
            'printf %s "`apply_patch` followed by one envelope"',
            "printf '%s\n' 'my_apply_patch followed by one envelope'",
            "my_`apply_patch` followed by one envelope",
        ):
            with self.subTest(command=command):
                projected = normalize_openmle_policy_observation(
                    "stdout: shell_command " + json.dumps({"command": command})
                )
                self.assertEqual(
                    dict(decode_inert_values(projected))["command"], command
                )
                self.assertEqual(
                    normalize_openmle_policy_observation(projected), projected
                )

        legacy_submit = (
            "- `submit {}` grades `submission.csv` exactly once and always terminates"
        )
        patch = (
            "*** Begin Patch\n"
            "*** Add File: note.md\n"
            f"+{legacy_submit}\n"
            "+my_apply_patch followed by one envelope\n"
            "+my_`apply_patch` followed by one envelope\n"
            "*** End Patch"
        )
        projected = normalize_openmle_policy_observation(
            "next_action: apply_patch\n" + patch
        )
        self.assertEqual(dict(decode_inert_values(projected))["patch"], patch)
        self.assertEqual(normalize_openmle_policy_observation(projected), projected)

    def test_webshop_availability_describes_search_parameter_not_literal(self) -> None:
        raw = (
            "Native WebShop actions currently available:\n"
            "- search[keywords]\n"
            "- click[B0123]\n"
        )
        visible = normalize_filesystem_webshop_policy_observation(raw)
        self.assertIn("required `keywords` parameter: one non-empty string", visible)
        self.assertNotIn('available `keywords` value: "keywords"', visible)
        self.assertNotIn("search[keywords]", visible)

    def test_effective_openmle_observation_hides_endpoint_legacy_actions(self) -> None:
        raw = """# OpenMLE-fast task

## Policy actions

You have one global budget of 30 ordinary actions. Every `shell_command`, `apply_patch`, and `submit` consumes one action, including rejected or failed actions. A compound shell command is one policy action, while every managed Python start is counted separately. Use only one action per response.

- `shell_command {\"command\":\"...\",\"workdir\":\".\",\"timeout_ms\":20000}`
- `apply_patch` followed by one `*** Begin Patch` / `*** End Patch` envelope
- `submit {}` grades `submission.csv` exactly once and always terminates

`TASK.md` and `data/` are read-only."""
        visible = normalize_openmle_policy_observation(raw)
        for legacy in (
            'shell_command {"command"',
            "`apply_patch` followed by",
            "submit {}",
        ):
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, visible)
        self.assertIn("Qwen XML", visible)
        self.assertIn("`shell_command` function-call", visible)
        self.assertIn("`apply_patch` function-call", visible)
        self.assertIn("`submit` function-call", visible)
        self.assertEqual(normalize_openmle_policy_observation(visible), visible)
        self.assertIn("30 ordinary actions", visible)
        self.assertIn("`TASK.md` and `data/` are read-only", visible)

        command = (
            "python train.py && printf "
            "'shell_command {\\\"command\\\":\\\"x\\\"}; "
            "apply_patch followed by one envelope'"
        )
        patch = (
            "*** Begin Patch\n"
            "*** Update File: train.py\n"
            "+print('submit {}')\n"
            "*** End Patch"
        )
        dynamic = (
            "stdout record: "
            f"shell_command {json.dumps({'command': command, 'workdir': '.', 'timeout_ms': 20000})}\n"
            f"checkpoint next_action: apply_patch\n{patch}\n"
            'prior final action: submit {"candidate":"submission.csv"}\n'
        )
        projected = normalize_openmle_policy_observation(dynamic)
        self.assertEqual(normalize_openmle_policy_observation(projected), projected)
        for callable_fragment in (
            "shell_command {",
            "submit {",
            "apply_patch\n*** Begin Patch",
            "apply_patch followed by",
        ):
            with self.subTest(callable_fragment=callable_fragment):
                self.assertNotIn(callable_fragment, projected)
        self.assertIn("shell_command \\u007b", projected)
        self.assertIn("submit \\u007b}", projected)
        decoded_values = decode_inert_values(projected)
        self.assertIn(("command", command), decoded_values)
        self.assertIn(("patch", patch), decoded_values)
        self.assertIn(("candidate", "submission.csv"), decoded_values)

    def test_policy_clients_expose_only_deconflicted_observations(self) -> None:
        webshop = AgentMemoryEnvClient.__new__(AgentMemoryEnvClient)
        webshop.is_filesystem = True
        raw_webshop_observation = (
                "Native WebShop actions currently available:\n"
                "- search[keywords]\n"
                "- click[Buy Now]\n\n"
                "Persistent workspace tools:\n"
                "The private workspace persists across shopping sessions in this episode.\n"
                'Canonical shell form: shell_command {"command":"rg -n pattern .","workdir":".","timeout_ms":10000}\n'
                "The literal shell_command prefix and one separating space are required; a bare JSON object, markdown code fence, or explanation is invalid.\n"
                "apply_patch followed on the next line by one *** Begin Patch ... *** End Patch patch.\n"
                "shell_command runs in a networkless, resource-bounded workspace sandbox.\n"
                "apply_patch supports Add File, Update File, Delete File, and Move to.\n"
                "Both tools have zero task reward. Paths and workdir are workspace-relative."
        )
        webshop.info = {"observation": raw_webshop_observation}
        webshop_visible = webshop.observe()
        self.assertIn("Qwen XML", webshop_visible)
        self.assertNotIn("search[keywords]", webshop_visible)
        self.assertNotIn("Canonical shell form", webshop_visible)
        self.assertEqual(webshop.info["observation"], raw_webshop_observation)

        openmle = OpenMLEFastEnvClient.__new__(OpenMLEFastEnvClient)
        openmle.info = {
            "observation": (
                '- `shell_command {"command":"...","workdir":".","timeout_ms":20000}`\n'
                "- `apply_patch` followed by one `*** Begin Patch` / `*** End Patch` envelope\n"
                "- `submit {}` grades `submission.csv` exactly once and always terminates"
            )
        }
        openmle_visible = openmle.observe()
        self.assertIn("Qwen XML", openmle_visible)
        self.assertNotIn('shell_command {"command"', openmle_visible)
        self.assertNotIn("submit {}", openmle_visible)

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
            normalize_filesystem_webshop_policy_action,
            qwen_call("ask", field="color"),
            'ASK {"field": "color"}',
            allow_ask=True,
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

    def test_upstream_native_truncated_final_parameter_translates_everywhere(self) -> None:
        self.assert_valid_translation(
            normalize_filesystem_webshop_policy_action,
            qwen_call_with_truncated_final_parameter("search", keywords="red mug"),
            "search[red mug]",
        )
        self.assert_valid_translation(
            normalize_swesmith_policy_action,
            qwen_call_with_truncated_final_parameter(
                "shell_command", command="pwd", workdir="."
            ),
            'shell_command {"command":"pwd","workdir":"."}',
        )
        self.assert_valid_translation(
            normalize_literesearcher_policy_action,
            qwen_call_with_truncated_final_parameter(
                "visit",
                url="https://literesearcher.local/page/one",
                goal="find evidence",
                page=1,
            ),
            "<tool_call>\n"
            "<function=visit>\n"
            "<parameter=url>\n"
            '"https://literesearcher.local/page/one"\n'
            "</parameter>\n"
            "<parameter=goal>\n"
            '"find evidence"\n'
            "</parameter>\n"
            "<parameter=page>\n"
            "1\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
        )
        self.assert_valid_translation(
            _normalize_openmle_policy_action,
            qwen_call_with_truncated_final_parameter(
                "shell_command", command="pwd", workdir="."
            ),
            'shell_command {"command": "pwd", "workdir": "."}',
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

    def test_plain_prose_around_one_native_call_is_tolerated_everywhere(self) -> None:
        call = qwen_call("shell_command", command="pwd", workdir=".")
        for normalizer in (
            normalize_filesystem_webshop_policy_action,
            normalize_swesmith_policy_action,
            normalize_literesearcher_policy_action,
            _normalize_openmle_policy_action,
        ):
            for action in ("I will inspect first.\n" + call, call + "\nDone."):
                with self.subTest(environment=normalizer.__module__, action=action[:24]):
                    submitted, evidence = normalizer(action)
                    self.assertTrue(submitted.startswith("shell_command "))
                    self.assertIs(evidence["tool_parser_normalized"], True)

    def test_think_and_multiple_calls_are_rejected_everywhere(self) -> None:
        call = qwen_call("shell_command", command="pwd", workdir=".")
        malformed = ("<think>inspect</think>\n" + call, call + "\n" + call)
        for normalizer in (
            normalize_filesystem_webshop_policy_action,
            normalize_swesmith_policy_action,
            normalize_literesearcher_policy_action,
            _normalize_openmle_policy_action,
        ):
            for action in malformed:
                with self.subTest(environment=normalizer.__module__, action=action[:24]):
                    self.assert_rejected(normalizer, action)

    def test_unknown_tool_and_missing_required_argument_are_rejected_everywhere(self) -> None:
        malformed = (
            qwen_call("unknown_tool", command="pwd"),
            qwen_call("shell_command", workdir="."),
        )
        for normalizer in (
            normalize_filesystem_webshop_policy_action,
            normalize_swesmith_policy_action,
            normalize_literesearcher_policy_action,
            _normalize_openmle_policy_action,
        ):
            for action in malformed:
                with self.subTest(environment=normalizer.__module__, action=action):
                    self.assert_rejected(normalizer, action)

    def test_truncated_final_parameter_does_not_bypass_route_validation(self) -> None:
        for action in (
            qwen_call_with_truncated_final_parameter(
                "shell_command", command="pwd", workdir="/workspace"
            ),
            qwen_call_with_truncated_final_parameter(
                "shell_command", command="pwd", timeout_ms=20001
            ),
            qwen_call_with_truncated_final_parameter("submit", reason="done"),
        ):
            with self.subTest(action=action):
                self.assert_rejected(_normalize_openmle_policy_action, action)

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
