from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentenv_openmle_fast.actions import (
    OpenMLEFastActionError,
    apply_workspace_patch,
    parse_policy_action,
)


class OpenMLEFastActionsTest(unittest.TestCase):
    def test_accepts_only_three_exact_tools(self) -> None:
        shell = parse_policy_action(
            'shell_command {"command":"python solution.py","timeout_ms":15000}'
        )
        self.assertEqual(shell.kind, "shell_command")
        self.assertEqual(shell.arguments["command"], "python solution.py")
        patch = parse_policy_action(
            "apply_patch\n*** Begin Patch\n*** Add File: solution.py\n+print(1)\n*** End Patch"
        )
        self.assertEqual(patch.kind, "apply_patch")
        self.assertEqual(parse_policy_action("submit").kind, "submit")
        self.assertEqual(parse_policy_action("submit {}").kind, "submit")
        runtime_shell = parse_policy_action(
            'shell_command {"command":"pwd","workdir":".","timeout_ms":20000}'
        )
        self.assertEqual(runtime_shell.kind, "shell_command")
        self.assertNotIn("workdir", runtime_shell.arguments)
        for raw in (
            "execute python solution.py",
            "submit now",
            'shell_command {"command":"pwd"}\nsubmit',
            'shell_command {"command":"pwd","workdir":"/tmp"}',
            "final",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_policy_action(raw).kind, "parser_error")

    def test_rejects_oversize_nul_and_timeout_above_frozen_limit(self) -> None:
        self.assertEqual(
            parse_policy_action('shell_command {"command":"a\\u0000b"}').kind,
            "parser_error",
        )
        self.assertEqual(
            parse_policy_action(
                'shell_command {"command":"pwd","timeout_ms":20001}'
            ).kind,
            "parser_error",
        )
        self.assertEqual(
            parse_policy_action(
                'shell_command {"command":"' + ("x" * (32 * 1024 + 1)) + '"}'
            ).kind,
            "parser_error",
        )
        self.assertEqual(
            parse_policy_action('shell_command {"command":"\ud800"}').kind,
            "parser_error",
        )

    def test_qwen3_xml_reuses_frozen_action_validation(self) -> None:
        for raw in (
            """<tool_call>
<function=shell_command>
<parameter=command>pwd</parameter>
<parameter=workdir>/tmp</parameter>
</function>
</tool_call>""",
            """<tool_call>
<function=shell_command>
<parameter=command>pwd</parameter>
<parameter=timeout_ms>20001</parameter>
</function>
</tool_call>""",
            """<tool_call>
<function=apply_patch>
<parameter=patch>not a patch</parameter>
</function>
</tool_call>""",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_policy_action(raw).kind, "parser_error")

    def test_patch_outputs_are_accessible_to_fresh_sandbox_uid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="openmle-patch-mode-test-") as raw:
            workspace = Path(raw)
            apply_workspace_patch(
                workspace,
                "*** Begin Patch\n*** Add File: src/solution.py\n"
                "+print('ok')\n*** End Patch",
            )
            self.assertEqual(
                (workspace / "src" / "solution.py").stat().st_mode & 0o777,
                0o666,
            )
            self.assertEqual((workspace / "src").stat().st_mode & 0o777, 0o777)

    def test_patch_is_workspace_scoped_and_protects_task_data(self) -> None:
        with tempfile.TemporaryDirectory(prefix="openmle-patch-test-") as raw:
            workspace = Path(raw)
            (workspace / "TASK.md").write_text("contract", encoding="utf-8")
            (workspace / "data").mkdir()
            (workspace / "data/input.csv").write_text("x\n", encoding="utf-8")
            result = apply_workspace_patch(
                workspace,
                "*** Begin Patch\n*** Add File: solution.py\n+print('ok')\n*** End Patch",
            )
            self.assertEqual(result.changed_paths, ("solution.py",))
            self.assertEqual(
                (workspace / "solution.py").read_text(encoding="utf-8"),
                "print('ok')\n",
            )
            for patch in (
                "*** Begin Patch\n*** Add File: ../escape.py\n+x\n*** End Patch",
                "*** Begin Patch\n*** Update File: TASK.md\n@@\n-contract\n+changed\n*** End Patch",
                "*** Begin Patch\n*** Delete File: data/input.csv\n*** End Patch",
                "*** Begin Patch\n*** Add File: Data/escape.csv\n+x\n*** End Patch",
                "*** Begin Patch\n*** Update File: task.MD\n@@\n-contract\n+changed\n*** End Patch",
            ):
                with (
                    self.subTest(patch=patch),
                    self.assertRaises(OpenMLEFastActionError),
                ):
                    apply_workspace_patch(workspace, patch)


    def test_accepts_qwen3_xml_action_envelope(self) -> None:
        shell = parse_policy_action(
            """<tool_call>
<function=shell_command>
<parameter=command>
cat .agent_memory/CONTINUATION.md
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
20000
</parameter>
</function>
</tool_call>"""
        )
        self.assertEqual(shell.kind, "shell_command")
        self.assertEqual(
            shell.arguments,
            {
                "command": "cat .agent_memory/CONTINUATION.md",
                "timeout_ms": 20000,
            },
        )

        patch = parse_policy_action(
            """<tool_call>
<function=apply_patch>
<parameter=patch>
*** Begin Patch
*** Add File: train.py
+print(1)
*** End Patch
</parameter>
</function>
</tool_call>"""
        )
        self.assertEqual(patch.kind, "apply_patch")
        self.assertIn("*** Add File: train.py", patch.patch)

        submit = parse_policy_action(
            """<tool_call>
<function=submit>
</function>
</tool_call>"""
        )
        self.assertEqual(submit.kind, "submit")

    def test_qwen3_xml_action_remains_exactly_one_strict_call(self) -> None:
        malformed = (
            "<tool_call><function=shell_command>"
            "<parameter=command>pwd</parameter>"
            "<parameter=timeout_ms>20000</parameter>"
            "<parameter=timeout_ms>20000</parameter>"
            "</function></tool_call>"
        )
        self.assertEqual(parse_policy_action(malformed).kind, "parser_error")
        self.assertEqual(
            parse_policy_action(
                "<tool_call><function=submit></function></tool_call> trailing"
            ).kind,
            "parser_error",
        )
        self.assertEqual(
            parse_policy_action(
                "<tool_call><function=submit>"
                "<parameter=unexpected>x</parameter>"
                "</function></tool_call>"
            ).kind,
            "parser_error",
        )


    def test_qwen3_xml_rejects_endpoint_only_name_aliases(self) -> None:
        for alias in (
            "<tool_call><function=SUBMIT></function></tool_call>",
            "<tool_call><function=shell_command><parameter=Command>pwd</parameter></function></tool_call>",
            "<tool_call><function=shell_command><parameter='command'>pwd</parameter></function></tool_call>",
        ):
            with self.subTest(alias=alias):
                self.assertEqual(parse_policy_action(alias).kind, "parser_error")

    def test_qwen3_xml_preserves_native_parameter_edge_whitespace(self) -> None:
        parsed = parse_policy_action(
            "<tool_call><function=shell_command>"
            "<parameter=command>\n  printf x  \n</parameter>"
            "<parameter=workdir>\n.\n</parameter>"
            "</function></tool_call>"
        )
        self.assertEqual(parsed.kind, "shell_command")
        self.assertEqual(parsed.arguments["command"], "  printf x  ")

if __name__ == "__main__":
    unittest.main()
