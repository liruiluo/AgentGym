from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import tempfile
import unittest


ENV_PATH = Path(__file__).resolve().parents[1] / "agentenv" / "envs"
FILESYSTEM_CHECKPOINT_PATH = ENV_PATH / "filesystem_checkpoint.py"
SWESMITH_PATH = ENV_PATH / "swesmith.py"


def extract_static_string_assignments() -> dict[str, str]:
    values: dict[str, str | int] = {}

    def resolve(node: ast.expr) -> str | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return resolve(node.left) * resolve(node.right)
        if isinstance(node, ast.JoinedStr):
            pieces: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    pieces.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    pieces.append(str(resolve(value.value)))
                else:
                    raise ValueError(
                        f"unsupported f-string expression: {ast.dump(value)}"
                    )
            return "".join(pieces)
        raise ValueError(f"unsupported static string expression: {ast.dump(node)}")

    for source in (FILESYSTEM_CHECKPOINT_PATH, SWESMITH_PATH):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = resolve(node.value)
            except ValueError:
                continue
    return {key: value for key, value in values.items() if isinstance(value, str)}


class SwesmithJointMemoryPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        values = extract_static_string_assignments()
        self.prompt = values["SWE_POLICY_SYSTEM_PROMPT"]
        self.compaction = values["SWE_CONTEXT_COMPACTION_REQUEST"]
        self.checkpoint_example = values["SWE_CHECKPOINT_SHELL_SAFE_EXAMPLE"]
        self.continuation_marker = values["SWE_POLICY_CONTINUATION_MARKER"]
        self.exact_read = values["SWE_CHECKPOINT_READ_ACTION"]
        self.memory_contract = values["SWE_MEMORY_CONTRACT"]

    def test_contract_has_a_distinct_joint_memory_identity(self) -> None:
        self.assertEqual(
            self.memory_contract,
            "policy_filesystem_checkpoint_then_client_replace_v3",
        )

    def test_prompt_exposes_optional_durable_debugging_notes(self) -> None:
        for fragment in (
            "# Durable debugging notes",
            "maintain a concise evidence ledger incrementally",
            "instead of waiting for the compaction request",
            "a hypothesis is introduced or ruled out",
            "a root cause or partial fix is verified",
            "hypotheses, commands or tests already tried",
            "failed approaches and why they failed",
            "no filename or note format is prescribed",
            "a short task may not need notes at all",
            "Before context compaction, keep detailed evidence",
            "After replacement, read the checkpoint",
            "does not replace source files",
            "Writing or reading a note has no separate reward",
        ):
            self.assertIn(fragment, self.prompt)

    def test_prompt_requires_action_only_and_upstream_submission_sentinel(self) -> None:
        for fragment in (
            "response channel is an action parser, not a chat channel",
            "starting at byte zero",
            "Never prefix an action with narration",
            "workdir is relative to /testbed",
            "never `/testbed` or `./testbed`",
            "Never submit a plain-text final response",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ):
            self.assertIn(fragment, self.prompt)
        self.assertNotIn("a plain final response may summarize", self.prompt)

    def test_prompt_uses_qwen_xml_for_every_ordinary_tool_example(self) -> None:
        for fragment in (
            "output exactly one Qwen XML tool call",
            "<tool_call>\n<function=shell_command>\n<parameter=command>\n",
            "<function=apply_patch>\n<parameter=patch>\n",
            "Start at byte zero with <tool_call>",
            "Do not use the bare shell_command JSON form",
        ):
            self.assertIn(fragment, self.prompt)
        for forbidden in (
            'shell_command {"command"',
            "Start at byte zero with shell_command or apply_patch",
            "no XML tags",
        ):
            self.assertNotIn(forbidden, self.prompt)

    def test_prompt_uses_only_codex_general_tools_for_memory(self) -> None:
        self.assertIn("shell_command", self.prompt)
        self.assertIn("apply_patch", self.prompt)
        for forbidden in (
            "memory_add",
            "memory_search",
            "memory_update",
            "memory_delete",
            "ADD requires",
            "RETRIEVE requires",
        ):
            self.assertNotIn(forbidden, self.prompt)

    def test_prompt_includes_a_shell_safe_separate_turn_note_contract(self) -> None:
        self.assertIn(
            "quoted heredoc inside one shell_command command parameter",
            self.prompt,
        )
        self.assertIn("rg -n 'hypothesis|evidence|next check' .agent_memory", self.prompt)
        self.assertIn("only a syntax illustration", self.prompt)
        self.assertNotIn("printf '%s", self.prompt)

    def test_checkpoint_control_turns_are_exact_and_cannot_edit_task_source(self) -> None:
        self.assertIn(self.checkpoint_example, self.compaction)
        self.assertIn("on this turn only, overwrite the checkpoint", self.compaction)
        self.assertIn("do not inspect, read, test, edit source, or write another path", self.compaction)
        self.assertIn(self.exact_read, self.continuation_marker)
        self.assertIn("until the required checkpoint read succeeds", self.continuation_marker)
        source = SWESMITH_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            source.count(
                "continuation_marker=SWE_POLICY_CONTINUATION_MARKER"
            ),
            2,
        )

    def test_compaction_is_a_real_bounded_checkpoint_write(self) -> None:
        for fragment in (
            "exactly one normal executable shell_command or apply_patch action",
            ".agent_memory/CONTINUATION.md",
            "at most 8192 bytes",
            "executed normally and consumes one policy-action step",
            "removed only after the environment verifies this exact file write",
            "reserved `.agent_memory` directory already exists",
        ):
            self.assertIn(fragment, self.compaction)
        for forbidden in (
            "not be sent to the environment",
            "Do not claim that this response executed a shell command",
        ):
            self.assertNotIn(forbidden, self.compaction)

    def test_compaction_requires_immediate_shell_safe_overwrite(self) -> None:
        for fragment in (
            "If the repair is already complete, submit with the normal terminal sentinel",
            "Otherwise, on this turn only, overwrite the checkpoint",
            "do not inspect, read, test, edit source, or write another path",
            "Qwen XML shell action",
            "cat > .agent_memory/CONTINUATION.md <<'AGENT_MEMORY_EOF'",
            "do not use printf/echo for checkpoint text",
        ):
            self.assertIn(fragment, self.compaction)
        self.assertNotIn("do not submit", self.compaction)
        self.assertNotIn("printf '%s", self.compaction)

    def test_checkpoint_example_is_valid_xml_and_executes_multiline_content(self) -> None:
        self.assertTrue(self.checkpoint_example.startswith("<tool_call>\n"))
        self.assertTrue(self.checkpoint_example.endswith("\n</tool_call>"))
        command = _qwen_parameter(self.checkpoint_example, "command")
        workdir = _qwen_parameter(self.checkpoint_example, "workdir")
        self.assertEqual(workdir, ".")
        self.assertIn("\n", command)
        self.assertIn("user's task", command)
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir) / ".agent_memory"
            memory_dir.mkdir()
            result = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoint = memory_dir / "CONTINUATION.md"
            self.assertTrue(checkpoint.is_file())
            content = checkpoint.read_text()
            self.assertIn("user's task", content)
            self.assertIn("next: <concrete action>", content)

    def test_shell_safe_shape_preserves_apostrophe_quotes_multiline_and_paths(self) -> None:
        command = (
            "cat > .agent_memory/CONTINUATION.md <<'AGENT_MEMORY_EOF'\n"
            "objective: user's \"quoted\" task\n"
            "evidence: source says $HOME and `pwd` literally\n"
            "paths: src/pkg/example.py; tests/test_example.py\n"
            "next: run the focused test\n"
            "AGENT_MEMORY_EOF"
        )
        action = (
            "<tool_call>\n"
            "<function=shell_command>\n"
            "<parameter=command>\n"
            + command
            + "\n</parameter>\n"
            "<parameter=workdir>\n"
            ".\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        decoded_command = _qwen_parameter(action, "command")
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".agent_memory").mkdir()
            result = subprocess.run(
                ["/bin/bash", "-c", decoded_command],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            content = (Path(tmpdir) / ".agent_memory/CONTINUATION.md").read_text()
            self.assertIn('user\'s "quoted" task', content)
            self.assertIn("$HOME and `pwd` literally", content)
            self.assertIn("src/pkg/example.py", content)


def _qwen_parameter(action: str, name: str) -> str:
    start = f"<parameter={name}>\n"
    end = "\n</parameter>"
    self_start = action.index(start) + len(start)
    self_end = action.index(end, self_start)
    return action[self_start:self_end]


if __name__ == "__main__":
    unittest.main()
