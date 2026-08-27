from __future__ import annotations

import ast
from pathlib import Path
import unittest


ENV_DIR = Path(__file__).resolve().parents[1] / "agentenv" / "envs"
FILESYSTEM_CHECKPOINT_PATH = ENV_DIR / "filesystem_checkpoint.py"
SWESMITH_PATH = ENV_DIR / "swesmith.py"


def extract_static_string_assignments() -> dict[str, str]:
    values: dict[str, str] = {}

    def resolve(node: ast.expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return resolve(node.left) * resolve(node.right)
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for item in node.values:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    parts.append(item.value)
                elif isinstance(item, ast.FormattedValue):
                    parts.append(str(resolve(item.value)))
                else:
                    raise ValueError(
                        f"unsupported f-string component: {ast.dump(item)}"
                    )
            return "".join(parts)
        raise ValueError(f"unsupported static string expression: {ast.dump(node)}")

    for source_path in (FILESYSTEM_CHECKPOINT_PATH, SWESMITH_PATH):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
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
    return values


class SwesmithJointMemoryPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        values = extract_static_string_assignments()
        self.prompt = values["SWE_POLICY_SYSTEM_PROMPT"]
        self.compaction = values["SWE_CONTEXT_COMPACTION_REQUEST"]
        self.continuation_marker = values["SWE_POLICY_CONTINUATION_MARKER"]
        self.memory_contract = values["SWE_MEMORY_CONTRACT"]

    def test_contract_has_a_distinct_joint_memory_identity(self) -> None:
        self.assertEqual(
            self.memory_contract,
            "policy_filesystem_checkpoint_then_client_replace_v2",
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
            "overwrite `.agent_memory/CONTINUATION.md`",
            "read the checkpoint with a normal command",
            "Writing or reading a note has no separate reward",
        ):
            self.assertIn(fragment, self.prompt)

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

    def test_prompt_includes_a_separate_turn_write_then_read_example(self) -> None:
        self.assertIn(">> .agent_memory/debugging.md", self.prompt)
        self.assertIn("rg -n 'hypothesis|evidence|next check' .agent_memory", self.prompt)
        self.assertIn("followed in a later action", self.prompt)
        self.assertIn("only a syntax illustration", self.prompt)

    def test_compaction_is_an_executed_checkpoint_write_then_later_read(self) -> None:
        for fragment in (
            "exactly one normal executable shell_command or apply_patch action",
            ".agent_memory/CONTINUATION.md",
            "executed normally and consumes one policy-action step",
            "removed only after the environment verifies this exact file write",
        ):
            self.assertIn(fragment, self.compaction)
        self.assertIn(
            "Use the next normal action to read that file",
            self.continuation_marker,
        )
        for forbidden in (
            "will not be sent to the environment",
            "Do not claim that this response executed a shell command",
        ):
            self.assertNotIn(forbidden, self.compaction)


if __name__ == "__main__":
    unittest.main()
