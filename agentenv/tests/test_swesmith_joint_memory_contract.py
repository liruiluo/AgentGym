from __future__ import annotations

import ast
from pathlib import Path
import unittest


SWESMITH_PATH = Path(__file__).resolve().parents[1] / "agentenv" / "envs" / "swesmith.py"


def extract_static_string_assignments() -> dict[str, str]:
    tree = ast.parse(SWESMITH_PATH.read_text(encoding="utf-8"))
    values: dict[str, str] = {}

    def resolve(node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        raise ValueError(f"unsupported static string expression: {ast.dump(node)}")

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
        self.memory_contract = values["SWE_MEMORY_CONTRACT"]

    def test_contract_has_a_distinct_joint_memory_identity(self) -> None:
        self.assertEqual(
            self.memory_contract,
            "policy_compaction_plus_optional_durable_filesystem_v1",
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
            "Before context compaction, make sure",
            "rediscover and read the notes",
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

    def test_prompt_preserves_the_upstream_repair_workflow(self) -> None:
        for fragment in (
            "# Required repair workflow",
            "Reproduce the reported behavior",
            "Modify only the necessary non-test source files",
            "Rerun the reproduction",
            "Run relevant existing tests and check edge cases",
            "submit immediately instead of continuing to inspect",
        ):
            self.assertIn(fragment, self.prompt)

    def test_compaction_is_short_state_and_locator_not_file_execution(self) -> None:
        for fragment in (
            "Keep this response short",
            "immediate objective",
            "path/search key of any durable notes you already wrote",
            "not be sent to the environment",
            "Do not claim that this response executed a shell command",
        ):
            self.assertIn(fragment, self.compaction)


if __name__ == "__main__":
    unittest.main()
