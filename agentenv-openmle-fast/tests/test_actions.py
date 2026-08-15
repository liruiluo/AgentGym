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


if __name__ == "__main__":
    unittest.main()
