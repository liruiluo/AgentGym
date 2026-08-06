from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.persistent_workspace import (
    WORKSPACE_STATE_SCHEMA,
    WORKSPACE_TOOL_CONTRACT,
    WORKSPACE_TOOL_NAMES,
    PersistentWorkspace,
    WorkspaceActionError,
    WorkspaceLimits,
    parse_workspace_action,
)
from tests.workspace_test_support import InProcessTestShellSandbox


def shell_action(command: str, *, workdir: str = ".", timeout_ms: int = 10_000) -> str:
    return "shell_command " + json.dumps(
        {"command": command, "workdir": workdir, "timeout_ms": timeout_ms},
        separators=(",", ":"),
    )


class PersistentWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.parent = Path(self.temporary.name)
        self.workspace = PersistentWorkspace(
            "test-env",
            shell_sandbox=InProcessTestShellSandbox(),
            root_parent=self.parent,
        )
        self.workspace.reset_episode("episode-1")

    def tearDown(self) -> None:
        self.workspace.close()
        self.temporary.cleanup()

    def apply(self, action: str, *, step: int = 1, phase: int = 0):
        result = self.workspace.apply(
            action,
            env_step=step,
            phase_index=phase,
        )
        self.assertIsNotNone(result)
        return result

    def test_contract_exposes_only_canonical_codex_tools(self) -> None:
        self.assertEqual(WORKSPACE_TOOL_NAMES, ("shell_command", "apply_patch"))
        self.assertEqual(
            WORKSPACE_TOOL_CONTRACT,
            "codex_shell_command_apply_patch_v1",
        )
        rendered = self.workspace.render_contract()
        self.assertIn('shell_command {"command":', rendered)
        self.assertIn("apply_patch", rendered)
        for stale in ("Read {", "Write {", "Edit {", "Grep {", "Glob {"):
            self.assertNotIn(stale, rendered)

    def test_action_parser_is_typed_and_fail_closed(self) -> None:
        parsed = parse_workspace_action(
            'shell_command {"command":"pwd","workdir":".","timeout_ms":10000}'
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tool_name, "shell_command")
        self.assertEqual(parsed.arguments["command"], "pwd")

        parsed = parse_workspace_action(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+x\n*** End Patch"
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tool_name, "apply_patch")
        self.assertIsNone(parsed.arguments)

        self.assertIsNone(parse_workspace_action("search[desk lamp]"))
        for invalid in (
            "Shell_Command {}",
            "shell_command []",
            "shell_command {bad json}",
            "apply_patch *** Begin Patch",
            'Read {"path":"note.md"}',
        ):
            with self.subTest(invalid=invalid):
                if invalid.startswith(("Shell_Command", "shell_command", "apply_patch")):
                    with self.assertRaises(WorkspaceActionError):
                        parse_workspace_action(invalid)
                else:
                    self.assertIsNone(parse_workspace_action(invalid))

    def test_shell_write_then_later_read_persists_and_is_audited(self) -> None:
        write = self.apply(
            shell_action(
                "mkdir -p .agent_memory && "
                "printf '%s\\n' 'selected finish: black' > .agent_memory/MEMORY.md"
            )
        )
        self.assertEqual(write.op, "SHELL_COMMAND")
        self.assertEqual(
            write.workspace_diff["added"][0]["path"],
            ".agent_memory/MEMORY.md",
        )
        self.assertEqual(write.tool_op["exit_code"], 0)
        self.assertNotIn(str(self.workspace.host_root), write.message)

        read = self.apply(
            shell_action("cat MEMORY.md", workdir=".agent_memory"),
            step=2,
            phase=1,
        )
        self.assertIn("selected finish: black", read.message)
        self.assertEqual(read.workspace_diff["added"], [])
        self.assertEqual(read.workspace_diff["modified"], [])
        self.assertEqual(read.tool_op["phase_index"], 1)
        self.assertEqual(
            self.workspace.audit_events[0]["workspace_tree_sha256_after"],
            self.workspace.audit_events[1]["workspace_tree_sha256_before"],
        )

    def test_shell_has_no_semantic_command_allowlist(self) -> None:
        result = self.apply(
            shell_action(
                "for value in alpha beta; do printf '%s\\n' \"$value\"; done | "
                "while read value; do printf '[%s]' \"$value\"; done"
            )
        )
        self.assertIn("[alpha][beta]", result.message)
        self.assertEqual(result.tool_op["exit_code"], 0)

    def test_shell_nonzero_exit_is_a_valid_zero_mutation_event(self) -> None:
        before = self.workspace.snapshot()
        result = self.apply(shell_action("printf problem >&2; exit 7"))
        self.assertEqual(result.tool_op["exit_code"], 7)
        self.assertIn("[stderr]", result.message)
        self.assertEqual(before, self.workspace.snapshot())
        self.assertEqual(len(self.workspace.audit_events), 1)

    def test_shell_rejects_extra_fields_bad_workdir_and_timeout(self) -> None:
        bad_actions = (
            'shell_command {"command":"pwd","env":{}}',
            shell_action("pwd", workdir="../outside"),
            shell_action("pwd", timeout_ms=30_001),
        )
        for action in bad_actions:
            with self.subTest(action=action), self.assertRaises(WorkspaceActionError):
                self.workspace.apply(action, env_step=1, phase_index=0)
        self.assertEqual(self.workspace.snapshot()["file_count"], 0)
        self.assertEqual(self.workspace.audit_events, ())

    def test_apply_patch_lifecycle_is_transactional_and_audited(self) -> None:
        add = self.apply(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Add File: .agent_memory/MEMORY.md\n"
            "+finish=black\n"
            "*** End Patch"
        )
        self.assertEqual(add.op, "APPLY_PATCH")
        self.assertEqual(add.tool_op["added_paths"], [".agent_memory/MEMORY.md"])
        self.assertEqual(add.message, "Done!")

        update = self.apply(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: .agent_memory/MEMORY.md\n"
            "@@\n"
            "-finish=black\n"
            "+finish=gray\n"
            "*** End Patch",
            step=2,
        )
        self.assertEqual(
            update.workspace_diff["modified"][0]["after"]["path"],
            ".agent_memory/MEMORY.md",
        )

        move = self.apply(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: .agent_memory/MEMORY.md\n"
            "*** Move to: .agent_memory/preferences.md\n"
            "@@\n"
            " finish=gray\n"
            "*** End Patch",
            step=3,
        )
        self.assertEqual(move.tool_op["deleted_paths"], [".agent_memory/MEMORY.md"])
        self.assertEqual(move.tool_op["added_paths"], [".agent_memory/preferences.md"])

        delete = self.apply(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Delete File: .agent_memory/preferences.md\n"
            "*** End Patch",
            step=4,
        )
        self.assertEqual(delete.tool_op["deleted_paths"], [".agent_memory/preferences.md"])
        self.assertEqual(self.workspace.snapshot()["file_count"], 0)
        self.assertEqual(
            [event["op"] for event in self.workspace.audit_events],
            ["APPLY_PATCH"] * 4,
        )

    def test_failed_multi_file_patch_rolls_back_every_operation(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: one.md\n+one\n*** End Patch"
        )
        before = self.workspace.snapshot()
        with self.assertRaisesRegex(WorkspaceActionError, "context was not found"):
            self.workspace.apply(
                "apply_patch\n"
                "*** Begin Patch\n"
                "*** Add File: two.md\n"
                "+two\n"
                "*** Update File: one.md\n"
                "@@\n"
                "-missing\n"
                "+changed\n"
                "*** End Patch",
                env_step=2,
                phase_index=0,
            )
        self.assertEqual(before, self.workspace.snapshot())
        self.assertFalse((self.workspace.host_root / "two.md").exists())

    def test_patch_quota_failure_rolls_back(self) -> None:
        workspace = PersistentWorkspace(
            "quota",
            shell_sandbox=InProcessTestShellSandbox(),
            root_parent=self.parent,
            limits=WorkspaceLimits(max_files=1, max_directories=1, max_file_bytes=8, max_total_bytes=8),
        )
        workspace.reset_episode("quota-episode")
        try:
            workspace.apply(
                "apply_patch\n*** Begin Patch\n*** Add File: one.md\n+1234567\n*** End Patch",
                env_step=1,
                phase_index=0,
            )
            before = workspace.snapshot()
            with self.assertRaises(WorkspaceActionError):
                workspace.apply(
                    "apply_patch\n*** Begin Patch\n*** Add File: two.md\n+x\n*** End Patch",
                    env_step=2,
                    phase_index=0,
                )
            self.assertEqual(before, workspace.snapshot())
        finally:
            workspace.close()

    def test_reset_and_close_remove_episode_workspace(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+old\n*** End Patch"
        )
        old_root = self.workspace.host_root
        self.workspace.reset_episode("episode-2")
        self.assertFalse(old_root.exists())
        self.assertEqual(self.workspace.snapshot()["file_count"], 0)
        current_root = self.workspace.host_root
        self.workspace.close()
        self.assertFalse(current_root.exists())

    def test_harness_seed_files_are_ordinary_files_with_separate_provenance(self) -> None:
        manifest = self.workspace.install_seed_files(
            {
                ".agent_memory/inbox/profile-02.md": "other customer: gray\n",
                ".agent_memory/inbox/profile-01.md": "old preference: black\n",
            },
            source_label="distractor_orbit:7",
        )
        self.assertEqual(
            [item["path"] for item in manifest["files"]],
            [
                ".agent_memory/inbox/profile-01.md",
                ".agent_memory/inbox/profile-02.md",
            ],
        )
        self.assertEqual(self.workspace.audit_events, ())
        self.assertEqual(self.workspace.snapshot()["file_count"], 2)
        provenance = self.workspace.provenance_summary
        self.assertTrue(provenance["contains_harness_seed"])
        self.assertFalse(provenance["policy_authored"])
        self.assertEqual(provenance["seed_file_count"], 2)
        self.assertEqual(len(provenance["unchanged_seed_paths"]), 2)

        result = self.apply(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Add File: .agent_memory/MEMORY.md\n"
            "+current preference: green\n"
            "*** End Patch"
        )
        self.assertEqual(result.op, "APPLY_PATCH")
        provenance = self.workspace.provenance_summary
        self.assertTrue(provenance["policy_authored"])
        self.assertEqual(
            provenance["policy_created_paths"],
            [".agent_memory/MEMORY.md"],
        )

    def test_seed_state_survives_export_and_intervention_without_becoming_policy_seed(self) -> None:
        self.workspace.install_seed_files(
            {"notes/distractor.md": "archived preference: gray\n"},
            source_label="distractor_task:clean-pair",
        )
        self.apply(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Add File: notes/current.md\n"
            "+current preference: black\n"
            "*** End Patch"
        )
        state = self.workspace.export_state()
        self.assertEqual(state["schema"], WORKSPACE_STATE_SCHEMA)
        self.assertEqual(
            state["seed_manifest"]["manifest_sha256"],
            self.workspace.seed_manifest["manifest_sha256"],
        )

        target = PersistentWorkspace(
            "seed-target",
            shell_sandbox=InProcessTestShellSandbox(),
            root_parent=self.parent,
        )
        try:
            target.reset_episode("target-episode")
            target.install_causal_intervention("correct", state=state)
            self.assertEqual(target.seed_manifest, state["seed_manifest"])
            self.assertEqual(target.audit_events, ())
            provenance = target.provenance_summary
            self.assertTrue(provenance["contains_harness_seed"])
            self.assertTrue(provenance["policy_authored"])
            self.assertEqual(
                provenance["policy_created_paths"],
                ["notes/current.md"],
            )
        finally:
            target.close()

    def test_seed_install_is_one_shot_before_policy_and_blank_clears_provenance(self) -> None:
        self.workspace.install_seed_files(
            {"seed.md": "distractor\n"},
            source_label="fixture",
        )
        with self.assertRaisesRegex(WorkspaceActionError, "only once"):
            self.workspace.install_seed_files(
                {"second.md": "x\n"},
                source_label="fixture-2",
            )
        self.workspace.install_causal_intervention("blank")
        self.assertIsNone(self.workspace.seed_manifest)
        self.assertFalse(self.workspace.provenance_summary["contains_harness_seed"])

        self.workspace.reset_episode("episode-policy-first")
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+x\n*** End Patch"
        )
        with self.assertRaisesRegex(WorkspaceActionError, "precede policy"):
            self.workspace.install_seed_files(
                {"seed.md": "late\n"},
                source_label="late",
            )

    def test_tampered_seed_manifest_is_rejected_transactionally(self) -> None:
        self.workspace.install_seed_files(
            {"seed.md": "distractor\n"},
            source_label="fixture",
        )
        state = self.workspace.export_state()
        tampered = deepcopy(state)
        tampered["seed_manifest"]["source_label"] = "changed"
        before = self.workspace.snapshot()
        with self.assertRaisesRegex(WorkspaceActionError, "manifest digest"):
            self.workspace.install_causal_intervention("correct", state=tampered)
        self.assertEqual(self.workspace.snapshot(), before)
        self.assertIsNone(self.workspace.causal_arm)

    def test_genuine_no_workspace_arm_has_no_host_directory(self) -> None:
        old_root = self.workspace.host_root
        self.workspace.reset_episode("episode-no-workspace", enabled=False)
        self.assertFalse(old_root.exists())
        self.assertFalse(self.workspace.enabled)
        self.assertEqual(self.workspace.snapshot()["file_count"], 0)
        self.assertIn("unavailable", self.workspace.render_contract())
        with self.assertRaisesRegex(WorkspaceActionError, "does not provide"):
            self.workspace.apply(
                shell_action("pwd"),
                env_step=1,
                phase_index=0,
            )
        with self.assertRaises(RuntimeError):
            _ = self.workspace.host_root

    def test_export_and_correct_intervention_restore_exact_policy_tree(self) -> None:
        self.apply(
            shell_action(
                "mkdir -p notes/empty && printf 'finish=black\\n' > notes/MEMORY.md"
            )
        )
        state = self.workspace.export_state()
        expected = self.workspace.snapshot()
        self.apply(shell_action("printf changed > notes/MEMORY.md"), step=2)

        event = self.workspace.install_causal_intervention(
            "correct",
            state=state,
            source_label="target-source",
        )
        self.assertEqual(self.workspace.snapshot(), expected)
        self.assertEqual(self.workspace.audit_events, ())
        self.assertEqual(self.workspace.causal_arm, "correct")
        self.assertEqual(event["arm"], "correct")
        self.assertEqual(event["source_tree_sha256"], expected["tree_sha256"])
        self.assertEqual(
            (self.workspace.host_root / "notes/MEMORY.md").read_text(
                encoding="utf-8"
            ),
            "finish=black\n",
        )
        with self.assertRaisesRegex(WorkspaceActionError, "only once"):
            self.workspace.install_causal_intervention("blank")

    def test_blank_and_swapped_interventions_are_out_of_band(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: source.md\n+black\n*** End Patch"
        )
        self.workspace.install_causal_intervention("blank")
        self.assertEqual(self.workspace.snapshot()["file_count"], 0)
        self.assertEqual(self.workspace.snapshot()["directory_count"], 0)
        self.assertEqual(self.workspace.causal_arm, "blank")
        self.assertEqual(self.workspace.audit_events, ())

        swapped = PersistentWorkspace(
            "paired-env",
            shell_sandbox=InProcessTestShellSandbox(),
            root_parent=self.parent,
        )
        try:
            swapped.reset_episode("paired-episode")
            result = swapped.apply(
                "apply_patch\n*** Begin Patch\n*** Add File: pair.md\n+gray\n*** End Patch",
                env_step=1,
                phase_index=0,
            )
            self.assertIsNotNone(result)
            state = swapped.export_state()

            self.workspace.reset_episode("episode-swapped")
            event = self.workspace.install_causal_intervention(
                "swapped",
                state=state,
                source_label="paired-source",
            )
            self.assertEqual(self.workspace.causal_arm, "swapped")
            self.assertEqual(event["source_label"], "paired-source")
            self.assertEqual(
                (self.workspace.host_root / "pair.md").read_text(encoding="utf-8"),
                "gray\n",
            )
        finally:
            swapped.close()

    def test_no_workspace_intervention_removes_tree_and_rejects_tools(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+x\n*** End Patch"
        )
        old_root = self.workspace.host_root
        event = self.workspace.install_causal_intervention("no_workspace")
        self.assertFalse(old_root.exists())
        self.assertFalse(self.workspace.enabled)
        self.assertEqual(self.workspace.causal_arm, "no_workspace")
        self.assertFalse(event["workspace_enabled_after"])
        self.assertEqual(self.workspace.snapshot()["file_count"], 0)
        with self.assertRaisesRegex(WorkspaceActionError, "does not provide"):
            self.workspace.apply(
                shell_action("pwd"),
                env_step=2,
                phase_index=1,
            )

    def test_no_workspace_delete_failure_preserves_live_workspace_state(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+original\n*** End Patch"
        )
        root = self.workspace.host_root
        before = self.workspace.snapshot()
        with patch(
            "agentenv_agentmemory.persistent_workspace.shutil.rmtree",
            side_effect=OSError("simulated deletion failure"),
        ):
            with self.assertRaisesRegex(OSError, "simulated deletion failure"):
                self.workspace.install_causal_intervention("no_workspace")
        self.assertTrue(self.workspace.enabled)
        self.assertEqual(self.workspace.host_root, root)
        self.assertEqual(self.workspace.snapshot(), before)
        self.assertIsNone(self.workspace.causal_arm)
        self.assertIsNone(self.workspace.control_event)

    def test_tampered_intervention_state_is_rejected_transactionally(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+original\n*** End Patch"
        )
        state = self.workspace.export_state()
        before = self.workspace.snapshot()
        tampered = deepcopy(state)
        tampered["files"][0]["content_base64"] = "dGFtcGVyZWQ="
        with self.assertRaises(WorkspaceActionError):
            self.workspace.install_causal_intervention(
                "swapped",
                state=tampered,
            )
        self.assertEqual(self.workspace.snapshot(), before)
        self.assertIsNone(self.workspace.causal_arm)

    def test_invalid_base64_intervention_state_is_rejected_transactionally(self) -> None:
        self.apply(
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+original\n*** End Patch"
        )
        state = self.workspace.export_state()
        before = self.workspace.snapshot()
        for invalid in ("%%%", "\N{SNOWMAN}"):
            with self.subTest(invalid=invalid):
                tampered = deepcopy(state)
                tampered["files"][0]["content_base64"] = invalid
                with self.assertRaisesRegex(WorkspaceActionError, "canonical base64"):
                    self.workspace.install_causal_intervention(
                        "swapped",
                        state=tampered,
                    )
                self.assertEqual(self.workspace.snapshot(), before)
                self.assertIsNone(self.workspace.causal_arm)


if __name__ == "__main__":
    unittest.main()
