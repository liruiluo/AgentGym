from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.filesystem_webshop_env import (
    FILESYSTEM_REWARD_CONTRACT,
    PersistentWorkspaceWebShopEnv,
)
from agentenv_agentmemory.filesystem_wrapper import (
    ProceduralFilesystemAgentMemoryWrapper,
)
from agentenv_agentmemory.memoryarena_webshop_env import MemoryArenaWebShopEnv
from agentenv_agentmemory.persistent_workspace import WORKSPACE_TOOL_OPS, WorkspaceLimits
from agentenv_agentmemory.procedural_wrapper import ProceduralAgentMemoryWrapper
from agentenv_agentmemory.reward_hierarchy import (
    INVALID_ACTION_PENALTY,
    WRONG_BUY_TERMINAL_FAILURE,
)
from tests.test_memoryarena_webshop_native import (
    TARGETS,
    FakeNativeBackend,
    make_bundle,
    purchase,
)
from tests.workspace_test_support import InProcessTestShellSandbox


def shell_action(command: str, *, workdir: str = ".") -> str:
    return "shell_command " + json.dumps(
        {"command": command, "workdir": workdir, "timeout_ms": 10_000},
        separators=(",", ":"),
    )


class PersistentWorkspaceWebShopEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.backend = FakeNativeBackend()
        self.env = PersistentWorkspaceWebShopEnv(
            bundles=[make_bundle()],
            backend=self.backend,
            env_uid="filesystem-test",
            shell_sandbox=InProcessTestShellSandbox(),
            workspace_root_parent=Path(self.temporary.name),
        )
        self.observation, self.info = self.env.reset()

    def tearDown(self) -> None:
        self.env.close()
        self.temporary.cleanup()

    def test_observation_exposes_codex_tools_without_memory_api(self) -> None:
        self.assertIn('shell_command {"command":', self.observation)
        self.assertIn("apply_patch", self.observation)
        for stale in (
            "Read {",
            "Write {",
            "Edit {",
            "Grep {",
            "Glob {",
            "ADD {",
            "RETRIEVE {",
            "UPDATE {",
            "DELETE {",
            "SUMMARY {",
            "FILTER {",
        ):
            self.assertNotIn(stale, self.observation)
        self.assertIn("persists across shopping sessions", self.observation)
        self.assertNotIn(str(self.env.workspace.host_root), self.observation)

    def test_patch_then_later_shell_read_are_zero_reward_and_persistent(self) -> None:
        _, reward, done, _, patch_info = self.env.step(
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Add File: .agent_memory/MEMORY.md\n"
            "+selected finish: black\n"
            "*** End Patch"
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(patch_info["tool_ops"][0]["op"], "APPLY_PATCH")
        self.assertEqual(patch_info["reward_components"][0]["name"], "apply_patch_transition")
        self.assertEqual(patch_info["memory_ops"], [])
        self.assertEqual(patch_info["workspace_ops"][0]["op"], "APPLY_PATCH")
        self.assertEqual(patch_info["workspace_snapshot"]["file_count"], 1)

        observation, reward, done, _, buy_info = purchase(self.env, TARGETS[0])
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)
        self.assertEqual(buy_info["current_subtask_index"], 1)
        self.assertEqual(buy_info["workspace_snapshot"]["file_count"], 1)
        self.assertNotIn("selected finish: black", observation)

        observation, reward, done, _, shell_info = self.env.step(
            shell_action("cat MEMORY.md", workdir=".agent_memory")
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertIn("selected finish: black", observation)
        self.assertEqual(shell_info["tool_ops"][0]["op"], "SHELL_COMMAND")
        self.assertEqual(shell_info["workspace_audit_event_count"], 2)
        self.assertEqual(shell_info["workspace_latest_event"]["phase_index"], 1)

    def test_repeated_workspace_action_has_no_dedicated_shaping(self) -> None:
        action = shell_action("printf same")
        for expected_step in (1, 2):
            _, reward, done, _, info = self.env.step(action)
            self.assertEqual(reward, 0.0)
            self.assertFalse(done)
            self.assertEqual(
                info["reward_components"],
                [
                    {
                        "name": "shell_command_transition",
                        "value": 0.0,
                        "op": "SHELL_COMMAND",
                        "step": expected_step,
                    }
                ],
            )

    def test_wrong_buy_reward_contract_matches_runtime(self) -> None:
        _, reward, done, _, info = purchase(self.env, TARGETS[1])
        self.assertEqual(reward, WRONG_BUY_TERMINAL_FAILURE)
        self.assertTrue(done)
        self.assertEqual(info["status"], "failed_purchase")
        self.assertEqual(
            info["reward_contract"]["wrong_buy_terminal_reward"],
            WRONG_BUY_TERMINAL_FAILURE,
        )
        self.assertTrue(info["tool_ops"][0]["terminal"])
        self.assertFalse(info["tool_ops"][0]["purchase_correct"])

    def test_legacy_and_claude_style_actions_are_invalid_on_v2(self) -> None:
        for action in (
            'ADD {"key":"finish","value":"black"}',
            'Read {"path":"notes.md"}',
        ):
            with self.subTest(action=action):
                observation, reward, done, _, info = self.env.step(action)
                self.assertEqual(reward, INVALID_ACTION_PENALTY)
                self.assertFalse(done)
                self.assertIn("does not expose ADD", observation)
                self.assertEqual(info["tool_ops"], [])
                self.assertEqual(info["workspace_snapshot"]["file_count"], 0)

    def test_malformed_workspace_action_is_invalid_without_mutation(self) -> None:
        _, reward, done, _, info = self.env.step(
            shell_action("pwd", workdir="../outside")
        )
        self.assertEqual(reward, INVALID_ACTION_PENALTY)
        self.assertFalse(done)
        self.assertEqual(info["workspace_snapshot"]["file_count"], 0)
        self.assertEqual(info["workspace_audit_event_count"], 0)

    def test_reset_replaces_workspace_and_removes_prior_episode(self) -> None:
        self.env.step(
            "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+old episode\n*** End Patch"
        )
        old_root = self.env.workspace.host_root
        _, info = self.env.reset(data_idx=0)
        self.assertFalse(old_root.exists())
        self.assertEqual(info["workspace_snapshot"]["file_count"], 0)
        self.assertEqual(info["workspace_audit_event_count"], 0)

    def test_no_workspace_is_genuinely_unavailable(self) -> None:
        self.env.set_workspace_enabled(False)
        observation, info = self.env.reset(data_idx=0)
        self.assertIn("unavailable", observation)
        self.assertEqual(info["workspace_intervention"], "no_workspace")
        self.assertFalse(info["workspace_shell_enabled"])
        _, reward, done, _, step_info = self.env.step(shell_action("pwd"))
        self.assertEqual(reward, INVALID_ACTION_PENALTY)
        self.assertFalse(done)
        self.assertEqual(step_info["workspace_audit_event_count"], 0)
        with self.assertRaises(RuntimeError):
            _ = self.env.workspace.host_root

    def test_causal_intervention_is_frozen_at_first_session_boundary(self) -> None:
        self.env.step(
            "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+black\n*** End Patch"
        )
        with self.assertRaisesRegex(RuntimeError, "cross-session boundary"):
            self.env.install_workspace_causal_intervention("blank")
        purchase(self.env, TARGETS[0])
        observation, info = self.env.install_workspace_causal_intervention("blank")
        self.assertIn("Persistent workspace tools", observation)
        self.assertEqual(info["workspace_causal_arm"], "blank")
        self.assertEqual(info["workspace_intervention"], "enabled")
        self.assertEqual(info["workspace_snapshot"]["file_count"], 0)
        self.assertEqual(info["workspace_audit_event_count"], 0)
        self.assertFalse(info["workspace_control_event"]["policy_action"])

        self.env.reset()
        purchase(self.env, TARGETS[0])
        observation, info = self.env.install_workspace_causal_intervention(
            "no_workspace"
        )
        self.assertIn("unavailable in this intervention", observation)
        self.assertEqual(info["workspace_causal_arm"], "no_workspace")
        self.assertEqual(info["workspace_intervention"], "no_workspace")
        self.assertFalse(info["workspace_shell_enabled"])
        self.assertFalse(info["workspace_apply_patch_enabled"])

    def test_info_and_reward_contract_do_not_expose_host_path(self) -> None:
        serialized = json.dumps(self.info)
        self.assertNotIn(str(self.env.workspace.host_root), serialized)
        self.assertEqual(self.info["reward_contract"], FILESYSTEM_REWARD_CONTRACT)
        self.assertEqual(
            self.info["memory_management"],
            "policy_managed_persistent_workspace",
        )
        self.assertTrue(self.info["workspace_shell_enabled"])
        self.assertTrue(self.info["workspace_apply_patch_enabled"])

    def test_legacy_surface_keeps_accepting_add_after_parser_hook(self) -> None:
        legacy = MemoryArenaWebShopEnv(
            bundles=[make_bundle()],
            backend=FakeNativeBackend(),
            env_uid="legacy-control",
        )
        try:
            legacy.reset()
            _, reward, done, _, info = legacy.step(
                'ADD {"key":"finish","value":"black"}'
            )
            self.assertGreater(reward, 0.0)
            self.assertFalse(done)
            self.assertEqual(info["tool_ops"][0]["op"], "ADD")
        finally:
            legacy.close()


class ProceduralFilesystemWrapperTests(unittest.TestCase):
    @staticmethod
    def _initialize_base(wrapper, *, prompt_mode="natural_filesystem", reward=0.0):
        wrapper.memory_prompt_mode = prompt_mode
        wrapper.reward_contract = {
            "first_valid_add_reward": reward,
            "first_valid_later_session_retrieve_reward": 0.0,
        }

    def _construct(self, *, intervention: bool = False):
        sandbox = InProcessTestShellSandbox()
        environment = {
            "AGENTMEMORY_WORKSPACE_RG_BINARY": "/tmp/test-rg",
            "AGENTMEMORY_WORKSPACE_RG_SHA256": "c" * 64,
        }
        if intervention:
            environment.update(
                {
                    "AGENTMEMORY_SERVICE_ROLE": "intervention_eval",
                    "AGENTMEMORY_WORKSPACE_INTERVENTION_TOKEN": "t" * 48,
                }
            )
        with (
            patch.object(
                ProceduralAgentMemoryWrapper,
                "__init__",
                lambda wrapper: self._initialize_base(wrapper),
            ),
            patch(
                "agentenv_agentmemory.filesystem_wrapper.LinuxNamespaceShellSandbox.from_environment",
                return_value=sandbox,
            ),
            patch.dict(
                "os.environ",
                environment,
                clear=True,
            ),
        ):
            wrapper = ProceduralFilesystemAgentMemoryWrapper()
        return wrapper, sandbox

    def test_wrapper_requires_natural_prompt_and_zero_legacy_shaping(self) -> None:
        wrapper, sandbox = self._construct()
        self.assertEqual(wrapper.reward_contract, FILESYSTEM_REWARD_CONTRACT)
        self.assertIs(wrapper.shell_sandbox, sandbox)
        self.assertIsInstance(wrapper.workspace_limits, WorkspaceLimits)

        for prompt_mode, reward in (("legacy", 0.0), ("natural_filesystem", 0.1)):
            with (
                self.subTest(prompt_mode=prompt_mode, reward=reward),
                patch.object(
                    ProceduralAgentMemoryWrapper,
                    "__init__",
                    lambda wrapper, mode=prompt_mode, value=reward: self._initialize_base(
                        wrapper,
                        prompt_mode=mode,
                        reward=value,
                    ),
                ),
                patch.dict(
                    "os.environ",
                    {
                        "AGENTMEMORY_WORKSPACE_RG_BINARY": "/tmp/test-rg",
                        "AGENTMEMORY_WORKSPACE_RG_SHA256": "c" * 64,
                    },
                    clear=True,
                ),
                self.assertRaises(RuntimeError),
            ):
                ProceduralFilesystemAgentMemoryWrapper()

    def test_wrapper_metadata_attests_codex_workspace(self) -> None:
        wrapper, _ = self._construct()
        base_metadata = {
            "surface": wrapper.surface,
            "paper_eligible": False,
            "memory_prompt_mode": "natural_filesystem",
            "reward_contract": {"legacy": True},
            "ltm_inventory_mode": "hidden",
            "ltm_transition_notice_mode": "none",
            "ltm_inventory_key_max_chars": 24,
            "ltm_inventory_key_format": "ascii_identifier",
        }
        with patch.object(
            ProceduralAgentMemoryWrapper,
            "metadata",
            return_value=base_metadata,
        ):
            metadata = wrapper.metadata()
        for legacy_key in (
            "ltm_inventory_mode",
            "ltm_transition_notice_mode",
            "ltm_inventory_key_max_chars",
            "ltm_inventory_key_format",
        ):
            self.assertNotIn(legacy_key, metadata)
        self.assertEqual(metadata["reward_contract"], FILESYSTEM_REWARD_CONTRACT)
        self.assertEqual(metadata["workspace_tool_ops"], list(WORKSPACE_TOOL_OPS))
        self.assertEqual(metadata["workspace_surface"], "codex_workspace_v2")
        self.assertTrue(metadata["workspace_shell_enabled"])
        self.assertTrue(metadata["workspace_apply_patch_enabled"])
        self.assertFalse(metadata["workspace_host_path_exposed"])
        self.assertEqual(
            metadata["workspace_limits"],
            wrapper.workspace_limits.as_metadata(),
        )
        self.assertEqual(
            metadata["workspace_sandbox"],
            dict(wrapper.shell_sandbox.metadata),
        )
        self.assertFalse(metadata["workspace_intervention_control"]["enabled"])
        self.assertTrue(
            metadata["workspace_intervention_control"]["authenticated_export"]
        )

    def test_wrapper_environment_configuration_injects_sandbox(self) -> None:
        wrapper, sandbox = self._construct()
        wrapper.workspace_root_parent = Path("/tmp/workspaces")
        configuration = wrapper._environment_configuration()
        self.assertEqual(configuration["workspace_root_parent"], Path("/tmp/workspaces"))
        self.assertEqual(configuration["workspace_limits"], wrapper.workspace_limits)
        self.assertIs(configuration["shell_sandbox"], sandbox)

    def test_authenticated_control_copies_only_paired_policy_workspace(self) -> None:
        wrapper, sandbox = self._construct(intervention=True)
        backend = FakeNativeBackend()
        target = PersistentWorkspaceWebShopEnv(
            bundles=[make_bundle()],
            backend=backend,
            env_uid="target",
            shell_sandbox=sandbox,
        )
        paired = PersistentWorkspaceWebShopEnv(
            bundles=[make_bundle()],
            backend=backend,
            env_uid="paired",
            shell_sandbox=sandbox,
        )
        try:
            target.reset(data_idx=0)
            paired.reset(data_idx=1)
            target.step(
                "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+black\n*** End Patch"
            )
            paired.step(
                "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+gray\n*** End Patch"
            )
            purchase(target, TARGETS[0])
            purchase(paired, TARGETS[0])
            wrapper.envs = {0: target, 1: paired}
            wrapper.info = {}
            wrapper.env_locks = {0: threading.RLock(), 1: threading.RLock()}

            with self.assertRaises(PermissionError):
                wrapper.workspace_export(0, token="wrong")
            exported = wrapper.workspace_export(0, token="t" * 48)
            self.assertEqual(
                exported["schema"],
                "agentmemory_workspace_authenticated_export_v1",
            )
            self.assertEqual(exported["data_idx"], 0)
            self.assertTrue(exported["policy_authored"])
            self.assertFalse(exported["hidden_answer_injection"])
            self.assertEqual(exported["workspace_state"]["file_count"], 1)

            with self.assertRaises(PermissionError):
                wrapper.workspace_intervention(
                    0,
                    arm="swapped",
                    source_env_id=1,
                    token="wrong",
                )
            result = wrapper.workspace_intervention(
                0,
                arm="swapped",
                source_env_id=1,
                token="t" * 48,
            )
            self.assertEqual(result["info"]["workspace_causal_arm"], "swapped")
            self.assertEqual(
                (target.workspace.host_root / "note.md").read_text(encoding="utf-8"),
                "gray\n",
            )
            self.assertEqual(
                paired.workspace.snapshot()["files"][0]["sha256"],
                target.workspace.snapshot()["files"][0]["sha256"],
            )
        finally:
            target.close()
            paired.close()

    def test_intervention_control_rejects_ineligible_sources_and_boundaries(self) -> None:
        wrapper, sandbox = self._construct(intervention=True)
        backend = FakeNativeBackend()
        environments = {
            identifier: PersistentWorkspaceWebShopEnv(
                bundles=[make_bundle()],
                backend=backend,
                env_uid=f"env-{identifier}",
                shell_sandbox=sandbox,
            )
            for identifier in range(4)
        }
        try:
            target, paired, nonpaired, preboundary = (
                environments[index] for index in range(4)
            )
            target.reset(data_idx=0)
            paired.reset(data_idx=1)
            nonpaired.reset(data_idx=3)
            preboundary.reset(data_idx=0)
            for environment, value in (
                (target, "black"),
                (nonpaired, "blue"),
            ):
                environment.step(
                    "apply_patch\n*** Begin Patch\n*** Add File: note.md\n"
                    f"+{value}\n*** End Patch"
                )
            for environment in (target, paired, nonpaired):
                purchase(environment, TARGETS[0])

            wrapper.envs = environments
            wrapper.info = {}
            wrapper.env_locks = {
                identifier: threading.RLock() for identifier in environments
            }

            with self.assertRaisesRegex(ValueError, "first-session boundary"):
                wrapper.workspace_export(3, token="t" * 48)
            with self.assertRaisesRegex(ValueError, "exact counterfactual pair"):
                wrapper.workspace_intervention(
                    0,
                    arm="swapped",
                    source_env_id=2,
                    token="t" * 48,
                )
            with self.assertRaisesRegex(ValueError, "source workspace is empty"):
                wrapper.workspace_intervention(
                    0,
                    arm="swapped",
                    source_env_id=1,
                    token="t" * 48,
                )
            with self.assertRaisesRegex(ValueError, "must not name a source"):
                wrapper.workspace_intervention(
                    0,
                    arm="blank",
                    source_env_id=1,
                    token="t" * 48,
                )
            with self.assertRaisesRegex(ValueError, "first-session boundary"):
                wrapper.workspace_intervention(
                    3,
                    arm="no_workspace",
                    source_env_id=None,
                    token="t" * 48,
                )
            self.assertIsNone(target.workspace.causal_arm)
            self.assertTrue(target.workspace.enabled)
        finally:
            for environment in environments.values():
                environment.close()


if __name__ == "__main__":
    unittest.main()
