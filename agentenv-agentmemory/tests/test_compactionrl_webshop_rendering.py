from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentenv.controller.policy_turn import (
    bind_initial_policy_context,
    complete_policy_turn,
    prepare_policy_turn,
)
from agentenv.controller.types import ActionFormat
from agentenv.envs.agentmemory import (
    AgentMemoryEnvClient,
    FilesystemAgentMemoryAdapter,
)
from agentenv.envs.context_compaction import configure_compactionrl_controller
from agentenv_agentmemory.filesystem_webshop_env import (
    PersistentWorkspaceWebShopEnv,
)
from tests.test_memoryarena_webshop_native import (
    TARGETS,
    WRONG,
    FakeNativeBackend,
    make_bundle,
)
from tests.workspace_test_support import InProcessTestShellSandbox


def _count_characters(messages) -> int:
    return sum(len(message["content"]) for message in messages)


class CompactionRLWebShopRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = PersistentWorkspaceWebShopEnv(
            bundles=[make_bundle()],
            backend=FakeNativeBackend(),
            env_uid="compactionrl-webshop-fixture",
            shell_sandbox=InProcessTestShellSandbox(),
            workspace_root_parent=Path(self.temporary.name),
        )
        observation, env_info = self.environment.reset()

        client = object.__new__(AgentMemoryEnvClient)
        client.adapter_cls = FilesystemAgentMemoryAdapter
        client.action_format = ActionFormat.REACT
        client.env_id = "compactionrl-webshop-fixture"
        client.metadata = {"surface": self.environment.surface}
        client.is_v3 = False
        client.is_filesystem = True
        client._policy_system_prompt = "fixture system prompt"
        configure_compactionrl_controller(client, mode="compactionrl")
        client.info = {
            "observation": observation,
            "reward": 0.0,
            "done": False,
            "env_info": env_info,
            "metadata": client.metadata,
        }
        client.last_action_submission = None
        client._reset_policy_transition_state(env_info)

        def post(path, data):
            self.assertEqual(path, "step")
            observation, reward, terminated, truncated, info = self.environment.step(
                data["action"]
            )
            return {
                "observation": observation,
                "reward": reward,
                "done": bool(terminated or truncated),
                "info": info,
            }

        client.post = post
        self.client = client
        self.messages = bind_initial_policy_context(
            client,
            [{"role": "user", "content": observation}],
        )

    def tearDown(self) -> None:
        self.environment.close()
        self.temporary.cleanup()

    def _step(self, action: str):
        prepared = prepare_policy_turn(
            self.client,
            self.messages,
            count_prompt_tokens=_count_characters,
            max_prompt_tokens=1_000_000,
            max_model_tokens=1_000_128,
            max_response_tokens=128,
            max_observation_tokens=16_384,
            action_observation_envelope_tokens=16,
        )
        self.assertIsNone(prepared.control_request)
        output, self.messages = complete_policy_turn(
            self.client,
            prepared,
            action,
        )
        return output

    def test_real_server_trace_is_omitted_only_from_policy_rendering(self) -> None:
        self.assertNotIn("Current-session action trace:", self.messages[-1]["content"])

        searched = self._step("search[item]")
        self.assertIn(
            "Current-session action trace:",
            self.client.info["observation"],
        )
        self.assertIn("search[item]", self.client.info["observation"])
        self.assertNotIn("Current-session action trace:", searched.state)
        self.assertEqual(
            searched.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertTrue(
            searched.info["wrapper_evidence"]["native_session_trace_retained"]
        )
        self.assertEqual(
            searched.info["wrapper_evidence"]["policy_session_trace_rendering"],
            "omitted_redundant_cumulative_trace",
        )
        self.assertGreater(len(self.environment.session_trace), 0)

        self._step(f"click[{TARGETS[0]}]")
        advanced = self._step("click[Buy Now]")
        self.assertEqual(self.environment.current_session_index, 1)
        self.assertEqual(self.client._session_epoch, 1)
        self.assertIsNone(self.client._pending_session_handoff)
        self.assertFalse(advanced.done)
        self.assertNotIn("Current-session action trace:", advanced.state)
        self.assertEqual(
            advanced.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertIn("Progress: 1/6", advanced.state)

        # The wrapper keeps the ordinary CompactionRL action/observation history;
        # only the server's duplicated cumulative rendering is omitted.
        assistant_turns = [
            message["content"]
            for message in self.messages
            if message["role"] == "assistant"
        ]
        self.assertEqual(
            assistant_turns,
            ["search[item]", f"click[{TARGETS[0]}]", "click[Buy Now]"],
        )
        self.assertTrue(
            all(
                "Current-session action trace:" not in message["content"]
                for message in self.messages
                if message["role"] == "user"
            )
        )

    def test_terminal_observation_without_workspace_sections_is_preserved(self) -> None:
        self._step("search[item]")
        self._step(f"click[{WRONG}]")
        terminal = self._step("click[Buy Now]")

        self.assertTrue(terminal.done)
        self.assertEqual(
            terminal.state,
            "The shopping episode has ended.\n\n"
            "Task family: bundled_shopping\nProgress: 0/6",
        )
        self.assertEqual(
            terminal.info["wrapper_evidence"]["policy_session_trace_rendering"],
            "terminal_observation_preserved",
        )


if __name__ == "__main__":
    unittest.main()
