from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from agentenv.envs.letta_code import (
    LETTA_CODE_SOURCE_REVISION,
    LETTA_PROMPT_MARKER,
    LettaCodeAdapterConfig,
    LettaCodeEnvClientAdapter,
    parse_letta_action,
    parse_memory_filesystem_read,
)
from agentenv.envs.literesearcher import LITERESEARCHER_SYSTEM_PROMPT
from agentenv.envs.openmle_fast import OPENMLE_FAST_POLICY_SYSTEM_PROMPT
from agentenv.envs.swesmith import SWE_POLICY_SYSTEM_PROMPT


def memory_action(name: str, **arguments: object) -> str:
    return (
        "<letta_memory_call>"
        + json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "</letta_memory_call>"
    )


class FakeEnvClient(BaseEnvClient):
    def __init__(
        self,
        *,
        system_prompt: str = "native system",
        strict_initial_framing: bool = False,
    ) -> None:
        super().__init__("react")
        self.system_prompt = system_prompt
        self.strict_initial_framing = strict_initial_framing
        self.native_actions: list[str] = []
        self.bound: list[list[dict[str, str]]] = []
        self.episode_source_identity: dict[str, object] | None = None
        self.control_candidate: str | None = None
        self.select_control = False

    def __len__(self) -> int:
        return 2

    def observe(self) -> str:
        return "native observation"

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}]

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        return [dict(message) for message in messages]

    def bind_policy_context(
        self, messages: Sequence[Mapping[str, str]], *, initial: bool = False
    ) -> None:
        normalized = [dict(message) for message in messages]
        if initial and self.strict_initial_framing:
            expected = self.policy_framing() + [
                {"role": "user", "content": self.observe()}
            ]
            if normalized != expected:
                raise ValueError("native initial policy context differs from framing")
        self.bound.append(normalized)
        self.assert_no_letta_prompt(normalized)

    @staticmethod
    def assert_no_letta_prompt(messages: Sequence[Mapping[str, str]]) -> None:
        if any(LETTA_PROMPT_MARKER in message["content"] for message in messages):
            raise AssertionError("Letta prompt leaked into native wrapper")

    def policy_turn_candidate(self) -> str | None:
        return self.control_candidate

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        del pressure
        return self.control_candidate if self.select_control else None

    def step(self, action: str) -> StepOutput:
        self.native_actions.append(action)
        return StepOutput(
            state="native result",
            reward=0.25,
            done=False,
            info=build_task_neutral_transition_info(
                action_submission={"raw_policy_output": action},
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE,
                    messages=[
                        {"role": "system", "content": "native system"},
                        {"role": "user", "content": "native replacement"},
                    ],
                ),
                wrapper_evidence={"event": "native_action"},
            ),
        )

    def reset(self, idx: int = 0) -> None:
        self.episode_source_identity = {
            "schema": "camg_native_episode_source_identity_v1",
            "route_id": "swesmith",
            "data_idx": idx,
            "instance_id": f"repo.issue-{idx}",
        }

    def finalize_policy_horizon(self) -> StepOutput | None:
        return None

    def close(self) -> None:
        return None


class LettaCodeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        native = FakeEnvClient()
        self.adapter = LettaCodeEnvClientAdapter(
            native,
            LettaCodeAdapterConfig(
                runtime_root=self.temp.name,
                max_file_bytes=256,
                max_total_bytes=1024,
                max_files=16,
            ),
        )
        self.adapter.reset(0)
        messages = self.adapter.policy_framing() + [
            {"role": "user", "content": self.adapter.observe()}
        ]
        messages = self.adapter.normalize_initial_policy_context(messages)
        self.adapter.bind_policy_context(messages, initial=True)
        self.messages = messages

    def tearDown(self) -> None:
        self.adapter.close()
        self.temp.cleanup()

    def rebind(self, output: StepOutput) -> None:
        replacement = output.info["context_transition"]["messages"]
        self.adapter.bind_policy_context(replacement)
        self.messages = replacement

    def test_official_write_tools_exclude_read(self) -> None:
        prompt = self.messages[0]["content"]
        self.assertIn(LETTA_CODE_SOURCE_REVISION, prompt)
        self.assertNotIn("update_description|read", prompt)
        output = self.adapter.step(
            memory_action(
                "memory",
                command="read",
                reason="read child",
                file_path="notes.md",
            )
        )
        payload = json.loads(output.state.split("\n", 1)[1])
        self.assertEqual(payload["error_code"], "unsupported_command")

    def test_initial_context_round_trip_preserves_native_framing_exactly(self) -> None:
        cases = {
            "webshop": "webshop system framing",
            "swesmith": SWE_POLICY_SYSTEM_PROMPT,
            "literesearcher": LITERESEARCHER_SYSTEM_PROMPT,
            "openmle_fast": OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
        }
        for route_id, system_prompt in cases.items():
            with self.subTest(route_id=route_id), tempfile.TemporaryDirectory() as root:
                native = FakeEnvClient(
                    system_prompt=system_prompt,
                    strict_initial_framing=True,
                )
                adapter = LettaCodeEnvClientAdapter(
                    native,
                    LettaCodeAdapterConfig(runtime_root=root),
                )
                try:
                    adapter.reset(0)
                    messages = adapter.policy_framing() + [
                        {"role": "user", "content": adapter.observe()}
                    ]
                    normalized = adapter.normalize_initial_policy_context(messages)
                    adapter.bind_policy_context(normalized, initial=True)
                    self.assertEqual(
                        native.bound[-1],
                        native.policy_framing()
                        + [{"role": "user", "content": native.observe()}],
                    )
                finally:
                    adapter.close()

    def test_create_recompiles_core_and_commits(self) -> None:
        output = self.adapter.step(
            memory_action(
                "memory",
                command="create",
                reason="Remember exact callback",
                file_path="callback",
                description="Verified callback path",
                file_text="Use /oauth/callback exactly.\n",
            )
        )
        self.assertEqual(
            output.info["context_transition"]["operation"],
            CONTEXT_OPERATION_REPLACE,
        )
        evidence = output.info["wrapper_evidence"]["letta_code_adapter"]
        self.assertRegex(evidence["commit_sha"], r"^[0-9a-f]{40}$")
        self.assertIn(
            "Use /oauth/callback exactly",
            output.info["context_transition"]["messages"][0]["content"],
        )
        self.rebind(output)
        self.assertNotIn(
            LETTA_PROMPT_MARKER,
            self.adapter.native_client.bound[-1][0]["content"],
        )

    def test_child_memory_requires_index_then_reads_via_shell(self) -> None:
        missing = self.adapter.step(
            memory_action(
                "memory",
                command="create",
                reason="create fact",
                file_path="reference/fact.md",
                description="A verified fact",
                file_text="alpha\n",
            )
        )
        self.assertEqual(
            json.loads(missing.state.split("\n", 1)[1])["error_code"],
            "missing_memory_index",
        )

        index = self.adapter.step(
            memory_action(
                "memory",
                command="create",
                reason="index reference",
                file_path="reference/MEMORY.md",
                file_text="# Reference\n",
            )
        )
        self.rebind(index)
        created = self.adapter.step(
            memory_action(
                "memory",
                command="create",
                reason="create fact",
                file_path="reference/fact",
                description="A verified fact",
                file_text="alpha\n",
            )
        )
        self.rebind(created)
        read = self.adapter.step(
            'shell_command {"command":"cat $MEMORY_DIR/reference/fact.md"}'
        )
        evidence = read.info["wrapper_evidence"]["letta_code_adapter"]
        self.assertEqual(evidence["event"], "memory_filesystem_read")
        self.assertEqual(evidence["operation"], "read")
        self.assertIn("alpha", read.state)
        self.assertEqual(self.adapter.native_client.native_actions, [])

    def test_memory_read_rejects_shell_composition(self) -> None:
        output = self.adapter.step(
            'shell_command {"command":"cat $MEMORY_DIR/MEMORY.md; uname -a"}'
        )
        payload = json.loads(output.state.split("\n", 1)[1])
        self.assertEqual(payload["error_code"], "invalid_memory_read")
        self.assertEqual(self.adapter.native_client.native_actions, [])

    def test_failed_capacity_write_rolls_back(self) -> None:
        before = self.adapter._git_output("rev-parse", "HEAD").strip()
        output = self.adapter.step(
            memory_action(
                "memory",
                command="create",
                reason="oversized write",
                file_path="too-large.md",
                description="Too large",
                file_text="x" * 300,
            )
        )
        payload = json.loads(output.state.split("\n", 1)[1])
        self.assertEqual(payload["error_code"], "memory_capacity_exceeded")
        self.assertEqual(self.adapter._git_output("rev-parse", "HEAD").strip(), before)
        self.assertEqual(self.adapter._git_output("status", "--porcelain"), "")
        self.assertFalse((self.adapter._require_repo() / "too-large.md").exists())

    def test_apply_patch_adds_exact_frontmatter(self) -> None:
        output = self.adapter.step(
            memory_action(
                "memory_apply_patch",
                reason="add preferences",
                input=(
                    "*** Begin Patch\n"
                    "*** Add File: preferences.md\n"
                    "+Prefer small diffs.\n"
                    "*** End Patch"
                ),
            )
        )
        self.rebind(output)
        text = (self.adapter._require_repo() / "preferences.md").read_text()
        self.assertIn('name: "Preferences"', text)
        self.assertIn('description: "Memory block preferences"', text)

    def test_native_action_and_native_context_replacement_remain_native(self) -> None:
        output = self.adapter.step('shell_command {"command":"pwd"}')
        self.assertEqual(output.reward, 0.25)
        self.assertEqual(
            self.adapter.native_client.native_actions,
            ['shell_command {"command":"pwd"}'],
        )
        self.assertIn(
            LETTA_PROMPT_MARKER,
            output.info["context_transition"]["messages"][0]["content"],
        )
        self.assertEqual(
            output.info["wrapper_evidence"]["letta_code_adapter"]["event"],
            "native_action_passthrough",
        )

    def test_parser_and_read_parser_are_exact(self) -> None:
        parsed = parse_letta_action(
            memory_action(
                "memory",
                command="create",
                reason="x",
                file_path="x.md",
                description="x",
            )
        )
        self.assertEqual(parsed.name, "memory")
        self.assertEqual(
            parse_memory_filesystem_read(
                'shell_command {"command":"cat -- ${MEMORY_DIR}/MEMORY.md"}'
            ),
            "${MEMORY_DIR}/MEMORY.md",
        )
        self.assertIsNone(
            parse_memory_filesystem_read('shell_command {"command":"pwd"}')
        )


if __name__ == "__main__":
    unittest.main()
