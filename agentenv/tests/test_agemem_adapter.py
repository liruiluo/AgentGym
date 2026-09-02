from __future__ import annotations

import json
import unittest
from typing import Mapping, Sequence

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_REPLACE,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from agentenv.envs.agemem import (
    AGEMEM_PROMPT_MARKER,
    AgeMemAdapterConfig,
    AgeMemEnvClientAdapter,
    parse_agemem_action,
)


def memory_action(name: str, **arguments: object) -> str:
    return "<agemem_tool_call>" + json.dumps(
        [{"name": name, "arguments": arguments}],
        sort_keys=True,
        separators=(",", ":"),
    ) + "</agemem_tool_call>"


class FakeEnvClient(BaseEnvClient):
    def __init__(self, *, label: str = "fake") -> None:
        super().__init__("react")
        self.label = label
        self.info = {"observation": f"{label}-observation"}
        self.bound: list[tuple[list[dict[str, str]], bool]] = []
        self.native_actions: list[str] = []
        self.control_candidate: str | None = None
        self.select_control = False
        self.closed = False
        self.reset_count = 0
        self.fail_reset = False

    def __len__(self) -> int:
        return 4

    def observe(self) -> str:
        return str(self.info["observation"])

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": f"{self.label}-system"}]

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        normalized = [dict(message) for message in messages]
        return self.policy_framing() + [dict(normalized[-1])]

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = [dict(message) for message in messages]
        if initial:
            self.assert_plain_framing(normalized)
        self.bound.append((normalized, initial))

    def assert_plain_framing(self, messages: list[dict[str, str]]) -> None:
        if messages[0] != self.policy_framing()[0]:
            raise AssertionError("adapter leaked its prompt into the native client")

    def policy_turn_candidate(self) -> str | None:
        return self.control_candidate

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        del pressure
        return self.control_candidate if self.select_control else None

    def step(self, action: str) -> StepOutput:
        self.native_actions.append(action)
        replacement = [
            {"role": "system", "content": f"{self.label}-system"},
            {"role": "user", "content": "native replacement"},
        ]
        return StepOutput(
            state="native-state",
            reward=0.25,
            done=False,
            info=build_task_neutral_transition_info(
                action_submission={"raw_policy_output": action},
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE,
                    messages=replacement,
                ),
                wrapper_evidence={"event": "native_action"},
            ),
        )

    def reset(self, idx: int) -> None:
        self.reset_count += 1
        self.info = {"observation": f"{self.label}-observation-{idx}"}
        if self.fail_reset:
            raise RuntimeError("synthetic reset failure")

    def finalize_policy_horizon(self) -> StepOutput | None:
        return None

    def close(self) -> bool:
        self.closed = True
        return True


class AgeMemAdapterTests(unittest.TestCase):
    def make_adapter(self, *, label: str = "fake") -> AgeMemEnvClientAdapter:
        return AgeMemEnvClientAdapter(
            FakeEnvClient(label=label),
            AgeMemAdapterConfig(max_memories=8, max_content_bytes=2048),
        )

    def bind(self, adapter: AgeMemEnvClientAdapter) -> list[dict[str, str]]:
        adapter.reset(0)
        messages = adapter.policy_framing() + [
            {"role": "user", "content": adapter.observe()}
        ]
        normalized = list(adapter.normalize_initial_policy_context(messages))
        adapter.bind_policy_context(normalized, initial=True)
        return normalized

    def test_prompt_is_injected_once_and_stripped_before_native_bind(self) -> None:
        adapter = self.make_adapter()
        messages = self.bind(adapter)
        self.assertEqual(sum(AGEMEM_PROMPT_MARKER in m["content"] for m in messages), 1)
        native = adapter.native_client
        self.assertNotIn(AGEMEM_PROMPT_MARKER, native.bound[-1][0][0]["content"])
        normalized = list(adapter.normalize_initial_policy_context(messages))
        self.assertEqual(sum(AGEMEM_PROMPT_MARKER in m["content"] for m in normalized), 1)

    def test_prompt_round_trip_preserves_native_trailing_whitespace(self) -> None:
        class TrailingPromptEnvClient(FakeEnvClient):
            def policy_framing(self) -> list[dict[str, str]]:
                return [{"role": "system", "content": "native-system\n"}]

        native = TrailingPromptEnvClient()
        adapter = AgeMemEnvClientAdapter(native)
        adapter.reset(0)
        messages = adapter.policy_framing() + [
            {"role": "user", "content": adapter.observe()}
        ]
        normalized = list(adapter.normalize_initial_policy_context(messages))
        adapter.bind_policy_context(normalized, initial=True)
        self.assertEqual(native.bound[-1][0][0]["content"], "native-system\n")

    def test_add_retrieve_update_delete_and_reset_are_episode_private(self) -> None:
        adapter = self.make_adapter()
        messages = self.bind(adapter)
        adapter.bind_policy_context(messages)

        added = adapter.step(
            memory_action(
                "Add_memory",
                content="oauth redirect URI must preserve the exact callback path",
                metadata={"domain": "coding"},
                memory_type="constraint",
            )
        )
        self.assertEqual(added.reward, 0.0)
        self.assertFalse(added.done)
        add_payload = json.loads(added.state.removeprefix("[AgeMem tool result]\n"))
        memory_id = add_payload["memory_id"]
        self.assertEqual(add_payload["memory_size_after"], 1)
        self.assertEqual(
            added.info["context_transition"]["operation"], CONTEXT_OPERATION_APPEND
        )

        retrieved = adapter.step(
            memory_action(
                "Retrieve_memory",
                query="exact oauth callback path",
                top_k=3,
                metadata_filter={"domain": "coding"},
            )
        )
        retrieve_payload = json.loads(
            retrieved.state.removeprefix("[AgeMem tool result]\n")
        )
        self.assertEqual(retrieve_payload["memories"][0]["memory_id"], memory_id)
        self.assertGreater(retrieve_payload["memories"][0]["score"], 0.0)
        self.assertEqual(
            retrieved.info["wrapper_evidence"]["agemem_adapter"][
                "retrieved_memory_ids"
            ],
            [memory_id],
        )

        updated = adapter.step(
            memory_action(
                "Update_memory",
                memory_id=memory_id,
                content="oauth redirect URI must preserve /oauth/callback exactly",
                metadata={"domain": "coding", "verified": True},
            )
        )
        self.assertIn("updated", updated.state)
        deleted = adapter.step(
            memory_action(
                "Delete_memory", memory_id=memory_id, confirmation=True
            )
        )
        self.assertIn("deleted", deleted.state)
        self.assertEqual(adapter.memory_size, 0)

        adapter.step(memory_action("Add_memory", content="episode zero"))
        adapter.reset(1)
        self.assertEqual(adapter.memory_size, 0)
        empty = adapter.step(memory_action("Retrieve_memory", query="episode zero"))
        self.assertEqual(
            json.loads(empty.state.removeprefix("[AgeMem tool result]\n"))["memories"],
            [],
        )

    def test_reset_clears_episode_memory_even_when_native_reset_raises(self) -> None:
        adapter = self.make_adapter()
        self.bind(adapter)
        adapter.step(memory_action("Add_memory", content="episode zero"))
        native = adapter.native_client
        native.fail_reset = True
        with self.assertRaisesRegex(RuntimeError, "synthetic reset failure"):
            adapter.reset(1)
        self.assertEqual(adapter.memory_size, 0)

    def test_retrieval_survives_a_context_replacement(self) -> None:
        adapter = self.make_adapter()
        messages = self.bind(adapter)
        added = adapter.step(memory_action("Add_memory", content="durable fact"))
        memory_id = json.loads(added.state.split("\n", 1)[1])["memory_id"]
        adapter.bind_policy_context(
            messages
            + [
                {"role": "assistant", "content": "old detail"},
                {"role": "user", "content": "old observation"},
            ]
        )
        summarized = adapter.step(
            memory_action(
                "Summary_context",
                span="2",
                summary="compressed observation",
            )
        )
        adapter.bind_policy_context(
            summarized.info["context_transition"]["messages"]
        )
        retrieved = adapter.step(
            memory_action("Retrieve_memory", query="durable fact")
        )
        payload = json.loads(retrieved.state.split("\n", 1)[1])
        self.assertEqual(payload["retrieved_memory_ids"], [memory_id])

    def test_update_and_delete_require_a_prior_retrieval(self) -> None:
        adapter = self.make_adapter()
        self.bind(adapter)
        add = adapter.step(memory_action("Add_memory", content="keep this"))
        memory_id = json.loads(add.state.split("\n", 1)[1])["memory_id"]
        rejected = adapter.step(
            memory_action("Update_memory", memory_id=memory_id, content="changed")
        )
        payload = json.loads(rejected.state.split("\n", 1)[1])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "memory_id_not_retrieved")

    def test_summary_and_filter_return_task_neutral_replacements(self) -> None:
        adapter = self.make_adapter()
        messages = self.bind(adapter)
        messages += [
            {"role": "assistant", "content": "searched old cache"},
            {"role": "user", "content": "old cache produced stale result"},
            {"role": "assistant", "content": "verified current source"},
            {"role": "user", "content": "current source says callback is /oauth/callback"},
        ]
        adapter.bind_policy_context(messages)
        summarized = adapter.step(
            memory_action(
                "Summary_context",
                span="2",
                summary="Verified callback is /oauth/callback.",
            )
        )
        transition = summarized.info["context_transition"]
        self.assertEqual(transition["operation"], CONTEXT_OPERATION_REPLACE)
        replacement = transition["messages"]
        self.assertTrue(any(AGEMEM_PROMPT_MARKER in m["content"] for m in replacement))
        self.assertTrue(any("Verified callback" in m["content"] for m in replacement))
        self.assertFalse(any("current source says" in m["content"] for m in replacement))

        adapter.bind_policy_context(replacement)
        filtered = adapter.step(
            memory_action("Filter_context", criteria="old cache stale result")
        )
        filtered_transition = filtered.info["context_transition"]
        self.assertEqual(filtered_transition["operation"], CONTEXT_OPERATION_REPLACE)
        self.assertFalse(
            any(
                "old cache produced stale result" in message["content"]
                for message in filtered_transition["messages"]
            )
        )

    def test_native_actions_and_replacements_stay_wrapper_owned(self) -> None:
        adapter = self.make_adapter(label="coding")
        self.bind(adapter)
        native = adapter.step('shell_command {"command":"pwd"}')
        self.assertEqual(native.reward, 0.25)
        self.assertEqual(adapter.native_client.native_actions, ['shell_command {"command":"pwd"}'])
        replacement = native.info["context_transition"]["messages"]
        self.assertIn(AGEMEM_PROMPT_MARKER, replacement[0]["content"])
        evidence = native.info["wrapper_evidence"]
        self.assertEqual(evidence["event"], "native_action")
        self.assertEqual(evidence["agemem_adapter"]["memory_size_after"], 0)

    def test_truncated_native_null_reward_is_preserved_for_reschedule(self) -> None:
        class ExcludedNativeClient(FakeEnvClient):
            @property
            def sample_excluded(self) -> bool:
                return True

            def step(self, action: str) -> StepOutput:
                self.native_actions.append(action)
                return StepOutput(
                    state="infrastructure fault",
                    reward=None,
                    done=True,
                    info=build_task_neutral_transition_info(
                        env_info={
                            "truncated": True,
                            "terminal_reason": "grader_infrastructure_fault",
                        },
                        action_submission={"raw_policy_output": action},
                    ),
                )

        adapter = AgeMemEnvClientAdapter(ExcludedNativeClient())
        adapter.reset(0)
        output = adapter.step('shell_command {"command":"python train.py"}')
        self.assertIsNone(output.reward)
        self.assertTrue(output.done)
        self.assertTrue(adapter.sample_excluded)

    def test_null_native_reward_without_exclusion_is_rejected(self) -> None:
        class InvalidNullRewardClient(FakeEnvClient):
            def step(self, action: str) -> StepOutput:
                self.native_actions.append(action)
                return StepOutput(
                    state="invalid null reward",
                    reward=None,
                    done=True,
                    info=build_task_neutral_transition_info(
                        env_info={"truncated": False},
                        action_submission={"raw_policy_output": action},
                    ),
                )

        adapter = AgeMemEnvClientAdapter(InvalidNullRewardClient())
        adapter.reset(0)
        with self.assertRaisesRegex(RuntimeError, "null reward"):
            adapter.step('shell_command {"command":"python train.py"}')

    def test_native_control_has_priority_over_memory_interception(self) -> None:
        adapter = self.make_adapter()
        self.bind(adapter)
        native = adapter.native_client
        native.control_candidate = "write the required continuation"
        native.select_control = True
        candidate = adapter.policy_turn_candidate()
        self.assertEqual(candidate, native.control_candidate)
        pressure = PolicyContextPressure(100, 120, 1000, 1200, 100, 100)
        self.assertEqual(adapter.prepare_policy_turn(pressure), candidate)
        action = memory_action("Add_memory", content="must be forwarded")
        adapter.step(action)
        self.assertEqual(native.native_actions[-1], action)
        self.assertEqual(adapter.memory_size, 0)

    def test_parser_accepts_exactly_one_tool_and_rejects_mixed_output(self) -> None:
        parsed = parse_agemem_action(memory_action("Add_memory", content="x"))
        self.assertEqual(parsed.name, "Add_memory")
        with self.assertRaises(ValueError):
            parse_agemem_action(
                "prefix " + memory_action("Add_memory", content="x")
            )
        with self.assertRaises(ValueError):
            parse_agemem_action(
                "<agemem_tool_call>"
                '[{"name":"Add_memory","arguments":{"content":"x"}},'
                '{"name":"Retrieve_memory","arguments":{"query":"x"}}]'
                "</agemem_tool_call>"
            )

    def test_retrieve_top_k_requires_a_json_integer(self) -> None:
        adapter = self.make_adapter()
        self.bind(adapter)
        for invalid in (True, 1.5, "1"):
            output = adapter.step(
                memory_action("Retrieve_memory", query="fact", top_k=invalid)
            )
            payload = json.loads(output.state.split("\n", 1)[1])
            self.assertEqual(payload["error_code"], "invalid_top_k")

    def test_only_the_exact_mandatory_memory_path_is_allowed(self) -> None:
        adapter = self.make_adapter()
        self.bind(adapter)
        native = adapter.native_client
        allowed = adapter.step(
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}'
        )
        self.assertEqual(allowed.reward, 0.25)
        self.assertEqual(len(native.native_actions), 1)

        forbidden_actions = (
            'shell_command {"command":"cat .agent_memory/research.md"}',
            'shell_command {"command":"cat .agent_memory"}',
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md.bak"}',
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md/../notes"}',
        )
        for action in forbidden_actions:
            output = adapter.step(action)
            payload = json.loads(output.state.split("\n", 1)[1])
            self.assertEqual(
                payload["error_code"], "filesystem_memory_namespace_disabled"
            )
        self.assertEqual(len(native.native_actions), 1)


if __name__ == "__main__":
    unittest.main()
