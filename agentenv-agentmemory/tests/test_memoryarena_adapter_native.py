from __future__ import annotations

import json
import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("AgentGym adapter tests require the 9N torch runtime.")

from agentenv.controller.types import ActionFormat
from agentenv.envs.agentmemory import (
    AGENTMEMORY_FUNCTION_DESCRIPTION,
    AgentMemoryAdapter,
    AgentMemoryEnvClient,
    extract_bare_env_action,
    parse_env_action,
)


class NativeAgentMemoryAdapterTests(unittest.TestCase):
    def test_react_accepts_original_native_actions(self) -> None:
        parsed = AgentMemoryAdapter.parse_react("Thought:\nlook\n\nAction:\nsearch[red shoes]")
        self.assertEqual(parsed.action, "search[red shoes]")
        parsed = AgentMemoryAdapter.parse_react("click[B000000001]")
        self.assertEqual(parsed.action, "click[B000000001]")

    def test_function_calling_maps_search_and_click_to_bracket_actions(self) -> None:
        search = json.dumps(
            {
                "thought": "look",
                "function_name": "search",
                "arguments": {"keywords": "red shoes"},
            }
        )
        click = json.dumps(
            {
                "thought": "open",
                "function_name": "click",
                "arguments": {"item": "B000000001"},
            }
        )
        self.assertEqual(AgentMemoryAdapter.parse_function_calling(search).action, "search[red shoes]")
        self.assertEqual(AgentMemoryAdapter.parse_function_calling(click).action, "click[B000000001]")

    def test_memory_tools_remain_uppercase_json(self) -> None:
        parsed = AgentMemoryAdapter.parse_react('ADD {"key":"k","value":"v"}')
        self.assertEqual(parsed.action, 'ADD {"key": "k", "value": "v"}')

    def test_thinking_suffix_accepts_exactly_one_action(self) -> None:
        native = AgentMemoryAdapter.parse_react(
            "<think>compare the visible candidates</think>\nclick[B000000001]"
        )
        self.assertEqual(native.action, "click[B000000001]")
        self.assertEqual(native.thought, "compare the visible candidates")

        memory = AgentMemoryAdapter.parse_react(
            '<think>save the selected product</think>\nADD {"key":"item","value":"B000000001"}'
        )
        self.assertEqual(memory.action, 'ADD {"key": "item", "value": "B000000001"}')

    def test_thinking_suffix_rejects_extra_or_unclosed_text(self) -> None:
        for response in (
            "<think>reason</think>\nclick[B000000001]\nextra",
            "<think>reason\nclick[B000000001]",
            "unwrapped reasoning\nclick[B000000001]",
        ):
            with self.subTest(response=response):
                self.assertEqual(AgentMemoryAdapter.parse_react(response).action, "")

    def test_surrogate_actions_and_multi_actions_are_rejected(self) -> None:
        for action in [
            'SEARCH {"query":"x"}',
            'BUY {"product_id":"x"}',
            'GROUND {"candidate_id":"x"}',
            "search[x]\nclick[y]",
            "click[x]\nextra",
        ]:
            with self.subTest(action=action):
                self.assertEqual(extract_bare_env_action(action), "")
                with self.assertRaises(ValueError):
                    parse_env_action(action)

    def test_formal_prompt_and_functions_have_no_surrogate_purchase(self) -> None:
        function_names = {item["name"] for item in AGENTMEMORY_FUNCTION_DESCRIPTION}
        self.assertEqual(
            function_names,
            {"search", "click", "add", "update", "delete", "retrieve", "summary", "filter"},
        )
        prompt = AgentMemoryAdapter.conversation_start_dict[ActionFormat.REACT][0]["value"]
        self.assertIn("original MemoryArena WebShop", prompt)
        self.assertIn("click[Buy Now]", prompt)
        self.assertNotIn(" BUY ", prompt)
        self.assertNotIn("GROUND", prompt)

    def test_client_sends_unparsed_action_to_environment_for_authoritative_rejection(self) -> None:
        class RejectingAdapter:
            @staticmethod
            def action_parser(action, action_format):
                del action, action_format
                raise ValueError("malformed policy action")

        client = AgentMemoryEnvClient.__new__(AgentMemoryEnvClient)
        client.adapter_cls = RejectingAdapter
        client.action_format = ActionFormat.REACT
        client.metadata = {"surface": "memoryarena_webshop_native_v1"}
        client.info = {
            "observation": "old observation",
            "reward": 0.0,
            "done": False,
            "env_info": {"step": 3},
            "metadata": client.metadata,
        }
        raw_action = "not a valid action"
        component = {
            "name": "invalid_action",
            "value": -0.01,
            "op": "INVALID",
            "step": 4,
        }

        def post(path, data):
            self.assertEqual(path, "step")
            self.assertEqual(data, {"action": raw_action})
            return {
                "observation": "authoritative invalid action response",
                "reward": -0.01,
                "done": False,
                "info": {"step": 4, "reward_components": [component]},
            }

        client.post = post
        output = client.step(raw_action)

        self.assertEqual(output.state, "authoritative invalid action response")
        self.assertEqual(output.reward, -0.01)
        self.assertFalse(output.done)
        self.assertEqual(client.info["env_info"]["step"], 4)
        self.assertEqual(client.info["env_info"]["reward_components"], [component])


if __name__ == "__main__":
    unittest.main()
