from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    Product,
    ShoppingSubtask,
    ShoppingTask,
)


def cross_session_task() -> ShoppingTask:
    return ShoppingTask(
        task_id="repeated_empty_retrieve_unit",
        title="Repeated empty retrieval reward regression",
        source="cross_session_unit",
        memory_dependency="cross_session_unit",
        subtasks=(
            ShoppingSubtask(
                instruction="Select the source product.",
                candidate_products=(
                    Product("src_a", "Source A", {}),
                    Product("src_b", "Source B", {}),
                ),
                target_product_id="src_a",
            ),
            ShoppingSubtask(
                instruction="Select a product compatible with the previous product.",
                candidate_products=(
                    Product("dep_a", "Dependent A", {}),
                    Product("dep_b", "Dependent B", {}),
                ),
                target_product_id="dep_a",
            ),
        ),
    )


def component_names(info: dict) -> list[str]:
    return [component["name"] for component in info["reward_components"]]


class RepeatedEmptyRetrievePenaltyTests(unittest.TestCase):
    def make_env(self, shaping: str = "chain_v1") -> AgentMemoryEnv:
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_MEMORY_SHAPING": shaping},
            clear=True,
        ):
            env = AgentMemoryEnv(tasks=[cross_session_task()])
            env.reset()
        return env

    def retrieve(self, env: AgentMemoryEnv, query: str):
        return env.step(
            f'RETRIEVE {{"query": {json.dumps(query)}, "top_k": 3}}'
        )

    def assert_reward_ledger(self, reward: float, info: dict) -> None:
        self.assertAlmostEqual(
            sum(float(component["value"]) for component in info["reward_components"]),
            reward,
        )

    def test_first_empty_retrieve_in_source_session_is_neutral(self) -> None:
        env = self.make_env()

        observation, reward, done, _, info = self.retrieve(env, "cake mix")

        self.assertEqual((reward, done), (0.0, False))
        self.assertNotIn("memory_retrieve_empty_repeat_same_query_noop", component_names(info))
        self.assertNotIn("returned no matches more than once", observation)
        self.assert_reward_ledger(reward, info)

    def test_second_identical_empty_retrieve_in_source_session_is_minus_point04(self) -> None:
        env = self.make_env()
        self.retrieve(env, "cake mix")

        observation, reward, done, _, info = self.retrieve(env, "cake mix")

        self.assertEqual((reward, done), (-0.04, False))
        self.assertIn("returned no matches more than once", observation)
        self.assertEqual(
            component_names(info).count("memory_retrieve_empty_repeat_same_query_noop"),
            1,
        )
        component = next(
            item
            for item in info["reward_components"]
            if item["name"] == "memory_retrieve_empty_repeat_same_query_noop"
        )
        self.assertEqual(component["repeat_count"], 2)
        self.assertEqual(component["query"], "cake mix")
        self.assertEqual(component["op"], "RETRIEVE")
        self.assertEqual(component["step"], 2)
        self.assert_reward_ledger(reward, info)

    def test_third_and_later_identical_empty_retrieves_use_constant_penalty(self) -> None:
        env = self.make_env()

        rewards = [self.retrieve(env, "cake mix")[1] for _ in range(4)]

        self.assertEqual(rewards, [0.0, -0.04, -0.04, -0.04])

    def test_empty_query_identity_is_case_and_whitespace_normalized(self) -> None:
        env = self.make_env()
        self.retrieve(env, "  Cake   MIX ")

        _, reward, _, _, info = self.retrieve(env, "cake mix")

        self.assertEqual(reward, -0.04)
        self.assertEqual(
            next(
                item["repeat_count"]
                for item in info["reward_components"]
                if item["name"] == "memory_retrieve_empty_repeat_same_query_noop"
            ),
            2,
        )

    def test_different_empty_query_gets_its_own_neutral_first_attempt(self) -> None:
        env = self.make_env()
        self.retrieve(env, "cake mix")
        self.retrieve(env, "cake mix")

        first_other = self.retrieve(env, "facial cleanser")
        second_other = self.retrieve(env, "facial cleanser")

        self.assertEqual(first_other[1], 0.0)
        self.assertEqual(second_other[1], -0.04)
        self.assertNotIn(
            "memory_retrieve_empty_repeat_same_query_noop",
            component_names(first_other[4]),
        )

    def test_episode_reset_clears_empty_query_counts(self) -> None:
        env = self.make_env()
        self.retrieve(env, "cake mix")
        self.assertEqual(self.retrieve(env, "cake mix")[1], -0.04)

        env.reset()
        _, reward, _, _, info = self.retrieve(env, "cake mix")

        self.assertEqual(reward, 0.0)
        self.assertNotIn("memory_retrieve_empty_repeat_same_query_noop", component_names(info))

    def test_successful_session_advance_clears_empty_query_counts(self) -> None:
        env = self.make_env()
        self.retrieve(env, "cake mix")
        self.assertEqual(self.retrieve(env, "cake mix")[1], -0.04)

        _, _, buy_done, _, buy_info = env.step('BUY {"product_id": "src_a"}')
        self.assertFalse(buy_done)
        self.assertTrue(buy_info["tool_ops"][0]["session_advanced"])
        _, reward, _, _, info = self.retrieve(env, "cake mix")

        self.assertEqual(env.current_subtask_index, 1)
        self.assertEqual(reward, 0.0)
        self.assertNotIn("memory_retrieve_empty_repeat_same_query_noop", component_names(info))

    def test_repeated_empty_penalty_also_applies_in_dependent_session(self) -> None:
        env = self.make_env()
        env.step('BUY {"product_id": "src_a"}')

        first = self.retrieve(env, "prior product")
        second = self.retrieve(env, "prior product")

        self.assertEqual(first[1], 0.0)
        self.assertEqual(second[1], -0.04)
        self.assertIn("memory_retrieve_empty_in_dependent_session", component_names(second[4]))
        self.assertIn("memory_retrieve_empty_repeat_same_query_noop", component_names(second[4]))
        self.assert_reward_ledger(second[1], second[4])

    def test_repeated_empty_penalty_does_not_require_ltm_to_be_empty(self) -> None:
        env = self.make_env()
        env.step('ADD {"key": "unrelated", "value": "stored context"}')

        with patch("agentenv_agentmemory.environment.rank_memory_entries_bm25", return_value=[]):
            first = self.retrieve(env, "missing memory")
            second = self.retrieve(env, "missing memory")

        self.assertTrue(env.long_term_memory)
        self.assertEqual(first[1], 0.0)
        self.assertEqual(second[1], -0.04)

    def test_prior_nonempty_same_query_does_not_consume_first_empty_attempt(self) -> None:
        env = self.make_env()
        env.step('ADD {"key": "note", "value": "alpha beta"}')
        nonempty = self.retrieve(env, "alpha beta")
        self.assertEqual(nonempty[4]["memory_ops"][0]["retrieved_count"], 1)
        env.step('DELETE {"memory_id": "mem_0000"}')

        first_empty = self.retrieve(env, " ALPHA   BETA ")
        second_empty = self.retrieve(env, "alpha beta")

        self.assertEqual(first_empty[1], 0.0)
        self.assertEqual(second_empty[1], -0.04)
        self.assertNotIn(
            "memory_retrieve_empty_repeat_same_query_noop",
            component_names(first_empty[4]),
        )

    def test_nonempty_source_retrieve_repeat_keeps_existing_logic(self) -> None:
        env = self.make_env()
        env.step('ADD {"key": "chosen", "value": "src_a is selected"}')

        first = self.retrieve(env, "src_a")
        second = self.retrieve(env, "src_a")

        self.assertEqual(first[1], 0.0)
        self.assertEqual(second[1], -0.05)
        self.assertIn("memory_retrieve_source_same_session_check", component_names(first[4]))
        self.assertIn(
            "memory_retrieve_source_same_session_repeat_noop",
            component_names(second[4]),
        )
        self.assertNotIn(
            "memory_retrieve_empty_repeat_same_query_noop",
            component_names(second[4]),
        )

    def test_nonempty_dependent_retrieve_keeps_existing_reward_and_repeat_logic(self) -> None:
        env = self.make_env()
        env.step('ADD {"key": "chosen", "value": "src_a is selected"}')
        env.step('BUY {"product_id": "src_a"}')

        first = self.retrieve(env, "src_a")
        second = self.retrieve(env, "src_a")

        self.assertEqual(first[1], 0.06)
        self.assertEqual(second[1], -0.03)
        self.assertIn(
            "memory_retrieve_nonempty_before_dependent_buy",
            component_names(first[4]),
        )
        self.assertIn(
            "memory_retrieve_nonempty_repeat_same_session",
            component_names(second[4]),
        )
        self.assertNotIn(
            "memory_retrieve_empty_repeat_same_query_noop",
            component_names(second[4]),
        )

    def test_shaping_off_does_not_add_a_blanket_retrieve_tax(self) -> None:
        env = self.make_env("off")

        results = [self.retrieve(env, "cake mix") for _ in range(3)]

        self.assertEqual([result[1] for result in results], [0.0, 0.0, 0.0])
        for _, reward, _, _, info in results:
            self.assertNotIn(
                "memory_retrieve_empty_repeat_same_query_noop",
                component_names(info),
            )
            self.assert_reward_ledger(reward, info)


if __name__ == "__main__":
    unittest.main()
