from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agentenv_agentmemory.environment import AgentMemoryEnv, Product, ShoppingSubtask, ShoppingTask


def two_session_task() -> ShoppingTask:
    return ShoppingTask(
        task_id="buy_semantics_test",
        title="BUY semantics regression",
        source="memoryarena_test",
        subtasks=(
            ShoppingSubtask(
                instruction="Choose the requested first product.",
                candidate_products=(
                    Product("first_correct", "First correct", {}),
                    Product(
                        "first_wrong",
                        "First wrong",
                        {"compatible_tv_min": 10, "compatible_tv_max": 20},
                    ),
                ),
                target_product_id="first_correct",
            ),
            ShoppingSubtask(
                instruction="Choose the requested second product.",
                candidate_products=(
                    Product("second_correct", "Second correct", {}),
                    Product("second_wrong", "Second wrong", {}),
                ),
                target_product_id="second_correct",
            ),
        ),
    )


class BuySemanticsTests(unittest.TestCase):
    def make_env(self, buy_semantics: str | None = None) -> AgentMemoryEnv:
        return AgentMemoryEnv(tasks=[two_session_task()], buy_semantics=buy_semantics)

    def test_terminate_is_the_default_and_wrong_buy_ends_without_text_oracle(self) -> None:
        with patch.dict(os.environ, {"AGENTMEMORY_BUY_SEMANTICS": ""}):
            env = self.make_env()
        self.assertEqual(env.buy_semantics, "terminate")
        env.reset()

        observation, reward, done, _, info = env.step('BUY {"product_id": "first_wrong"}')

        self.assertEqual(reward, -0.5)
        self.assertTrue(done)
        self.assertFalse(info["episode_success"])
        self.assertEqual(info["current_subtask_index"], 0)
        self.assertEqual(info["purchase_history"][0]["product_id"], "first_wrong")
        self.assertFalse(info["purchase_history"][0]["purchase_correct"])
        self.assertNotIn("Choose the requested second product.", observation)
        self.assertNotIn("Purchase rejected", observation)
        self.assertNotIn("incorrect", observation.lower())
        self.assertNotIn("no previously purchased TV size exists in hidden bundle state", observation)
        self.assertNotIn("try a different", observation.lower())
        self.assertNotIn("Rejected product_ids", observation)
        self.assertEqual(
            info["tool_ops"],
            [
                {
                    "op": "BUY",
                    "product_id": "first_wrong",
                    "step": 1,
                    "committed": True,
                    "purchase_correct": False,
                    "outcome": "incorrect",
                    "session_advanced": False,
                    "terminal": True,
                    "memory_shaping_bonus": 0.0,
                }
            ],
        )

    def test_correct_buy_advances(self) -> None:
        env = self.make_env("terminate")
        env.reset()

        _, reward, done, _, info = env.step('BUY {"product_id": "first_correct"}')

        self.assertEqual(reward, 1.0)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 1)
        self.assertTrue(info["purchase_history"][0]["purchase_correct"])

    def test_correct_buy_needs_no_preceding_tool_action(self) -> None:
        task = two_session_task()
        task = ShoppingTask(
            **{
                **task.__dict__,
                "curriculum_flags": frozenset({"require_memory_before_source_buy"}),
            }
        )
        env = AgentMemoryEnv(tasks=[task], buy_semantics="terminate")
        env.reset()

        _, reward, done, _, info = env.step('BUY {"product_id": "first_correct"}')

        self.assertEqual(reward, 1.0)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 1)
        self.assertTrue(info["tool_ops"][0]["committed"])

    def test_buy_rejects_memory_evidence_fields(self) -> None:
        env = self.make_env("terminate")
        env.reset()

        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "first_correct", "memory_ids": ["C0"], "why": "because"}'
        )

        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 0)
        self.assertIn("BUY accepts exactly one field: product_id", observation)

    def test_ground_is_not_an_active_action(self) -> None:
        env = self.make_env("terminate")
        env.reset()

        observation, reward, done, _, info = env.step(
            'GROUND {"candidate_id": "first_correct", "memory_ids": ["C0"], "why": "because"}'
        )

        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 0)
        self.assertIn("Unsupported action 'GROUND'", observation)

    def test_add_preserves_exact_policy_authored_value(self) -> None:
        env = self.make_env("terminate")
        env.reset()
        value = "remember first_correct only"

        env.step(f'ADD {{"key": "choice", "value": "{value}"}}')

        self.assertEqual(env.long_term_memory["mem_0000"].value, value)
        self.assertNotIn("title=", env.long_term_memory["mem_0000"].value)

    def test_observation_contains_no_strategy_workflow_or_legacy_action(self) -> None:
        env = self.make_env("terminate")
        observation, _ = env.reset()

        self.assertNotIn("GROUND", observation)
        self.assertNotIn("memory_ids", observation)
        self.assertNotIn("Recommended workflow", observation)
        self.assertNotIn("ADD the selected", observation)

    def test_wrong_buy_after_progress_finishes_episode_without_advancing(self) -> None:
        env = self.make_env("terminate")
        env.reset()
        env.step('BUY {"product_id": "first_correct"}')

        observation, reward, done, _, info = env.step('BUY {"product_id": "second_wrong"}')

        self.assertEqual(reward, -0.5)
        self.assertTrue(done)
        self.assertFalse(info["episode_success"])
        self.assertEqual(info["current_subtask_index"], 1)
        self.assertEqual([item["product_id"] for item in info["purchase_history"]], ["first_correct", "second_wrong"])
        self.assertIn("Shopping episode is complete.", observation)
        self.assertNotIn("Purchase rejected", observation)

    def test_all_correct_buys_keep_success_and_final_bonus(self) -> None:
        env = self.make_env("terminate")
        env.reset()
        env.step('BUY {"product_id": "first_correct"}')

        _, reward, done, _, info = env.step('BUY {"product_id": "second_correct"}')

        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertTrue(info["episode_success"])

    def test_continue_mode_is_explicit_benchmark_replay_only(self) -> None:
        env = self.make_env("continue")
        env.reset()

        _, reward, done, _, info = env.step('BUY {"product_id": "first_wrong"}')

        self.assertEqual(reward, -0.5)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 1)
        self.assertFalse(info["tool_ops"][0]["purchase_correct"])
        self.assertTrue(info["tool_ops"][0]["session_advanced"])

    def test_retry_is_available_only_when_explicitly_requested(self) -> None:
        env = self.make_env("retry")
        env.reset()

        observation, reward, done, _, info = env.step('BUY {"product_id": "first_wrong"}')

        self.assertEqual(reward, -0.5)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 0)
        self.assertEqual(info["purchase_history"], [])
        self.assertIn("Purchase rejected", observation)


if __name__ == "__main__":
    unittest.main()
