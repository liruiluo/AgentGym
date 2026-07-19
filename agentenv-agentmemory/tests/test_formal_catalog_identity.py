from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.catalog_search import (
    CatalogSearchResponse,
    CatalogSearchResult,
    execute_catalog_search,
)
from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    FORMAL_ACTION_NAMES,
    InitialMemorySpec,
    Product,
    ShoppingSubtask,
    ShoppingTask,
    load_tasks_from_jsonl,
)
from agentenv_agentmemory.memoryarena_converter import convert_record


def asin(index: int) -> str:
    return f"B{index:09d}"


def result(index: int, *, price: float | str = 10.0) -> CatalogSearchResult:
    return CatalogSearchResult(
        asin=asin(index),
        title=f"Catalog product {index}",
        price_usd=price,
        average_rating=4.5,
        total_reviews=100 + index,
        match_score=1000 - index,
        backend_rank=index,
    )


class FakeBackend:
    def __init__(self, response: CatalogSearchResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int) -> CatalogSearchResponse:
        self.calls.append((query, limit))
        return self.response


def formal_task(*, target_asin: str = asin(15)) -> ShoppingTask:
    return ShoppingTask(
        task_id="memoryarena_identity_unit",
        title="Formal catalog identity unit",
        source="memoryarena_bundled_shopping_v1",
        subtasks=(
            ShoppingSubtask(
                instruction="Select one catalog product.",
                candidate_products=(
                    Product("ma_a_a_a", "Candidate description A", {"source_option": "a"}),
                    Product("ma_a_a_b", "Candidate description B", {"source_option": "b"}),
                ),
                target_asin=target_asin,
            ),
        ),
    )


class FormalCatalogIdentityTests(unittest.TestCase):
    def make_env(self, response: CatalogSearchResponse, **kwargs) -> tuple[AgentMemoryEnv, FakeBackend]:
        backend = FakeBackend(response)
        with patch.dict(os.environ, {}, clear=True):
            env = AgentMemoryEnv(
                tasks=[formal_task()],
                catalog_search_backend=backend,
                **kwargs,
            )
            env.reset()
        return env, backend

    def test_inner_dispatch_uses_exact_ten_action_names(self) -> None:
        self.assertEqual(
            FORMAL_ACTION_NAMES,
            (
                "ADD",
                "UPDATE",
                "DELETE",
                "RETRIEVE",
                "SUMMARY",
                "FILTER",
                "SEARCH",
                "PAGE",
                "BUY",
                "ANSWER",
            ),
        )

    def test_retrieve_typos_do_not_reach_bm25(self) -> None:
        env, _ = self.make_env(
            CatalogSearchResponse(status="empty", backend_name="fake")
        )
        with patch(
            "agentenv_agentmemory.environment.rank_memory_entries_bm25",
            side_effect=AssertionError("BM25 must not run for an invalid action name"),
        ) as ranker:
            for typo in ("RETIEVE", "RETRIVE", "RETIREVE"):
                observation, reward, done, _, info = env.step(
                    f'{typo} {{"query":"prior product","top_k":3}}'
                )
                self.assertEqual((reward, done), (-0.1, False))
                self.assertIn(f"Unsupported action '{typo}'", observation)
                self.assertEqual(info["tool_ops"], [])
                components = info["reward_components"]
                self.assertTrue(components)
                self.assertEqual(len(components), 1)
                self.assertEqual(components[0]["name"], "invalid_action")
                self.assertEqual(components[0]["raw_action"].split(None, 1)[0], typo)
                self.assertIn("Unsupported action", components[0]["error"])
                self.assertEqual({item["step"] for item in components}, {env.step_count})
                self.assertEqual({item["op"] for item in components}, {typo})
                self.assertAlmostEqual(
                    sum(float(item["value"]) for item in components),
                    reward,
                )
        ranker.assert_not_called()

    def test_reward_component_ledger_exactly_matches_each_action_reward(self) -> None:
        rows = tuple(result(index) for index in range(1, 12))
        env, _ = self.make_env(
            CatalogSearchResponse(status="ok", results=rows, backend_name="fake")
        )
        for expected_op, action in (
            ("SEARCH", 'SEARCH {"query":"catalog"}'),
            ("BUY", f'BUY {{"product_id":"{asin(1)}"}}'),
        ):
            _, reward, _, _, info = env.step(action)
            components = info["reward_components"]
            self.assertTrue(components)
            self.assertEqual({item["step"] for item in components}, {env.step_count})
            self.assertEqual({item["op"] for item in components}, {expected_op})
            self.assertAlmostEqual(
                sum(float(item["value"]) for item in components),
                reward,
            )

        with patch.dict(
            os.environ,
            {"AGENTMEMORY_MEMORY_SHAPING": "chain_v1"},
            clear=True,
        ):
            env = AgentMemoryEnv(
                tasks=[formal_task()],
                catalog_search_backend=FakeBackend(
                    CatalogSearchResponse(status="empty", backend_name="fake")
                ),
            )
            env.reset()
        _, reward, _, _, info = env.step(
            'UPDATE {"memory_id":"C0","value":"invalid context"}'
        )
        self.assertEqual(reward, -0.35)
        self.assertEqual(len(info["reward_components"]), 1)
        component = info["reward_components"][0]
        self.assertEqual(component["name"], "invalid_action")
        self.assertEqual(component["value"], -0.35)
        self.assertEqual(component["op"], "UPDATE")
        self.assertIn("Unknown memory_id", component["error"])

    def test_formal_bootstrap_rejects_memory_or_prefill_curriculum(self) -> None:
        variants = (
            replace(
                formal_task(),
                initial_memories=(
                    InitialMemorySpec(
                        key="prior_product",
                        value="environment-authored memory",
                    ),
                ),
            ),
            replace(
                formal_task(),
                curriculum_flags=frozenset({"prefill_initial_memories_active"}),
            ),
        )
        for task in variants:
            with self.subTest(task=task):
                with patch.dict(os.environ, {}, clear=True):
                    env = AgentMemoryEnv(
                        tasks=[task],
                        catalog_search_backend=FakeBackend(
                            CatalogSearchResponse(
                                status="empty",
                                backend_name="fake",
                            )
                        ),
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "must start without preloaded memories",
                    ):
                        env.reset()

    def test_pool50_page10_opaque_cursor_and_single_visible_copy(self) -> None:
        rows = tuple(result(index) for index in range(1, 26))
        env, backend = self.make_env(
            CatalogSearchResponse(status="ok", results=rows, backend_name="fake")
        )

        observation, reward, done, _, info = env.step('SEARCH {"query":"cake mix"}')
        search = info["tool_ops"][0]
        self.assertEqual(backend.calls, [("cake mix", 50)])
        self.assertEqual((reward, done), (0.0, False))
        self.assertEqual(search["pool_size"], 25)
        self.assertEqual(search["result_count"], 10)
        self.assertEqual(observation.count("- asin="), 10)
        cursor_1 = search["next_cursor"]
        self.assertRegex(cursor_1, r"^cur_[0-9a-f]{24}$")
        self.assertNotIn("cake", cursor_1)

        observation, _, _, _, info = env.step(f'PAGE {{"cursor":"{cursor_1}"}}')
        page_2 = info["tool_ops"][0]
        self.assertEqual(page_2["page"], 2)
        self.assertEqual(page_2["result_count"], 10)
        self.assertEqual(observation.count("- asin="), 10)
        cursor_2 = page_2["next_cursor"]
        self.assertNotEqual(cursor_1, cursor_2)

        _, reward, done, _, info = env.step(f'PAGE {{"cursor":"{cursor_1}"}}')
        self.assertEqual((reward, done), (-0.1, False))
        self.assertEqual(info["tool_ops"][0]["status"], "contract_error")

        observation, _, _, _, info = env.step(f'PAGE {{"cursor":"{cursor_2}"}}')
        page_3 = info["tool_ops"][0]
        self.assertEqual(page_3["result_count"], 5)
        self.assertIsNone(page_3["next_cursor"])
        self.assertEqual(observation.count("- asin="), 5)

    def test_buy_accepts_only_observed_asin_and_rejects_ma_alias(self) -> None:
        rows = tuple(result(index) for index in range(1, 12))
        env, _ = self.make_env(
            CatalogSearchResponse(status="ok", results=rows, backend_name="fake")
        )
        env.step('SEARCH {"query":"catalog"}')

        for product_id in ("ma_a_a_a", asin(11), asin(99)):
            _, reward, done, _, _ = env.step(
                f'BUY {{"product_id":"{product_id}"}}'
            )
            self.assertEqual((reward, done), (-0.1, False))

    def test_cursor_and_observed_asins_expire_on_reset(self) -> None:
        rows = tuple(result(index) for index in range(1, 12))
        env, _ = self.make_env(
            CatalogSearchResponse(status="ok", results=rows, backend_name="fake")
        )
        _, _, _, _, info = env.step('SEARCH {"query":"catalog"}')
        cursor = info["tool_ops"][0]["next_cursor"]
        env.reset()
        _, _, _, _, info = env.step('SEARCH {"query":"catalog"}')
        reset_cursor = info["tool_ops"][0]["next_cursor"]
        self.assertNotEqual(cursor, reset_cursor)

        _, reward, done, _, info = env.step(f'PAGE {{"cursor":"{cursor}"}}')
        self.assertEqual((reward, done), (-0.1, False))
        self.assertEqual(info["tool_ops"][0]["status"], "contract_error")
        env.reset()
        _, reward, done, _, _ = env.step(
            f'BUY {{"product_id":"{asin(1)}"}}'
        )
        self.assertEqual((reward, done), (-0.1, False))

    def test_observed_wrong_buy_is_terminal_minus_half_even_in_continue_mode(self) -> None:
        rows = tuple(result(index) for index in range(1, 12))
        env, _ = self.make_env(
            CatalogSearchResponse(status="ok", results=rows, backend_name="fake"),
            buy_semantics="continue",
        )
        env.step('SEARCH {"query":"catalog"}')
        _, reward, done, _, info = env.step(
            f'BUY {{"product_id":"{asin(1)}"}}'
        )
        buy = info["tool_ops"][0]
        self.assertEqual(reward, -0.5)
        self.assertTrue(done)
        self.assertTrue(buy["terminal"])
        self.assertFalse(buy["session_advanced"])
        self.assertEqual(buy["outcome"], "incorrect")

    def test_raw_target_asin_on_second_page_is_correct(self) -> None:
        rows = tuple(result(index) for index in range(1, 26))
        env, _ = self.make_env(
            CatalogSearchResponse(status="ok", results=rows, backend_name="fake")
        )
        _, _, _, _, info = env.step('SEARCH {"query":"catalog"}')
        cursor = info["tool_ops"][0]["next_cursor"]
        env.step(f'PAGE {{"cursor":"{cursor}"}}')
        _, reward, done, _, info = env.step(
            f'BUY {{"product_id":"{asin(15)}"}}'
        )
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertTrue(info["episode_success"])

    def test_statuses_and_contract_error_are_distinct(self) -> None:
        for status in ("empty", "timeout", "backend_error"):
            env, _ = self.make_env(
                CatalogSearchResponse(
                    status=status,
                    backend_name="fake",
                    error_message=status if status != "empty" else None,
                )
            )
            observation, _, done, _, info = env.step('SEARCH {"query":"catalog"}')
            self.assertFalse(done)
            self.assertIn(f"status={status}", observation)
            self.assertEqual(info["tool_ops"][0]["status"], status)

        env, _ = self.make_env(CatalogSearchResponse(status="empty", backend_name="fake"))
        _, _, _, _, info = env.step('SEARCH {"query":"x","top_k":3}')
        self.assertEqual(info["tool_ops"][0]["status"], "contract_error")

    def test_malformed_backend_numeric_field_fails_closed(self) -> None:
        backend = FakeBackend(
            CatalogSearchResponse(
                status="ok",
                results=(result(1, price="not-a-number"),),
                backend_name="fake",
            )
        )
        response = execute_catalog_search(backend, "catalog", limit=50)
        self.assertEqual(response.status, "backend_error")
        self.assertFalse(response.results)

    def test_converter_emits_raw_asin_not_fuzzy_runtime_target(self) -> None:
        record = {
            "id": "identity_0",
            "category": "unit",
            "questions": [
                "\n".join(
                    [
                        "Product 1:",
                        "### Select cake mix",
                        "**Available Options:**",
                        "- Alpha cake mix",
                        "- Beta cake mix",
                    ]
                )
            ],
            "answers": [
                {"target_asin": asin(15), "attributes": ["Beta", "cake", "mix"]}
            ],
        }
        task, report = convert_record(
            record,
            position=0,
            split_mode="ratio",
            min_match_score=1,
            ambiguous_policy="first",
            catalog_index={},
            candidate_metadata_matches={},
            enrich_candidate_metadata=False,
            candidate_metadata_min_score=90,
        )
        self.assertEqual(task["subtasks"][0]["target_asin"], asin(15))
        self.assertNotIn("target_product_id", task["subtasks"][0])
        self.assertTrue(report[0]["matched_product_id"].startswith("ma_"))

    def test_memoryarena_loader_rejects_legacy_fuzzy_target_only(self) -> None:
        record = {
            "task_id": "memoryarena_legacy",
            "title": "legacy",
            "source": "memoryarena_bundled_shopping_v0",
            "subtasks": [
                {
                    "instruction": "select",
                    "target_product_id": "ma_a",
                    "candidate_products": [
                        {"product_id": "ma_a", "title": "A", "attributes": {}}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "require raw target_asin"):
                load_tasks_from_jsonl(path)


if __name__ == "__main__":
    unittest.main()
