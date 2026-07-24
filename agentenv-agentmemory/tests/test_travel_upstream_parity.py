from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from agentenv_agentmemory.domains.travel import (
    TRAVEL_MAX_STEPS_PER_PHASE,
    TravelPlannerFactory,
    _format_plan,
)
from agentenv_agentmemory.domains.memoryarena_dataset import (
    attest_injected_test_dataset,
)


MEMORYARENA_ROOT = os.environ.get("MEMORYARENA_ROOT")
TRAVEL_DATABASE_PATH = os.environ.get("MEMORYARENA_TRAVEL_DATABASE_PATH")


def _travel_row(task_id: int, *, second: bool = False) -> dict:
    base_name = "Marcus" if second else "Jennifer"
    first_name = "Noah" if second else "Eric"
    second_name = "Mia" if second else "Dana"
    city = "Austin" if second else "Dallas"
    return {
        "id": task_id,
        "base_person": {
            "name": base_name,
            "query": f"I am {base_name}. Base itinerary.",
            "daily_plans": [
                {
                    "days": 1,
                    "current_city": city,
                    "transportation": "-",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "-",
                    "accommodation": "-",
                }
            ],
        },
        "questions": [
            {
                "round_idx": 1,
                "name": first_name,
                "query": f"I am {first_name}. Plan the first trip.",
            },
            {
                "round_idx": 2,
                "name": second_name,
                "query": f"I am {second_name}. Plan the second trip.",
            },
        ],
        "answers": [
            {
                "round_idx": 1,
                "daily_plans": [
                    {
                        "days": 1,
                        "current_city": city,
                        "transportation": "-",
                        "breakfast": "-",
                        "attraction": "-",
                        "lunch": "-",
                        "dinner": "-",
                        "accommodation": "-",
                    }
                ],
            },
            {
                "round_idx": 2,
                "daily_plans": [
                    {
                        "days": 1,
                        "current_city": city,
                        "transportation": "-",
                        "breakfast": "-",
                        "attraction": "-",
                        "lunch": "-",
                        "dinner": "-",
                        "accommodation": "-",
                    }
                ],
            },
        ],
    }


@unittest.skipUnless(
    MEMORYARENA_ROOT,
    "set MEMORYARENA_ROOT to run direct MemoryArena parity replay",
)
class TravelUpstreamParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(MEMORYARENA_ROOT).expanduser().resolve()
        sys.path.insert(0, str(cls.root))
        cls._installed_datasets_stub = False
        try:
            import datasets  # noqa: F401
        except ImportError:
            datasets_stub = types.ModuleType("datasets")

            def unavailable_dataset(*args, **kwargs):
                raise RuntimeError("datasets is required for the opt-in 270-row smoke")

            datasets_stub.load_dataset = unavailable_dataset
            sys.modules["datasets"] = datasets_stub
            cls._installed_datasets_stub = True
        cls.travel_module = importlib.import_module("env.env_systems.travel_env")
        cls.executor_module = importlib.import_module(
            "env.env_systems.travel_planner_env.tool_executor"
        )
        cities_module = importlib.import_module(
            "env.env_systems.travel_planner_env.tools.cities"
        )
        cls.executor = cls.executor_module.ToolExecutor.__new__(
            cls.executor_module.ToolExecutor
        )
        database = cls.root / "env/env_systems/travel_planner_env/database"
        cls.executor.db_path = str(database)
        cls.executor.cities = cities_module.Cities(
            path=str(database / "background/citySet_with_states.txt")
        )

    @classmethod
    def tearDownClass(cls):
        if sys.path and sys.path[0] == str(cls.root):
            sys.path.pop(0)
        if cls._installed_datasets_stub:
            sys.modules.pop("datasets", None)

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.rows = [_travel_row(7), _travel_row(99, second=True)]
        self.tasks_path = Path(self.tempdir.name) / "travel.jsonl"
        self.tasks_path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.factory = TravelPlannerFactory(
            contract_mode="paper_eval",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="group_travel_planner",
            ),
            memoryarena_root=self.root,
            tool_executor=self.executor,
        )
        self.driver = self.factory.create("parity")
        with mock.patch.object(
            self.travel_module,
            "load_travel_data",
            return_value=self.rows,
        ):
            self.native = self.travel_module.TravelPlannerEnvironment(
                {"judgement_mode": "none"}
            )

    def tearDown(self):
        self.native.close()
        self.driver.close()
        self.tempdir.cleanup()

    def test_reset_uses_dataset_position_and_records_source_id(self):
        for position, source_id in ((0, 7), (1, 99)):
            native = self.native.reset(seed=source_id)
            adapted = self.driver.reset(position)
            expected_id = source_id
            self.assertEqual(native["group_id"], expected_id)
            self.assertEqual(adapted.domain_evidence["source_id"], expected_id)
            self.assertEqual(adapted.domain_evidence["dataset_position"], position)
            self.assertEqual(adapted.phase_index, 0)
            self.assertFalse(adapted.done)
        for invalid in (-1, 2):
            with self.assertRaisesRegex(IndexError, "position out of range"):
                self.driver.reset(invalid)

        native = self.native.reset(seed=7)
        adapted = self.driver.reset(0)
        base = native["base_person"]
        self.assertIn(base["query"], adapted.observation)
        self.assertIn(
            _format_plan(base["name"], base["daily_plans"]),
            adapted.observation,
        )
        self.assertIn(native["questions"][0]["query"], adapted.observation)
        self.assertNotIn(repr(native["answers"]), adapted.observation)

    def test_real_tool_success_and_error_results_are_exactly_preserved(self):
        self.driver.reset(0)
        native_result = self.executor.execute("CitySearch", {"state": "Texas"})
        adapted = self.driver.step('CitySearch {"state": "Texas"}', env_step=1)
        self.assertEqual(adapted.reward, 0.0)
        self.assertFalse(adapted.done)
        self.assertEqual(adapted.action_execution["status"], "executed")
        self.assertIn(f"Tool result (CitySearch):\n{native_result}\n\n", adapted.observation)

        arguments = {
            "origin": "Dallas",
            "destination": "Austin",
            "date": "2022-03-01",
        }
        native_error = self.executor.execute("FlightSearch", arguments)
        self.assertTrue(native_error.startswith("Error executing FlightSearch:"))
        adapted_error = self.driver.step(
            f"FlightSearch {json.dumps(arguments)}",
            env_step=2,
        )
        self.assertEqual(adapted_error.reward, 0.0)
        self.assertFalse(adapted_error.done)
        self.assertEqual(
            adapted_error.action_execution["status"],
            "executed_with_error",
        )
        self.assertIn(f"Tool result (FlightSearch):\n{native_error}\n\n", adapted_error.observation)

    @unittest.skipUnless(
        TRAVEL_DATABASE_PATH,
        "set MEMORYARENA_TRAVEL_DATABASE_PATH for the complete six-tool replay",
    )
    def test_factory_initializes_and_preserves_all_six_native_tools(self):
        factory = TravelPlannerFactory(
            contract_mode="paper_eval",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="group_travel_planner",
            ),
            memoryarena_root=self.root,
            database_path=TRAVEL_DATABASE_PATH,
        )
        driver = factory.create("six-tool-parity")
        driver.reset(0)
        cases = (
            (
                "FlightSearch",
                {
                    "origin": "Grand Junction",
                    "destination": "Denver",
                    "date": "2022-04-04",
                },
            ),
            ("RestaurantSearch", {"city": "Dallas"}),
            ("AccommodationSearch", {"city": "Dallas"}),
            ("AttractionSearch", {"city": "Dallas"}),
            (
                "DistanceMatrix",
                {
                    "origin": "Dallas",
                    "destination": "Austin",
                    "mode": "driving",
                },
            ),
            ("CitySearch", {"state": "Texas"}),
        )
        for env_step, (op, arguments) in enumerate(cases, start=1):
            native_result = factory.tool_executor.execute(op, arguments)
            self.assertFalse(native_result.startswith("Error"))
            adapted = driver.step(f"{op} {json.dumps(arguments)}", env_step=env_step)
            self.assertEqual(adapted.reward, 0.0)
            self.assertFalse(adapted.done)
            self.assertEqual(adapted.action_execution["status"], "executed")
            self.assertIn(f"Tool result ({op}):\n{native_result}\n\n", adapted.observation)
        driver.close()

    def test_paper_eval_native_judge_and_continuation_match_upstream(self):
        self.assertEqual(self.factory.contract_mode, "paper_eval")
        self.assertEqual(
            self.factory.metadata()["wrong_submission_semantics"],
            "continue_to_next_traveler",
        )
        row = self.rows[0]
        self.native.reset(seed=7)
        self.driver.reset(0)

        first_answer = row["answers"][0]
        first_plan = _format_plan("Eric", first_answer["daily_plans"])
        native_observation, native_reward, native_info = self.native.step(
            first_plan,
            ground_truth={
                "name": "Eric",
                "daily_plans": first_answer["daily_plans"],
                "judgement_mode": "none",
            },
            need_judge=True,
        )
        adapted = self.driver.step(
            f'SUBMIT_PLAN {json.dumps({"plan": first_plan})}',
            env_step=1,
        )
        self.assertEqual(native_observation["reward"], native_reward)
        self.assertEqual(native_info, {})
        self.assertEqual(adapted.reward, float(native_reward))
        self.assertEqual(adapted.phase_index, 1)
        self.assertFalse(adapted.done)

        second_answer = row["answers"][1]
        wrong_plan = "=== Dana's Plan ===\nDay 1:\nCurrent City: nowhere"
        native_observation, native_reward, native_info = self.native.step(
            wrong_plan,
            ground_truth={
                "name": "Dana",
                "daily_plans": second_answer["daily_plans"],
                "judgement_mode": "none",
            },
            need_judge=True,
        )
        adapted = self.driver.step(
            f'SUBMIT_PLAN {json.dumps({"plan": wrong_plan})}',
            env_step=2,
        )
        self.assertEqual(native_observation["reward"], native_reward)
        self.assertEqual(native_info, {})
        self.assertEqual(adapted.reward, float(native_reward))
        self.assertEqual(adapted.phase_index, 2)
        self.assertTrue(adapted.done)
        self.assertFalse(adapted.episode_success)
        self.assertEqual(adapted.status, "completed_with_errors")

    def test_online_current_city_reward_is_separate_from_paper_metrics(self):
        row = self.rows[0]
        first_answer = row["answers"][0]
        second_answer = row["answers"][1]
        wrong_city_plan = _format_plan(
            "Eric",
            first_answer["daily_plans"],
        ).replace("Current City: Dallas", "Current City: nowhere")
        exact_second_plan = _format_plan("Dana", second_answer["daily_plans"])

        self.driver.reset(0)
        online = self.driver.step(
            f'SUBMIT_PLAN {json.dumps({"plan": wrong_city_plan})}',
            env_step=1,
        )
        self.assertEqual(online.reward, 0.0)
        self.assertEqual(
            online.reward_components[0]["name"],
            "travel_plan_incorrect",
        )

        paper = self.factory.evaluate_paper_predictions(
            {
                (7, 1): self.factory.parse_submitted_plan(
                    wrong_city_plan,
                    "Eric",
                ),
                (7, 2): self.factory.parse_submitted_plan(
                    exact_second_plan,
                    "Dana",
                ),
            }
        )
        self.assertEqual(paper["ps"], 100.0)
        self.assertEqual(paper["sr"], 100.0)
        self.assertNotIn("current_city", paper["slots"])
        self.assertTrue(paper["online_reward_is_separate"])

    def test_paper_parser_matches_official_header_independent_parser(self):
        answer = self.rows[0]["answers"][0]
        official_body = "\n".join(
            _format_plan("Eric", answer["daily_plans"]).splitlines()[1:]
        )
        parsed = self.factory.parse_submitted_plan(
            "=== Wrong Name's Plan ===\n" + official_body,
            "Eric",
        )
        self.assertEqual(parsed, self.factory.paper_evaluator.parse_prediction(official_body))
        self.assertTrue(parsed)
        self.assertEqual(parsed[0]["day"], 1)

    def test_terminal_transition_emits_official_group_contribution(self):
        row = self.rows[0]
        self.driver.reset(0)
        first = _format_plan("Eric", row["answers"][0]["daily_plans"])
        second = _format_plan("Dana", row["answers"][1]["daily_plans"])
        self.driver.step(
            f'SUBMIT_PLAN {json.dumps({"plan": first})}',
            env_step=1,
        )
        terminal = self.driver.step(
            f'SUBMIT_PLAN {json.dumps({"plan": second})}',
            env_step=2,
        )
        ledger = terminal.domain_evidence["paper_evaluation"]
        self.assertEqual(
            ledger["metric_contract"],
            "memoryarena_travel_eval_py_ps_sps_sr_v1",
        )
        self.assertEqual(ledger["dataset_scope"], "injected_test_fixture")
        self.assertEqual(ledger["source_id"], 7)
        self.assertTrue(ledger["complete"])
        self.assertEqual(ledger["full_pass_people"], 2)
        self.assertEqual(ledger["total_people"], 2)
        self.assertTrue(ledger["group_success"])
        self.assertEqual(ledger["constraint_people"], 0)
        self.assertIsNone(ledger["group_constraint_rate"])

    @unittest.skipUnless(
        os.environ.get("MEMORYARENA_FULL_TRAVEL_SMOKE") == "1",
        "set MEMORYARENA_FULL_TRAVEL_SMOKE=1 for the 270-group replay",
    )
    def test_all_270_groups_match_native_oracle_replay(self):
        rows = self.travel_module.load_travel_data()
        self.assertEqual(len(rows), 270)
        tasks_path = Path(self.tempdir.name) / "all_travel.jsonl"
        tasks_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        factory = TravelPlannerFactory(
            contract_mode="paper_eval",
            tasks_path=tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                tasks_path,
                config="group_travel_planner",
            ),
            memoryarena_root=self.root,
            tool_executor=self.executor,
        )
        self.assertEqual(factory.max_phase_count, 8)
        self.assertEqual(
            factory.contract.max_steps,
            8 * TRAVEL_MAX_STEPS_PER_PHASE,
        )
        driver = factory.create("full-parity")
        with mock.patch.object(
            self.travel_module,
            "load_travel_data",
            return_value=rows,
        ):
            native = self.travel_module.TravelPlannerEnvironment(
                {"judgement_mode": "none"}
            )

        phase_count = 0
        try:
            for position, row in enumerate(rows):
                native_reset = native.reset(seed=row["id"])
                adapted_reset = driver.reset(position)
                self.assertEqual(native_reset["group_id"], row["id"])
                self.assertEqual(
                    adapted_reset.domain_evidence["dataset_position"],
                    position,
                )
                self.assertEqual(
                    adapted_reset.domain_evidence["source_id"],
                    row["id"],
                )
                for question, answer in zip(row["questions"], row["answers"]):
                    name = question["name"]
                    plan = _format_plan(name, answer["daily_plans"])
                    _, native_reward, native_info = native.step(
                        plan,
                        ground_truth={
                            "name": name,
                            "daily_plans": answer["daily_plans"],
                            "judgement_mode": "none",
                        },
                        need_judge=True,
                    )
                    adapted = driver.step(
                        f'SUBMIT_PLAN {json.dumps({"plan": plan})}',
                        env_step=phase_count + 1,
                    )
                    self.assertEqual(native_info, {})
                    self.assertEqual(adapted.reward, float(native_reward))
                    self.assertEqual(adapted.reward, 1.0)
                    phase_count += 1
                self.assertTrue(adapted.done)
                self.assertTrue(adapted.episode_success)
                self.assertEqual(adapted.phase_index, len(row["questions"]))
        finally:
            native.close()
            driver.close()
        self.assertGreater(phase_count, 270)


if __name__ == "__main__":
    unittest.main()
