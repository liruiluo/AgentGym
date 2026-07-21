from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentenv_agentmemory.domains.travel import (
    TRAVEL_MAX_STEPS_PER_PHASE,
    TravelPlannerFactory,
    _format_plan,
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

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.rows = [_travel_row(7), _travel_row(99, second=True)]
        self.tasks_path = Path(self.tempdir.name) / "travel.jsonl"
        self.tasks_path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.factory = TravelPlannerFactory(
            tasks_path=self.tasks_path,
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

    def test_reset_matches_native_id_index_and_fallback_selection(self):
        for seed, expected_id in ((99, 99), (1, 99), (12345, 7)):
            native = self.native.reset(seed=seed)
            adapted = self.driver.reset(seed)
            self.assertEqual(native["group_id"], expected_id)
            self.assertEqual(adapted.domain_evidence["source_id"], expected_id)
            self.assertEqual(adapted.phase_index, 0)
            self.assertFalse(adapted.done)

        native = self.native.reset(seed=7)
        adapted = self.driver.reset(7)
        base = native["base_person"]
        self.assertIn(base["query"], adapted.observation)
        self.assertIn(
            _format_plan(base["name"], base["daily_plans"]),
            adapted.observation,
        )
        self.assertIn(native["questions"][0]["query"], adapted.observation)
        self.assertNotIn(repr(native["answers"]), adapted.observation)

    def test_real_tool_success_and_error_results_are_exactly_preserved(self):
        self.driver.reset(7)
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
            tasks_path=self.tasks_path,
            memoryarena_root=self.root,
            database_path=TRAVEL_DATABASE_PATH,
        )
        driver = factory.create("six-tool-parity")
        driver.reset(7)
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

    def test_native_judge_reward_and_outer_phase_progress_match_adapter(self):
        row = self.rows[0]
        self.native.reset(seed=7)
        self.driver.reset(7)

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
            tasks_path=tasks_path,
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
            for row in rows:
                native_reset = native.reset(seed=row["id"])
                adapted_reset = driver.reset(row["id"])
                self.assertEqual(native_reset["group_id"], row["id"])
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
