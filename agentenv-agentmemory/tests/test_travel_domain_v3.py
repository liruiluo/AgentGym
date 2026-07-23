from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.domains.travel import (
    TRAVEL_CONTRACTS,
    TRAVEL_DATABASE_ASSET_SPECS,
    TRAVEL_MAX_STEPS_PER_PHASE,
    TRAVEL_SURFACES,
    TRAVEL_UPSTREAM_RELATIVE_PATHS,
    TravelPaperEvaluator,
    TravelPlannerDriver,
    TravelPlannerFactory,
    attest_travel_database,
    attest_travel_upstream,
    load_travel_tasks,
)
from agentenv_agentmemory.domains.memoryarena_dataset import (
    attest_injected_test_dataset,
)
from agentenv_agentmemory.runtime.memory import MemoryRewardPolicy
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


class FakeTravelTools:
    def __init__(self, result="flight F123 at 09:00", error=None):
        self.calls = []
        self.result = result
        self.error = error

    def execute(self, op, payload):
        self.calls.append((op, dict(payload)))
        if self.error is not None:
            raise self.error
        return self.result


class FakePaperParser:
    @staticmethod
    def parse_plan_text(plan):
        return [{"day": 1, "answer": plan}]


class FakePaperEval:
    SLOTS = ("answer",)

    @staticmethod
    def find_constraint_slots(ground_truth, base_plans, source_id, round_index):
        return {(None, "answer")}

    @staticmethod
    def check_slot_pass(truth, prediction, day_index, slot):
        return bool(prediction) and truth[0]["secret"] == prediction[0]["answer"]

    @staticmethod
    def check_person_full_pass(truth, prediction):
        return bool(prediction) and truth[0]["secret"] == prediction[0]["answer"]


def exact_judge(plan, name, ground_truth_plans):
    return plan == ground_truth_plans[0]["secret"] and name in {"Eric", "Dana"}


class TravelDomainV3Test(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tasks_path = Path(self.tempdir.name) / "travel.jsonl"
        payload = {
            "id": 7,
            "base_person": {
                "name": "Jennifer",
                "query": "base query",
                "daily_plans": [
                    {
                        "days": 1,
                        "current_city": "Base City",
                        "transportation": "F000",
                        "breakfast": "-",
                        "attraction": "-",
                        "lunch": "-",
                        "dinner": "-",
                        "accommodation": "-",
                    }
                ],
            },
            "questions": ["I am Eric. first query", "I am Dana. second query"],
            "answers": [[{"secret": "SECRET_A"}], [{"secret": "SECRET_B"}]],
        }
        self.tasks_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        self.tools = FakeTravelTools()
        self.factory = TravelPlannerFactory(
            contract_mode="failfast",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="group_travel_planner",
            ),
            tool_executor=self.tools,
            judge=exact_judge,
        )
        self.wrapper = DomainEnvWrapper(
            self.factory,
            reward_policy=MemoryRewardPolicy(
                first_add=0.1,
                first_later_phase_retrieve=0.1,
                exact_repeat=-0.01,
            ),
        )
        self.created = self.wrapper.create()
        self.env_id = self.created["id"]

    def tearDown(self):
        if self.env_id in self.wrapper.envs:
            self.wrapper.close(self.env_id)
        self.tempdir.cleanup()

    def test_reset_exposes_base_and_question_but_not_private_answers(self):
        observation = self.created["observation"]
        self.assertIn("Fixed base traveler", observation)
        self.assertIn("I am Eric. first query", observation)
        self.assertNotIn("SECRET_A", observation)
        self.assertNotIn("SECRET_B", observation)
        self.assertNotIn("answers", repr(self.wrapper.metadata()).lower())
        self.assertEqual(
            self.wrapper.metadata()["wrong_submission_semantics"],
            "terminal_failure_without_phase_advance",
        )
        self.assertEqual(self.wrapper.metadata()["contract_mode"], "failfast")
        self.assertEqual(
            self.wrapper.metadata()["surface"], TRAVEL_SURFACES["failfast"]
        )
        self.assertFalse(self.wrapper.metadata()["paper_evaluation"]["available"])
        self.assertFalse(
            self.wrapper.metadata()["paper_evaluation"]["canonical_semantics"]
        )
        self.assertIsNone(self.factory.paper_evaluator)
        with self.assertRaisesRegex(RuntimeError, "paper evaluation requires"):
            self.factory.evaluate_paper_predictions({})
        self.assertEqual(
            self.wrapper.metadata()["max_actions_per_phase"],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )
        self.assertFalse(
            self.wrapper.metadata()["native_agent_turn_budget_equivalent"]
        )
        self.assertFalse(
            self.wrapper.metadata()["action_granularity"][
                "upstream_batched_model_turn_parity"
            ]
        )
        self.assertEqual(self.wrapper.metadata()["max_episode_steps"], 60)
        prompt = self.wrapper.metadata()["system_prompt"]
        for slot in (
            "Current City",
            "Transportation",
            "Breakfast",
            "Attraction",
            "Lunch",
            "Dinner",
            "Accommodation",
        ):
            self.assertIn(slot, prompt)

    def test_two_explicit_contracts_have_distinct_surfaces_and_metadata(self):
        self.assertEqual(set(TRAVEL_SURFACES), {"failfast", "paper_eval"})
        self.assertEqual(set(TRAVEL_CONTRACTS), {"failfast", "paper_eval"})
        self.assertNotEqual(
            TRAVEL_CONTRACTS["failfast"].contract_id,
            TRAVEL_CONTRACTS["paper_eval"].contract_id,
        )
        self.assertNotEqual(
            TRAVEL_CONTRACTS["failfast"].sha256,
            TRAVEL_CONTRACTS["paper_eval"].sha256,
        )
        paper_factory = TravelPlannerFactory(
            contract_mode="paper_eval",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="group_travel_planner",
            ),
            tool_executor=self.tools,
            judge=exact_judge,
        )
        metadata = paper_factory.metadata()
        self.assertEqual(paper_factory.surface, TRAVEL_SURFACES["paper_eval"])
        self.assertEqual(metadata["contract_mode"], "paper_eval")
        self.assertEqual(
            metadata["wrong_submission_semantics"],
            "continue_to_next_traveler",
        )
        self.assertEqual(
            metadata["phase_timeout_semantics"],
            "record_empty_incorrect_plan_and_advance",
        )
        self.assertTrue(metadata["paper_evaluation"]["canonical_semantics"])
        self.assertFalse(metadata["paper_evaluation"]["available"])
        self.assertFalse(metadata["paper_evaluation"]["paper_panel_complete"])
        self.assertFalse(metadata["paper_evaluation"]["paper_column_eligible"])
        self.assertFalse(
            metadata["action_granularity"]["upstream_batched_model_turn_parity"]
        )
        with (
            patch(
                "agentenv_agentmemory.domains.travel.attest_travel_upstream",
                return_value={"mode": "injected_test_upstream"},
            ),
            patch(
                "agentenv_agentmemory.domains.travel._load_upstream",
                return_value=(object(), None, FakePaperEval, FakePaperParser),
            ),
        ):
            fixture_with_evaluator = TravelPlannerFactory(
                contract_mode="paper_eval",
                tasks_path=self.tasks_path,
                dataset_provenance=attest_injected_test_dataset(
                    self.tasks_path,
                    config="group_travel_planner",
                ),
                memoryarena_root=self.tempdir.name,
                tool_executor=self.tools,
                judge=exact_judge,
            )
        fixture_metadata = fixture_with_evaluator.metadata()["paper_evaluation"]
        self.assertTrue(fixture_metadata["available"])
        self.assertFalse(fixture_metadata["paper_panel_complete"])
        self.assertFalse(fixture_metadata["paper_column_eligible"])
        tasks = load_travel_tasks(self.tasks_path)
        paper_evaluator = TravelPaperEvaluator(
            tasks=tasks,
            paper_eval_module=FakePaperEval,
            paper_parser_module=FakePaperParser,
            dataset_scope="injected_test_fixture",
        )
        with self.assertRaisesRegex(ValueError, "cannot attach a paper evaluator"):
            TravelPlannerDriver(
                contract_mode="failfast",
                tasks=tasks,
                tool_executor=self.tools,
                judge=exact_judge,
                env_uid="invalid-failfast-paper",
                paper_evaluator=paper_evaluator,
            )
        with self.assertRaisesRegex(ValueError, "contract_mode"):
            TravelPlannerFactory(
                contract_mode="ambiguous",
                tasks_path=self.tasks_path,
                dataset_provenance=attest_injected_test_dataset(
                    self.tasks_path,
                    config="group_travel_planner",
                ),
                tool_executor=self.tools,
                judge=exact_judge,
            )

    def test_non_boolean_judge_result_is_excluded_infrastructure_failure(self):
        factory = TravelPlannerFactory(
            contract_mode="failfast",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="group_travel_planner",
            ),
            tool_executor=self.tools,
            judge=lambda plan, name, truth: 1,
        )
        driver = factory.create("bad-judge")
        driver.reset(0)
        transition = driver.step(
            'SUBMIT_PLAN {"plan": "SECRET_A"}',
            env_step=1,
        )
        self.assertTrue(transition.done)
        self.assertTrue(transition.sample_excluded)
        self.assertEqual(transition.status, "infra_error")
        self.assertEqual(transition.phase_index, 0)

    def test_native_tool_is_one_zero_reward_transition(self):
        stepped = self.wrapper.step(
            self.env_id,
            'Action: FlightSearch {"origin": "A", "destination": "B", "date": "2022-01-01"}',
        )
        self.assertEqual(stepped["reward"], 0.0)
        self.assertFalse(stepped["done"])
        self.assertEqual(stepped["info"]["action_execution"]["op"], "FlightSearch")
        self.assertEqual(len(self.tools.calls), 1)

    def test_native_tool_payload_is_delegated_without_stricter_adapter_rules(self):
        stepped = self.wrapper.step(
            self.env_id,
            'Action: DistanceMatrix {"origin": "A", "destination": "B", '
            '"mode": "walking", "upstream_ignored": true}',
        )
        self.assertEqual(stepped["reward"], 0.0)
        self.assertFalse(stepped["done"])
        self.assertEqual(
            self.tools.calls,
            [
                (
                    "DistanceMatrix",
                    {
                        "origin": "A",
                        "destination": "B",
                        "mode": "walking",
                        "upstream_ignored": True,
                    },
                )
            ],
        )

    def test_upstream_error_string_remains_an_executed_tool_result(self):
        self.tools.result = "Error executing FlightSearch: missing flight table"
        stepped = self.wrapper.step(
            self.env_id,
            'Action: FlightSearch {"origin": "A", "destination": "B", '
            '"date": "2022-01-01"}',
        )
        self.assertEqual(stepped["reward"], 0.0)
        self.assertFalse(stepped["done"])
        self.assertEqual(
            stepped["info"]["action_execution"]["status"],
            "executed_with_error",
        )
        self.assertIn(self.tools.result, stepped["observation"])
        self.assertEqual(
            stepped["info"]["tool_ops"][0]["upstream_error_result"],
            True,
        )

    def test_executor_exception_is_excluded_infrastructure_failure(self):
        self.tools.error = RuntimeError("database unavailable")
        stepped = self.wrapper.step(
            self.env_id,
            'Action: CitySearch {"state": "Texas"}',
        )
        self.assertTrue(stepped["done"])
        self.assertEqual(stepped["info"]["status"], "infra_error")
        self.assertTrue(stepped["info"]["sample_excluded"])
        self.assertNotIn("database unavailable", stepped["observation"])

    def test_correct_plan_advances_and_requires_retrieve_for_hidden_ltm(self):
        added = self.wrapper.step(
            self.env_id,
            'Action: ADD {"key": "base", "value": "Base City F000"}',
        )
        self.assertAlmostEqual(added["reward"], 0.1)
        submitted = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_A"}',
        )
        self.assertEqual(submitted["reward"], 1.0)
        self.assertEqual(submitted["info"]["phase_index"], 1)
        evidence = submitted["info"]["domain_evidence"]
        self.assertEqual(
            evidence["transition_event"]["completed_round_index"],
            1,
        )
        self.assertEqual(evidence["round_index"], 2)
        self.assertEqual(evidence["active_round_index"], 2)
        self.assertNotIn("Fixed base traveler", submitted["observation"])
        self.assertNotIn("Base City F000", submitted["observation"])

        retrieved = self.wrapper.step(
            self.env_id,
            'Action: RETRIEVE {"query": "Base City", "top_k": 3}',
        )
        self.assertAlmostEqual(retrieved["reward"], 0.1)
        self.assertIn("Base City F000", retrieved["observation"])
        retrieved_evidence = retrieved["info"]["domain_evidence"]
        self.assertEqual(retrieved_evidence["round_index"], 2)
        self.assertEqual(retrieved_evidence["active_round_index"], 2)
        self.assertNotIn("transition_event", retrieved_evidence)
        self.assertNotIn("paper_evaluation", retrieved_evidence)

    def test_failfast_wrong_plan_terminates_without_advance_or_label_leakage(self):
        wrong = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "wrong"}',
        )
        self.assertTrue(wrong["done"])
        self.assertAlmostEqual(wrong["reward"], 0.0)
        self.assertEqual(wrong["info"]["phase_index"], 0)
        self.assertEqual(wrong["info"]["status"], "failed_on_incorrect_plan")
        self.assertFalse(wrong["info"]["domain_evidence"]["phase_advanced"])
        self.assertFalse(wrong["info"]["action_execution"]["phase_advanced"])
        self.assertFalse(wrong["info"]["tool_ops"][0]["phase_advanced"])
        self.assertTrue(wrong["info"]["tool_ops"][0]["terminal"])
        self.assertEqual(
            wrong["info"]["domain_evidence"]["transition_event"]["type"],
            "travel_phase_failed",
        )
        self.assertNotIn(
            "completed_round_index",
            wrong["info"]["domain_evidence"]["transition_event"],
        )
        self.assertNotIn("paper_evaluation", wrong["info"]["domain_evidence"])
        self.assertNotIn("SECRET_A", wrong["observation"])
        self.assertNotIn("SECRET_A", repr(wrong["info"]))

    def test_failfast_preserves_earlier_correct_reward_when_later_plan_fails(self):
        first = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_A"}',
        )
        failed = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "wrong"}',
        )
        self.assertEqual(first["reward"], 1.0)
        self.assertEqual(failed["reward"], 0.0)
        self.assertEqual(first["reward"] + failed["reward"], 1.0)
        self.assertTrue(failed["done"])
        self.assertEqual(failed["info"]["phase_index"], 1)
        self.assertFalse(failed["info"]["episode_success"])

    def test_paper_eval_wrong_plan_advances_and_continues(self):
        factory = TravelPlannerFactory(
            contract_mode="paper_eval",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="group_travel_planner",
            ),
            tool_executor=self.tools,
            judge=exact_judge,
        )
        driver = factory.create("paper-continue")
        driver.reset(0)
        wrong = driver.step('SUBMIT_PLAN {"plan": "wrong"}', env_step=1)
        self.assertFalse(wrong.done)
        self.assertEqual(wrong.reward, 0.0)
        self.assertEqual(wrong.phase_index, 1)
        self.assertTrue(wrong.tool_ops[0]["phase_advanced"])
        self.assertNotIn("SECRET_A", wrong.observation)
        completed = driver.step(
            'SUBMIT_PLAN {"plan": "SECRET_B"}',
            env_step=2,
        )
        self.assertTrue(completed.done)
        self.assertFalse(completed.episode_success)

    def test_full_chain_succeeds(self):
        first = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_A"}',
        )
        second = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_B"}',
        )
        self.assertFalse(first["done"])
        self.assertTrue(second["done"])
        self.assertTrue(second["info"]["episode_success"])
        self.assertEqual(second["info"]["phase_index"], 2)
        self.assertEqual(
            second["info"]["domain_evidence"]["transition_event"][
                "completed_round_index"
            ],
            2,
        )
        self.assertIsNone(
            second["info"]["domain_evidence"]["active_round_index"]
        )

    def test_terminal_paper_ledger_is_covered_without_optional_upstream(self):
        tasks = load_travel_tasks(self.tasks_path)
        paper_evaluator = TravelPaperEvaluator(
            tasks=tasks,
            paper_eval_module=FakePaperEval,
            paper_parser_module=FakePaperParser,
            dataset_scope="injected_test_fixture",
        )
        driver = TravelPlannerDriver(
            contract_mode="paper_eval",
            tasks=tasks,
            tool_executor=self.tools,
            judge=exact_judge,
            env_uid="paper-ledger",
            paper_evaluator=paper_evaluator,
        )
        driver.reset(0)
        driver.step('SUBMIT_PLAN {"plan": "SECRET_A"}', env_step=1)
        terminal = driver.step(
            'SUBMIT_PLAN {"plan": "SECRET_B"}',
            env_step=2,
        )
        ledger = terminal.domain_evidence["paper_evaluation"]
        self.assertEqual(
            set(ledger),
            {
                "metric_contract",
                "dataset_scope",
                "source_id",
                "complete",
                "full_pass_people",
                "total_people",
                "group_success",
                "group_constraint_rate",
                "constraint_people",
                "online_reward_is_separate",
            },
        )
        self.assertEqual(ledger["full_pass_people"], 2)
        self.assertEqual(ledger["total_people"], 2)
        self.assertTrue(ledger["group_success"])
        self.assertEqual(ledger["constraint_people"], 2)
        self.assertEqual(ledger["group_constraint_rate"], 1.0)
        self.assertTrue(ledger["complete"])

    def test_failfast_one_action_30_action_limit_terminates_without_advance(self):
        for step in range(1, TRAVEL_MAX_STEPS_PER_PHASE):
            transition = self.wrapper.step(
                self.env_id,
                f'Action: CitySearch {{"state": "State {step}"}}',
            )
            self.assertEqual(transition["info"]["phase_index"], 0)
            self.assertFalse(transition["done"])

        timed_out = self.wrapper.step(
            self.env_id,
            f'Action: CitySearch {{"state": "State {TRAVEL_MAX_STEPS_PER_PHASE}"}}',
        )
        info = timed_out["info"]
        self.assertEqual(timed_out["reward"], 0.0)
        self.assertTrue(timed_out["done"])
        self.assertEqual(info["phase_index"], 0)
        self.assertEqual(info["status"], "failed_on_phase_timeout")
        self.assertTrue(info["action_execution"]["phase_timeout"])
        self.assertFalse(info["action_execution"]["phase_advanced"])
        self.assertEqual(
            info["domain_evidence"]["transition_event"][
                "failure_reason"
            ],
            "one_action_variant_limit",
        )
        self.assertEqual(
            info["domain_evidence"]["transition_event"][
                "failed_phase_step_count"
            ],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )
        self.assertEqual(info["tool_ops"][-1]["op"], "PHASE_TIMEOUT")
        self.assertFalse(info["tool_ops"][-1]["phase_advanced"])
        self.assertNotIn("I am Dana. second query", timed_out["observation"])
        self.assertNotIn("paper_evaluation", info["domain_evidence"])

    def test_plan_submitted_as_thirtieth_action_is_judged_before_timeout(self):
        for step in range(1, TRAVEL_MAX_STEPS_PER_PHASE):
            self.wrapper.step(
                self.env_id,
                f'Action: CitySearch {{"state": "State {step}"}}',
            )
        submitted = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_A"}',
        )
        self.assertEqual(submitted["reward"], 1.0)
        self.assertFalse(submitted["done"])
        self.assertEqual(submitted["info"]["phase_index"], 1)
        self.assertNotIn("phase_timeout", submitted["info"]["action_execution"])
        self.assertEqual(
            submitted["info"]["domain_evidence"]["transition_event"][
                "completed_phase_step_count"
            ],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )

    def test_paper_eval_timeout_advances_and_terminal_group_emits_ledger(self):
        tasks = load_travel_tasks(self.tasks_path)
        paper_evaluator = TravelPaperEvaluator(
            tasks=tasks,
            paper_eval_module=FakePaperEval,
            paper_parser_module=FakePaperParser,
            dataset_scope="injected_test_fixture",
        )
        driver = TravelPlannerDriver(
            contract_mode="paper_eval",
            tasks=tasks,
            tool_executor=self.tools,
            judge=exact_judge,
            env_uid="paper-timeout",
            paper_evaluator=paper_evaluator,
        )
        driver.reset(0)
        for step in range(1, TRAVEL_MAX_STEPS_PER_PHASE + 1):
            timed_out = driver.step(
                f'CitySearch {{"state": "State {step}"}}',
                env_step=step,
            )
        self.assertFalse(timed_out.done)
        self.assertEqual(timed_out.phase_index, 1)
        self.assertTrue(timed_out.action_execution["phase_advanced"])
        terminal = driver.step(
            'SUBMIT_PLAN {"plan": "SECRET_B"}',
            env_step=TRAVEL_MAX_STEPS_PER_PHASE + 1,
        )
        self.assertTrue(terminal.done)
        self.assertFalse(terminal.episode_success)
        self.assertEqual(
            terminal.domain_evidence["transition_event"][
                "completed_phase_step_count"
            ],
            1,
        )
        ledger = terminal.domain_evidence["paper_evaluation"]
        self.assertTrue(ledger["complete"])
        self.assertEqual(ledger["full_pass_people"], 1)
        self.assertEqual(ledger["total_people"], 2)
        self.assertFalse(ledger["group_success"])

    def test_memory_actions_do_not_consume_one_action_phase_budget(self):
        for index in range(15):
            self.wrapper.step(
                self.env_id,
                f'Action: ADD {{"key": "memo-{index}", "value": "State {index}"}}',
            )
            self.wrapper.step(
                self.env_id,
                f'Action: RETRIEVE {{"query": "State {index}", "top_k": 1}}',
            )

        for step in range(1, TRAVEL_MAX_STEPS_PER_PHASE):
            transition = self.wrapper.step(
                self.env_id,
                f'Action: CitySearch {{"state": "Native {step}"}}',
            )
            self.assertEqual(transition["info"]["phase_index"], 0)
            self.assertFalse(
                transition["info"]["action_execution"].get("phase_timeout", False)
            )

        timed_out = self.wrapper.step(
            self.env_id,
            f'Action: CitySearch '
            f'{{"state": "Native {TRAVEL_MAX_STEPS_PER_PHASE}"}}',
        )
        self.assertTrue(timed_out["done"])
        self.assertEqual(timed_out["info"]["phase_index"], 0)
        self.assertTrue(timed_out["info"]["action_execution"]["phase_timeout"])
        self.assertFalse(timed_out["info"]["action_execution"]["phase_advanced"])
        self.assertEqual(
            timed_out["info"]["domain_evidence"]["transition_event"][
                "failed_phase_step_count"
            ],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )

    def test_reset_clears_memory_phase_state_and_rejects_bad_position(self):
        self.wrapper.step(
            self.env_id,
            'Action: ADD {"key": "base", "value": "Base City F000"}',
        )
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_A"}',
        )
        with self.assertRaisesRegex(IndexError, "position out of range"):
            self.wrapper.reset(self.env_id, data_idx=999)
        reset = self.wrapper.reset(self.env_id, data_idx=0)
        self.assertEqual(reset["info"]["phase_index"], 0)
        self.assertEqual(reset["info"]["domain_evidence"]["dataset_position"], 0)
        self.assertEqual(reset["info"]["domain_evidence"]["source_id"], 7)
        self.assertEqual(reset["info"]["domain_evidence"]["phase_step_count"], 0)
        self.assertIn("Fixed base traveler", reset["observation"])
        self.assertNotIn("Base City F000", reset["observation"])
        self.assertEqual(
            reset["info"]["domain_evidence"]["memory_inventory_count"],
            0,
        )

    def test_round_index_join_matches_upstream_when_answers_are_reordered(self):
        payload = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        payload["answers"] = [
            {"round_idx": 2, "daily_plans": [{"secret": "SECRET_B"}]},
            {"round_idx": 1, "daily_plans": [{"secret": "SECRET_A"}]},
        ]
        reordered_path = Path(self.tempdir.name) / "travel-reordered.jsonl"
        reordered_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        factory = TravelPlannerFactory(
            contract_mode="failfast",
            tasks_path=reordered_path,
            dataset_provenance=attest_injected_test_dataset(
                reordered_path,
                config="group_travel_planner",
            ),
            tool_executor=self.tools,
            judge=exact_judge,
        )
        driver = factory.create("reordered")
        driver.reset(0)
        first = driver.step('SUBMIT_PLAN {"plan": "SECRET_A"}', env_step=1)
        second = driver.step('SUBMIT_PLAN {"plan": "SECRET_B"}', env_step=2)
        self.assertEqual(first.reward, 1.0)
        self.assertEqual(second.reward, 1.0)
        self.assertTrue(second.done)
        self.assertTrue(second.episode_success)

    def test_270_dataset_positions_cover_source_ids_exactly_once(self):
        path = Path(self.tempdir.name) / "travel-270.jsonl"
        base_payload = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        rows = []
        for source_id in range(1, 271):
            payload = dict(base_payload)
            payload["id"] = source_id
            rows.append(json.dumps(payload) + "\n")
        path.write_text("".join(rows), encoding="utf-8")
        factory = TravelPlannerFactory(
            contract_mode="failfast",
            tasks_path=path,
            dataset_provenance=attest_injected_test_dataset(
                path,
                config="group_travel_planner",
            ),
            tool_executor=self.tools,
            judge=exact_judge,
        )
        driver = factory.create("position-coverage")
        observed = []
        for position in range(270):
            reset = driver.reset(position)
            evidence = reset.domain_evidence
            self.assertEqual(evidence["dataset_position"], position)
            observed.append(evidence["source_id"])
        self.assertEqual(observed, list(range(1, 271)))
        self.assertEqual(len(set(observed)), 270)
        for invalid in (-1, 270):
            with self.assertRaisesRegex(IndexError, "position out of range"):
                driver.reset(invalid)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            driver.reset("1")


class TravelDatabaseAttestationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.expected = {}
        for asset_class, (relative_path, _) in TRAVEL_DATABASE_ASSET_SPECS.items():
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"fixture-{asset_class}\n".encode("utf-8")
            path.write_bytes(content)
            self.expected[asset_class] = (
                relative_path,
                hashlib.sha256(content).hexdigest(),
            )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_attests_exactly_six_asset_classes(self):
        evidence = attest_travel_database(
            self.root,
            injected_test_assets=self.expected,
        )
        self.assertEqual(evidence["mode"], "injected_test_database_manifest")
        self.assertEqual(evidence["asset_count"], 6)
        self.assertEqual(set(evidence["assets"]), set(TRAVEL_DATABASE_ASSET_SPECS))
        self.assertRegex(evidence["manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_missing_tampered_and_incomplete_manifests(self):
        flights_path = self.root / self.expected["flights"][0]
        flights_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing Travel database asset"):
            attest_travel_database(self.root, injected_test_assets=self.expected)
        flights_path.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            attest_travel_database(self.root, injected_test_assets=self.expected)
        with self.assertRaisesRegex(RuntimeError, "exactly six"):
            attest_travel_database(
                self.root,
                injected_test_assets={
                    key: value
                    for key, value in self.expected.items()
                    if key != "cities"
                },
            )


class TravelUpstreamAttestationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for relative_path in TRAVEL_UPSTREAM_RELATIVE_PATHS:
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# pristine {relative_path}\n", encoding="utf-8")
        self._git("init")
        self._git("config", "user.email", "agentmemory-test@example.invalid")
        self._git("config", "user.name", "AgentMemory Test")
        self._git("add", ".")
        self._git("commit", "-m", "fixture")
        self.commit = self._git("rev-parse", "HEAD").strip()

    def tearDown(self):
        self.tempdir.cleanup()

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_attests_exact_commit_and_source_bundle(self):
        evidence = attest_travel_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)
        self.assertEqual(
            set(evidence["source_files_sha256"]),
            set(TRAVEL_UPSTREAM_RELATIVE_PATHS),
        )
        self.assertEqual(len(evidence["source_bundle_sha256"]), 64)

    def test_rejects_modified_travel_source(self):
        path = self.root / TRAVEL_UPSTREAM_RELATIVE_PATHS[0]
        path.write_text("# changed semantics\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_travel_upstream(self.root, expected_commit=self.commit)

    def test_allows_unrelated_worktree_changes(self):
        unrelated = self.root / "env/env_systems/web_shopping_env/local_patch.py"
        unrelated.parent.mkdir(parents=True, exist_ok=True)
        unrelated.write_text("# unrelated\n", encoding="utf-8")
        evidence = attest_travel_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)

    def test_rejects_wrong_commit(self):
        with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
            attest_travel_upstream(self.root, expected_commit="0" * 40)


if __name__ == "__main__":
    unittest.main()
