from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.domains.travel import (
    TRAVEL_MAX_STEPS_PER_PHASE,
    TRAVEL_UPSTREAM_RELATIVE_PATHS,
    TravelPlannerFactory,
    attest_travel_upstream,
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
        factory = TravelPlannerFactory(
            tasks_path=self.tasks_path,
            tool_executor=self.tools,
            judge=exact_judge,
        )
        self.wrapper = DomainEnvWrapper(
            factory,
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
            "continue_to_next_traveler",
        )
        self.assertEqual(
            self.wrapper.metadata()["max_steps_per_phase"],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )
        self.assertEqual(self.wrapper.metadata()["max_episode_steps"], 60)

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
        self.assertNotIn("Fixed base traveler", submitted["observation"])
        self.assertNotIn("Base City F000", submitted["observation"])

        retrieved = self.wrapper.step(
            self.env_id,
            'Action: RETRIEVE {"query": "Base City", "top_k": 3}',
        )
        self.assertAlmostEqual(retrieved["reward"], 0.1)
        self.assertIn("Base City F000", retrieved["observation"])

    def test_wrong_plan_preserves_original_continue_semantics_without_label_feedback(self):
        wrong = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "wrong"}',
        )
        self.assertFalse(wrong["done"])
        self.assertAlmostEqual(wrong["reward"], 0.0)
        self.assertEqual(wrong["info"]["phase_index"], 1)
        self.assertNotIn("SECRET_A", wrong["observation"])
        self.assertNotIn("SECRET_A", repr(wrong["info"]))

        completed = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_B"}',
        )
        self.assertTrue(completed["done"])
        self.assertFalse(completed["info"]["episode_success"])

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

    def test_native_30_step_limit_is_phase_local_and_visible_in_evidence(self):
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
        self.assertFalse(timed_out["done"])
        self.assertEqual(info["phase_index"], 1)
        self.assertTrue(info["action_execution"]["phase_timeout"])
        self.assertEqual(info["domain_evidence"]["phase_completion_reason"], "native_step_limit")
        self.assertEqual(
            info["domain_evidence"]["completed_phase_step_count"],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )
        self.assertEqual(info["tool_ops"][-1]["op"], "PHASE_TIMEOUT")
        self.assertIn("I am Dana. second query", timed_out["observation"])

        completed = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_B"}',
        )
        self.assertTrue(completed["done"])
        self.assertFalse(completed["info"]["episode_success"])
        self.assertEqual(
            completed["info"]["domain_evidence"]["completed_phase_step_count"],
            1,
        )

    def test_memory_actions_do_not_consume_native_phase_budget(self):
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
        self.assertEqual(timed_out["info"]["phase_index"], 1)
        self.assertTrue(timed_out["info"]["action_execution"]["phase_timeout"])
        self.assertEqual(
            timed_out["info"]["domain_evidence"]["completed_phase_step_count"],
            TRAVEL_MAX_STEPS_PER_PHASE,
        )

    def test_reset_clears_memory_phase_state_and_uses_original_seed_fallback(self):
        self.wrapper.step(
            self.env_id,
            'Action: ADD {"key": "base", "value": "Base City F000"}',
        )
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_PLAN {"plan": "SECRET_A"}',
        )
        reset = self.wrapper.reset(self.env_id, data_idx=999)
        self.assertEqual(reset["info"]["phase_index"], 0)
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
            tasks_path=reordered_path,
            tool_executor=self.tools,
            judge=exact_judge,
        )
        driver = factory.create("reordered")
        driver.reset(7)
        first = driver.step('SUBMIT_PLAN {"plan": "SECRET_A"}', env_step=1)
        second = driver.step('SUBMIT_PLAN {"plan": "SECRET_B"}', env_step=2)
        self.assertEqual(first.reward, 1.0)
        self.assertEqual(second.reward, 1.0)
        self.assertTrue(second.done)
        self.assertTrue(second.episode_success)


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
