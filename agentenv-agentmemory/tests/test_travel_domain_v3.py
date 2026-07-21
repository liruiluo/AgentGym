from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.domains.travel import (
    TRAVEL_UPSTREAM_RELATIVE_PATHS,
    TravelPlannerFactory,
    attest_travel_upstream,
)
from agentenv_agentmemory.runtime.memory import MemoryRewardPolicy
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


class FakeTravelTools:
    def __init__(self):
        self.calls = []

    def execute(self, op, payload):
        self.calls.append((op, dict(payload)))
        return "flight F123 at 09:00"


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
