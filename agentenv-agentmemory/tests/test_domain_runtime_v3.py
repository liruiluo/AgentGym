from __future__ import annotations

import hashlib
import unittest

from agentenv_agentmemory.domains import V3_SURFACES
from agentenv_agentmemory.domains.fake import FakeTwoPhaseFactory
from agentenv_agentmemory.runtime.domain import (
    MEMORY_ACTION_DESCRIPTIONS,
    DomainContract,
    DomainTransition,
)
from agentenv_agentmemory.runtime.memory import MemoryRewardPolicy
from agentenv_agentmemory.runtime.registry import DomainRegistry
from agentenv_agentmemory.runtime.server_factory import build_domain_registry
from agentenv_agentmemory.runtime.wrapper import (
    DomainEnvWrapper,
    extract_submitted_action,
)


class DomainTransitionTest(unittest.TestCase):
    def test_reward_ledger_must_sum_to_reward(self):
        with self.assertRaisesRegex(ValueError, "sum to reward"):
            DomainTransition(
                observation="x",
                reward=1.0,
                done=False,
                status="active",
                phase_index=0,
                phase_count=2,
                episode_success=False,
                reward_components=({"name": "wrong", "value": 0.0},),
            )

    def test_contract_digest_is_stable(self):
        contract = DomainContract("c", "prompt", ("ACTION",), 3)
        self.assertEqual(contract.sha256, contract.sha256)
        self.assertEqual(len(contract.sha256), 64)


class DomainRegistryTest(unittest.TestCase):
    def test_production_registry_covers_exactly_the_four_v3_surfaces(self):
        self.assertEqual(
            build_domain_registry().surfaces(),
            tuple(sorted(V3_SURFACES)),
        )

    def test_registry_is_lazy_and_fail_closed(self):
        calls = []
        registry = DomainRegistry()
        registry.register(
            FakeTwoPhaseFactory.surface,
            lambda: calls.append("built") or FakeTwoPhaseFactory(),
        )
        self.assertEqual(calls, [])
        self.assertEqual(registry.build(FakeTwoPhaseFactory.surface).task_count, 2)
        self.assertEqual(calls, ["built"])
        with self.assertRaisesRegex(RuntimeError, "unknown AgentMemoryGym"):
            registry.build("missing")


class DomainWrapperTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = DomainEnvWrapper(
            FakeTwoPhaseFactory(),
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

    def test_metadata_has_v3_contract_and_no_ground_truth(self):
        metadata = self.wrapper.metadata()
        self.assertEqual(metadata["formal_schema_version"], "agentmemory_formal_step_v3")
        self.assertEqual(metadata["domain_id"], "fake_two_phase")
        self.assertEqual(len(metadata["contract_sha256"]), 64)
        self.assertEqual(
            metadata["system_prompt_sha256"],
            hashlib.sha256(metadata["system_prompt"].encode("utf-8")).hexdigest(),
        )
        prompt = metadata["system_prompt"]
        self.assertEqual(prompt.count('ADVANCE {"value": "..."}'), 1)
        for action in MEMORY_ACTION_DESCRIPTIONS:
            self.assertEqual(prompt.count(action), 1)
        self.assertIn("Cross-phase memory lifecycle:", prompt)
        self.assertIn(
            "A native phase advance clears the current phase's short-term/page trace",
            prompt,
        )
        self.assertIn(
            "Long-term memory is retained, but it is not automatically visible",
            prompt,
        )
        self.assertIn("RETRIEVE queries text previously written with ADD", prompt)
        self.assertEqual(prompt.count("Action:"), 1)
        self.assertNotIn("ground_truth", repr(metadata).lower())
        self.assertEqual(
            metadata["reward_overlay"],
            "agentmemory_policy_memory_shaping_v1",
        )

    def test_latest_observation_does_not_repeat_static_action_contract(self):
        observation = self.created["observation"]
        self.assertNotIn('ADVANCE {"value": "..."}', observation)
        for action in MEMORY_ACTION_DESCRIPTIONS:
            self.assertNotIn(action, observation)

        stepped = self.wrapper.step(
            self.env_id,
            'Action: RETRIEVE {"query": "none", "top_k": 3}',
        )
        self.assertNotIn('ADVANCE {"value": "..."}', stepped["observation"])
        for action in MEMORY_ACTION_DESCRIPTIONS:
            self.assertNotIn(action, stepped["observation"])

    def test_default_runtime_is_reward_neutral(self):
        wrapper = DomainEnvWrapper(FakeTwoPhaseFactory())
        self.assertEqual(wrapper.metadata()["reward_overlay"], "none")
        self.assertEqual(
            wrapper.metadata()["memory_reward_policy"],
            {
                "first_add": 0.0,
                "first_later_phase_retrieve": 0.0,
                "exact_repeat": 0.0,
                "invalid_action": 0.0,
            },
        )

    def test_default_invalid_overlay_preserves_domain_reward(self):
        wrapper = DomainEnvWrapper(FakeTwoPhaseFactory())
        env_id = wrapper.create()["id"]
        invalid = wrapper.step(env_id, "not a fake-domain action")
        self.assertEqual(invalid["reward"], -0.01)
        self.assertEqual(
            invalid["info"]["reward_components"],
            [
                {
                    "name": "invalid_action",
                    "value": -0.01,
                    "op": "INVALID",
                    "step": 1,
                }
            ],
        )
        wrapper.close(env_id)

    def test_explicit_invalid_overlay_replaces_domain_penalty_once(self):
        wrapper = DomainEnvWrapper(
            FakeTwoPhaseFactory(),
            invalid_action_penalty=-0.02,
        )
        env_id = wrapper.create()["id"]
        invalid = wrapper.step(env_id, "not a fake-domain action")
        components = invalid["info"]["reward_components"]
        self.assertEqual(invalid["reward"], -0.02)
        self.assertEqual(
            [item["name"] for item in components].count("invalid_action"),
            1,
        )
        self.assertEqual(sum(item["value"] for item in components), -0.02)
        self.assertEqual(components[0]["value"], -0.02)
        wrapper.close(env_id)

    def test_ltm_survives_phase_advance_but_visible_context_clears(self):
        added = self.wrapper.step(
            self.env_id,
            'Thought: store it\n\nAction: ADD {"key": "phase0", "value": "alpha"}',
        )
        self.assertAlmostEqual(added["reward"], 0.1)
        self.assertIn("mem_0000", added["observation"])

        advanced = self.wrapper.step(
            self.env_id,
            'Thought: continue\n\nAction: ADVANCE {"value": "first"}',
        )
        self.assertEqual(advanced["info"]["phase_index"], 1)
        self.assertNotIn("[mem_0000] phase0: alpha", advanced["observation"])

        retrieved = self.wrapper.step(
            self.env_id,
            'Action: RETRIEVE {"query": "alpha", "top_k": 3}',
        )
        self.assertAlmostEqual(retrieved["reward"], 0.1)
        self.assertIn("[mem_0000] phase0: alpha", retrieved["observation"])

    def test_repeat_penalty_compares_submitted_action_not_thought(self):
        first = self.wrapper.step(
            self.env_id,
            'Thought: one\n\nAction: RETRIEVE {"query": "none", "top_k": 3}',
        )
        second = self.wrapper.step(
            self.env_id,
            'Thought: a different thought\n\nAction: RETRIEVE {"query": "none", "top_k": 3}',
        )
        self.assertEqual(first["reward"], 0.0)
        self.assertAlmostEqual(second["reward"], -0.01)
        self.assertEqual(
            second["info"]["reward_components"][-1]["name"],
            "exact_repeated_valid_zero_reward_action",
        )

    def test_raw_and_submitted_actions_are_both_recorded(self):
        raw = 'Thought: advance\n\nAction: ADVANCE {"value": "x"}'
        stepped = self.wrapper.step(self.env_id, raw)
        execution = stepped["info"]["action_execution"]
        self.assertEqual(execution["raw_policy_output"], raw)
        self.assertEqual(execution["submitted_action"], 'ADVANCE {"value": "x"}')
        self.assertTrue(stepped["info"]["domain_evidence"]["phase_advanced"])

    def test_action_envelope_does_not_truncate_payload_text(self):
        raw = (
            "Thought: build the itinerary\n\n"
            'Action:\nSUBMIT_PLAN {"plan":"Day 1: Action: use flight"}'
        )

        self.assertEqual(
            extract_submitted_action(raw),
            'SUBMIT_PLAN {"plan":"Day 1: Action: use flight"}',
        )

        plain_answer = "Action:\nfirst line\nAction: quoted source label"
        self.assertEqual(
            extract_submitted_action(plain_answer),
            "first line\nAction: quoted source label",
        )

    def test_memory_transitions_preserve_domain_evidence(self):
        added = self.wrapper.step(
            self.env_id,
            'Action: ADD {"key": "phase0", "value": "alpha"}',
        )
        evidence = added["info"]["domain_evidence"]
        self.assertEqual(evidence["data_idx"], 0)
        self.assertEqual(evidence["phase_index_before"], 0)
        self.assertFalse(evidence["phase_advanced"])
        self.assertEqual(
            evidence["memory_state_diff"]["added"][0]["value"],
            "alpha",
        )

        invalid = self.wrapper.step(self.env_id, "Action: ADD {bad json}")
        invalid_evidence = invalid["info"]["domain_evidence"]
        self.assertEqual(invalid_evidence["data_idx"], 0)
        self.assertEqual(invalid_evidence["phase_index_before"], 0)
        self.assertFalse(invalid_evidence["phase_advanced"])
        self.assertIn("valid JSON", invalid_evidence["memory_action_error"])
        self.assertEqual(
            invalid_evidence["memory_state_diff"],
            {"added": [], "updated": [], "deleted": []},
        )

    def test_episode_reset_clears_long_term_memory(self):
        self.wrapper.step(
            self.env_id,
            'Action: ADD {"key": "phase0", "value": "alpha"}',
        )
        reset = self.wrapper.reset(self.env_id, 1)
        retrieved = self.wrapper.step(
            self.env_id,
            'Action: RETRIEVE {"query": "alpha", "top_k": 3}',
        )
        self.assertIn("Long-term memory is empty", retrieved["observation"])
        self.assertEqual(reset["info"]["phase_index"], 0)


if __name__ == "__main__":
    unittest.main()
