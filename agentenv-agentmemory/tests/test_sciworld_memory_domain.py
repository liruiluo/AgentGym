from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

from agentenv_agentmemory.domains import V3_SURFACES
from agentenv_agentmemory.domains.sciworld import (
    SCIWORLD_CALIBRATION_SURFACE,
    SCIWORLD_CONTEXTUAL_RULE_SURFACE,
    SCIWORLD_CONDUCTIVITY_SURFACE,
    SCIWORLD_FRICTION_SURFACE,
    SCIWORLD_GOAL_PROGRESS_SURFACE,
    SCIWORLD_HYPOTHESIS_TRACKING_SURFACE,
    SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
    SCIWORLD_MELTINGPOINT_SURFACE,
    SCIWORLD_NEGATIVE_EVIDENCE_SURFACE,
    SCIWORLD_RULE_MEMORY_SURFACE,
    SCIWORLD_SOP_MEMORY_SURFACE,
    SCIWORLD_STATE_CHANGE_SURFACE,
    SCIWORLD_SURFACES,
    SciWorldMemoryFactory,
)
from agentenv_agentmemory.runtime.memory import MemoryRewardPolicy
from agentenv_agentmemory.runtime.server_factory import build_domain_registry
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


_FORBIDDEN_HISTORY_HELP = (
    "latest N",
    "latest 5",
    "latest 10",
    "last N",
    "last 5",
    "last 10",
    "recent N",
    "recent 5",
    "recent 10",
    "rolling summary",
    "environment-written summary",
    "harness summary",
    "ground-truth lab notes",
)

_SOLUTIONS = {
    SCIWORLD_CONDUCTIVITY_SURFACE: [
        ("test conductivity of alpha", "unknown sample alpha is conductive", "conductive"),
        (
            None,
            (("unknown sample alpha", "unknown sample alpha is conductive"),),
            "unknown sample alpha",
        ),
    ],
    SCIWORLD_MELTINGPOINT_SURFACE: [
        ("melt unknown crystal mira", "unknown crystal mira melts at 70 celsius", "70"),
        (None, (("mira", "unknown crystal mira melts at 70 celsius"),), "mira"),
    ],
    SCIWORLD_FRICTION_SURFACE: [
        ("measure friction of slate and glass", "slate A has higher friction than glass B", "slate higher"),
        (None, (("slate", "slate A has higher friction than glass B"),), "slate"),
    ],
    SCIWORLD_RULE_MEMORY_SURFACE: [
        ("mix red paint with yellow paint", "red plus yellow makes orange", "orange"),
        ("mix orange paint with yellow paint", "orange plus yellow makes amber", "amber"),
        (
            None,
            (
                ("orange", "red plus yellow makes orange"),
                ("amber", "orange plus yellow makes amber"),
            ),
            "mix red and yellow to make orange, then mix orange and yellow",
        ),
    ],
    SCIWORLD_SOP_MEMORY_SURFACE: [
        ("assemble a circuit with battery wire bulb and sample", "conductivity SOP uses battery wire bulb sample", "battery bulb sample"),
        (
            None,
            (("conductivity SOP", "conductivity SOP uses battery wire bulb sample"),),
            "battery bulb sample",
        ),
    ],
    SCIWORLD_NEGATIVE_EVIDENCE_SURFACE: [
        (
            "test zeta with vinegar",
            "powder zeta does not fizz with vinegar",
            "does not fizz",
        ),
        (
            None,
            (("zeta vinegar", "powder zeta does not fizz with vinegar"),),
            "powder eta",
        ),
    ],
    SCIWORLD_HYPOTHESIS_TRACKING_SURFACE: [
        (
            "move lamp east",
            "light direction hypothesis supported by lamp east",
            "light",
        ),
        (
            "change soil color",
            "soil-color hypothesis is ruled out",
            "rules out soil",
        ),
        (
            None,
            (
                ("light hypothesis", "light direction hypothesis supported"),
                ("soil hypothesis", "soil-color hypothesis is ruled out"),
            ),
            "light direction",
        ),
    ],
    SCIWORLD_CALIBRATION_SURFACE: [
        (
            "compare thermometer with reference bath",
            "thermometer T reads 5 celsius high",
            "5 high",
        ),
        (
            None,
            (("thermometer offset", "thermometer T reads 5 celsius high"),),
            "70",
        ),
    ],
    SCIWORLD_CONTEXTUAL_RULE_SURFACE: [
        (
            "test sugar in cold water and hot water",
            "sugar dissolves quickly in hot water and slowly in cold water",
            "hot quickly",
        ),
        (
            None,
            (("sugar hot water", "sugar dissolves quickly in hot water"),),
            "hot water",
        ),
    ],
    SCIWORLD_STATE_CHANGE_SURFACE: [
        (
            "test riva with indicator strip",
            "preliminary strip says riva acidic",
            "acidic",
        ),
        (
            "test riva with calibrated meter",
            "calibrated meter supersedes strip: riva neutral",
            "neutral",
        ),
        (
            None,
            (("riva neutral", "calibrated meter supersedes strip: riva neutral"),),
            "neutral",
        ),
    ],
    SCIWORLD_GOAL_PROGRESS_SURFACE: [
        (
            "collect sample",
            "sample collected; next heat sample then record final color",
            "sample collected",
        ),
        (
            "heat sample",
            "sample heated; next record final color",
            "sample heated",
        ),
        (
            None,
            (("record final color", "sample heated; next record final color"),),
            "record final color",
        ),
    ],
    SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE: [
        ("inspect station 1", "station 1 color blue", "blue"),
        ("inspect station 2", "station 2 color green", "green"),
        ("inspect station 3", "station 3 color orange", "orange"),
        ("inspect station 4", "station 4 color silver", "silver"),
        ("inspect station 5", "station 5 color violet", "violet"),
        ("inspect station 6", "station 6 color black", "black"),
        (None, (("station 5", "station 5 color violet"),), "station 5"),
    ],
}


class SciWorldMemoryContractTest(unittest.TestCase):
    def test_all_planned_sciworld_memory_surfaces_are_registered_without_importing_scienceworld(self):
        expected = {
            SCIWORLD_CONDUCTIVITY_SURFACE,
            SCIWORLD_MELTINGPOINT_SURFACE,
            SCIWORLD_FRICTION_SURFACE,
            SCIWORLD_RULE_MEMORY_SURFACE,
            SCIWORLD_SOP_MEMORY_SURFACE,
            SCIWORLD_NEGATIVE_EVIDENCE_SURFACE,
            SCIWORLD_HYPOTHESIS_TRACKING_SURFACE,
            SCIWORLD_CALIBRATION_SURFACE,
            SCIWORLD_CONTEXTUAL_RULE_SURFACE,
            SCIWORLD_STATE_CHANGE_SURFACE,
            SCIWORLD_GOAL_PROGRESS_SURFACE,
            SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
        }
        self.assertEqual(set(SCIWORLD_SURFACES.values()), expected)
        self.assertTrue(expected.issubset(set(V3_SURFACES)))
        self.assertTrue(expected.issubset(set(build_domain_registry().surfaces())))

    def test_server_factory_can_build_fixture_surfaces_for_static_tests(self):
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_SCIWORLD_BACKEND": "fixture"},
            clear=False,
        ):
            for surface in SCIWORLD_SURFACES.values():
                with self.subTest(surface=surface):
                    factory = build_domain_registry().build(surface)
                    self.assertIsInstance(factory, SciWorldMemoryFactory)
                    self.assertEqual(factory.backend, "fixture")
                    self.assertEqual(factory.surface, surface)
                    self.assertGreaterEqual(factory.task_count, 1)

    def test_scienceworld_backend_fails_closed_when_dependency_missing(self):
        if importlib.util.find_spec("scienceworld") is not None:
            self.skipTest("scienceworld is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "requires the optional 'scienceworld'"):
            SciWorldMemoryFactory(
                surface=SCIWORLD_CONDUCTIVITY_SURFACE,
                backend="scienceworld",
            )

    def test_contract_says_model_manages_external_memory_not_manual_window(self):
        for surface in SCIWORLD_SURFACES.values():
            with self.subTest(surface=surface):
                factory = SciWorldMemoryFactory(surface=surface, backend="fixture")
                metadata = factory.metadata()
                prompt = factory.contract.canonical_system_prompt
                self.assertEqual(
                    metadata["memory_management"],
                    "policy_managed_external_notebook",
                )
                self.assertEqual(
                    metadata["history_policy"],
                    "no_harness_recent_n_no_environment_summary",
                )
                self.assertEqual(
                    metadata["episode_structure"],
                    "fixture_stages_only_not_capability_evidence",
                )
                self.assertFalse(metadata["artificial_session_boundaries"])
                self.assertEqual(metadata["context_compaction_owner"], "policy")
                self.assertFalse(metadata["harness_summarizes_history"])
                self.assertIsNone(metadata["manual_recent_n_window"])
                self.assertIn("one continuous episode", prompt)
                self.assertIn("decide when to use SUMMARY/FILTER", prompt)
                self.assertIn("create artificial session boundaries", prompt)
                self.assertIn("does not write lab notes", prompt)
                self.assertIn("memory_kind", metadata)
                for forbidden in _FORBIDDEN_HISTORY_HELP:
                    self.assertNotIn(forbidden.lower(), prompt.lower())
                self.assertNotIn("keep only", prompt.lower())

    def test_each_fixture_surface_has_minimal_memory_chain(self):
        for surface, solution in _SOLUTIONS.items():
            with self.subTest(surface=surface):
                wrapper = DomainEnvWrapper(
                    SciWorldMemoryFactory(surface=surface, backend="fixture"),
                    reward_policy=MemoryRewardPolicy(
                        first_add=0.1,
                        first_later_phase_retrieve=0.1,
                    ),
                )
                env_id = wrapper.create()["id"]
                try:
                    for index, (command, memory_value, answer) in enumerate(solution):
                        if command is not None:
                            observed = wrapper.step(
                                env_id,
                                f'Action: SCI_ACTION {{"action": "{command}"}}',
                            )
                            self.assertFalse(observed["done"])
                            self.assertTrue(
                                observed["info"]["action_execution"]["experiment_matched"]
                            )
                            self.assertEqual(
                                observed["info"]["reward_components"][0]["name"],
                                "sciworld_experiment_observed",
                            )
                            stored = wrapper.step(
                                env_id,
                                'Action: ADD '
                                + '{"key": "phase_%d", "value": "%s"}'
                                % (index, memory_value),
                            )
                            self.assertIn("mem_", stored["observation"])
                        else:
                            for query, expected_memory in memory_value:
                                retrieved = wrapper.step(
                                    env_id,
                                    'Action: RETRIEVE {"query": "%s", "top_k": 1}'
                                    % query,
                                )
                                self.assertIn(
                                    "Retrieved memories", retrieved["observation"]
                                )
                                self.assertIn(
                                    expected_memory, retrieved["observation"]
                                )
                        result = wrapper.step(
                            env_id,
                            f'Action: ANSWER {{"answer": "{answer}"}}',
                        )
                    self.assertTrue(result["done"])
                    self.assertTrue(result["info"]["episode_success"])
                    self.assertEqual(result["info"]["status"], "success")
                finally:
                    if env_id in wrapper.envs:
                        wrapper.close(env_id)

    def test_phase_advance_does_not_repeat_prior_result_without_retrieve(self):
        wrapper = DomainEnvWrapper(
            SciWorldMemoryFactory(
                surface=SCIWORLD_CONDUCTIVITY_SURFACE,
                backend="fixture",
            ),
            reward_policy=MemoryRewardPolicy(first_add=0.1),
        )
        env_id = wrapper.create()["id"]
        try:
            tested = wrapper.step(
                env_id,
                'Action: SCI_ACTION {"action": "test conductivity of alpha"}',
            )
            self.assertIn("unknown sample alpha is conductive", tested["observation"])
            wrapper.step(
                env_id,
                'Action: ADD {"key": "alpha test", "value": "unknown sample alpha is conductive"}',
            )
            advanced = wrapper.step(env_id, 'Action: ANSWER {"answer": "conductive"}')
            self.assertEqual(advanced["info"]["phase_index"], 1)
            self.assertNotIn("unknown sample alpha is conductive", advanced["observation"])
            self.assertIn("prior lab result is not repeated", advanced["observation"])
        finally:
            if env_id in wrapper.envs:
                wrapper.close(env_id)

    def test_state_change_surface_can_revise_memory_instead_of_using_stale_fact(self):
        wrapper = DomainEnvWrapper(
            SciWorldMemoryFactory(
                surface=SCIWORLD_STATE_CHANGE_SURFACE,
                backend="fixture",
            ),
            reward_policy=MemoryRewardPolicy(first_add=0.1),
        )
        env_id = wrapper.create()["id"]
        try:
            wrapper.step(
                env_id,
                'Action: SCI_ACTION {"action": "test riva with indicator strip"}',
            )
            added = wrapper.step(
                env_id,
                'Action: ADD {"key": "riva state", "value": "preliminary strip says riva acidic"}',
            )
            self.assertIn("mem_0000", added["observation"])
            wrapper.step(env_id, 'Action: ANSWER {"answer": "acidic"}')
            wrapper.step(
                env_id,
                'Action: SCI_ACTION {"action": "test riva with calibrated meter"}',
            )
            updated = wrapper.step(
                env_id,
                'Action: UPDATE {"memory_id": "mem_0000", "value": "calibrated meter supersedes strip: riva neutral"}',
            )
            self.assertIn("Updated memory [mem_0000]", updated["observation"])
            wrapper.step(env_id, 'Action: ANSWER {"answer": "neutral"}')
            retrieved = wrapper.step(
                env_id,
                'Action: RETRIEVE {"query": "riva neutral", "top_k": 1}',
            )
            self.assertIn(
                "calibrated meter supersedes strip: riva neutral",
                retrieved["observation"],
            )
            self.assertNotIn(
                "preliminary strip says riva acidic",
                retrieved["observation"],
            )
            result = wrapper.step(env_id, 'Action: ANSWER {"answer": "neutral"}')
            self.assertTrue(result["done"])
            self.assertTrue(result["info"]["episode_success"])
        finally:
            if env_id in wrapper.envs:
                wrapper.close(env_id)


if __name__ == "__main__":
    unittest.main()
