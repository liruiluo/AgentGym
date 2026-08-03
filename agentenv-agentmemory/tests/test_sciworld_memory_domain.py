from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

from agentenv_agentmemory.domains import V3_SURFACES
from agentenv_agentmemory.domains.sciworld import (
    SCIWORLD_CONDUCTIVITY_SURFACE,
    SCIWORLD_FRICTION_SURFACE,
    SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
    SCIWORLD_MELTINGPOINT_SURFACE,
    SCIWORLD_RULE_MEMORY_SURFACE,
    SCIWORLD_SOP_MEMORY_SURFACE,
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
                self.assertFalse(metadata["harness_summarizes_history"])
                self.assertIsNone(metadata["manual_recent_n_window"])
                self.assertIn("use your own memory actions", prompt)
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


if __name__ == "__main__":
    unittest.main()
