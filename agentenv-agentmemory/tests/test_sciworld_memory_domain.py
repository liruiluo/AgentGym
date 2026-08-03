from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

from agentenv_agentmemory.domains import V3_SURFACES
from agentenv_agentmemory.domains.sciworld import (
    SCIWORLD_CONDUCTIVITY_SURFACE,
    SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
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


class SciWorldMemoryContractTest(unittest.TestCase):
    def test_surfaces_are_registered_without_importing_scienceworld(self):
        self.assertIn(SCIWORLD_CONDUCTIVITY_SURFACE, V3_SURFACES)
        self.assertIn(SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE, V3_SURFACES)
        self.assertIn(SCIWORLD_CONDUCTIVITY_SURFACE, build_domain_registry().surfaces())
        self.assertIn(
            SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
            build_domain_registry().surfaces(),
        )

    def test_server_factory_can_build_fixture_surface_for_static_tests(self):
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_SCIWORLD_BACKEND": "fixture"},
            clear=False,
        ):
            factory = build_domain_registry().build(SCIWORLD_CONDUCTIVITY_SURFACE)
        self.assertIsInstance(factory, SciWorldMemoryFactory)
        self.assertEqual(factory.backend, "fixture")
        self.assertEqual(factory.surface, SCIWORLD_CONDUCTIVITY_SURFACE)

    def test_scienceworld_backend_fails_closed_when_dependency_missing(self):
        if importlib.util.find_spec("scienceworld") is not None:
            self.skipTest("scienceworld is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "requires the optional 'scienceworld'"):
            SciWorldMemoryFactory(
                surface=SCIWORLD_CONDUCTIVITY_SURFACE,
                backend="scienceworld",
            )

    def test_contract_says_model_manages_external_memory_not_recent_n(self):
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
                for forbidden in _FORBIDDEN_HISTORY_HELP:
                    self.assertNotIn(forbidden.lower(), prompt.lower())
                self.assertNotIn("keep only", prompt.lower())

    def test_fixture_requires_policy_authored_memory_across_phase_boundary(self):
        factory = SciWorldMemoryFactory(
            surface=SCIWORLD_CONDUCTIVITY_SURFACE,
            backend="fixture",
        )
        wrapper = DomainEnvWrapper(
            factory,
            reward_policy=MemoryRewardPolicy(
                first_add=0.1,
                first_later_phase_retrieve=0.1,
            ),
        )
        env_id = wrapper.create()["id"]
        try:
            tested = wrapper.step(
                env_id,
                'Action: SCI_ACTION {"action": "test conductivity of the unknown sample"}',
            )
            self.assertIn("unknown sample alpha is conductive", tested["observation"])

            stored = wrapper.step(
                env_id,
                'Action: ADD {"key": "alpha test", "value": "unknown sample alpha is conductive"}',
            )
            self.assertIn("mem_0000", stored["observation"])

            advanced = wrapper.step(
                env_id,
                'Action: ANSWER {"answer": "conductive"}',
            )
            self.assertEqual(advanced["info"]["phase_index"], 1)
            self.assertNotIn("unknown sample alpha is conductive", advanced["observation"])
            self.assertIn("prior lab result is not repeated", advanced["observation"])

            retrieved = wrapper.step(
                env_id,
                'Action: RETRIEVE {"query": "alpha conductive", "top_k": 3}',
            )
            self.assertIn("[mem_0000] alpha test", retrieved["observation"])
            self.assertAlmostEqual(retrieved["reward"], 0.1)

            solved = wrapper.step(
                env_id,
                'Action: ANSWER {"answer": "unknown sample alpha"}',
            )
            self.assertTrue(solved["done"])
            self.assertTrue(solved["info"]["episode_success"])
            self.assertEqual(solved["info"]["status"], "success")
        finally:
            if env_id in wrapper.envs:
                wrapper.close(env_id)


if __name__ == "__main__":
    unittest.main()
