from __future__ import annotations

import json
import unittest
from dataclasses import replace

from agentenv_agentmemory.distractor_robustness import (
    PROVIDER_MODE_RESEEDED_STREAM,
    DistractorRobustnessDataError,
    DistractorRobustnessGenerator,
    VerifiedDistractorRobustnessBundleProvider,
    verify_distractor_robustness_orbit,
)
from agentenv_agentmemory.distractor_robustness_webshop_env import (
    DistractorRobustnessWebShopEnv,
)
from tests.test_recency_override_data import (
    _FakeRecencyNativeBackend,
    make_fixture_pool,
)


class DistractorRobustnessGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = DistractorRobustnessGenerator(pool=self.pool, seed=233)

    def test_one_hundred_tasks_are_deterministic_and_strict_top1(self) -> None:
        semantic_hashes: set[str] = set()
        for orbit_index in range(50):
            orbit = self.generator.generate_orbit(orbit_index, split="train")
            repeated = DistractorRobustnessGenerator(
                pool=self.pool,
                seed=233,
            ).generate_orbit(orbit_index, split="train")
            self.assertEqual(orbit.as_dict(), repeated.as_dict())
            proof = verify_distractor_robustness_orbit(
                orbit,
                pool=self.pool,
                expected_generator_version=self.generator.version,
                expected_generator_seed=self.generator.seed,
            )
            self.assertEqual(proof.source_recency_valid_solution_counts, (1, 1))
            self.assertGreater(proof.canonical_top1_score, proof.runner_up_score)
            self.assertGreater(proof.canonical_top1_margin, 0.0)
            self.assertEqual(proof.distractor_count, 8)
            self.assertEqual(proof.visible_question_identity_checks, 6)
            self.assertEqual(proof.target_identity_checks, 6)
            self.assertEqual(proof.leak_checked_memory_count, 8)
            self.assertEqual(
                dict(proof.similarity_tier_counts),
                {"high": 4, "medium": 2, "low": 2},
            )
            semantic_hashes.update(task.semantic_sha256 for task in orbit.tasks)
        self.assertEqual(len(semantic_hashes), 100)

    def test_clean_and_distracted_only_differ_in_initial_memory(self) -> None:
        clean, distracted = self.generator.generate_orbit(7, split="dev").tasks
        self.assertEqual(clean.source_task, distracted.source_task)
        self.assertEqual(clean.questions, distracted.questions)
        self.assertEqual(clean.target_asins, distracted.target_asins)
        self.assertEqual(clean.budget_cents, distracted.budget_cents)
        self.assertEqual(clean.canonical_query, distracted.canonical_query)
        self.assertEqual(clean.initial_memories, ())
        self.assertEqual(len(distracted.initial_memories), 8)
        self.assertEqual(
            {item.key for item in distracted.initial_memories},
            {clean.canonical_memory_key},
        )
        medium_values = [
            item.value
            for item in distracted.initial_memories
            if item.similarity_tier == "medium"
        ]
        self.assertTrue(medium_values)
        self.assertTrue(all(": color is " not in value for value in medium_values))

    def test_supported_distractor_counts_all_have_strict_top1_margin(self) -> None:
        for count in (1, 2, 3, 4, 8, 16, 32, 64):
            with self.subTest(count=count):
                generator = DistractorRobustnessGenerator(
                    pool=self.pool,
                    seed=233,
                    distractor_count=count,
                )
                proof = verify_distractor_robustness_orbit(
                    generator.generate_orbit(0, split="test"),
                    pool=self.pool,
                )
                self.assertEqual(proof.distractor_count, count)
                self.assertGreater(proof.canonical_top1_margin, 0.0)

    def test_seed_changes_task_stream(self) -> None:
        original = self.generator.generate_orbit(11, split="train")
        changed = DistractorRobustnessGenerator(
            pool=self.pool,
            seed=234,
        ).generate_orbit(11, split="train")
        self.assertNotEqual(original.as_dict(), changed.as_dict())


class DistractorRobustnessTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = DistractorRobustnessGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def _replace_task(self, branch: int, task):
        tasks = list(self.orbit.tasks)
        tasks[branch] = task
        return replace(self.orbit, tasks=tuple(tasks))

    def test_rejects_changed_distractor(self) -> None:
        task = self.orbit.tasks[1]
        memories = list(task.initial_memories)
        memories[0] = replace(memories[0], value=memories[0].value + " tampered")
        with self.assertRaisesRegex(
            DistractorRobustnessDataError,
            "canonical deterministic generation",
        ):
            verify_distractor_robustness_orbit(
                self._replace_task(
                    1,
                    replace(task, initial_memories=tuple(memories)),
                ),
                pool=self.pool,
            )

    def test_rejects_changed_query(self) -> None:
        task = self.orbit.tasks[0]
        with self.assertRaisesRegex(
            DistractorRobustnessDataError,
            "canonical deterministic generation",
        ):
            verify_distractor_robustness_orbit(
                self._replace_task(
                    0,
                    replace(task, canonical_query="customer profile"),
                ),
                pool=self.pool,
            )

    def test_rejects_target_leak_in_initial_memory(self) -> None:
        task = self.orbit.tasks[1]
        memories = list(task.initial_memories)
        memories[0] = replace(
            memories[0],
            value=f"target product {task.target_asins[0]}",
        )
        tampered_task = replace(task, initial_memories=tuple(memories))
        tampered_orbit = self._replace_task(1, tampered_task)
        # Bypass canonical-regeneration comparison only for this focused unit so
        # the independent leakage guard is exercised directly.
        from unittest.mock import patch

        with patch.object(
            DistractorRobustnessGenerator,
            "generate_orbit",
            autospec=True,
            side_effect=lambda _self, *_args, **_kwargs: tampered_orbit,
        ):
            with self.assertRaisesRegex(
                DistractorRobustnessDataError,
                "ASIN leaked",
            ):
                verify_distractor_robustness_orbit(
                    tampered_orbit,
                    pool=self.pool,
                )


class DistractorRobustnessProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = DistractorRobustnessGenerator(pool=self.pool, seed=233)

    def test_fixed_provider_keeps_counterfactual_pair_together(self) -> None:
        provider = VerifiedDistractorRobustnessBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=12,
            start_orbit=3,
            cache_orbits=2,
        )
        clean = provider.get(0)
        distracted = provider.get(1)
        self.assertEqual(clean.orbit_id, distracted.orbit_id)
        self.assertEqual(clean.questions, distracted.questions)
        self.assertEqual(clean.target_asins, distracted.target_asins)
        self.assertEqual(clean.proof_sha256, distracted.proof_sha256)
        self.assertEqual(len(clean.initial_memories), 0)
        self.assertEqual(len(distracted.initial_memories), 8)
        with self.assertRaises(IndexError):
            provider.get(12)

    def test_reseeded_training_stream_accepts_next_seed_epoch(self) -> None:
        provider = VerifiedDistractorRobustnessBundleProvider(
            generator=self.generator,
            split="train",
            task_count=2,
            mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        initial = provider.get(0)
        next_epoch = provider.get(provider.seed_epoch_task_count)
        self.assertNotEqual(initial.task_id, next_epoch.task_id)
        metadata = provider.metadata()
        self.assertEqual(metadata["accepted_index_domain"], "all_nonnegative_integers")
        self.assertEqual(metadata["retrieve_policy"], "query_top1")
        self.assertEqual(metadata["preloaded_distractor_count"], 8)


class DistractorRobustnessRuntimeTests(unittest.TestCase):
    def _make_env(self, branch: int):
        pool = make_fixture_pool()
        generator = DistractorRobustnessGenerator(pool=pool, seed=233)
        orbit = generator.generate_orbit(0, split="train")
        task = orbit.tasks[branch]
        provider = VerifiedDistractorRobustnessBundleProvider(
            generator=generator,
            split="train",
            task_count=2,
        )
        backend = _FakeRecencyNativeBackend(task.source_task)
        env = DistractorRobustnessWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"distractor-{branch}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        observation, info = env.reset(data_idx=branch)
        return env, backend, task, observation, info

    @staticmethod
    def _purchase(env, target_asin: str, expected_index: int):
        _, _, done, truncated, _ = env.step("search[product]")
        assert not done and not truncated
        _, _, done, truncated, _ = env.step(f"click[{target_asin}]")
        assert not done and not truncated
        _, reward, done, truncated, info = env.step("click[Buy Now]")
        assert not truncated
        assert info["tool_ops"][0]["op"] == "BUY"
        assert info["tool_ops"][0]["purchase_correct"] is True
        assert info["current_subtask_index"] == expected_index + 1
        assert reward == (2.0 if expected_index == 5 else 1.0)
        return done, info

    def test_clean_and_distracted_reset_observations_are_byte_identical(self) -> None:
        clean_env, _, _, clean_observation, clean_info = self._make_env(0)
        dirty_env, _, dirty_task, dirty_observation, dirty_info = self._make_env(1)
        try:
            self.assertEqual(clean_observation, dirty_observation)
            self.assertEqual(clean_info["ltm_inventory_count"], 0)
            self.assertEqual(dirty_info["ltm_inventory_count"], 8)
            self.assertEqual(
                dirty_info["memory_state_diff"],
                {"added": [], "updated": [], "deleted": []},
            )
            self.assertEqual(dirty_info["retrieve_policy"], "query_top1")
            self.assertNotIn(dirty_task.canonical_query, dirty_observation)
            for item in dirty_task.initial_memories:
                self.assertNotIn(item.value, dirty_observation)
            for asin in dirty_task.target_asins:
                self.assertNotIn(asin, dirty_observation)
        finally:
            clean_env.close()
            dirty_env.close()

    def test_distracted_branch_adds_correct_memory_then_retrieves_top1(self) -> None:
        env, backend, task, _, _ = self._make_env(1)
        correct_memory_id = f"mem_{len(task.initial_memories):04d}"
        try:
            add_action = "ADD " + json.dumps(
                {
                    "key": task.canonical_memory_key,
                    "value": task.canonical_memory_value,
                }
            )
            _, _, done, _, info = env.step(add_action)
            self.assertFalse(done)
            self.assertEqual(info["memory_ops"][0]["memory_id"], correct_memory_id)

            retrieve_action = "RETRIEVE " + json.dumps(
                {"query": task.canonical_query}
            )
            observation, _, done, _, info = env.step(retrieve_action)
            self.assertFalse(done)
            self.assertEqual(
                info["memory_ops"][0]["retrieved_memory_ids"],
                [correct_memory_id],
            )
            self.assertIn(task.canonical_memory_value, observation)
            for item in task.initial_memories:
                self.assertNotIn(item.value, observation)

            for phase_index, target_asin in enumerate(task.target_asins):
                if phase_index > 0:
                    _, _, done, _, info = env.step(retrieve_action)
                    self.assertFalse(done)
                    self.assertEqual(
                        info["memory_ops"][0]["retrieved_memory_ids"],
                        [correct_memory_id],
                    )
                done, info = self._purchase(env, target_asin, phase_index)
            self.assertTrue(done)
            self.assertTrue(info["episode_success"])
            self.assertEqual(backend.sessions, {})
        finally:
            env.close()

    def test_reset_discards_policy_memory_and_reloads_only_distractors(self) -> None:
        env, backend, task, _, _ = self._make_env(1)
        try:
            env.step(
                "ADD "
                + json.dumps(
                    {
                        "key": task.canonical_memory_key,
                        "value": task.canonical_memory_value,
                    }
                )
            )
            self.assertEqual(len(env.long_term_memory), 9)
            observation, info = env.reset(data_idx=1)
            self.assertEqual(len(env.long_term_memory), 8)
            self.assertEqual(info["ltm_inventory_count"], 8)
            self.assertEqual(
                info["memory_state_diff"],
                {"added": [], "updated": [], "deleted": []},
            )
            self.assertNotIn(task.canonical_memory_value, observation)
            self.assertEqual(env.memory_id_counter, 8)
            self.assertEqual(len(backend.sessions), 1)
        finally:
            env.close()
        self.assertEqual(backend.sessions, {})

    def test_surface_rejects_visible_inventory_or_broader_retrieval(self) -> None:
        pool = make_fixture_pool()
        generator = DistractorRobustnessGenerator(pool=pool, seed=233)
        provider = VerifiedDistractorRobustnessBundleProvider(
            generator=generator,
            split="train",
            task_count=2,
        )
        task = generator.generate_orbit(0, split="train").tasks[0]
        backend = _FakeRecencyNativeBackend(task.source_task)
        with self.assertRaisesRegex(ValueError, "hidden LTM"):
            DistractorRobustnessWebShopEnv(
                provider=provider,
                backend=backend,
                ltm_inventory_mode="keys",
            )
        with self.assertRaisesRegex(ValueError, "query_top1"):
            DistractorRobustnessWebShopEnv(
                provider=provider,
                backend=backend,
                retrieve_policy="standard",
            )


if __name__ == "__main__":
    unittest.main()
