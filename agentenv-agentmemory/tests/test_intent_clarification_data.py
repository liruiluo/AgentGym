from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from agentenv_agentmemory.intent_clarification import (
    PROVIDER_MODE_RESEEDED_STREAM,
    IntentClarificationDataError,
    IntentClarificationGenerator,
    VerifiedIntentClarificationBundleProvider,
    verify_intent_clarification_orbit,
)
from agentenv_agentmemory.intent_clarification_webshop_env import (
    IntentClarificationFilesystemWebShopEnv,
    IntentClarificationWebShopEnv,
)
from tests.test_recency_override_data import (
    _FakeRecencyNativeBackend,
    make_fixture_pool,
)
from tests.workspace_test_support import InProcessTestShellSandbox


class IntentClarificationGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = IntentClarificationGenerator(pool=self.pool, seed=233)

    def test_one_hundred_tasks_are_deterministic_and_exhaustively_verified(self) -> None:
        semantic_hashes: set[str] = set()
        for orbit_index in range(50):
            orbit = self.generator.generate_orbit(orbit_index, split="train")
            repeated = IntentClarificationGenerator(
                pool=self.pool,
                seed=233,
            ).generate_orbit(orbit_index, split="train")
            self.assertEqual(orbit.as_dict(), repeated.as_dict())
            proof = verify_intent_clarification_orbit(
                orbit,
                pool=self.pool,
                expected_generator_version=self.generator.version,
                expected_generator_seed=self.generator.seed,
            )
            self.assertEqual(proof.enumerated_path_count_per_branch, 64)
            self.assertEqual(proof.valid_solution_counts, (1, 1))
            self.assertEqual(proof.pre_ask_observation_identity_checks, 6)
            self.assertEqual(proof.post_clarification_target_flip_checks, 6)
            self.assertEqual(proof.later_session_memory_dependency_checks, 5)
            self.assertGreater(proof.top1_retrieval_min_score, 0.0)
            self.assertTrue(proof.payload()["verification"]["training_ready"])
            semantic_hashes.update(task.semantic_sha256 for task in orbit.tasks)
        self.assertEqual(len(semantic_hashes), 100)

    def test_counterfactual_pair_is_ambiguous_until_ask(self) -> None:
        left, right = self.generator.generate_orbit(7, split="dev").tasks
        self.assertEqual(left.questions, right.questions)
        self.assertEqual(
            tuple(phase.candidates for phase in left.source_task.phases),
            tuple(phase.candidates for phase in right.source_task.phases),
        )
        self.assertNotEqual(left.clarification_answer, right.clarification_answer)
        self.assertTrue(
            all(a != b for a, b in zip(left.target_asins, right.target_asins))
        )
        for question, phase in zip(left.questions, left.source_task.phases):
            for candidate in phase.candidates:
                self.assertNotIn(candidate.asin, question)
                self.assertEqual(question.count(candidate.title), 1)

    def test_seed_changes_stream(self) -> None:
        original = self.generator.generate_orbit(11, split="test")
        changed = IntentClarificationGenerator(
            pool=self.pool,
            seed=234,
        ).generate_orbit(11, split="test")
        self.assertNotEqual(original.as_dict(), changed.as_dict())


class IntentClarificationTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = IntentClarificationGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def test_rejects_visible_branch_hint(self) -> None:
        task = self.orbit.tasks[0]
        tasks = list(self.orbit.tasks)
        questions = list(task.questions)
        questions[0] += " Prefer the first listing."
        tasks[0] = replace(task, questions=tuple(questions))
        with self.assertRaisesRegex(
            IntentClarificationDataError,
            "canonical deterministic generation",
        ):
            verify_intent_clarification_orbit(
                replace(self.orbit, tasks=tuple(tasks)),
                pool=self.pool,
            )

    def test_rejects_unflipped_target(self) -> None:
        task = self.orbit.tasks[1]
        targets = list(task.target_asins)
        targets[3] = self.orbit.tasks[0].target_asins[3]
        with self.assertRaises((IntentClarificationDataError, ValueError)):
            tasks = list(self.orbit.tasks)
            tasks[1] = replace(task, target_asins=tuple(targets))
            verify_intent_clarification_orbit(
                replace(self.orbit, tasks=tuple(tasks)),
                pool=self.pool,
            )


class IntentClarificationProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = IntentClarificationGenerator(pool=self.pool, seed=233)

    def test_fixed_provider_keeps_twins_together(self) -> None:
        provider = VerifiedIntentClarificationBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=12,
            start_orbit=3,
            cache_orbits=2,
        )
        left, right = provider.get(0), provider.get(1)
        self.assertEqual(left.orbit_id, right.orbit_id)
        self.assertEqual(left.proof_sha256, right.proof_sha256)
        self.assertEqual(left.questions, right.questions)
        self.assertNotEqual(left.target_asins, right.target_asins)
        with self.assertRaises(IndexError):
            provider.get(12)

    def test_reseeded_training_stream_accepts_next_seed_epoch(self) -> None:
        provider = VerifiedIntentClarificationBundleProvider(
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
        self.assertEqual(metadata["required_action"], "ASK")
        self.assertEqual(metadata["retrieve_policy"], "query_top1")
        self.assertEqual(
            metadata["task_prompt_product_identity"],
            "complete_native_title",
        )
        self.assertIs(metadata["target_asin_in_task_prompt"], False)
        self.assertIs(
            metadata["native_search_result_asin_handles_visible"],
            True,
        )
        self.assertIs(metadata["native_click_action_uses_asin_handle"], True)
        self.assertIs(metadata["purchase_receipt_asin_verification"], True)


class IntentClarificationRuntimeTests(unittest.TestCase):
    def _make_env(self, branch: int):
        pool = make_fixture_pool()
        generator = IntentClarificationGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[branch]
        provider = VerifiedIntentClarificationBundleProvider(
            generator=generator,
            split="train",
            task_count=2,
        )
        backend = _FakeRecencyNativeBackend(task.source_task)
        env = IntentClarificationWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"clarification-{branch}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        observation, info = env.reset(data_idx=branch)
        return env, backend, task, observation, info

    @staticmethod
    def _purchase(env, target_asin: str, expected_index: int):
        env.step("search[product]")
        env.step(f"click[{target_asin}]")
        _, reward, done, truncated, info = env.step("click[Buy Now]")
        assert not truncated
        assert info["tool_ops"][0]["purchase_correct"] is True
        assert info["current_subtask_index"] == expected_index + 1
        assert reward == (2.0 if expected_index == 5 else 1.0)
        return done, info

    def test_pre_ask_twins_have_identical_runtime_observation(self) -> None:
        left = self._make_env(0)
        right = self._make_env(1)
        try:
            self.assertEqual(left[3], right[3])
            self.assertFalse(left[4]["ask_completed"])
            self.assertNotIn(left[2].clarification_answer, left[3])
            self.assertNotIn(right[2].clarification_answer, right[3])
        finally:
            left[0].close()
            right[0].close()

    def test_ask_add_and_top1_retrieve_complete_all_sessions(self) -> None:
        env, backend, task, observation, info = self._make_env(0)
        fact = task.canonical_memory
        try:
            self.assertIn("Intent clarification action", observation)
            self.assertIn('- ASK {"field":"..."}', observation)
            self.assertNotIn(
                f'ASK {{"field":"{task.clarification_field}"}}',
                observation,
            )
            _, reward, done, _, info = env.step(
                f'ASK {{"field":"{task.clarification_field}"}}'
            )
            self.assertEqual(reward, 0.0)
            self.assertFalse(done)
            self.assertTrue(info["ask_completed"])
            self.assertEqual(info["tool_ops"][0]["op"], "CLARIFY")
            self.assertNotIn("answer", info["tool_ops"][0])

            _, reward, _, _, info = env.step(
                "ADD " + json.dumps({"key": fact.key, "value": fact.value})
            )
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["memory_ops"][0]["memory_id"], "mem_0000")

            _, repeat_reward, repeat_done, _, repeat_info = env.step(
                f'ASK {{"field":"{task.clarification_field}"}}'
            )
            self.assertLess(repeat_reward, 0.0)
            self.assertFalse(repeat_done)
            self.assertEqual(repeat_info["reward_components"][0]["name"], "invalid_action")
            self._purchase(env, task.target_asins[0], 0)

            for phase_index in range(1, 6):
                _, reward, done, _, info = env.step(
                    "RETRIEVE " + json.dumps({"query": fact.query})
                )
                self.assertEqual(reward, 0.0)
                self.assertFalse(done)
                self.assertEqual(
                    info["memory_ops"][0]["retrieved_memory_ids"],
                    ["mem_0000"],
                )
                done, info = self._purchase(
                    env,
                    task.target_asins[phase_index],
                    phase_index,
                )
            self.assertTrue(done)
            self.assertTrue(info["episode_success"])
            self.assertEqual(len(backend.sessions), 0)
        finally:
            env.close()

    def test_purchase_before_ask_and_malformed_ask_are_invalid(self) -> None:
        env, _, task, _, _ = self._make_env(1)
        try:
            env.step("search[product]")
            env.step(f"click[{task.target_asins[0]}]")
            _, reward, done, _, info = env.step("click[Buy Now]")
            self.assertLess(reward, 0.0)
            self.assertFalse(done)
            self.assertEqual(info["current_subtask_index"], 0)
            self.assertFalse(info["ask_completed"])

            _, reward, done, _, info = env.step("ASK not-json")
            self.assertLess(reward, 0.0)
            self.assertFalse(done)
            self.assertFalse(info["ask_completed"])
            self.assertEqual(info["reward_components"][0]["op"], "INVALID")
        finally:
            env.close()


class IntentClarificationFilesystemRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.pool = make_fixture_pool()
        self.generator = IntentClarificationGenerator(pool=self.pool, seed=233)
        self.provider = VerifiedIntentClarificationBundleProvider(
            generator=self.generator,
            split="train",
            task_count=2,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_env(self, branch: int = 0):
        task = self.generator.generate_orbit(0, split="train").tasks[branch]
        backend = _FakeRecencyNativeBackend(task.source_task)
        env = IntentClarificationFilesystemWebShopEnv(
            provider=self.provider,
            backend=backend,
            env_uid=f"clarification-filesystem-{branch}",
            shell_sandbox=InProcessTestShellSandbox(),
            workspace_root_parent=Path(self.temporary.name),
        )
        observation, info = env.reset(data_idx=branch)
        return env, backend, task, observation, info

    def test_ask_is_only_path_specific_to_intent_surface(self) -> None:
        env, backend, task, observation, info = self._make_env()
        try:
            self.assertIn('- ASK {"field":"..."}', observation)
            self.assertNotIn(
                f'ASK {{"field":"{task.clarification_field}"}}',
                observation,
            )
            self.assertNotIn("ltm_inventory_mode", info)
            _, reward, done, _, info = env.step(
                f'ASK {{"field":"{task.clarification_field}"}}'
            )
            self.assertEqual(reward, 0.0)
            self.assertFalse(done)
            self.assertTrue(info["ask_completed"])
            self.assertEqual(info["tool_ops"][0]["op"], "CLARIFY")
            self.assertEqual(len(backend.sessions), 1)
        finally:
            env.close()

    def test_unclarified_purchase_is_normal_invalid_action_and_ask_repeats_fail(self) -> None:
        env, _, task, _, _ = self._make_env(1)
        try:
            env.step("search[product]")
            env.step(f"click[{task.target_asins[0]}]")
            _, reward, done, _, info = env.step("click[Buy Now]")
            self.assertLess(reward, 0.0)
            self.assertFalse(done)
            self.assertEqual(info["current_subtask_index"], 0)
            self.assertEqual(info["reward_components"][0]["name"], "invalid_action")
            self.assertEqual(info["reward_components"][0]["op"], "CLICK")
            self.assertFalse(info["ask_completed"])

            malformed_actions = (
                "ASK not-json",
                "ASK {bad json}",
                "ASK [1]",
                "shell_command {bad json}",
                "apply_patch *** Begin Patch",
            )
            for action in malformed_actions:
                with self.subTest(action=action):
                    _, reward, done, _, info = env.step(action)
                    self.assertLess(reward, 0.0)
                    self.assertFalse(done)
                    self.assertFalse(info["ask_completed"])
                    self.assertEqual(
                        info["reward_components"][0]["name"], "invalid_action"
                    )
                    self.assertEqual(info["reward_components"][0]["op"], "INVALID")

            failed_ask_attempts = (
                "ASK {}",
                'ASK {"field":""}',
                'ASK {"field":"wrong"}',
                'ASK {"field":"color","extra":true}',
            )
            for action in failed_ask_attempts:
                with self.subTest(action=action):
                    _, reward, done, _, info = env.step(action)
                    self.assertLess(reward, 0.0)
                    self.assertFalse(done)
                    self.assertFalse(info["ask_completed"])
                    self.assertEqual(info["reward_components"][0]["op"], "ASK")

            env.step(f'ASK {{"field":"{task.clarification_field}"}}')
            _, reward, done, _, info = env.step(
                f'ASK {{"field":"{task.clarification_field}"}}'
            )
            self.assertLess(reward, 0.0)
            self.assertFalse(done)
            self.assertEqual(info["reward_components"][0]["name"], "invalid_action")
            self.assertEqual(info["reward_components"][0]["op"], "ASK")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
