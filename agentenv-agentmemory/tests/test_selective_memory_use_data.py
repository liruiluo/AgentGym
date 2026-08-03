from __future__ import annotations

import json
import unittest
from dataclasses import replace

from agentenv_agentmemory.selective_memory_use import (
    PROVIDER_MODE_RESEEDED_STREAM,
    SelectiveMemoryUseDataError,
    SelectiveMemoryUseGenerator,
    VerifiedSelectiveMemoryUseBundleProvider,
    verify_selective_memory_use_orbit,
)
from agentenv_agentmemory.selective_memory_use_webshop_env import (
    SelectiveMemoryUseWebShopEnv,
)
from agentenv_agentmemory.selective_memory_use.verifier import (
    _request_leaks_memory_use_instruction,
)
from tests.test_recency_override_data import (
    _FakeRecencyNativeBackend,
    make_fixture_pool,
)


class SelectiveMemoryUseGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = SelectiveMemoryUseGenerator(pool=self.pool, seed=233)

    def test_one_hundred_tasks_are_deterministic_and_exhaustively_verified(self) -> None:
        semantic_hashes: set[str] = set()
        for orbit_index in range(25):
            orbit = self.generator.generate_orbit(orbit_index, split="train")
            repeated = SelectiveMemoryUseGenerator(
                pool=self.pool,
                seed=233,
            ).generate_orbit(orbit_index, split="train")
            self.assertEqual(orbit.as_dict(), repeated.as_dict())

            proof = verify_selective_memory_use_orbit(
                orbit,
                pool=self.pool,
                expected_generator_version=self.generator.version,
                expected_generator_seed=self.generator.seed,
            )
            self.assertEqual(proof.enumerated_path_count_per_branch, 64)
            self.assertEqual(proof.valid_solution_counts, (1, 1, 1, 1))
            self.assertEqual(proof.required_observation_identity_checks, 6)
            self.assertEqual(proof.required_target_flip_checks, 6)
            self.assertEqual(proof.explicit_constraint_checks, 12)
            self.assertEqual(proof.stale_memory_conflict_checks, 12)
            self.assertEqual(proof.top1_positive_score_checks, 4)
            self.assertEqual(proof.certified_candidate_checks, 48)
            self.assertFalse(
                proof.payload()["verification"]["memory_action_positive_shaping_allowed"]
            )
            semantic_hashes.update(task.semantic_sha256 for task in orbit.tasks)

        self.assertEqual(len(semantic_hashes), 100)

    def test_factorial_orbit_separates_memory_need_from_preference(self) -> None:
        required_a, not_required_a, required_b, not_required_b = (
            self.generator.generate_orbit(7, split="dev").tasks
        )

        self.assertEqual(required_a.questions, required_b.questions)
        self.assertNotEqual(required_a.target_asins, required_b.target_asins)
        self.assertEqual(required_a.target_asins, not_required_a.target_asins)
        self.assertEqual(required_b.target_asins, not_required_b.target_asins)
        self.assertEqual(required_a.initial_memory.state, "current")
        self.assertEqual(not_required_a.initial_memory.state, "stale")
        self.assertNotIn("stale", not_required_a.initial_memory.value.casefold())
        self.assertNotIn("superseded", not_required_a.initial_memory.value.casefold())
        recipe = self.pool.recipe_by_id(required_a.source_task.recipe_id)
        display_a = recipe.value_display_name(required_a.preferred_attribute_value)
        display_b = recipe.value_display_name(required_b.preferred_attribute_value)
        for question in required_a.questions:
            request_text = question.split("\n\nApproved listings:", 1)[0]
            self.assertNotIn(display_a, request_text)
            self.assertNotIn(display_b, request_text)
        for task, expected_display, opposite_display in (
            (not_required_a, display_a, display_b),
            (not_required_b, display_b, display_a),
        ):
            for question in task.questions:
                request_text = question.split("\n\n", 1)[1].split(
                    "\n\nApproved listings:", 1
                )[0]
                self.assertIn(
                    f"required {recipe.axis_display_name} value is {expected_display}",
                    request_text,
                )
                self.assertNotIn(opposite_display, request_text)
                for forbidden in (
                    "ADD",
                    "RETRIEVE",
                    "memory",
                    "saved current profile",
                    "older profile history",
                ):
                    self.assertNotIn(forbidden.casefold(), request_text.casefold())

    def test_instruction_leak_check_ignores_ids_and_listing_content(self) -> None:
        question = (
            "Customer: shopper.train.5815cad430eadd22\n"
            "Purchase 1 of 6: mattress topper\n"
            "Total six-purchase budget: $380.00\n\n"
            "The customer wants their usual choice for firmness; this request "
            "does not state which value that is. Both approved listings satisfy "
            "every other current requirement.\n\n"
            "Approved listings:\n"
            "1. Memory Foam Padded Topper\n"
            "2. Standard Topper\n\n"
            "Purchase exactly one approved listing."
        )
        self.assertFalse(_request_leaks_memory_use_instruction(question))
        leaked = question.replace(
            "The customer wants their usual choice for firmness",
            "Use RETRIEVE to find the customer's usual choice for firmness",
        )
        self.assertTrue(_request_leaks_memory_use_instruction(leaked))

    def test_seed_changes_task_stream(self) -> None:
        original = self.generator.generate_orbit(11, split="train")
        changed = SelectiveMemoryUseGenerator(
            pool=self.pool,
            seed=234,
        ).generate_orbit(11, split="train")
        self.assertNotEqual(original.as_dict(), changed.as_dict())


class SelectiveMemoryUseTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = SelectiveMemoryUseGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def _replace_task(self, branch: int, task):
        tasks = list(self.orbit.tasks)
        tasks[branch] = task
        return replace(self.orbit, tasks=tuple(tasks))

    def test_rejects_changed_explicit_constraint(self) -> None:
        task = self.orbit.tasks[1]
        questions = list(task.questions)
        questions[0] += " tampered"
        with self.assertRaisesRegex(
            SelectiveMemoryUseDataError,
            "canonical deterministic generation",
        ):
            verify_selective_memory_use_orbit(
                self._replace_task(1, replace(task, questions=tuple(questions))),
                pool=self.pool,
            )

    def test_rejects_changed_seeded_memory(self) -> None:
        task = self.orbit.tasks[3]
        memory = replace(task.initial_memory, value=task.initial_memory.value + " tampered")
        with self.assertRaisesRegex(
            SelectiveMemoryUseDataError,
            "canonical deterministic generation",
        ):
            verify_selective_memory_use_orbit(
                self._replace_task(3, replace(task, initial_memory=memory)),
                pool=self.pool,
            )

    def test_task_schema_rejects_wrong_memory_state(self) -> None:
        task = self.orbit.tasks[0]
        with self.assertRaisesRegex(
            SelectiveMemoryUseDataError,
            "seeded memory state",
        ):
            replace(
                task,
                initial_memory=replace(task.initial_memory, state="stale"),
            )


class SelectiveMemoryUseProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = SelectiveMemoryUseGenerator(pool=self.pool, seed=233)

    def test_fixed_provider_keeps_factorial_orbit_together(self) -> None:
        provider = VerifiedSelectiveMemoryUseBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=12,
            start_orbit=3,
            cache_orbits=2,
        )
        tasks = tuple(provider.get(index) for index in range(4))
        self.assertEqual(len({task.orbit_id for task in tasks}), 1)
        self.assertEqual(len({task.proof_sha256 for task in tasks}), 1)
        self.assertEqual(
            tuple(task.branch_kind for task in tasks),
            (
                "memory_required_a",
                "memory_not_required_a",
                "memory_required_b",
                "memory_not_required_b",
            ),
        )
        self.assertEqual(
            provider.metadata()["fixed_window"],
            {"start_orbit": 3, "end_orbit_exclusive": 6},
        )
        with self.assertRaises(IndexError):
            provider.get(12)

    def test_reseeded_stream_keeps_factorial_orbit_inside_seed_epoch(self) -> None:
        provider = VerifiedSelectiveMemoryUseBundleProvider(
            generator=self.generator,
            split="train",
            task_count=4,
            mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        initial = provider.get(0)
        next_epoch = provider.get(provider.seed_epoch_task_count)
        self.assertNotEqual(initial.task_id, next_epoch.task_id)
        metadata = provider.metadata()
        self.assertEqual(metadata["accepted_index_domain"], "all_nonnegative_integers")
        self.assertEqual(metadata["retrieve_policy"], "query_top1")
        self.assertEqual(metadata["memory_required_fraction"], 0.5)
        self.assertEqual(metadata["memory_not_required_fraction"], 0.5)
        self.assertFalse(metadata["memory_action_positive_shaping_allowed"])

    def test_task_count_must_preserve_complete_factorial_orbits(self) -> None:
        with self.assertRaisesRegex(SelectiveMemoryUseDataError, "multiple of four"):
            VerifiedSelectiveMemoryUseBundleProvider(
                generator=self.generator,
                split="train",
                task_count=2,
            )


class SelectiveMemoryUseRuntimeTests(unittest.TestCase):
    def _make_env(self, data_idx: int):
        pool = make_fixture_pool()
        generator = SelectiveMemoryUseGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[data_idx]
        provider = VerifiedSelectiveMemoryUseBundleProvider(
            generator=generator,
            split="train",
            task_count=4,
        )
        backend = _FakeRecencyNativeBackend(task.source_task)
        env = SelectiveMemoryUseWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"selective-{data_idx}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        observation, info = env.reset(data_idx=data_idx)
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

    def test_required_branch_retrieves_hidden_current_profile_by_query_top1(self) -> None:
        env, backend, task, observation, info = self._make_env(0)
        retrieve = "RETRIEVE " + json.dumps({"query": task.canonical_query})
        try:
            self.assertEqual(info["ltm_inventory_count"], 1)
            self.assertNotIn(task.initial_memory.value, observation)
            for phase_index, target_asin in enumerate(task.target_asins):
                observation, reward, done, _, info = env.step(retrieve)
                self.assertFalse(done)
                self.assertEqual(reward, 0.0)
                self.assertEqual(
                    info["memory_ops"][0]["retrieved_memory_ids"],
                    ["mem_0000"],
                )
                self.assertIn(task.initial_memory.value, observation)
                done, info = self._purchase(env, target_asin, phase_index)
            self.assertTrue(done)
            self.assertTrue(info["episode_success"])
            self.assertEqual(backend.sessions, {})
        finally:
            env.close()

    def test_not_required_branch_completes_without_any_memory_action(self) -> None:
        env, backend, task, observation, info = self._make_env(1)
        try:
            self.assertIn("For this purchase", observation)
            self.assertNotIn(task.initial_memory.value, observation)
            for phase_index, target_asin in enumerate(task.target_asins):
                done, info = self._purchase(env, target_asin, phase_index)
            self.assertTrue(done)
            self.assertTrue(info["episode_success"])
            self.assertFalse(
                any(
                    component["name"] == "memory_action_not_required"
                    for component in info["reward_components"]
                )
            )
            self.assertEqual(backend.sessions, {})
        finally:
            env.close()

    def test_not_required_retrieve_exposes_conflict_and_is_penalized(self) -> None:
        env, _, task, observation, _ = self._make_env(3)
        try:
            current_value = task.preferred_attribute_value
            self.assertIn(current_value, observation.casefold())
            retrieve = "RETRIEVE " + json.dumps({"query": task.canonical_query})
            observation, reward, done, truncated, info = env.step(retrieve)
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(reward, -0.01)
            self.assertIn(task.initial_memory.value, observation)
            component = info["reward_components"][-1]
            self.assertEqual(component["name"], "memory_action_not_required")
            self.assertEqual(component["memory_requirement"], "memory_not_required")
            self.assertEqual(component["value"], -0.01)
        finally:
            env.close()

    def test_required_add_is_unnecessary_and_penalized(self) -> None:
        env, _, _, _, _ = self._make_env(2)
        try:
            action = 'ADD {"key":"duplicate","value":"unneeded"}'
            _, reward, done, truncated, info = env.step(action)
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(reward, -0.01)
            self.assertEqual(
                info["reward_components"][-1]["name"],
                "memory_action_not_required",
            )
        finally:
            env.close()

    def test_surface_fails_closed_on_bonus_inventory_or_retrieval_override(self) -> None:
        pool = make_fixture_pool()
        generator = SelectiveMemoryUseGenerator(pool=pool, seed=233)
        provider = VerifiedSelectiveMemoryUseBundleProvider(
            generator=generator,
            split="train",
            task_count=4,
        )
        task = generator.generate_orbit(0, split="train").tasks[0]
        backend = _FakeRecencyNativeBackend(task.source_task)
        with self.assertRaisesRegex(ValueError, "positive memory-action shaping"):
            SelectiveMemoryUseWebShopEnv(
                provider=provider,
                backend=backend,
                first_valid_add_reward=0.1,
            )
        with self.assertRaisesRegex(ValueError, "hidden LTM"):
            SelectiveMemoryUseWebShopEnv(
                provider=provider,
                backend=backend,
                ltm_inventory_mode="keys",
            )
        with self.assertRaisesRegex(ValueError, "query_top1"):
            SelectiveMemoryUseWebShopEnv(
                provider=provider,
                backend=backend,
                retrieve_policy="standard",
            )


if __name__ == "__main__":
    unittest.main()
