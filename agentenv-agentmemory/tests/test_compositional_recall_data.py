from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.compositional_recall import (
    PROVIDER_MODE_RESEEDED_STREAM,
    CompositionalRecallDataError,
    CompositionalRecallGenerator,
    VerifiedCompositionalRecallBundleProvider,
    verify_compositional_recall_orbit,
)
from agentenv_agentmemory.compositional_recall_webshop_env import (
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    CompositionalRecallFilesystemWebShopEnv,
    CompositionalRecallWebShopEnv,
)
from tests.test_recency_override_data import (
    _FakeRecencyNativeBackend,
    make_fixture_pool,
)
from tests.workspace_test_support import InProcessTestShellSandbox


class CompositionalRecallGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = CompositionalRecallGenerator(pool=self.pool, seed=233)

    def test_one_hundred_tasks_are_deterministic_and_exhaustively_verified(self) -> None:
        semantic_hashes: set[str] = set()
        for orbit_index in range(25):
            orbit = self.generator.generate_orbit(orbit_index, split="train")
            repeated = CompositionalRecallGenerator(
                pool=self.pool,
                seed=233,
            ).generate_orbit(orbit_index, split="train")
            self.assertEqual(orbit.as_dict(), repeated.as_dict())
            proof = verify_compositional_recall_orbit(
                orbit,
                pool=self.pool,
                expected_generator_version=self.generator.version,
                expected_generator_seed=self.generator.seed,
            )
            self.assertEqual(proof.valid_solution_counts, (1, 1, 1, 1))
            self.assertEqual(proof.enumerated_path_count_per_branch, 64)
            self.assertGreater(proof.hop1_min_top1_margin, 0.0)
            self.assertGreater(proof.hop2_min_top1_margin, 0.0)
            self.assertEqual(proof.sequential_token_bridge_checks, 4)
            self.assertEqual(proof.mapping_leave_one_out_checks, 10)
            self.assertEqual(proof.directory_leave_one_out_checks, 8)
            semantic_hashes.update(task.semantic_sha256 for task in orbit.tasks)
        self.assertEqual(len(semantic_hashes), 100)

    def test_two_by_two_factorial_is_xor_over_application_targets(self) -> None:
        tasks = self.generator.generate_orbit(7, split="dev").tasks
        aa, ab, ba, bb = tasks
        self.assertEqual({task.questions[2:] for task in tasks}, {aa.questions[2:]})
        self.assertEqual(aa.target_asins[2:], bb.target_asins[2:])
        self.assertEqual(ab.target_asins[2:], ba.target_asins[2:])
        self.assertNotEqual(aa.target_asins[2:], ab.target_asins[2:])

        self.assertEqual(aa.canonical_memories[0], ab.canonical_memories[0])
        self.assertEqual(ba.canonical_memories[0], bb.canonical_memories[0])
        self.assertEqual(
            aa.canonical_memories[1].value,
            ba.canonical_memories[1].value,
        )
        self.assertEqual(
            ab.canonical_memories[1].value,
            bb.canonical_memories[1].value,
        )
        self.assertNotEqual(
            aa.canonical_memories[0].value,
            ba.canonical_memories[0].value,
        )
        self.assertNotEqual(
            aa.canonical_memories[1].value,
            ab.canonical_memories[1].value,
        )

    def test_application_hides_both_opaque_profile_tokens(self) -> None:
        orbit = self.generator.generate_orbit(3, split="test")
        for task in orbit.tasks:
            for question in task.questions[2:]:
                for token in task.profile_tokens:
                    self.assertNotIn(token, question)
            for phase, question in zip(task.source_task.phases, task.questions):
                for candidate in phase.candidates:
                    self.assertNotIn(candidate.asin, question)

    def test_seed_changes_task_stream(self) -> None:
        original = self.generator.generate_orbit(11, split="train")
        changed = CompositionalRecallGenerator(
            pool=self.pool,
            seed=234,
        ).generate_orbit(11, split="train")
        self.assertNotEqual(original.as_dict(), changed.as_dict())

    def test_verifier_uses_recipe_display_names_not_internal_value_ids(self) -> None:
        display_by_value = {
            "black": "charcoal tone",
            "gray": "ash tone",
        }
        recipe = replace(
            self.pool.recipes[0],
            value_display_names=tuple(
                display_by_value[value] for value in self.pool.recipes[0].values
            ),
        )
        pool = replace(
            self.pool,
            recipes=(recipe,),
            products=tuple(
                replace(
                    product,
                    attribute_display_name=display_by_value[
                        product.attribute_value
                    ],
                )
                for product in self.pool.products
            ),
        )
        generator = CompositionalRecallGenerator(pool=pool, seed=233)
        orbit = generator.generate_orbit(0, split="train")

        proof = verify_compositional_recall_orbit(
            orbit,
            pool=pool,
            expected_generator_version=generator.version,
            expected_generator_seed=generator.seed,
        )

        self.assertGreater(proof.hop2_min_top1_margin, 0.0)
        for task in orbit.tasks:
            directory = task.canonical_memories[1]
            self.assertNotIn(task.preferred_attribute_value, directory.value)
            self.assertIn(
                recipe.value_display_name(task.preferred_attribute_value),
                directory.value,
            )


class CompositionalRecallTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = CompositionalRecallGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def _replace_task(self, branch: int, task):
        tasks = list(self.orbit.tasks)
        tasks[branch] = task
        return replace(self.orbit, tasks=tuple(tasks))

    def test_rejects_changed_bridge_query(self) -> None:
        task = self.orbit.tasks[0]
        memories = list(task.canonical_memories)
        memories[1] = replace(memories[1], query="profile preference")
        with self.assertRaisesRegex(
            CompositionalRecallDataError,
            "canonical deterministic generation",
        ):
            verify_compositional_recall_orbit(
                self._replace_task(
                    0,
                    replace(task, canonical_memories=tuple(memories)),
                ),
                pool=self.pool,
            )

    def test_independent_factorial_guard_rejects_unflipped_target(self) -> None:
        task = self.orbit.tasks[2]
        targets = list(task.target_asins)
        targets[3] = self.orbit.tasks[0].target_asins[3]
        tampered = self._replace_task(2, replace(task, target_asins=tuple(targets)))
        with patch.object(
            CompositionalRecallGenerator,
            "generate_orbit",
            autospec=True,
            side_effect=lambda _self, *_args, **_kwargs: tampered,
        ):
            with self.assertRaisesRegex(
                CompositionalRecallDataError,
                "mapping bit must flip",
            ):
                verify_compositional_recall_orbit(tampered, pool=self.pool)

    def test_independent_top1_guard_rejects_wrong_query(self) -> None:
        task = self.orbit.tasks[0]
        memories = list(task.canonical_memories)
        memories[0] = replace(memories[0], query=memories[1].query)
        tampered = self._replace_task(
            0,
            replace(task, canonical_memories=tuple(memories)),
        )
        with patch.object(
            CompositionalRecallGenerator,
            "generate_orbit",
            autospec=True,
            side_effect=lambda _self, *_args, **_kwargs: tampered,
        ):
            with self.assertRaisesRegex(
                CompositionalRecallDataError,
                "hop 1 canonical query",
            ):
                verify_compositional_recall_orbit(tampered, pool=self.pool)


class CompositionalRecallProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = CompositionalRecallGenerator(pool=self.pool, seed=233)

    def test_fixed_provider_keeps_factorial_orbit_together(self) -> None:
        provider = VerifiedCompositionalRecallBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=12,
            start_orbit=3,
            cache_orbits=2,
        )
        bundles = tuple(provider.get(index) for index in range(4))
        self.assertEqual(len({bundle.orbit_id for bundle in bundles}), 1)
        self.assertEqual(len({bundle.proof_sha256 for bundle in bundles}), 1)
        self.assertEqual(len({bundle.questions[2:] for bundle in bundles}), 1)
        with self.assertRaises(IndexError):
            provider.get(12)

    def test_reseeded_training_stream_accepts_next_seed_epoch(self) -> None:
        provider = VerifiedCompositionalRecallBundleProvider(
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
        self.assertEqual(metadata["required_sequential_retrievals"], 2)
        self.assertTrue(metadata["leave_one_memory_out_certified"])


class CompositionalRecallRuntimeTests(unittest.TestCase):
    def _make_env(self, data_idx: int = 1):
        pool = make_fixture_pool()
        generator = CompositionalRecallGenerator(pool=pool, seed=233)
        orbit = generator.generate_orbit(0, split="train")
        task = orbit.tasks[data_idx]
        provider = VerifiedCompositionalRecallBundleProvider(
            generator=generator,
            split="train",
            task_count=4,
        )
        backend = _FakeRecencyNativeBackend(task.source_task)
        env = CompositionalRecallWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"compositional-{data_idx}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        observation, info = env.reset(data_idx=data_idx)
        return env, backend, task, observation, info

    @staticmethod
    def _add(env, fact):
        return env.step(
            "ADD " + json.dumps({"key": fact.key, "value": fact.value})
        )

    @staticmethod
    def _retrieve(env, fact):
        return env.step(
            "RETRIEVE " + json.dumps({"query": fact.query})
        )

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

    def test_two_sequential_top1_retrievals_complete_six_sessions(self) -> None:
        env, backend, task, observation, info = self._make_env(1)
        mapping, directory = task.canonical_memories
        recipe = make_fixture_pool().recipe_by_id(task.source_task.recipe_id)
        preferred_display = recipe.value_display_name(
            task.preferred_attribute_value
        )
        try:
            self.assertEqual(info["retrieve_policy"], "query_top1")
            self.assertNotIn(task.target_asins[0], observation)

            _, _, done, _, info = self._add(env, mapping)
            self.assertFalse(done)
            self.assertEqual(info["memory_ops"][0]["memory_id"], "mem_0000")
            self._purchase(env, task.target_asins[0], 0)

            _, _, done, _, info = self._add(env, directory)
            self.assertFalse(done)
            self.assertEqual(info["memory_ops"][0]["memory_id"], "mem_0001")

            first, _, done, _, info = self._retrieve(env, mapping)
            self.assertFalse(done)
            self.assertEqual(
                info["memory_ops"][0]["retrieved_memory_ids"],
                ["mem_0000"],
            )
            self.assertIn(task.active_profile_token, first)
            second, _, done, _, info = self._retrieve(env, directory)
            self.assertFalse(done)
            self.assertEqual(
                info["memory_ops"][0]["retrieved_memory_ids"],
                ["mem_0001"],
            )
            self.assertIn(task.active_profile_token, second)
            self.assertIn(preferred_display, second)
            self._purchase(env, task.target_asins[1], 1)

            for phase_index in range(2, 6):
                first, _, done, _, info = self._retrieve(env, mapping)
                self.assertFalse(done)
                self.assertIn(task.active_profile_token, first)
                second, _, done, _, info = self._retrieve(env, directory)
                self.assertFalse(done)
                self.assertIn(preferred_display, second)
                done, info = self._purchase(
                    env,
                    task.target_asins[phase_index],
                    phase_index,
                )
            self.assertTrue(done)
            self.assertTrue(info["episode_success"])
            self.assertEqual(backend.sessions, {})
        finally:
            env.close()

    def test_reset_clears_both_policy_authored_memories(self) -> None:
        env, backend, task, _, _ = self._make_env(2)
        try:
            self._add(env, task.canonical_memories[0])
            self._add(env, task.canonical_memories[1])
            self.assertEqual(len(env.long_term_memory), 2)
            observation, info = env.reset(data_idx=2)
            self.assertEqual(len(env.long_term_memory), 0)
            self.assertEqual(info["ltm_inventory_count"], 0)
            self.assertEqual(env.memory_id_counter, 0)
            for fact in task.canonical_memories:
                self.assertNotIn(fact.value, observation)
            self.assertEqual(len(backend.sessions), 1)
        finally:
            env.close()
        self.assertEqual(backend.sessions, {})

    def test_filesystem_surface_persists_both_hops_to_session_two(self) -> None:
        pool = make_fixture_pool()
        generator = CompositionalRecallGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[1]
        provider = VerifiedCompositionalRecallBundleProvider(
            generator=generator,
            split="train",
            task_count=4,
        )
        backend = _FakeRecencyNativeBackend(task.source_task)
        with tempfile.TemporaryDirectory() as root:
            env = CompositionalRecallFilesystemWebShopEnv(
                provider=provider,
                backend=backend,
                env_uid="compositional-filesystem-1",
                shell_sandbox=InProcessTestShellSandbox(),
                workspace_root_parent=Path(root),
            )
            try:
                _, info = env.reset(data_idx=1)
                self.assertEqual(info["surface"], COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE)
                mapping, directory = task.canonical_memories
                for path, fact, phase_index in (
                    ("mapping.md", mapping, 0),
                    ("directory.md", directory, 1),
                ):
                    env.step(
                        "apply_patch\n*** Begin Patch\n"
                        f"*** Add File: {path}\n+{fact.value}\n"
                        "*** End Patch"
                    )
                    self._purchase(env, task.target_asins[phase_index], phase_index)
                self.assertEqual(env.current_session_index, 2)
                state = env.workspace.export_state()
                self.assertEqual(state["file_count"], 2)
                _, info = env.install_workspace_causal_intervention("blank")
                self.assertEqual(info["workspace_causal_arm"], "blank")
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
