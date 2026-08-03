from __future__ import annotations

import json
import unittest
from dataclasses import replace

from agentenv_agentmemory.latent_preference.schema import (
    canonical_sha256,
    normalize_native_title,
)
from agentenv_agentmemory.native_webshop_backend import NativePage, NativePurchase
from agentenv_agentmemory.negative_constraint import (
    PROVIDER_MODE_RESEEDED_STREAM,
    NegativeConstraintCandidate,
    NegativeConstraintDataError,
    NegativeConstraintGenerator,
    NegativeConstraintProductPool,
    NegativeConstraintRecipe,
    VerifiedNegativeConstraintBundleProvider,
    verify_negative_constraint_orbit,
)
from agentenv_agentmemory.negative_constraint_webshop_env import (
    NegativeConstraintWebShopEnv,
)


CATEGORIES = (
    ("area_rug", "area rug"),
    ("phone_case", "phone case"),
    ("pillowcase", "pillowcase"),
    ("window_curtain", "window curtain"),
)
VALUES = (("black", "black"), ("gray", "gray"), ("red", "red"))
SPLITS = ("train", "dev", "test")


def make_negative_fixture_pool() -> NegativeConstraintProductPool:
    recipe = NegativeConstraintRecipe(
        recipe_id="color.black_gray_red",
        axis="color",
        axis_display_name="color",
        values=tuple(value for value, _ in VALUES),
        value_display_names=tuple(display for _, display in VALUES),
        categories=tuple(category for category, _ in CATEGORIES),
        category_display_names=tuple(display for _, display in CATEGORIES),
    )
    candidates = []
    ordinal = 1
    for category_id, category_display in CATEGORIES:
        for value, value_display in VALUES:
            for split in SPLITS:
                for cell_index in range(2):
                    asin = f"N{ordinal:09d}"
                    title = (
                        f"{value_display.title()} {category_display.title()} "
                        f"Native Model {cell_index} {split}"
                    )
                    candidates.append(
                        NegativeConstraintCandidate(
                            asin=asin,
                            title=title,
                            normalized_title=normalize_native_title(title),
                            product_category=f"Fixture > {category_display}",
                            category_id=category_id,
                            category_display_name=category_display,
                            axis="color",
                            attribute_value=value,
                            attribute_display_name=value_display,
                            title_evidence=(value_display.title(),),
                            split=split,
                            source_classification_sha256=canonical_sha256(
                                {"classification": asin}
                            ),
                            source_row_sha256=canonical_sha256({"source_row": asin}),
                        )
                    )
                    ordinal += 1
    candidates.sort(
        key=lambda item: (
            item.axis,
            item.category_id,
            item.attribute_value,
            item.split,
            item.asin,
        )
    )
    return NegativeConstraintProductPool(
        pool_id="fixture_negative_constraint_pool",
        products_per_cell=2,
        recipes=(recipe,),
        candidates=tuple(candidates),
        candidate_artifact_sha256="1" * 64,
        split_policy="fixture_asin_split_v1",
        selection_policy="fixture_cell_selection_v1",
        native_certified=False,
    )


class NegativeConstraintGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_negative_fixture_pool()
        self.generator = NegativeConstraintGenerator(pool=self.pool, seed=233)

    def test_one_hundred_tasks_are_deterministic_and_exhaustively_verified(self) -> None:
        semantic_hashes: set[str] = set()
        checked = 0
        for orbit_index in range(34):
            orbit = self.generator.generate_orbit(orbit_index, split="train")
            repeated = NegativeConstraintGenerator(
                pool=self.pool,
                seed=233,
            ).generate_orbit(orbit_index, split="train")
            self.assertEqual(orbit.as_dict(), repeated.as_dict())
            proof = verify_negative_constraint_orbit(
                orbit,
                pool=self.pool,
                expected_generator_version=self.generator.version,
                expected_generator_seed=self.generator.seed,
            )
            self.assertEqual(proof.enumerated_path_count_per_branch, 729)
            self.assertEqual(proof.valid_solution_counts, (1, 1, 1))
            self.assertEqual(proof.application_observation_identity_checks, 5)
            self.assertEqual(proof.application_three_target_permutation_checks, 5)
            self.assertGreater(proof.top1_retrieval_min_score, 0.0)
            self.assertFalse(proof.payload()["verification"]["training_ready"])
            for task in orbit.tasks:
                if checked < 100:
                    semantic_hashes.add(task.semantic_sha256)
                    checked += 1
        self.assertEqual(checked, 100)
        self.assertEqual(len(semantic_hashes), 100)

    def test_three_way_counterfactual_shares_application_observations(self) -> None:
        tasks = self.generator.generate_orbit(7, split="dev").tasks
        self.assertEqual(len({task.questions[0] for task in tasks}), 3)
        for phase_index in range(1, 6):
            self.assertEqual(
                len({task.questions[phase_index] for task in tasks}),
                1,
            )
            candidate_asins = {
                item.asin for item in tasks[0].phases[phase_index].candidates
            }
            self.assertEqual(
                {task.target_asins[phase_index] for task in tasks},
                candidate_asins,
            )

    def test_seed_changes_stream(self) -> None:
        original = self.generator.generate_orbit(11, split="test")
        changed = NegativeConstraintGenerator(
            pool=self.pool,
            seed=234,
        ).generate_orbit(11, split="test")
        self.assertNotEqual(original.as_dict(), changed.as_dict())


class NegativeConstraintTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_negative_fixture_pool()
        self.generator = NegativeConstraintGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def _replace_task(self, branch: int, task):
        tasks = list(self.orbit.tasks)
        tasks[branch] = task
        return replace(self.orbit, tasks=tuple(tasks))

    def test_rejects_wrong_declared_target(self) -> None:
        task = self.orbit.tasks[0]
        phase = task.phases[3]
        wrong = next(
            item.asin for item in phase.candidates if item.asin != phase.target_asin
        )
        phases = list(task.phases)
        phases[3] = replace(
            phase,
            target_asin=wrong,
            allowed_attribute_value=next(
                item.attribute_value for item in phase.candidates if item.asin == wrong
            ),
        )
        with self.assertRaises(NegativeConstraintDataError):
            verify_negative_constraint_orbit(
                self._replace_task(0, replace(task, phases=tuple(phases))),
                pool=self.pool,
            )

    def test_rejects_noncanonical_application_question(self) -> None:
        task = self.orbit.tasks[1]
        phases = list(task.phases)
        phases[5] = replace(phases[5], question=phases[5].question + " tampered")
        with self.assertRaisesRegex(
            NegativeConstraintDataError,
            "canonical deterministic generation",
        ):
            verify_negative_constraint_orbit(
                self._replace_task(1, replace(task, phases=tuple(phases))),
                pool=self.pool,
            )


class NegativeConstraintProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_negative_fixture_pool()
        self.generator = NegativeConstraintGenerator(pool=self.pool, seed=233)

    def test_fixed_provider_keeps_three_branches_together(self) -> None:
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=9,
            start_orbit=3,
            cache_orbits=2,
        )
        bundles = tuple(provider.get(index) for index in range(3))
        self.assertEqual(len({bundle.orbit_id for bundle in bundles}), 1)
        self.assertEqual(len({bundle.proof_sha256 for bundle in bundles}), 1)
        self.assertEqual(len({bundle.questions[1:] for bundle in bundles}), 1)
        with self.assertRaises(IndexError):
            provider.get(9)

    def test_reseeded_provider_enters_next_seed_epoch(self) -> None:
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=self.generator,
            split="train",
            task_count=3,
            mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        initial = provider.get(0)
        next_epoch = provider.get(provider.seed_epoch_task_count)
        self.assertNotEqual(initial.task_id, next_epoch.task_id)
        metadata = provider.metadata()
        self.assertEqual(metadata["accepted_index_domain"], "all_nonnegative_integers")
        self.assertTrue(metadata["rules_only"])
        self.assertFalse(metadata["native_certified"])
        self.assertFalse(metadata["training_ready"])


class _FakeNegativeNativeBackend:
    surface = "memoryarena_webshop_native_v1"

    def __init__(self, task) -> None:
        self.phases = {phase.phase_index: phase for phase in task.phases}
        self.sessions: dict[str, dict[str, object]] = {}
        self.prices = {
            candidate.asin: 1_000
            for phase in task.phases
            for candidate in phase.candidates
        }

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        phase_index = int(session_token.rsplit("_", 1)[-1])
        self.sessions[session_token] = {
            "phase_index": phase_index,
            "instruction": instruction,
            "asin": None,
        }
        return NativePage(
            observation=f"Fixture WebShop [SEP] {instruction} [SEP] Search",
            url=f"http://fixture/{session_token}",
            has_search_bar=True,
            clickables=(),
        )

    def step(self, session_token: str, action: str) -> NativePage:
        session = self.sessions[session_token]
        phase = self.phases[int(session["phase_index"])]
        instruction = str(session["instruction"])
        if action.startswith("search["):
            handles = tuple(candidate.asin for candidate in phase.candidates)
            return NativePage(
                observation=f"{instruction} [SEP] Results [SEP] " + " [SEP] ".join(handles),
                url=f"http://fixture/{session_token}/search",
                has_search_bar=True,
                clickables=handles,
            )
        if action.startswith("click[") and action.lower() != "click[buy now]":
            asin = action[6:-1].upper()
            if asin not in {item.asin for item in phase.candidates}:
                raise AssertionError(f"click outside current results: {asin}")
            session["asin"] = asin
            return NativePage(
                observation=f"{instruction} [SEP] Product {asin} [SEP] Buy Now",
                url=f"http://fixture/{session_token}/item/{asin}",
                has_search_bar=False,
                clickables=("Buy Now",),
            )
        if action.lower() == "click[buy now]":
            asin = str(session["asin"] or "")
            return NativePage(
                observation=f"{instruction} [SEP] purchase receipt",
                url=f"http://fixture/{session_token}/done/{asin}",
                has_search_bar=False,
                clickables=(),
                purchase=NativePurchase(
                    asin=asin,
                    price_cents=self.prices[asin],
                    selected_options={},
                ),
            )
        raise AssertionError(f"unsupported fixture action: {action}")

    def close_session(self, session_token: str) -> None:
        self.sessions.pop(session_token, None)

    def has_product(self, asin: str) -> bool:
        return asin.upper() in self.prices

    def metadata(self) -> dict[str, object]:
        return {"surface": self.surface}


class NegativeConstraintRuntimeTests(unittest.TestCase):
    def _make_env(self, data_idx: int = 0):
        pool = make_negative_fixture_pool()
        generator = NegativeConstraintGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[data_idx]
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=generator,
            split="train",
            task_count=3,
        )
        backend = _FakeNegativeNativeBackend(task)
        env = NegativeConstraintWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"negative-{data_idx}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        observation, info = env.reset(data_idx=data_idx)
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

    def test_add_then_query_top1_retrieve_completes_six_sessions(self) -> None:
        env, backend, task, observation, info = self._make_env(2)
        try:
            self.assertTrue(info["rules_only"])
            self.assertFalse(info["training_ready"])
            self.assertNotIn(task.canonical_memory_value, observation)
            payload = json.dumps(
                {
                    "key": task.canonical_memory_key,
                    "value": task.canonical_memory_value,
                }
            )
            _, _, done, _, info = env.step("ADD " + payload)
            self.assertFalse(done)
            self.assertEqual(info["memory_ops"][0]["memory_id"], "mem_0000")
            self._purchase(env, task.target_asins[0], 0)

            for phase_index in range(1, 6):
                retrieve = json.dumps(
                    {"query": task.canonical_retrieval_query}
                )
                observation, _, done, _, info = env.step("RETRIEVE " + retrieve)
                self.assertFalse(done)
                self.assertEqual(
                    info["memory_ops"][0]["retrieved_memory_ids"],
                    ["mem_0000"],
                )
                self.assertIn(task.canonical_memory_value, observation)
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

    def test_reset_clears_policy_authored_memory(self) -> None:
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
            self.assertEqual(len(env.long_term_memory), 1)
            observation, info = env.reset(data_idx=1)
            self.assertEqual(len(env.long_term_memory), 0)
            self.assertEqual(info["ltm_inventory_count"], 0)
            self.assertEqual(env.memory_id_counter, 0)
            self.assertNotIn(task.canonical_memory_value, observation)
            self.assertEqual(len(backend.sessions), 1)
        finally:
            env.close()
        self.assertEqual(backend.sessions, {})

    def test_surface_rejects_visible_inventory_or_broader_retrieval(self) -> None:
        pool = make_negative_fixture_pool()
        generator = NegativeConstraintGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[0]
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=generator,
            split="train",
            task_count=3,
        )
        backend = _FakeNegativeNativeBackend(task)
        with self.assertRaises(ValueError):
            NegativeConstraintWebShopEnv(
                provider=provider,
                backend=backend,
                ltm_inventory_mode="keys",
            )
        with self.assertRaises(ValueError):
            NegativeConstraintWebShopEnv(
                provider=provider,
                backend=backend,
                retrieve_policy="standard",
            )


if __name__ == "__main__":
    unittest.main()
