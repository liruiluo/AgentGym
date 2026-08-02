from __future__ import annotations

import unittest
from dataclasses import replace

from agentenv_agentmemory.latent_preference.schema import (
    SPLITS,
    CertifiedPreferenceProduct,
    PreferenceProductPool,
    PreferenceRecipe,
    canonical_sha256,
    normalize_native_title,
    preference_classification_payload,
)
from agentenv_agentmemory.native_webshop_backend import NativePage, NativePurchase
from agentenv_agentmemory.recency_override import (
    PROVIDER_MODE_RESEEDED_STREAM,
    RecencyOverrideDataError,
    RecencyOverrideGenerator,
    VerifiedRecencyOverrideBundleProvider,
    verify_recency_override_orbit,
)
from agentenv_agentmemory.recency_override_webshop_env import (
    RecencyOverrideWebShopEnv,
)


CATEGORIES = (
    ("area_rug", "area rug"),
    ("phone_case", "phone case"),
    ("pillowcase", "pillowcase"),
    ("window_curtain", "window curtain"),
)
VALUES = (("black", "black"), ("gray", "gray"))


def make_fixture_pool(products_per_cell: int = 2) -> PreferenceProductPool:
    recipe = PreferenceRecipe(
        recipe_id="color.black_gray.home",
        axis="color",
        axis_display_name="color",
        values=tuple(value for value, _ in VALUES),
        value_display_names=tuple(display for _, display in VALUES),
        categories=tuple(category for category, _ in CATEGORIES),
        category_display_names=tuple(display for _, display in CATEGORIES),
    )
    products = []
    ordinal = 1
    for category_id, category_display in CATEGORIES:
        for attribute_value, attribute_display in VALUES:
            for split in SPLITS:
                for cell_index in range(products_per_cell):
                    asin = f"R{ordinal:09d}"
                    title = (
                        f"{attribute_display.title()} {category_display.title()} "
                        f"Native Model {cell_index} {split}"
                    )
                    source_hash = canonical_sha256({"source_candidate": asin})
                    product_category = f"Fixture > {category_display}"
                    category_evidence = (category_display.title(),)
                    title_evidence = (attribute_display.title(),)
                    classification = preference_classification_payload(
                        asin=asin,
                        title=title,
                        product_category=product_category,
                        category_title_evidence=category_evidence,
                        category_id=category_id,
                        axis="color",
                        attribute_value=attribute_value,
                        title_evidence=title_evidence,
                        guard_matches=(attribute_value,),
                        source_candidate_sha256=source_hash,
                    )
                    products.append(
                        CertifiedPreferenceProduct(
                            asin=asin,
                            title=title,
                            native_title_normalized=normalize_native_title(title),
                            price_cents=1_000 + ordinal,
                            product_category=product_category,
                            category_title_evidence=category_evidence,
                            category_id=category_id,
                            category_display_name=category_display,
                            axis="color",
                            attribute_value=attribute_value,
                            attribute_display_name=attribute_display,
                            split=split,
                            search_query=title,
                            search_rank=1,
                            catalog_record_sha256=canonical_sha256(
                                {"asin": asin, "title": title}
                            ),
                            title_evidence=title_evidence,
                            guard_matches=(attribute_value,),
                            classification_sha256=canonical_sha256(classification),
                            source_candidate_sha256=source_hash,
                        )
                    )
                    ordinal += 1
    products.sort(
        key=lambda item: (
            item.axis,
            item.category_id,
            item.attribute_value,
            item.split,
            item.asin,
        )
    )
    return PreferenceProductPool(
        pool_id="fixture_recency_override_pool",
        certifier_version="fixture_certifier_v1",
        products_per_cell=products_per_cell,
        recipes=(recipe,),
        products=tuple(products),
        catalog_sha256="1" * 64,
        attributes_sha256="2" * 64,
        price_table_sha256="3" * 64,
        lucene_index_sha256="4" * 64,
        candidate_artifact_sha256="5" * 64,
        rules_sha256="6" * 64,
        source_manifest_sha256="7" * 64,
    )


class RecencyOverrideGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = RecencyOverrideGenerator(pool=self.pool, seed=233)

    def test_orbit_is_deterministic_and_exhaustively_verified(self) -> None:
        orbit = self.generator.generate_orbit(0, split="train")
        repeated = RecencyOverrideGenerator(pool=self.pool, seed=233).generate_orbit(
            0, split="train"
        )
        changed = RecencyOverrideGenerator(pool=self.pool, seed=234).generate_orbit(
            0, split="train"
        )
        self.assertEqual(orbit.as_dict(), repeated.as_dict())
        self.assertNotEqual(orbit.as_dict(), changed.as_dict())

        proof = verify_recency_override_orbit(
            orbit,
            pool=self.pool,
            expected_generator_version=self.generator.version,
            expected_generator_seed=self.generator.seed,
        )
        self.assertEqual(proof.enumerated_path_count_per_branch, 64)
        self.assertEqual(proof.valid_solution_counts, (1, 1))
        self.assertEqual(proof.application_observation_identity_checks, 3)
        self.assertEqual(proof.application_target_flip_count, 4)
        self.assertEqual(proof.override_transition_checks, 1)
        self.assertTrue(proof.payload()["verification"]["canonical_memory_update_required_for_flip"])

    def test_counterfactual_pair_shares_visible_application_but_flips_target(self) -> None:
        stay, flip = self.generator.generate_orbit(7, split="dev").tasks
        for index, (left, right) in enumerate(zip(stay.phases, flip.phases)):
            self.assertEqual(left.candidates, right.candidates)
            if index < 2:
                self.assertEqual(left.question, right.question)
                self.assertEqual(left.target_asin, right.target_asin)
            elif index == 2:
                self.assertNotEqual(left.question, right.question)
                self.assertNotEqual(left.target_asin, right.target_asin)
            else:
                self.assertEqual(left.question, right.question)
                self.assertEqual(left.confirmed_attribute_value, right.confirmed_attribute_value)
                self.assertNotEqual(left.active_attribute_value, right.active_attribute_value)
                self.assertNotEqual(left.target_asin, right.target_asin)

    def test_reseeded_provider_keeps_pairs_inside_each_epoch(self) -> None:
        provider = VerifiedRecencyOverrideBundleProvider(
            generator=self.generator,
            split="train",
            task_count=2,
            mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        first = provider.get(0)
        next_epoch = provider.get(provider.seed_epoch_orbit_count * 2)
        self.assertNotEqual(first.task_id, next_epoch.task_id)
        self.assertEqual(provider.get(0).orbit_id, provider.get(1).orbit_id)
        self.assertEqual(provider.metadata()["accepted_index_domain"], "all_nonnegative_integers")
        self.assertEqual(
            provider.metadata()["reseeded_stream"]["tasks_per_seed_epoch"],
            provider.seed_epoch_task_count,
        )

    def test_fixed_provider_reports_window_boundary(self) -> None:
        provider = VerifiedRecencyOverrideBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=4,
            start_orbit=3,
        )
        self.assertEqual(
            provider.metadata()["fixed_window"],
            {"start_orbit": 3, "end_orbit_exclusive": 5},
        )


class RecencyOverrideTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = RecencyOverrideGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def _replace_task(self, branch: int, task):
        tasks = list(self.orbit.tasks)
        tasks[branch] = task
        return replace(self.orbit, tasks=tuple(tasks))

    def test_rejects_wrong_declared_target(self) -> None:
        task = self.orbit.tasks[0]
        phase = task.phases[3]
        wrong = next(item.asin for item in phase.candidates if item.asin != phase.target_asin)
        phases = list(task.phases)
        phases[3] = replace(phase, target_asin=wrong)
        with self.assertRaises(RecencyOverrideDataError):
            verify_recency_override_orbit(
                self._replace_task(0, replace(task, phases=tuple(phases))),
                pool=self.pool,
            )

    def test_rejects_noncanonical_question(self) -> None:
        task = self.orbit.tasks[1]
        phases = list(task.phases)
        phases[5] = replace(phases[5], question=phases[5].question + " tampered")
        with self.assertRaisesRegex(RecencyOverrideDataError, "not canonical"):
            verify_recency_override_orbit(
                self._replace_task(1, replace(task, phases=tuple(phases))),
                pool=self.pool,
            )

    def test_rejects_flip_without_canonical_update_contract(self) -> None:
        task = self.orbit.tasks[1]
        with self.assertRaises(RecencyOverrideDataError):
            verify_recency_override_orbit(
                self._replace_task(1, replace(task, override_mode="none")),
                pool=self.pool,
            )


class _FakeRecencyNativeBackend:
    """Small native action surface used to exercise the real recency env."""

    surface = "memoryarena_webshop_native_v1"

    def __init__(self, task) -> None:
        self.task = task
        self.phases = {phase.phase_index: phase for phase in task.phases}
        self.sessions: dict[str, dict[str, object]] = {}
        self.closed_tokens: list[str] = []
        self.prices = {
            candidate.asin: candidate.price_cents
            for phase in task.phases
            for candidate in phase.candidates
        }

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        phase_index = int(session_token.rsplit("_", 1)[-1])
        if phase_index not in self.phases:
            raise AssertionError(f"unexpected phase token: {session_token}")
        if session_token in self.sessions:
            raise AssertionError(f"duplicate session token: {session_token}")
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
                observation=(
                    f"{instruction} [SEP] Results [SEP] "
                    + " [SEP] ".join(handles)
                ),
                url=f"http://fixture/{session_token}/search",
                has_search_bar=True,
                clickables=handles,
            )
        if action.startswith("click[") and action.lower() != "click[buy now]":
            asin = action[6:-1].upper()
            handles = {candidate.asin for candidate in phase.candidates}
            if asin not in handles:
                raise AssertionError(f"click outside current search results: {asin}")
            session["asin"] = asin
            return NativePage(
                observation=(
                    f"{instruction} [SEP] Product {asin} [SEP] Buy Now"
                ),
                url=f"http://fixture/{session_token}/item/{asin}",
                has_search_bar=False,
                clickables=("Buy Now",),
            )
        if action.lower() == "click[buy now]":
            asin = str(session["asin"] or "")
            if not asin:
                raise AssertionError("Buy Now was submitted before opening a product")
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
        raise AssertionError(f"unsupported native fixture action: {action}")

    def close_session(self, session_token: str) -> None:
        self.sessions.pop(session_token, None)
        self.closed_tokens.append(session_token)

    def has_product(self, asin: str) -> bool:
        return asin.upper() in self.prices

    def metadata(self) -> dict[str, object]:
        return {"surface": self.surface}


class RecencyOverrideRuntimeTests(unittest.TestCase):
    def _make_env(self, branch: int):
        pool = make_fixture_pool()
        generator = RecencyOverrideGenerator(pool=pool, seed=233)
        orbit = generator.generate_orbit(0, split="train")
        task = orbit.tasks[branch]
        provider = VerifiedRecencyOverrideBundleProvider(
            generator=generator,
            split="train",
            task_count=2,
        )
        backend = _FakeRecencyNativeBackend(task)
        env = RecencyOverrideWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"recency-{branch}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        observation, info = env.reset(data_idx=branch)
        return env, backend, task, observation, info

    @staticmethod
    def _purchase(env, target_asin: str, expected_index: int):
        observation, reward, done, truncated, info = env.step("search[product]")
        assert not done and not truncated
        assert info["tool_ops"][0]["op"] == "SEARCH"

        observation, reward, done, truncated, info = env.step(
            f"click[{target_asin}]"
        )
        assert not done and not truncated
        assert info["tool_ops"][0]["op"] == "CLICK"

        observation, reward, done, truncated, info = env.step("click[Buy Now]")
        assert not truncated
        assert info["tool_ops"][0]["op"] == "BUY"
        assert info["tool_ops"][0]["purchase_correct"] is True
        # A correct final purchase still advances the completed subtask; the
        # terminal bit distinguishes it from the five intermediate purchases.
        assert info["tool_ops"][0]["session_advanced"] is True
        assert info["tool_ops"][0]["terminal"] is (expected_index == 5)
        assert info["current_subtask_index"] == expected_index + 1
        assert reward == (2.0 if expected_index == 5 else 1.0)
        return observation, reward, done, info

    def test_stay_branch_retrieves_old_value_and_buys_all_phases(self) -> None:
        env, backend, task, observation, info = self._make_env(branch=0)
        try:
            self.assertNotIn(task.target_asins[0], observation)
            self.assertEqual(info["current_subtask_index"], 0)
            self.assertEqual(info["ltm_inventory_count"], 0)

            _, reward, done, _, info = env.step(
                f'ADD {{"key":"user_preference","value":"{task.old_attribute_value}"}}'
            )
            self.assertFalse(done)
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["memory_ops"][0]["op"], "ADD")
            self.assertEqual(
                info["memory_state_diff"]["added"][0]["value"],
                task.old_attribute_value,
            )

            self._purchase(env, task.target_asins[0], expected_index=0)

            _, reward, done, _, info = env.step(
                'RETRIEVE {"memory_id":"mem_0000"}'
            )
            self.assertFalse(done)
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["memory_ops"][0]["op"], "RETRIEVE")
            self.assertEqual(info["memory_ops"][0]["retrieved_memory_ids"], ["mem_0000"])
            self.assertEqual(info["memory_state_diff"], {"added": [], "updated": [], "deleted": []})

            self._purchase(env, task.target_asins[1], expected_index=1)
            for phase_index in range(2, 6):
                _, _, done, info = self._purchase(
                    env,
                    task.target_asins[phase_index],
                    expected_index=phase_index,
                )
            self.assertTrue(done)
            self.assertEqual(info["current_subtask_index"], 6)
            self.assertTrue(info["episode_success"])
            self.assertEqual(backend.sessions, {})
        finally:
            env.close()
        self.assertEqual(backend.sessions, {})

    def test_flip_branch_updates_same_memory_id_before_new_value_buys(self) -> None:
        env, backend, task, _, _ = self._make_env(branch=1)
        try:
            env.step(
                f'ADD {{"key":"user_preference","value":"{task.old_attribute_value}"}}'
            )
            self._purchase(env, task.target_asins[0], expected_index=0)

            _, _, done, _, info = env.step('RETRIEVE {"memory_id":"mem_0000"}')
            self.assertFalse(done)
            self.assertEqual(info["memory_ops"][0]["op"], "RETRIEVE")
            self._purchase(env, task.target_asins[1], expected_index=1)

            _, reward, done, _, info = env.step(
                f'UPDATE {{"memory_id":"mem_0000","value":"{task.new_attribute_value}"}}'
            )
            self.assertFalse(done)
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["memory_ops"][0]["op"], "UPDATE")
            update = info["memory_state_diff"]["updated"]
            self.assertEqual(len(update), 1)
            self.assertEqual(update[0]["before"]["memory_id"], "mem_0000")
            self.assertEqual(update[0]["before"]["value"], task.old_attribute_value)
            self.assertEqual(update[0]["after"]["memory_id"], "mem_0000")
            self.assertEqual(update[0]["after"]["value"], task.new_attribute_value)

            for phase_index in range(2, 6):
                _, _, done, info = self._purchase(
                    env,
                    task.target_asins[phase_index],
                    expected_index=phase_index,
                )
            self.assertTrue(done)
            self.assertEqual(info["current_subtask_index"], 6)
            self.assertTrue(info["episode_success"])
            self.assertEqual(backend.sessions, {})
        finally:
            env.close()

    def test_reset_clears_memory_trace_and_native_session(self) -> None:
        env, backend, task, _, _ = self._make_env(branch=0)
        try:
            env.step(
                f'ADD {{"key":"user_preference","value":"{task.old_attribute_value}"}}'
            )
            self.assertEqual(len(backend.sessions), 1)

            observation, info = env.reset(data_idx=0)
            self.assertEqual(info["current_subtask_index"], 0)
            self.assertEqual(info["ltm_inventory_count"], 0)
            self.assertEqual(info["session_trace"], [])
            self.assertEqual(info["memory_state_diff"], {"added": [], "updated": [], "deleted": []})
            self.assertNotIn("[mem_0000] user_preference:", observation)
            self.assertEqual(len(backend.sessions), 1)
        finally:
            env.close()
        self.assertEqual(backend.sessions, {})


if __name__ == "__main__":
    unittest.main()
