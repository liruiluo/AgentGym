from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from agentenv_agentmemory.latent_preference import (
    CATEGORY_SCHEDULES,
    PROVIDER_MODE_RESEEDED_STREAM,
    CertifiedPreferenceProduct,
    LatentPreferenceDataError,
    LatentPreferenceGenerator,
    PreferenceProductPool,
    PreferenceRecipe,
    VerifiedLatentPreferenceBundleProvider,
    load_preference_product_pool,
    verify_latent_preference_orbit,
    write_preference_product_pool_manifest,
)
from agentenv_agentmemory.latent_preference.schema import (
    SPLITS,
    canonical_sha256,
    normalize_native_title,
    preference_classification_payload,
)
from agentenv_agentmemory.latent_preference.question_format import (
    render_preference_question,
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
                    asin = f"L{ordinal:09d}"
                    title = (
                        f"{attribute_display.title()} {category_display.title()} "
                        f"Native Model {cell_index} {split}"
                    )
                    source_candidate_sha256 = canonical_sha256(
                        {"source_candidate": asin}
                    )
                    title_evidence = (attribute_display.title(),)
                    guard_matches = (attribute_value,)
                    products.append(
                        CertifiedPreferenceProduct(
                            asin=asin,
                            title=title,
                            native_title_normalized=normalize_native_title(title),
                            price_cents=1_000 + ordinal,
                            product_category=f"Fixture > {category_display}",
                            category_title_evidence=(category_display.title(),),
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
                            guard_matches=guard_matches,
                            classification_sha256=canonical_sha256(
                                preference_classification_payload(
                                    asin=asin,
                                    title=title,
                                    product_category=f"Fixture > {category_display}",
                                    category_title_evidence=(
                                        category_display.title(),
                                    ),
                                    category_id=category_id,
                                    axis="color",
                                    attribute_value=attribute_value,
                                    title_evidence=title_evidence,
                                    guard_matches=guard_matches,
                                    source_candidate_sha256=(
                                        source_candidate_sha256
                                    ),
                                )
                            ),
                            source_candidate_sha256=source_candidate_sha256,
                        )
                    )
                    ordinal += 1
    products.sort(
        key=lambda product: (
            product.axis,
            product.category_id,
            product.attribute_value,
            product.split,
            product.asin,
        )
    )
    return PreferenceProductPool(
        pool_id="fixture_latent_preference_pool",
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


class LatentPreferenceGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = LatentPreferenceGenerator(pool=self.pool, seed=233)

    def test_one_two_three_evidence_each_verify_one_hundred_tasks(self) -> None:
        task_counts: Counter[int] = Counter()
        semantic_hashes: set[str] = set()
        orbit_index = 0
        while min((task_counts[count] for count in (1, 2, 3)), default=0) < 100:
            orbit = self.generator.generate_orbit(orbit_index, split="train")
            proof = verify_latent_preference_orbit(
                orbit,
                pool=self.pool,
                expected_generator_version=self.generator.version,
                expected_generator_seed=233,
            )
            evidence_count = orbit.tasks[0].supporting_evidence_count
            task_counts[evidence_count] += 2
            semantic_hashes.update(task.semantic_sha256 for task in orbit.tasks)

            self.assertEqual(proof.enumerated_path_count_per_branch, 64)
            self.assertEqual(proof.valid_solution_counts, (1, 1))
            self.assertEqual(
                proof.hypothesis_counts_after_evidence,
                ((1,) * evidence_count, (1,) * evidence_count),
            )
            self.assertEqual(proof.evidence_target_flip_count, evidence_count)
            self.assertEqual(
                proof.application_target_flip_count,
                6 - evidence_count,
            )
            self.assertEqual(
                proof.application_observation_identity_checks,
                6 - evidence_count,
            )
            self.assertFalse(proof.payload()["verification"]["human_review_required"])
            self.assertFalse(proof.payload()["verification"]["llm_judge_required"])
            orbit_index += 1
            self.assertLess(orbit_index, 1_000)

        self.assertEqual(task_counts, Counter({1: 100, 2: 100, 3: 100}))
        self.assertEqual(len(semantic_hashes), 300)

    def test_counterfactual_pair_keeps_application_visible_state_and_flips_action(self) -> None:
        for orbit_index in range(12):
            orbit = self.generator.generate_orbit(orbit_index, split="dev")
            left, right = orbit.tasks
            evidence_count = left.supporting_evidence_count
            self.assertEqual(left.user_id, right.user_id)
            for phase_index, (left_phase, right_phase) in enumerate(
                zip(left.phases, right.phases)
            ):
                self.assertEqual(left_phase.candidates, right_phase.candidates)
                self.assertNotEqual(left_phase.target_asin, right_phase.target_asin)
                if phase_index < evidence_count:
                    self.assertNotEqual(left_phase.question, right_phase.question)
                else:
                    self.assertEqual(
                        left_phase.as_dict(include_target=False),
                        right_phase.as_dict(include_target=False),
                    )

    def test_question_hides_asins_and_preserves_candidate_order(self) -> None:
        orbit = self.generator.generate_orbit(7, split="test")
        for task in orbit.tasks:
            for phase in task.phases:
                for candidate in phase.candidates:
                    self.assertEqual(phase.question.count(candidate.title), 1)
                    self.assertNotIn(candidate.asin, phase.question)
            self.assertNotIn("other confirmed choice", task.phases[0].question)

    def test_category_schedule_crosses_to_unseen_application_category(self) -> None:
        for orbit_index in range(9):
            task = self.generator.generate_orbit(
                orbit_index, split="train"
            ).tasks[0]
            schedule = CATEGORY_SCHEDULES[task.supporting_evidence_count]
            observed = tuple(phase.category_id for phase in task.phases)
            recipe = self.pool.recipe_by_id(task.recipe_id)
            self.assertEqual(
                observed,
                tuple(recipe.categories[position] for position in schedule),
            )
            evidence_categories = set(
                observed[: task.supporting_evidence_count]
            )
            self.assertNotIn(
                observed[task.supporting_evidence_count],
                evidence_categories,
            )

    def test_generation_is_byte_deterministic_and_seed_changes_stream(self) -> None:
        first = self.generator.generate_orbit(31, split="train")
        repeated = LatentPreferenceGenerator(
            pool=self.pool, seed=233
        ).generate_orbit(31, split="train")
        changed = LatentPreferenceGenerator(
            pool=self.pool, seed=234
        ).generate_orbit(31, split="train")
        self.assertEqual(first.as_dict(), repeated.as_dict())
        self.assertNotEqual(first.as_dict(), changed.as_dict())


class LatentPreferenceVerifierTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = LatentPreferenceGenerator(pool=self.pool, seed=233)
        self.orbit = self.generator.generate_orbit(0, split="train")

    def _replace_task(self, branch: int, task):
        tasks = list(self.orbit.tasks)
        tasks[branch] = task
        return replace(self.orbit, tasks=tuple(tasks))

    def test_rejects_generator_declared_wrong_target(self) -> None:
        task = self.orbit.tasks[0]
        phase = task.phases[0]
        wrong_target = next(
            candidate.asin
            for candidate in phase.candidates
            if candidate.asin != phase.target_asin
        )
        phases = list(task.phases)
        phases[0] = replace(phase, target_asin=wrong_target)
        tampered = self._replace_task(0, replace(task, phases=tuple(phases)))
        with self.assertRaisesRegex(
            LatentPreferenceDataError,
            "declared targets disagree",
        ):
            verify_latent_preference_orbit(tampered, pool=self.pool)

    def test_rejects_hidden_preference_that_disagrees_with_history(self) -> None:
        task = self.orbit.tasks[0]
        other_value = next(
            value
            for value in self.pool.recipes[0].values
            if value != task.preferred_attribute_value
        )
        tampered = self._replace_task(
            0,
            replace(task, preferred_attribute_value=other_value),
        )
        with self.assertRaisesRegex(
            LatentPreferenceDataError,
            "hidden preference disagrees",
        ):
            verify_latent_preference_orbit(tampered, pool=self.pool)

    def test_rejects_noncanonical_question(self) -> None:
        task = self.orbit.tasks[0]
        phases = list(task.phases)
        phases[-1] = replace(
            phases[-1],
            question=phases[-1].question + " Hidden hint.",
        )
        tampered = self._replace_task(0, replace(task, phases=tuple(phases)))
        with self.assertRaisesRegex(
            LatentPreferenceDataError,
            "not canonical visible text",
        ):
            verify_latent_preference_orbit(tampered, pool=self.pool)

    def test_rejects_reused_product_in_repeated_category(self) -> None:
        task = self.orbit.tasks[0]
        phases = list(task.phases)
        repeated_pair = next(
            (left_index, right_index)
            for left_index in range(6)
            for right_index in range(left_index + 1, 6)
            if phases[left_index].category_id == phases[right_index].category_id
        )
        source_index, destination_index = repeated_pair
        source = phases[source_index]
        destination = phases[destination_index]
        replacement_target = next(
            candidate.asin
            for candidate in source.candidates
            if candidate.attribute_value == task.preferred_attribute_value
        )
        phases[destination_index] = replace(
            destination,
            candidates=source.candidates,
            target_asin=replacement_target,
        )
        tampered = self._replace_task(0, replace(task, phases=tuple(phases)))
        with self.assertRaisesRegex(LatentPreferenceDataError, "reuses a product"):
            verify_latent_preference_orbit(tampered, pool=self.pool)

    def test_rejects_budget_shortcut(self) -> None:
        task = self.orbit.tasks[0]
        recipe = self.pool.recipe_by_id(task.recipe_id)
        phases = tuple(
            replace(
                phase,
                question=render_preference_question(
                    user_id=task.user_id,
                    phase_index=phase.phase_index,
                    phase_kind=phase.phase_kind,
                    supporting_evidence_count=task.supporting_evidence_count,
                    recipe=recipe,
                    category_id=phase.category_id,
                    candidates=phase.candidates,
                    budget_cents=1,
                    confirmed_attribute_value=phase.confirmed_attribute_value,
                ),
            )
            for phase in task.phases
        )
        tampered_task = replace(task, budget_cents=1, phases=phases)
        tampered = self._replace_task(0, tampered_task)
        with self.assertRaisesRegex(LatentPreferenceDataError, "budget prunes"):
            verify_latent_preference_orbit(tampered, pool=self.pool)


class LatentPreferenceProviderAndPoolIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_fixture_pool()
        self.generator = LatentPreferenceGenerator(pool=self.pool, seed=233)

    def test_fixed_provider_keeps_counterfactual_pair_together(self) -> None:
        provider = VerifiedLatentPreferenceBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=12,
            start_orbit=3,
            cache_orbits=2,
        )
        left = provider.get(0)
        right = provider.get(1)
        self.assertEqual(left.orbit_id, right.orbit_id)
        self.assertEqual(left.proof_sha256, right.proof_sha256)
        self.assertNotEqual(left.target_asins, right.target_asins)
        with self.assertRaises(IndexError):
            provider.get(12)

    def test_reseeded_training_stream_accepts_next_seed_epoch(self) -> None:
        provider = VerifiedLatentPreferenceBundleProvider(
            generator=self.generator,
            split="train",
            task_count=2,
            mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        initial = provider.get(0)
        next_epoch = provider.get(provider.seed_epoch_task_count)
        self.assertNotEqual(initial.task_id, next_epoch.task_id)
        self.assertEqual(
            provider.metadata()["accepted_index_domain"],
            "all_nonnegative_integers",
        )
        self.assertEqual(
            provider.metadata()["reseeded_stream"],
            {
                "tasks_per_seed_epoch": provider.seed_epoch_task_count,
                "orbits_per_seed_epoch": provider.seed_epoch_orbit_count,
                "counterfactual_pair_never_crosses_seed_epoch": True,
                "seed_epoch_zero_uses_base_seed": True,
                "later_seed_epoch_derivation": "sha256_v1",
                "collision_free_within_complete_seed_epoch": True,
                "semantic_uniqueness_guaranteed_through_task_index": (
                    provider.seed_epoch_task_count - 1
                ),
                "cross_seed_epoch_semantic_uniqueness_guaranteed": False,
            },
        )

    def test_pool_manifest_requires_exact_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pool.json"
            file_sha256 = write_preference_product_pool_manifest(self.pool, path)
            loaded = load_preference_product_pool(
                path,
                expected_file_sha256=file_sha256,
            )
            self.assertEqual(loaded, self.pool)
            with self.assertRaisesRegex(LatentPreferenceDataError, "SHA256 mismatch"):
                load_preference_product_pool(
                    path,
                    expected_file_sha256="0" * 64,
                )
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                file_sha256,
            )


if __name__ == "__main__":
    unittest.main()
