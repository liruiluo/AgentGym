from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from agentenv_agentmemory.native_webshop_backend import NativePage, NativePurchase
from agentenv_agentmemory.procedural import (
    SCENARIOS,
    CertifiedProduct,
    NaturalAttributeChainGenerator,
    NativeCertificationConfig,
    NativeProductPoolCertificationError,
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    ProceduralMemoryDataError,
    ProductPool,
    VerifiedProceduralBundleProvider,
    certify_native_product_pool,
    classify_product_record,
    load_certified_product_pool,
    normalize_native_title,
    require_unique_product_classification,
    scenario_by_id,
    verify_counterfactual_orbit,
)
from agentenv_agentmemory.procedural.pool_io import write_product_pool_manifest
from agentenv_agentmemory.procedural.scenarios import SCENARIO_DEFINITION_SHA256
from agentenv_agentmemory.procedural_webshop_env import ProceduralMemoryWebShopEnv
from agentenv_agentmemory.procedural_wrapper import (
    ProceduralAgentMemoryWrapper,
    attest_procedural_runtime_inputs,
)


SPLITS = ("train", "dev", "test")
HOME_TITLES = {
    "seating": {"leather": "Leather Sofa", "velvet": "Velvet Sofa"},
    "footrest": {"leather": "Leather Ottoman", "velvet": "Velvet Ottoman"},
    "coffee_table": {"wood": "Wood Coffee Table", "glass": "Glass Coffee Table"},
    "side_table": {"wood": "Wood End Table", "glass": "Glass End Table"},
    "rug": {"beige": "Beige Area Rug", "gray": "Gray Area Rug"},
    "curtains": {"linen": "Linen Window Curtains", "silk": "Silk Window Curtains"},
}
HOME_PRODUCT_CATEGORIES = {
    "seating": "Home & Kitchen › Furniture › Living Room Furniture › Sofas & Couches",
    "footrest": "Home & Kitchen › Furniture › Accent Furniture › Ottomans",
    "coffee_table": (
        "Home & Kitchen › Furniture › Living Room Furniture › Tables › Coffee Tables"
    ),
    "side_table": (
        "Home & Kitchen › Furniture › Living Room Furniture › Tables › End Tables"
    ),
    "rug": "Home & Kitchen › Home Décor Products › Rugs, Pads & Protectors › Area Rugs",
    "curtains": (
        "Home & Kitchen › Home Décor Products › Window Treatments › "
        "Curtains & Drapes › Panels"
    ),
}


def _record(title: str, *, product_category: str) -> dict[str, str]:
    return {
        "Title": title,
        "category": "garden",
        "query": "fixture home product",
        "product_category": product_category,
    }


def _record_hash(record: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def make_fixture_pool(
    products_per_cell: int = 2,
    *,
    catalog_sha256: str = "1" * 64,
    attributes_sha256: str = "2" * 64,
    price_table_sha256: str = "3" * 64,
    lucene_index_sha256: str = "4" * 64,
) -> tuple[ProductPool, dict[str, dict[str, str]], dict[str, int]]:
    products = []
    records: dict[str, dict[str, str]] = {}
    prices: dict[str, int] = {}
    ordinal = 1
    for slot in scenario_by_id("home").slots:
        for value in slot.values:
            for split in SPLITS:
                for cell_index in range(products_per_cell):
                    asin = f"H{ordinal:09d}"
                    title = (
                        f"{HOME_TITLES[slot.slot_id][value.value_id]} "
                        f"Model {cell_index} {split}"
                    )
                    record = _record(
                        title,
                        product_category=HOME_PRODUCT_CATEGORIES[slot.slot_id],
                    )
                    classification = require_unique_product_classification(
                        record,
                        scenario_ids=("home",),
                    )
                    price = 1_000 + ordinal
                    products.append(
                        CertifiedProduct.from_classification(
                            classification=classification,
                            asin=asin,
                            title=title,
                            split=split,
                            price_cents=price,
                            search_query=title,
                            search_rank=1,
                            catalog_record_sha256=_record_hash(record),
                            native_title_catalog_match_count=1,
                            native_title_globally_unique=True,
                        )
                    )
                    records[asin] = record
                    prices[asin] = price
                    ordinal += 1
    return (
        ProductPool(
            pool_id="fixture_natural_pool",
            certifier_version="fixture_certifier_v1",
            scenario_ids=("home",),
            products_per_cell=products_per_cell,
            products=tuple(products),
            catalog_sha256=catalog_sha256,
            attributes_sha256=attributes_sha256,
            price_table_sha256=price_table_sha256,
            lucene_index_sha256=lucene_index_sha256,
            source_manifest_sha256="5" * 64,
        ),
        records,
        prices,
    )


class FakeNativeBackend:
    surface = "fixture_native"

    def __init__(
        self,
        records: dict[str, dict[str, str]],
        prices: dict[str, int],
        *,
        price_table_sha256: str = "3" * 64,
    ) -> None:
        self.records = {asin: dict(record) for asin, record in records.items()}
        self.prices = dict(prices)
        self.price_table_sha256 = price_table_sha256
        self.sessions: dict[str, dict[str, str | None]] = {}
        self.actions: list[tuple[str, str]] = []

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        self.sessions[session_token] = {"asin": None, "instruction": instruction}
        return NativePage(
            observation=instruction,
            url="fixture://search",
            has_search_bar=True,
            clickables=(),
        )

    def step(self, session_token: str, action: str) -> NativePage:
        self.actions.append((session_token, action))
        session = self.sessions[session_token]
        if action.startswith("search["):
            title = action[len("search[") : -1]
            matches = tuple(
                asin
                for asin, record in self.records.items()
                if record["Title"] == title
            )
            return NativePage(
                observation="Search results",
                url="fixture://results",
                has_search_bar=True,
                clickables=matches,
            )
        if action.startswith("click["):
            value = action[len("click[") : -1]
            if value == "Buy Now":
                asin = str(session["asin"])
                return NativePage(
                    observation="Purchase complete",
                    url="fixture://done",
                    has_search_bar=False,
                    clickables=(),
                    purchase=NativePurchase(
                        asin=asin,
                        price_cents=self.prices[asin],
                        selected_options={},
                    ),
                )
            if value not in self.records:
                raise KeyError(value)
            session["asin"] = value
            return NativePage(
                observation=self.records[value]["Title"],
                url=f"fixture://item/{value}",
                has_search_bar=True,
                clickables=("Buy Now",),
            )
        raise ValueError(action)

    def close_session(self, session_token: str) -> None:
        self.sessions.pop(session_token, None)

    def close(self) -> None:
        self.sessions.clear()

    def has_product(self, asin: str) -> bool:
        return asin in self.records

    def product_asins(self):
        return self.records

    def product_title(self, asin: str) -> str:
        return self.records[asin]["Title"]

    def product_record(self, asin: str) -> dict[str, str]:
        return dict(self.records[asin])

    def product_price_cents(self, asin: str) -> int:
        return self.prices[asin]

    def product_record_sha256(self, asin: str) -> str:
        return _record_hash(self.records[asin])

    def metadata(self):
        return {
            "surface": self.surface,
            "price_seed": 233,
            "product_count": len(self.records),
            "price_table_sha256": self.price_table_sha256,
            "upstream_provenance": {"fixture": True},
        }


class SubstringSearchBackend(FakeNativeBackend):
    """Fixture backend whose native search accepts title-derived phrases."""

    def step(self, session_token: str, action: str) -> NativePage:
        if not action.startswith("search["):
            return super().step(session_token, action)
        self.actions.append((session_token, action))
        query = action[len("search[") : -1]
        normalized_query = normalize_native_title(query)
        matches = tuple(
            asin
            for asin, record in self.records.items()
            if normalized_query in normalize_native_title(record["Title"])
        )
        return NativePage(
            observation="Search results",
            url="fixture://results",
            has_search_bar=True,
            clickables=matches,
        )


class RejectFirstCandidatePerBaseCellBackend(FakeNativeBackend):
    """Reject every query for the first native probe in each attribute cell."""

    def __init__(
        self,
        records: dict[str, dict[str, str]],
        prices: dict[str, int],
    ) -> None:
        super().__init__(records, prices)
        self._seen_base_cells: set[tuple[str, str, str]] = set()
        self._blocked_sessions: set[str] = set()

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        asin = session_token.rsplit("-", 1)[-1]
        classification = require_unique_product_classification(
            self.records[asin],
            scenario_ids=("home",),
        )
        base_cell = (
            classification.scenario_id,
            classification.slot_id,
            classification.attribute_value,
        )
        if base_cell not in self._seen_base_cells:
            self._seen_base_cells.add(base_cell)
            self._blocked_sessions.add(session_token)
        return super().open_session(session_token, instruction)

    def step(self, session_token: str, action: str) -> NativePage:
        if session_token in self._blocked_sessions and action.startswith("search["):
            self.actions.append((session_token, action))
            return NativePage(
                observation="Search results",
                url="fixture://results",
                has_search_bar=True,
                clickables=(),
            )
        return super().step(session_token, action)

    def close_session(self, session_token: str) -> None:
        self._blocked_sessions.discard(session_token)
        super().close_session(session_token)


class ScenarioDefinitionTests(unittest.TestCase):
    def test_memoryarena_five_scenarios_have_six_binary_natural_slots(self) -> None:
        self.assertEqual(
            [scenario.scenario_id for scenario in SCENARIOS],
            ["baking", "beauty", "electronics", "grocery", "home"],
        )
        self.assertEqual(sum(len(scenario.slots) for scenario in SCENARIOS), 30)
        self.assertTrue(all(len(scenario.slots) == 6 for scenario in SCENARIOS))
        self.assertTrue(
            all(
                len(slot.values) == 2
                for scenario in SCENARIOS
                for slot in scenario.slots
            )
        )
        self.assertRegex(SCENARIO_DEFINITION_SHA256, r"^[0-9a-f]{64}$")

    def test_classifier_requires_real_category_and_one_unambiguous_value(self) -> None:
        seating_category = HOME_PRODUCT_CATEGORIES["seating"]
        match = require_unique_product_classification(
            _record("Velvet Loveseat Sofa", product_category=seating_category),
            scenario_ids=("home",),
        )
        self.assertEqual((match.slot_id, match.attribute_value), ("seating", "velvet"))
        self.assertEqual(
            classify_product_record(
                {
                    **_record(
                        "Velvet Loveseat Sofa",
                        product_category=seating_category,
                    ),
                    "category": "electronics",
                },
                scenario_ids=("home",),
            ),
            (),
        )
        self.assertEqual(
            classify_product_record(
                _record(
                    "Leather and Velvet Loveseat Sofa",
                    product_category=seating_category,
                ),
                scenario_ids=("home",),
            ),
            (),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            require_unique_product_classification(
                _record(
                    "Leather and Velvet Loveseat Sofa",
                    product_category=seating_category,
                ),
                scenario_ids=("home",),
            )

    def test_product_category_gate_rejects_title_keyword_false_positives(self) -> None:
        false_positives = (
            {
                "Title": "4K Projector Compatible with TV Stick",
                "category": "electronics",
                "query": "4k tv",
                "product_category": "Electronics › Video Projectors",
            },
            {
                "Title": "Thermal Label Printer Compatible with UPS Shipping",
                "category": "electronics",
                "query": "ups printer",
                "product_category": (
                    "Office Products › Office Electronics › Printers & Accessories "
                    "› Printers › Label Printers"
                ),
            },
            {
                "Title": "Leather Sofa Throw Pillow Cover",
                "category": "garden",
                "query": "leather sofa",
                "product_category": (
                    "Home & Kitchen › Bedding › Decorative Pillows, Inserts & Covers "
                    "› Throw Pillow Covers"
                ),
            },
            {
                "Title": "Replacement Remote for 4K Blu-ray Player",
                "category": "electronics",
                "query": "blu ray player",
                "product_category": (
                    "Electronics › Accessories & Supplies › Audio & Video Accessories "
                    "› Remote Controls & Accessories › Remote Controls"
                ),
            },
        )
        for record in false_positives:
            with self.subTest(title=record["Title"]):
                self.assertEqual(classify_product_record(record), ())


class ProceduralNaturalChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool, cls.records, cls.prices = make_fixture_pool()
        cls.generator = NaturalAttributeChainGenerator(pool=cls.pool, seed=59)

    def test_generation_is_deterministic_and_semantically_nonrepeating(self) -> None:
        first = self.generator.generate_orbit(123, split="train")
        repeated = self.generator.generate_orbit(123, split="train")
        self.assertEqual(first.as_dict(), repeated.as_dict())
        fingerprints = {
            self.generator.generate_orbit(index, split="train").semantic_sha256
            for index in range(200)
        }
        self.assertEqual(len(fingerprints), 200)
        self.assertEqual(self.generator.semantic_period_orbits, 8_388_608)
        self.assertEqual(self.generator.semantic_period_tasks, 16_777_216)
        self.assertEqual(
            5 * (16**6) * (2**5) * 2,
            5_368_709_120,
        )

    def test_pair_keeps_later_observations_and_flips_all_answers(self) -> None:
        orbit = self.generator.generate_orbit(7, split="train")
        left, right = orbit.tasks
        self.assertNotEqual(left.phases[0].question, right.phases[0].question)
        self.assertEqual(left.questions[1:], right.questions[1:])
        self.assertEqual(
            tuple(phase.candidates for phase in left.phases),
            tuple(phase.candidates for phase in right.phases),
        )
        self.assertTrue(all(len(phase.candidates) == 2 for phase in left.phases))
        self.assertTrue(
            all(a != b for a, b in zip(left.target_asins, right.target_asins))
        )
        combined = "\n".join(left.questions)
        for synthetic_term in (
            "pairing number",
            "receive_code",
            "next_code",
            "mix-",
        ):
            self.assertNotIn(synthetic_term, combined)
        self.assertIn("certified natural attribute", combined)
        for phase in left.phases:
            self.assertIn(
                "Only those two exact listings are eligible for this order",
                phase.question,
            )
            self.assertIn("Customer-approved product cards", phase.question)
            for candidate in phase.candidates:
                self.assertEqual(phase.question.count(candidate.title), 1)
                self.assertNotIn(candidate.asin, phase.question)

    def test_each_later_answer_uses_only_previous_selected_attribute(self) -> None:
        orbit = self.generator.generate_orbit(71, split="train")
        for task in orbit.tasks:
            for phase_index in range(1, 6):
                previous_value = task.phases[phase_index - 1].target_attribute_value
                transition = task.phases[phase_index].transition
                self.assertIsNotNone(transition)
                expected = transition.resolve(previous_value)  # type: ignore[union-attr]
                self.assertEqual(
                    expected,
                    task.phases[phase_index].target_attribute_value,
                )

    def test_exhaustive_verifier_proves_one_of_64_without_budget_shortcut(self) -> None:
        orbit = self.generator.generate_orbit(9, split="train")
        proof = verify_counterfactual_orbit(
            orbit,
            pool=self.pool,
            expected_generator_version=self.generator.version,
            expected_generator_seed=self.generator.seed,
        )
        self.assertEqual(proof.enumerated_path_count_per_branch, 64)
        self.assertEqual(proof.valid_solution_counts, (1, 1))
        self.assertEqual(proof.counterfactual_target_flip_count, 6)
        self.assertEqual(proof.complete_bijection_count, 5)
        self.assertEqual(proof.later_observation_identity_checks, 5)
        self.assertLess(proof.max_path_cost_cents, proof.budget_cents)
        self.assertEqual(proof.as_dict()["budget"]["paths_pruned"], 0)
        verification = proof.as_dict()["verification"]
        self.assertEqual(
            verification["answer_domain"],
            "current_phase_approved_titles_resolved_by_hidden_asin_receipt",
        )
        self.assertTrue(verification["approved_candidate_titles_in_task_prompt"])
        self.assertFalse(verification["approved_candidate_asins_in_task_prompt"])
        self.assertFalse(verification["target_asin_in_task_prompt"])
        self.assertTrue(verification["native_search_result_asin_handles_visible"])
        self.assertTrue(verification["native_click_action_uses_asin_handle"])
        self.assertTrue(verification["native_purchase_receipt_asin_verification"])
        self.assertFalse(verification["out_of_shortlist_purchase_is_legal"])
        self.assertFalse(verification["global_catalog_attribute_uniqueness_required"])
        self.assertFalse(verification["global_catalog_attribute_uniqueness_claimed"])
        self.assertTrue(
            verification["global_catalog_normalized_title_uniqueness_required"]
        )
        self.assertTrue(
            verification["global_catalog_normalized_title_uniqueness_claimed"]
        )
        self.assertFalse(proof.as_dict()["verification"]["human_review_required"])
        self.assertFalse(proof.as_dict()["verification"]["llm_judge_required"])

    def test_verifier_rejects_corrupt_declared_target(self) -> None:
        orbit = self.generator.generate_orbit(10, split="train")
        task = orbit.tasks[0]
        phase = task.phases[3]
        wrong = next(
            candidate.asin
            for candidate in phase.candidates
            if candidate.asin != phase.target_asin
        )
        phases = list(task.phases)
        phases[3] = replace(phase, target_asin=wrong)
        tasks = (replace(task, phases=tuple(phases)), orbit.tasks[1])
        corrupted = replace(orbit, tasks=tasks)
        with self.assertRaisesRegex(
            ProceduralMemoryDataError,
            "independently enumerated solution",
        ):
            verify_counterfactual_orbit(corrupted, pool=self.pool)

    def test_verifier_rejects_noncanonical_or_pair_specific_text(self) -> None:
        orbit = self.generator.generate_orbit(11, split="train")
        task = orbit.tasks[0]
        phases = list(task.phases)
        phases[2] = replace(phases[2], question=phases[2].question + "\nleaked branch")
        corrupted = replace(
            orbit,
            tasks=(replace(task, phases=tuple(phases)), orbit.tasks[1]),
        )
        with self.assertRaisesRegex(ProceduralMemoryDataError, "canonical visible text"):
            verify_counterfactual_orbit(corrupted, pool=self.pool)

    def test_verifier_rejects_budget_pruning(self) -> None:
        orbit = self.generator.generate_orbit(12, split="train")
        low_budget = max(
            sum(phase.candidates[choice].price_cents for phase in task.phases)
            for task in orbit.tasks
            for choice in (0, 1)
        )
        tasks = tuple(replace(task, budget_cents=low_budget) for task in orbit.tasks)
        corrupted = replace(orbit, tasks=tasks)
        with self.assertRaisesRegex(ProceduralMemoryDataError, "canonical visible text"):
            verify_counterfactual_orbit(corrupted, pool=self.pool)

    def test_product_splits_are_asin_disjoint(self) -> None:
        memberships = {
            split: {product.asin for product in self.pool.products if product.split == split}
            for split in SPLITS
        }
        self.assertFalse(memberships["train"] & memberships["dev"])
        self.assertFalse(memberships["train"] & memberships["test"])
        self.assertFalse(memberships["dev"] & memberships["test"])

    def test_provider_has_no_modulo_wrap_and_uses_bounded_cache(self) -> None:
        provider = VerifiedProceduralBundleProvider(
            generator=self.generator,
            split="train",
            task_count=20,
            cache_orbits=2,
        )
        self.assertNotEqual(provider.get(0).task_id, provider.get(2).task_id)
        with self.assertRaises(IndexError):
            provider.get(20)
        for index in range(0, 20, 2):
            provider.get(index)
        self.assertLessEqual(len(provider._cache), 2)
        metadata = provider.metadata()
        self.assertEqual(metadata["memory_dependency"], "previous_purchased_natural_attribute")

    def test_reseeded_stream_exhausts_full_period_before_next_seed_epoch(
        self,
    ) -> None:
        provider = VerifiedProceduralBundleProvider(
            generator=self.generator,
            split="train",
            task_count=8,
            mode=PROVIDER_MODE_RESEEDED_STREAM,
            cache_orbits=2,
        )

        epoch_task_count = self.generator.semantic_period_orbits * 2
        last_pair_in_epoch_zero = (
            provider.get(epoch_task_count - 2),
            provider.get(epoch_task_count - 1),
        )
        first_pair_in_epoch_one = (
            provider.get(epoch_task_count),
            provider.get(epoch_task_count + 1),
        )
        self.assertEqual(
            last_pair_in_epoch_zero[0].orbit_id,
            last_pair_in_epoch_zero[1].orbit_id,
        )
        self.assertEqual(
            first_pair_in_epoch_one[0].orbit_id,
            first_pair_in_epoch_one[1].orbit_id,
        )
        self.assertNotEqual(
            last_pair_in_epoch_zero[0].orbit_id,
            first_pair_in_epoch_one[0].orbit_id,
        )
        self.assertEqual(provider.get(epoch_task_count), first_pair_in_epoch_one[0])
        self.assertEqual(
            provider.proof_for_index(epoch_task_count - 2).generator_seed,
            self.generator.seed,
        )
        self.assertEqual(
            provider.proof_for_index(epoch_task_count - 2).proof_sha256,
            provider.proof_for_index(epoch_task_count - 1).proof_sha256,
        )
        self.assertEqual(
            provider.proof_for_index(epoch_task_count).proof_sha256,
            provider.proof_for_index(epoch_task_count + 1).proof_sha256,
        )
        self.assertNotEqual(
            provider.proof_for_index(epoch_task_count).generator_seed,
            self.generator.seed,
        )

        # task_count is the DataLoader epoch window, not a cap on the stream.
        far_index = 1_000_000
        self.assertEqual(provider.get(far_index), provider.get(far_index))
        metadata = provider.metadata()
        self.assertEqual(metadata["accepted_index_domain"], "all_nonnegative_integers")
        self.assertTrue(
            metadata["reseeded_stream"][
                "counterfactual_pair_never_crosses_seed_epoch"
            ]
        )
        self.assertEqual(
            metadata["reseeded_stream"]["tasks_per_seed_epoch"],
            epoch_task_count,
        )

    def test_stream_is_train_only_and_fixed_windows_are_strictly_bounded(self) -> None:
        for split in ("dev", "test"):
            with self.subTest(split=split), self.assertRaisesRegex(
                ProceduralMemoryDataError,
                "training-only",
            ):
                VerifiedProceduralBundleProvider(
                    generator=self.generator,
                    split=split,
                    task_count=2,
                    mode=PROVIDER_MODE_RESEEDED_STREAM,
                )

        provider = VerifiedProceduralBundleProvider(
            generator=self.generator,
            split="dev",
            task_count=2,
            mode=PROVIDER_MODE_FIXED_WINDOW,
            start_orbit=3,
        )
        expected = self.generator.generate_orbit(3, split="dev")
        self.assertEqual(provider.get(0).task_id, expected.tasks[0].task_id)
        self.assertEqual(provider.get(1).task_id, expected.tasks[1].task_id)
        for invalid_index in (-1, 2, True, 1.5):
            with self.subTest(index=invalid_index), self.assertRaises(IndexError):
                provider.get(invalid_index)  # type: ignore[arg-type]


class ProductPoolAndRuntimeTests(unittest.TestCase):
    def test_wrapper_creation_does_not_use_transport_id_as_fixed_window_index(
        self,
    ) -> None:
        pool, records, prices = make_fixture_pool(1)
        wrapper = ProceduralAgentMemoryWrapper.__new__(ProceduralAgentMemoryWrapper)
        wrapper.reward_contract = {
            "first_valid_add_reward": 0.0,
            "first_valid_later_session_retrieve_reward": 0.0,
        }
        wrapper.ltm_inventory_mode = "hidden"
        wrapper.ltm_transition_notice_mode = "none"
        wrapper.action_listing_mode = "separate"
        wrapper.provider = VerifiedProceduralBundleProvider(
            generator=NaturalAttributeChainGenerator(pool=pool, seed=233),
            split="dev",
            task_count=2,
            mode=PROVIDER_MODE_FIXED_WINDOW,
        )
        wrapper.backend = FakeNativeBackend(records, prices)
        wrapper.max_id = 0
        wrapper.envs = {}
        wrapper.info = {}
        wrapper.env_locks = {}
        wrapper.lock = threading.RLock()

        try:
            for expected_env_id in range(3):
                created = wrapper.create()
                self.assertEqual(created["id"], expected_env_id)
                self.assertEqual(wrapper.envs[expected_env_id].data_idx, 0)
                wrapper.close(expected_env_id)
        finally:
            wrapper.backend.close()

    def test_pool_round_trip_is_hash_pinned(self) -> None:
        pool, _, _ = make_fixture_pool(1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pool.json"
            digest = write_product_pool_manifest(pool, path)
            loaded = load_certified_product_pool(path, expected_file_sha256=digest)
            self.assertEqual(pool.semantic_manifest(), loaded.semantic_manifest())
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ProceduralMemoryDataError, "SHA256 mismatch"):
                load_certified_product_pool(path, expected_file_sha256=digest)

    def test_native_certifier_fills_all_home_attribute_split_cells(self) -> None:
        records: dict[str, dict[str, str]] = {}
        prices: dict[str, int] = {}
        ordinal = 1
        for slot in scenario_by_id("home").slots:
            for value in slot.values:
                for candidate_index in range(6):
                    asin = f"C{ordinal:09d}"
                    title = (
                        f"{HOME_TITLES[slot.slot_id][value.value_id]} "
                        f"Certified Candidate {candidate_index}"
                    )
                    records[asin] = _record(
                        title,
                        product_category=HOME_PRODUCT_CATEGORIES[slot.slot_id],
                    )
                    prices[asin] = 2_000 + ordinal
                    ordinal += 1
        backend = FakeNativeBackend(records, prices)
        pool, audit = certify_native_product_pool(
            backend,
            catalog_sha256="a" * 64,
            attributes_sha256="b" * 64,
            lucene_index_sha256="c" * 64,
            config=NativeCertificationConfig(
                pool_id="fixture_certified_pool",
                scenario_ids=("home",),
                products_per_cell=1,
                probe_cap_per_cell_split=2,
            ),
        )
        self.assertEqual(len(pool.products), 6 * 2 * 3)
        self.assertTrue(audit["verification"]["category_and_attribute_from_native_record"])
        self.assertFalse(
            audit["verification"]["approved_shortlist_asins_in_task_prompt"]
        )
        self.assertTrue(
            audit["verification"]["native_search_result_asin_handles_visible"]
        )
        self.assertTrue(
            audit["verification"]["native_click_action_uses_asin_handle"]
        )
        self.assertTrue(audit["verification"]["native_title_globally_unique"])
        self.assertEqual(
            audit["counts"]["catalog_title_identity_records_scanned"],
            len(records),
        )
        self.assertFalse(audit["verification"]["human_review_required"])
        self.assertTrue(any(action.startswith("search[") for _, action in backend.actions))
        self.assertTrue(any(action == "click[Buy Now]" for _, action in backend.actions))

    def test_native_certifier_rejects_catalog_wide_duplicate_titles(self) -> None:
        records: dict[str, dict[str, str]] = {}
        prices: dict[str, int] = {}
        ordinal = 1
        duplicate_title = ""
        for slot in scenario_by_id("home").slots:
            for value in slot.values:
                for candidate_index in range(7):
                    asin = f"U{ordinal:09d}"
                    title = (
                        f"{HOME_TITLES[slot.slot_id][value.value_id]} "
                        f"Unique Candidate {candidate_index}"
                    )
                    if slot.slot_id == "seating" and value.value_id == "leather":
                        if candidate_index == 0:
                            duplicate_title = title
                        elif candidate_index == 1:
                            title = duplicate_title.swapcase()
                    records[asin] = _record(
                        title,
                        product_category=HOME_PRODUCT_CATEGORIES[slot.slot_id],
                    )
                    prices[asin] = 3_000 + ordinal
                    ordinal += 1

        pool, audit = certify_native_product_pool(
            FakeNativeBackend(records, prices),
            catalog_sha256="6" * 64,
            attributes_sha256="7" * 64,
            lucene_index_sha256="8" * 64,
            config=NativeCertificationConfig(
                pool_id="fixture_unique_title_pool",
                scenario_ids=("home",),
                products_per_cell=1,
                probe_cap_per_cell_split=2,
            ),
        )
        selected_titles = {product.title.casefold() for product in pool.products}
        self.assertNotIn(duplicate_title.casefold(), selected_titles)
        self.assertEqual(audit["counts"]["duplicate_normalized_title_groups"], 1)
        self.assertEqual(
            audit["counts"]["rejected_nonunique_normalized_title_candidates"],
            2,
        )
        self.assertTrue(
            all(product.native_title_globally_unique for product in pool.products)
        )
        self.assertTrue(
            all(
                product.native_title_catalog_match_count == 1
                for product in pool.products
            )
        )

    def test_native_certifier_audits_successes_before_balanced_split_assignment(
        self,
    ) -> None:
        records: dict[str, dict[str, str]] = {}
        prices: dict[str, int] = {}
        ordinal = 1
        for slot in scenario_by_id("home").slots:
            for value in slot.values:
                for candidate_index in range(4):
                    asin = f"B{ordinal:09d}"
                    title = (
                        f"{HOME_TITLES[slot.slot_id][value.value_id]} "
                        f"Balanced Candidate {candidate_index}"
                    )
                    records[asin] = _record(
                        title,
                        product_category=HOME_PRODUCT_CATEGORIES[slot.slot_id],
                    )
                    prices[asin] = 4_000 + ordinal
                    ordinal += 1

        pool, audit = certify_native_product_pool(
            RejectFirstCandidatePerBaseCellBackend(records, prices),
            catalog_sha256="9" * 64,
            attributes_sha256="a" * 64,
            lucene_index_sha256="b" * 64,
            config=NativeCertificationConfig(
                pool_id="fixture_post_audit_split_pool",
                scenario_ids=("home",),
                products_per_cell=1,
                probe_cap_per_cell_split=2,
            ),
        )

        self.assertEqual(len(pool.products), 6 * 2 * 3)
        self.assertTrue(audit["verification"]["native_audit_before_split_assignment"])
        self.assertEqual(
            audit["counts"]["rejections"],
            {"target_absent_from_all_title_derived_first_pages": 12},
        )
        self.assertTrue(
            all(
                cell["certified"] == 1
                for cell in audit["counts"]["per_split_cell"].values()
            )
        )

    def test_native_certifier_derives_safe_query_from_long_bracketed_title(self) -> None:
        records: dict[str, dict[str, str]] = {}
        prices: dict[str, int] = {}
        ordinal = 1
        long_suffix = (
            " - Artisan Home Collection with Reinforced Everyday Construction "
            "and Coordinated Decorative Details for Seasonal Room Styling "
            "[Gift Ready Retail Package]"
        )
        for slot in scenario_by_id("home").slots:
            for value in slot.values:
                for candidate_index in range(3):
                    asin = f"L{ordinal:09d}"
                    title = (
                        f"{HOME_TITLES[slot.slot_id][value.value_id]} "
                        f"Natural Candidate {candidate_index}{long_suffix}"
                    )
                    self.assertGreater(len(title), 160)
                    self.assertLessEqual(len(title), 240)
                    records[asin] = _record(
                        title,
                        product_category=HOME_PRODUCT_CATEGORIES[slot.slot_id],
                    )
                    prices[asin] = 5_000 + ordinal
                    ordinal += 1

        pool, audit = certify_native_product_pool(
            SubstringSearchBackend(records, prices),
            catalog_sha256="c" * 64,
            attributes_sha256="d" * 64,
            lucene_index_sha256="e" * 64,
            config=NativeCertificationConfig(
                pool_id="fixture_title_derived_query_pool",
                scenario_ids=("home",),
                products_per_cell=1,
                probe_cap_per_cell_split=1,
            ),
        )

        self.assertEqual(len(pool.products), 6 * 2 * 3)
        self.assertTrue(audit["verification"]["native_search_query_title_derived"])
        for product in pool.products:
            self.assertNotEqual(product.search_query, product.title)
            self.assertNotIn("[", product.search_query)
            self.assertNotIn("]", product.search_query)
            self.assertIn(
                normalize_native_title(product.search_query),
                normalize_native_title(product.title),
            )
            self.assertTrue(
                any(
                    normalize_native_title(evidence)
                    in normalize_native_title(product.search_query)
                    for evidence in product.attribute_title_evidence
                )
            )

    def test_native_certifier_failure_audit_keeps_candidate_query_evidence(
        self,
    ) -> None:
        records: dict[str, dict[str, str]] = {}
        prices: dict[str, int] = {}
        ordinal = 1
        for slot in scenario_by_id("home").slots:
            for value in slot.values:
                for candidate_index in range(3):
                    asin = f"F{ordinal:09d}"
                    title = (
                        f"{HOME_TITLES[slot.slot_id][value.value_id]} "
                        f"Failure Audit Candidate {candidate_index}"
                    )
                    records[asin] = _record(
                        title,
                        product_category=HOME_PRODUCT_CATEGORIES[slot.slot_id],
                    )
                    prices[asin] = 6_000 + ordinal
                    ordinal += 1

        with self.assertRaises(NativeProductPoolCertificationError) as raised:
            certify_native_product_pool(
                RejectFirstCandidatePerBaseCellBackend(records, prices),
                catalog_sha256="f" * 64,
                attributes_sha256="0" * 64,
                lucene_index_sha256="1" * 64,
                config=NativeCertificationConfig(
                    pool_id="fixture_failed_audit_pool",
                    scenario_ids=("home",),
                    products_per_cell=1,
                    probe_cap_per_cell_split=1,
                ),
            )

        audit = raised.exception.audit
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(len(audit["counts"]["missing_split_cells"]), 12)
        rejected = [
            probe for probe in audit["candidate_probes"] if probe["status"] == "rejected"
        ]
        self.assertEqual(len(rejected), 12)
        for probe in rejected:
            self.assertRegex(probe["asin"], r"^[A-Z0-9]{10}$")
            self.assertTrue(probe["title"])
            self.assertEqual(
                probe["rejection_reason"],
                "target_absent_from_all_title_derived_first_pages",
            )
            self.assertTrue(probe["search_attempts"])
            for attempt in probe["search_attempts"]:
                self.assertTrue(attempt["query"])
                self.assertEqual(attempt["result_asins"], [])
                self.assertIsNone(attempt["target_rank"])
                self.assertFalse(attempt["within_limit"])

    def test_runtime_attestation_rechecks_record_classification_and_all_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = root / "items.json"
            attributes = root / "attributes.json"
            items.write_bytes(b"catalog\n")
            attributes.write_bytes(b"attributes\n")
            search_root = root / "search"
            index = search_root / "indexes-full" / "segment"
            index.parent.mkdir(parents=True)
            index.write_bytes(b"lucene\n")
            manifest = root / "lucene.sha256"
            manifest.write_text(
                f"{hashlib.sha256(index.read_bytes()).hexdigest()}  segment\n"
            )
            pool, records, prices = make_fixture_pool(
                1,
                catalog_sha256=hashlib.sha256(items.read_bytes()).hexdigest(),
                attributes_sha256=hashlib.sha256(attributes.read_bytes()).hexdigest(),
                price_table_sha256="d" * 64,
                lucene_index_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            backend = FakeNativeBackend(
                records,
                prices,
                price_table_sha256="d" * 64,
            )
            attest_procedural_runtime_inputs(
                pool,
                backend,
                items_file=items,
                attributes_file=attributes,
                search_root=search_root,
                lucene_manifest=manifest,
            )
            first = pool.products[0]
            backend.records[first.asin]["Title"] = "Unclassified fixture title"
            with self.assertRaisesRegex(RuntimeError, "title"):
                attest_procedural_runtime_inputs(
                    pool,
                    backend,
                    items_file=items,
                    attributes_file=attributes,
                    search_root=search_root,
                    lucene_manifest=manifest,
                )

    def test_buy_info_keeps_formal_evidence_without_visible_answer_feedback(self) -> None:
        pool, records, prices = make_fixture_pool(1)
        generator = NaturalAttributeChainGenerator(pool=pool, seed=233)
        provider = VerifiedProceduralBundleProvider(
            generator=generator,
            split="train",
            task_count=2,
        )
        backend = FakeNativeBackend(records, prices)
        env = ProceduralMemoryWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid="procedural-test",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        try:
            _, reset_info = env.reset(data_idx=0)
            forbidden = {
                "task_id",
                "purchase_history",
                "root_attribute_value",
                "target_asin",
                "proof_sha256",
                "product_pool_sha256",
                "orbit_id",
            }
            self.assertFalse(forbidden & set(reset_info))
            bundle = provider.get(0)
            target = bundle.target_asins[0]
            title = backend.product_title(target)
            env.step(f"search[{title}]")
            env.step(f"click[{target}]")
            observation, reward, done, _, info = env.step("click[Buy Now]")
            self.assertEqual(reward, 1.0)
            self.assertFalse(done)
            info_json = json.dumps(info, sort_keys=True)
            self.assertNotIn(target, info_json)
            self.assertNotIn("actual_asin", info_json)
            self.assertTrue(info["tool_ops"][0]["purchase_correct"])
            self.assertEqual(info["tool_ops"][0]["raw_action"], "click[Buy Now]")
            self.assertTrue(info["tool_ops"][0]["committed"])
            self.assertTrue(info["tool_ops"][0]["session_advanced"])
            self.assertFalse(info["tool_ops"][0]["terminal"])
            self.assertNotIn("purchase_history", info)
            self.assertNotIn("purchase_correct", observation)
            self.assertNotIn(target, observation)

            env.reset(data_idx=0)
            wrong = next(
                product.asin
                for product in generator.generate_orbit(0, split="train").tasks[0].phases[0].candidates
                if product.asin != target
            )
            wrong_title = backend.product_title(wrong)
            env.step(f"search[{wrong_title}]")
            env.step(f"click[{wrong}]")
            observation, _, wrong_done, _, wrong_info = env.step("click[Buy Now]")
            self.assertTrue(wrong_done)
            self.assertEqual(observation, "The shopping episode has ended.\n\nTask family: bundled_shopping\nProgress: 0/6")
            wrong_json = json.dumps(wrong_info, sort_keys=True)
            self.assertNotIn(target, wrong_json)
            self.assertNotIn(wrong, wrong_json)
            self.assertFalse(wrong_info["tool_ops"][0]["purchase_correct"])
            self.assertEqual(
                wrong_info["tool_ops"][0]["raw_action"], "click[Buy Now]"
            )
            self.assertTrue(wrong_info["tool_ops"][0]["committed"])
            self.assertFalse(wrong_info["tool_ops"][0]["session_advanced"])
            self.assertTrue(wrong_info["tool_ops"][0]["terminal"])
            self.assertNotIn("purchase_correct", observation)
        finally:
            env.close()

    def test_same_attribute_product_outside_approved_shortlist_is_rejected(self) -> None:
        pool, records, prices = make_fixture_pool(2)
        generator = NaturalAttributeChainGenerator(pool=pool, seed=233)
        provider = VerifiedProceduralBundleProvider(
            generator=generator,
            split="train",
            task_count=2,
        )
        backend = FakeNativeBackend(records, prices)
        env = ProceduralMemoryWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid="procedural-shortlist-domain-test",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
        )
        try:
            observation, info = env.reset(data_idx=0)
            phase = generator.generate_orbit(0, split="train").tasks[0].phases[0]
            approved_asins = {candidate.asin for candidate in phase.candidates}
            target = next(
                candidate for candidate in phase.candidates
                if candidate.asin == phase.target_asin
            )
            outside = next(
                product
                for product in pool.products
                if product.split == "train"
                and product.scenario_id == phase.scenario_id
                and product.slot_id == phase.slot_id
                and product.attribute_value == target.attribute_value
                and product.asin not in approved_asins
            )
            self.assertNotIn(f"[{outside.asin}]", observation)
            self.assertEqual(
                info["purchase_eligibility_scope"],
                "current_phase_two_approved_listings",
            )
            self.assertEqual(
                info["task_prompt_product_identity"],
                "complete_native_title",
            )
            self.assertFalse(info["target_asin_in_task_prompt"])
            self.assertTrue(info["native_search_result_asin_handles_visible"])
            self.assertTrue(info["native_click_action_uses_asin_handle"])
            self.assertTrue(info["purchase_receipt_asin_verification"])
            self.assertFalse(info["global_catalog_attribute_uniqueness_claimed"])
            self.assertTrue(
                info["global_catalog_normalized_title_uniqueness_claimed"]
            )

            search_observation, _, _, _, _ = env.step(f"search[{outside.title}]")
            self.assertIn(f"click[{outside.asin}]", search_observation)
            env.step(f"click[{outside.asin}]")
            terminal, reward, done, _, terminal_info = env.step("click[Buy Now]")
            self.assertTrue(done)
            self.assertEqual(reward, -0.01)
            self.assertIn("The shopping episode has ended.", terminal)
            public_json = json.dumps(terminal_info, sort_keys=True)
            self.assertNotIn(outside.asin, public_json)
            self.assertNotIn(phase.target_asin, public_json)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
