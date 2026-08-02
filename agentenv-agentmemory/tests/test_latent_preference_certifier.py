from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.latent_preference import (
    LatentPreferenceGenerator,
    NativePreferenceCertificationConfig,
    NativePreferencePoolCertificationError,
    VerifiedLatentPreferenceBundleProvider,
    attest_latent_preference_runtime_inputs,
    certify_native_preference_product_pool,
    verify_latent_preference_orbit,
    write_preference_product_pool_manifest,
)
from agentenv_agentmemory.latent_preference_webshop_env import (
    LATENT_PREFERENCE_SURFACE,
    LatentPreferenceWebShopEnv,
)
from agentenv_agentmemory.latent_preference.certifier import (
    CANDIDATE_SCHEMA,
    _Candidate,
    _Slot,
    _attribute_context_is_valid,
    _category_title_evidence,
    _category_title_exclusions,
    _deterministic_unique_asin_matching,
    _guard_matches,
    _load_candidates,
    file_sha256,
)
from agentenv_agentmemory.latent_preference.schema import (
    LatentPreferenceDataError,
    canonical_sha256,
    normalize_native_title,
)
from agentenv_agentmemory.native_webshop_backend import NativePage, NativePurchase


COLOR_RECIPE_ID = "color.black_gray"
COLOR_CATEGORIES = ("phone_case", "pillowcase", "watch_band", "window_curtain")


class FakeNativePreferenceBackend:
    surface = "memoryarena_webshop_native_v1"

    def __init__(
        self,
        records: dict[str, dict[str, object]],
        *,
        unsearchable_asins: set[str] | None = None,
    ) -> None:
        self.records = records
        self.unsearchable_asins = set(unsearchable_asins or ())
        self.sessions: dict[str, dict[str, object]] = {}

    def metadata(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "price_seed": 233,
            "product_count": len(self.records),
            "price_table_sha256": canonical_sha256(
                [
                    [asin, self.product_price_cents(asin)]
                    for asin in sorted(self.records)
                ]
            ),
            "upstream_provenance": {
                "commit": "f" * 40,
                "runtime_tree_sha256": "e" * 64,
            },
        }

    def product_asins(self):
        return self.records.keys()

    def has_product(self, asin: str) -> bool:
        return asin.upper() in self.records

    def product_title(self, asin: str) -> str:
        return str(self.records[asin.upper()]["Title"])

    def product_record(self, asin: str) -> dict[str, object]:
        record = self.records[asin.upper()]
        return {
            key: record.get(key)
            for key in ("Title", "category", "query", "product_category")
        }

    def product_price_cents(self, asin: str) -> int:
        return int(self.records[asin.upper()]["price_cents"])

    def product_record_sha256(self, asin: str) -> str:
        return canonical_sha256(self.records[asin.upper()])

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        self.sessions[session_token] = {
            "instruction": instruction,
            "search_results": (),
            "asin": None,
        }
        return NativePage(
            observation=instruction,
            url=f"http://native/search/{session_token}",
            has_search_bar=True,
            clickables=(),
        )

    def step(self, session_token: str, action: str) -> NativePage:
        session = self.sessions[session_token]
        if action.startswith("search[") and action.endswith("]"):
            query = normalize_native_title(action[7:-1])
            results = tuple(
                asin
                for asin, record in sorted(self.records.items())
                if asin not in self.unsearchable_asins
                and query in normalize_native_title(str(record["Title"]))
            )
            session["search_results"] = results
            return NativePage(
                observation="\n".join(results),
                url=f"http://native/search/{session_token}",
                has_search_bar=True,
                clickables=results,
            )
        if action.startswith("click[") and action.endswith("]"):
            argument = action[6:-1]
            if argument.casefold() == "buy now":
                asin = str(session["asin"])
                return NativePage(
                    observation=f"Purchased {asin}",
                    url=f"http://native/done/{session_token}/{asin}",
                    has_search_bar=False,
                    clickables=(),
                    purchase=NativePurchase(
                        asin=asin,
                        price_cents=self.product_price_cents(asin),
                        selected_options={},
                    ),
                )
            asin = argument.upper()
            if asin not in session["search_results"]:
                raise ValueError(f"ASIN {asin} was not in the visible search results")
            session["asin"] = asin
            return NativePage(
                observation=self.product_title(asin),
                url=f"http://native/item/{session_token}/{asin}",
                has_search_bar=True,
                clickables=("Buy Now",),
            )
        raise ValueError(action)

    def close_session(self, session_token: str) -> None:
        self.sessions.pop(session_token, None)

    def close(self) -> None:
        self.sessions.clear()


def _source_row(
    *,
    asin: str,
    category_id: str,
    color: str,
    ordinal: int,
    second_color: str | None = None,
) -> dict[str, object]:
    color_text = color.title()
    if second_color is not None:
        color_text += f"/{second_color.title()}"
    category_text = category_id.replace("_", " ").title()
    title = f"{color_text} {category_text} Certified Model {ordinal} Cotton"
    semantic: dict[str, object] = {
        "category_id": category_id,
        "axis": "color",
        "attribute_value": color,
        "asin": asin,
        "title": title,
        "product_category": f"Fixture > {category_text}",
        "title_evidence": [color.title()],
    }
    return {
        "schema": CANDIDATE_SCHEMA,
        **semantic,
        "classification_sha256": canonical_sha256(semantic),
        "normalized_title": normalize_native_title(title),
    }


def _write_fixture(
    root: Path,
    *,
    candidates_per_cell: int,
    ambiguous_first: bool = False,
) -> tuple[Path, FakeNativePreferenceBackend]:
    rows: list[dict[str, object]] = []
    records: dict[str, dict[str, object]] = {}
    ordinal = 1
    for category in COLOR_CATEGORIES:
        for color in ("black", "gray"):
            for cell_index in range(candidates_per_cell):
                asin = f"B{ordinal:09d}"
                row = _source_row(
                    asin=asin,
                    category_id=category,
                    color=color,
                    ordinal=ordinal,
                    second_color=(
                        "gold"
                        if ambiguous_first
                        and category == "phone_case"
                        and color == "black"
                        and cell_index == 0
                        else None
                    ),
                )
                rows.append(row)
                records[asin] = {
                    "Title": row["title"],
                    "category": "fixture",
                    "query": category,
                    "product_category": row["product_category"],
                    "price_cents": 1_000 + ordinal,
                }
                ordinal += 1
    path = root / "candidates.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path, FakeNativePreferenceBackend(records)


def _config() -> NativePreferenceCertificationConfig:
    return NativePreferenceCertificationConfig(
        recipe_ids=(COLOR_RECIPE_ID,),
        products_per_cell=2,
        candidate_cap_per_cell=24,
    )


class LatentPreferenceCertifierTests(unittest.TestCase):
    def test_runtime_attestation_rechecks_every_pinned_native_input(self) -> None:
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
                f"{hashlib.sha256(index.read_bytes()).hexdigest()}  segment\n",
                encoding="utf-8",
            )
            candidates, backend = _write_fixture(
                root,
                candidates_per_cell=6,
            )
            pool, _ = certify_native_preference_product_pool(
                backend,
                candidate_artifact=candidates,
                expected_candidate_artifact_sha256=file_sha256(candidates),
                catalog_sha256=hashlib.sha256(items.read_bytes()).hexdigest(),
                attributes_sha256=hashlib.sha256(attributes.read_bytes()).hexdigest(),
                lucene_index_sha256=hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                config=_config(),
            )
            attest_latent_preference_runtime_inputs(
                pool,
                backend,
                items_file=items,
                attributes_file=attributes,
                search_root=search_root,
                lucene_manifest=manifest,
            )
            first = pool.products[0]
            backend.records[first.asin]["Title"] = "Changed native product title"
            with self.assertRaisesRegex(RuntimeError, "title"):
                attest_latent_preference_runtime_inputs(
                    pool,
                    backend,
                    items_file=items,
                    attributes_file=attributes,
                    search_root=search_root,
                    lucene_manifest=manifest,
                )

    def test_certified_pool_runs_on_redacted_latent_preference_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, backend = _write_fixture(
                Path(temp),
                candidates_per_cell=6,
            )
            pool, _ = certify_native_preference_product_pool(
                backend,
                candidate_artifact=path,
                expected_candidate_artifact_sha256=file_sha256(path),
                catalog_sha256="1" * 64,
                attributes_sha256="2" * 64,
                lucene_index_sha256="3" * 64,
                config=_config(),
            )
            provider = VerifiedLatentPreferenceBundleProvider(
                generator=LatentPreferenceGenerator(pool=pool, seed=233),
                split="train",
                task_count=2,
            )
            env = LatentPreferenceWebShopEnv(
                provider=provider,
                backend=backend,
                first_valid_add_reward=0.0,
                first_valid_later_session_retrieve_reward=0.0,
            )
            try:
                _, info = env.reset(data_idx=0)
                self.assertEqual(info["surface"], LATENT_PREFERENCE_SURFACE)
                self.assertEqual(
                    info["task_family"],
                    "procedural_latent_user_preference_shopping",
                )
                self.assertNotIn("task_id", info)
                self.assertNotIn("purchase_history", info)

                target_asin = provider.get(0).target_asins[0]
                product = pool.product_by_asin(target_asin)
                env.step(f"search[{product.search_query}]")
                env.step(f"click[{target_asin}]")
                observation, reward, done, _, purchase_info = env.step(
                    "click[Buy Now]"
                )
                self.assertEqual(reward, 1.0)
                self.assertFalse(done)
                self.assertNotIn("actual_asin", json.dumps(purchase_info))
                self.assertNotIn("purchase_correct", observation)
                self.assertNotIn(target_asin, observation)
                self.assertEqual(
                    purchase_info["tool_ops"],
                    [
                        {
                            "op": "BUY",
                            "raw_action": "click[Buy Now]",
                            "committed": True,
                            "purchase_correct": True,
                            "terminal": False,
                            "session_advanced": True,
                            "step": 3,
                            "session_index": 0,
                        }
                    ],
                )

                env.reset(data_idx=0)
                first_phase = provider.generator.generate_orbit(
                    0, split="train"
                ).tasks[0].phases[0]
                wrong_asin = next(
                    candidate.asin
                    for candidate in first_phase.candidates
                    if candidate.asin != target_asin
                )
                wrong_product = pool.product_by_asin(wrong_asin)
                env.step(f"search[{wrong_product.search_query}]")
                env.step(f"click[{wrong_asin}]")
                observation, reward, done, _, purchase_info = env.step(
                    "click[Buy Now]"
                )
                self.assertEqual(reward, -0.01)
                self.assertTrue(done)
                self.assertNotIn("purchase_correct", observation)
                self.assertNotIn(target_asin, observation)
                self.assertNotIn(wrong_asin, observation)
                self.assertNotIn("actual_asin", json.dumps(purchase_info))
                self.assertEqual(
                    purchase_info["tool_ops"],
                    [
                        {
                            "op": "BUY",
                            "raw_action": "click[Buy Now]",
                            "committed": True,
                            "purchase_correct": False,
                            "terminal": True,
                            "session_advanced": False,
                            "step": 3,
                            "session_index": 0,
                        }
                    ],
                )
            finally:
                env.close()

    def test_certifies_native_pool_and_generated_orbit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, backend = _write_fixture(
                Path(temp),
                candidates_per_cell=6,
            )
            pool, audit = certify_native_preference_product_pool(
                backend,
                candidate_artifact=path,
                expected_candidate_artifact_sha256=file_sha256(path),
                catalog_sha256="1" * 64,
                attributes_sha256="2" * 64,
                lucene_index_sha256="3" * 64,
                config=_config(),
            )

        self.assertEqual(len(pool.products), 48)
        self.assertEqual(len({product.asin for product in pool.products}), 48)
        self.assertTrue(
            all(
                product.native_search_verified
                and product.native_open_verified
                and product.native_purchase_verified
                and product.native_title_globally_unique
                for product in pool.products
            )
        )
        self.assertEqual(audit["status"], "certified")
        self.assertEqual(audit["counts"]["native_probes"], 48)
        self.assertTrue(
            audit["verification"]["global_asin_uniqueness_across_axes_cells_splits"]
        )

        generator = LatentPreferenceGenerator(pool=pool, seed=233)
        orbit = generator.generate_orbit(0, split="train")
        proof = verify_latent_preference_orbit(
            orbit,
            pool=pool,
            expected_generator_seed=233,
        )
        self.assertEqual(proof.valid_solution_counts, (1, 1))
        self.assertEqual(proof.enumerated_path_count_per_branch, 64)

    def test_candidate_hash_mismatch_fails_before_native_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, backend = _write_fixture(
                Path(temp),
                candidates_per_cell=6,
            )
            with self.assertRaisesRegex(
                LatentPreferenceDataError,
                "candidate artifact SHA256 mismatch",
            ):
                certify_native_preference_product_pool(
                    backend,
                    candidate_artifact=path,
                    expected_candidate_artifact_sha256="0" * 64,
                    catalog_sha256="1" * 64,
                    attributes_sha256="2" * 64,
                    lucene_index_sha256="3" * 64,
                    config=_config(),
                )
        self.assertFalse(backend.sessions)

    def test_dataset_audit_cli_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates, backend = _write_fixture(
                root,
                candidates_per_cell=6,
            )
            pool, _ = certify_native_preference_product_pool(
                backend,
                candidate_artifact=candidates,
                expected_candidate_artifact_sha256=file_sha256(candidates),
                catalog_sha256="1" * 64,
                attributes_sha256="2" * 64,
                lucene_index_sha256="3" * 64,
                config=_config(),
            )
            pool_path = root / "pool.json"
            pool_sha256 = write_preference_product_pool_manifest(pool, pool_path)
            manifest_one = root / "manifest-one.json"
            manifest_two = root / "manifest-two.json"
            package_root = Path(__file__).resolve().parents[1]
            script = (
                package_root
                / "scripts"
                / "audits"
                / "verify_latent_preference_dataset.py"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(package_root)
            command = [
                sys.executable,
                str(script),
                "--product-pool",
                str(pool_path),
                "--product-pool-sha256",
                pool_sha256,
                "--split",
                "train",
                "--generator-seed",
                "233",
                "--task-count",
                "24",
            ]
            first = subprocess.run(
                [*command, "--output-manifest", str(manifest_one)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            manifest_sha256 = file_sha256(manifest_one)
            second = subprocess.run(
                [
                    *command,
                    "--output-manifest",
                    str(manifest_two),
                    "--expected-manifest-sha256",
                    manifest_sha256,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(manifest_one.read_bytes(), manifest_two.read_bytes())
            self.assertIn("tasks=24", first.stdout)
            self.assertIn(f"manifest_sha256={manifest_sha256}", second.stdout)
            manifest = json.loads(manifest_one.read_bytes())
            self.assertEqual(
                manifest["verification"]["total_enumerated_paths"],
                24 * 64,
            )
            self.assertTrue(manifest["verification"]["unique_semantic_tasks"])

    def test_multicolor_title_is_rejected_instead_of_accepted_as_black(self) -> None:
        self.assertEqual(
            _guard_matches("color", "Black/Gold geometric comforter"),
            ("black", "gold"),
        )
        with tempfile.TemporaryDirectory() as temp:
            path, backend = _write_fixture(
                Path(temp),
                candidates_per_cell=6,
                ambiguous_first=True,
            )
            with self.assertRaises(NativePreferencePoolCertificationError) as caught:
                certify_native_preference_product_pool(
                    backend,
                    candidate_artifact=path,
                    expected_candidate_artifact_sha256=file_sha256(path),
                    catalog_sha256="1" * 64,
                    attributes_sha256="2" * 64,
                    lucene_index_sha256="3" * 64,
                    config=_config(),
                )
        self.assertEqual(caught.exception.audit["status"], "failed")
        self.assertEqual(
            caught.exception.audit["counts"][
                "rejected_broad_guard_multiple_or_conflicting_values"
            ],
            1,
        )

    def test_color_guard_rejects_secondary_and_transparent_colors(self) -> None:
        self.assertEqual(
            _guard_matches("color", "Black/Lime silicone watch band"),
            ("black", "lime"),
        )
        self.assertEqual(
            _guard_matches("color", "Lavender Gray phone case"),
            ("gray", "lavender"),
        )
        self.assertEqual(
            _guard_matches("color", "Transparent crystal clear case - Black"),
            ("black", "clear", "transparent"),
        )

    def test_dietary_guard_allows_compatible_labels_but_not_recipe_pair(self) -> None:
        self.assertEqual(
            _guard_matches(
                "dietary_profile",
                "Gluten-Free Vegan Non-GMO chocolate drink powder",
            ),
            ("gluten_free",),
        )
        self.assertEqual(
            _guard_matches(
                "dietary_profile",
                "Organic Gluten-Free vanilla drink mix",
            ),
            ("gluten_free", "organic"),
        )

    def test_food_category_guards_reject_wrong_product_forms(self) -> None:
        self.assertTrue(
            _category_title_evidence(
                "drink_mix",
                "Organic chocolate superfood drink mix powder",
            )
        )
        for title in (
            "Ready-to-drink chocolate protein mix, 11 fl oz bottles",
            "Vanilla drink mix syrup in a bottle",
            "Coconut milk drink with pulp and smoothie mix",
        ):
            self.assertTrue(_category_title_exclusions("drink_mix", title))

        for title in (
            "Chocolate chewy granola bars",
            "Vanilla granola bites and minis",
            "Organic vanilla granola butter alternative",
            "Vanilla cereal: it is not granola",
        ):
            self.assertTrue(_category_title_exclusions("granola", title))

        self.assertTrue(
            _category_title_evidence(
                "cake_mix",
                "Gluten-Free Chocolate Cake Mix",
            )
        )
        self.assertFalse(
            _category_title_evidence(
                "cake_mix",
                "Low Carb Vanilla Frosting Mix",
            )
        )
        self.assertFalse(
            _category_title_evidence(
                "cake_mix",
                "Pancake and Waffle Mix with Chocolate Brownie Mix",
            )
        )
        self.assertFalse(
            _category_title_evidence(
                "nutrition_bar",
                "Organic dark chocolate bean to bar",
            )
        )
        self.assertTrue(
            _category_title_evidence(
                "nutrition_bar",
                "Organic dark chocolate protein bar",
            )
        )

    def test_organic_must_describe_the_product_not_only_an_ingredient(self) -> None:
        self.assertTrue(
            _attribute_context_is_valid(
                axis="dietary_profile",
                attribute_value="organic",
                category_id="cookies",
                title="Emmy's Organic Chocolate Chip Coconut Cookies",
            )
        )
        self.assertFalse(
            _attribute_context_is_valid(
                axis="dietary_profile",
                attribute_value="organic",
                category_id="cookies",
                title=(
                    "Lactation Cookies made with brewer's yeast and Organic "
                    "Flaxseed Meal"
                ),
            )
        )

    def test_failed_native_edge_is_removed_and_matching_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, backend = _write_fixture(
                Path(temp),
                candidates_per_cell=7,
            )
            candidates_by_cell, _ = _load_candidates(path, config=_config())
            rejected = candidates_by_cell[("color", "phone_case", "black")][0]
            backend.unsearchable_asins.add(rejected.asin)
            pool, audit = certify_native_preference_product_pool(
                backend,
                candidate_artifact=path,
                expected_candidate_artifact_sha256=file_sha256(path),
                catalog_sha256="1" * 64,
                attributes_sha256="2" * 64,
                lucene_index_sha256="3" * 64,
                config=_config(),
            )

        self.assertNotIn(rejected.asin, {product.asin for product in pool.products})
        self.assertGreater(audit["counts"]["matching_rebuilds"], 1)
        self.assertEqual(audit["counts"]["blocked_asin_cell_edges"], 1)
        self.assertEqual(
            audit["counts"]["native_probe_rejections"],
            {"target_absent_from_all_title_derived_first_pages": 1},
        )

    def test_matching_enforces_global_asin_uniqueness_across_axes(self) -> None:
        shared_asin = "B000000001"

        def candidate(
            asin: str,
            *,
            axis: str,
            category: str,
            value: str,
            selection: str,
        ) -> _Candidate:
            return _Candidate(
                asin=asin,
                title=f"{value} {category} fixture title",
                normalized_title=f"{value} {category} fixture title",
                product_category="fixture",
                category_title_evidence=(category,),
                category_id=category,
                axis=axis,
                attribute_value=value,
                title_evidence=(value,),
                guard_matches=(value,),
                source_candidate_sha256="1" * 64,
                classification_sha256="2" * 64,
                selection_sha256=selection * 64,
                catalog_title_match_count=1,
            )

        color_slot = _Slot("color", "pillowcase", "black", "train", 0)
        pattern_slot = _Slot("pattern", "pillowcase", "floral", "train", 0)
        shared_color = candidate(
            shared_asin,
            axis="color",
            category="pillowcase",
            value="black",
            selection="0",
        )
        color_alternative = candidate(
            "B000000002",
            axis="color",
            category="pillowcase",
            value="black",
            selection="1",
        )
        shared_pattern = candidate(
            shared_asin,
            axis="pattern",
            category="pillowcase",
            value="floral",
            selection="0",
        )
        matching = _deterministic_unique_asin_matching(
            (color_slot, pattern_slot),
            {
                color_slot.base_cell: (shared_color, color_alternative),
                pattern_slot.base_cell: (shared_pattern,),
            },
            blocked_edges=set(),
        )
        self.assertIsNotNone(matching)
        assert matching is not None
        self.assertEqual(matching[pattern_slot].asin, shared_asin)
        self.assertEqual(matching[color_slot].asin, "B000000002")

    def test_title_normalization_uses_unicode_nfkc(self) -> None:
        self.assertEqual(
            normalize_native_title("Ｆｕｌｌ　Ｗｉｄｔｈ"),
            "full width",
        )


if __name__ == "__main__":
    unittest.main()
