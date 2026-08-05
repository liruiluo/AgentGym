from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.latent_preference.schema import (
    canonical_sha256,
    normalize_native_title,
)
from agentenv_agentmemory.native_webshop_backend import NativePage, NativePurchase
from agentenv_agentmemory.negative_constraint import (
    NEGATIVE_CONSTRAINT_RECIPES,
    PROVIDER_MODE_RESEEDED_STREAM,
    NegativeConstraintCandidate,
    NegativeConstraintDataError,
    NegativeConstraintGenerator,
    NegativeConstraintProductPool,
    NegativeConstraintRecipe,
    NativeNegativeConstraintCertificationConfig,
    NativeNegativeConstraintPoolCertificationError,
    VerifiedNegativeConstraintBundleProvider,
    certify_native_negative_constraint_product_pool,
    certify_native_negative_constraint_product_pool_with_reselection,
    load_negative_constraint_native_product_pool,
    load_negative_constraint_product_pool,
    split_for_asin,
    verify_negative_constraint_orbit,
    write_negative_constraint_product_pool_manifest,
)
from agentenv_agentmemory.negative_constraint.runtime_attestation import (
    attest_negative_constraint_runtime_inputs,
)
from agentenv_agentmemory.negative_constraint_webshop_env import (
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    NegativeConstraintFilesystemWebShopEnv,
    NegativeConstraintWebShopEnv,
)
from tests.workspace_test_support import InProcessTestShellSandbox


CATEGORIES = (
    ("area_rug", "area rug"),
    ("phone_case", "phone case"),
    ("pillowcase", "pillowcase"),
    ("window_curtain", "window curtain"),
)
VALUES = (("black", "black"), ("gray", "gray"), ("red", "red"))
SPLITS = ("train", "dev", "test")


def shell_action(command: str, *, workdir: str = ".") -> str:
    return "shell_command " + json.dumps(
        {"command": command, "workdir": workdir, "timeout_ms": 10_000},
        separators=(",", ":"),
    )


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


def write_negative_candidate_fixture(path: Path) -> str:
    rows = []
    counter = 1
    for recipe in NEGATIVE_CONSTRAINT_RECIPES:
        for category_id in recipe.categories:
            category_display = recipe.category_display_name(category_id)
            for value in recipe.values:
                value_display = recipe.value_display_name(value)
                for split in SPLITS:
                    for cell_index in range(3):
                        while True:
                            asin = f"T{counter:09d}"
                            counter += 1
                            if split_for_asin(asin) == split:
                                break
                        title = (
                            f"{value_display.title()} {category_display.title()} "
                            f"Fixture Product {cell_index} {split} {asin[-4:]}"
                        )
                        product_category = f"Fixture > {category_display}"
                        title_evidence = [value_display.title()]
                        classification = {
                            "category_id": category_id,
                            "axis": recipe.axis,
                            "attribute_value": value,
                            "asin": asin,
                            "title": title,
                            "product_category": product_category,
                            "title_evidence": title_evidence,
                        }
                        rows.append(
                            {
                                "schema": (
                                    "agentmemory_latent_preference_rule_candidate_v2"
                                ),
                                "asin": asin,
                                "axis": recipe.axis,
                                "attribute_value": value,
                                "category_id": category_id,
                                "classification_sha256": canonical_sha256(
                                    classification
                                ),
                                "normalized_title": normalize_native_title(title),
                                "product_category": product_category,
                                "title": title,
                                "title_evidence": title_evidence,
                            }
                        )
    data = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


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
            allow_rules_only=True,
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
            allow_rules_only=True,
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


class _FakeNegativeCertificationBackend:
    surface = "memoryarena_webshop_native_v1"

    def __init__(self, pool: NegativeConstraintProductPool) -> None:
        self.records = {
            candidate.asin: {
                "Title": candidate.title,
                "category": [candidate.category_id],
                "query": candidate.category_display_name,
                "product_category": candidate.product_category,
                "price_cents": 1_000 + index,
            }
            for index, candidate in enumerate(pool.candidates)
        }
        self.sessions: dict[str, dict[str, object]] = {}
        self.unsearchable_asins: set[str] = set()

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
            "upstream_provenance": {"memoryarena_commit": "f" * 40},
        }

    def active_session_count(self) -> int:
        return len(self.sessions)

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
        self.sessions[session_token] = {"asin": None, "search_results": ()}
        return NativePage(
            observation=instruction,
            url=f"http://fixture/search/{session_token}",
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
                url=f"http://fixture/search/{session_token}",
                has_search_bar=True,
                clickables=results,
            )
        if action.startswith("click[") and action.endswith("]"):
            argument = action[6:-1]
            if argument.casefold() == "buy now":
                asin = str(session["asin"])
                return NativePage(
                    observation=f"Purchased {asin}",
                    url=f"http://fixture/done/{asin}",
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
                raise ValueError(f"ASIN {asin} was not in the search results")
            session["asin"] = asin
            return NativePage(
                observation=self.product_title(asin),
                url=f"http://fixture/item/{asin}",
                has_search_bar=True,
                clickables=("Buy Now",),
            )
        raise ValueError(action)

    def close_session(self, session_token: str) -> None:
        self.sessions.pop(session_token, None)

    def close(self) -> None:
        self.sessions.clear()


class NegativeConstraintNativeCertificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules_pool = make_negative_fixture_pool()
        self.backend = _FakeNegativeCertificationBackend(self.rules_pool)

    def _certify(self):
        return certify_native_negative_constraint_product_pool(
            self.backend,
            rules_pool=self.rules_pool,
            catalog_sha256="1" * 64,
            attributes_sha256="2" * 64,
            lucene_index_sha256="3" * 64,
            expected_memoryarena_commit="f" * 40,
            config=NativeNegativeConstraintCertificationConfig(
                pool_id="fixture_negative_constraint_native_v2"
            ),
        )

    def test_rules_only_provider_is_explicitly_test_only(self) -> None:
        with self.assertRaisesRegex(NegativeConstraintDataError, "rules-only"):
            VerifiedNegativeConstraintBundleProvider(
                generator=NegativeConstraintGenerator(
                    pool=self.rules_pool,
                    seed=233,
                ),
                split="train",
                task_count=3,
            )

    def test_certifies_round_trips_and_enables_training(self) -> None:
        pool, audit = self._certify()
        self.assertTrue(pool.native_certified)
        self.assertEqual(
            len(pool.native_certificates),
            len(self.rules_pool.candidates),
        )
        self.assertEqual(audit["status"], "certified")
        self.assertTrue(audit["verification"]["training_ready"])
        self.assertEqual(self.backend.active_session_count(), 0)

        generator = NegativeConstraintGenerator(pool=pool, seed=233)
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=generator,
            split="train",
            task_count=3,
        )
        self.assertTrue(provider.metadata()["training_ready"])
        proof = verify_negative_constraint_orbit(
            generator.generate_orbit(0, split="train"),
            pool=pool,
        )
        self.assertTrue(proof.payload()["verification"]["training_ready"])
        self.assertGreater(proof.native_certificate_checks, 0)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "negative-pool.json"
            digest = write_negative_constraint_product_pool_manifest(pool, path)
            loaded = load_negative_constraint_native_product_pool(
                path,
                expected_file_sha256=digest,
            )
        self.assertEqual(pool.semantic_manifest(), loaded.semantic_manifest())

    def test_native_pool_rejects_non_boolean_certification_flag(self) -> None:
        pool, _ = self._certify()
        with self.assertRaisesRegex(NegativeConstraintDataError, "must be a boolean"):
            replace(pool, native_certified=1)

    def test_native_pool_rejects_missing_certifier_version(self) -> None:
        pool, _ = self._certify()
        with self.assertRaisesRegex(NegativeConstraintDataError, "must be a string"):
            replace(pool, certifier_version=None)

    def test_native_pool_rejects_duplicate_certificate(self) -> None:
        pool, _ = self._certify()
        duplicated = tuple(
            sorted(
                (*pool.native_certificates, pool.native_certificates[0]),
                key=lambda certificate: certificate.asin,
            )
        )
        with self.assertRaisesRegex(NegativeConstraintDataError, "exactly once"):
            replace(pool, native_certificates=duplicated)

    def test_unsearchable_selected_product_fails_closed_and_cleans_session(self) -> None:
        failed_asin = min(self.backend.records)
        self.backend.unsearchable_asins.add(failed_asin)
        with self.assertRaises(
            NativeNegativeConstraintPoolCertificationError
        ) as caught:
            self._certify()
        self.assertEqual(caught.exception.audit["status"], "failed")
        self.assertEqual(
            caught.exception.audit["failed_product_asin"],
            failed_asin,
        )
        self.assertEqual(self.backend.active_session_count(), 0)

    def test_candidate_local_failure_reselects_same_cell_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate_path = Path(temp) / "candidates.jsonl"
            candidate_sha256 = write_negative_candidate_fixture(candidate_path)
            initial = load_negative_constraint_product_pool(
                candidate_path,
                expected_file_sha256=candidate_sha256,
            )
            failed_asin = min(candidate.asin for candidate in initial.candidates)
            expected = load_negative_constraint_product_pool(
                candidate_path,
                expected_file_sha256=candidate_sha256,
                blocked_asins={failed_asin},
            )

            backend = _FakeNegativeCertificationBackend(initial)
            for index, candidate in enumerate(expected.candidates, start=1):
                backend.records.setdefault(
                    candidate.asin,
                    {
                        "Title": candidate.title,
                        "category": [candidate.category_id],
                        "query": candidate.category_display_name,
                        "product_category": candidate.product_category,
                        "price_cents": 10_000 + index,
                    },
                )
            backend.unsearchable_asins.add(failed_asin)
            pool, audit = (
                certify_native_negative_constraint_product_pool_with_reselection(
                    backend,
                    candidate_artifact=candidate_path,
                    expected_candidate_artifact_sha256=candidate_sha256,
                    catalog_sha256="1" * 64,
                    attributes_sha256="2" * 64,
                    lucene_index_sha256="3" * 64,
                    expected_memoryarena_commit="f" * 40,
                    config=NativeNegativeConstraintCertificationConfig(
                        pool_id="fixture_negative_constraint_native_v2"
                    ),
                )
            )

        self.assertEqual(pool.candidates, expected.candidates)
        self.assertNotIn(failed_asin, {item.asin for item in pool.candidates})
        self.assertEqual(audit["selection"]["rebuild_count"], 1)
        self.assertEqual(
            audit["selection"]["blocked_candidate_asins"],
            [failed_asin],
        )
        self.assertEqual(
            audit["selection"]["candidate_rejections"][0][
                "rejection_reason"
            ],
            "target_absent_from_all_title_derived_first_pages",
        )
        self.assertTrue(
            audit["verification"]["deterministic_same_cell_split_reselection"]
        )

    def test_native_title_mismatch_audit_records_both_titles(self) -> None:
        failed_asin = min(self.backend.records)
        source_title = str(self.backend.records[failed_asin]["Title"])
        native_title = source_title.upper()
        self.backend.records[failed_asin]["Title"] = native_title
        with self.assertRaises(NativeNegativeConstraintPoolCertificationError) as caught:
            self._certify()
        rejected = caught.exception.audit["probes"][-1]
        self.assertEqual(rejected["rejection_reason"], "native_title_mismatch")
        self.assertEqual(rejected["source_title"], source_title)
        self.assertEqual(rejected["native_title"], native_title)

    def test_manifest_nested_field_tamper_is_rejected(self) -> None:
        pool, _ = self._certify()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "negative-pool.json"
            write_negative_constraint_product_pool_manifest(pool, path)
            payload = json.loads(path.read_bytes())
            del payload["native_certificates"][0]["purchase_receipt_sha256"]
            data = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            with self.assertRaisesRegex(
                NegativeConstraintDataError,
                "certificate fields mismatch",
            ):
                load_negative_constraint_native_product_pool(
                    path,
                    expected_file_sha256=digest,
                )

    def test_dataset_manifest_is_byte_deterministic(self) -> None:
        pool, _ = self._certify()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pool_path = root / "negative-pool.json"
            pool_sha256 = write_negative_constraint_product_pool_manifest(
                pool,
                pool_path,
            )
            first = root / "manifest-one.json"
            second = root / "manifest-two.json"
            package_root = Path(__file__).resolve().parents[1]
            script = (
                package_root
                / "scripts"
                / "audits"
                / "verify_negative_constraint_dataset.py"
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
                "6",
            ]
            subprocess.run(
                [*command, "--output-manifest", str(first)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            digest = hashlib.sha256(first.read_bytes()).hexdigest()
            replay = subprocess.run(
                [
                    *command,
                    "--output-manifest",
                    str(second),
                    "--expected-manifest-sha256",
                    digest,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertIn("tasks=6", replay.stdout)

    def test_runtime_attestation_detects_price_tamper(self) -> None:
        pool, _ = self._certify()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = root / "items.json"
            attributes = root / "attributes.json"
            lucene = root / "lucene.sha256"
            for path in (items, attributes, lucene):
                path.write_text("fixture", encoding="utf-8")

            def frozen_hash(path):
                if Path(path) == items:
                    return pool.catalog_sha256
                if Path(path) == attributes:
                    return pool.attributes_sha256
                if Path(path) == lucene:
                    return pool.lucene_index_sha256
                raise AssertionError(path)

            with patch(
                "agentenv_agentmemory.negative_constraint.runtime_attestation.file_sha256",
                side_effect=frozen_hash,
            ), patch(
                "agentenv_agentmemory.negative_constraint.runtime_attestation.verify_lucene_index_manifest",
                return_value=1,
            ):
                attest_negative_constraint_runtime_inputs(
                    pool,
                    self.backend,
                    items_file=items,
                    attributes_file=attributes,
                    search_root=root,
                    lucene_manifest=lucene,
                )
                asin = pool.candidates[0].asin
                self.backend.records[asin]["price_cents"] = 999_999
                with self.assertRaisesRegex(RuntimeError, "price table"):
                    attest_negative_constraint_runtime_inputs(
                        pool,
                        self.backend,
                        items_file=items,
                        attributes_file=attributes,
                        search_root=root,
                        lucene_manifest=lucene,
                    )


class NegativeConstraintRuntimeTests(unittest.TestCase):
    def _make_env(self, data_idx: int = 0):
        pool = make_negative_fixture_pool()
        generator = NegativeConstraintGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[data_idx]
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=generator,
            split="train",
            task_count=3,
            allow_rules_only=True,
        )
        backend = _FakeNegativeNativeBackend(task)
        env = NegativeConstraintWebShopEnv(
            provider=provider,
            backend=backend,
            env_uid=f"negative-{data_idx}",
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
            allow_rules_only=True,
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

    def test_filesystem_surface_keeps_two_exclusions_at_session_one(self) -> None:
        pool = make_negative_fixture_pool()
        generator = NegativeConstraintGenerator(pool=pool, seed=233)
        task = generator.generate_orbit(0, split="train").tasks[2]
        provider = VerifiedNegativeConstraintBundleProvider(
            generator=generator,
            split="train",
            task_count=3,
            allow_rules_only=True,
        )
        backend = _FakeNegativeNativeBackend(task)
        with tempfile.TemporaryDirectory() as root:
            env = NegativeConstraintFilesystemWebShopEnv(
                provider=provider,
                backend=backend,
                env_uid="negative-filesystem-2",
                shell_sandbox=InProcessTestShellSandbox(),
                workspace_root_parent=Path(root),
                allow_rules_only=True,
            )
            try:
                _, info = env.reset(data_idx=2)
                self.assertEqual(info["surface"], NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE)
                env.step(
                    "apply_patch\n*** Begin Patch\n"
                    "*** Add File: exclusions.md\n"
                    f"+Standing exclusions: {task.canonical_memory_value}\n"
                    "*** End Patch"
                )
                self._purchase(env, task.target_asins[0], 0)
                self.assertEqual(env.current_session_index, 1)
                state = env.workspace.export_state()
                self.assertEqual(state["file_count"], 1)
                self.assertNotIn("content", state["files"][0])
                observation, reward, done, _, read_info = env.step(
                    shell_action("cat exclusions.md")
                )
                self.assertEqual(reward, 0.0)
                self.assertFalse(done)
                self.assertIn("Standing exclusions", observation)
                self.assertEqual(read_info["workspace_ops"][0]["op"], "SHELL_COMMAND")
                _, info = env.install_workspace_causal_intervention("blank")
                self.assertEqual(info["workspace_causal_arm"], "blank")
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
            allow_rules_only=True,
        )
        backend = _FakeNegativeNativeBackend(task)
        with self.assertRaises(ValueError):
            NegativeConstraintWebShopEnv(
                provider=provider,
                backend=backend,
                ltm_inventory_mode="keys",
                allow_rules_only=True,
            )
        with self.assertRaises(ValueError):
            NegativeConstraintWebShopEnv(
                provider=provider,
                backend=backend,
                retrieve_policy="standard",
                allow_rules_only=True,
            )


if __name__ == "__main__":
    unittest.main()
