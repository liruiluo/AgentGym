from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agentenv_agentmemory.env_wrapper import AgentMemoryWrapper
from agentenv_agentmemory.memoryarena_dataset import (
    ACTION_SURFACE_VERSION,
    EXPECTED_DOMAIN_DATA_SHA256,
    EXPECTED_MEMORYARENA_COMMIT,
    MemoryArenaDatasetProvenance,
    MemoryArenaDatasetError,
    load_memoryarena_bundles,
    load_memoryarena_dataset,
    parse_budget_cents,
)


def make_asin(source_row_id: int, session_index: int) -> str:
    return f"B{source_row_id * 6 + session_index:09d}"


def make_question(
    step_index: int,
    *,
    budget: str = "70",
    marker_step: int | None = None,
    suffix: str = "",
) -> str:
    marker_step = step_index if marker_step is None else marker_step
    return (
        "You are an intelligent Shopping Agent operating in a webshop.\n\n"
        "*** GLOBAL RULES ***\n"
        "1. **Evaluate All:** Compare all candidates.\n"
        f"2. **Total Budget:** All items combined must not exceed ${budget}.\n"
        "3. **Product Purchase:** Buy Product 1 first, then Product 2.\n\n"
        + "-" * 64
        + "\n"
        + f"Product {marker_step}:\n"
        + f"### Select product {step_index}\n"
        + f"**Goal:** Preserve exact instruction {step_index}.\n"
        + "**Available Options:**\n"
        + f"- Candidate {step_index} alpha\n"
        + f"- Candidate {step_index} beta{suffix}"
    )


def make_record(source_row_id: int, *, budget: str = "70") -> dict:
    return {
        "id": source_row_id,
        "questions": [
            make_question(step_index, budget=budget)
            for step_index in range(1, 7)
        ],
        "answers": [
            {
                "target_asin": make_asin(source_row_id, session_index),
                "attributes": [f"attribute-{session_index}", "exact source value"],
            }
            for session_index in range(6)
        ],
        "category": f"fixture_item_{source_row_id}",
    }


def encode_jsonl(records: list[dict]) -> bytes:
    return (
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        )
    ).encode("utf-8")


def target_asins(records: list[dict]) -> set[str]:
    return {
        answer["target_asin"]
        for record in records
        for answer in record["answers"]
    }


class MemoryArenaDatasetTests(unittest.TestCase):
    def load_records(
        self,
        records: list[dict],
        *,
        catalog_asins: set[str] | None = None,
        payload: bytes | None = None,
        expected_sha256: str | None = None,
    ):
        payload = encode_jsonl(records) if payload is None else payload
        expected_sha256 = expected_sha256 or hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bundled_shopping_data.jsonl"
            path.write_bytes(payload)
            return load_memoryarena_dataset(
                path,
                frozen_product_asins=(
                    target_asins(records) if catalog_asins is None else catalog_asins
                ),
                expected_raw_sha256=expected_sha256,
                expected_bundle_count=len(records),
            )

    def assert_rejected(self, records: list[dict], message: str) -> None:
        with self.assertRaisesRegex(MemoryArenaDatasetError, message):
            self.load_records(records)

    def test_loads_150_complete_bundles_with_exact_split_and_provenance(self) -> None:
        records = [
            make_record(source_row_id, budget="1,234.56")
            for source_row_id in range(150)
        ]

        dataset = self.load_records(records)

        self.assertEqual(len(dataset), 150)
        self.assertEqual(sum(len(bundle.sessions) for bundle in dataset), 900)
        self.assertEqual(len(dataset.for_split("train")), 120)
        self.assertEqual(len(dataset.for_split("dev")), 15)
        self.assertEqual(len(dataset.for_split("test")), 15)
        self.assertEqual(dataset.bundles[0].task_id, "fixture_item_0")
        self.assertEqual(dataset.bundles[0].source_row_id, 0)
        self.assertEqual(dataset.bundles[0].budget_cents, 123456)
        self.assertEqual(
            dataset.bundles[0].target_asins,
            tuple(make_asin(0, i) for i in range(6)),
        )
        self.assertEqual(dataset.bundles[8].split, "dev")
        self.assertEqual(dataset.bundles[9].split, "test")
        self.assertEqual(dataset.bundles[10].split, "train")
        self.assertEqual(
            dataset.provenance.split_counts,
            (("train", 120), ("dev", 15), ("test", 15)),
        )
        self.assertEqual(dataset.provenance.session_count, 900)
        self.assertEqual(dataset.provenance.memoryarena_commit, EXPECTED_MEMORYARENA_COMMIT)
        self.assertEqual(dataset.provenance.domain_data_sha256, EXPECTED_DOMAIN_DATA_SHA256)
        self.assertEqual(dataset.provenance.action_surface_version, ACTION_SURFACE_VERSION)
        self.assertTrue(dataset.provenance.target_asin_membership_verified)
        self.assertRegex(dataset.provenance.raw_dataset_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(dataset.provenance.split_manifest_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            dataset.bundles[149].provenance.split_manifest_sha256,
            dataset.provenance.split_manifest_sha256,
        )
        self.assertEqual(dataset.bundles[149].provenance.source_line_number, 150)

    def test_webshop_metadata_exposes_complete_dataset_provenance(self) -> None:
        provenance = MemoryArenaDatasetProvenance(
            raw_dataset_path="/frozen/bundled_shopping_data.jsonl",
            raw_dataset_sha256="1" * 64,
            memoryarena_commit="2" * 40,
            domain_data_sha256="3" * 64,
            action_surface_version=ACTION_SURFACE_VERSION,
            split_strategy="source_position_mod10_8_1_1_v1",
            split_manifest_sha256="4" * 64,
            split_counts=(("train", 120), ("dev", 15), ("test", 15)),
            bundle_count=150,
            sessions_per_bundle=6,
            session_count=900,
            target_asin_membership_verified=True,
        )
        wrapper = AgentMemoryWrapper.__new__(AgentMemoryWrapper)
        wrapper.dataset = SimpleNamespace(provenance=provenance)
        wrapper.tasks = tuple(
            SimpleNamespace(split=split)
            for split in ("train", "dev", "test")
        )
        wrapper.annotation_gate = SimpleNamespace(
            mode="strict",
            manifest_sha256="5" * 64,
            allowed_task_ids_sha256="6" * 64,
            allowed_task_ids=("a", "b", "c"),
        )
        wrapper.reward_contract = {"contract": "fixture"}
        wrapper.ltm_inventory_mode = "hidden"
        wrapper.backend = SimpleNamespace(metadata=lambda: {"backend": "fixture"})

        metadata = wrapper.metadata()

        self.assertEqual(metadata["task_count"], 3)
        self.assertEqual(metadata["dataset_sha256"], "1" * 64)
        self.assertEqual(metadata["raw_dataset_sha256"], "1" * 64)
        self.assertEqual(metadata["dataset_provenance"], provenance.as_manifest())
        self.assertEqual(metadata["dataset_provenance"]["bundle_count"], 150)
        self.assertEqual(metadata["dataset_provenance"]["session_count"], 900)

    def test_preserves_question_instruction_candidate_context_and_answer(self) -> None:
        records = [make_record(0)]
        exact_question = make_question(1, suffix=" with trailing words")
        records[0]["questions"][0] = exact_question
        records[0]["answers"][0]["target_asin"] = "b000000000"
        catalog_asins = {asin.upper() for asin in target_asins(records)}

        dataset = self.load_records(records, catalog_asins=catalog_asins)
        bundle = dataset.bundles[0]
        session = bundle.sessions[0]

        self.assertEqual(bundle.questions[0], exact_question)
        self.assertEqual(session.question, exact_question)
        self.assertEqual(
            session.instruction,
            "Product 1:\n### Select product 1\n**Goal:** Preserve exact instruction 1.\n",
        )
        self.assertEqual(
            session.candidate_context,
            "**Available Options:**\n"
            "- Candidate 1 alpha\n"
            "- Candidate 1 beta with trailing words",
        )
        self.assertEqual(
            session.candidate_options,
            ("Candidate 1 alpha", "Candidate 1 beta with trailing words"),
        )
        self.assertEqual(session.raw_target_asin, "b000000000")
        self.assertEqual(session.target_asin, "B000000000")
        self.assertEqual(session.answer_attributes, ("attribute-0", "exact source value"))
        self.assertEqual(bundle.answer_attributes[0], session.answer_attributes)

    def test_bundle_only_loader_exports_parent_facing_tuple_api(self) -> None:
        records = [make_record(0)]
        payload = encode_jsonl(records)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "raw.jsonl"
            path.write_bytes(payload)

            bundles = load_memoryarena_bundles(
                path,
                frozen_product_asins=target_asins(records),
                expected_raw_sha256=hashlib.sha256(payload).hexdigest(),
                expected_bundle_count=1,
            )

        self.assertIsInstance(bundles, tuple)
        self.assertEqual(bundles[0].task_id, "fixture_item_0")
        self.assertEqual(len(bundles[0].questions), 6)
        self.assertEqual(len(bundles[0].target_asins), 6)
        self.assertEqual(bundles[0].budget_cents, 7000)

    def test_budget_parser_is_cent_exact(self) -> None:
        self.assertEqual(parse_budget_cents(make_question(1, budget="70")), 7000)
        self.assertEqual(parse_budget_cents(make_question(1, budget="70.5")), 7050)
        self.assertEqual(parse_budget_cents(make_question(1, budget="1,234.56")), 123456)

    def test_rejects_missing_or_misaligned_session_rows(self) -> None:
        missing_question = make_record(0)
        missing_question["questions"].pop()
        self.assert_rejected([missing_question], "exactly 6 aligned questions and answers")

        misaligned_marker = make_record(0)
        misaligned_marker["questions"][2] = make_question(3, marker_step=4)
        self.assert_rejected([misaligned_marker], "product marker is misaligned")

        misordered_source_id = make_record(1)
        self.assert_rejected([misordered_source_id], "is misordered")

    def test_rejects_inconsistent_or_malformed_budget(self) -> None:
        inconsistent = make_record(0)
        inconsistent["questions"][5] = make_question(6, budget="71")
        self.assert_rejected([inconsistent], "inconsistent six-session budgets")

        malformed = make_record(0)
        malformed["questions"][0] = malformed["questions"][0].replace(
            "**Total Budget:**", "**Budget:**"
        )
        self.assert_rejected([malformed], "exactly one canonical Total Budget")

    def test_rejects_invalid_or_catalog_missing_asin(self) -> None:
        invalid = make_record(0)
        invalid["answers"][0]["target_asin"] = "BAD"
        with self.assertRaisesRegex(MemoryArenaDatasetError, "invalid target ASIN"):
            self.load_records(
                [invalid],
                catalog_asins=target_asins([make_record(0)]),
            )

        missing = make_record(0)
        catalog = target_asins([missing]) - {missing["answers"][3]["target_asin"]}
        with self.assertRaisesRegex(
            MemoryArenaDatasetError,
            "absent from the frozen native product dictionary",
        ):
            self.load_records([missing], catalog_asins=catalog)

    def test_rejects_malformed_answer_and_candidate_context(self) -> None:
        missing_attributes = make_record(0)
        del missing_attributes["answers"][1]["attributes"]
        self.assert_rejected([missing_attributes], "missing required key 'attributes'")

        no_candidates = make_record(0)
        no_candidates["questions"][0] = no_candidates["questions"][0].replace(
            "- Candidate 1 alpha\n- Candidate 1 beta", ""
        )
        self.assert_rejected([no_candidates], "has no candidate options")

        duplicate_candidates = make_record(0)
        duplicate_candidates["questions"][0] = duplicate_candidates["questions"][0].replace(
            "Candidate 1 beta", "Candidate 1 alpha"
        )
        self.assert_rejected([duplicate_candidates], "duplicate candidate option text")

    def test_rejects_duplicate_task_ids_and_wrong_dataset_hash(self) -> None:
        records = [make_record(0), make_record(1)]
        records[1]["category"] = records[0]["category"]
        self.assert_rejected(records, "category/task IDs must be unique")

        good = [make_record(0)]
        payload = encode_jsonl(good)
        with self.assertRaisesRegex(MemoryArenaDatasetError, "SHA256 mismatch"):
            self.load_records(
                good,
                payload=payload,
                expected_sha256="0" * 64,
            )

    def test_rejects_blank_jsonl_rows(self) -> None:
        records = [make_record(0)]
        payload = encode_jsonl(records) + b"\n"
        with self.assertRaisesRegex(MemoryArenaDatasetError, "blank JSONL row"):
            self.load_records(records, payload=payload)


if __name__ == "__main__":
    unittest.main()
