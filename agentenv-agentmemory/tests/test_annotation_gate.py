from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from agentenv_agentmemory.annotation_gate import (
    ANNOTATION_GATE_MODES,
    AnnotationGateBindings,
    AnnotationGateError,
    AnnotationGateTrustRoot,
    build_annotation_gate_bindings,
    build_annotation_gate_manifest,
    fingerprint_memoryarena_source_tree,
    fingerprint_memoryarena_price_table,
    hash_task_ids,
    validate_annotation_gate_manifest,
    write_annotation_gate_manifest,
    _build_annotation_gate_manifest,
    _validate_annotation_gate_manifest,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def asin(source_id: int, step: int) -> str:
    return f"B{source_id * 6 + step:09d}"


def detailed_checks(status: str) -> dict[str, str]:
    checks = {
        "target_alignment": "pass",
        "target_semantics": "pass",
        "compatibility": "pass",
        "metric_parse": "pass",
        "ranking": "pass",
        "bundle_budget": "pass",
    }
    if status == "unknown":
        checks["target_alignment"] = "unknown"
        checks["ranking"] = "unknown"
    elif status == "fail":
        checks["bundle_budget"] = "fail"
    elif status == "semantic_ambiguity":
        checks["target_alignment"] = "semantic_ambiguity"
    return checks


class GateFixture:
    statuses = ("unknown", "fail", "semantic_ambiguity", "pass")

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw = root / "bundled_shopping_data.jsonl"
        self.domain = root / "domain_data.json"
        self.items = root / "items_shuffle.json"
        self.attributes = root / "items_ins_v2.json"
        self.lucene = root / "original_lucene_index_files.sha256"
        self.lucene_root = root / "indexes-full"
        self.manual = root / "manual_candidate_evidence.json"
        self.chains = root / "chains.jsonl"
        self.summary = root / "summary.json"
        self.repo = root / "MemoryArena"
        self.manifest = root / "annotation_gate.json"
        self.expected_manifest_sha256: str | None = None

        raw_rows = []
        chain_rows = []
        for source_id, status in enumerate(self.statuses):
            target_asins = [asin(source_id, step) for step in range(6)]
            task_id = f"fixture_task_{source_id}"
            raw_rows.append(
                {
                    "id": source_id,
                    "category": task_id,
                    "questions": [f"question {source_id}:{step}" for step in range(6)],
                    "answers": [
                        {"target_asin": value, "attributes": []}
                        for value in target_asins
                    ],
                }
            )
            chain_rows.append(
                {
                    "source_id": source_id,
                    "category": task_id,
                    "status": status,
                    "step_statuses": [status] * 6,
                    "step_check_statuses": [detailed_checks(status) for _ in range(6)],
                    "budget_status": "fail" if status == "fail" else "pass",
                    "target_asins": target_asins,
                }
            )
        write_jsonl(self.raw, raw_rows)
        write_json(self.domain, {"domains": ["fixture"]})
        write_json(self.items, [{"asin": asin(0, 0), "pricing": "$1.00 to $2.00"}])
        write_json(self.attributes, {asin(0, 0): {"attributes": []}})
        self.lucene_root.mkdir()
        (self.lucene_root / "segments_1").write_bytes(b"fixture-lucene-index")
        self.lucene.write_text(
            sha256(self.lucene_root / "segments_1") + "  ./segments_1\n",
            encoding="utf-8",
        )
        write_json(self.manual, {"schema_version": 1, "mappings": []})
        write_jsonl(self.chains, chain_rows)

        self.repo.mkdir()
        (self.repo / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "add", "."],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            cwd=self.repo,
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

        counts = dict(sorted(Counter(self.statuses).items()))
        summary = {
            "schema_version": 2,
            "formal_unit": "one_complete_six_step_chain",
            "chain_count": len(self.statuses),
            "chain_status_counts": counts,
            "proven_correct_chain_count": 1,
            "confirmed_annotation_issue_chain_count": 2,
            "method_guards": {
                "upstream_annotation_audit_uses_amg_sqlite_search": False,
            },
            "inputs": {
                "raw_hf_jsonl": {"sha256": sha256(self.raw)},
                "domain_data": {"sha256": sha256(self.domain)},
                "items_shuffle_sha256": sha256(self.items),
                "items_ins_v2_sha256": sha256(self.attributes),
                "original_lucene_index_manifest": {"sha256": sha256(self.lucene)},
                "manual_evidence": {"sha256": sha256(self.manual)},
                "memoryarena_repo": {"commit": self.commit},
            },
        }
        write_json(self.summary, summary)
        self.trust_root = AnnotationGateTrustRoot(
            raw_dataset_sha256=sha256(self.raw),
            domain_data_sha256=sha256(self.domain),
            items_shuffle_sha256=sha256(self.items),
            items_ins_v2_sha256=sha256(self.attributes),
            lucene_index_manifest_sha256=sha256(self.lucene),
            audit_summary_sha256=sha256(self.summary),
            audit_chains_sha256=sha256(self.chains),
            manual_evidence_sha256=sha256(self.manual),
            memoryarena_base_commit=self.commit,
            price_seed=233,
            chain_status_counts=tuple(counts.items()),
        )
        self.bindings = self.make_bindings()

    def make_bindings(
        self,
        *,
        price_seed: int = 233,
    ) -> AnnotationGateBindings:
        return build_annotation_gate_bindings(
            raw_dataset_path=self.raw,
            domain_data_path=self.domain,
            items_shuffle_path=self.items,
            items_ins_v2_path=self.attributes,
            lucene_index_manifest_path=self.lucene,
            lucene_index_root=self.lucene_root,
            audit_summary_path=self.summary,
            audit_chains_path=self.chains,
            manual_evidence_path=self.manual,
            memoryarena_repo_path=self.repo,
            memoryarena_base_commit=self.commit,
            price_seed=price_seed,
        )

    def build(self, mode: str = "provisional") -> dict:
        return _build_annotation_gate_manifest(
            run_id="fixture-run",
            mode=mode,
            raw_dataset_path=self.raw,
            audit_summary_path=self.summary,
            audit_chains_path=self.chains,
            manual_evidence_path=self.manual,
            bindings=self.bindings,
            requested_task_ids=None,
            trust_root=self.trust_root,
        )

    def validate(
        self,
        selected_task_ids: list[str],
        *,
        mode: str = "provisional",
        expected_manifest_sha256: str | None = None,
        price_seed: int = 233,
    ):
        expected_hash = expected_manifest_sha256 or self.expected_manifest_sha256
        if expected_hash is None:
            raise AssertionError("The fixture manifest must be written before validation.")
        return _validate_annotation_gate_manifest(
            self.manifest,
            expected_mode=mode,
            expected_run_id="fixture-run",
            expected_manifest_sha256=expected_hash,
            selected_task_ids=selected_task_ids,
            raw_dataset_path=self.raw,
            domain_data_path=self.domain,
            items_shuffle_path=self.items,
            items_ins_v2_path=self.attributes,
            lucene_index_manifest_path=self.lucene,
            lucene_index_root=self.lucene_root,
            audit_summary_path=self.summary,
            audit_chains_path=self.chains,
            manual_evidence_path=self.manual,
            memoryarena_repo_path=self.repo,
            memoryarena_base_commit=self.commit,
            price_seed=price_seed,
            trust_root=self.trust_root,
        )

    def write(self, manifest: dict) -> str:
        self.expected_manifest_sha256 = write_annotation_gate_manifest(
            manifest, self.manifest
        )
        return self.expected_manifest_sha256


class AnnotationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.fixture = GateFixture(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_provisional_allows_only_unknown_and_pass_whole_chains(self) -> None:
        manifest = self.fixture.build("provisional")
        manifest_sha256 = self.fixture.write(manifest)

        self.assertEqual(
            manifest["allowed_task_ids"],
            ["fixture_task_0", "fixture_task_3"],
        )
        self.assertEqual(
            manifest["excluded_task_ids"],
            ["fixture_task_1", "fixture_task_2"],
        )
        self.assertEqual(
            manifest["allowed_task_ids_sha256"],
            hash_task_ids(["fixture_task_0", "fixture_task_3"]),
        )
        self.assertFalse(manifest["policy"]["unknown_is_proven_correct"])
        decision = self.fixture.validate(
            ["fixture_task_0", "fixture_task_3"],
            expected_manifest_sha256=manifest_sha256,
        )
        self.assertEqual(decision.allowed_task_ids_sha256, manifest["allowed_task_ids_sha256"])

        with self.assertRaisesRegex(AnnotationGateError, "blocks 1 selected whole chains"):
            self.fixture.validate(["fixture_task_1"])
        with self.assertRaisesRegex(AnnotationGateError, "blocks 1 selected whole chains"):
            self.fixture.validate(["fixture_task_2"])

    def test_strict_allows_pass_only_and_fails_closed_without_one(self) -> None:
        manifest = self.fixture.build("strict")
        self.fixture.write(manifest)

        self.assertEqual(manifest["allowed_task_ids"], ["fixture_task_3"])
        self.fixture.validate(["fixture_task_3"], mode="strict")
        with self.assertRaisesRegex(AnnotationGateError, "blocks 1 selected whole chains"):
            self.fixture.validate(["fixture_task_0"], mode="strict")

        manifest["task_verdicts"][-1]["verdict"] = "unknown"
        manifest["audit"]["chain_status_counts"] = {
            "fail": 1,
            "semantic_ambiguity": 1,
            "unknown": 2,
        }
        manifest["allowed_task_ids"] = []
        manifest["allowed_task_ids_sha256"] = hash_task_ids([])
        write_annotation_gate_manifest(manifest, self.fixture.manifest)
        with self.assertRaisesRegex(AnnotationGateError, "manifest SHA256 mismatch"):
            self.fixture.validate(["fixture_task_3"], mode="strict")

    def test_trust_all_allows_every_verdict_without_relabeling(self) -> None:
        manifest = self.fixture.build("trust_all")
        manifest_sha256 = self.fixture.write(manifest)
        all_task_ids = [f"fixture_task_{index}" for index in range(4)]

        self.assertEqual(
            ANNOTATION_GATE_MODES,
            ("provisional", "strict", "trust_all"),
        )
        self.assertEqual(manifest["allowed_task_ids"], all_task_ids)
        self.assertEqual(manifest["excluded_task_ids"], [])
        self.assertEqual(
            manifest["policy"]["allowed_verdicts"],
            ["fail", "pass", "semantic_ambiguity", "unknown"],
        )
        self.assertEqual(
            [row["verdict"] for row in manifest["task_verdicts"]],
            list(self.fixture.statuses),
        )
        self.assertFalse(manifest["policy"]["unknown_is_proven_correct"])

        decision = self.fixture.validate(
            all_task_ids,
            mode="trust_all",
            expected_manifest_sha256=manifest_sha256,
        )
        self.assertEqual(decision.mode, "trust_all")
        self.assertEqual(decision.allowed_task_ids, tuple(all_task_ids))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            AnnotationGateError,
            "expected one of provisional, strict, trust_all",
        ):
            self.fixture.build("accept_anything")

    def test_manifest_tamper_is_rebuilt_from_canonical_evidence(self) -> None:
        manifest = self.fixture.build()
        self.fixture.write(manifest)
        manifest["allowed_task_ids"].append("fixture_task_1")
        manifest["allowed_task_ids_sha256"] = hash_task_ids(manifest["allowed_task_ids"])
        tampered_sha256 = write_annotation_gate_manifest(manifest, self.fixture.manifest)

        with self.assertRaisesRegex(AnnotationGateError, "manifest SHA256 mismatch"):
            self.fixture.validate(["fixture_task_1"])
        with self.assertRaisesRegex(AnnotationGateError, "does not match"):
            self.fixture.validate(
                ["fixture_task_1"], expected_manifest_sha256=tampered_sha256
            )

    def test_seed_and_price_table_mismatch_fail_closed(self) -> None:
        manifest = self.fixture.build()
        self.fixture.write(manifest)

        with self.assertRaisesRegex(AnnotationGateError, "trust-root mismatch"):
            self.fixture.validate(["fixture_task_0"], price_seed=234)

        manifest["bindings"]["runtime_prices"]["price_table_sha256"] = (
            hashlib.sha256(b"different-prices").hexdigest()
        )
        tampered_sha256 = write_annotation_gate_manifest(
            manifest, self.fixture.manifest
        )
        with self.assertRaisesRegex(AnnotationGateError, "does not match"):
            self.fixture.validate(
                ["fixture_task_0"],
                expected_manifest_sha256=tampered_sha256,
            )

    def test_price_table_is_derived_from_items_and_seed(self) -> None:
        seed_233 = fingerprint_memoryarena_price_table(
            self.fixture.items,
            price_seed=233,
        )
        seed_234 = fingerprint_memoryarena_price_table(
            self.fixture.items,
            price_seed=234,
        )

        self.assertEqual(seed_233.sha256, self.fixture.bindings.price_table_sha256)
        self.assertEqual(
            seed_233.row_count, self.fixture.bindings.price_table_row_count
        )
        self.assertEqual(seed_233.row_count, 1)
        self.assertNotEqual(seed_233.sha256, seed_234.sha256)

    def test_artifact_hash_changes_are_rejected(self) -> None:
        self.fixture.write(self.fixture.build())

        for path in (self.fixture.items, self.fixture.attributes, self.fixture.lucene):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper\n")
                with self.assertRaises(AnnotationGateError):
                    self.fixture.validate(["fixture_task_0"])
                path.write_bytes(original)

        index_path = self.fixture.lucene_root / "segments_1"
        original = index_path.read_bytes()
        index_path.write_bytes(b"changed-index")
        with self.assertRaisesRegex(AnnotationGateError, "Lucene index SHA256 mismatch"):
            self.fixture.validate(["fixture_task_0"])
        index_path.write_bytes(original)

    def test_chain_verdict_tamper_and_manual_positive_evidence_are_rejected(self) -> None:
        self.fixture.write(self.fixture.build())
        original_chains = self.fixture.chains.read_bytes()
        rows = [json.loads(line) for line in original_chains.decode().splitlines()]
        rows[1]["status"] = "unknown"
        write_jsonl(self.fixture.chains, rows)
        with self.assertRaisesRegex(AnnotationGateError, "trust-root mismatch"):
            self.fixture.validate(["fixture_task_0"])
        self.fixture.chains.write_bytes(original_chains)

        write_json(self.fixture.manual, {"schema_version": 1, "mappings": [{"x": 1}]})
        with self.assertRaisesRegex(AnnotationGateError, "trust-root mismatch"):
            self.fixture.validate(["fixture_task_0"])

    def test_resealed_top_level_pass_cannot_override_detailed_failure(self) -> None:
        rows = [json.loads(line) for line in self.fixture.chains.read_text().splitlines()]
        rows[1]["status"] = "pass"
        rows[1]["step_statuses"] = ["pass"] * 6
        rows[1]["budget_status"] = "pass"
        write_jsonl(self.fixture.chains, rows)

        summary = json.loads(self.fixture.summary.read_text())
        summary["chain_status_counts"] = {
            "pass": 2,
            "semantic_ambiguity": 1,
            "unknown": 1,
        }
        summary["proven_correct_chain_count"] = 2
        summary["confirmed_annotation_issue_chain_count"] = 1
        write_json(self.fixture.summary, summary)
        resealed_trust = AnnotationGateTrustRoot(
            raw_dataset_sha256=sha256(self.fixture.raw),
            domain_data_sha256=sha256(self.fixture.domain),
            items_shuffle_sha256=sha256(self.fixture.items),
            items_ins_v2_sha256=sha256(self.fixture.attributes),
            lucene_index_manifest_sha256=sha256(self.fixture.lucene),
            audit_summary_sha256=sha256(self.fixture.summary),
            audit_chains_sha256=sha256(self.fixture.chains),
            manual_evidence_sha256=sha256(self.fixture.manual),
            memoryarena_base_commit=self.fixture.commit,
            price_seed=233,
            chain_status_counts=(
                ("pass", 2),
                ("semantic_ambiguity", 1),
                ("unknown", 1),
            ),
        )
        bindings = self.fixture.make_bindings()

        with self.assertRaisesRegex(AnnotationGateError, "contradict detailed checks"):
            _build_annotation_gate_manifest(
                run_id="reseal-probe",
                mode="strict",
                raw_dataset_path=self.fixture.raw,
                audit_summary_path=self.fixture.summary,
                audit_chains_path=self.fixture.chains,
                manual_evidence_path=self.fixture.manual,
                bindings=bindings,
                requested_task_ids=None,
                trust_root=resealed_trust,
            )

    def test_source_tree_change_and_base_commit_mismatch_are_rejected(self) -> None:
        self.fixture.write(self.fixture.build())
        source = self.fixture.repo / "runtime.py"
        source.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(AnnotationGateError, "does not match"):
            self.fixture.validate(["fixture_task_0"])

        with self.assertRaisesRegex(AnnotationGateError, "base commit mismatch"):
            fingerprint_memoryarena_source_tree(
                self.fixture.repo,
                expected_base_commit="0" * 40,
            )

    def test_requested_ids_are_run_specific_and_hash_protected(self) -> None:
        manifest = _build_annotation_gate_manifest(
            run_id="fixture-run",
            mode="provisional",
            raw_dataset_path=self.fixture.raw,
            audit_summary_path=self.fixture.summary,
            audit_chains_path=self.fixture.chains,
            manual_evidence_path=self.fixture.manual,
            bindings=self.fixture.bindings,
            requested_task_ids=["fixture_task_0", "fixture_task_1"],
            trust_root=self.fixture.trust_root,
        )
        self.fixture.write(manifest)
        self.assertEqual(manifest["allowed_task_ids"], ["fixture_task_0"])
        with self.assertRaisesRegex(AnnotationGateError, "not requested"):
            self.fixture.validate(["fixture_task_3"])

        tampered = copy.deepcopy(manifest)
        tampered["run"]["requested_task_ids_sha256"] = "0" * 64
        write_annotation_gate_manifest(tampered, self.fixture.manifest)
        with self.assertRaisesRegex(AnnotationGateError, "manifest SHA256 mismatch"):
            self.fixture.validate(["fixture_task_0"])


if __name__ == "__main__":
    unittest.main()
