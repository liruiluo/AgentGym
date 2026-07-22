from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.domains.browsecomp import (
    BROWSECOMP_FINAL_MAX_ITERATIONS,
    BROWSECOMP_FROZEN_CORPUS_REPOSITORY,
    BROWSECOMP_FROZEN_CORPUS_REVISION,
    BROWSECOMP_SUBQUERY_MAX_ITERATIONS,
    BROWSECOMP_UPSTREAM_RELATIVE_PATHS,
    BrowseCompPlusFactory,
    attest_browsecomp_search_assets,
    attest_browsecomp_upstream,
    load_browsecomp_tasks,
)
from agentenv_agentmemory.runtime.memory import MemoryRewardPolicy
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


class FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, query: str) -> str:
        self.calls.append(query)
        if query == "explode":
            raise RuntimeError("temporary retrieval failure")
        return json.dumps(
            [
                {
                    "docid": "DOC-1",
                    "score": 0.75,
                    "snippet": f"evidence for {query}",
                }
            ],
            indent=2,
        )


class FakeJudge:
    def __init__(self):
        self.calls = []
        self.failure = None

    def __call__(self, question: str, predicted: str, correct: str):
        self.calls.append((question, predicted, correct))
        if self.failure is not None:
            raise self.failure
        passed = predicted == correct
        return {
            "correct": passed,
            "confidence": 100.0 if passed else 0.0,
            "reasoning": "private judge rationale",
            "extracted_final_answer": predicted,
            "parse_error": False,
        }


class BrowseCompDomainV3Test(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.ground_truth = root / "browsecomp_plus_decrypted.jsonl"
        self.decomposition = root / "browsecomp_all_jsons.jsonl"
        self.ground_truth.write_text(
            json.dumps(
                {
                    "query_id": "116",
                    "query": "final combined question",
                    "answer": "SECRET_FINAL",
                    "evidence_docs": ["DOC-1"],
                    "gold_docs": ["DOC-1"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.decomposition.write_text(
            json.dumps(
                {
                    "id": "116",
                    "question": [
                        "first research question",
                        "second research question",
                        "decomposition copy of final question",
                    ],
                    "answer": ["PRIVATE_SUB_1", "PRIVATE_SUB_2", "PRIVATE_FINAL"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.search = FakeSearch()
        self.judge = FakeJudge()
        factory = BrowseCompPlusFactory(
            ground_truth_path=self.ground_truth,
            decomposition_path=self.decomposition,
            search_tool=self.search,
            judge=self.judge,
        )
        self.wrapper = DomainEnvWrapper(
            factory,
            reward_policy=MemoryRewardPolicy(),
        )
        self.created = self.wrapper.create()
        self.env_id = self.created["id"]

    def tearDown(self):
        if self.env_id in self.wrapper.envs:
            self.wrapper.close(self.env_id)
        self.tempdir.cleanup()

    def test_reset_exposes_first_subquery_without_private_answers(self):
        observation = self.created["observation"]
        self.assertIn("first research question", observation)
        self.assertNotIn("final combined question", observation)
        self.assertNotIn("SECRET_FINAL", observation)
        self.assertNotIn("PRIVATE_SUB_1", observation)
        self.assertNotIn("SECRET_FINAL", repr(self.wrapper.metadata()))
        self.assertEqual(self.created["info"]["phase_count"], 3)
        self.assertEqual(
            self.wrapper.metadata()["intermediate_submission_reward"],
            0.0,
        )
        self.assertEqual(self.wrapper.metadata()["max_steps"], 100)
        prompt = self.wrapper.metadata()["system_prompt"]
        for action in self.wrapper.metadata()["native_action_descriptions"]:
            self.assertEqual(prompt.count(action), 1)
            self.assertNotIn(action, observation)

    def test_reset_rejects_out_of_range_data_index_instead_of_wrapping(self):
        with self.assertRaisesRegex(IndexError, "outside"):
            self.wrapper.reset(self.env_id, data_idx=1)

    def test_lowercase_native_search_is_one_zero_reward_transition(self):
        stepped = self.wrapper.step(
            self.env_id,
            'Action: search {"query": "target facts"}',
        )
        self.assertEqual(stepped["reward"], 0.0)
        self.assertFalse(stepped["done"])
        self.assertEqual(stepped["info"]["phase_index"], 0)
        self.assertEqual(stepped["info"]["action_execution"]["op"], "search")
        self.assertEqual(self.search.calls, ["target facts"])
        self.assertIn("DOC-1", stepped["observation"])
        self.assertEqual(
            stepped["info"]["tool_ops"][0]["retrieved_docids"],
            ["DOC-1"],
        )
        self.assertEqual(
            stepped["info"]["formal_schema_version"],
            "agentmemory_formal_step_v3",
        )
        self.assertEqual(
            stepped["info"]["domain_evidence"]["native_iteration_count"],
            1,
        )
        for action in self.wrapper.metadata()["native_action_descriptions"]:
            self.assertNotIn(action, stepped["observation"])

    def test_phase_budget_counts_native_turns_but_not_memory_actions(self):
        for index in range(5):
            memory = self.wrapper.step(
                self.env_id,
                f'Action: ADD {{"key": "k{index}", "value": "v{index}"}}',
            )
            self.assertEqual(memory["info"]["phase_index"], 0)

        invalid = self.wrapper.step(self.env_id, "Action: malformed-native-action")
        self.assertEqual(
            invalid["info"]["domain_evidence"]["native_iteration_count"],
            1,
        )
        for index in range(BROWSECOMP_SUBQUERY_MAX_ITERATIONS - 2):
            searched = self.wrapper.step(
                self.env_id,
                f'Action: search {{"query": "query {index}"}}',
            )
        self.assertEqual(searched["info"]["phase_index"], 0)
        self.assertEqual(
            searched["info"]["domain_evidence"]["native_iteration_count"],
            BROWSECOMP_SUBQUERY_MAX_ITERATIONS - 1,
        )

        exhausted = self.wrapper.step(
            self.env_id,
            'Action: search {"query": "last allowed query"}',
        )
        self.assertEqual(exhausted["reward"], 0.0)
        self.assertFalse(exhausted["done"])
        self.assertEqual(exhausted["info"]["phase_index"], 1)
        self.assertTrue(exhausted["info"]["action_execution"]["phase_budget_exhausted"])
        self.assertEqual(
            exhausted["info"]["tool_ops"][-1]["op"],
            "PHASE_BUDGET_EXHAUSTED",
        )
        self.assertEqual(
            exhausted["info"]["domain_evidence"]["native_iteration_count"],
            BROWSECOMP_SUBQUERY_MAX_ITERATIONS,
        )
        self.assertIn("second research question", exhausted["observation"])
        self.assertEqual(self.judge.calls, [])

    def test_final_budget_exhaustion_without_answer_ends_incorrect(self):
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft one"}',
        )
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft two"}',
        )
        for index in range(BROWSECOMP_FINAL_MAX_ITERATIONS - 1):
            searched = self.wrapper.step(
                self.env_id,
                f'Action: search {{"query": "final query {index}"}}',
            )
        self.assertFalse(searched["done"])
        self.assertEqual(
            searched["info"]["domain_evidence"]["native_iteration_count"],
            BROWSECOMP_FINAL_MAX_ITERATIONS - 1,
        )

        exhausted = self.wrapper.step(
            self.env_id,
            'Action: search {"query": "last final query"}',
        )
        self.assertEqual(exhausted["reward"], 0.0)
        self.assertTrue(exhausted["done"])
        self.assertFalse(exhausted["info"]["episode_success"])
        self.assertEqual(exhausted["info"]["status"], "completed_incorrect")
        self.assertEqual(exhausted["info"]["phase_index"], 3)
        self.assertFalse(
            exhausted["info"]["domain_evidence"]["extractable_answer"]
        )
        self.assertEqual(self.judge.calls, [])

    def test_search_case_and_strict_payload_match_native_tool_contract(self):
        uppercase = self.wrapper.step(
            self.env_id,
            'Action: SEARCH {"query": "target facts"}',
        )
        extra = self.wrapper.step(
            self.env_id,
            'Action: search {"query": "target facts", "k": 100}',
        )
        self.assertEqual(uppercase["info"]["action_execution"]["op"], "INVALID")
        self.assertEqual(extra["info"]["action_execution"]["op"], "INVALID")
        self.assertEqual(self.search.calls, [])

    def test_search_failure_is_a_nonterminal_tool_message_like_upstream(self):
        stepped = self.wrapper.step(
            self.env_id,
            'Action: search {"query": "explode"}',
        )
        self.assertEqual(stepped["reward"], 0.0)
        self.assertFalse(stepped["done"])
        self.assertFalse(stepped["info"]["sample_excluded"])
        self.assertEqual(stepped["info"]["action_execution"]["status"], "error")
        self.assertIn("Error executing search", stepped["observation"])

    def test_subquery_submission_advances_unjudged_with_zero_reward(self):
        searched = self.wrapper.step(
            self.env_id,
            'Action: search {"query": "target facts"}',
        )
        self.assertIn("DOC-1", searched["observation"])
        submitted = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft one"}',
        )
        self.assertEqual(submitted["reward"], 0.0)
        self.assertFalse(submitted["done"])
        self.assertEqual(submitted["info"]["phase_index"], 1)
        self.assertEqual(
            submitted["info"]["action_execution"]["status"],
            "committed_unjudged",
        )
        self.assertEqual(self.judge.calls, [])
        self.assertIn("second research question", submitted["observation"])
        self.assertNotIn("evidence for target facts", submitted["observation"])

    def test_only_final_submission_calls_original_judge_and_rewards_success(self):
        first = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft one"}',
        )
        second = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft two"}',
        )
        final = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "SECRET_FINAL"}',
        )
        self.assertEqual(first["reward"], 0.0)
        self.assertEqual(second["reward"], 0.0)
        self.assertEqual(final["reward"], 1.0)
        self.assertTrue(final["done"])
        self.assertTrue(final["info"]["episode_success"])
        self.assertEqual(final["info"]["phase_index"], 3)
        self.assertEqual(
            self.judge.calls,
            [("final combined question", "SECRET_FINAL", "SECRET_FINAL")],
        )
        self.assertNotIn("SECRET_FINAL", final["observation"])

    def test_incorrect_final_is_terminal_zero_reward_without_answer_leak(self):
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft one"}',
        )
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft two"}',
        )
        final = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "wrong"}',
        )
        self.assertEqual(final["reward"], 0.0)
        self.assertTrue(final["done"])
        self.assertFalse(final["info"]["episode_success"])
        self.assertNotIn("SECRET_FINAL", final["observation"])
        self.assertNotIn("SECRET_FINAL", repr(final["info"]))

    def test_judge_failure_is_excluded_infrastructure_failure(self):
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft one"}',
        )
        self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "draft two"}',
        )
        self.judge.failure = RuntimeError("judge unavailable")
        final = self.wrapper.step(
            self.env_id,
            'Action: SUBMIT_ANSWER {"answer": "candidate"}',
        )
        self.assertEqual(final["reward"], 0.0)
        self.assertTrue(final["done"])
        self.assertTrue(final["info"]["sample_excluded"])
        self.assertEqual(final["info"]["status"], "infra_error")


class BrowseCompLoaderTest(unittest.TestCase):
    def test_progressive_search_schema_is_ground_truth_and_decomposition(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "progressive_search.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": 7,
                                "questions": ["subquery one", "subquery two", "final"],
                                "answers": ["private one", "private two", "final answer"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": 8,
                                "questions": ["only final"],
                                "answers": ["another answer"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tasks = load_browsecomp_tasks(source, source)
            self.assertEqual([task.query_id for task in tasks], ["7", "8"])
            self.assertEqual(
                [(phase.kind, phase.query) for phase in tasks[0].phases],
                [
                    ("subquery", "subquery one"),
                    ("subquery", "subquery two"),
                    ("final", "final"),
                ],
            )
            self.assertEqual(tasks[0].final_answer, "final answer")
            self.assertEqual(
                [(phase.kind, phase.query) for phase in tasks[1].phases],
                [("final", "only final")],
            )

    def test_progressive_source_metadata_records_single_source_provenance(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "progressive_search.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": 7,
                        "questions": ["subquery", "final"],
                        "answers": ["private", "final answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            factory = BrowseCompPlusFactory(
                ground_truth_path=source,
                search_tool=lambda query: "[]",
                judge=lambda question, predicted, correct: {"correct": False},
            )
            metadata = factory.metadata()
            self.assertEqual(metadata["dataset_schema"], "progressive_search")
            self.assertEqual(metadata["decomposition_mode"], "progressive_search_direct")
            self.assertEqual(metadata["dataset_sha256"], metadata["decomposition_sha256"])

    def test_progressive_search_schema_fails_closed_on_misaligned_lists(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "progressive_search.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": 7,
                        "questions": ["subquery", "final"],
                        "answers": ["only one answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "misaligned"):
                load_browsecomp_tasks(source)

    def test_progressive_search_schema_rejects_mixed_legacy_rows(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "mixed.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": 7,
                                "questions": ["final"],
                                "answers": ["answer"],
                            }
                        ),
                        json.dumps(
                            {
                                "query_id": "8",
                                "query": "final",
                                "answer": "answer",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mixes progressive"):
                load_browsecomp_tasks(source)

    def test_missing_decomposition_replays_run_search_original_query_fallback(self):
        with tempfile.TemporaryDirectory() as tempdir:
            ground_truth = Path(tempdir) / "ground_truth.jsonl"
            ground_truth.write_text(
                json.dumps(
                    {
                        "query_id": 5,
                        "query": "original query",
                        "answer": "private answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tasks = load_browsecomp_tasks(
                ground_truth,
                Path(tempdir) / "missing_decomposition.jsonl",
            )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(
            [(phase.kind, phase.query) for phase in tasks[0].phases],
            [
                ("subquery", "original query"),
                ("final", "original query"),
            ],
        )

    def test_duplicate_decomposition_id_matches_native_first_row_semantics(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ground_truth = root / "ground_truth.jsonl"
            decomposition = root / "decomposition.jsonl"
            ground_truth.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "final",
                        "answer": "private",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decomposition.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "q1",
                                "question": ["first row", "final"],
                                "answer": ["a", "b"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "q1",
                                "question": ["second row", "final"],
                                "answer": ["a", "b"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tasks = load_browsecomp_tasks(ground_truth, decomposition)
        self.assertEqual(tasks[0].phases[0].query, "first row")

    @unittest.skipUnless(
        os.environ.get("MEMORYARENA_BROWSECOMP_REAL_DECOMPOSITION"),
        "real upstream decomposition path is not configured",
    )
    def test_real_upstream_hf_decomposition_replay(self):
        source = Path(os.environ["MEMORYARENA_BROWSECOMP_REAL_DECOMPOSITION"])
        expected_sha256 = "6f6b1f6c40ae37196e23fe4053747568ed2031bffd3da3733748f99c6631b46f"
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_sha256,
        )
        selected = None
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("id")) == "116":
                    selected = row
                    break
        self.assertIsNotNone(selected)
        assert selected is not None
        with tempfile.TemporaryDirectory() as tempdir:
            ground_truth = Path(tempdir) / "browsecomp_plus_decrypted.jsonl"
            ground_truth.write_text(
                json.dumps(
                    {
                        "query_id": selected["id"],
                        "query": selected["question"][-1],
                        "answer": selected["answer"][-1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tasks = load_browsecomp_tasks(ground_truth, source)
        self.assertEqual(len(tasks[0].phases), len(selected["question"]))
        self.assertEqual(
            [phase.query for phase in tasks[0].phases[:-1]],
            selected["question"][:-1],
        )
        self.assertEqual(tasks[0].phases[-1].query, selected["question"][-1])


class BrowseCompSearchAssetAttestationTest(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path):
        index_path = root / "shard0.index"
        id_map_path = root / "shard0_id_map.json"
        corpus_path = root / "corpus.jsonl"
        manifest_path = root / "corpus.manifest.json"
        index_path.write_bytes(b"fixture-index")
        id_map_path.write_text(
            json.dumps({"ids": ["0", "1"]}) + "\n",
            encoding="utf-8",
        )
        corpus_path.write_text(
            json.dumps({"docid": "0", "text": "zero"})
            + "\n"
            + json.dumps({"docid": "1", "text": "one"})
            + "\n",
            encoding="utf-8",
        )
        source_sha = "a" * 64
        manifest = {
            "format": "agentmemory_browsecomp_corpus_manifest_v1",
            "source": {
                "input_glob": "/frozen/*.parquet",
                "repository": BROWSECOMP_FROZEN_CORPUS_REPOSITORY,
                "revision": BROWSECOMP_FROZEN_CORPUS_REVISION,
                "file_count": 1,
                "files": [{"sha256": source_sha}],
                "columns": ["docid", "text", "url"],
            },
            "projection": {"output_columns": ["docid", "text"]},
            "output": {
                "path": str(corpus_path.resolve()),
                "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "row_count": 2,
            },
        }
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        kwargs = {
            "index_pattern": str(root / "shard*.index"),
            "corpus_path": corpus_path,
            "corpus_manifest_path": manifest_path,
            "embedding_model": "text-embedding-3-small",
            "expected_index_shards": (
                {
                    "name": "shard0.index",
                    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    "id_map_sha256": hashlib.sha256(id_map_path.read_bytes()).hexdigest(),
                    "vector_count": 2,
                },
            ),
            "expected_corpus_shards": (source_sha,),
            "expected_document_count": 2,
        }
        return manifest, manifest_path, kwargs

    def test_manifest_binds_frozen_repository_revision_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _, _, kwargs = self._fixture(Path(tempdir))
            evidence = attest_browsecomp_search_assets(**kwargs)

        self.assertEqual(
            evidence["corpus_repository"],
            BROWSECOMP_FROZEN_CORPUS_REPOSITORY,
        )
        self.assertEqual(
            evidence["corpus_revision"],
            BROWSECOMP_FROZEN_CORPUS_REVISION,
        )
        self.assertEqual(evidence["document_count"], 2)

    def test_malformed_manifest_fields_fail_closed_with_runtime_errors(self):
        cases = (
            ("top_level", "JSON object"),
            ("repository", "repository/revision"),
            ("source_file", "source files"),
            ("row_count", "row count"),
        )
        for case, message in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tempdir:
                manifest, manifest_path, kwargs = self._fixture(Path(tempdir))
                if case == "top_level":
                    payload = []
                else:
                    payload = manifest
                    if case == "repository":
                        payload["source"]["repository"] = "unbound/repository"
                    elif case == "source_file":
                        payload["source"]["files"] = ["not-an-object"]
                    else:
                        payload["output"]["row_count"] = "2"
                manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, message):
                    attest_browsecomp_search_assets(**kwargs)


class BrowseCompUpstreamAttestationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for relative_path in BROWSECOMP_UPSTREAM_RELATIVE_PATHS:
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# pristine {relative_path}\n", encoding="utf-8")
        self._git("init")
        self._git("config", "user.email", "agentmemory-test@example.invalid")
        self._git("config", "user.name", "AgentMemory Test")
        self._git("add", ".")
        self._git("commit", "-m", "fixture")
        self.commit = self._git("rev-parse", "HEAD").strip()

    def tearDown(self):
        self.tempdir.cleanup()

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_attests_exact_commit_and_source_bundle(self):
        evidence = attest_browsecomp_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)
        self.assertEqual(
            set(evidence["source_files_sha256"]),
            set(BROWSECOMP_UPSTREAM_RELATIVE_PATHS),
        )
        self.assertEqual(len(evidence["source_bundle_sha256"]), 64)

    def test_rejects_modified_browsecomp_source(self):
        path = self.root / BROWSECOMP_UPSTREAM_RELATIVE_PATHS[0]
        path.write_text("# changed semantics\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_browsecomp_upstream(self.root, expected_commit=self.commit)


if __name__ == "__main__":
    unittest.main()
