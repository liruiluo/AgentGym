from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.domains.browsecomp import (
    BROWSECOMP_FROZEN_CORPUS_REPOSITORY,
    BROWSECOMP_FROZEN_CORPUS_REVISION,
    BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
    BROWSECOMP_FROZEN_SEARCH_K,
    BROWSECOMP_FROZEN_SNIPPET_TOKENS,
    BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE,
    BROWSECOMP_OPENROUTER_ENDPOINT,
    BROWSECOMP_SURFACES,
    BROWSECOMP_UPSTREAM_REQUIRED_FILES,
    BrowseCompPlusFactory,
    _import_upstream_search_client_without_api_key,
    _judge_provenance,
    _load_frozen_snippet_tokenizer,
    aggregate_browsecomp_paper_metrics,
    attest_browsecomp_cross_source_parity,
    attest_browsecomp_search_assets,
    attest_browsecomp_upstream,
    attest_frozen_snippet_tokenizer,
    load_browsecomp_tasks,
    validate_loaded_browsecomp_searcher,
)
from agentenv_agentmemory.domains.memoryarena_dataset import (
    attest_frozen_memoryarena_dataset,
    attest_injected_test_dataset,
)
from agentenv_agentmemory.runtime.memory import MemoryRewardPolicy
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


class FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, op: str, arguments: dict[str, str]) -> str:
        self.calls.append((op, dict(arguments)))
        if arguments.get("query") == "explode":
            raise RuntimeError("temporary retrieval failure")
        if op == "search":
            return json.dumps(
                [
                    {
                        "docid": "DOC-1",
                        "score": 0.75,
                        "snippet": f"evidence for {arguments['query']}",
                    }
                ]
            )
        return json.dumps({"docid": arguments["docid"], "text": "complete document"})


class FakeJudge:
    def __init__(self):
        self.calls = []
        self.failure: Exception | None = None
        self.parse_error = False

    def __call__(self, question: str, predicted: str, correct: str):
        self.calls.append((question, predicted, correct))
        if self.failure is not None:
            raise self.failure
        return {
            "correct": predicted == correct,
            "confidence": 100.0 if predicted == correct else 0.0,
            "reasoning": "private rationale",
            "extracted_final_answer": predicted,
            "parse_error": self.parse_error,
        }


def write_tasks(path: Path, rows=None) -> None:
    if rows is None:
        rows = [
            {
                "id": 7,
                "questions": ["phase one", "phase two", "final question"],
                "answers": ["ANSWER_ONE", "ANSWER_TWO", "FINAL_ANSWER"],
            }
        ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BrowseCompContractTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tasks_path = Path(self.tempdir.name) / "progressive_search.jsonl"
        write_tasks(self.tasks_path)
        self.provenance = attest_injected_test_dataset(
            self.tasks_path,
            config="progressive_search",
        )
        self.search = FakeSearch()
        self.judge = FakeJudge()
        self.wrappers = []

    def tearDown(self):
        for wrapper, env_id in self.wrappers:
            if env_id in wrapper.envs:
                wrapper.close(env_id)
        self.tempdir.cleanup()

    def _create(self, mode: str):
        factory = BrowseCompPlusFactory(
            contract_mode=mode,
            tasks_path=self.tasks_path,
            dataset_provenance=self.provenance,
            search_tool=self.search,
            judge=self.judge,
            test_mode=True,
        )
        wrapper = DomainEnvWrapper(factory, reward_policy=MemoryRewardPolicy())
        created = wrapper.create()
        self.wrappers.append((wrapper, created["id"]))
        return factory, wrapper, created

    def test_upstream_client_import_cannot_print_openai_api_key(self):
        secret = "unit-test-secret-must-not-appear"
        imported = object()

        def fake_import(name):
            print("ENV OPENAI_API_KEY:", repr(os.getenv("OPENAI_API_KEY")))
            return imported

        output = io.StringIO()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": secret}, clear=True),
            patch(
                "agentenv_agentmemory.domains.browsecomp.importlib.import_module",
                side_effect=fake_import,
            ),
            patch.object(sys, "stdout", output),
        ):
            self.assertIs(
                _import_upstream_search_client_without_api_key(),
                imported,
            )
            self.assertEqual(os.environ["OPENAI_API_KEY"], secret)

        self.assertNotIn(secret, output.getvalue())
        self.assertIn("None", output.getvalue())

    def test_metadata_names_public_panel_and_exact_search_contract(self):
        self.assertEqual(
            BROWSECOMP_SURFACES,
            {
                "paper_eval": (
                    "memoryarena_progressive_search_paper_eval_public221_"
                    "one_action_v3"
                ),
                "failfast": (
                    "memoryarena_progressive_search_failfast_public221_"
                    "one_action_v3"
                ),
            },
        )
        for mode, surface in BROWSECOMP_SURFACES.items():
            with self.subTest(mode=mode):
                factory, _, _ = self._create(mode)
                metadata = factory.metadata()
                self.assertIn("public221", surface)
                self.assertTrue(surface.endswith("one_action_v3"))
                self.assertEqual(metadata["contract_mode"], mode)
                self.assertEqual(
                    metadata["action_granularity"]["policy_actions_per_turn"],
                    1,
                )
                self.assertFalse(
                    metadata["action_granularity"][
                        "upstream_batched_model_turn_parity"
                    ]
                )
                self.assertEqual(
                    metadata["max_total_actions"], factory.contract.max_steps
                )
                self.assertEqual(
                    metadata["total_action_budget"]["counts"],
                    ["native", "memory", "invalid"],
                )
                self.assertEqual(
                    metadata["total_action_budget"][
                        "memory_action_allowance_per_phase"
                    ],
                    BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE,
                )
                self.assertEqual(
                    metadata["total_action_budget"]["native_action_allowance"],
                    100,
                )
                self.assertEqual(
                    metadata["total_action_budget"]["memory_action_allowance"],
                    3 * BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE,
                )
                self.assertEqual(factory.contract.max_steps, 148)
                self.assertIn(
                    "exact global cap for this dataset is 148",
                    factory.contract.canonical_system_prompt,
                )
                self.assertEqual(
                    metadata["native_iteration_budget"]["subquery_per_phase"],
                    35,
                )
                self.assertEqual(
                    metadata["native_iteration_budget"]["final_phase"],
                    30,
                )
                self.assertEqual(
                    metadata["native_iteration_budget"]["counts"],
                    ["native", "invalid"],
                )
                self.assertFalse(
                    metadata["native_iteration_budget"]
                    ["memory_actions_consume_budget"]
                )
                self.assertIn(
                    "one-action adapter", factory.contract.canonical_system_prompt
                )
                self.assertEqual(
                    metadata["native_tool_ops"], ["search", "get_document"]
                )
                self.assertEqual(
                    metadata["native_search_k"], BROWSECOMP_FROZEN_SEARCH_K
                )
                self.assertEqual(
                    metadata["native_snippet_max_tokens"],
                    BROWSECOMP_FROZEN_SNIPPET_TOKENS,
                )
                self.assertFalse(metadata["release_scope"]["paper_panel_complete"])
                self.assertEqual(metadata["release_scope"]["public_task_count"], 221)
                self.assertEqual(metadata["release_scope"]["paper_task_count"], 256)
                self.assertEqual(
                    metadata["release_scope"]["missing_private_task_count"], 35
                )
                self.assertEqual(
                    metadata["paper_evaluation"]["id"],
                    "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1",
                )
                self.assertEqual(
                    metadata["paper_evaluation"]["available"],
                    mode == "paper_eval",
                )

    def test_embedding_route_is_hashed_and_paper_eval_rejects_openrouter(self):
        failfast = BrowseCompPlusFactory(
            contract_mode="failfast",
            tasks_path=self.tasks_path,
            dataset_provenance=self.provenance,
            embedding_model="text-embedding-3-small",
            provider="openrouter",
            embedding_endpoint=BROWSECOMP_OPENROUTER_ENDPOINT,
            search_tool=self.search,
            judge=self.judge,
            test_mode=True,
        )
        route = failfast.metadata()["embedding_route_provenance"]
        self.assertEqual(route["provider"], "openrouter")
        self.assertEqual(route["model"], "text-embedding-3-small")
        self.assertEqual(
            route["route_variant"],
            "failfast_openrouter_nonpaper_embedding_v1",
        )
        self.assertEqual(len(route["endpoint_sha256"]), 64)
        self.assertEqual(len(route["config_sha256"]), 64)
        self.assertNotIn("openrouter.ai", repr(route))

        with self.assertRaisesRegex(RuntimeError, "requires the OpenAI"):
            BrowseCompPlusFactory(
                contract_mode="paper_eval",
                tasks_path=self.tasks_path,
                dataset_provenance=self.provenance,
                provider="openrouter",
                embedding_endpoint=BROWSECOMP_OPENROUTER_ENDPOINT,
                search_tool=self.search,
                judge=self.judge,
                test_mode=True,
            )

    def test_memory_action_does_not_consume_longest_task_native_quota(self):
        questions = [f"phase {index}" for index in range(16)]
        answers = [f"ANSWER_{index}" for index in range(16)]
        write_tasks(
            self.tasks_path,
            rows=[{"id": 0, "questions": questions, "answers": answers}],
        )
        provenance = attest_injected_test_dataset(
            self.tasks_path,
            config="progressive_search",
        )
        factory = BrowseCompPlusFactory(
            contract_mode="paper_eval",
            tasks_path=self.tasks_path,
            dataset_provenance=provenance,
            search_tool=self.search,
            judge=self.judge,
            test_mode=True,
        )
        wrapper = DomainEnvWrapper(factory, reward_policy=MemoryRewardPolicy())
        created = wrapper.create()
        env_id = created["id"]
        self.wrappers.append((wrapper, env_id))
        wrapper.reset(env_id, 0)

        add = wrapper.step(env_id, 'Action: ADD {"key":"plan","value":"seed"}')
        self.assertFalse(add["done"])
        for phase_index, answer in enumerate(answers):
            native_limit = 30 if phase_index == len(answers) - 1 else 35
            for _ in range(native_limit - 1):
                transition = wrapper.step(
                    env_id,
                    'Action: search {"query":"evidence"}',
                )
                self.assertFalse(transition["done"])
            transition = wrapper.step(
                env_id,
                f'Action: SUBMIT_ANSWER {{"answer":"{answer}"}}',
            )
        self.assertTrue(transition["done"])
        self.assertTrue(transition["info"]["episode_success"])
        self.assertEqual(factory.contract.max_steps, 811)

    def test_judge_provenance_hashes_endpoint_and_complete_public_config(self):
        provenance = _judge_provenance(
            {
                "backend": "openai_responses",
                "model_name": "judge-model",
                "base_url": "https://secret-route.example/v1",
                "max_tokens": 4321,
            },
            mode="upstream_memoryarena_judge",
        )
        self.assertEqual(provenance["backend"], "openai_responses")
        self.assertEqual(provenance["model"], "judge-model")
        self.assertEqual(provenance["max_tokens"], 4321)
        self.assertEqual(len(provenance["endpoint_sha256"]), 64)
        self.assertEqual(len(provenance["prompt_template_sha256"]), 64)
        self.assertEqual(len(provenance["config_sha256"]), 64)
        self.assertNotIn("secret-route", repr(provenance))

    def test_paper_eval_judges_every_phase_continues_and_emits_metrics(self):
        _, wrapper, created = self._create("paper_eval")
        env_id = created["id"]
        first = wrapper.step(
            env_id,
            'Action: SUBMIT_ANSWER {"answer": "WRONG"}',
        )
        self.assertEqual(first["reward"], 0.0)
        self.assertFalse(first["done"])
        self.assertEqual(first["info"]["phase_index"], 1)
        self.assertIn("phase two", first["observation"])
        second = wrapper.step(
            env_id,
            'Action: SUBMIT_ANSWER {"answer": "ANSWER_TWO"}',
        )
        final = wrapper.step(
            env_id,
            'Action: SUBMIT_ANSWER {"answer": "FINAL_ANSWER"}',
        )
        self.assertEqual(second["reward"], 0.0)
        self.assertEqual(final["reward"], 0.0)
        self.assertTrue(final["done"])
        self.assertTrue(final["info"]["episode_success"])
        self.assertEqual(len(self.judge.calls), 3)
        paper = final["info"]["domain_evidence"]["paper_evaluation"]
        self.assertEqual(
            [item["correct"] for item in paper["phase_verdicts"]],
            [False, True, True],
        )
        self.assertEqual(paper["process_score_numerator"], 2)
        self.assertEqual(paper["process_score_denominator"], 3)
        self.assertEqual(paper["process_score"], 2 / 3)
        self.assertEqual(
            paper["sr_at_k"],
            {
                "1": {"correct": False, "numerator": 0, "denominator": 1},
                "2": {"correct": True, "numerator": 1, "denominator": 1},
                "3": {"correct": True, "numerator": 1, "denominator": 1},
            },
        )
        self.assertEqual(paper["final_sr_numerator"], 1)
        self.assertEqual(paper["final_sr_denominator"], 1)
        self.assertTrue(paper["final_success"])
        self.assertTrue(paper["complete"])
        ledger = final["info"]["domain_evidence"]["phase_verdict_ledger"]
        self.assertEqual(
            set(ledger[0]),
            {
                "phase_index",
                "phase_kind",
                "correct",
                "verdict_source",
                "answer_sha256",
                "judge_response_sha256",
                "judge_confidence",
                "judge_parse_error",
                "retrieved_docids",
            },
        )
        self.assertNotIn("ANSWER_ONE", repr(final))
        self.assertNotIn("private rationale", repr(final))

    def test_paper_metric_aggregation_is_task_macro_and_depth_specific(self):
        snapshots = [
            {
                "metric_contract": "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1",
                "dataset_scope": "public221_of_paper256",
                "metric_scale": "unit_interval",
                "complete": True,
                "process_score_denominator": 3,
                "phase_verdicts": [
                    {"phase_index": 0, "correct": False},
                    {"phase_index": 1, "correct": True},
                    {"phase_index": 2, "correct": True},
                ],
            },
            {
                "metric_contract": "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1",
                "dataset_scope": "public221_of_paper256",
                "metric_scale": "unit_interval",
                "complete": True,
                "process_score_denominator": 2,
                "phase_verdicts": [
                    {"phase_index": 0, "correct": True},
                    {"phase_index": 1, "correct": False},
                ],
            },
        ]
        metrics = aggregate_browsecomp_paper_metrics(snapshots)
        self.assertEqual(metrics["task_count"], 2)
        self.assertEqual(metrics["process_score"], ((2 / 3) + (1 / 2)) / 2)
        self.assertEqual(
            metrics["sr_at_k"]["1"],
            {"correct_tasks": 1, "eligible_tasks": 2, "rate": 0.5},
        )
        self.assertEqual(
            metrics["sr_at_k"]["3"],
            {"correct_tasks": 1, "eligible_tasks": 1, "rate": 1.0},
        )
        self.assertEqual(metrics["final_success_rate"], 0.5)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aggregate_browsecomp_paper_metrics(
                [
                    {
                        "complete": False,
                        "process_score_denominator": 1,
                        "phase_verdicts": [],
                    }
                ]
            )

    def test_failfast_retains_earlier_return_and_stops_on_wrong_phase(self):
        _, wrapper, created = self._create("failfast")
        env_id = created["id"]
        first = wrapper.step(
            env_id,
            'Action: SUBMIT_ANSWER {"answer": "ANSWER_ONE"}',
        )
        wrong = wrapper.step(
            env_id,
            'Action: SUBMIT_ANSWER {"answer": "WRONG"}',
        )
        self.assertEqual(first["reward"], 1.0)
        self.assertFalse(first["done"])
        self.assertEqual(wrong["reward"], 0.0)
        self.assertTrue(wrong["done"])
        self.assertEqual(wrong["info"]["phase_index"], 1)
        self.assertFalse(wrong["info"]["episode_success"])
        self.assertEqual(wrong["info"]["status"], "failed_on_incorrect_answer")
        self.assertEqual(
            [
                item["correct"]
                for item in wrong["info"]["domain_evidence"]["phase_verdict_ledger"]
            ],
            [True, False],
        )
        self.assertEqual(first["reward"] + wrong["reward"], 1.0)

    def test_search_and_get_document_are_strict_zero_reward_transitions(self):
        _, wrapper, created = self._create("paper_eval")
        env_id = created["id"]
        searched = wrapper.step(
            env_id,
            'Action: search {"query": "target facts"}',
        )
        document = wrapper.step(
            env_id,
            'Action: get_document {"docid": "DOC-1"}',
        )
        self.assertEqual(searched["reward"], 0.0)
        self.assertEqual(document["reward"], 0.0)
        self.assertEqual(
            self.search.calls,
            [
                ("search", {"query": "target facts"}),
                ("get_document", {"docid": "DOC-1"}),
            ],
        )
        self.assertIn("complete document", document["observation"])
        self.assertEqual(
            document["info"]["domain_evidence"]["retrieved_docids"],
            ["DOC-1"],
        )
        uppercase = wrapper.step(
            env_id,
            'Action: SEARCH {"query": "target facts"}',
        )
        extra = wrapper.step(
            env_id,
            'Action: search {"query": "target facts", "k": 10}',
        )
        self.assertEqual(uppercase["info"]["action_execution"]["op"], "INVALID")
        self.assertEqual(extra["info"]["action_execution"]["op"], "INVALID")

    def test_judge_exception_and_parse_failure_are_excluded(self):
        for mode in BROWSECOMP_SURFACES:
            with self.subTest(mode=mode):
                _, wrapper, created = self._create(mode)
                self.judge.parse_error = True
                failed = wrapper.step(
                    created["id"],
                    'Action: SUBMIT_ANSWER {"answer": "candidate"}',
                )
                self.assertTrue(failed["done"])
                self.assertTrue(failed["info"]["sample_excluded"])
                self.assertEqual(
                    failed["info"]["status"], "infrastructure_error"
                )
                self.assertNotIn(
                    "phase_verdict_ledger", failed["info"]["domain_evidence"]
                )
                self.judge.parse_error = False

    def test_search_failure_is_terminal_excluded_without_phase_verdict(self):
        _, wrapper, created = self._create("paper_eval")
        failed = wrapper.step(
            created["id"],
            'Action: search {"query": "explode"}',
        )
        self.assertTrue(failed["done"])
        self.assertTrue(failed["info"]["sample_excluded"])
        self.assertEqual(failed["info"]["status"], "infrastructure_error")
        self.assertEqual(failed["info"]["action_execution"]["status"], "error")
        self.assertNotIn("temporary retrieval failure", failed["observation"])
        self.assertNotIn(
            "phase_verdict_ledger", failed["info"]["domain_evidence"]
        )

    def test_phase_budget_advance_clears_current_docids_but_keeps_ledger(self):
        _, wrapper, created = self._create("paper_eval")
        env_id = created["id"]
        exhausted = None
        for index in range(35):
            exhausted = wrapper.step(
                env_id,
                f'Action: search {{"query": "evidence-{index}"}}',
            )
        assert exhausted is not None
        self.assertFalse(exhausted["done"])
        self.assertEqual(exhausted["info"]["phase_index"], 1)
        self.assertIn("Native iteration budget: 0/35", exhausted["observation"])
        evidence = exhausted["info"]["domain_evidence"]
        self.assertEqual(evidence["retrieved_docids"], [])
        self.assertEqual(
            evidence["phase_verdict_ledger"][0]["retrieved_docids"],
            ["DOC-1"],
        )

        submitted = wrapper.step(
            env_id,
            'Action: SUBMIT_ANSWER {"answer": "ANSWER_TWO"}',
        )
        self.assertEqual(
            submitted["info"]["domain_evidence"]["phase_verdict_ledger"][1]
            ["retrieved_docids"],
            [],
        )


class BrowseCompDatasetTest(unittest.TestCase):
    def test_loader_retains_every_private_phase_answer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "progressive_search.jsonl"
            write_tasks(path)
            tasks = load_browsecomp_tasks(path)
        self.assertEqual(
            [(phase.kind, phase.question, phase.answer) for phase in tasks[0].phases],
            [
                ("subquery", "phase one", "ANSWER_ONE"),
                ("subquery", "phase two", "ANSWER_TWO"),
                ("final", "final question", "FINAL_ANSWER"),
            ],
        )

    def test_loader_refuses_legacy_ground_truth_and_misaligned_rows(self):
        cases = (
            {"query_id": "1", "query": "q", "answer": "a"},
            {"id": "1", "questions": ["q1", "q2"], "answers": ["a1"]},
        )
        for row in cases:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as tempdir:
                path = Path(tempdir) / "tasks.jsonl"
                write_tasks(path, [row])
                with self.assertRaises(ValueError):
                    load_browsecomp_tasks(path)

    def test_factory_refuses_injected_dataset_outside_explicit_test_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "tasks.jsonl"
            write_tasks(path)
            provenance = attest_injected_test_dataset(
                path,
                config="progressive_search",
            )
            with self.assertRaisesRegex(RuntimeError, "frozen public"):
                BrowseCompPlusFactory(
                    contract_mode="paper_eval",
                    tasks_path=path,
                    dataset_provenance=provenance,
                    search_tool=FakeSearch(),
                    judge=FakeJudge(),
                )

    def test_production_factory_attests_the_exact_assets_it_loads(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks_path = root / "tasks.jsonl"
            memoryarena_root = root / "MemoryArena"
            corpus_path = root / "corpus.jsonl"
            manifest_path = root / "corpus.manifest.json"
            index_pattern = str(root / "shard*.index")
            write_tasks(tasks_path)
            memoryarena_root.mkdir()
            corpus_path.write_text("{}\n", encoding="utf-8")
            manifest_path.write_text("{}\n", encoding="utf-8")
            dataset_provenance = types.SimpleNamespace(
                mode="frozen_public_hf_dataset",
                record_count=1,
                phase_count=3,
            )
            asset_provenance = {
                "mode": "frozen_public_assets",
                "corpus_sha256": "a" * 64,
            }
            search = FakeSearch()
            judge = FakeJudge()
            with (
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "verify_memoryarena_dataset_provenance"
                ),
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "attest_browsecomp_upstream",
                    return_value={"mode": "pinned_pristine_upstream"},
                ),
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "attest_browsecomp_search_assets",
                    return_value=asset_provenance,
                ) as asset_attester,
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "_build_upstream_search",
                    return_value=search,
                ) as search_builder,
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "_build_upstream_judge",
                    return_value=judge,
                ),
            ):
                factory = BrowseCompPlusFactory(
                    contract_mode="paper_eval",
                    tasks_path=tasks_path,
                    dataset_provenance=dataset_provenance,
                    memoryarena_root=memoryarena_root,
                    index_path=index_pattern,
                    corpus_path=corpus_path,
                    corpus_manifest_path=manifest_path,
                    embedding_model="text-embedding-3-small",
                    provider="openai",
                    embedding_endpoint="https://api.example/v1",
                    judge_config={
                        "backend": "openai_responses",
                        "model_name": "gpt-4.1",
                        "base_url": "https://api.example/v1",
                        "max_tokens": 8000,
                    },
                    expected_memoryarena_commit=(
                        BROWSECOMP_FROZEN_MEMORYARENA_COMMIT
                    ),
                )

            self.assertEqual(factory.search_asset_provenance, asset_provenance)
            asset_attester.assert_called_once_with(
                index_pattern=str((root / "shard*.index").resolve()),
                corpus_path=corpus_path.resolve(),
                corpus_manifest_path=manifest_path.resolve(),
                embedding_model="text-embedding-3-small",
            )
            search_builder.assert_called_once_with(
                memoryarena_root.resolve(),
                index_path=str((root / "shard*.index").resolve()),
                corpus_path=corpus_path.resolve(),
                embedding_model="text-embedding-3-small",
                provider="openai",
                embedding_endpoint="https://api.example/v1",
            )

    def test_production_factory_refuses_injected_asset_provenance(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks_path = root / "tasks.jsonl"
            memoryarena_root = root / "MemoryArena"
            write_tasks(tasks_path)
            memoryarena_root.mkdir()
            dataset_provenance = types.SimpleNamespace(
                mode="frozen_public_hf_dataset",
                record_count=1,
                phase_count=3,
            )
            with (
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "verify_memoryarena_dataset_provenance"
                ),
                self.assertRaisesRegex(RuntimeError, "refuses injected provenance"),
            ):
                BrowseCompPlusFactory(
                    contract_mode="paper_eval",
                    tasks_path=tasks_path,
                    dataset_provenance=dataset_provenance,
                    memoryarena_root=memoryarena_root,
                    index_path=str(root / "shard*.index"),
                    corpus_path=root / "corpus.jsonl",
                    corpus_manifest_path=root / "corpus.manifest.json",
                    embedding_endpoint="https://api.example/v1",
                    search_asset_provenance={"mode": "frozen_public_assets"},
                    expected_memoryarena_commit=(
                        BROWSECOMP_FROZEN_MEMORYARENA_COMMIT
                    ),
                )

    def test_task_bytes_changed_after_attestation_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "tasks.jsonl"
            write_tasks(path)
            provenance = attest_injected_test_dataset(
                path,
                config="progressive_search",
            )
            write_tasks(
                path,
                [{"id": 7, "questions": ["changed"], "answers": ["answer"]}],
            )
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                BrowseCompPlusFactory(
                    contract_mode="paper_eval",
                    tasks_path=path,
                    dataset_provenance=provenance,
                    search_tool=FakeSearch(),
                    judge=FakeJudge(),
                    test_mode=True,
                )

    def test_cross_source_parity_ignores_ids_but_not_questions_or_answers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks_path = root / "tasks.jsonl"
            reference_path = root / "reference.jsonl"
            write_tasks(tasks_path)
            reference_path.write_text(
                json.dumps(
                    {
                        "id": "different-id",
                        "question": ["phase one", "phase two", "final question"],
                        "answer": ["ANSWER_ONE", "ANSWER_TWO", "FINAL_ANSWER"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            evidence = attest_browsecomp_cross_source_parity(
                tasks_path,
                reference_path,
                expected_reference_sha256=reference_sha,
            )
            self.assertEqual(evidence["task_count"], 1)
            changed = json.loads(reference_path.read_text())
            changed["answer"][0] = "changed"
            reference_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            changed_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "questions/answers mismatch"):
                attest_browsecomp_cross_source_parity(
                    tasks_path,
                    reference_path,
                    expected_reference_sha256=changed_sha,
                )


class BrowseCompSearchAssetAttestationTest(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path):
        index_path = root / "shard0.index"
        id_map_path = root / "shard0_id_map.json"
        source_path = root / "source-000.parquet"
        corpus_path = root / "corpus.jsonl"
        manifest_path = root / "corpus.manifest.json"
        index_path.write_bytes(b"fixture-index")
        id_map_path.write_text(json.dumps({"ids": ["0", "1"]}) + "\n")
        source_path.write_bytes(b"frozen parquet bytes")
        corpus_path.write_text(
            json.dumps({"docid": "0", "text": "zero"}, separators=(",", ":"))
            + "\n"
            + json.dumps({"docid": "1", "text": "one"}, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        manifest = {
            "format": "agentmemory_browsecomp_corpus_manifest_v2",
            "path_base": "manifest_directory",
            "source": {
                "input_glob": str(root / "*.parquet"),
                "repository": BROWSECOMP_FROZEN_CORPUS_REPOSITORY,
                "revision": BROWSECOMP_FROZEN_CORPUS_REVISION,
                "file_count": 1,
                "files": [
                    {
                        "path": source_path.name,
                        "sha256": source_sha,
                        "size_bytes": source_path.stat().st_size,
                        "row_count": 2,
                        "columns": ["docid", "text", "url"],
                    }
                ],
                "columns": ["docid", "text", "url"],
            },
            "projection": {"output_columns": ["docid", "text"]},
            "output": {
                "path": corpus_path.name,
                "sha256": corpus_sha,
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
                    "id_map_sha256": hashlib.sha256(
                        id_map_path.read_bytes()
                    ).hexdigest(),
                    "vector_count": 2,
                },
            ),
            "expected_corpus_shards": (source_sha,),
            "expected_document_count": 2,
            "expected_corpus_sha256": corpus_sha,
        }
        return manifest, manifest_path, source_path, corpus_path, kwargs

    def test_attestation_hashes_actual_source_and_canonical_output_bytes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _, _, _, _, kwargs = self._fixture(Path(tempdir))
            evidence = attest_browsecomp_search_assets(**kwargs)
        self.assertEqual(evidence["document_count"], 2)
        self.assertEqual(len(evidence["corpus_source_shards"]), 1)
        self.assertEqual(evidence["embedding_dimension"], 1536)

    def test_manifest_cannot_self_report_forged_source_bytes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _, _, source_path, _, kwargs = self._fixture(Path(tempdir))
            source_path.write_bytes(b"tampered after manifest")
            with self.assertRaisesRegex(RuntimeError, "self-inconsistent"):
                attest_browsecomp_search_assets(**kwargs)

    def test_manifest_matching_fake_corpus_still_fails_canonical_sha(self):
        with tempfile.TemporaryDirectory() as tempdir:
            manifest, manifest_path, _, corpus_path, kwargs = self._fixture(
                Path(tempdir)
            )
            corpus_path.write_text(
                json.dumps({"docid": "0", "text": "forged"})
                + "\n"
                + json.dumps({"docid": "1", "text": "one"})
                + "\n",
                encoding="utf-8",
            )
            manifest["output"]["sha256"] = hashlib.sha256(
                corpus_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "canonical freeze"):
                attest_browsecomp_search_assets(**kwargs)

    def test_v2_manifest_rejects_mount_specific_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tempdir:
            manifest, manifest_path, source_path, _, kwargs = self._fixture(
                Path(tempdir)
            )
            manifest["source"]["files"][0]["path"] = str(source_path.resolve())
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "must be relative"):
                attest_browsecomp_search_assets(**kwargs)

    def test_missing_canonical_corpus_pin_fails_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _, _, _, _, kwargs = self._fixture(Path(tempdir))
            del kwargs["expected_corpus_sha256"]
            with (
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "BROWSECOMP_FROZEN_MATERIALIZED_CORPUS_SHA256",
                    None,
                ),
                self.assertRaisesRegex(RuntimeError, "not configured"),
            ):
                attest_browsecomp_search_assets(**kwargs)

    @unittest.skipUnless(
        all(
            os.environ.get(key)
            for key in (
                "MEMORYARENA_BROWSECOMP_REAL_INDEX_PATTERN",
                "MEMORYARENA_BROWSECOMP_REAL_CORPUS",
                "MEMORYARENA_BROWSECOMP_REAL_CORPUS_MANIFEST",
            )
        ),
        "real frozen BrowseComp assets are not configured",
    )
    def test_real_frozen_search_assets(self):
        evidence = attest_browsecomp_search_assets(
            index_pattern=os.environ["MEMORYARENA_BROWSECOMP_REAL_INDEX_PATTERN"],
            corpus_path=Path(os.environ["MEMORYARENA_BROWSECOMP_REAL_CORPUS"]),
            corpus_manifest_path=Path(
                os.environ["MEMORYARENA_BROWSECOMP_REAL_CORPUS_MANIFEST"]
            ),
            embedding_model="text-embedding-3-small",
        )
        self.assertEqual(evidence["document_count"], 100195)


class BrowseCompLoadedSearcherTest(unittest.TestCase):
    def test_loaded_searcher_checks_dimension_count_and_exact_coverage(self):
        indexes = [types.SimpleNamespace(d=1536, ntotal=2)]
        searcher = types.SimpleNamespace(
            indexes=indexes,
            id_maps=[["0", "1"]],
            docid_to_text={"0": "zero", "1": "one"},
        )
        specs = ({"vector_count": 2},)
        validate_loaded_browsecomp_searcher(
            searcher,
            embedding_model="text-embedding-3-small",
            expected_index_shards=specs,
            expected_document_count=2,
        )
        indexes[0].d = 768
        with self.assertRaisesRegex(RuntimeError, "dimension"):
            validate_loaded_browsecomp_searcher(
                searcher,
                embedding_model="text-embedding-3-small",
                expected_index_shards=specs,
                expected_document_count=2,
            )


class BrowseCompTokenizerTest(unittest.TestCase):
    def test_tokenizer_uses_exact_revision_and_never_network_fallback(self):
        calls = []
        sentinel = object()

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(repository, **kwargs):
                calls.append((repository, kwargs))
                return sentinel

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = FakeAutoTokenizer
        snapshot = Path("/frozen/Qwen3-0.6B/c1899de")
        with (
            patch.dict(sys.modules, {"transformers": transformers}),
            patch(
                "agentenv_agentmemory.domains.browsecomp."
                "_resolve_frozen_tokenizer_snapshot",
                return_value=snapshot,
            ),
            patch(
                "agentenv_agentmemory.domains.browsecomp."
                "attest_frozen_snippet_tokenizer"
            ) as attester,
        ):
            self.assertIs(_load_frozen_snippet_tokenizer(), sentinel)
        attester.assert_called_once_with(snapshot)
        self.assertEqual(
            calls,
            [
                (
                    str(snapshot),
                    {
                        "local_files_only": True,
                        "trust_remote_code": False,
                    },
                )
            ],
        )

    def test_snapshot_attestation_hashes_resolved_file_bytes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            files = {
                "config.json": b"config",
                "tokenizer.json": b"tokenizer",
            }
            for name, content in files.items():
                (root / name).write_bytes(content)
            expected = {
                name: hashlib.sha256(content).hexdigest()
                for name, content in files.items()
            }
            bundle = hashlib.sha256(
                json.dumps(
                    expected,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            evidence = attest_frozen_snippet_tokenizer(
                root,
                expected_files_sha256=expected,
                expected_bundle_sha256=bundle,
            )
            self.assertEqual(evidence["files_sha256"], expected)
            (root / "config.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                attest_frozen_snippet_tokenizer(
                    root,
                    expected_files_sha256=expected,
                    expected_bundle_sha256=bundle,
                )


class BrowseCompUpstreamAttestationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source_files = set(BROWSECOMP_UPSTREAM_REQUIRED_FILES) | {
            "agent/__init__.py",
            "agent/search_helper.py",
            "env/env_systems/web_search_env/__init__.py",
            "env/env_systems/web_search_env/search_agent/__init__.py",
            "env/env_systems/web_search_env/search_agent/openai_client.py",
            "env/env_systems/web_search_env/searcher/__init__.py",
            "env/env_systems/web_search_env/searcher/searchers/__init__.py",
            "env/env_systems/web_search_env/searcher/searchers/custom_searcher.py",
        }
        for relative_path in self.source_files:
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
            self.source_files,
        )

    def test_rejects_modified_browsecomp_source(self):
        path = self.root / BROWSECOMP_UPSTREAM_REQUIRED_FILES[0]
        path.write_text("# changed semantics\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_browsecomp_upstream(self.root, expected_commit=self.commit)

    def test_rejects_modified_imported_searcher_dependency(self):
        path = (
            self.root
            / "env/env_systems/web_search_env/searcher/searchers/custom_searcher.py"
        )
        path.write_text("# changed imported dependency\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_browsecomp_upstream(self.root, expected_commit=self.commit)

    def test_rejects_untracked_executable_dependency(self):
        path = self.root / "env/env_systems/web_search_env/search_agent/injected.py"
        path.write_text("# untracked executable dependency\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source set is not pristine"):
            attest_browsecomp_upstream(self.root, expected_commit=self.commit)


@unittest.skipUnless(
    os.environ.get("MEMORYARENA_PROGRESSIVE_SEARCH_PUBLIC221"),
    "real frozen public221 dataset is not configured",
)
class BrowseCompRealDatasetTest(unittest.TestCase):
    def test_real_public221_dataset_attests_and_loads_all_phase_answers(self):
        path = Path(os.environ["MEMORYARENA_PROGRESSIVE_SEARCH_PUBLIC221"])
        provenance = attest_frozen_memoryarena_dataset(
            path,
            config="progressive_search",
        )
        tasks = load_browsecomp_tasks(path)
        self.assertEqual(provenance.record_count, 221)
        self.assertEqual(provenance.phase_count, 1641)
        self.assertEqual(sum(len(task.phases) for task in tasks), 1641)
        self.assertTrue(all(phase.answer for task in tasks for phase in task.phases))


if __name__ == "__main__":
    unittest.main()
