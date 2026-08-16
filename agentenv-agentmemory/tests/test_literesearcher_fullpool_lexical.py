from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from agentenv_agentmemory.literesearcher import (
    FullPoolLiteResearcherTask,
    FullPoolLiteResearcherTasks,
    LITERESEARCHER_FULLPOOL_SURFACE,
    LiteResearchBackendError,
    LiteResearchJudgeResult,
    LiteResearchRequestError,
    LiteResearcherWrapper,
    SQLiteFTSLiteResearchBackend,
    UPSTREAM_LLM_JUDGE_CONTRACT,
)


class _SemanticJudgeStub:
    contract_id = UPSTREAM_LLM_JUDGE_CONTRACT

    def __init__(
        self,
        *,
        correct: bool = True,
        fallback_reason: str | None = None,
    ) -> None:
        self.correct = correct
        self.fallback_reason = fallback_reason
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    def judge(self, question, targets, answer):
        self.calls.append((question, tuple(targets), answer))
        return LiteResearchJudgeResult(
            correct=self.correct,
            method="semantic_judge_stub",
            attempts=1,
            primary_model="kimi-k2.6",
            fallback_reason=self.fallback_reason,
        )

    def metadata(self):
        return {
            "contract": self.contract_id,
            "primary": "semantic_judge_stub",
            "fallback": "upstream_em_v1",
            "semantic_equivalence": True,
        }


class _FormalBackendStub:
    contract_id = "literesearcher_upstream_hybrid_diskann_v1"

    def __init__(self, tasks, *, fail_search: bool = False) -> None:
        self.tasks_source = tasks
        self.split = "train"
        self.fail_search = fail_search

    def metadata(self):
        return {
            "backend_contract": self.contract_id,
            "service_identity_verified": True,
        }

    def search(self, query, *, top_k=None, mask_url=""):
        del query, top_k
        hits = [
            {
                "url": "https://public.example/evidence",
                "title": "Independent evidence",
                "snippet": "Alpha research evidence identifies the public result.",
                "rank": 1,
            }
        ]
        return [hit for hit in hits if hit["url"] != mask_url]

    def search_with_telemetry(self, query, *, top_k=None, mask_url=""):
        if self.fail_search:
            raise LiteResearchBackendError("test backend failure")
        return _BackendCallStub(
            self.search(query, top_k=top_k, mask_url=mask_url),
            {
                "backend_telemetry_schema": "agentmemory_literesearcher_backend_timing_v1",
                "backend_query_count": 1,
                "backend_reported_search_seconds_by_query": [0.3],
                "backend_reported_embedding_seconds_by_query": [0.1],
                "backend_reported_milvus_seconds_by_query": [0.2],
            },
        )

    def visit(self, url, *, goal="", page=1):
        if url != "https://public.example/evidence":
            raise LiteResearchRequestError("visit URL is outside the released corpus")
        return {
            "url": url,
            "title": "Independent evidence",
            "content": "Alpha research evidence identifies the public result Beta Fact.",
            "goal": goal,
            "page": page,
            "page_count": 1,
            "next_page": None,
        }

    def visit_with_telemetry(self, url, *, goal="", page=1):
        return _BackendCallStub(
            self.visit(url, goal=goal, page=page),
            {
                "backend_telemetry_schema": "agentmemory_literesearcher_backend_timing_v1",
                "backend_reported_visit_seconds": 0.4,
            },
        )


class _BackendCallStub:
    def __init__(self, value, timing) -> None:
        self.value = value
        self.timing = timing

    def public_results(self):
        return list(self.value)

    def public_page(self):
        return dict(self.value)

    def timing_evidence(self):
        return dict(self.timing)


class _WorkspaceResultStub:
    message = "workspace result"
    op = "SHELL_COMMAND"


class _WorkspaceStub:
    def reset_episode(self, episode_id, *, enabled=True) -> None:
        del episode_id, enabled

    def apply(self, action, *, env_step, phase_index):
        del action, env_step, phase_index
        return _WorkspaceResultStub()

    def close(self) -> None:
        pass


class LiteResearcherFullPoolContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "corpus.sqlite"
        connection = sqlite3.connect(self.database)
        connection.execute(
            "CREATE TABLE documents ("
            "id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, "
            "title TEXT NOT NULL, document TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO documents(url, title, document) VALUES (?, ?, ?)",
            [
                (
                    "https://public.example/answer",
                    "Hidden source",
                    "The private answer is Alpha Secret and must be masked.",
                ),
                (
                    "https://public.example/evidence",
                    "Independent evidence",
                    "Alpha research evidence identifies the public result Beta Fact.",
                ),
                (
                    "https://public.example/other",
                    "Other source",
                    "Unrelated material for concurrency checks.",
                ),
            ],
        )
        connection.execute(
            "CREATE VIRTUAL TABLE documents_fts USING fts5("
            "title, document, content='documents', content_rowid='id')"
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
        connection.commit()
        connection.close()

        self.tasks = FullPoolLiteResearcherTasks(
            (
                FullPoolLiteResearcherTask(
                    index=0,
                    question="Which public result is supported by the Alpha research evidence?",
                    targets=("Beta Fact",),
                    mask_url="https://public.example/answer",
                    row_identity="a" * 64,
                    parquet_path="train.parquet",
                    physical_row=0,
                    data_source="test",
                    upstream_curriculum_stage=1,
                ),
            ),
            manifest_sha256="b" * 64,
            dataset_revision="revision",
            upstream_commit="commit",
        )
        self.backend = SQLiteFTSLiteResearchBackend(self.tasks, self.database, top_k=3)
        self.formal_backend = _FormalBackendStub(self.tasks)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_masked_url_is_excluded_and_search_result_can_be_visited(self) -> None:
        hits = self.backend.search(
            "Alpha research evidence",
            mask_url=self.tasks.train[0].mask_url,
        )
        self.assertTrue(hits)
        self.assertNotIn(self.tasks.train[0].mask_url, {hit["url"] for hit in hits})
        page = self.backend.visit(hits[0]["url"], goal="Beta Fact")
        self.assertEqual(page["url"], "https://public.example/evidence")
        self.assertIn("Beta Fact", page["content"])

    def test_unknown_visit_fails_closed(self) -> None:
        with self.assertRaisesRegex(LiteResearchRequestError, "outside the released corpus"):
            self.backend.visit("https://unknown.example/not-in-corpus")

    def test_wrapper_never_exposes_private_target_or_mask_metadata(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.tasks,
            self.formal_backend,
            split="train",
            surface=LITERESEARCHER_FULLPOOL_SURFACE,
            judge=_SemanticJudgeStub(),
        )
        created = wrapper.create(data_idx=0)
        public = json.dumps(
            {"observation": created["observation"], "metadata": wrapper.metadata()},
            ensure_ascii=False,
        )
        self.assertNotIn("Beta Fact", public)
        self.assertNotIn(self.tasks.train[0].mask_url, public)

        searched = wrapper.step(
            created["id"],
            '<tool_call>{"name":"search","arguments":{"query":"Alpha research evidence"}}</tool_call>',
        )
        self.assertNotIn(self.tasks.train[0].mask_url, searched["observation"])
        self.assertNotIn("Alpha Secret", searched["observation"])
        wrapper.close(created["id"])

    def test_sixty_four_read_only_threads_are_stable(self) -> None:
        def query(_: int) -> tuple[str, str]:
            hit = self.backend.search(
                "Alpha research evidence",
                mask_url=self.tasks.train[0].mask_url,
                top_k=1,
            )[0]
            page = self.backend.visit(hit["url"], goal="Beta Fact")
            return hit["url"], page["content"]

        with ThreadPoolExecutor(max_workers=64) as executor:
            results = list(executor.map(query, range(64)))
        self.assertEqual(len(results), 64)
        self.assertEqual({url for url, _ in results}, {"https://public.example/evidence"})
        self.assertTrue(all("Beta Fact" in content for _, content in results))

    def test_full_pool_terminal_reward_comes_from_semantic_judge(self) -> None:
        judge = _SemanticJudgeStub(correct=True)
        wrapper = LiteResearcherWrapper(
            self.tasks,
            self.formal_backend,
            split="train",
            surface=LITERESEARCHER_FULLPOOL_SURFACE,
            judge=judge,
        )
        created = wrapper.create(data_idx=0)
        result = wrapper.step(created["id"], "<answer>Beta</answer>")
        self.assertTrue(result["done"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(
            result["info"]["wrapper_evidence"]["judge_method"],
            "semantic_judge_stub",
        )
        self.assertEqual(
            result["info"]["wrapper_evidence"]["judge_latency_seconds"],
            0.0,
        )
        self.assertEqual(
            result["info"]["wrapper_evidence"]["judge_primary_model"],
            "kimi-k2.6",
        )
        self.assertFalse(
            result["info"]["wrapper_evidence"]["judge_fallback_used"]
        )
        self.assertIsNone(
            result["info"]["wrapper_evidence"]["judge_fallback_reason"]
        )
        self.assertEqual(
            judge.calls,
            [
                (
                    self.tasks.train[0].question,
                    self.tasks.train[0].targets,
                    "Beta",
                )
            ],
        )
        wrapper.close(created["id"])

    def test_full_pool_receipts_preserve_per_call_phase_telemetry(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.tasks,
            self.formal_backend,
            split="train",
            surface=LITERESEARCHER_FULLPOOL_SURFACE,
            judge=_SemanticJudgeStub(),
        )
        created = wrapper.create(data_idx=0)
        searched = wrapper.step(
            created["id"],
            '<tool_call>{"name":"search","arguments":{"query":"Alpha"}}</tool_call>',
        )
        search_evidence = searched["info"]["wrapper_evidence"]
        self.assertEqual(
            search_evidence["telemetry_schema"],
            "agentmemory_literesearcher_env_timing_v1",
        )
        self.assertEqual(search_evidence["phase"], "search")
        self.assertGreaterEqual(search_evidence["env_step_wall_seconds"], 0.0)
        self.assertEqual(search_evidence["other_env_seconds"], 0.0)
        self.assertEqual(
            search_evidence["backend_reported_search_seconds_by_query"], [0.3]
        )
        self.assertEqual(
            search_evidence["backend_reported_embedding_seconds_by_query"], [0.1]
        )
        self.assertEqual(
            search_evidence["backend_reported_milvus_seconds_by_query"], [0.2]
        )

        visited = wrapper.step(
            created["id"],
            '<tool_call>{"name":"visit","arguments":'
            '{"url":"https://public.example/evidence"}}</tool_call>',
        )
        visit_evidence = visited["info"]["wrapper_evidence"]
        self.assertEqual(visit_evidence["phase"], "visit")
        self.assertEqual(visit_evidence["backend_reported_visit_seconds"], 0.4)
        self.assertGreaterEqual(visit_evidence["env_step_wall_seconds"], 0.0)

        answered = wrapper.step(created["id"], "<answer>Beta</answer>")
        judge_evidence = answered["info"]["wrapper_evidence"]
        self.assertEqual(judge_evidence["phase"], "judge")
        self.assertEqual(judge_evidence["judge_method"], "semantic_judge_stub")
        self.assertFalse(judge_evidence["judge_fallback_used"])
        self.assertGreaterEqual(judge_evidence["env_step_wall_seconds"], 0.0)
        receipt = json.dumps(judge_evidence, sort_keys=True)
        self.assertNotIn("Beta Fact", receipt)
        self.assertNotIn(self.tasks.train[0].mask_url, receipt)
        wrapper.close(created["id"])

    def test_full_pool_other_env_timing_covers_workspace_invalid_and_backend_error(self) -> None:
        cases = (
            (
                self.formal_backend,
                _WorkspaceStub(),
                'shell_command {"command":"printf ok"}',
                "workspace",
                False,
            ),
            (self.formal_backend, None, "malformed action", "invalid", False),
            (
                _FormalBackendStub(self.tasks, fail_search=True),
                None,
                '<tool_call>{"name":"search","arguments":{"query":"Alpha"}}</tool_call>',
                "backend_error",
                True,
            ),
        )
        for backend, workspace, action, phase, done in cases:
            with self.subTest(phase=phase):
                wrapper = LiteResearcherWrapper(
                    self.tasks,
                    backend,
                    workspace=workspace,
                    split="train",
                    surface=LITERESEARCHER_FULLPOOL_SURFACE,
                    judge=_SemanticJudgeStub(),
                )
                created = wrapper.create(data_idx=0)
                result = wrapper.step(created["id"], action)
                evidence = result["info"]["wrapper_evidence"]
                self.assertEqual(evidence["phase"], phase)
                self.assertGreaterEqual(evidence["env_step_wall_seconds"], 0.0)
                self.assertEqual(
                    evidence["other_env_seconds"],
                    evidence["env_step_wall_seconds"],
                )
                self.assertEqual(result["done"], done)
                wrapper.close(created["id"])

    def test_full_pool_terminal_receipt_records_judge_fallback_reason(self) -> None:
        judge = _SemanticJudgeStub(
            correct=False,
            fallback_reason="timeout",
        )
        wrapper = LiteResearcherWrapper(
            self.tasks,
            self.formal_backend,
            split="train",
            surface=LITERESEARCHER_FULLPOOL_SURFACE,
            judge=judge,
        )
        created = wrapper.create(data_idx=0)
        result = wrapper.step(created["id"], "<answer>Beta</answer>")
        evidence = result["info"]["wrapper_evidence"]
        self.assertTrue(evidence["judge_fallback_used"])
        self.assertEqual(evidence["judge_fallback_reason"], "timeout")
        self.assertEqual(evidence["judge_primary_model"], "kimi-k2.6")
        wrapper.close(created["id"])

    def test_full_pool_rejects_lexical_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "upstream hybrid DISKANN"):
            LiteResearcherWrapper(
                self.tasks,
                self.backend,
                split="train",
                surface=LITERESEARCHER_FULLPOOL_SURFACE,
                judge=_SemanticJudgeStub(),
            )


if __name__ == "__main__":
    unittest.main()
