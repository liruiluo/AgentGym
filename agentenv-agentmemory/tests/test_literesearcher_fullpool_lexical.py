from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agentenv_agentmemory.literesearcher import (
    FullPoolLiteResearcherTask,
    FullPoolLiteResearcherTasks,
    LITERESEARCHER_FULLPOOL_SURFACE,
    LiteResearchJudgeResult,
    LiteResearchRequestError,
    LiteResearcherWrapper,
    SQLiteFTSLiteResearchBackend,
    TantivyLiteResearchBackend,
    UPSTREAM_LLM_JUDGE_CONTRACT,
)


class _FakeTantivyDocument:
    def __init__(self, document_id: int) -> None:
        self.document_id = document_id

    def get_first(self, field: str) -> int:
        if field != "id":
            raise KeyError(field)
        return self.document_id


class _FakeTantivySearcher:
    num_docs = 3

    def search(self, query, *, limit: int, count: bool):
        del query, count
        return type(
            "SearchResult",
            (),
            {"hits": [(3.0 - index, index + 1) for index in range(min(limit, 3))]},
        )()

    @staticmethod
    def doc(address: int) -> _FakeTantivyDocument:
        return _FakeTantivyDocument(address)


class _FakeTantivyIndex:
    schema = object()

    @staticmethod
    def reload() -> None:
        return None

    @staticmethod
    def searcher() -> _FakeTantivySearcher:
        return _FakeTantivySearcher()


class _FakeTantivy:
    __version__ = "0.25.1-test"

    class Occur:
        Should = "should"

    class Query:
        @staticmethod
        def term_query(schema, field: str, token: str):
            return schema, field, token

        @staticmethod
        def boolean_query(clauses):
            return tuple(clauses)

    class Index:
        @staticmethod
        def open(path: str) -> _FakeTantivyIndex:
            del path
            return _FakeTantivyIndex()


class _SemanticJudgeStub:
    contract_id = UPSTREAM_LLM_JUDGE_CONTRACT

    def __init__(self, *, correct: bool = True) -> None:
        self.correct = correct
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    def judge(self, question, targets, answer):
        self.calls.append((question, tuple(targets), answer))
        return LiteResearchJudgeResult(
            correct=self.correct,
            method="semantic_judge_stub",
            attempts=1,
        )

    def metadata(self):
        return {
            "contract": self.contract_id,
            "primary": "semantic_judge_stub",
            "fallback": "upstream_em_v1",
            "semantic_equivalence": True,
        }


class LiteResearcherFullPoolLexicalTests(unittest.TestCase):
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
            self.backend,
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

    def test_tantivy_backend_masks_url_and_uses_released_document_store(self) -> None:
        index = Path(self.temporary.name) / "tantivy-index"
        index.mkdir()
        (index / "agentmemory-index.json").write_text(
            json.dumps(
                {
                    "contract": "combined_title2_document_bm25_v1",
                    "document_count": 3,
                    "tantivy_version": "0.25.1-test",
                }
            ),
            encoding="utf-8",
        )
        with patch(
            "agentenv_agentmemory.literesearcher.lexical_backend."
            "importlib.import_module",
            return_value=_FakeTantivy,
        ):
            backend = TantivyLiteResearchBackend(
                self.tasks,
                self.database,
                index,
                top_k=2,
            )
            hits = backend.search(
                "Alpha research evidence",
                mask_url=self.tasks.train[0].mask_url,
            )
        self.assertEqual(
            [hit["url"] for hit in hits],
            [
                "https://public.example/evidence",
                "https://public.example/other",
            ],
        )
        self.assertIn("Beta Fact", hits[0]["snippet"])
        metadata = backend.metadata()
        self.assertEqual(
            metadata["backend_contract"],
            "literesearcher_released_corpus_tantivy_bm25_v1",
        )
        self.assertEqual(metadata["tantivy_document_count"], 3)
        page = backend.visit(hits[0]["url"], goal="Beta Fact")
        self.assertIn("Beta Fact", page["content"])

    def test_tantivy_backend_routes_unicode_queries_to_sqlite(self) -> None:
        index = Path(self.temporary.name) / "tantivy-unicode-index"
        index.mkdir()
        (index / "agentmemory-index.json").write_text(
            json.dumps(
                {
                    "contract": "combined_title2_document_bm25_v1",
                    "document_count": 3,
                    "tantivy_version": "0.25.1-test",
                }
            ),
            encoding="utf-8",
        )
        with (
            patch(
                "agentenv_agentmemory.literesearcher.lexical_backend."
                "importlib.import_module",
                return_value=_FakeTantivy,
            ),
            patch.object(
                SQLiteFTSLiteResearchBackend,
                "search",
                return_value=[{"url": "sqlite-fallback"}],
            ) as sqlite_search,
        ):
            backend = TantivyLiteResearchBackend(
                self.tasks,
                self.database,
                index,
            )
            result = backend.search(
                "\u7d22\u6069\u6cb3\u53d1\u6e90\u4e8e\u5df4\u683c\u9a6c\u5c3c",
                mask_url="masked",
            )
        self.assertEqual(result, [{"url": "sqlite-fallback"}])
        sqlite_search.assert_called_once_with(
            "\u7d22\u6069\u6cb3\u53d1\u6e90\u4e8e\u5df4\u683c\u9a6c\u5c3c",
            top_k=5,
            mask_url="masked",
        )

    def test_full_pool_terminal_reward_comes_from_semantic_judge(self) -> None:
        judge = _SemanticJudgeStub(correct=True)
        wrapper = LiteResearcherWrapper(
            self.tasks,
            self.backend,
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


if __name__ == "__main__":
    unittest.main()
