from __future__ import annotations

import importlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .backend import (
    LiteResearchBackendError,
    LiteResearchRequestError,
    SearchHit,
    _rank_windows_by_goal,
)


_STOP_WORDS = {
    "about", "after", "also", "and", "are", "been", "before", "being",
    "between", "did", "does", "during", "for", "from", "had", "has",
    "have", "how", "into", "its", "not", "that", "the", "their", "then",
    "there", "these", "they", "this", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would",
}


def _query_tokens(value: str) -> list[str]:
    tokens = []
    seen = set()
    for token in re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE):
        if len(token) < 2 or token in _STOP_WORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 16:
            break
    if not tokens:
        raise LiteResearchRequestError("search query has no indexable terms")
    return tokens


def _normalize_search_request(
    query: str | list[str],
    *,
    top_k: int | None,
    default_top_k: int,
) -> tuple[str, int]:
    queries = [query] if isinstance(query, str) else query
    if not isinstance(queries, list) or not queries or any(
        not isinstance(item, str) for item in queries
    ):
        raise LiteResearchRequestError(
            "search query must be a non-empty string or list of strings"
        )
    query_text = " ".join(item.strip() for item in queries).strip()
    if not query_text:
        raise LiteResearchRequestError("search query must not be empty")
    limit = default_top_k if top_k is None else top_k
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise LiteResearchRequestError("search top_k must be a positive integer")
    return query_text, limit


def _requires_sqlite_tokenization(value: str) -> bool:
    fallback_ranges = (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0x3040, 0x30FF),
        (0xAC00, 0xD7AF),
    )
    return any(
        start <= ord(character) <= end
        for character in value
        for start, end in fallback_ranges
    )


class SQLiteFTSLiteResearchBackend:
    """Read-only lexical fallback over LiteResearcher's released corpus."""

    contract_id = "literesearcher_released_corpus_sqlite_fts_v1"

    def __init__(self, tasks, database_path: str | Path, *, top_k: int = 5) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        path = Path(database_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"LiteResearcher FTS database does not exist: {path}")
        self.tasks_source = tasks
        self.split = "train"
        self.database_path = path
        self.top_k = top_k
        self._local = threading.local()
        connection = self._connection()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise ValueError(f"LiteResearcher FTS quick_check failed: {quick_check}")
        self.document_count = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            uri = f"file:{quote(str(self.database_path), safe='/')}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-262144")
            self._local.connection = connection
        return connection

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_contract": self.contract_id,
            "backend_class": "lexical_fallback",
            "upstream_hybrid_backend": False,
            "released_search_corpus": True,
            "document_count": self.document_count,
            "search_result_url_namespace": "released_public_url_v1",
            "search_exposes_mask_url": False,
            "search_exposes_targets": False,
            "visit_exposes_mask_url": False,
            "visit_pagination_contract": "goal_bm25_overlapping_chars_v1",
            "live_network": False,
            "resident_index": True,
            "failure_mode": "fail_closed",
        }

    @staticmethod
    def _query(value: str) -> str:
        return " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"'
            for token in _query_tokens(value)
        )

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> list[dict[str, Any]]:
        query_text, limit = _normalize_search_request(
            query,
            top_k=top_k,
            default_top_k=self.top_k,
        )
        try:
            rows = self._connection().execute(
                "SELECT d.url, d.title, "
                "snippet(documents_fts, 1, '', '', ' ... ', 48), "
                "bm25(documents_fts, 2.0, 1.0) AS score "
                "FROM documents_fts JOIN documents AS d "
                "ON d.id = documents_fts.rowid "
                "WHERE documents_fts MATCH ? AND d.url <> ? "
                "ORDER BY score LIMIT ?",
                (self._query(query_text), str(mask_url), limit),
            ).fetchall()
        except sqlite3.Error as exc:
            raise LiteResearchBackendError("LiteResearcher FTS search failed") from exc
        return [
            SearchHit(
                url=str(url),
                title=str(title),
                snippet=str(snippet),
                rank=rank,
            ).public_record()
            for rank, (url, title, snippet, _) in enumerate(rows, start=1)
        ]

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise LiteResearchRequestError("visit URL must be a non-empty string")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise LiteResearchRequestError("visit page must be a positive integer")
        try:
            row = self._connection().execute(
                "SELECT url, title, document FROM documents WHERE url = ?",
                (url.strip(),),
            ).fetchone()
        except sqlite3.Error as exc:
            raise LiteResearchBackendError("LiteResearcher FTS visit failed") from exc
        if row is None:
            raise LiteResearchRequestError("visit URL is outside the released corpus")
        resolved_url, title, document = map(str, row)
        pages = _rank_windows_by_goal(document, str(goal))
        if page > len(pages):
            raise LiteResearchRequestError(
                f"visit page {page} exceeds page_count {len(pages)}"
            )
        return {
            "url": resolved_url,
            "title": title,
            "content": pages[page - 1],
            "goal": str(goal),
            "page": page,
            "page_count": len(pages),
            "next_page": page + 1 if page < len(pages) else None,
        }


class TantivyLiteResearchBackend(SQLiteFTSLiteResearchBackend):
    """Tantivy BM25 retrieval with the released SQLite corpus as document store."""

    contract_id = "literesearcher_released_corpus_tantivy_bm25_v1"

    def __init__(
        self,
        tasks,
        database_path: str | Path,
        index_path: str | Path,
        *,
        top_k: int = 5,
    ) -> None:
        super().__init__(tasks, database_path, top_k=top_k)
        resolved_index = Path(index_path).expanduser().resolve()
        if not resolved_index.is_dir():
            raise ValueError(
                f"LiteResearcher Tantivy index does not exist: {resolved_index}"
            )
        manifest_path = resolved_index / "agentmemory-index.json"
        try:
            index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "LiteResearcher Tantivy index requires agentmemory-index.json"
            ) from exc
        if (
            index_manifest.get("contract")
            != "combined_title2_document_bm25_lead512_v2"
        ):
            raise ValueError("LiteResearcher Tantivy index contract mismatch")
        try:
            tantivy = importlib.import_module("tantivy")
        except ImportError as exc:
            raise RuntimeError(
                "LiteResearcher Tantivy backend requires tantivy==0.25.1"
            ) from exc
        runtime_version = getattr(tantivy, "__version__", "unknown")
        if index_manifest.get("tantivy_version") != runtime_version:
            raise ValueError(
                "LiteResearcher Tantivy index/runtime version mismatch: "
                f"{index_manifest.get('tantivy_version')} != {runtime_version}"
            )
        try:
            index = tantivy.Index.open(str(resolved_index))
            index.reload()
            searcher = index.searcher()
        except Exception as exc:
            raise ValueError(
                f"LiteResearcher Tantivy index cannot be opened: {resolved_index}"
            ) from exc
        if int(searcher.num_docs) != self.document_count:
            raise ValueError(
                "LiteResearcher Tantivy/SQLite document count mismatch: "
                f"{searcher.num_docs} != {self.document_count}"
            )
        if int(index_manifest.get("document_count", -1)) != self.document_count:
            raise ValueError(
                "LiteResearcher Tantivy manifest/SQLite document count mismatch"
            )
        self.index_path = resolved_index
        self._tantivy = tantivy
        self._index = index
        self._searcher = searcher
        self._schema = index.schema
        self._index_manifest = index_manifest
        self._search_slots = threading.BoundedSemaphore(8)
        try:
            probe = tantivy.Query.term_query(self._schema, "content", "probe")
            searcher.search(probe, limit=1, count=False)
        except Exception as exc:
            raise ValueError(
                "LiteResearcher Tantivy index requires indexed field 'content'"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "backend_contract": self.contract_id,
                "backend_class": "lexical_accelerated",
                "retrieval_engine": "tantivy_bm25",
                "retrieval_engine_version": getattr(
                    self._tantivy, "__version__", "unknown"
                ),
                "search_ranking_contract": (
                    "combined_title2_document_bm25_lead512_v2"
                ),
                "search_snippet_contract": "tantivy_lead_512_chars_v1",
                "search_document_store": "tantivy_lead512_stored_v1",
                "max_concurrent_searches": 8,
                "cjk_query_fallback": "sqlite_fts_v1",
                "index_manifest_contract": self._index_manifest["contract"],
                "tantivy_document_count": int(self._searcher.num_docs),
                "sqlite_document_store": True,
            }
        )
        return metadata

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> list[dict[str, Any]]:
        with self._search_slots:
            return self._search_unthrottled(
                query,
                top_k=top_k,
                mask_url=mask_url,
            )

    def _search_unthrottled(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> list[dict[str, Any]]:
        query_text, limit = _normalize_search_request(
            query,
            top_k=top_k,
            default_top_k=self.top_k,
        )
        if _requires_sqlite_tokenization(query_text):
            return super().search(
                query,
                top_k=limit,
                mask_url=mask_url,
            )
        tokens = _query_tokens(query_text)
        clauses = [
            (
                self._tantivy.Occur.Should,
                self._tantivy.Query.term_query(self._schema, "content", token),
            )
            for token in tokens
        ]
        try:
            result = self._searcher.search(
                self._tantivy.Query.boolean_query(clauses),
                limit=limit + 1,
                count=False,
            )
            documents = [
                self._searcher.doc(address) for _, address in result.hits
            ]
            if not documents:
                return []
        except Exception as exc:
            raise LiteResearchBackendError(
                "LiteResearcher Tantivy search failed"
            ) from exc
        hits = []
        for document in documents:
            url = document.get_first("url")
            title = document.get_first("title")
            snippet = document.get_first("snippet")
            if not all(isinstance(value, str) for value in (url, title, snippet)):
                raise LiteResearchBackendError(
                    "LiteResearcher Tantivy result lacks stored public fields"
                )
            if url == str(mask_url):
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=title,
                    snippet=str(snippet),
                    rank=len(hits) + 1,
                ).public_record()
            )
            if len(hits) == limit:
                break
        return hits
