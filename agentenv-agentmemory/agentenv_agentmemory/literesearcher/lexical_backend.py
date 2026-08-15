from __future__ import annotations

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
        tokens = []
        seen = set()
        for token in re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE):
            if len(token) < 2 or token in _STOP_WORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token.replace('"', '""'))
            if len(tokens) >= 16:
                break
        if not tokens:
            raise LiteResearchRequestError("search query has no indexable terms")
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> list[dict[str, Any]]:
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
        limit = self.top_k if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise LiteResearchRequestError("search top_k must be a positive integer")
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
