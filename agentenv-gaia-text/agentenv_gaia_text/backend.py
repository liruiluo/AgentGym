from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BackendError(RuntimeError):
    """Backend infrastructure failed; callers must not fall back to live web."""


class RequestError(ValueError):
    """A policy request is invalid for the frozen backend."""


@dataclass(frozen=True)
class _Document:
    url: str
    title: str
    content: str


class FixtureBackend:
    contract_id = "gaia_text_external_fixture_search_visit_v1"

    def __init__(
        self,
        documents: tuple[_Document, ...],
        *,
        asset_sha256: str,
        page_chars: int = 8192,
    ) -> None:
        if type(page_chars) is not int or page_chars <= 0:
            raise ValueError("page_chars must be a positive integer")
        self._documents = documents
        self._by_url = {document.url: document for document in documents}
        self.asset_sha256 = asset_sha256
        self.page_chars = page_chars
        self._trace: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        asset_path: str | Path,
        expected_sha256: str,
        *,
        page_chars: int = 8192,
    ) -> FixtureBackend:
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError("backend expected SHA-256 must be a lowercase digest")
        path = Path(asset_path).expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError("backend asset path must be a real file")
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected_sha256:
            raise ValueError(
                "backend asset SHA-256 mismatch: "
                f"expected {expected_sha256}, got {observed}"
            )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("backend asset must be a UTF-8 JSON object") from exc
        if not isinstance(payload, dict) or set(payload) != {"schema", "documents"}:
            raise ValueError("backend asset must contain exactly schema and documents")
        if payload["schema"] != "gaia_text_fixture_backend_v1":
            raise ValueError("backend asset has an unsupported schema")
        records = payload["documents"]
        if not isinstance(records, list) or not records:
            raise ValueError("backend asset documents must be a non-empty list")
        documents = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {
                "url",
                "title",
                "content",
            }:
                raise ValueError(f"backend document {index} has an invalid schema")
            url = _text(record["url"], f"backend document {index} URL")
            title = _text(record["title"], f"backend document {index} title")
            content = _text(record["content"], f"backend document {index} content")
            if not url.startswith("gaia-text://"):
                raise ValueError(
                    "fixture backend URLs must use the opaque gaia-text scheme"
                )
            documents.append(_Document(url=url, title=title, content=content))
        if len({document.url for document in documents}) != len(documents):
            raise ValueError("backend document URLs must be unique")
        return cls(tuple(documents), asset_sha256=observed, page_chars=page_chars)

    @property
    def call_trace(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(deepcopy(self._trace))

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_contract": self.contract_id,
            "asset_sha256": self.asset_sha256,
            "document_count": len(self._documents),
            "live_network": False,
            "failure_mode": "fail_closed",
            "page_chars": self.page_chars,
        }

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        queries = [query] if isinstance(query, str) else query
        if (
            not isinstance(queries, list)
            or not queries
            or any(not isinstance(item, str) for item in queries)
        ):
            raise RequestError(
                "search query must be a non-empty string or list of strings"
            )
        query_text = " ".join(item.strip() for item in queries).strip()
        if not query_text:
            raise RequestError("search query must be non-empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RequestError("search top_k must be a positive integer")
        query_tokens = set(_tokens(query_text))
        scored = []
        for index, document in enumerate(self._documents):
            overlap = len(
                query_tokens & set(_tokens(document.title + " " + document.content))
            )
            if overlap:
                scored.append((-overlap, index, document))
        scored.sort(key=lambda item: (item[0], item[1]))
        results = [
            {
                "url": document.url,
                "title": document.title,
                "snippet": document.content[:240],
                "rank": rank,
            }
            for rank, (_, _, document) in enumerate(scored[:top_k], 1)
        ]
        self._record({"action": "search", "query": list(queries), "top_k": top_k})
        return results

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise RequestError("visit URL must be a non-empty string")
        if not isinstance(goal, str):
            raise RequestError("visit goal must be text")
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise RequestError("visit page must be a positive integer")
        try:
            document = self._by_url[url.strip()]
        except KeyError as exc:
            raise RequestError(
                "visit URL is outside the verified backend asset"
            ) from exc
        pages = tuple(
            document.content[start : start + self.page_chars]
            for start in range(0, len(document.content), self.page_chars)
        )
        if goal.strip() and len(pages) > 1:
            goal_tokens = set(_tokens(goal))
            pages = tuple(
                value
                for _, _, value in sorted(
                    (
                        (-len(goal_tokens & set(_tokens(value))), index, value)
                        for index, value in enumerate(pages)
                    ),
                    key=lambda item: (item[0], item[1]),
                )
            )
        if page > len(pages):
            raise RequestError(f"visit page {page} exceeds page_count {len(pages)}")
        result = {
            "url": document.url,
            "title": document.title,
            "content": pages[page - 1],
            "goal": goal,
            "page": page,
            "page_count": len(pages),
            "next_page": page + 1 if page < len(pages) else None,
        }
        self._record(
            {"action": "visit", "url": document.url, "goal": goal, "page": page}
        )
        return result

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._trace.append(deepcopy(event))


def _tokens(value: str) -> Iterable[str]:
    return re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text without NUL bytes")
    return value
