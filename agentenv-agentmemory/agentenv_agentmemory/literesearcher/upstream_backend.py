from __future__ import annotations

import hashlib
import json
import math
import socket
from collections.abc import Mapping
from typing import Any
from urllib import error, parse, request

from .backend import (
    LiteResearchBackendError,
    LiteResearchRequestError,
    SearchHit,
    _rank_windows_by_goal,
)


UPSTREAM_SERVER_ID = "local_rag_diskann_server"
UPSTREAM_SERVICE_ID = "literesearcher-i-bgem3-diskann-v1"
UPSTREAM_INDEX_TYPE = "DISKANN"
UPSTREAM_SPARSE_INDEX_TYPE = "SPARSE_INVERTED_INDEX"
UPSTREAM_VECTOR_DTYPE = "FP32"
UPSTREAM_SEARCH_TYPE = "hybrid"
UPSTREAM_SPARSE_WEIGHT = 0.7
UPSTREAM_DENSE_WEIGHT = 1.0
UPSTREAM_DOCUMENT_COUNT = 32_127_370
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class UpstreamHybridLiteResearchBackend:
    """Fail-closed client for LiteResearcher's released hybrid RAG service."""

    contract_id = "literesearcher_upstream_hybrid_diskann_v1"

    def __init__(
        self,
        tasks,
        service_url: str,
        *,
        top_k: int = 5,
        timeout_seconds: float = 120.0,
    ) -> None:
        parsed = parse.urlparse(service_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("service_url must be a plain HTTP(S) service URL")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be an integer in [1, 50]")
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.tasks_source = tasks
        self.split = "train"
        self.service_url = service_url.rstrip("/")
        self.top_k = top_k
        self.timeout_seconds = float(timeout_seconds)

        health = self._request_json("/health")
        self.document_count = self._validate_health(health)
        probe = self._request_json(
            "/search",
            {
                "query": "LiteResearcher service identity check",
                "limit": 1,
                "search_type": UPSTREAM_SEARCH_TYPE,
                "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
                "dense_weight": UPSTREAM_DENSE_WEIGHT,
            },
        )
        self._parse_search_response(probe, expected_count=1)

    def _request_json(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        method = "GET"
        if payload is not None:
            data = json.dumps(dict(payload), ensure_ascii=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        http_request = request.Request(
            f"{self.service_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            raise LiteResearchBackendError(
                f"LiteResearcher upstream returned HTTP {exc.code}"
            ) from exc
        except (error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise LiteResearchBackendError(
                "LiteResearcher upstream request failed"
            ) from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LiteResearchBackendError("LiteResearcher upstream response is too large")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiteResearchBackendError(
                "LiteResearcher upstream returned malformed JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise LiteResearchBackendError(
                "LiteResearcher upstream response must be a JSON object"
            )
        return decoded

    @staticmethod
    def _validate_health(payload: Mapping[str, Any]) -> int:
        if payload.get("status") != "healthy":
            raise LiteResearchBackendError("LiteResearcher upstream is not healthy")
        if payload.get("server_id") != UPSTREAM_SERVER_ID:
            raise LiteResearchBackendError("LiteResearcher upstream server identity mismatch")
        if payload.get("service_id") != UPSTREAM_SERVICE_ID:
            raise LiteResearchBackendError("LiteResearcher upstream service identity mismatch")
        if payload.get("milvus_loaded") is not True:
            raise LiteResearchBackendError("LiteResearcher Milvus collection is not loaded")
        if payload.get("redis_connected") is not True:
            raise LiteResearchBackendError("LiteResearcher embedding queue is not connected")
        queue_size = payload.get("embedding_queue_size")
        if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 0:
            raise LiteResearchBackendError("LiteResearcher embedding queue size is invalid")
        if payload.get("index_type") != UPSTREAM_INDEX_TYPE:
            raise LiteResearchBackendError("LiteResearcher upstream is not using DISKANN")
        if payload.get("vector_precision") != UPSTREAM_VECTOR_DTYPE:
            raise LiteResearchBackendError("LiteResearcher upstream is not using FP32")
        if payload.get("vector_dtype") != UPSTREAM_VECTOR_DTYPE:
            raise LiteResearchBackendError("LiteResearcher upstream vector dtype mismatch")
        if payload.get("sparse_index_type") != UPSTREAM_SPARSE_INDEX_TYPE:
            raise LiteResearchBackendError("LiteResearcher sparse index type mismatch")
        if payload.get("default_search_type") != UPSTREAM_SEARCH_TYPE:
            raise LiteResearchBackendError("LiteResearcher default search type mismatch")
        weights = payload.get("hybrid_weights")
        if not isinstance(weights, Mapping):
            raise LiteResearchBackendError("LiteResearcher hybrid weights are missing")
        sparse_weight = UpstreamHybridLiteResearchBackend._finite_number(
            weights.get("sparse"), "health sparse weight"
        )
        dense_weight = UpstreamHybridLiteResearchBackend._finite_number(
            weights.get("dense"), "health dense weight"
        )
        if not math.isclose(sparse_weight, UPSTREAM_SPARSE_WEIGHT, abs_tol=1e-12):
            raise LiteResearchBackendError("LiteResearcher health sparse weight mismatch")
        if not math.isclose(dense_weight, UPSTREAM_DENSE_WEIGHT, abs_tol=1e-12):
            raise LiteResearchBackendError("LiteResearcher health dense weight mismatch")
        document_count = payload.get("collection_entities")
        if (
            isinstance(document_count, bool)
            or not isinstance(document_count, int)
            or document_count != UPSTREAM_DOCUMENT_COUNT
            or payload.get("document_count") != document_count
            or payload.get("loaded_count") != document_count
        ):
            raise LiteResearchBackendError(
                "LiteResearcher upstream collection entity count mismatch"
            )
        return document_count

    @staticmethod
    def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LiteResearchBackendError(f"LiteResearcher {field} must be numeric")
        result = float(value)
        if not math.isfinite(result) or (nonnegative and result < 0):
            raise LiteResearchBackendError(f"LiteResearcher {field} is invalid")
        return result

    def _parse_search_response(
        self,
        payload: Mapping[str, Any],
        *,
        expected_count: int,
    ) -> list[dict[str, Any]]:
        if payload.get("server_id") != UPSTREAM_SERVER_ID:
            raise LiteResearchBackendError("LiteResearcher upstream server identity mismatch")
        if payload.get("service_id") != UPSTREAM_SERVICE_ID:
            raise LiteResearchBackendError("LiteResearcher upstream service identity mismatch")
        if payload.get("index_type") != UPSTREAM_INDEX_TYPE:
            raise LiteResearchBackendError("LiteResearcher search did not use DISKANN")
        if payload.get("sparse_index_type") != UPSTREAM_SPARSE_INDEX_TYPE:
            raise LiteResearchBackendError("LiteResearcher search sparse index mismatch")
        if (
            payload.get("document_count") != UPSTREAM_DOCUMENT_COUNT
            or payload.get("loaded_count") != UPSTREAM_DOCUMENT_COUNT
        ):
            raise LiteResearchBackendError("LiteResearcher search document count mismatch")
        if payload.get("vector_dtype") != UPSTREAM_VECTOR_DTYPE:
            raise LiteResearchBackendError("LiteResearcher search did not use FP32")
        if payload.get("search_type") != UPSTREAM_SEARCH_TYPE:
            raise LiteResearchBackendError("LiteResearcher search was not hybrid")
        sparse_weight = self._finite_number(payload.get("sparse_weight"), "sparse_weight")
        dense_weight = self._finite_number(payload.get("dense_weight"), "dense_weight")
        if not math.isclose(sparse_weight, UPSTREAM_SPARSE_WEIGHT, abs_tol=1e-12):
            raise LiteResearchBackendError("LiteResearcher sparse weight mismatch")
        if not math.isclose(dense_weight, UPSTREAM_DENSE_WEIGHT, abs_tol=1e-12):
            raise LiteResearchBackendError("LiteResearcher dense weight mismatch")
        for field in ("search_time", "embedding_time", "milvus_time"):
            self._finite_number(payload.get(field), field, nonnegative=True)

        results = payload.get("results")
        total = payload.get("total")
        if not isinstance(results, list):
            raise LiteResearchBackendError("LiteResearcher results must be a list")
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total != len(results)
            or total != expected_count
        ):
            raise LiteResearchBackendError("LiteResearcher result count mismatch")

        parsed: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for rank, item in enumerate(results, start=1):
            if not isinstance(item, Mapping):
                raise LiteResearchBackendError("LiteResearcher result must be an object")
            link = item.get("link")
            title = item.get("title")
            snippet = item.get("snippet")
            if not isinstance(link, str) or not link.strip():
                raise LiteResearchBackendError("LiteResearcher result link is invalid")
            if not isinstance(title, str) or not isinstance(snippet, str):
                raise LiteResearchBackendError("LiteResearcher result text is invalid")
            link = link.strip()
            if link in seen_urls:
                raise LiteResearchBackendError("LiteResearcher returned a duplicate URL")
            seen_urls.add(link)
            score = self._finite_number(item.get("score"), "result score")
            parsed.append(
                SearchHit(
                    url=link,
                    title=title,
                    snippet=snippet,
                    rank=rank,
                    score=score,
                ).public_record()
            )
        return parsed

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_contract": self.contract_id,
            "backend_class": "upstream_hybrid",
            "upstream_hybrid_backend": True,
            "server_id": UPSTREAM_SERVER_ID,
            "service_id": UPSTREAM_SERVICE_ID,
            "index_type": UPSTREAM_INDEX_TYPE,
            "sparse_index_type": UPSTREAM_SPARSE_INDEX_TYPE,
            "vector_dtype": UPSTREAM_VECTOR_DTYPE,
            "search_type": UPSTREAM_SEARCH_TYPE,
            "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
            "dense_weight": UPSTREAM_DENSE_WEIGHT,
            "document_count": self.document_count,
            "service_identity_verified": True,
            "service_endpoint_sha256": hashlib.sha256(
                self.service_url.encode("utf-8")
            ).hexdigest(),
            "search_result_url_namespace": "released_public_url_v1",
            "search_exposes_mask_url": False,
            "search_exposes_targets": False,
            "visit_exposes_mask_url": False,
            "visit_pagination_contract": "goal_bm25_overlapping_chars_v1",
            "live_network": False,
            "lexical_fallback": False,
            "failure_mode": "fail_closed",
            "timeout_seconds": self.timeout_seconds,
        }

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
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise LiteResearchRequestError("search top_k must be an integer in [1, 50]")
        masked = str(mask_url).strip()
        request_limit = min(50, limit + int(bool(masked)))
        payload = self._request_json(
            "/search",
            {
                "query": query_text,
                "limit": request_limit,
                "search_type": UPSTREAM_SEARCH_TYPE,
                "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
                "dense_weight": UPSTREAM_DENSE_WEIGHT,
            },
        )
        hits = self._parse_search_response(payload, expected_count=request_limit)
        if masked:
            hits = [hit for hit in hits if hit["url"] != masked]
        return hits[:limit]

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise LiteResearchRequestError("visit URL must be a non-empty string")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise LiteResearchRequestError("visit page must be a positive integer")
        requested_url = url.strip()
        payload = self._request_json("/web_parser", {"url": requested_url})
        found = payload.get("found")
        resolved_url = payload.get("url")
        if payload.get("service_id") != UPSTREAM_SERVICE_ID:
            raise LiteResearchBackendError("LiteResearcher web_parser identity mismatch")
        if payload.get("backend") != "postgresql_exact_url":
            raise LiteResearchBackendError("LiteResearcher web_parser backend mismatch")
        if not isinstance(found, bool) or not isinstance(resolved_url, str):
            raise LiteResearchBackendError("LiteResearcher web_parser schema mismatch")
        if resolved_url != requested_url:
            raise LiteResearchBackendError("LiteResearcher web_parser URL mismatch")
        if not found:
            raise LiteResearchRequestError("visit URL is outside the released corpus")
        title = payload.get("title")
        text = payload.get("text")
        if not isinstance(title, str) or not isinstance(text, str) or not text:
            raise LiteResearchBackendError("LiteResearcher web_parser text is invalid")
        pages = _rank_windows_by_goal(text, str(goal))
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
