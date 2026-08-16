from __future__ import annotations

import hashlib
import json
import math
import socket
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
UPSTREAM_DOCUMENT_COUNT = 32_127_370
UPSTREAM_INDEX_TYPE = "DISKANN"
UPSTREAM_SPARSE_INDEX_TYPE = "SPARSE_INVERTED_INDEX"
UPSTREAM_VECTOR_DTYPE = "FP32"
UPSTREAM_SEARCH_TYPE = "hybrid"
UPSTREAM_SPARSE_WEIGHT = 0.7
UPSTREAM_DENSE_WEIGHT = 1.0
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
UPSTREAM_BACKEND_CONTRACT = "literesearcher_upstream_hybrid_diskann_v1"
UPSTREAM_TELEMETRY_CONTRACT = "agentmemory_literesearcher_backend_timing_v1"


@dataclass(frozen=True)
class _ParsedSearchResponse:
    results: tuple[dict[str, Any], ...]
    search_seconds: float
    embedding_seconds: float
    milvus_seconds: float


@dataclass(frozen=True)
class UpstreamSearchCall:
    results: tuple[dict[str, Any], ...]
    search_seconds_by_query: tuple[float, ...]
    embedding_seconds_by_query: tuple[float, ...]
    milvus_seconds_by_query: tuple[float, ...]

    def public_results(self) -> list[dict[str, Any]]:
        return [dict(result) for result in self.results]

    def timing_evidence(self) -> dict[str, Any]:
        return {
            "backend_telemetry_schema": UPSTREAM_TELEMETRY_CONTRACT,
            "backend_query_count": len(self.search_seconds_by_query),
            "backend_reported_search_seconds_by_query": list(
                self.search_seconds_by_query
            ),
            "backend_reported_embedding_seconds_by_query": list(
                self.embedding_seconds_by_query
            ),
            "backend_reported_milvus_seconds_by_query": list(
                self.milvus_seconds_by_query
            ),
        }


@dataclass(frozen=True)
class UpstreamVisitCall:
    page: Mapping[str, Any]
    visit_seconds: float

    def public_page(self) -> dict[str, Any]:
        return dict(self.page)

    def timing_evidence(self) -> dict[str, Any]:
        return {
            "backend_telemetry_schema": UPSTREAM_TELEMETRY_CONTRACT,
            "backend_reported_visit_seconds": self.visit_seconds,
        }


class UpstreamHybridLiteResearchBackend:
    """Fail-closed client for LiteResearcher's released hybrid RAG service."""

    contract_id = UPSTREAM_BACKEND_CONTRACT

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
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")

        self.tasks_source = tasks
        self.split = "train"
        self.service_url = service_url.rstrip("/")
        self.top_k = top_k
        self.timeout_seconds = float(timeout_seconds)

        health = self._request_json("/health")
        self._validate_health(health)
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
        self._parse_search_response(probe)

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
    def _validate_health(payload: Mapping[str, Any]) -> None:
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
        for field in ("document_count", "loaded_count"):
            value = payload.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != UPSTREAM_DOCUMENT_COUNT
            ):
                raise LiteResearchBackendError(
                    f"LiteResearcher {field} does not match the released corpus"
                )
        if payload.get("index_type") != UPSTREAM_INDEX_TYPE:
            raise LiteResearchBackendError("LiteResearcher upstream is not using DISKANN")
        if payload.get("sparse_index_type") != UPSTREAM_SPARSE_INDEX_TYPE:
            raise LiteResearchBackendError(
                "LiteResearcher upstream is not using the sparse inverted index"
            )
        if payload.get("vector_precision") != UPSTREAM_VECTOR_DTYPE:
            raise LiteResearchBackendError("LiteResearcher upstream is not using FP32")
        if payload.get("default_search_type") != UPSTREAM_SEARCH_TYPE:
            raise LiteResearchBackendError("LiteResearcher upstream default is not hybrid")
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
    ) -> _ParsedSearchResponse:
        if payload.get("server_id") != UPSTREAM_SERVER_ID:
            raise LiteResearchBackendError("LiteResearcher upstream server identity mismatch")
        if payload.get("service_id") != UPSTREAM_SERVICE_ID:
            raise LiteResearchBackendError("LiteResearcher upstream service identity mismatch")
        for field in ("document_count", "loaded_count"):
            value = payload.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != UPSTREAM_DOCUMENT_COUNT
            ):
                raise LiteResearchBackendError(
                    f"LiteResearcher search {field} does not match the released corpus"
                )
        if payload.get("index_type") != UPSTREAM_INDEX_TYPE:
            raise LiteResearchBackendError("LiteResearcher search did not use DISKANN")
        if payload.get("sparse_index_type") != UPSTREAM_SPARSE_INDEX_TYPE:
            raise LiteResearchBackendError(
                "LiteResearcher search did not use the sparse inverted index"
            )
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
        search_seconds = self._finite_number(
            payload.get("search_time"), "search_time", nonnegative=True
        )
        embedding_seconds = self._finite_number(
            payload.get("embedding_time"), "embedding_time", nonnegative=True
        )
        milvus_seconds = self._finite_number(
            payload.get("milvus_time"), "milvus_time", nonnegative=True
        )

        results = payload.get("results")
        total = payload.get("total")
        if not isinstance(results, list):
            raise LiteResearchBackendError("LiteResearcher results must be a list")
        if isinstance(total, bool) or not isinstance(total, int) or total != len(results):
            raise LiteResearchBackendError("LiteResearcher result count mismatch")

        parsed: list[dict[str, Any]] = []
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
            self._finite_number(item.get("score"), "result score")
            parsed.append(
                SearchHit(
                    url=link,
                    title=title,
                    snippet=snippet,
                    rank=rank,
                ).public_record()
            )
        return _ParsedSearchResponse(
            results=tuple(parsed),
            search_seconds=search_seconds,
            embedding_seconds=embedding_seconds,
            milvus_seconds=milvus_seconds,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_contract": self.contract_id,
            "backend_class": "upstream_hybrid",
            "upstream_hybrid_backend": True,
            "server_id": UPSTREAM_SERVER_ID,
            "service_id": UPSTREAM_SERVICE_ID,
            "document_count": UPSTREAM_DOCUMENT_COUNT,
            "index_type": UPSTREAM_INDEX_TYPE,
            "sparse_index_type": UPSTREAM_SPARSE_INDEX_TYPE,
            "vector_dtype": UPSTREAM_VECTOR_DTYPE,
            "search_type": UPSTREAM_SEARCH_TYPE,
            "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
            "dense_weight": UPSTREAM_DENSE_WEIGHT,
            "service_identity_verified": True,
            "service_endpoint_sha256": hashlib.sha256(
                self.service_url.encode("utf-8")
            ).hexdigest(),
            "search_result_url_namespace": "released_public_url_v1",
            "search_exposes_mask_url": False,
            "search_exposes_targets": False,
            "search_exposes_scores": False,
            "search_query_contract": "parallel_per_query_exact_url_dedup_v1",
            "visit_exposes_mask_url": False,
            "visit_pagination_contract": "goal_bm25_overlapping_chars_v1",
            "live_network": False,
            "lexical_fallback": False,
            "failure_mode": "fail_closed",
            "timeout_seconds": self.timeout_seconds,
            "per_call_telemetry_contract": UPSTREAM_TELEMETRY_CONTRACT,
        }

    def _search_one(self, query_text: str, limit: int) -> _ParsedSearchResponse:
        payload = self._request_json(
            "/search",
            {
                "query": query_text,
                "limit": limit,
                "search_type": UPSTREAM_SEARCH_TYPE,
                "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
                "dense_weight": UPSTREAM_DENSE_WEIGHT,
            },
        )
        parsed = self._parse_search_response(payload)
        if len(parsed.results) != limit:
            raise LiteResearchBackendError(
                "LiteResearcher upstream did not return the requested result count"
            )
        return parsed

    def search_with_telemetry(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> UpstreamSearchCall:
        queries = [query] if isinstance(query, str) else query
        if not isinstance(queries, list) or not queries or any(
            not isinstance(item, str) for item in queries
        ):
            raise LiteResearchRequestError(
                "search query must be a non-empty string or list of strings"
            )
        query_texts = [item.strip() for item in queries]
        if any(not item for item in query_texts):
            raise LiteResearchRequestError("search queries must not be empty")
        limit = self.top_k if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise LiteResearchRequestError("search top_k must be an integer in [1, 50]")

        if len(query_texts) == 1:
            result_batches = [self._search_one(query_texts[0], limit)]
        else:
            with ThreadPoolExecutor(max_workers=min(len(query_texts), 32)) as executor:
                futures = [
                    executor.submit(self._search_one, query_text, limit)
                    for query_text in query_texts
                ]
                result_batches = [future.result() for future in futures]

        masked = str(mask_url).strip()
        hits: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for query_text, batch in zip(query_texts, result_batches):
            query_rank = 0
            for hit in batch.results:
                url = hit["url"]
                if (masked and masked == url) or url in seen_urls:
                    continue
                seen_urls.add(url)
                query_rank += 1
                public_hit = dict(hit)
                public_hit["rank"] = query_rank
                public_hit["query"] = query_text
                hits.append(public_hit)
        return UpstreamSearchCall(
            results=tuple(hits),
            search_seconds_by_query=tuple(
                batch.search_seconds for batch in result_batches
            ),
            embedding_seconds_by_query=tuple(
                batch.embedding_seconds for batch in result_batches
            ),
            milvus_seconds_by_query=tuple(
                batch.milvus_seconds for batch in result_batches
            ),
        )

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> list[dict[str, Any]]:
        return self.search_with_telemetry(
            query,
            top_k=top_k,
            mask_url=mask_url,
        ).public_results()

    def visit_with_telemetry(
        self, url: str, *, goal: str = "", page: int = 1
    ) -> UpstreamVisitCall:
        if not isinstance(url, str) or not url.strip():
            raise LiteResearchRequestError("visit URL must be a non-empty string")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise LiteResearchRequestError("visit page must be a positive integer")
        requested_url = url.strip()
        payload = self._request_json("/web_parser", {"url": requested_url})
        found = payload.get("found")
        resolved_url = payload.get("url")
        if not isinstance(found, bool) or not isinstance(resolved_url, str):
            raise LiteResearchBackendError("LiteResearcher web_parser schema mismatch")
        if resolved_url != requested_url:
            raise LiteResearchBackendError("LiteResearcher web_parser URL mismatch")
        if payload.get("service_id") != UPSTREAM_SERVICE_ID:
            raise LiteResearchBackendError("LiteResearcher web_parser identity mismatch")
        if payload.get("backend") != "postgresql_exact_url":
            raise LiteResearchBackendError("LiteResearcher web_parser backend mismatch")
        visit_seconds = self._finite_number(
            payload.get("visit_time"), "visit_time", nonnegative=True
        )
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
        return UpstreamVisitCall(
            page={
                "url": resolved_url,
                "title": title,
                "content": pages[page - 1],
                "goal": str(goal),
                "page": page,
                "page_count": len(pages),
                "next_page": page + 1 if page < len(pages) else None,
            },
            visit_seconds=visit_seconds,
        )

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        return self.visit_with_telemetry(url, goal=goal, page=page).public_page()
