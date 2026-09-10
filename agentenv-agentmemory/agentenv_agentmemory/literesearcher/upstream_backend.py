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
UPSTREAM_VISIT_ENDPOINT = "/visit_page"
UPSTREAM_VISIT_BACKEND = "postgresql_exact_url_goal_bm25_page_v1"
UPSTREAM_VISIT_PAGINATION_CONTRACT = "goal_bm25_overlapping_chars_v1"
UPSTREAM_VISIT_PAGE_CHARS = 8192
UPSTREAM_VISIT_PAGE_OVERLAP_CHARS = 1024
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
VISITABLE_SEARCH_PATH = "/search_visitable"


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
        filter_visitable: bool = False,
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
        if not isinstance(filter_visitable, bool):
            raise ValueError("filter_visitable must be a boolean")
        self.filter_visitable = filter_visitable
        self.search_path = VISITABLE_SEARCH_PATH if filter_visitable else "/search"

        health = self._request_json("/health")
        self.document_count = self._validate_health(health)
        probe = self._request_json(
            self.search_path,
            {
                "query": "LiteResearcher service identity check",
                "limit": 1,
                "search_type": UPSTREAM_SEARCH_TYPE,
                "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
                "dense_weight": UPSTREAM_DENSE_WEIGHT,
            },
        )
        self._parse_search_response(
            probe,
            expected_count=1,
            allow_partial=filter_visitable,
        )

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
        if payload.get("visit_page_endpoint") != UPSTREAM_VISIT_ENDPOINT:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit endpoint mismatch"
            )
        if payload.get("visit_page_backend") != UPSTREAM_VISIT_BACKEND:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit backend mismatch"
            )
        if (
            payload.get("visit_pagination_contract")
            != UPSTREAM_VISIT_PAGINATION_CONTRACT
        ):
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit pagination contract mismatch"
            )
        if payload.get("visit_page_chars") != UPSTREAM_VISIT_PAGE_CHARS:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit page size mismatch"
            )
        if (
            payload.get("visit_page_overlap_chars")
            != UPSTREAM_VISIT_PAGE_OVERLAP_CHARS
        ):
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit overlap mismatch"
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
        allow_partial: bool = False,
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
        count_valid = (
            isinstance(total, int)
            and not isinstance(total, bool)
            and total == len(results)
            and (
                total == expected_count
                if not allow_partial
                else 0 <= total <= expected_count
            )
        )
        if not count_valid:
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
            "search_path": self.search_path,
            "filter_visitable": self.filter_visitable,
            "visitable_search_contract": (
                "milvus_ranked_candidates_postgresql_exact_url_v1"
                if self.filter_visitable
                else None
            ),
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
        if self.filter_visitable:
            # Over-sample before the exact URL join; the service may return
            # fewer than ``limit`` rows after filtering, but never an
            # unvisitable URL.
            request_limit = min(50, max(limit * 4 + int(bool(masked)), limit))
        else:
            request_limit = min(50, limit + int(bool(masked)))
        payload = self._request_json(
            self.search_path,
            {
                "query": query_text,
                "limit": request_limit,
                "search_type": UPSTREAM_SEARCH_TYPE,
                "sparse_weight": UPSTREAM_SPARSE_WEIGHT,
                "dense_weight": UPSTREAM_DENSE_WEIGHT,
            },
        )
        hits = self._parse_search_response(
            payload,
            expected_count=request_limit,
            allow_partial=self.filter_visitable,
        )
        if masked:
            hits = [hit for hit in hits if hit["url"] != masked]
        hits = hits[:limit]
        return hits

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise LiteResearchRequestError("visit URL must be a non-empty string")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise LiteResearchRequestError("visit page must be a positive integer")
        requested_url = url.strip()
        goal_text = str(goal)
        payload = self._request_json(
            UPSTREAM_VISIT_ENDPOINT,
            {"url": requested_url, "goal": goal_text, "page": page},
        )
        if payload.get("service_id") != UPSTREAM_SERVICE_ID:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit identity mismatch"
            )
        if payload.get("backend") != UPSTREAM_VISIT_BACKEND:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit backend mismatch"
            )
        if (
            payload.get("pagination_contract")
            != UPSTREAM_VISIT_PAGINATION_CONTRACT
        ):
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit pagination contract mismatch"
            )
        if payload.get("page_chars") != UPSTREAM_VISIT_PAGE_CHARS:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit page size mismatch"
            )
        if (
            payload.get("page_overlap_chars")
            != UPSTREAM_VISIT_PAGE_OVERLAP_CHARS
        ):
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit overlap mismatch"
            )

        found = payload.get("found")
        resolved_url = payload.get("url")
        title = payload.get("title")
        content = payload.get("content")
        response_goal = payload.get("goal")
        response_page = payload.get("page")
        page_count = payload.get("page_count")
        next_page = payload.get("next_page")
        if (
            not isinstance(found, bool)
            or not isinstance(resolved_url, str)
            or not isinstance(title, str)
            or not isinstance(content, str)
            or not isinstance(response_goal, str)
            or isinstance(response_page, bool)
            or not isinstance(response_page, int)
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 0
            or (
                next_page is not None
                and (isinstance(next_page, bool) or not isinstance(next_page, int))
            )
        ):
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit schema mismatch"
            )
        self._finite_number(
            payload.get("visit_time"), "bounded Visit time", nonnegative=True
        )
        if resolved_url != requested_url:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit URL mismatch"
            )
        if response_goal != goal_text or response_page != page:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit request echo mismatch"
            )
        if not found:
            if title or content or page_count != 0 or next_page is not None:
                raise LiteResearchBackendError(
                    "LiteResearcher missing Visit response is inconsistent"
                )
            raise LiteResearchRequestError(
                "visit URL is outside the released corpus"
            )
        if page_count < 1:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit page count is invalid"
            )
        if page > page_count:
            if content or next_page is not None:
                raise LiteResearchBackendError(
                    "LiteResearcher out-of-range Visit response is inconsistent"
                )
            raise LiteResearchRequestError(
                f"visit page {page} exceeds page_count {page_count}"
            )
        expected_next_page = page + 1 if page < page_count else None
        if not content or len(content) > UPSTREAM_VISIT_PAGE_CHARS:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit content is invalid"
            )
        if next_page != expected_next_page:
            raise LiteResearchBackendError(
                "LiteResearcher bounded Visit next page mismatch"
            )
        return {
            "url": resolved_url,
            "title": title,
            "content": content,
            "goal": response_goal,
            "page": response_page,
            "page_count": page_count,
            "next_page": next_page,
        }
