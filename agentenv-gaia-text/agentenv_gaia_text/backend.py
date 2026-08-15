from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx

LITERESEARCHER_SOURCE_REVISION = "779e7d5f6a043d4100149ba0992a39507f69a974"
_LITERESEARCHER_ENDPOINT_CONTRACT = {
    "schema": "literesearcher_local_rag_http_v1",
    "health": {
        "method": "GET",
        "path": "/health",
        "response_fields": [
            "status",
            "milvus_loaded",
            "collection_entities",
            "redis_connected",
            "embedding_queue_size",
            "vector_precision",
            "index_type",
            "stats",
        ],
        "stats_fields": [
            "active_requests",
            "total_requests",
            "success_count",
            "error_count",
            "requests_per_second",
            "avg_response_ms",
        ],
        "required_ready_fields": {
            "status": "healthy",
            "milvus_loaded": True,
            "redis_connected": True,
            "vector_precision": "FP32",
            "index_type": "DISKANN",
        },
    },
    "search": {
        "method": "POST",
        "path": "/search",
        "request_fields": [
            "query",
            "limit",
            "search_type",
            "sparse_weight",
            "dense_weight",
        ],
        "fixed_request": {
            "search_type": "hybrid",
            "sparse_weight": 0.7,
            "dense_weight": 1.0,
        },
        "fixed_response": {
            "search_type": "hybrid",
            "sparse_weight": 0.7,
            "dense_weight": 1.0,
            "server_id": "local_rag_diskann_server",
            "vector_dtype": "FP32",
            "index_type": "DISKANN",
        },
        "limit_range": [1, 50],
        "response_fields": [
            "results",
            "total",
            "search_time",
            "embedding_time",
            "milvus_time",
            "search_type",
            "sparse_weight",
            "dense_weight",
            "server_id",
            "vector_dtype",
            "index_type",
        ],
        "result_fields": ["link", "title", "snippet", "score"],
    },
    "visit": {
        "method": "POST",
        "path": "/web_parser",
        "request_fields": ["url"],
        "response_fields": ["found", "url", "title", "text"],
        "not_found_is_terminal_backend_error": False,
    },
    "redirects": "forbidden",
    "live_web_fallback": False,
}
LITERESEARCHER_ENDPOINT_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _LITERESEARCHER_ENDPOINT_CONTRACT,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_CERTIFICATE_SCHEMA = "gaia_text_literesearcher_service_certificate_v1"
_CERTIFICATE_FIELDS = {
    "schema",
    "upstream_source_revision",
    "endpoint_contract_sha256",
    "endpoint_origin_sha256",
    "search_corpus_certificate_sha256",
    "search_index_certificate_sha256",
    "browse_store_certificate_sha256",
}
_SEARCH_RESPONSE_FIELDS = set(
    _LITERESEARCHER_ENDPOINT_CONTRACT["search"]["response_fields"]
)
_SEARCH_RESULT_FIELDS = set(
    _LITERESEARCHER_ENDPOINT_CONTRACT["search"]["result_fields"]
)
_VISIT_RESPONSE_FIELDS = set(
    _LITERESEARCHER_ENDPOINT_CONTRACT["visit"]["response_fields"]
)
_HEALTH_RESPONSE_FIELDS = set(
    _LITERESEARCHER_ENDPOINT_CONTRACT["health"]["response_fields"]
)
_HEALTH_STATS_FIELDS = set(
    _LITERESEARCHER_ENDPOINT_CONTRACT["health"]["stats_fields"]
)
_HEALTH_REQUIRED_READY_FIELDS = _LITERESEARCHER_ENDPOINT_CONTRACT["health"][
    "required_ready_fields"
]
_SEARCH_FIXED_REQUEST = _LITERESEARCHER_ENDPOINT_CONTRACT["search"][
    "fixed_request"
]
_SEARCH_FIXED_RESPONSE = _LITERESEARCHER_ENDPOINT_CONTRACT["search"][
    "fixed_response"
]
_RETRIABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class BackendError(RuntimeError):
    """Backend infrastructure failed; callers must not fall back to live web."""


class BackendTimeoutError(BackendError):
    """The configured production service did not respond within its timeout."""


class BackendConnectionError(BackendError):
    """The configured production service could not be reached."""


class BackendHTTPError(BackendError):
    """The configured production service returned a non-success status."""


class BackendProtocolError(BackendError):
    """The configured production service violated its pinned response contract."""


class RequestError(ValueError):
    """A policy request is invalid for the frozen backend."""


@runtime_checkable
class SearchVisitBackend(Protocol):
    """Arm-neutral backend contract consumed by the GAIA-Text dispatcher."""

    def metadata(self) -> dict[str, Any]: ...

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]: ...

    def visit(
        self,
        url: str,
        *,
        goal: str = "",
        page: int = 1,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _ServiceCertificate:
    service_certificate_sha256: str
    endpoint_origin_sha256: str
    search_corpus_certificate_sha256: str
    search_index_certificate_sha256: str
    browse_store_certificate_sha256: str

    def metadata(self) -> dict[str, str]:
        return {
            "service_certificate_sha256": self.service_certificate_sha256,
            "upstream_source_revision": LITERESEARCHER_SOURCE_REVISION,
            "endpoint_contract_sha256": LITERESEARCHER_ENDPOINT_CONTRACT_SHA256,
            "endpoint_origin_sha256": self.endpoint_origin_sha256,
            "search_corpus_certificate_sha256": (
                self.search_corpus_certificate_sha256
            ),
            "search_index_certificate_sha256": self.search_index_certificate_sha256,
            "browse_store_certificate_sha256": self.browse_store_certificate_sha256,
        }


class LiteResearcherBackend:
    """Strict client for the pinned LiteResearcher local retrieval stack."""

    contract_id = "gaia_text_literesearcher_search_visit_v1"

    def __init__(
        self,
        certificate: _ServiceCertificate,
        *,
        base_url: str,
        connect_timeout_ms: int,
        read_timeout_ms: int,
        retry_count: int,
        retry_backoff_ms: int,
        result_limit: int,
        page_chars: int,
        page_limit: int,
    ) -> None:
        self._base_url = base_url
        self._certificate = certificate
        self.connect_timeout_ms = connect_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.retry_count = retry_count
        self.retry_backoff_ms = retry_backoff_ms
        self.result_limit = result_limit
        self.page_chars = page_chars
        self.page_limit = page_limit
        self.service_certificate_sha256 = certificate.service_certificate_sha256
        self._client = httpx.Client(
            base_url=base_url,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                connect=connect_timeout_ms / 1000,
                read=read_timeout_ms / 1000,
                write=read_timeout_ms / 1000,
                pool=connect_timeout_ms / 1000,
            ),
            headers={
                "Accept": "application/json",
                "User-Agent": "agentenv-gaia-text/literesearcher-pinned",
            },
        )

    @classmethod
    def load(
        cls,
        certificate_path: str | Path,
        expected_certificate_sha256: str,
        *,
        base_url: str,
        connect_timeout_ms: int,
        read_timeout_ms: int,
        retry_count: int,
        retry_backoff_ms: int,
        result_limit: int,
        page_chars: int,
        page_limit: int,
    ) -> LiteResearcherBackend:
        normalized_origin = _normalize_origin(base_url)
        _bounded_integer(
            connect_timeout_ms, "connect_timeout_ms", minimum=1, maximum=300_000
        )
        _bounded_integer(
            read_timeout_ms, "read_timeout_ms", minimum=1, maximum=300_000
        )
        _bounded_integer(retry_count, "retry_count", minimum=0, maximum=10)
        _bounded_integer(
            retry_backoff_ms, "retry_backoff_ms", minimum=0, maximum=60_000
        )
        _bounded_integer(result_limit, "result_limit", minimum=1, maximum=50)
        _bounded_integer(page_chars, "page_chars", minimum=1, maximum=1_000_000)
        _bounded_integer(page_limit, "page_limit", minimum=1, maximum=10_000)
        if page_chars * page_limit > 16_000_000:
            raise ValueError("configured visit page capacity exceeds the safe limit")
        certificate = _load_service_certificate(
            certificate_path,
            expected_certificate_sha256,
            endpoint_origin_sha256=hashlib.sha256(
                normalized_origin.encode("utf-8")
            ).hexdigest(),
        )
        backend = cls(
            certificate,
            base_url=normalized_origin,
            connect_timeout_ms=connect_timeout_ms,
            read_timeout_ms=read_timeout_ms,
            retry_count=retry_count,
            retry_backoff_ms=retry_backoff_ms,
            result_limit=result_limit,
            page_chars=page_chars,
            page_limit=page_limit,
        )
        try:
            backend.probe()
        except BaseException:
            backend.close()
            raise
        return backend

    def metadata(self) -> dict[str, Any]:
        identity = {
            "backend_contract": self.contract_id,
            "backend_kind": "production",
            **self._certificate.metadata(),
            "connect_timeout_ms": self.connect_timeout_ms,
            "read_timeout_ms": self.read_timeout_ms,
            "retry_count": self.retry_count,
            "retry_backoff_ms": self.retry_backoff_ms,
            "result_limit": self.result_limit,
            "page_chars": self.page_chars,
            "page_limit": self.page_limit,
            "network_access": "configured_literesearcher_origin_only",
            "live_web_fallback": False,
            "failure_mode": "fail_closed",
        }
        identity_sha256 = hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            **identity,
            # Legacy client field: for production this is the digest of every
            # stable backend identity field above, not a filesystem asset.
            "asset_sha256": identity_sha256,
            "runtime_identity_sha256": identity_sha256,
        }

    def probe(self) -> None:
        payload = self._request(
            "health",
            "GET",
            "/health",
            body=None,
            response_byte_limit=256_000,
        )
        if not _valid_health_response(payload):
            raise BackendProtocolError(
                "production backend health contract is not ready"
            )

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
        if "\x00" in query_text or len(query_text) > 16_384:
            raise RequestError("search query exceeds the pinned text contract")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RequestError("search top_k must be a positive integer")
        if top_k > self.result_limit:
            raise RequestError(
                f"search top_k exceeds configured result limit {self.result_limit}"
            )
        payload = self._request(
            "search",
            "POST",
            "/search",
            body={
                "query": query_text,
                "limit": top_k,
                **_SEARCH_FIXED_REQUEST,
            },
            response_byte_limit=2_000_000,
        )
        results = _validate_search_response(payload, requested_limit=top_k)
        return [
            {
                "url": result["link"],
                "title": result["title"],
                "snippet": result["snippet"],
                "rank": rank,
            }
            for rank, result in enumerate(results, 1)
        ]

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise RequestError("visit URL must be a non-empty string")
        normalized_url = url.strip()
        if "\x00" in normalized_url or len(normalized_url) > 16_384:
            raise RequestError("visit URL exceeds the pinned text contract")
        if not isinstance(goal, str) or "\x00" in goal or len(goal) > 16_384:
            raise RequestError("visit goal must satisfy the pinned text contract")
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise RequestError("visit page must be a positive integer")
        if page > self.page_limit:
            raise RequestError(
                f"visit page exceeds configured page limit {self.page_limit}"
            )
        payload = self._request(
            "visit",
            "POST",
            "/web_parser",
            body={"url": normalized_url},
            response_byte_limit=(self.page_chars * self.page_limit * 6) + 65_536,
        )
        title, text = _validate_visit_response(payload, requested_url=normalized_url)
        if text is None:
            raise RequestError("visit URL is outside the verified browse store")
        if len(text) > self.page_chars * self.page_limit:
            raise BackendProtocolError(
                "production backend visit response exceeds configured page limits"
            )
        pages = tuple(
            text[start : start + self.page_chars]
            for start in range(0, len(text), self.page_chars)
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
        return {
            "url": normalized_url,
            "title": title,
            "content": pages[page - 1],
            "goal": goal,
            "page": page,
            "page_count": len(pages),
            "next_page": page + 1 if page < len(pages) else None,
        }

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        operation: str,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None,
        response_byte_limit: int,
    ) -> Any:
        for attempt in range(self.retry_count + 1):
            try:
                with self._client.stream(method, path, json=body) as response:
                    if not 200 <= response.status_code < 300:
                        if (
                            response.status_code in _RETRIABLE_HTTP_STATUSES
                            and attempt < self.retry_count
                        ):
                            response.close()
                            self._backoff(attempt)
                            continue
                        raise BackendHTTPError(
                            f"production backend {operation} returned HTTP status "
                            f"{response.status_code}"
                        )
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > response_byte_limit:
                            raise BackendProtocolError(
                                f"production backend {operation} response exceeds "
                                "its size limit"
                            )
            except httpx.TimeoutException:
                if attempt < self.retry_count:
                    self._backoff(attempt)
                    continue
                raise BackendTimeoutError(
                    f"production backend {operation} request timed out"
                ) from None
            except httpx.TransportError:
                if attempt < self.retry_count:
                    self._backoff(attempt)
                    continue
                raise BackendConnectionError(
                    f"production backend {operation} connection failed"
                ) from None
            try:
                return json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise BackendProtocolError(
                    f"production backend {operation} response is not valid JSON"
                ) from None
        raise AssertionError("bounded backend request loop exhausted")

    def _backoff(self, attempt: int) -> None:
        if self.retry_backoff_ms:
            time.sleep((self.retry_backoff_ms * (attempt + 1)) / 1000)


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


def _normalize_origin(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("production backend origin must be non-empty canonical text")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("production backend origin is invalid") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "production backend origin must contain only HTTP(S) scheme, host, and port"
        )
    hostname = parsed.hostname.casefold()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = hostname if port is None else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _bounded_integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _load_service_certificate(
    certificate_path: str | Path,
    expected_sha256: str,
    *,
    endpoint_origin_sha256: str,
) -> _ServiceCertificate:
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise ValueError(
            "production backend expected certificate SHA-256 must be a lowercase digest"
        )
    path = Path(certificate_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("production backend certificate must be a real file")
    try:
        raw = path.read_bytes()
    except OSError:
        raise ValueError("production backend certificate could not be read") from None
    if len(raw) > 65_536:
        raise ValueError("production backend certificate exceeds its size limit")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ValueError("production backend certificate SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(
            "production backend certificate must be a UTF-8 JSON object"
        ) from None
    if not isinstance(payload, dict) or set(payload) != _CERTIFICATE_FIELDS:
        raise ValueError("production backend certificate has an invalid schema")
    if payload["schema"] != _CERTIFICATE_SCHEMA:
        raise ValueError("production backend certificate has an unsupported schema")
    if payload["upstream_source_revision"] != LITERESEARCHER_SOURCE_REVISION:
        raise ValueError("production backend certificate source revision mismatch")
    if (
        payload["endpoint_contract_sha256"]
        != LITERESEARCHER_ENDPOINT_CONTRACT_SHA256
    ):
        raise ValueError("production backend certificate endpoint contract mismatch")
    for name in _CERTIFICATE_FIELDS - {"schema", "upstream_source_revision"}:
        value = payload[name]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(
                "production backend certificate digest fields must be lowercase SHA-256"
            )
    if payload["endpoint_origin_sha256"] != endpoint_origin_sha256:
        raise ValueError("production backend certificate origin digest mismatch")
    return _ServiceCertificate(
        service_certificate_sha256=observed,
        endpoint_origin_sha256=payload["endpoint_origin_sha256"],
        search_corpus_certificate_sha256=payload[
            "search_corpus_certificate_sha256"
        ],
        search_index_certificate_sha256=payload[
            "search_index_certificate_sha256"
        ],
        browse_store_certificate_sha256=payload[
            "browse_store_certificate_sha256"
        ],
    )


def _valid_health_response(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != _HEALTH_RESPONSE_FIELDS:
        return False
    stats = payload["stats"]
    return (
        all(
            payload[name] == expected
            for name, expected in _HEALTH_REQUIRED_READY_FIELDS.items()
        )
        and payload["milvus_loaded"] is True
        and payload["redis_connected"] is True
        and _is_nonnegative_integer(payload["collection_entities"])
        and payload["collection_entities"] > 0
        and _is_nonnegative_integer(payload["embedding_queue_size"])
        and isinstance(stats, dict)
        and set(stats) == _HEALTH_STATS_FIELDS
        and all(
            _is_nonnegative_number(stats[name]) for name in _HEALTH_STATS_FIELDS
        )
    )


def _validate_search_response(
    payload: Any,
    *,
    requested_limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != _SEARCH_RESPONSE_FIELDS:
        raise BackendProtocolError(
            "production backend search response schema mismatch"
        )
    results = payload["results"]
    if (
        not isinstance(results, list)
        or not _is_nonnegative_integer(payload["total"])
        or payload["total"] != len(results)
        or len(results) > requested_limit
        or not _finite_number(payload["sparse_weight"])
        or not _finite_number(payload["dense_weight"])
        or any(
            payload[name] != expected
            for name, expected in _SEARCH_FIXED_RESPONSE.items()
        )
        or any(
            not _is_nonnegative_number(payload[name])
            for name in ("search_time", "embedding_time", "milvus_time")
        )
    ):
        raise BackendProtocolError(
            "production backend search response schema mismatch"
        )
    for result in results:
        if (
            not isinstance(result, dict)
            or set(result) != _SEARCH_RESULT_FIELDS
            or not _nonempty_text(result["link"])
            or not _plain_text(result["title"])
            or not _plain_text(result["snippet"])
            or not _finite_number(result["score"])
        ):
            raise BackendProtocolError(
                "production backend search response schema mismatch"
            )
    return results


def _validate_visit_response(
    payload: Any,
    *,
    requested_url: str,
) -> tuple[str, str | None]:
    if (
        not isinstance(payload, dict)
        or set(payload) != _VISIT_RESPONSE_FIELDS
        or type(payload["found"]) is not bool
        or not _nonempty_text(payload["url"])
        or payload["url"] != requested_url
        or not _plain_text(payload["title"])
        or not _plain_text(payload["text"])
    ):
        raise BackendProtocolError(
            "production backend visit response schema mismatch"
        )
    if not payload["found"]:
        if payload["title"] or payload["text"]:
            raise BackendProtocolError(
                "production backend visit response schema mismatch"
            )
        return payload["title"], None
    if not payload["text"]:
        raise BackendProtocolError(
            "production backend visit response schema mismatch"
        )
    return payload["title"], payload["text"]


def _plain_text(value: Any) -> bool:
    return isinstance(value, str) and "\x00" not in value


def _nonempty_text(value: Any) -> bool:
    return _plain_text(value) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_nonnegative_number(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def _is_nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _tokens(value: str) -> Iterable[str]:
    return re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text without NUL bytes")
    return value
