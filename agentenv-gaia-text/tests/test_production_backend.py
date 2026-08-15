from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from agentenv_gaia_text.backend import (
    LITERESEARCHER_ENDPOINT_CONTRACT_SHA256,
    LITERESEARCHER_SOURCE_REVISION,
    BackendConnectionError,
    BackendHTTPError,
    BackendProtocolError,
    BackendTimeoutError,
    LiteResearcherBackend,
    RequestError,
)
from agentenv_gaia_text.contracts import EvaluationArm, ProtocolContract
from agentenv_gaia_text.dataset import GaiaTextDataset
from agentenv_gaia_text.launch import build_manager_from_environment
from agentenv_gaia_text.submission import SubmissionStore
from agentenv_gaia_text.wrapper import GaiaTextEpisodeManager
from support import FileWorkspace, protocol_kwargs, write_runtime_fixture


class _FakeState:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.health_status = 200
        self.health_payload: dict[str, Any] = {
            "status": "healthy",
            "milvus_loaded": True,
            "collection_entities": 32_000_000,
            "redis_connected": True,
            "embedding_queue_size": 0,
            "vector_precision": "FP32",
            "index_type": "DISKANN",
            "stats": {
                "active_requests": 0,
                "total_requests": 1,
                "success_count": 1,
                "error_count": 0,
                "requests_per_second": 1.0,
                "avg_response_ms": 2.0,
            },
        }
        self.search_status = 200
        self.search_payload: dict[str, Any] | None = None
        self.search_delay = 0.0
        self.visit_status = 200
        self.visit_payload: dict[str, Any] = {
            "found": True,
            "url": "https://documents.invalid/alpha",
            "title": "Alpha",
            "text": "alpha oneXdelta twoXomega",
        }

    def record(self, value: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(value)


class FakeLiteResearcher(AbstractContextManager["FakeLiteResearcher"]):
    def __init__(self) -> None:
        self.state = _FakeState()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.state = self.state  # type: ignore[attr-defined]
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class _Handler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        state: _FakeState = self.server.state  # type: ignore[attr-defined]
        state.record({"method": "GET", "path": self.path, "body": None})
        if self.path != "/health":
            self._reply(404, {"detail": "unknown"})
            return
        self._reply(state.health_status, state.health_payload)

    def do_POST(self) -> None:
        state: _FakeState = self.server.state  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        state.record({"method": "POST", "path": self.path, "body": body})
        if self.path == "/search":
            if state.search_delay:
                time.sleep(state.search_delay)
            payload = state.search_payload or _search_response(body)
            self._reply(state.search_status, payload)
            return
        if self.path == "/web_parser":
            self._reply(state.visit_status, state.visit_payload)
            return
        self._reply(404, {"detail": "unknown"})

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _search_response(request: dict[str, Any]) -> dict[str, Any]:
    results = [
        {
            "link": f"https://documents.invalid/{index}",
            "title": f"Result {index}",
            "snippet": f"Evidence for {request['query']}",
            "score": 1.0 / index,
        }
        for index in range(1, request["limit"] + 1)
    ]
    return {
        "results": results,
        "total": len(results),
        "search_time": 0.01,
        "embedding_time": 0.004,
        "milvus_time": 0.006,
        "search_type": "hybrid",
        "sparse_weight": 0.7,
        "dense_weight": 1.0,
        "server_id": "local_rag_diskann_server",
        "vector_dtype": "FP32",
        "index_type": "DISKANN",
    }


def _certificate(tmp_path: Path, base_url: str) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "gaia_text_literesearcher_service_certificate_v1",
        "upstream_source_revision": LITERESEARCHER_SOURCE_REVISION,
        "endpoint_contract_sha256": LITERESEARCHER_ENDPOINT_CONTRACT_SHA256,
        "endpoint_origin_sha256": hashlib.sha256(base_url.encode()).hexdigest(),
        "search_corpus_certificate_sha256": "1" * 64,
        "search_index_certificate_sha256": "2" * 64,
        "browse_store_certificate_sha256": "3" * 64,
    }
    raw = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    path = tmp_path / "private-service-certificate.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _backend(
    tmp_path: Path,
    service: FakeLiteResearcher,
    **overrides: Any,
) -> LiteResearcherBackend:
    certificate, digest = _certificate(tmp_path, service.url)
    values = {
        "base_url": service.url,
        "connect_timeout_ms": 200,
        "read_timeout_ms": 200,
        "retry_count": 0,
        "retry_backoff_ms": 0,
        "result_limit": 3,
        "page_chars": 10,
        "page_limit": 4,
    }
    values.update(overrides)
    return LiteResearcherBackend.load(certificate, digest, **values)


def test_exact_upstream_schema_and_local_visit_pagination(tmp_path: Path) -> None:
    with FakeLiteResearcher() as service:
        backend = _backend(tmp_path, service)
        results = backend.search(["alpha", "beta"], top_k=2)
        page = backend.visit(
            "https://documents.invalid/alpha", goal="delta", page=2
        )

        assert service.state.requests == [
            {"method": "GET", "path": "/health", "body": None},
            {
                "method": "POST",
                "path": "/search",
                "body": {
                    "query": "alpha beta",
                    "limit": 2,
                    "search_type": "hybrid",
                    "sparse_weight": 0.7,
                    "dense_weight": 1.0,
                },
            },
            {
                "method": "POST",
                "path": "/web_parser",
                "body": {"url": "https://documents.invalid/alpha"},
            },
        ]
        assert results == [
            {
                "url": "https://documents.invalid/1",
                "title": "Result 1",
                "snippet": "Evidence for alpha beta",
                "rank": 1,
            },
            {
                "url": "https://documents.invalid/2",
                "title": "Result 2",
                "snippet": "Evidence for alpha beta",
                "rank": 2,
            },
        ]
        assert page == {
            "url": "https://documents.invalid/alpha",
            "title": "Alpha",
            "content": "alpha oneX",
            "goal": "delta",
            "page": 2,
            "page_count": 3,
            "next_page": 3,
        }
        with pytest.raises(RequestError, match="page 4 exceeds page_count 3"):
            backend.visit("https://documents.invalid/alpha", page=4)
        with pytest.raises(RequestError, match="configured page limit 4"):
            backend.visit("https://documents.invalid/alpha", page=5)
        with pytest.raises(RequestError, match="configured result limit"):
            backend.search("alpha", top_k=4)


def test_metadata_binds_runtime_identity_without_endpoint_or_paths(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secret-token-directory"
    secret_dir.mkdir()
    with FakeLiteResearcher() as service:
        backend = _backend(secret_dir, service, retry_count=2, retry_backoff_ms=7)
        metadata = backend.metadata()

    serialized = json.dumps(metadata, sort_keys=True)
    assert str(secret_dir) not in serialized
    assert service.url not in serialized
    assert "secret-token" not in serialized
    expected_identity = {
        "backend_contract": "gaia_text_literesearcher_search_visit_v1",
        "backend_kind": "production",
        "browse_store_certificate_sha256": "3" * 64,
        "connect_timeout_ms": 200,
        "endpoint_contract_sha256": LITERESEARCHER_ENDPOINT_CONTRACT_SHA256,
        "endpoint_origin_sha256": hashlib.sha256(service.url.encode()).hexdigest(),
        "failure_mode": "fail_closed",
        "live_web_fallback": False,
        "network_access": "configured_literesearcher_origin_only",
        "page_chars": 10,
        "page_limit": 4,
        "read_timeout_ms": 200,
        "result_limit": 3,
        "retry_backoff_ms": 7,
        "retry_count": 2,
        "search_corpus_certificate_sha256": "1" * 64,
        "search_index_certificate_sha256": "2" * 64,
        "service_certificate_sha256": backend.service_certificate_sha256,
        "upstream_source_revision": LITERESEARCHER_SOURCE_REVISION,
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            expected_identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert metadata == {
        **expected_identity,
        "asset_sha256": expected_digest,
        "runtime_identity_sha256": expected_digest,
    }


def test_certificate_and_origin_digest_mismatches_fail_before_network(
    tmp_path: Path,
) -> None:
    with FakeLiteResearcher() as service:
        certificate, digest = _certificate(tmp_path, service.url)
        with pytest.raises(ValueError, match="certificate SHA-256 mismatch") as error:
            LiteResearcherBackend.load(
                certificate,
                "0" * 64,
                base_url=service.url,
                connect_timeout_ms=20,
                read_timeout_ms=20,
                retry_count=0,
                retry_backoff_ms=0,
                result_limit=3,
                page_chars=10,
                page_limit=4,
            )
        assert str(certificate) not in str(error.value)

        with pytest.raises(ValueError, match="origin digest mismatch"):
            LiteResearcherBackend.load(
                certificate,
                digest,
                base_url="http://127.0.0.1:1",
                connect_timeout_ms=20,
                read_timeout_ms=20,
                retry_count=0,
                retry_backoff_ms=0,
                result_limit=3,
                page_chars=10,
                page_limit=4,
            )
        assert service.state.requests == []


def test_unhealthy_probe_and_strict_response_schema_fail_closed(tmp_path: Path) -> None:
    with FakeLiteResearcher() as service:
        service.state.health_payload["status"] = "unhealthy"
        with pytest.raises(BackendProtocolError, match="health contract"):
            _backend(tmp_path / "health", service)

    with FakeLiteResearcher() as service:
        backend = _backend(tmp_path / "schema", service)
        malformed = _search_response({"query": "alpha", "limit": 1})
        del malformed["results"][0]["score"]
        service.state.search_payload = malformed
        with pytest.raises(BackendProtocolError, match="search response schema"):
            backend.search("alpha", top_k=1)

        boolean_weight = _search_response({"query": "alpha", "limit": 1})
        boolean_weight["dense_weight"] = True
        service.state.search_payload = boolean_weight
        with pytest.raises(BackendProtocolError, match="search response schema"):
            backend.search("alpha", top_k=1)

        service.state.visit_payload["text"] = "x" * 41
        with pytest.raises(BackendProtocolError, match="configured page limits"):
            backend.visit("https://documents.invalid/alpha")


def test_http_errors_retry_without_leaking_body_endpoint_or_certificate_path(
    tmp_path: Path,
) -> None:
    secret = "TOP-SECRET-UPSTREAM-BODY"
    with FakeLiteResearcher() as service:
        backend = _backend(
            tmp_path / "private-certificate-path",
            service,
            retry_count=1,
        )
        service.state.search_status = 503
        service.state.search_payload = {"detail": secret}
        with pytest.raises(BackendHTTPError, match="HTTP status 503") as error:
            backend.search("alpha", top_k=1)

        message = str(error.value)
        assert secret not in message
        assert service.url not in message
        assert str(tmp_path) not in message
        search_requests = [
            request
            for request in service.state.requests
            if request["path"] == "/search"
        ]
        assert len(search_requests) == 2


def test_timeout_and_connection_failures_are_classified_and_bounded(
    tmp_path: Path,
) -> None:
    with FakeLiteResearcher() as service:
        backend = _backend(
            tmp_path / "timeout",
            service,
            read_timeout_ms=20,
            retry_count=1,
        )
        service.state.search_delay = 0.2
        with pytest.raises(BackendTimeoutError, match="timed out"):
            backend.search("alpha", top_k=1)
        search_requests = [
            request
            for request in service.state.requests
            if request["path"] == "/search"
        ]
        assert len(search_requests) == 2

    with FakeLiteResearcher() as service:
        backend = _backend(tmp_path / "connection", service, retry_count=1)
    with pytest.raises(BackendConnectionError, match="connection failed"):
        backend.search("alpha", top_k=1)


def test_redirect_is_not_followed_as_a_network_fallback(tmp_path: Path) -> None:
    with FakeLiteResearcher() as service:
        backend = _backend(tmp_path, service)
        service.state.search_status = 307
        service.state.search_payload = {"location": "https://example.com/forbidden"}
        with pytest.raises(BackendHTTPError, match="HTTP status 307"):
            backend.search("alpha", top_k=1)
        assert len(service.state.requests) == 2


def test_not_found_visit_is_a_policy_request_error_without_fallback(
    tmp_path: Path,
) -> None:
    with FakeLiteResearcher() as service:
        backend = _backend(tmp_path, service)
        service.state.visit_payload = {
            "found": False,
            "url": "https://documents.invalid/missing",
            "title": "",
            "text": "",
        }
        with pytest.raises(RequestError, match="verified browse store"):
            backend.visit("https://documents.invalid/missing")
        assert [request["path"] for request in service.state.requests] == [
            "/health",
            "/web_parser",
        ]


def test_production_backend_identity_and_dispatch_are_equal_across_triad(
    tmp_path: Path,
) -> None:
    runtime = write_runtime_fixture(tmp_path / "runtime")
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=contract,
    )
    with FakeLiteResearcher() as service:
        backends = {
            arm: _backend(tmp_path / f"{arm.value}-cert", service)
            for arm in EvaluationArm
        }

        def workspace_factory(
            env_id: int, task_id: str, episode_index: int
        ) -> FileWorkspace:
            return FileWorkspace(
                tmp_path / "workspaces",
                f"{env_id}-{task_id}-{episode_index}",
            )

        managers = {
            arm: GaiaTextEpisodeManager(
                dataset,
                backends[arm],
                SubmissionStore(dataset.task_ids, tmp_path / f"{arm.value}.jsonl"),
                arm=arm,
                workspace_factory=(
                    workspace_factory if arm is EvaluationArm.AMG_MEMORY else None
                ),
            )
            for arm in EvaluationArm
        }
        metadata = [manager.metadata() for manager in managers.values()]
        assert len(
            {
                json.dumps(item["backend"], sort_keys=True)
                for item in metadata
            }
        ) == 1
        assert len(
            {
                json.dumps(item["paired_runtime_contract"], sort_keys=True)
                for item in metadata
            }
        ) == 1
        assert len(
            {item["paired_runtime_contract_sha256"] for item in metadata}
        ) == 1

        env_ids = {arm: manager.create()["id"] for arm, manager in managers.items()}
        reset_observations = {
            manager.reset(env_ids[arm], 0)["observation"]
            for arm, manager in managers.items()
        }
        assert len(reset_observations) == 1
        action = (
            '<tool_call>{"name":"search","arguments":{"query":"alpha",'
            '"top_k":1}}</tool_call>'
        )
        observations = {
            manager.step(env_ids[arm], action)["observation"]
            for arm, manager in managers.items()
        }
        assert len(observations) == 1

        search_requests = [
            request
            for request in service.state.requests
            if request["path"] == "/search"
        ]
        assert search_requests[-3:] == [search_requests[-1]] * 3


def test_launcher_requires_explicit_backend_and_selects_production(
    tmp_path: Path,
) -> None:
    runtime = write_runtime_fixture(tmp_path / "runtime")
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    base = {
        "GAIA_TEXT_ARM": "native",
        "GAIA_TEXT_MANIFEST": str(runtime.manifest),
        "GAIA_TEXT_QUESTIONS": str(runtime.questions),
        "GAIA_TEXT_QUESTIONS_SHA256": runtime.questions_sha256,
        "GAIA_TEXT_PREDICTIONS": str(runtime.predictions),
    }
    with pytest.raises(RuntimeError, match="GAIA_TEXT_BACKEND"):
        build_manager_from_environment(contract=contract, environment=base)

    with FakeLiteResearcher() as service:
        certificate, digest = _certificate(tmp_path, service.url)
        production = {
            **base,
            "GAIA_TEXT_BACKEND": "production",
            "GAIA_TEXT_LITERESEARCHER_BASE_URL": service.url,
            "GAIA_TEXT_LITERESEARCHER_CERTIFICATE": str(certificate),
            "GAIA_TEXT_LITERESEARCHER_CERTIFICATE_SHA256": digest,
            "GAIA_TEXT_LITERESEARCHER_CONNECT_TIMEOUT_MS": "200",
            "GAIA_TEXT_LITERESEARCHER_READ_TIMEOUT_MS": "200",
            "GAIA_TEXT_LITERESEARCHER_RETRY_COUNT": "0",
            "GAIA_TEXT_LITERESEARCHER_RETRY_BACKOFF_MS": "0",
            "GAIA_TEXT_SEARCH_RESULT_LIMIT": "3",
            "GAIA_TEXT_VISIT_PAGE_CHARS": "10",
            "GAIA_TEXT_VISIT_PAGE_LIMIT": "4",
        }
        manager = build_manager_from_environment(
            contract=contract, environment=production
        )
        assert isinstance(manager.backend, LiteResearcherBackend)
        assert manager.metadata()["backend"]["backend_kind"] == "production"

        with pytest.raises(RuntimeError, match="must not mix"):
            build_manager_from_environment(
                contract=contract,
                environment={
                    **production,
                    "GAIA_TEXT_BACKEND_ASSET": str(runtime.backend),
                    "GAIA_TEXT_BACKEND_SHA256": runtime.backend_sha256,
                },
            )


@pytest.mark.parametrize(
    "production_only_name",
    ["GAIA_TEXT_SEARCH_RESULT_LIMIT", "GAIA_TEXT_VISIT_PAGE_LIMIT"],
)
def test_fixture_selection_rejects_production_only_limits(
    production_only_name: str,
    tmp_path: Path,
) -> None:
    runtime = write_runtime_fixture(tmp_path)
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    environment = {
        "GAIA_TEXT_ARM": "native",
        "GAIA_TEXT_BACKEND": "fixture",
        "GAIA_TEXT_MANIFEST": str(runtime.manifest),
        "GAIA_TEXT_QUESTIONS": str(runtime.questions),
        "GAIA_TEXT_QUESTIONS_SHA256": runtime.questions_sha256,
        "GAIA_TEXT_BACKEND_ASSET": str(runtime.backend),
        "GAIA_TEXT_BACKEND_SHA256": runtime.backend_sha256,
        "GAIA_TEXT_PREDICTIONS": str(runtime.predictions),
        production_only_name: "3",
    }
    with pytest.raises(RuntimeError, match="must not mix"):
        build_manager_from_environment(contract=contract, environment=environment)
