from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentenv_agentmemory.literesearcher import (
    LiteResearchBackendError,
    LiteResearchRequestError,
    UpstreamHybridLiteResearchBackend,
)


class _Tasks:
    pass


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, dict | None]] = []
        self.mode = "ok"
        self.active = 0
        self.max_active = 0
        self.documents = {
            "https://docs.test/known": {
                "title": "Known page",
                "text": "alpha evidence " * 900,
            }
        }

    def record(self, path: str, payload: dict | None) -> None:
        with self.lock:
            self.requests.append((path, payload))

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _QuietServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        del request, client_address


def _handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            del format, args

        def _respond(self, status: int, payload, *, raw: bytes | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = raw if raw is not None else json.dumps(payload).encode("utf-8")
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:
            state.record(self.path, None)
            if self.path != "/health":
                self._respond(404, {"detail": "not found"})
                return
            payload = {
                "status": "healthy",
                "server_id": "local_rag_diskann_server",
                "service_id": "literesearcher-i-bgem3-diskann-v1",
                "milvus_loaded": True,
                "redis_connected": True,
                "document_count": 32_127_370,
                "loaded_count": 32_127_370,
                "index_type": "DISKANN",
                "sparse_index_type": "SPARSE_INVERTED_INDEX",
                "vector_precision": "FP32",
                "default_search_type": "hybrid",
                "hybrid_weights": {"dense": 1.0, "sparse": 0.7},
            }
            if state.mode == "bad_health_index":
                payload["index_type"] = "HNSW"
            if state.mode == "bad_health_count":
                payload["loaded_count"] -= 1
            self._respond(200, payload)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            state.record(self.path, payload)
            if state.mode == "http_error":
                self._respond(503, {"detail": "unavailable"})
                return
            if state.mode == "malformed":
                self._respond(200, None, raw=b"not-json")
                return
            if self.path == "/search":
                state.enter()
                try:
                    query = payload["query"]
                    if query == "slow":
                        time.sleep(0.15)
                    elif query.startswith("parallel-"):
                        time.sleep(0.02)
                    results = [
                        {
                            "link": f"https://docs.test/{query}",
                            "title": f"Result {query}",
                            "snippet": "matching evidence",
                            "score": 0.75,
                        },
                        {
                            "link": "https://docs.test/masked",
                            "title": "Masked result",
                            "snippet": "must not escape",
                            "score": 0.5,
                        },
                    ][: payload["limit"]]
                    response = {
                        "results": results,
                        "total": len(results),
                        "search_time": 0.02,
                        "embedding_time": 0.01,
                        "milvus_time": 0.01,
                        "search_type": "hybrid",
                        "sparse_weight": 0.7,
                        "dense_weight": 1.0,
                        "server_id": "local_rag_diskann_server",
                        "service_id": "literesearcher-i-bgem3-diskann-v1",
                        "document_count": 32_127_370,
                        "loaded_count": 32_127_370,
                        "vector_dtype": "FP32",
                        "index_type": "DISKANN",
                        "sparse_index_type": "SPARSE_INVERTED_INDEX",
                    }
                    if state.mode == "bad_schema":
                        response["index_type"] = "HNSW"
                    self._respond(200, response)
                finally:
                    state.leave()
                return
            if self.path == "/web_parser":
                url = payload["url"]
                document = state.documents.get(url)
                if document is None:
                    self._respond(
                        200,
                        {
                            "found": False,
                            "url": url,
                            "title": "",
                            "text": "",
                            "visit_time": 0.01,
                            "service_id": "literesearcher-i-bgem3-diskann-v1",
                            "backend": "postgresql_exact_url",
                        },
                    )
                else:
                    self._respond(
                        200,
                        {
                            "found": True,
                            "url": url,
                            **document,
                            "visit_time": 0.01,
                            "service_id": "literesearcher-i-bgem3-diskann-v1",
                            "backend": "postgresql_exact_url",
                        },
                    )
                return
            self._respond(404, {"detail": "not found"})

    return Handler


class UpstreamHybridLiteResearchBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _State()
        self.server = _QuietServer(("127.0.0.1", 0), _handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def backend(self, **kwargs) -> UpstreamHybridLiteResearchBackend:
        return UpstreamHybridLiteResearchBackend(_Tasks(), self.url, **kwargs)

    def test_startup_attests_health_and_search_identity(self) -> None:
        backend = self.backend(top_k=3, timeout_seconds=5)
        self.assertEqual([path for path, _ in self.state.requests[:2]], ["/health", "/search"])
        metadata = backend.metadata()
        self.assertTrue(metadata["upstream_hybrid_backend"])
        self.assertTrue(metadata["service_identity_verified"])
        self.assertEqual(metadata["document_count"], 32_127_370)
        self.assertEqual(metadata["index_type"], "DISKANN")
        self.assertEqual(metadata["sparse_index_type"], "SPARSE_INVERTED_INDEX")
        self.assertEqual(metadata["vector_dtype"], "FP32")
        self.assertFalse(metadata["lexical_fallback"])

    def test_search_uses_frozen_hybrid_weights_and_masks_target_url(self) -> None:
        backend = self.backend(top_k=2)
        hits = backend.search(
            ["alpha", "beta"],
            top_k=2,
            mask_url="https://docs.test/masked",
        )
        self.assertEqual(
            [hit["url"] for hit in hits],
            ["https://docs.test/alpha", "https://docs.test/beta"],
        )
        self.assertEqual([hit["query"] for hit in hits], ["alpha", "beta"])
        self.assertTrue(all("score" not in hit for hit in hits))
        payloads = {
            payload["query"]: payload
            for path, payload in self.state.requests
            if path == "/search"
            and payload is not None
            and payload["query"] in {"alpha", "beta"}
        }
        self.assertEqual(set(payloads), {"alpha", "beta"})
        for query, payload in payloads.items():
            self.assertEqual(
                payload,
                {
                    "query": query,
                    "limit": 2,
                    "search_type": "hybrid",
                    "sparse_weight": 0.7,
                    "dense_weight": 1.0,
                },
            )

    def test_multiple_queries_run_independently_and_deduplicate_urls(self) -> None:
        backend = self.backend(top_k=2)
        hits = backend.search(["parallel-first", "parallel-second"])
        self.assertEqual(
            [hit["url"] for hit in hits],
            [
                "https://docs.test/parallel-first",
                "https://docs.test/masked",
                "https://docs.test/parallel-second",
            ],
        )
        self.assertEqual(
            [hit["query"] for hit in hits],
            ["parallel-first", "parallel-first", "parallel-second"],
        )
        self.assertEqual([hit["rank"] for hit in hits], [1, 2, 1])
        self.assertGreater(self.state.max_active, 1)

    def test_mask_excludes_only_the_exact_released_url(self) -> None:
        backend = self.backend(top_k=2)
        hits = backend.search("alpha", mask_url="https://docs.test/mask")
        self.assertEqual(
            [hit["url"] for hit in hits],
            ["https://docs.test/alpha", "https://docs.test/masked"],
        )

    def test_visit_reads_only_released_web_parser_and_paginates(self) -> None:
        backend = self.backend()
        page = backend.visit("https://docs.test/known", goal="alpha", page=1)
        self.assertEqual(page["url"], "https://docs.test/known")
        self.assertEqual(page["title"], "Known page")
        self.assertIn("alpha evidence", page["content"])
        self.assertGreater(page["page_count"], 1)
        self.assertEqual(self.state.requests[-1][0], "/web_parser")

    def test_unknown_visit_fails_closed(self) -> None:
        backend = self.backend()
        with self.assertRaises(LiteResearchRequestError):
            backend.visit("https://outside.test/unknown")
        self.assertEqual(self.state.requests[-1][0], "/web_parser")

    def test_transport_and_schema_failures_raise_backend_error(self) -> None:
        backend = self.backend(timeout_seconds=0.05)
        for mode, query in (
            ("http_error", "http"),
            ("malformed", "json"),
            ("bad_schema", "identity"),
            ("ok", "slow"),
        ):
            with self.subTest(mode=mode):
                self.state.mode = mode
                with self.assertRaises(LiteResearchBackendError):
                    backend.search(query, top_k=1)
        self.state.mode = "ok"

    def test_constructor_rejects_non_diskann_health(self) -> None:
        self.state.mode = "bad_health_index"
        with self.assertRaises(LiteResearchBackendError):
            self.backend()

    def test_constructor_rejects_incomplete_loaded_count(self) -> None:
        self.state.mode = "bad_health_count"
        with self.assertRaises(LiteResearchBackendError):
            self.backend()

    def test_sixty_four_concurrent_searches_are_independent(self) -> None:
        backend = self.backend(top_k=1, timeout_seconds=5)
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [
                executor.submit(backend.search, f"parallel-{index}", top_k=1)
                for index in range(64)
            ]
        hits = [future.result() for future in futures]
        self.assertEqual(len(hits), 64)
        self.assertEqual(
            {result[0]["url"] for result in hits},
            {f"https://docs.test/parallel-{index}" for index in range(64)},
        )
        self.assertGreater(self.state.max_active, 1)


if __name__ == "__main__":
    unittest.main()
