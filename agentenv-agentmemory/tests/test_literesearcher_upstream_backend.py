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
from agentenv_agentmemory.literesearcher.backend import _rank_windows_by_goal


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
                "document_count": 32_127_370,
                "loaded_count": 32_127_370,
                "collection_entities": 32_127_370,
                "redis_connected": True,
                "embedding_queue_size": 0,
                "index_type": "DISKANN",
                "sparse_index_type": "SPARSE_INVERTED_INDEX",
                "default_search_type": "hybrid",
                "hybrid_weights": {"sparse": 0.7, "dense": 1.0},
                "vector_dtype": "FP32",
                "vector_precision": "FP32",
                "visit_pagination_contract": "goal_bm25_overlapping_chars_v1",
                "visit_page_chars": 8192,
                "visit_page_overlap_chars": 1024,
                "visit_page_endpoint": "/visit_page",
                "visit_page_backend": "postgresql_exact_url_goal_bm25_page_v1",
            }
            if state.mode == "bad_health_index":
                payload["index_type"] = "HNSW"
            elif state.mode == "partial_collection":
                payload["collection_entities"] -= 1
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
                        {
                            "link": "https://docs.test/alternate",
                            "title": "Alternate result",
                            "snippet": "independent evidence",
                            "score": 0.25,
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
                        "vector_dtype": "FP32",
                        "index_type": "DISKANN",
                        "sparse_index_type": "SPARSE_INVERTED_INDEX",
                        "document_count": 32_127_370,
                        "loaded_count": 32_127_370,
                    }
                    if state.mode == "bad_schema":
                        response["index_type"] = "HNSW"
                    self._respond(200, response)
                finally:
                    state.leave()
                return
            if self.path == "/visit_page":
                url = payload["url"]
                goal = payload["goal"]
                page = payload["page"]
                document = state.documents.get(url)
                response = {
                    "found": document is not None,
                    "url": url,
                    "title": "",
                    "content": "",
                    "goal": goal,
                    "page": page,
                    "page_count": 0,
                    "next_page": None,
                    "visit_time": 0.01,
                    "service_id": "literesearcher-i-bgem3-diskann-v1",
                    "backend": "postgresql_exact_url_goal_bm25_page_v1",
                    "pagination_contract": "goal_bm25_overlapping_chars_v1",
                    "page_chars": 8192,
                    "page_overlap_chars": 1024,
                }
                if document is not None:
                    pages = _rank_windows_by_goal(document["text"], goal)
                    response.update(
                        title=document["title"],
                        content=pages[page - 1] if page <= len(pages) else "",
                        page_count=len(pages),
                        next_page=page + 1 if page < len(pages) else None,
                    )
                if state.mode == "bad_visit_schema":
                    response["backend"] = "postgresql_exact_url"
                self._respond(200, response)
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
        self.assertEqual(metadata["index_type"], "DISKANN")
        self.assertEqual(metadata["vector_dtype"], "FP32")
        self.assertEqual(metadata["document_count"], 32_127_370)
        self.assertFalse(metadata["lexical_fallback"])

    def test_search_uses_frozen_hybrid_weights_and_masks_target_url(self) -> None:
        backend = self.backend(top_k=2)
        hits = backend.search(
            ["alpha", "beta"],
            top_k=2,
            mask_url="https://docs.test/masked",
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["url"], "https://docs.test/alpha beta")
        self.assertEqual(hits[1]["url"], "https://docs.test/alternate")
        self.assertEqual(hits[0]["score"], 0.75)
        _, payload = self.state.requests[-1]
        self.assertEqual(
            payload,
            {
                "query": "alpha beta",
                "limit": 3,
                "search_type": "hybrid",
                "sparse_weight": 0.7,
                "dense_weight": 1.0,
            },
        )

    def test_visit_uses_bounded_server_page_contract(self) -> None:
        backend = self.backend()
        page = backend.visit("https://docs.test/known", goal="alpha", page=1)
        self.assertEqual(page["url"], "https://docs.test/known")
        self.assertEqual(page["title"], "Known page")
        self.assertIn("alpha evidence", page["content"])
        self.assertLessEqual(len(page["content"]), 8192)
        self.assertGreater(page["page_count"], 1)
        self.assertEqual(
            self.state.requests[-1],
            (
                "/visit_page",
                {
                    "url": "https://docs.test/known",
                    "goal": "alpha",
                    "page": 1,
                },
            ),
        )

    def test_unknown_and_out_of_range_visit_fail_as_requests(self) -> None:
        backend = self.backend()
        with self.assertRaises(LiteResearchRequestError):
            backend.visit("https://outside.test/unknown")
        self.assertEqual(self.state.requests[-1][0], "/visit_page")
        with self.assertRaises(LiteResearchRequestError):
            backend.visit("https://docs.test/known", goal="alpha", page=99)
        self.assertEqual(self.state.requests[-1][0], "/visit_page")

    def test_bounded_visit_schema_failure_is_backend_error(self) -> None:
        backend = self.backend()
        self.state.mode = "bad_visit_schema"
        with self.assertRaises(LiteResearchBackendError):
            backend.visit("https://docs.test/known", goal="alpha", page=1)

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

    def test_constructor_rejects_partial_collection(self) -> None:
        self.state.mode = "partial_collection"
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
