#!/usr/bin/env python3
"""Long-lived, label-free MemoryArena/WebShop Lucene catalog service."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


POOL_LIMIT = 50


class CatalogSingleton:
    def __init__(
        self,
        index_path: Path,
        items_path: Path,
        *,
        price_seed: int,
        expected_index_manifest_sha256: str,
        expected_items_sha256: str,
    ) -> None:
        from pyserini.pyclass import autoclass

        index_manifest_sha256 = index_tree_manifest_sha256(index_path)
        items_sha256 = file_sha256(items_path)
        require_digest(
            "index manifest",
            index_manifest_sha256,
            expected_index_manifest_sha256,
        )
        require_digest("items", items_sha256, expected_items_sha256)
        searcher_class = autoclass("io.anserini.search.SimpleSearcher")
        self.searcher = searcher_class(str(index_path))
        self.products = load_products(items_path, price_seed=price_seed)
        self.asset_info = {
            "price_seed": price_seed,
            "product_count": len(self.products),
            "items_sha256": items_sha256,
            "index_manifest_sha256": index_manifest_sha256,
            "asin_price_digest_sha256": asin_price_digest(self.products),
            "asin_price_digest_format": "sorted asin + TAB + price_usd:.6f + LF",
        }
        self.lock = threading.Lock()

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            hits = self.searcher.search(query, limit)
            rows = []
            for hit in hits:
                asin = str(hit.docid).strip().upper()
                product = self.products.get(asin)
                if product is None:
                    continue
                rows.append(
                    {
                        "asin": asin,
                        "title": product["title"],
                        "price_usd": product["price_usd"],
                        "average_rating": product["average_rating"],
                        "total_reviews": product["total_reviews"],
                        "match_score": int(round(float(hit.score) * 1000)),
                        "backend_rank": len(rows) + 1,
                    }
                )
                if len(rows) >= limit:
                    break
        return rows


def load_products(items_path: Path, *, price_seed: int) -> dict[str, dict[str, Any]]:
    import ijson

    products: dict[str, dict[str, Any]] = {}
    with items_path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            if not isinstance(row, dict):
                continue
            asin = str(row.get("asin", "")).strip().upper()
            title = str(row.get("name", row.get("Title", ""))).strip()
            if len(asin) != 10 or not asin.isalnum() or not title:
                continue
            products.setdefault(
                asin,
                {
                    "title": title,
                    "price_usd": deterministic_price(row.get("pricing"), asin, price_seed),
                    "average_rating": optional_float(row.get("average_rating")),
                    "total_reviews": optional_int(row.get("total_reviews")),
                },
            )
    return products


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_tree_manifest_sha256(index_path: Path) -> str:
    """Hash `sha256<two spaces>relative_path<LF>` lines in C path order."""
    files = [path for path in index_path.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.relative_to(index_path).as_posix().encode("utf-8"))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(index_path).as_posix()
        digest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def asin_price_digest(products: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for asin in sorted(products):
        digest.update(
            f"{asin}\t{float(products[asin]['price_usd']):.6f}\n".encode("ascii")
        )
    return digest.hexdigest()


def require_digest(name: str, actual: str, expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"expected {name} SHA256 must be 64 lowercase hex characters")
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {actual}")


def deterministic_price(raw: Any, asin: str, seed: int) -> float:
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(raw or ""))]
    if not values:
        return 100.0
    if len(values) == 1:
        return values[0]
    digest = hashlib.sha256(f"{seed}:{asin}".encode("ascii")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return round(rng.uniform(min(values[:2]), max(values[:2])), 2)


def optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 5 else None


def optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


class CatalogHandler(BaseHTTPRequestHandler):
    server: "CatalogServer"

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"status": "backend_error", "error": "not found"})
            return
        self.send_json(
            200,
            {
                "status": "ok",
                "backend_name": "memoryarena_lucene",
                **self.server.catalog.asset_info,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/search":
            self.send_json(404, {"status": "backend_error", "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = payload.get("query") if isinstance(payload, dict) else None
            limit = payload.get("limit", POOL_LIMIT) if isinstance(payload, dict) else None
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= POOL_LIMIT:
                raise ValueError(f"limit must be an integer within 1..{POOL_LIMIT}")
            rows = self.server.catalog.search(query.strip(), limit=limit)
            self.send_json(
                200,
                {
                    "status": "ok" if rows else "empty",
                    "backend_name": "memoryarena_lucene",
                    "results": rows,
                },
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.send_json(400, {"status": "backend_error", "error": str(exc), "results": []})
        except Exception as exc:
            self.send_json(
                500,
                {"status": "backend_error", "error": f"{type(exc).__name__}: {exc}", "results": []},
            )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CatalogServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], catalog: CatalogSingleton) -> None:
        super().__init__(address, CatalogHandler)
        self.catalog = catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65490)
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument("--expected-index-manifest-sha256", required=True)
    parser.add_argument("--expected-items-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = CatalogSingleton(
        args.index,
        args.items,
        price_seed=args.price_seed,
        expected_index_manifest_sha256=args.expected_index_manifest_sha256,
        expected_items_sha256=args.expected_items_sha256,
    )
    CatalogServer((args.host, args.port), catalog).serve_forever()


if __name__ == "__main__":
    main()
