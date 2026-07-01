from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .memoryarena_converter import (
    catalog_lookup_tokens,
    catalog_title_match_score,
    expand_catalog_paths,
    extract_product_asin,
    extract_product_metadata,
    extract_product_title,
    iter_catalog_products,
    metadata_quality,
)


@dataclass(frozen=True)
class CatalogSearchResult:
    title: str
    average_rating: float | None
    price_usd: float | None
    total_reviews: int | None
    match_score: int

    def render(self) -> str:
        fields = {
            "average_rating": self.average_rating,
            "price_usd": self.price_usd,
            "total_reviews": self.total_reviews,
            "match_score": self.match_score,
        }
        attrs = ", ".join(f"{key}={render_value(value)}" for key, value in fields.items())
        return f"- {self.title} ({attrs})"


def build_sqlite_catalog_index(
    catalog_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    replace: bool = True,
    limit: int | None = None,
) -> int:
    """Build a compact SQLite/FTS product-title index from MemoryArena catalog files."""
    index_path = Path(output_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if replace and index_path.exists():
        index_path.unlink()
    with sqlite3.connect(index_path) as db:
        configure_for_bulk_load(db)
        ensure_schema(db)
        product_count = 0
        for catalog_path in expand_catalog_paths(catalog_paths):
            for product in iter_catalog_products(catalog_path):
                title = extract_product_title(product)
                if not title:
                    continue
                metadata = extract_product_metadata(product)
                asin = extract_product_asin(product) or ""
                insert_product(db, asin=asin, title=title, metadata=metadata, source_path=str(catalog_path))
                product_count += 1
                if limit is not None and product_count >= limit:
                    db.commit()
                    return product_count
        db.commit()
        return product_count


def configure_for_bulk_load(db: sqlite3.Connection) -> None:
    # The index is a reproducible derived artifact on the shared disk. Favor
    # bounded build time over crash-safe journaling during this one-shot build.
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA locking_mode=EXCLUSIVE")


def ensure_schema(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            asin TEXT,
            title TEXT NOT NULL,
            average_rating REAL,
            price_usd REAL,
            total_reviews INTEGER,
            source_path TEXT
        )
        """
    )
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(title)")


def insert_product(
    db: sqlite3.Connection,
    *,
    asin: str,
    title: str,
    metadata: dict[str, Any],
    source_path: str,
) -> None:
    cursor = db.execute(
        """
        INSERT INTO products(asin, title, average_rating, price_usd, total_reviews, source_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            asin,
            title,
            metadata.get("average_rating"),
            metadata.get("price_usd"),
            metadata.get("total_reviews"),
            source_path,
        ),
    )
    rowid = cursor.lastrowid
    db.execute("INSERT INTO products_fts(rowid, title) VALUES (?, ?)", (rowid, title))


def search_sqlite_catalog(index_path: str | Path, query: str, *, top_k: int = 3) -> list[CatalogSearchResult]:
    tokens = catalog_lookup_tokens(query)
    if not tokens:
        return []
    candidate_limit = max(top_k * 25, 50)
    with sqlite3.connect(index_path) as db:
        rows = []
        for match_query in fts_match_queries(tokens):
            rows = query_fts_rows(db, match_query, candidate_limit)
            if rows:
                break
    results = []
    for title, average_rating, price_usd, total_reviews, rank in rows:
        del rank
        score = catalog_title_match_score(query, title)
        meta = {"average_rating": average_rating, "price_usd": price_usd, "total_reviews": total_reviews}
        results.append(
            (
                score,
                metadata_quality(meta),
                CatalogSearchResult(
                    title=title,
                    average_rating=average_rating,
                    price_usd=price_usd,
                    total_reviews=total_reviews,
                    match_score=score,
                ),
            )
        )
    results.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in dedupe_titles(results)[:top_k]]


def fts_match_queries(tokens: list[str]) -> list[str]:
    unique_tokens = list(dict.fromkeys(tokens))
    # Exact-ish title searches should be narrow and fast. FTS5's whitespace
    # operator is AND, while wide OR over million-row catalogs can be very slow.
    queries = []
    for width in (12, 8, 6, 5, 4, 3):
        if len(unique_tokens) >= width:
            queries.append(" ".join(quote_fts_token(token) for token in unique_tokens[:width]))
    relaxed_tokens = sorted(unique_tokens, key=lambda token: (len(token), token), reverse=True)
    for width in (5, 4, 3):
        if len(relaxed_tokens) >= width:
            queries.append(" ".join(quote_fts_token(token) for token in relaxed_tokens[:width]))
    # Last-resort broad search is intentionally tiny; large OR queries over the
    # full MemoryArena catalog are too slow for interactive agent steps.
    if relaxed_tokens:
        queries.append(" OR ".join(quote_fts_token(token) for token in relaxed_tokens[:3]))
    return [query for query in queries if query]


def query_fts_rows(db: sqlite3.Connection, match_query: str, candidate_limit: int) -> list[tuple[Any, ...]]:
    return db.execute(
        """
        SELECT
            p.title,
            p.average_rating,
            p.price_usd,
            p.total_reviews,
            bm25(products_fts) AS rank
        FROM products_fts
        JOIN products AS p ON p.rowid = products_fts.rowid
        WHERE products_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (match_query, candidate_limit),
    ).fetchall()


def dedupe_titles(
    scored_results: Iterable[tuple[int, int, CatalogSearchResult]]
) -> list[tuple[int, int, CatalogSearchResult]]:
    seen: set[str] = set()
    deduped = []
    for item in scored_results:
        key = item[2].title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def quote_fts_token(token: str) -> str:
    safe = "".join(ch for ch in token.lower() if ch.isalnum())
    return json.dumps(safe)


def render_value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)
