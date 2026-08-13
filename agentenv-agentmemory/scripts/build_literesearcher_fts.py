#!/usr/bin/env python3
"""Build the resident lexical LiteResearcher corpus used by the AMG fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import zstandard


SCHEMA = "agentmemory_literesearcher_released_search_corpus_fts_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--progress-file")
    return parser.parse_args()


def write_progress(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    progress = Path(args.progress_file).resolve() if args.progress_file else None
    if not source.is_file() or args.batch_size <= 0:
        raise SystemExit("invalid source or batch size")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    started = time.time()
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-2097152")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        "CREATE TABLE documents ("
        "id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, "
        "title TEXT NOT NULL, document TEXT NOT NULL)"
    )
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    rows_seen = 0
    rows_inserted = 0
    batch: list[tuple[str, str, str]] = []
    with source.open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        import io

        with io.TextIOWrapper(reader, encoding="utf-8") as text:
            for line in text:
                rows_seen += 1
                try:
                    row = json.loads(line)
                    url = str(row["url"]).strip()
                    title = str(row.get("title", "")).strip()
                    document = str(row.get("doc", "")).strip()
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not url or not document:
                    continue
                batch.append((url, title, document))
                if len(batch) < args.batch_size:
                    continue
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO documents(url, title, document) VALUES (?, ?, ?)",
                    batch,
                )
                rows_inserted += connection.total_changes - before
                batch.clear()
                if rows_seen % (args.batch_size * 10) == 0:
                    connection.commit()
                    write_progress(
                        progress,
                        {
                            "phase": "load_documents",
                            "rows_seen": rows_seen,
                            "rows_inserted": rows_inserted,
                            "elapsed_seconds": round(time.time() - started, 3),
                        },
                    )
    if batch:
        before = connection.total_changes
        connection.executemany(
            "INSERT OR IGNORE INTO documents(url, title, document) VALUES (?, ?, ?)",
            batch,
        )
        rows_inserted += connection.total_changes - before
    connection.commit()

    write_progress(
        progress,
        {
            "phase": "build_fts",
            "rows_seen": rows_seen,
            "rows_inserted": rows_inserted,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )
    connection.execute(
        "CREATE VIRTUAL TABLE documents_fts USING fts5("
        "title, document, content='documents', content_rowid='id', "
        "tokenize='porter unicode61 remove_diacritics 2')"
    )
    connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
    connection.execute("CREATE INDEX documents_url_idx ON documents(url)")
    metadata = {
        "schema": SCHEMA,
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_sha256(source),
        "rows_seen": rows_seen,
        "rows_inserted": rows_inserted,
        "built_at_unix": int(time.time()),
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
    )
    connection.execute("PRAGMA optimize")
    connection.commit()
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    if quick_check != "ok":
        raise RuntimeError(f"SQLite quick_check failed: {quick_check}")
    connection.close()
    final = {
        **metadata,
        "phase": "complete",
        "database_path": str(output),
        "database_bytes": output.stat().st_size,
        "elapsed_seconds": round(time.time() - started, 3),
        "quick_check": quick_check,
    }
    write_progress(progress, final)
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
