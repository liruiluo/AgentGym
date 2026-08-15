"""Build the accelerated LiteResearcher Tantivy index from the released corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time

import tantivy


INDEX_CONTRACT = "combined_title2_document_bm25_v1"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--heap-bytes", type=int, default=16_000_000_000)
    parser.add_argument("--commit-every", type=int, default=1_000_000)
    parser.add_argument("--fetch-size", type=int, default=10_000)
    args = parser.parse_args()

    database = Path(args.database).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()
    if not database.is_file():
        parser.error(f"SQLite corpus does not exist: {database}")
    if index_path.exists():
        parser.error(f"refusing to reuse Tantivy index path: {index_path}")
    if args.threads < 1 or args.heap_bytes < 15_000_000:
        parser.error("threads and heap-bytes must be positive")
    if args.commit_every < 1 or args.fetch_size < 1:
        parser.error("commit-every and fetch-size must be positive")
    index_path.mkdir(parents=True)

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("id", stored=True, fast=True)
    schema_builder.add_text_field("content", stored=False)
    schema = schema_builder.build()
    index = tantivy.Index(schema, path=str(index_path), reuse=True)
    writer = index.writer(heap_size=args.heap_bytes, num_threads=args.threads)

    connection = sqlite3.connect(
        f"file:{database}?mode=ro&immutable=1",
        uri=True,
    )
    connection.execute("PRAGMA query_only=ON")
    document_count = int(
        connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    )
    cursor = connection.execute(
        "SELECT id, title, document FROM documents ORDER BY id"
    )
    progress_path = index_path / "build-progress.json"
    started = time.perf_counter()
    indexed = 0
    last_commit = 0
    while True:
        rows = cursor.fetchmany(args.fetch_size)
        if not rows:
            break
        for document_id, title, document in rows:
            writer.add_document(
                tantivy.Document(
                    id=[int(document_id)],
                    content=[f"{title} {title} {document}"],
                )
            )
            indexed += 1
        if indexed - last_commit >= args.commit_every:
            writer.commit()
            index.reload()
            committed = int(index.searcher().num_docs)
            elapsed = time.perf_counter() - started
            progress = {
                "contract": INDEX_CONTRACT,
                "documents_per_second": committed / max(elapsed, 1e-9),
                "elapsed_seconds": elapsed,
                "indexed": committed,
                "status": "building",
                "target": document_count,
            }
            _write_json(progress_path, progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            last_commit = indexed

    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    committed = int(index.searcher().num_docs)
    elapsed = time.perf_counter() - started
    if committed != document_count:
        raise RuntimeError(
            f"Tantivy document count mismatch: {committed} != {document_count}"
        )
    progress = {
        "contract": INDEX_CONTRACT,
        "documents_per_second": committed / max(elapsed, 1e-9),
        "elapsed_seconds": elapsed,
        "indexed": committed,
        "status": "complete",
        "target": document_count,
        "tantivy_version": getattr(tantivy, "__version__", "unknown"),
    }
    _write_json(progress_path, progress)
    _write_json(
        index_path / "agentmemory-index.json",
        {
            "contract": INDEX_CONTRACT,
            "document_count": committed,
            "tantivy_version": getattr(tantivy, "__version__", "unknown"),
        },
    )
    print(json.dumps(progress, sort_keys=True))


if __name__ == "__main__":
    main()
