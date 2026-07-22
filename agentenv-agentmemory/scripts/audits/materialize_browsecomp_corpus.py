#!/usr/bin/env python3
"""Materialize the frozen BrowseComp parquet corpus for OpenAISearcher.

The upstream parquet shards contain exactly ``docid``, ``text`` and ``url``.
OpenAISearcher reads only the first two fields from JSONL, so this audit tool
projects those fields in deterministic shard/row order and records hashes and
the source schema in a sidecar manifest. It deliberately refuses unknown or
incomplete schemas; it is not a downloader or a surrogate corpus builder.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SOURCE_COLUMNS = ("docid", "text", "url")
OUTPUT_COLUMNS = ("docid", "text")
FROZEN_CORPUS_REPOSITORY = "Tevatron/browsecomp-plus-corpus"
FROZEN_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"


class CorpusMaterializationError(ValueError):
    """Raised when a parquet corpus cannot be proven compatible."""


def materialize_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, str]]:
    """Validate source rows and yield the lossless OpenAISearcher projection.

    The function is dependency-free so schema behavior can be unit-tested on a
    small fixture without installing pyarrow.
    """

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        keys = set(row)
        missing = set(SOURCE_COLUMNS) - keys
        unknown = keys - set(SOURCE_COLUMNS)
        if missing or unknown:
            raise CorpusMaterializationError(
                f"row {row_number} schema mismatch; missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        docid = row["docid"]
        text = row["text"]
        url = row["url"]
        if not isinstance(docid, str) or not docid.strip():
            raise CorpusMaterializationError(f"row {row_number} has invalid docid")
        if not isinstance(text, str) or not text:
            raise CorpusMaterializationError(f"row {row_number} has invalid text")
        if not isinstance(url, str) or not url.strip():
            raise CorpusMaterializationError(f"row {row_number} has invalid url")
        if docid in seen:
            raise CorpusMaterializationError(f"duplicate docid {docid!r}")
        seen.add(docid)
        yield {"docid": docid, "text": text}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_parquet_corpus(
    input_glob: str,
    output_path: Path,
    manifest_path: Path,
    *,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Project sorted parquet shards and write a reproducible manifest."""

    if batch_size <= 0:
        raise CorpusMaterializationError("batch_size must be positive")
    input_paths = tuple(Path(value).resolve() for value in sorted(glob.glob(input_glob)))
    if not input_paths:
        raise FileNotFoundError(f"no parquet files matched --input-glob {input_glob!r}")
    if any(not path.is_file() for path in input_paths):
        raise FileNotFoundError("one or more parquet inputs are not regular files")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise SystemExit(
            "pyarrow is required to materialize parquet; install it in the audit environment"
        ) from exc

    output_path = output_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    source_files: list[dict[str, Any]] = []
    total_rows = 0
    seen_docids: set[str] = set()
    output_digest = hashlib.sha256()
    temporary_output: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_output = Path(handle.name)
            for input_path in input_paths:
                parquet_file = parquet.ParquetFile(input_path)
                columns = tuple(parquet_file.schema_arrow.names)
                if columns != SOURCE_COLUMNS:
                    raise CorpusMaterializationError(
                        f"{input_path} schema {columns!r} does not exactly match "
                        f"{SOURCE_COLUMNS!r}"
                    )
                shard_rows = 0
                for batch in parquet_file.iter_batches(
                    columns=list(SOURCE_COLUMNS),
                    batch_size=batch_size,
                ):
                    for projected in materialize_rows(batch.to_pylist()):
                        if projected["docid"] in seen_docids:
                            raise CorpusMaterializationError(
                                f"duplicate docid {projected['docid']!r} across shards"
                            )
                        seen_docids.add(projected["docid"])
                        line = (
                            json.dumps(
                                projected,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                        handle.write(line)
                        output_digest.update(line)
                        total_rows += 1
                        shard_rows += 1
                source_files.append(
                    {
                        "path": str(input_path),
                        "sha256": sha256_file(input_path),
                        "size_bytes": input_path.stat().st_size,
                        "row_count": shard_rows,
                        "columns": list(columns),
                    }
                )
        os.replace(temporary_output, output_path)
        temporary_output = None
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

    manifest = {
        "format": "agentmemory_browsecomp_corpus_manifest_v1",
        "source": {
            "input_glob": input_glob,
            "repository": FROZEN_CORPUS_REPOSITORY,
            "revision": FROZEN_CORPUS_REVISION,
            "file_count": len(source_files),
            "files": source_files,
            "columns": list(SOURCE_COLUMNS),
        },
        "projection": {
            "output_columns": list(OUTPUT_COLUMNS),
            "discarded_columns": ["url"],
            "consumer": "MemoryArena OpenAISearcher._load_corpus",
            "ordering": "lexicographic input path, then parquet row order",
        },
        "output": {
            "path": str(output_path),
            "sha256": output_digest.hexdigest(),
            "row_count": total_rows,
        },
    }
    encoded_manifest = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_manifest: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
            delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            handle.write(encoded_manifest)
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
    finally:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically project frozen BrowseComp parquet shards to "
            "OpenAISearcher-compatible {docid,text} JSONL plus a hash manifest."
        )
    )
    parser.add_argument("--input-glob", required=True, help="Glob for parquet shards.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--manifest", required=True, help="Output manifest JSON path.")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_parquet_corpus(
        args.input_glob,
        Path(args.output),
        Path(args.manifest),
        batch_size=args.batch_size,
    )
    print(
        "BROWSECOMP_CORPUS_MATERIALIZED",
        f"rows={manifest['output']['row_count']}",
        f"output_sha256={manifest['output']['sha256']}",
        f"manifest={Path(args.manifest).expanduser().resolve()}",
    )


if __name__ == "__main__":
    main()
