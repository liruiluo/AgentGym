from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .schema import ProceduralMemoryDataError


_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lucene_index_manifest(
    manifest: str | Path,
    *,
    index_dir: str | Path,
) -> int:
    """Verify every index byte and reject unlisted or missing files."""

    manifest_path = Path(manifest)
    index_path = Path(index_dir)
    if not index_path.is_dir():
        raise ProceduralMemoryDataError(
            f"Lucene index directory is missing: {index_path}"
        )
    seen: set[str] = set()
    count = 0
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = _MANIFEST_LINE_RE.fullmatch(raw_line)
        if match is None:
            raise ProceduralMemoryDataError(
                f"invalid Lucene manifest line {line_number}: {raw_line!r}."
            )
        expected, raw_relative = match.groups()
        relative = raw_relative[2:] if raw_relative.startswith("./") else raw_relative
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ProceduralMemoryDataError(
                f"unsafe Lucene manifest path at line {line_number}: {raw_relative!r}."
            )
        canonical = relative_path.as_posix()
        if canonical in seen:
            raise ProceduralMemoryDataError(
                f"duplicate Lucene manifest path: {canonical!r}."
            )
        seen.add(canonical)
        path = index_path / relative_path
        if not path.is_file():
            raise ProceduralMemoryDataError(f"Lucene index file is missing: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise ProceduralMemoryDataError(
                f"Lucene index file SHA256 mismatch for {canonical}: "
                f"expected {expected}, observed {observed}."
            )
        count += 1
    if count == 0:
        raise ProceduralMemoryDataError("Lucene index manifest is empty.")
    filesystem_files = {
        path.relative_to(index_path).as_posix()
        for path in index_path.rglob("*")
        if path.is_file()
    }
    if filesystem_files != seen:
        raise ProceduralMemoryDataError(
            "Lucene index file set differs from the frozen manifest: "
            f"missing={sorted(seen - filesystem_files)} "
            f"extra={sorted(filesystem_files - seen)}."
        )
    return count
