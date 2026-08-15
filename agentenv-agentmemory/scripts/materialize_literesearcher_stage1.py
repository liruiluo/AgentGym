#!/usr/bin/env python3
"""Freeze the semantically reviewed LiteResearcher Stage-1 intake corpus.

The script reads an exact Hugging Face dataset revision and a checked-in human
semantic audit.  It fetches only the audit-approved English Wikipedia rows,
verifies their question, target, URL, content hash, and supporting passage, and
writes a deterministic 64-train/8-held-out manifest.  Gold is used only as a
certifier: page bodies always come from the source fetch and are never
synthesized from the answer field.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASET = "simplex-ai-inc/LiteResearcher-Data"
CONFIG = "stage1"
SPLIT = "train"
DATA_REVISION = "fff6b0cfef718859543a16f542ea248d30d1ac34"
UPSTREAM_COMMIT = "779e7d5f6a043d4100149ba0992a39507f69a974"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"
STAGE1_PARQUET_SHA256 = "493f3d0cc87dc5f0f42340d3891d9df0f8b687d496c911847cd479250610371d"
JINA_READER = "https://r.jina.ai/http://en.wikipedia.org"
USER_AGENT = "AgentMemoryGym-LiteResearcher-intake/1.0"
LICENSE_NOTE = (
    "Wikipedia page text is available under CC BY-SA 4.0 and, unless noted "
    "otherwise, GFDL; attribution is available from the resolved page history."
)
EXTRACTION_METHOD = "jina_reader_wikipedia_plaintext_v1"
MAX_PAGE_CHARS = 128_000
DEFAULT_AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "literesearcher_stage1_semantic_audit.json"
)
DEFAULT_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "agentmemorygym"
    / "literesearcher-stage1"
    / DATA_REVISION[:12]
)


@dataclass(frozen=True)
class MaterializedRow:
    index: int
    question: str
    targets: tuple[str, ...]
    mask_url: str
    resolved_url: str
    page_title: str
    page_text: str
    content_sha256: str
    evidence_anchors: tuple[str, ...]
    source_normalization_reason: str


@dataclass(frozen=True)
class FetchOutcome:
    index: int
    row: MaterializedRow | None
    category: str
    detail: str
    source: str

    @property
    def ok(self) -> bool:
        return self.row is not None


class SourceMaterializationError(RuntimeError):
    def __init__(self, failures: Iterable[FetchOutcome]) -> None:
        self.failures = tuple(failures)
        details = "; ".join(
            f"{failure.index}:{failure.category}:{failure.detail}"
            for failure in self.failures
        )
        super().__init__(f"source materialization failed: {details}")


def _request_json(url: str, *, attempts: int = 5) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("JSON endpoint returned a non-object")
            return payload
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _dataset_rows(
    scan_rows: int, *, parquet_path: Path | None = None
) -> list[dict[str, Any]]:
    if parquet_path is not None:
        actual_digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        if actual_digest != STAGE1_PARQUET_SHA256:
            raise ValueError(
                f"Stage-1 parquet SHA256 mismatch: {actual_digest}"
            )
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "reading --parquet requires pyarrow; run with `uv run --with pyarrow`"
            ) from exc
        table = parquet.read_table(parquet_path).slice(0, scan_rows)
        return [
            {"row_idx": index, "row": row}
            for index, row in enumerate(table.to_pylist())
        ]

    rows: list[dict[str, Any]] = []
    for offset in range(0, scan_rows, 100):
        length = min(100, scan_rows - offset)
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": length,
                "revision": DATA_REVISION,
            }
        )
        payload = _request_json(f"{DATASET_SERVER}?{query}")
        page_rows = payload.get("rows")
        if not isinstance(page_rows, list) or len(page_rows) != length:
            raise RuntimeError(
                f"dataset server returned {len(page_rows or [])} rows at offset {offset}"
            )
        rows.extend(page_rows)
    return rows


def _normalize_source_url(raw_url: str) -> tuple[str, str]:
    source = raw_url.strip()
    resolved = source
    while resolved.endswith(")") and resolved.count(")") > resolved.count("("):
        resolved = resolved[:-1]
    reason = "none" if resolved == source else "removed_unbalanced_trailing_parenthesis"
    return resolved, reason


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _targets(record: dict[str, Any]) -> tuple[str, ...]:
    try:
        raw_targets = record["reward_model"]["ground_truth"]["target"]
    except (KeyError, TypeError) as exc:
        raise ValueError("dataset row has no ground-truth target list") from exc
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("dataset row target must be a non-empty list")
    targets = tuple(str(item).strip() for item in raw_targets)
    if any(not item for item in targets):
        raise ValueError("dataset row target contains an empty alias")
    return targets


def _candidate(record: dict[str, Any]) -> tuple[int, dict[str, Any], str, str] | None:
    row = record.get("row")
    index = record.get("row_idx")
    if not isinstance(row, dict) or isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("dataset server row has an invalid shape")
    try:
        mask_url = str(row["extra_info"]["mask_url"])
    except (KeyError, TypeError) as exc:
        raise ValueError("dataset row has no mask_url") from exc
    resolved_url, reason = _normalize_source_url(mask_url)
    parsed = urllib.parse.urlparse(resolved_url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "en.wikipedia.org":
        return None
    if not parsed.path.startswith("/wiki/"):
        return None
    return index, row, resolved_url, reason


def _cache_path(cache_dir: Path, expected_sha256: str) -> Path:
    return cache_dir / expected_sha256[:2] / f"{expected_sha256}.txt"


def _write_cache(path: Path, page_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(page_text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _materialized_row(
    candidate: tuple[int, dict[str, Any], str, str],
    page_text: str,
) -> FetchOutcome:
    index, row, resolved_url, reason = candidate
    targets = _targets(row)
    normalized_page = _normalized_text(page_text)
    anchors = tuple(
        target
        for target in targets
        if len(_normalized_text(target)) >= 2
        and _normalized_text(target) in normalized_page
    )
    if not anchors:
        return FetchOutcome(
            index=index,
            row=None,
            category="target_anchor_missing",
            detail="no verifier target alias occurs in the fetched page",
            source="validated_text",
        )
    question = str(row.get("question", "")).strip()
    if not question:
        return FetchOutcome(
            index=index,
            row=None,
            category="dataset_record_invalid",
            detail="question is empty",
            source="dataset",
        )
    parsed = urllib.parse.urlparse(resolved_url)
    title = urllib.parse.unquote(parsed.path[len("/wiki/") :]).replace("_", " ")
    return FetchOutcome(
        index=index,
        row=MaterializedRow(
            index=index,
            question=question,
            targets=targets,
            mask_url=str(row["extra_info"]["mask_url"]).strip(),
            resolved_url=resolved_url,
            page_title=title,
            page_text=page_text,
            content_sha256=hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
            evidence_anchors=anchors,
            source_normalization_reason=reason,
        ),
        category="verified",
        detail="source text passed hash and target-anchor checks",
        source="validated_text",
    )


def _fetch_candidate(
    candidate: tuple[int, dict[str, Any], str, str],
    *,
    expected_sha256: str,
    cache_dir: Path | None,
    attempts: int,
) -> FetchOutcome:
    index, _, resolved_url, _ = candidate
    cache_problem = ""
    if cache_dir is not None:
        path = _cache_path(cache_dir, expected_sha256)
        if path.is_file():
            cached_text = path.read_text(encoding="utf-8")
            actual_sha256 = hashlib.sha256(cached_text.encode("utf-8")).hexdigest()
            if actual_sha256 == expected_sha256:
                outcome = _materialized_row(candidate, cached_text)
                if outcome.ok:
                    return FetchOutcome(
                        index=index,
                        row=outcome.row,
                        category="verified",
                        detail="content-addressed cache hit",
                        source="cache",
                    )
                return outcome
            cache_problem = (
                f"cache hash mismatch: expected {expected_sha256}, got {actual_sha256}; "
            )

    parsed = urllib.parse.urlparse(resolved_url)
    reader_url = JINA_READER + parsed.path
    request = urllib.request.Request(
        reader_url,
        headers={"User-Agent": USER_AGENT, "X-Return-Format": "text"},
    )
    last_error = ""
    page_text = ""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=75) as response:
                raw = response.read(MAX_PAGE_CHARS + 1)
            if len(raw) > MAX_PAGE_CHARS:
                return FetchOutcome(
                    index=index,
                    row=None,
                    category="page_too_large",
                    detail=f"source response exceeds {MAX_PAGE_CHARS} bytes",
                    source="network",
                )
            page_text = raw.decode("utf-8", errors="replace").strip()
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    if not page_text:
        return FetchOutcome(
            index=index,
            row=None,
            category="network_fetch_failed",
            detail=cache_problem + (last_error or "empty source response"),
            source="network",
        )

    actual_sha256 = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        return FetchOutcome(
            index=index,
            row=None,
            category="content_hash_changed",
            detail=f"expected {expected_sha256}, got {actual_sha256}",
            source="network",
        )
    outcome = _materialized_row(candidate, page_text)
    if not outcome.ok:
        return outcome
    if cache_dir is not None:
        _write_cache(_cache_path(cache_dir, expected_sha256), page_text)
    return FetchOutcome(
        index=index,
        row=outcome.row,
        category="verified",
        detail="network source fetched and cached",
        source="network",
    )


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _targets_sha256(targets: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(targets),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_semantic_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic audit must be an object")
    if payload.get("schema") != "agentmemory_literesearcher_semantic_audit_v1":
        raise ValueError("unsupported semantic audit schema")
    if payload.get("data_revision") != DATA_REVISION:
        raise ValueError("semantic audit has the wrong data revision")
    unsigned = dict(payload)
    unsigned.pop("audit_sha256", None)
    digest = hashlib.sha256(_canonical_payload(unsigned)).hexdigest()
    if payload.get("audit_sha256") != digest:
        raise ValueError("semantic audit SHA256 mismatch")
    approved = payload.get("approved")
    rejected = payload.get("rejected")
    if not isinstance(approved, list) or len(approved) != 72:
        raise ValueError("semantic audit must approve exactly 72 rows")
    if not isinstance(rejected, list) or not rejected:
        raise ValueError("semantic audit must record rejected rows")
    return payload


def _record(row: MaterializedRow, *, retrieved_at: str) -> dict[str, Any]:
    return {
        "index": row.index,
        "question": row.question,
        "targets": list(row.targets),
        "mask_url": row.mask_url,
        "public_url": f"https://literesearcher.local/page/{row.index:05d}",
        "page_title": row.page_title,
        "page_text": row.page_text,
        "resolved_url": row.resolved_url,
        "retrieved_at": retrieved_at,
        "content_sha256": row.content_sha256,
        "extraction_method": EXTRACTION_METHOD,
        "license_note": LICENSE_NOTE,
        "evidence_anchors": list(row.evidence_anchors),
        "source_normalization_reason": row.source_normalization_reason,
    }


def _retrieved_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--retrieved-at must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def build_manifest(
    *,
    scan_rows: int,
    workers: int,
    retrieved_at: str,
    audit_path: Path,
    parquet_path: Path | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    attempts: int = 4,
    retry_rounds: int = 2,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if retry_rounds < 0:
        raise ValueError("retry_rounds must be non-negative")
    audit = _load_semantic_audit(audit_path)
    approved_records = audit["approved"]
    approved_indices = [record["index"] for record in approved_records]
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in approved_indices
    ):
        raise ValueError("semantic audit contains an invalid approved index")
    if len(set(approved_indices)) != 72:
        raise ValueError("semantic audit approved indices must be unique")
    if approved_indices != sorted(approved_indices):
        raise ValueError("semantic audit approved indices must be sorted")
    if scan_rows <= approved_indices[-1]:
        raise ValueError(
            f"--scan-rows must exceed the final approved index {approved_indices[-1]}"
        )
    approved_index_set = set(approved_indices)
    heldout_indices_raw = audit.get("heldout_indices")
    if heldout_indices_raw is None:
        heldout_indices = approved_indices[-8:]
    else:
        if not isinstance(heldout_indices_raw, list) or len(heldout_indices_raw) != 8:
            raise ValueError("semantic audit heldout_indices must list exactly 8 rows")
        heldout_indices = []
        for index in heldout_indices_raw:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("semantic audit heldout_indices contains an invalid index")
            heldout_indices.append(index)
        if len(set(heldout_indices)) != 8:
            raise ValueError("semantic audit heldout_indices must be unique")
        if not set(heldout_indices).issubset(approved_index_set):
            raise ValueError("semantic audit heldout_indices must be approved rows")
        heldout_indices.sort()
    heldout_index_set = set(heldout_indices)
    train_indices = [index for index in approved_indices if index not in heldout_index_set]
    if len(train_indices) != 64:
        raise ValueError("semantic audit split must produce exactly 64 train rows")
    candidates = [
        candidate
        for record in _dataset_rows(scan_rows, parquet_path=parquet_path)
        if record.get("row_idx") in approved_index_set
        and (candidate := _candidate(record)) is not None
    ]
    if {candidate[0] for candidate in candidates} != approved_index_set:
        raise RuntimeError("one or more approved rows are not English Wikipedia rows")
    audit_by_index = {record["index"]: record for record in approved_records}
    pending = {candidate[0]: candidate for candidate in candidates}
    outcomes: dict[int, FetchOutcome] = {}
    for retry_round in range(retry_rounds + 1):
        if not pending:
            break
        round_workers = workers if retry_round == 0 else 1
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=round_workers
        ) as executor:
            futures = {
                executor.submit(
                    _fetch_candidate,
                    candidate,
                    expected_sha256=audit_by_index[index]["content_sha256"],
                    cache_dir=cache_dir,
                    attempts=attempts,
                ): index
                for index, candidate in pending.items()
            }
            for future in concurrent.futures.as_completed(futures):
                outcome = future.result()
                outcomes[outcome.index] = outcome
        verified_count = sum(outcome.ok for outcome in outcomes.values())
        print(
            f"fetch round {retry_round + 1}/{retry_rounds + 1}: "
            f"checked {len(pending)} rows; verified {verified_count}/72",
            flush=True,
        )
        pending = {
            index: pending[index]
            for index, outcome in outcomes.items()
            if index in pending and outcome.category == "network_fetch_failed"
        }

    missing_indices = [index for index in approved_indices if index not in outcomes]
    if missing_indices:
        raise RuntimeError(
            f"source fetch produced no outcome for indices {missing_indices}"
        )
    failures = [
        outcomes[index] for index in approved_indices if not outcomes[index].ok
    ]
    if failures:
        raise SourceMaterializationError(failures)
    verified = [outcomes[index].row for index in approved_indices]
    if any(row is None for row in verified):
        raise AssertionError("verified source outcome unexpectedly lacks a row")
    verified = [row for row in verified if row is not None]
    if [row.index for row in verified] != approved_indices:
        raise RuntimeError("one or more semantic-audit rows failed source matching")
    for row in verified:
        record = audit_by_index[row.index]
        if record.get("review_status") != "approved":
            raise ValueError(f"row {row.index} is not approved")
        if record.get("question_sha256") != _text_sha256(row.question):
            raise ValueError(f"row {row.index} question SHA256 mismatch")
        if record.get("targets_sha256") != _targets_sha256(row.targets):
            raise ValueError(f"row {row.index} target SHA256 mismatch")
        if record.get("resolved_url") != row.resolved_url:
            raise ValueError(f"row {row.index} resolved URL mismatch")
        if record.get("content_sha256") != row.content_sha256:
            raise ValueError(f"row {row.index} source content changed; re-audit required")
        quote = str(record.get("evidence_quote", ""))
        if not quote or quote not in row.page_text:
            raise ValueError(f"row {row.index} evidence quote is absent")
        normalized_quote = _normalized_text(quote)
        if not any(_normalized_text(target) in normalized_quote for target in row.targets):
            raise ValueError(f"row {row.index} evidence quote lacks a target alias")

    verified_by_index = {row.index: row for row in verified}
    train = [verified_by_index[index] for index in train_indices]
    heldout = [verified_by_index[index] for index in heldout_indices]
    payload: dict[str, Any] = {
        "schema": "agentmemory_literesearcher_coverage_v3",
        "dataset": DATASET,
        "config": CONFIG,
        "data_revision": DATA_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
        "selection_contract": (
            "human_semantic_reviewed_64_train_8_heldout_explicit_subset_v2"
        ),
        "page_fixture_contract": (
            "source_backed_semantically_reviewed_frozen_text_v1"
        ),
        "materializer": "scripts/materialize_literesearcher_stage1.py",
        "dataset_parquet_sha256": STAGE1_PARQUET_SHA256,
        "retrieved_at": retrieved_at,
        "scan_rows": scan_rows,
        "semantic_audit": {
            "file": audit_path.name,
            "sha256": audit["audit_sha256"],
            "approved_count": len(audit["approved"]),
            "rejected_count": len(audit["rejected"]),
            "replacement_count": audit["summary"]["replacement_count"],
            "source_backed_ratio": audit["summary"]["source_backed_ratio"],
        },
        "train": [_record(row, retrieved_at=retrieved_at) for row in train],
        "heldout": [_record(row, retrieved_at=retrieved_at) for row in heldout],
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_payload(payload)).hexdigest()
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-rows", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--retry-rounds", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--failure-report", type=Path)
    parser.add_argument("--retrieved-at")
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scan_rows < 72:
        raise ValueError("--scan-rows must be at least 72")
    retrieved_at = _retrieved_at(args.retrieved_at)
    try:
        manifest = build_manifest(
            scan_rows=args.scan_rows,
            workers=args.workers,
            retrieved_at=retrieved_at,
            audit_path=args.audit.expanduser().resolve(),
            parquet_path=args.parquet,
            cache_dir=args.cache_dir.expanduser().resolve(),
            attempts=args.attempts,
            retry_rounds=args.retry_rounds,
        )
    except SourceMaterializationError as exc:
        if args.failure_report is not None:
            failure_report = {
                "schema": "agentmemory_literesearcher_source_failures_v1",
                "data_revision": DATA_REVISION,
                "failures": [
                    {
                        "index": failure.index,
                        "category": failure.category,
                        "detail": failure.detail,
                        "source": failure.source,
                    }
                    for failure in exc.failures
                ],
            }
            args.failure_report.parent.mkdir(parents=True, exist_ok=True)
            args.failure_report.write_text(
                json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_indices": [item["index"] for item in manifest["train"]],
                "heldout_indices": [item["index"] for item in manifest["heldout"]],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
