"""Fail-closed whole-chain annotation manifests for native MemoryArena runs.

The gate treats one complete six-purchase chain as the only clearance unit.
It rebuilds every run manifest from pinned raw rows and canonical audit ledgers
at validation time, then requires an independently pinned manifest digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


SCHEMA = "memoryarena_annotation_gate_v1"
SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 2
FORMAL_UNIT = "one_complete_six_step_chain"
VERDICTS = frozenset({"pass", "fail", "unknown", "semantic_ambiguity"})
ALLOWED_VERDICTS = {
    "provisional": frozenset({"pass", "unknown"}),
    "strict": frozenset({"pass"}),
}
STEP_CHECKS = frozenset(
    {
        "target_alignment",
        "target_semantics",
        "compatibility",
        "metric_parse",
        "ranking",
        "bundle_budget",
    }
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AnnotationGateError(ValueError):
    """Raised when annotation evidence cannot clear a requested run."""


@dataclass(frozen=True)
class AnnotationGateTrustRoot:
    raw_dataset_sha256: str
    domain_data_sha256: str
    items_shuffle_sha256: str
    items_ins_v2_sha256: str
    lucene_index_manifest_sha256: str
    audit_summary_sha256: str
    audit_chains_sha256: str
    manual_evidence_sha256: str
    memoryarena_base_commit: str
    price_seed: int
    chain_status_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "raw_dataset_sha256",
            "domain_data_sha256",
            "items_shuffle_sha256",
            "items_ins_v2_sha256",
            "lucene_index_manifest_sha256",
            "audit_summary_sha256",
            "audit_chains_sha256",
            "manual_evidence_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise AnnotationGateError(
                    f"Trust-root {field_name} must be a lowercase SHA256 digest."
                )
        if not _COMMIT_PATTERN.fullmatch(self.memoryarena_base_commit):
            raise AnnotationGateError("Trust-root MemoryArena commit is invalid.")
        if isinstance(self.price_seed, bool) or not isinstance(self.price_seed, int):
            raise AnnotationGateError("Trust-root price seed must be an integer.")
        counts = dict(self.chain_status_counts)
        if len(counts) != len(self.chain_status_counts):
            raise AnnotationGateError("Trust-root chain status counts contain duplicates.")
        if any(status not in VERDICTS for status in counts):
            raise AnnotationGateError("Trust-root chain status counts contain an invalid verdict.")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts.values()
        ):
            raise AnnotationGateError(
                "Trust-root chain status counts must be non-negative integers."
            )


CANONICAL_TRUST_ROOT = AnnotationGateTrustRoot(
    raw_dataset_sha256="4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef",
    domain_data_sha256="2576aa9637ab6691c14f26e5f0b022b3a16c325a312ebc856c271f8e641f2afc",
    items_shuffle_sha256="2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    items_ins_v2_sha256="1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e",
    lucene_index_manifest_sha256="f3e30552ca994607291fb617735d78d0708b59b6c15ab60ebe1eb3b640e1d81a",
    audit_summary_sha256="975bc47461652f0ae2e2a6f708c5ef13bb808c738a815f805623a1c29e300ed7",
    audit_chains_sha256="fd9f17449a2063aebf78e30131fdeddd24906ce528693182a52f88b15e3352e1",
    manual_evidence_sha256="741e97c3ae034ee2feca5647663b0c23135de772656867f809ec545cd7f2d801",
    memoryarena_base_commit="6cd9de14b71915e39ac742a20dc33785e14b6aab",
    price_seed=233,
    chain_status_counts=(("fail", 8), ("semantic_ambiguity", 1), ("unknown", 141)),
)


@dataclass(frozen=True)
class SourceTreeFingerprint:
    base_commit: str
    source_tree_sha256: str
    source_file_count: int

    @property
    def tracked_worktree_sha256(self) -> str:
        return self.source_tree_sha256

    @property
    def tracked_file_count(self) -> int:
        return self.source_file_count


@dataclass(frozen=True)
class PriceTableFingerprint:
    sha256: str
    row_count: int


@dataclass(frozen=True)
class AnnotationGateBindings:
    raw_dataset_sha256: str
    domain_data_sha256: str
    items_shuffle_sha256: str
    items_ins_v2_sha256: str
    lucene_index_manifest_sha256: str
    lucene_index_file_count: int
    audit_summary_sha256: str
    audit_chains_sha256: str
    manual_evidence_sha256: str
    memoryarena_base_commit: str
    memoryarena_source_tree_sha256: str
    memoryarena_source_file_count: int
    price_seed: int
    price_table_sha256: str
    price_table_row_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "raw_dataset_sha256",
            "domain_data_sha256",
            "items_shuffle_sha256",
            "items_ins_v2_sha256",
            "lucene_index_manifest_sha256",
            "audit_summary_sha256",
            "audit_chains_sha256",
            "manual_evidence_sha256",
            "memoryarena_source_tree_sha256",
            "price_table_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not _COMMIT_PATTERN.fullmatch(self.memoryarena_base_commit):
            raise AnnotationGateError(
                "memoryarena_base_commit must be a lowercase 40-character commit."
            )
        if isinstance(self.memoryarena_source_file_count, bool) or not isinstance(
            self.memoryarena_source_file_count, int
        ):
            raise AnnotationGateError("memoryarena_source_file_count must be an integer.")
        if self.memoryarena_source_file_count < 1:
            raise AnnotationGateError("memoryarena_source_file_count must be positive.")
        if isinstance(self.lucene_index_file_count, bool) or not isinstance(
            self.lucene_index_file_count, int
        ):
            raise AnnotationGateError("lucene_index_file_count must be an integer.")
        if self.lucene_index_file_count < 1:
            raise AnnotationGateError("lucene_index_file_count must be positive.")
        if isinstance(self.price_seed, bool) or not isinstance(self.price_seed, int):
            raise AnnotationGateError("price_seed must be an integer.")
        if self.price_seed < 0:
            raise AnnotationGateError("price_seed must be non-negative.")
        if isinstance(self.price_table_row_count, bool) or not isinstance(
            self.price_table_row_count, int
        ):
            raise AnnotationGateError("price_table_row_count must be an integer.")
        if self.price_table_row_count < 1:
            raise AnnotationGateError("price_table_row_count must be positive.")

    def as_manifest(self) -> dict[str, Any]:
        return {
            "artifacts": {
                "raw_dataset": {"sha256": self.raw_dataset_sha256},
                "domain_data": {"sha256": self.domain_data_sha256},
                "items_shuffle": {"sha256": self.items_shuffle_sha256},
                "items_ins_v2": {"sha256": self.items_ins_v2_sha256},
                "lucene_index_manifest": {
                    "sha256": self.lucene_index_manifest_sha256,
                    "verified_file_count": self.lucene_index_file_count,
                },
            },
            "memoryarena_source": {
                "base_commit": self.memoryarena_base_commit,
                "physical_source_tree_sha256": self.memoryarena_source_tree_sha256,
                "source_file_count": self.memoryarena_source_file_count,
                "fingerprint_algorithm": "git_tracked_plus_untracked_sha256_v1",
            },
            "runtime_prices": {
                "seed": self.price_seed,
                "price_table_sha256": self.price_table_sha256,
                "price_table_row_count": self.price_table_row_count,
                "price_table_algorithm": "sorted_asin_price_cents_jsonl_v1",
            },
        }


@dataclass(frozen=True)
class AnnotationGateDecision:
    run_id: str
    mode: str
    manifest_sha256: str
    selected_task_ids: tuple[str, ...]
    allowed_task_ids: tuple[str, ...]
    allowed_task_ids_sha256: str
    price_table_sha256: str
    price_table_row_count: int


def sha256_file(path: str | Path) -> str:
    resolved = _require_file(path, "artifact")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_task_ids(task_ids: Iterable[str]) -> str:
    normalized = _normalize_task_ids(task_ids, label="task IDs", allow_empty=True)
    digest = hashlib.sha256()
    for task_id in sorted(normalized):
        digest.update(task_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_target_asins(target_asins: Sequence[str]) -> str:
    if len(target_asins) != 6:
        raise AnnotationGateError("A whole-chain target must contain exactly six ASINs.")
    digest = hashlib.sha256()
    for value in target_asins:
        if not isinstance(value, str) or not _ASIN_PATTERN.fullmatch(value):
            raise AnnotationGateError(f"Invalid target ASIN in audit chain: {value!r}.")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_memoryarena_source_tree(
    repo_path: str | Path,
    *,
    expected_base_commit: str | None = None,
) -> SourceTreeFingerprint:
    """Fingerprint tracked and non-ignored untracked bytes above a base commit."""

    repo = Path(repo_path).expanduser().resolve()
    if not (repo / ".git").exists():
        raise AnnotationGateError(f"MemoryArena source is not a Git worktree: {repo}")
    commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise AnnotationGateError(f"Invalid MemoryArena HEAD commit: {commit!r}.")
    if expected_base_commit is not None and commit != expected_base_commit:
        raise AnnotationGateError(
            "MemoryArena base commit mismatch: "
            f"expected {expected_base_commit}, observed {commit}."
        )
    records = _git(repo, "ls-files", "--stage", "-z").split(b"\0")
    source_rows: list[tuple[str, str, bytes]] = []
    for record in records:
        if not record:
            continue
        try:
            header, relative_bytes = record.split(b"\t", 1)
            mode_bytes, _blob, stage_bytes = header.split(b" ", 2)
        except ValueError as exc:
            raise AnnotationGateError("Malformed git ls-files output.") from exc
        if stage_bytes != b"0":
            raise AnnotationGateError("MemoryArena source has unmerged tracked files.")
        if mode_bytes == b"160000":
            raise AnnotationGateError("Git submodules are not supported in the source hash.")
        try:
            relative = os.fsdecode(relative_bytes)
        except UnicodeDecodeError as exc:
            raise AnnotationGateError("MemoryArena has a non-UTF-8 tracked path.") from exc
        path = repo / relative
        if mode_bytes == b"120000":
            payload = os.readlink(path).encode("utf-8")
        elif path.is_file():
            payload = path.read_bytes()
        else:
            raise AnnotationGateError(f"Tracked MemoryArena source is missing: {relative}")
        source_rows.append((mode_bytes.decode("ascii"), relative, payload))

    untracked_records = _git(
        repo, "ls-files", "--others", "--exclude-standard", "-z"
    ).split(b"\0")
    for relative_bytes in untracked_records:
        if not relative_bytes:
            continue
        relative = os.fsdecode(relative_bytes)
        path = repo / relative
        if path.is_symlink():
            mode = "120000"
            payload = os.readlink(path).encode("utf-8")
        elif path.is_file():
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            payload = path.read_bytes()
        else:
            raise AnnotationGateError(f"Untracked MemoryArena source is not a file: {relative}")
        source_rows.append((mode, relative, payload))

    digest = hashlib.sha256()
    for mode, relative, payload in sorted(source_rows, key=lambda row: row[1]):
        row = json.dumps(
            [mode, relative, hashlib.sha256(payload).hexdigest()],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    file_count = len(source_rows)
    if file_count == 0:
        raise AnnotationGateError("MemoryArena source tree contains no source files.")
    return SourceTreeFingerprint(
        base_commit=commit,
        source_tree_sha256=digest.hexdigest(),
        source_file_count=file_count,
    )


def verify_lucene_index_manifest(
    index_root: str | Path,
    manifest_path: str | Path,
) -> int:
    root = Path(index_root).expanduser().resolve()
    manifest = _require_file(manifest_path, "Lucene index manifest")
    if not root.is_dir():
        raise AnnotationGateError(f"Lucene index root is not a directory: {root}")
    entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AnnotationGateError(f"Cannot read Lucene index manifest: {manifest}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise AnnotationGateError(
                f"Lucene index manifest contains blank row {line_number}."
            )
        try:
            expected_sha256, relative_text = line.split(None, 1)
        except ValueError as exc:
            raise AnnotationGateError(
                f"Malformed Lucene index manifest row {line_number}."
            ) from exc
        _require_sha256(expected_sha256, f"Lucene index row {line_number}")
        relative_text = relative_text.strip()
        if relative_text.startswith("./"):
            relative_text = relative_text[2:]
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in entries
        ):
            raise AnnotationGateError(
                f"Unsafe or duplicate Lucene index path: {relative_text!r}."
            )
        entries[relative_text] = expected_sha256
    if not entries:
        raise AnnotationGateError("Lucene index manifest is empty.")

    observed_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AnnotationGateError(f"Lucene index contains a symlink: {path}")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
    if observed_files != set(entries):
        missing = sorted(set(entries) - observed_files)
        extra = sorted(observed_files - set(entries))
        raise AnnotationGateError(
            "Lucene index files disagree with the pinned manifest: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    for relative_text, expected_sha256 in entries.items():
        observed_sha256 = sha256_file(root / relative_text)
        if observed_sha256 != expected_sha256:
            raise AnnotationGateError(
                f"Lucene index SHA256 mismatch for {relative_text}: "
                f"expected {expected_sha256}, observed {observed_sha256}."
            )
    return len(entries)


def fingerprint_memoryarena_price_table(
    items_shuffle_path: str | Path,
    *,
    price_seed: int,
) -> PriceTableFingerprint:
    """Reproduce original WebShop's deterministic ASIN-to-cent price table."""

    if isinstance(price_seed, bool) or not isinstance(price_seed, int) or price_seed < 0:
        raise AnnotationGateError("price_seed must be a non-negative integer.")
    items_path = _require_file(items_shuffle_path, "items_shuffle")
    try:
        with items_path.open("r", encoding="utf-8") as handle:
            products = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationGateError(f"Cannot decode items_shuffle: {items_path}") from exc
    if not isinstance(products, list) or not products:
        raise AnnotationGateError("items_shuffle must be a non-empty JSON list.")

    rng = random.Random(price_seed)
    seen_asins: set[str] = set()
    prices: dict[str, int] = {}
    for row_number, product in enumerate(products, start=1):
        if not isinstance(product, dict):
            raise AnnotationGateError(f"items_shuffle row {row_number} is not an object.")
        asin_value = product.get("asin")
        if not isinstance(asin_value, str):
            raise AnnotationGateError(f"items_shuffle row {row_number} has no string ASIN.")
        if asin_value == "nan" or len(asin_value) > 10 or asin_value in seen_asins:
            continue
        seen_asins.add(asin_value)
        bounds = _parse_webshop_price_bounds(product.get("pricing"), row_number)
        if not bounds:
            price = 100.0
        elif len(bounds) == 1:
            price = bounds[0]
        else:
            price = rng.uniform(bounds[0], bounds[1])
        prices[asin_value] = _price_to_cents(price)
    if not prices:
        raise AnnotationGateError("items_shuffle produced an empty WebShop price table.")

    digest = hashlib.sha256()
    for asin_value, price_cents in sorted(prices.items()):
        row = json.dumps([asin_value.upper(), price_cents], separators=(",", ":"))
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return PriceTableFingerprint(sha256=digest.hexdigest(), row_count=len(prices))


def build_annotation_gate_bindings(
    *,
    raw_dataset_path: str | Path,
    domain_data_path: str | Path,
    items_shuffle_path: str | Path,
    items_ins_v2_path: str | Path,
    lucene_index_manifest_path: str | Path,
    lucene_index_root: str | Path,
    audit_summary_path: str | Path,
    audit_chains_path: str | Path,
    manual_evidence_path: str | Path,
    memoryarena_repo_path: str | Path,
    memoryarena_base_commit: str,
    price_seed: int,
) -> AnnotationGateBindings:
    price_table = fingerprint_memoryarena_price_table(
        items_shuffle_path,
        price_seed=price_seed,
    )
    return _observe_annotation_gate_bindings(
        raw_dataset_path=raw_dataset_path,
        domain_data_path=domain_data_path,
        items_shuffle_path=items_shuffle_path,
        items_ins_v2_path=items_ins_v2_path,
        lucene_index_manifest_path=lucene_index_manifest_path,
        lucene_index_root=lucene_index_root,
        audit_summary_path=audit_summary_path,
        audit_chains_path=audit_chains_path,
        manual_evidence_path=manual_evidence_path,
        memoryarena_repo_path=memoryarena_repo_path,
        memoryarena_base_commit=memoryarena_base_commit,
        price_seed=price_seed,
        observed_price_table_sha256=price_table.sha256,
        observed_price_table_row_count=price_table.row_count,
    )


def _observe_annotation_gate_bindings(
    *,
    raw_dataset_path: str | Path,
    domain_data_path: str | Path,
    items_shuffle_path: str | Path,
    items_ins_v2_path: str | Path,
    lucene_index_manifest_path: str | Path,
    lucene_index_root: str | Path,
    audit_summary_path: str | Path,
    audit_chains_path: str | Path,
    manual_evidence_path: str | Path,
    memoryarena_repo_path: str | Path,
    memoryarena_base_commit: str,
    price_seed: int,
    observed_price_table_sha256: str,
    observed_price_table_row_count: int,
) -> AnnotationGateBindings:
    source = fingerprint_memoryarena_source_tree(
        memoryarena_repo_path,
        expected_base_commit=memoryarena_base_commit,
    )
    lucene_index_file_count = verify_lucene_index_manifest(
        lucene_index_root,
        lucene_index_manifest_path,
    )
    return AnnotationGateBindings(
        raw_dataset_sha256=sha256_file(raw_dataset_path),
        domain_data_sha256=sha256_file(domain_data_path),
        items_shuffle_sha256=sha256_file(items_shuffle_path),
        items_ins_v2_sha256=sha256_file(items_ins_v2_path),
        lucene_index_manifest_sha256=sha256_file(lucene_index_manifest_path),
        lucene_index_file_count=lucene_index_file_count,
        audit_summary_sha256=sha256_file(audit_summary_path),
        audit_chains_sha256=sha256_file(audit_chains_path),
        manual_evidence_sha256=sha256_file(manual_evidence_path),
        memoryarena_base_commit=source.base_commit,
        memoryarena_source_tree_sha256=source.tracked_worktree_sha256,
        memoryarena_source_file_count=source.tracked_file_count,
        price_seed=price_seed,
        price_table_sha256=observed_price_table_sha256,
        price_table_row_count=observed_price_table_row_count,
    )


def build_annotation_gate_manifest(
    *,
    run_id: str,
    mode: str,
    raw_dataset_path: str | Path,
    audit_summary_path: str | Path,
    audit_chains_path: str | Path,
    manual_evidence_path: str | Path,
    bindings: AnnotationGateBindings,
    requested_task_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    return _build_annotation_gate_manifest(
        run_id=run_id,
        mode=mode,
        raw_dataset_path=raw_dataset_path,
        audit_summary_path=audit_summary_path,
        audit_chains_path=audit_chains_path,
        manual_evidence_path=manual_evidence_path,
        bindings=bindings,
        requested_task_ids=requested_task_ids,
        trust_root=CANONICAL_TRUST_ROOT,
    )


def _build_annotation_gate_manifest(
    *,
    run_id: str,
    mode: str,
    raw_dataset_path: str | Path,
    audit_summary_path: str | Path,
    audit_chains_path: str | Path,
    manual_evidence_path: str | Path,
    bindings: AnnotationGateBindings,
    requested_task_ids: Iterable[str] | None,
    trust_root: AnnotationGateTrustRoot,
) -> dict[str, Any]:
    """Build a deterministic run-specific whole-chain gate manifest."""

    _validate_run_id(run_id)
    allowed_verdicts = _allowed_verdicts(mode)
    raw_path = _require_file(raw_dataset_path, "raw dataset")
    summary_path = _require_file(audit_summary_path, "audit summary")
    chains_path = _require_file(audit_chains_path, "audit chains")
    manual_path = _require_file(manual_evidence_path, "manual evidence")
    _validate_bindings_against_trust_root(bindings, trust_root)

    observed_hashes = {
        "raw_dataset_sha256": sha256_file(raw_path),
        "audit_summary_sha256": sha256_file(summary_path),
        "audit_chains_sha256": sha256_file(chains_path),
        "manual_evidence_sha256": sha256_file(manual_path),
    }
    for field_name, observed in observed_hashes.items():
        expected = getattr(bindings, field_name)
        if observed != expected:
            raise AnnotationGateError(
                f"{field_name} mismatch: expected {expected}, observed {observed}."
            )

    raw_rows = _read_jsonl(raw_path, "raw dataset")
    summary = _read_json(summary_path, "audit summary")
    chain_rows = _read_jsonl(chains_path, "audit chains")
    manual = _read_json(manual_path, "manual evidence")
    _validate_manual_evidence(manual)
    _validate_summary_bindings(summary, bindings)
    raw_by_source, raw_order = _validate_raw_rows(raw_rows)
    task_rows, verdict_by_task = _validate_chain_rows(
        chain_rows,
        raw_by_source=raw_by_source,
        raw_order=raw_order,
    )
    status_counts = dict(sorted(Counter(verdict_by_task.values()).items()))
    _validate_summary_counts(summary, status_counts, len(task_rows), trust_root)

    if requested_task_ids is None:
        requested = tuple(raw_order)
    else:
        requested = _normalize_task_ids(
            requested_task_ids,
            label="requested task IDs",
            allow_empty=False,
        )
    unknown_requested = sorted(set(requested) - set(verdict_by_task))
    if unknown_requested:
        raise AnnotationGateError(
            "Requested task IDs are absent from the canonical whole-chain audit: "
            + ", ".join(unknown_requested[:5])
        )

    requested_set = set(requested)
    allowed = tuple(
        task_id
        for task_id in raw_order
        if task_id in requested_set and verdict_by_task[task_id] in allowed_verdicts
    )
    allowed_set = set(allowed)
    excluded = tuple(
        task_id
        for task_id in raw_order
        if task_id in requested_set and task_id not in allowed_set
    )
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "mode": mode,
            "requested_task_ids": list(requested),
            "requested_task_ids_sha256": hash_task_ids(requested),
        },
        "policy": {
            "formal_unit": FORMAL_UNIT,
            "allowed_verdicts": sorted(allowed_verdicts),
            "unknown_is_proven_correct": False,
            "manual_positive_evidence_allowed": False,
            "sqlite_fts_evidence_allowed": False,
            "task_id_hash_algorithm": "sorted_utf8_newline_v1",
        },
        "bindings": bindings.as_manifest(),
        "audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "summary_sha256": bindings.audit_summary_sha256,
            "chains_sha256": bindings.audit_chains_sha256,
            "manual_evidence_sha256": bindings.manual_evidence_sha256,
            "chain_count": len(task_rows),
            "chain_status_counts": status_counts,
        },
        "task_verdicts": task_rows,
        "allowed_task_ids": list(allowed),
        "allowed_task_ids_sha256": hash_task_ids(allowed),
        "excluded_task_ids": list(excluded),
        "excluded_task_ids_sha256": hash_task_ids(excluded),
    }
    return manifest


def write_annotation_gate_manifest(manifest: Mapping[str, Any], path: str | Path) -> str:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(manifest) + b"\n"
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def validate_annotation_gate_manifest(
    manifest_path: str | Path,
    *,
    expected_mode: str,
    selected_task_ids: Iterable[str],
    raw_dataset_path: str | Path,
    domain_data_path: str | Path,
    items_shuffle_path: str | Path,
    items_ins_v2_path: str | Path,
    lucene_index_manifest_path: str | Path,
    lucene_index_root: str | Path,
    audit_summary_path: str | Path,
    audit_chains_path: str | Path,
    manual_evidence_path: str | Path,
    memoryarena_repo_path: str | Path,
    memoryarena_base_commit: str,
    price_seed: int,
    expected_manifest_sha256: str,
    expected_run_id: str | None = None,
) -> AnnotationGateDecision:
    return _validate_annotation_gate_manifest(
        manifest_path,
        expected_mode=expected_mode,
        selected_task_ids=selected_task_ids,
        raw_dataset_path=raw_dataset_path,
        domain_data_path=domain_data_path,
        items_shuffle_path=items_shuffle_path,
        items_ins_v2_path=items_ins_v2_path,
        lucene_index_manifest_path=lucene_index_manifest_path,
        lucene_index_root=lucene_index_root,
        audit_summary_path=audit_summary_path,
        audit_chains_path=audit_chains_path,
        manual_evidence_path=manual_evidence_path,
        memoryarena_repo_path=memoryarena_repo_path,
        memoryarena_base_commit=memoryarena_base_commit,
        price_seed=price_seed,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_run_id=expected_run_id,
        trust_root=CANONICAL_TRUST_ROOT,
    )


def _validate_annotation_gate_manifest(
    manifest_path: str | Path,
    *,
    expected_mode: str,
    selected_task_ids: Iterable[str],
    raw_dataset_path: str | Path,
    domain_data_path: str | Path,
    items_shuffle_path: str | Path,
    items_ins_v2_path: str | Path,
    lucene_index_manifest_path: str | Path,
    lucene_index_root: str | Path,
    audit_summary_path: str | Path,
    audit_chains_path: str | Path,
    manual_evidence_path: str | Path,
    memoryarena_repo_path: str | Path,
    memoryarena_base_commit: str,
    price_seed: int,
    expected_manifest_sha256: str,
    expected_run_id: str | None,
    trust_root: AnnotationGateTrustRoot,
) -> AnnotationGateDecision:
    """Rebuild and compare the gate before allowing any selected task."""

    manifest_file = _require_file(manifest_path, "annotation gate manifest")
    manifest_sha256 = sha256_file(manifest_file)
    _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
    if manifest_sha256 != expected_manifest_sha256:
        raise AnnotationGateError(
            "Annotation gate manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {manifest_sha256}."
        )
    manifest = _read_json(manifest_file, "annotation gate manifest")
    run = manifest.get("run")
    if not isinstance(run, dict):
        raise AnnotationGateError("Annotation gate manifest is missing run metadata.")
    run_id = run.get("run_id")
    mode = run.get("mode")
    requested = run.get("requested_task_ids")
    if not isinstance(run_id, str):
        raise AnnotationGateError("Annotation gate run_id must be a string.")
    if expected_run_id is not None and run_id != expected_run_id:
        raise AnnotationGateError(
            f"Annotation gate run_id mismatch: expected {expected_run_id}, observed {run_id}."
        )
    if mode != expected_mode:
        raise AnnotationGateError(
            f"Annotation gate mode mismatch: expected {expected_mode}, observed {mode}."
        )
    if not isinstance(requested, list):
        raise AnnotationGateError("Annotation gate requested_task_ids must be a list.")

    bindings = build_annotation_gate_bindings(
        raw_dataset_path=raw_dataset_path,
        domain_data_path=domain_data_path,
        items_shuffle_path=items_shuffle_path,
        items_ins_v2_path=items_ins_v2_path,
        lucene_index_manifest_path=lucene_index_manifest_path,
        lucene_index_root=lucene_index_root,
        audit_summary_path=audit_summary_path,
        audit_chains_path=audit_chains_path,
        manual_evidence_path=manual_evidence_path,
        memoryarena_repo_path=memoryarena_repo_path,
        memoryarena_base_commit=memoryarena_base_commit,
        price_seed=price_seed,
    )
    expected = _build_annotation_gate_manifest(
        run_id=run_id,
        mode=expected_mode,
        raw_dataset_path=raw_dataset_path,
        audit_summary_path=audit_summary_path,
        audit_chains_path=audit_chains_path,
        manual_evidence_path=manual_evidence_path,
        bindings=bindings,
        requested_task_ids=requested,
        trust_root=trust_root,
    )
    if _canonical_json_bytes(manifest) != _canonical_json_bytes(expected):
        raise AnnotationGateError(
            "Annotation gate content does not match the pinned raw data and canonical audit."
        )

    selected = _normalize_task_ids(
        selected_task_ids,
        label="selected task IDs",
        allow_empty=False,
    )
    requested_set = set(requested)
    unrequested = sorted(set(selected) - requested_set)
    if unrequested:
        raise AnnotationGateError(
            "Selected task IDs were not requested by this run manifest: "
            + ", ".join(unrequested[:5])
        )
    allowed = tuple(expected["allowed_task_ids"])
    blocked = sorted(set(selected) - set(allowed))
    if blocked:
        raise AnnotationGateError(
            f"Annotation gate mode {expected_mode!r} blocks {len(blocked)} selected "
            "whole chains: "
            + ", ".join(blocked[:5])
        )
    return AnnotationGateDecision(
        run_id=run_id,
        mode=expected_mode,
        manifest_sha256=manifest_sha256,
        selected_task_ids=selected,
        allowed_task_ids=allowed,
        allowed_task_ids_sha256=expected["allowed_task_ids_sha256"],
        price_table_sha256=bindings.price_table_sha256,
        price_table_row_count=bindings.price_table_row_count,
    )


def _validate_bindings_against_trust_root(
    bindings: AnnotationGateBindings,
    trust_root: AnnotationGateTrustRoot,
) -> None:
    for field_name in (
        "raw_dataset_sha256",
        "domain_data_sha256",
        "items_shuffle_sha256",
        "items_ins_v2_sha256",
        "lucene_index_manifest_sha256",
        "audit_summary_sha256",
        "audit_chains_sha256",
        "manual_evidence_sha256",
        "memoryarena_base_commit",
        "price_seed",
    ):
        expected = getattr(trust_root, field_name)
        observed = getattr(bindings, field_name)
        if observed != expected:
            raise AnnotationGateError(
                f"Canonical trust-root mismatch for {field_name}: "
                f"expected {expected!r}, observed {observed!r}."
            )


def _validate_summary_bindings(
    summary: Mapping[str, Any], bindings: AnnotationGateBindings
) -> None:
    if summary.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise AnnotationGateError(
            f"Audit summary schema_version must be {AUDIT_SCHEMA_VERSION}."
        )
    if summary.get("formal_unit") != FORMAL_UNIT:
        raise AnnotationGateError("Audit summary does not use the six-step chain unit.")
    guards = summary.get("method_guards")
    if not isinstance(guards, dict) or guards.get(
        "upstream_annotation_audit_uses_amg_sqlite_search"
    ) is not False:
        raise AnnotationGateError("Audit summary does not explicitly exclude SQLite FTS.")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise AnnotationGateError("Audit summary is missing input provenance.")
    expected = {
        "raw_dataset_sha256": _nested(inputs, "raw_hf_jsonl", "sha256"),
        "domain_data_sha256": _nested(inputs, "domain_data", "sha256"),
        "items_shuffle_sha256": inputs.get("items_shuffle_sha256"),
        "items_ins_v2_sha256": inputs.get("items_ins_v2_sha256"),
        "lucene_index_manifest_sha256": _nested(
            inputs, "original_lucene_index_manifest", "sha256"
        ),
        "manual_evidence_sha256": _nested(inputs, "manual_evidence", "sha256"),
        "memoryarena_base_commit": _nested(inputs, "memoryarena_repo", "commit"),
    }
    for field_name, expected_value in expected.items():
        observed = getattr(bindings, field_name)
        if observed != expected_value:
            raise AnnotationGateError(
                f"Audit binding {field_name} mismatch: "
                f"summary has {expected_value!r}, run has {observed!r}."
            )


def _validate_summary_counts(
    summary: Mapping[str, Any],
    status_counts: Mapping[str, int],
    chain_count: int,
    trust_root: AnnotationGateTrustRoot,
) -> None:
    if summary.get("chain_count") != chain_count:
        raise AnnotationGateError("Audit summary chain_count disagrees with chains.jsonl.")
    summary_counts = summary.get("chain_status_counts")
    if summary_counts != dict(status_counts):
        raise AnnotationGateError("Audit summary status counts disagree with chains.jsonl.")
    if dict(status_counts) != dict(trust_root.chain_status_counts):
        raise AnnotationGateError("Audit status counts disagree with the canonical trust root.")
    pass_count = status_counts.get("pass", 0)
    if summary.get("proven_correct_chain_count") != pass_count:
        raise AnnotationGateError("Audit summary proven-correct count is inconsistent.")
    confirmed_issues = status_counts.get("fail", 0) + status_counts.get(
        "semantic_ambiguity", 0
    )
    if summary.get("confirmed_annotation_issue_chain_count") != confirmed_issues:
        raise AnnotationGateError("Audit summary confirmed-issue count is inconsistent.")


def _validate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], tuple[str, ...]]:
    by_source: dict[int, dict[str, Any]] = {}
    task_ids: list[str] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AnnotationGateError(f"Raw row {position} is not an object.")
        source_id = row.get("id")
        task_id = row.get("category")
        questions = row.get("questions")
        answers = row.get("answers")
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            raise AnnotationGateError(f"Raw row {position} has an invalid source id.")
        if source_id != position:
            raise AnnotationGateError(
                f"Raw source id {source_id} is not aligned to row {position}."
            )
        if not isinstance(task_id, str) or not task_id.strip():
            raise AnnotationGateError(f"Raw row {position} has an invalid task ID.")
        if not isinstance(questions, list) or not isinstance(answers, list):
            raise AnnotationGateError(f"Raw task {task_id!r} lacks aligned sessions.")
        if len(questions) != 6 or len(answers) != 6:
            raise AnnotationGateError(f"Raw task {task_id!r} is not a six-step chain.")
        target_asins: list[str] = []
        for answer in answers:
            if not isinstance(answer, dict):
                raise AnnotationGateError(f"Raw task {task_id!r} has a malformed answer.")
            asin = answer.get("target_asin")
            if not isinstance(asin, str):
                raise AnnotationGateError(f"Raw task {task_id!r} has a missing target ASIN.")
            normalized = asin.upper()
            if not _ASIN_PATTERN.fullmatch(normalized):
                raise AnnotationGateError(f"Raw task {task_id!r} has an invalid target ASIN.")
            target_asins.append(normalized)
        by_source[source_id] = {
            "task_id": task_id,
            "target_asins": tuple(target_asins),
        }
        task_ids.append(task_id)
    normalized_task_ids = _normalize_task_ids(
        task_ids, label="raw task IDs", allow_empty=False
    )
    return by_source, normalized_task_ids


def _validate_chain_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_by_source: Mapping[int, Mapping[str, Any]],
    raw_order: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_task: dict[str, dict[str, Any]] = {}
    seen_sources: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AnnotationGateError(f"Audit chain row {row_number} is not an object.")
        source_id = row.get("source_id")
        task_id = row.get("category")
        verdict = row.get("status")
        target_asins = row.get("target_asins")
        step_statuses = row.get("step_statuses")
        step_check_statuses = row.get("step_check_statuses")
        budget_status = row.get("budget_status")
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            raise AnnotationGateError(f"Audit chain row {row_number} has invalid source_id.")
        if source_id in seen_sources:
            raise AnnotationGateError(f"Duplicate audit source_id {source_id}.")
        seen_sources.add(source_id)
        raw = raw_by_source.get(source_id)
        if raw is None:
            raise AnnotationGateError(f"Audit source_id {source_id} is absent from raw data.")
        if task_id != raw["task_id"]:
            raise AnnotationGateError(
                f"Audit task ID mismatch for source {source_id}: {task_id!r}."
            )
        if verdict not in VERDICTS:
            raise AnnotationGateError(f"Invalid whole-chain verdict {verdict!r}.")
        if not isinstance(step_statuses, list) or len(step_statuses) != 6:
            raise AnnotationGateError(f"Audit task {task_id!r} lacks six step verdicts.")
        if any(step_status not in VERDICTS for step_status in step_statuses):
            raise AnnotationGateError(f"Audit task {task_id!r} has an invalid step verdict.")
        if not isinstance(step_check_statuses, list) or len(step_check_statuses) != 6:
            raise AnnotationGateError(
                f"Audit task {task_id!r} lacks six detailed step-check verdicts."
            )
        derived_step_statuses: list[str] = []
        detailed_budget_statuses: list[str] = []
        for check_index, checks in enumerate(step_check_statuses, start=1):
            if not isinstance(checks, dict) or set(checks) != STEP_CHECKS:
                raise AnnotationGateError(
                    f"Audit task {task_id!r} step {check_index} has incomplete checks."
                )
            if any(status not in VERDICTS for status in checks.values()):
                raise AnnotationGateError(
                    f"Audit task {task_id!r} step {check_index} has an invalid check."
                )
            detailed_budget = checks["bundle_budget"]
            if detailed_budget not in {"pass", "fail"}:
                raise AnnotationGateError(
                    f"Audit task {task_id!r} step {check_index} has invalid budget evidence."
                )
            detailed_budget_statuses.append(detailed_budget)
            derived_step_statuses.append(_derive_step_verdict(checks))
        if tuple(step_statuses) != tuple(derived_step_statuses):
            raise AnnotationGateError(
                f"Audit task {task_id!r} step verdicts contradict detailed checks."
            )
        if budget_status not in {"pass", "fail"}:
            raise AnnotationGateError(f"Audit task {task_id!r} has an invalid budget verdict.")
        if set(detailed_budget_statuses) != {budget_status}:
            raise AnnotationGateError(
                f"Audit task {task_id!r} budget verdict contradicts detailed checks."
            )
        derived_verdict = _derive_chain_verdict(derived_step_statuses, budget_status)
        if verdict != derived_verdict:
            raise AnnotationGateError(
                f"Audit task {task_id!r} whole-chain verdict is inconsistent: "
                f"declared {verdict!r}, derived {derived_verdict!r}."
            )
        if not isinstance(target_asins, list) or tuple(target_asins) != raw["target_asins"]:
            raise AnnotationGateError(
                f"Audit target ASINs do not match raw task {task_id!r}."
            )
        by_task[task_id] = {
            "task_id": task_id,
            "source_id": source_id,
            "verdict": verdict,
            "target_asins_sha256": hash_target_asins(target_asins),
        }
    if seen_sources != set(raw_by_source):
        missing = sorted(set(raw_by_source) - seen_sources)
        raise AnnotationGateError(
            f"Canonical audit is missing {len(missing)} raw whole chains."
        )
    ordered = [by_task[task_id] for task_id in raw_order]
    return ordered, {task_id: by_task[task_id]["verdict"] for task_id in raw_order}


def _validate_manual_evidence(payload: Mapping[str, Any]) -> None:
    mappings = payload.get("mappings") if isinstance(payload, dict) else None
    if mappings != []:
        raise AnnotationGateError(
            "Manual evidence must contain an empty mappings list; it cannot create passes."
        )


def _derive_chain_verdict(step_statuses: Sequence[str], budget_status: str) -> str:
    if budget_status == "fail" or "fail" in step_statuses:
        return "fail"
    if "semantic_ambiguity" in step_statuses:
        return "semantic_ambiguity"
    if all(status == "pass" for status in step_statuses):
        return "pass"
    return "unknown"


def _derive_step_verdict(checks: Mapping[str, str]) -> str:
    statuses = tuple(checks.values())
    if "fail" in statuses:
        return "fail"
    if "semantic_ambiguity" in statuses:
        return "semantic_ambiguity"
    if all(status == "pass" for status in statuses):
        return "pass"
    return "unknown"


def _parse_webshop_price_bounds(value: Any, row_number: int) -> list[float]:
    if value is None or not value:
        return []
    if not isinstance(value, str):
        raise AnnotationGateError(
            f"items_shuffle row {row_number} has non-string pricing."
        )
    parsed: list[float] = []
    try:
        for fragment in value.split("$")[1:]:
            numeric = re.sub(r"[^\d.]", "", fragment)
            parsed.append(float(Decimal(numeric)))
    except (InvalidOperation, ValueError) as exc:
        raise AnnotationGateError(
            f"items_shuffle row {row_number} has malformed pricing {value!r}."
        ) from exc
    return parsed[:2]


def _price_to_cents(value: float) -> int:
    decimal_value = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(decimal_value * 100)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationGateError(f"Cannot decode {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise AnnotationGateError(f"{label} must be a JSON object.")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AnnotationGateError(f"Cannot read {label}: {path}") from exc
    if not lines:
        raise AnnotationGateError(f"{label} is empty.")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise AnnotationGateError(f"{label} contains blank row {line_number}.")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnnotationGateError(
                f"Cannot decode {label} row {line_number}."
            ) from exc
        if not isinstance(row, dict):
            raise AnnotationGateError(f"{label} row {line_number} is not an object.")
        rows.append(row)
    return rows


def _normalize_task_ids(
    values: Iterable[str], *, label: str, allow_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AnnotationGateError(f"{label} must be an iterable of strings.")
    normalized: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(character in value for character in "\x00\r\n")
        ):
            raise AnnotationGateError(f"{label} contain an invalid value: {value!r}.")
        normalized.append(value)
    if not normalized and not allow_empty:
        raise AnnotationGateError(f"{label} must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise AnnotationGateError(f"{label} contain duplicates.")
    return tuple(normalized)


def _allowed_verdicts(mode: str) -> frozenset[str]:
    try:
        return ALLOWED_VERDICTS[mode]
    except KeyError as exc:
        raise AnnotationGateError(
            f"Unsupported annotation gate mode {mode!r}; expected provisional or strict."
        ) from exc


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise AnnotationGateError(
            "run_id must be 1-128 characters from letters, digits, '.', '_' or '-'."
        )


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise AnnotationGateError(f"{label} must be a lowercase SHA256 digest.")


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise AnnotationGateError(f"{label} is not a file: {resolved}")
    return resolved


def _nested(payload: Mapping[str, Any], first: str, second: str) -> Any:
    inner = payload.get(first)
    return inner.get(second) if isinstance(inner, dict) else None


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnnotationGateError(
            f"Cannot inspect MemoryArena Git source at {repo}."
        ) from exc


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
