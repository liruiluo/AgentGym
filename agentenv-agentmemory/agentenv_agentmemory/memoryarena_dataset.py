from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Collection, Iterator


EXPECTED_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
EXPECTED_RAW_DATASET_SHA256 = "4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef"
EXPECTED_DOMAIN_DATA_SHA256 = "2576aa9637ab6691c14f26e5f0b022b3a16c325a312ebc856c271f8e641f2afc"
EXPECTED_BUNDLE_COUNT = 150
EXPECTED_SESSIONS_PER_BUNDLE = 6

SPLITS = ("train", "dev", "test")
SPLIT_STRATEGY = "source_position_mod10_8_1_1_v1"
ACTION_SURFACE_VERSION = "memoryarena_webshop_native_v1"

_ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$")
_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TOTAL_BUDGET_PATTERN = re.compile(
    r"\*\*Total Budget:\*\*\s*All items combined must not exceed\s*"
    r"\$\s*(?P<amount>[0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*\."
)
_PRODUCT_MARKER_PATTERN = re.compile(r"Product\s+(?P<step>[0-9]+):", re.IGNORECASE)
_SECTION_SEPARATOR = "-" * 64
_CANDIDATE_MARKER = "**Available Options:**"
_OPTION_PATTERN = re.compile(r"^\s*-\s+(?P<title>\S(?:.*\S)?)\s*$")


class MemoryArenaDatasetError(ValueError):
    """Raised when raw MemoryArena data violates the formal runtime contract."""


@dataclass(frozen=True)
class MemoryArenaSession:
    session_index: int
    question: str
    instruction: str
    candidate_context: str
    candidate_options: tuple[str, ...]
    raw_target_asin: str
    target_asin: str
    answer_attributes: tuple[str, ...]

    @property
    def step_index(self) -> int:
        return self.session_index + 1


@dataclass(frozen=True)
class MemoryArenaBundleProvenance:
    raw_dataset_path: str
    raw_dataset_sha256: str
    memoryarena_commit: str
    domain_data_sha256: str
    split_strategy: str
    split_manifest_sha256: str
    source_position: int
    source_line_number: int
    target_asin_membership_verified: bool


@dataclass(frozen=True)
class MemoryArenaBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    source_row_id: int
    provenance: MemoryArenaBundleProvenance
    sessions: tuple[MemoryArenaSession, ...]
    category: str
    answer_attributes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class MemoryArenaDatasetProvenance:
    raw_dataset_path: str
    raw_dataset_sha256: str
    memoryarena_commit: str
    domain_data_sha256: str
    action_surface_version: str
    split_strategy: str
    split_manifest_sha256: str
    split_counts: tuple[tuple[str, int], ...]
    bundle_count: int
    sessions_per_bundle: int
    session_count: int
    target_asin_membership_verified: bool

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema": "memoryarena_raw_dataset_provenance_v1",
            "raw_dataset_path": self.raw_dataset_path,
            "raw_dataset_sha256": self.raw_dataset_sha256,
            "memoryarena_commit": self.memoryarena_commit,
            "domain_data_sha256": self.domain_data_sha256,
            "action_surface_version": self.action_surface_version,
            "split_strategy": self.split_strategy,
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_counts": dict(self.split_counts),
            "bundle_count": self.bundle_count,
            "sessions_per_bundle": self.sessions_per_bundle,
            "session_count": self.session_count,
            "target_asin_membership_verified": self.target_asin_membership_verified,
        }


@dataclass(frozen=True)
class MemoryArenaDataset:
    bundles: tuple[MemoryArenaBundle, ...]
    provenance: MemoryArenaDatasetProvenance

    def __iter__(self) -> Iterator[MemoryArenaBundle]:
        return iter(self.bundles)

    def __len__(self) -> int:
        return len(self.bundles)

    def for_split(self, split: str) -> tuple[MemoryArenaBundle, ...]:
        if split not in SPLITS:
            raise MemoryArenaDatasetError(
                f"Unsupported split {split!r}; expected one of {SPLITS}."
            )
        return tuple(bundle for bundle in self.bundles if bundle.split == split)

    def get(self, task_id: str) -> MemoryArenaBundle:
        matches = [bundle for bundle in self.bundles if bundle.task_id == task_id]
        if len(matches) != 1:
            raise KeyError(task_id)
        return matches[0]


@dataclass(frozen=True)
class _ParsedBundle:
    task_id: str
    category: str
    source_row_id: int
    source_position: int
    split: str
    budget_cents: int
    sessions: tuple[MemoryArenaSession, ...]


def load_memoryarena_bundles(
    data_path: str | Path,
    *,
    frozen_product_asins: Collection[str],
    expected_raw_sha256: str = EXPECTED_RAW_DATASET_SHA256,
    expected_bundle_count: int = EXPECTED_BUNDLE_COUNT,
    expected_sessions_per_bundle: int = EXPECTED_SESSIONS_PER_BUNDLE,
    memoryarena_commit: str = EXPECTED_MEMORYARENA_COMMIT,
    domain_data_sha256: str = EXPECTED_DOMAIN_DATA_SHA256,
) -> tuple[MemoryArenaBundle, ...]:
    """Load formal raw bundles and return the immutable runtime task tuple."""

    return load_memoryarena_dataset(
        data_path,
        frozen_product_asins=frozen_product_asins,
        expected_raw_sha256=expected_raw_sha256,
        expected_bundle_count=expected_bundle_count,
        expected_sessions_per_bundle=expected_sessions_per_bundle,
        memoryarena_commit=memoryarena_commit,
        domain_data_sha256=domain_data_sha256,
    ).bundles


def load_memoryarena_dataset(
    data_path: str | Path,
    *,
    frozen_product_asins: Collection[str],
    expected_raw_sha256: str = EXPECTED_RAW_DATASET_SHA256,
    expected_bundle_count: int = EXPECTED_BUNDLE_COUNT,
    expected_sessions_per_bundle: int = EXPECTED_SESSIONS_PER_BUNDLE,
    memoryarena_commit: str = EXPECTED_MEMORYARENA_COMMIT,
    domain_data_sha256: str = EXPECTED_DOMAIN_DATA_SHA256,
) -> MemoryArenaDataset:
    """Load and validate the raw six-session MemoryArena shopping dataset.

    The catalog membership argument is deliberately mandatory. Formal callers
    must supply the ASIN keys from the frozen native product dictionary; raw
    labels alone are not evidence that a target exists on the action surface.
    """

    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise MemoryArenaDatasetError(f"Raw MemoryArena dataset is not a file: {path}")
    _validate_contract_arguments(
        expected_raw_sha256=expected_raw_sha256,
        expected_bundle_count=expected_bundle_count,
        expected_sessions_per_bundle=expected_sessions_per_bundle,
        memoryarena_commit=memoryarena_commit,
        domain_data_sha256=domain_data_sha256,
    )

    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_raw_sha256:
        raise MemoryArenaDatasetError(
            "Raw MemoryArena dataset SHA256 mismatch: "
            f"expected {expected_raw_sha256}, observed {actual_sha256}."
        )
    records = _decode_jsonl(payload, path)
    if len(records) != expected_bundle_count:
        raise MemoryArenaDatasetError(
            f"Raw MemoryArena dataset must contain {expected_bundle_count} bundles; "
            f"observed {len(records)}."
        )

    catalog_asins = _normalize_catalog_asins(frozen_product_asins)
    parsed = tuple(
        _parse_bundle(
            record,
            source_position=source_position,
            expected_sessions_per_bundle=expected_sessions_per_bundle,
            frozen_product_asins=catalog_asins,
        )
        for source_position, record in enumerate(records)
    )
    _validate_unique_source_identity(parsed)

    split_task_ids = {
        split: tuple(bundle.task_id for bundle in parsed if bundle.split == split)
        for split in SPLITS
    }
    _validate_split_coverage(parsed, split_task_ids)
    split_manifest_sha256 = _hash_split_manifest(split_task_ids)

    bundles = tuple(
        MemoryArenaBundle(
            task_id=bundle.task_id,
            questions=tuple(session.question for session in bundle.sessions),
            target_asins=tuple(session.target_asin for session in bundle.sessions),
            budget_cents=bundle.budget_cents,
            split=bundle.split,
            source_row_id=bundle.source_row_id,
            provenance=MemoryArenaBundleProvenance(
                raw_dataset_path=str(path),
                raw_dataset_sha256=actual_sha256,
                memoryarena_commit=memoryarena_commit,
                domain_data_sha256=domain_data_sha256,
                split_strategy=SPLIT_STRATEGY,
                split_manifest_sha256=split_manifest_sha256,
                source_position=bundle.source_position,
                source_line_number=bundle.source_position + 1,
                target_asin_membership_verified=True,
            ),
            sessions=bundle.sessions,
            category=bundle.category,
            answer_attributes=tuple(
                session.answer_attributes for session in bundle.sessions
            ),
        )
        for bundle in parsed
    )
    split_counts = tuple((split, len(split_task_ids[split])) for split in SPLITS)
    provenance = MemoryArenaDatasetProvenance(
        raw_dataset_path=str(path),
        raw_dataset_sha256=actual_sha256,
        memoryarena_commit=memoryarena_commit,
        domain_data_sha256=domain_data_sha256,
        action_surface_version=ACTION_SURFACE_VERSION,
        split_strategy=SPLIT_STRATEGY,
        split_manifest_sha256=split_manifest_sha256,
        split_counts=split_counts,
        bundle_count=len(bundles),
        sessions_per_bundle=expected_sessions_per_bundle,
        session_count=len(bundles) * expected_sessions_per_bundle,
        target_asin_membership_verified=True,
    )
    return MemoryArenaDataset(bundles=bundles, provenance=provenance)


def parse_budget_cents(question: str) -> int:
    matches = list(_TOTAL_BUDGET_PATTERN.finditer(question))
    if len(matches) != 1:
        raise MemoryArenaDatasetError(
            "Each MemoryArena question must contain exactly one canonical Total Budget value; "
            f"observed {len(matches)}."
        )
    amount_text = matches[0].group("amount").replace(",", "")
    try:
        amount = Decimal(amount_text)
    except InvalidOperation as exc:
        raise MemoryArenaDatasetError(
            f"Invalid Total Budget amount {amount_text!r}."
        ) from exc
    cents = amount * 100
    if amount <= 0 or cents != cents.to_integral_value():
        raise MemoryArenaDatasetError(
            f"Total Budget must be positive and cent-exact, got {amount_text!r}."
        )
    return int(cents)


def assign_split(source_position: int) -> str:
    if source_position < 0:
        raise MemoryArenaDatasetError(
            f"source_position must be non-negative, got {source_position}."
        )
    bucket = source_position % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "dev"
    return "test"


def _validate_contract_arguments(
    *,
    expected_raw_sha256: str,
    expected_bundle_count: int,
    expected_sessions_per_bundle: int,
    memoryarena_commit: str,
    domain_data_sha256: str,
) -> None:
    if not _HEX_SHA256_PATTERN.fullmatch(expected_raw_sha256):
        raise MemoryArenaDatasetError(
            f"expected_raw_sha256 must be a lowercase SHA256, got {expected_raw_sha256!r}."
        )
    if not _HEX_SHA256_PATTERN.fullmatch(domain_data_sha256):
        raise MemoryArenaDatasetError(
            f"domain_data_sha256 must be a lowercase SHA256, got {domain_data_sha256!r}."
        )
    if not _GIT_COMMIT_PATTERN.fullmatch(memoryarena_commit):
        raise MemoryArenaDatasetError(
            f"memoryarena_commit must be a full lowercase git commit, got {memoryarena_commit!r}."
        )
    if expected_bundle_count <= 0:
        raise MemoryArenaDatasetError("expected_bundle_count must be positive.")
    if expected_sessions_per_bundle <= 0:
        raise MemoryArenaDatasetError("expected_sessions_per_bundle must be positive.")


def _decode_jsonl(payload: bytes, path: Path) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemoryArenaDatasetError(f"Raw dataset is not valid UTF-8: {path}") from exc
    lines = text.splitlines()
    if not lines:
        raise MemoryArenaDatasetError(f"Raw dataset is empty: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise MemoryArenaDatasetError(
                f"Raw dataset contains a blank JSONL row at line {line_number}."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MemoryArenaDatasetError(
                f"Invalid JSON at raw dataset line {line_number}: {exc.msg}."
            ) from exc
        if not isinstance(record, dict):
            raise MemoryArenaDatasetError(
                f"Raw dataset line {line_number} must decode to an object."
            )
        records.append(record)
    return records


def _normalize_catalog_asins(values: Collection[str]) -> Collection[str]:
    if values is None:
        raise MemoryArenaDatasetError(
            "frozen_product_asins is required for formal target membership verification."
        )
    if not values:
        raise MemoryArenaDatasetError(
            "frozen_product_asins must contain the native product dictionary keys."
        )

    # Native WebShop already exposes an uppercase ASIN-keyed mapping. Reuse its
    # O(1) key view instead of copying roughly 1.1M keys during server startup.
    if isinstance(values, Mapping):
        return values.keys()
    if isinstance(values, AbstractSet):
        return values

    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise MemoryArenaDatasetError(
                "Every frozen product ASIN must be a string; "
                f"got {type(value).__name__}."
            )
        asin = value.strip().upper()
        if not _ASIN_PATTERN.fullmatch(asin):
            raise MemoryArenaDatasetError(
                f"Frozen product dictionary contains invalid ASIN {value!r}."
            )
        normalized.add(asin)
    return frozenset(normalized)


def _parse_bundle(
    record: dict[str, Any],
    *,
    source_position: int,
    expected_sessions_per_bundle: int,
    frozen_product_asins: Collection[str],
) -> _ParsedBundle:
    context = f"MemoryArena row at source position {source_position}"
    for key in ("id", "questions", "answers", "category"):
        if key not in record:
            raise MemoryArenaDatasetError(f"{context} is missing required key {key!r}.")

    source_row_id = record["id"]
    if isinstance(source_row_id, bool) or not isinstance(source_row_id, int):
        raise MemoryArenaDatasetError(f"{context} id must be an integer.")
    if source_row_id != source_position:
        raise MemoryArenaDatasetError(
            f"{context} is misordered: id={source_row_id}, expected {source_position}."
        )

    category = record["category"]
    if not isinstance(category, str) or not category or category != category.strip():
        raise MemoryArenaDatasetError(
            f"{context} category must be a non-empty canonical string."
        )

    questions = record["questions"]
    answers = record["answers"]
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise MemoryArenaDatasetError(
            f"{context} questions and answers must both be lists."
        )
    if (
        len(questions) != expected_sessions_per_bundle
        or len(answers) != expected_sessions_per_bundle
    ):
        raise MemoryArenaDatasetError(
            f"{context} must contain exactly {expected_sessions_per_bundle} aligned questions "
            f"and answers; observed {len(questions)} and {len(answers)}."
        )

    sessions = tuple(
        _parse_session(
            question,
            answer,
            source_row_id=source_row_id,
            session_index=session_index,
            frozen_product_asins=frozen_product_asins,
        )
        for session_index, (question, answer) in enumerate(zip(questions, answers))
    )
    budgets = tuple(parse_budget_cents(session.question) for session in sessions)
    if len(set(budgets)) != 1:
        raise MemoryArenaDatasetError(
            f"MemoryArena row {source_row_id} has inconsistent six-session budgets: {budgets}."
        )
    return _ParsedBundle(
        task_id=category,
        category=category,
        source_row_id=source_row_id,
        source_position=source_position,
        split=assign_split(source_position),
        budget_cents=budgets[0],
        sessions=sessions,
    )


def _parse_session(
    question: Any,
    answer: Any,
    *,
    source_row_id: int,
    session_index: int,
    frozen_product_asins: Collection[str],
) -> MemoryArenaSession:
    context = f"MemoryArena row {source_row_id} session {session_index + 1}"
    if not isinstance(question, str) or not question.strip():
        raise MemoryArenaDatasetError(f"{context} question must be a non-empty string.")
    if question.count(_SECTION_SEPARATOR) != 1:
        raise MemoryArenaDatasetError(
            f"{context} must contain exactly one canonical product-section separator."
        )
    section = question.split(_SECTION_SEPARATOR, maxsplit=1)[1]
    product_markers = list(_PRODUCT_MARKER_PATTERN.finditer(section))
    expected_step = session_index + 1
    if len(product_markers) != 1 or int(product_markers[0].group("step")) != expected_step:
        observed = tuple(int(match.group("step")) for match in product_markers)
        raise MemoryArenaDatasetError(
            f"{context} product marker is misaligned: "
            f"expected ({expected_step},), observed {observed}."
        )
    candidate_count = section.count(_CANDIDATE_MARKER)
    if candidate_count != 1:
        raise MemoryArenaDatasetError(
            f"{context} must contain exactly one {_CANDIDATE_MARKER!r} marker; "
            f"observed {candidate_count}."
        )
    candidate_offset = section.index(_CANDIDATE_MARKER)
    instruction = section[product_markers[0].start() : candidate_offset]
    candidate_context = section[candidate_offset:]
    candidate_options = _extract_candidate_options(candidate_context, context=context)

    if not isinstance(answer, dict):
        raise MemoryArenaDatasetError(f"{context} answer must be an object.")
    for key in ("target_asin", "attributes"):
        if key not in answer:
            raise MemoryArenaDatasetError(f"{context} answer is missing required key {key!r}.")
    raw_target_asin = answer["target_asin"]
    if not isinstance(raw_target_asin, str):
        raise MemoryArenaDatasetError(f"{context} target_asin must be a string.")
    target_asin = raw_target_asin.strip().upper()
    if not _ASIN_PATTERN.fullmatch(target_asin):
        raise MemoryArenaDatasetError(
            f"{context} has invalid target ASIN {raw_target_asin!r}."
        )
    if target_asin not in frozen_product_asins:
        raise MemoryArenaDatasetError(
            f"{context} target ASIN {target_asin} is absent from the frozen "
            "native product dictionary."
        )
    attributes = answer["attributes"]
    if not isinstance(attributes, list) or not all(isinstance(value, str) for value in attributes):
        raise MemoryArenaDatasetError(
            f"{context} answer attributes must be a list of strings."
        )

    return MemoryArenaSession(
        session_index=session_index,
        question=question,
        instruction=instruction,
        candidate_context=candidate_context,
        candidate_options=candidate_options,
        raw_target_asin=raw_target_asin,
        target_asin=target_asin,
        answer_attributes=tuple(attributes),
    )


def _extract_candidate_options(candidate_context: str, *, context: str) -> tuple[str, ...]:
    tail = candidate_context[len(_CANDIDATE_MARKER) :]
    options: list[str] = []
    for line in tail.splitlines():
        if not line.strip() and not options:
            continue
        match = _OPTION_PATTERN.fullmatch(line)
        if match:
            options.append(match.group("title"))
            continue
        if options:
            if line.strip():
                raise MemoryArenaDatasetError(
                    f"{context} candidate context contains trailing non-option text."
                )
            continue
        if line.strip():
            raise MemoryArenaDatasetError(
                f"{context} candidate context must begin with '- ' option lines."
            )
    if not options:
        raise MemoryArenaDatasetError(f"{context} has no candidate options.")
    if len(options) != len(set(options)):
        raise MemoryArenaDatasetError(f"{context} contains duplicate candidate option text.")
    return tuple(options)


def _validate_unique_source_identity(parsed: tuple[_ParsedBundle, ...]) -> None:
    task_ids = [bundle.task_id for bundle in parsed]
    source_row_ids = [bundle.source_row_id for bundle in parsed]
    if len(task_ids) != len(set(task_ids)):
        raise MemoryArenaDatasetError("MemoryArena category/task IDs must be unique.")
    if len(source_row_ids) != len(set(source_row_ids)):
        raise MemoryArenaDatasetError("MemoryArena source row IDs must be unique.")


def _validate_split_coverage(
    parsed: tuple[_ParsedBundle, ...],
    split_task_ids: dict[str, tuple[str, ...]],
) -> None:
    expected_task_ids = tuple(bundle.task_id for bundle in parsed)
    flattened = tuple(task_id for split in SPLITS for task_id in split_task_ids[split])
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected_task_ids):
        raise MemoryArenaDatasetError(
            "Deterministic split manifest must be disjoint and cover every task exactly once."
        )
    expected_counts = {
        split: sum(assign_split(position) == split for position in range(len(parsed)))
        for split in SPLITS
    }
    observed_counts = {split: len(split_task_ids[split]) for split in SPLITS}
    if observed_counts != expected_counts:
        raise MemoryArenaDatasetError(
            "Deterministic split counts are wrong: "
            f"expected {expected_counts}, observed {observed_counts}."
        )


def _hash_split_manifest(split_task_ids: dict[str, tuple[str, ...]]) -> str:
    manifest = {
        "schema": "memoryarena_task_split_v1",
        "strategy": SPLIT_STRATEGY,
        "splits": {split: list(split_task_ids[split]) for split in SPLITS},
    }
    payload = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACTION_SURFACE_VERSION",
    "EXPECTED_BUNDLE_COUNT",
    "EXPECTED_DOMAIN_DATA_SHA256",
    "EXPECTED_MEMORYARENA_COMMIT",
    "EXPECTED_RAW_DATASET_SHA256",
    "EXPECTED_SESSIONS_PER_BUNDLE",
    "MemoryArenaBundle",
    "MemoryArenaBundleProvenance",
    "MemoryArenaDataset",
    "MemoryArenaDatasetError",
    "MemoryArenaDatasetProvenance",
    "MemoryArenaSession",
    "SPLITS",
    "SPLIT_STRATEGY",
    "assign_split",
    "load_memoryarena_bundles",
    "load_memoryarena_dataset",
    "parse_budget_cents",
]
