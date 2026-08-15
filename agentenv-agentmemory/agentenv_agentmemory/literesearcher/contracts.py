from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


LITERESEARCHER_UPSTREAM_COMMIT = "779e7d5f6a043d4100149ba0992a39507f69a974"
LITERESEARCHER_DATA_REVISION = "fff6b0cfef718859543a16f542ea248d30d1ac34"
LITERESEARCHER_DATASET = "simplex-ai-inc/LiteResearcher-Data"
LITERESEARCHER_STAGE = "stage1"
LITERESEARCHER_STAGE1_ROWS = 10_398
LITERESEARCHER_STAGE1_TURN_LIMIT = 40
LITERESEARCHER_STAGE1_CONTEXT = 32_768


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_text_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(_require_text(item, field) for item in value)


@dataclass(frozen=True)
class LiteResearcherTask:
    """One server-private Stage-1 row plus its frozen source snapshot.

    Source provenance and ``targets`` are verifier inputs.  They are
    intentionally never returned by the policy-facing backend methods.
    """

    index: int
    question: str
    targets: tuple[str, ...]
    mask_url: str
    public_url: str
    page_title: str
    page_text: str
    resolved_url: str
    retrieved_at: str
    content_sha256: str
    extraction_method: str
    license_note: str
    evidence_anchors: tuple[str, ...]
    source_normalization_reason: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("task index must be a non-negative integer")
        _require_text(self.question, "task question")
        if not self.targets or any(
            not isinstance(item, str) or not item.strip() for item in self.targets
        ):
            raise ValueError("task targets must contain non-empty strings")
        _require_text(self.mask_url, "task mask_url")
        _require_text(self.public_url, "task public_url")
        _require_text(self.page_title, "task page_title")
        _require_text(self.page_text, "task page_text")
        _require_text(self.resolved_url, "task resolved_url")
        _require_text(self.retrieved_at, "task retrieved_at")
        _require_text(self.extraction_method, "task extraction_method")
        _require_text(self.license_note, "task license_note")
        _require_text(
            self.source_normalization_reason, "task source_normalization_reason"
        )
        if self.mask_url == self.public_url:
            raise ValueError("server-private mask_url must differ from public_url")
        if self.resolved_url == self.public_url:
            raise ValueError("server-private resolved_url must differ from public_url")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise ValueError("task content_sha256 must be a lowercase SHA256 digest")
        actual_digest = hashlib.sha256(self.page_text.encode("utf-8")).hexdigest()
        if actual_digest != self.content_sha256:
            raise ValueError("task page_text does not match content_sha256")
        try:
            timestamp = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("task retrieved_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("task retrieved_at must include a timezone")
        if not self.evidence_anchors or any(
            not isinstance(item, str) or not item.strip()
            for item in self.evidence_anchors
        ):
            raise ValueError("task evidence_anchors must contain non-empty strings")
        if any(not _normalized_evidence_text(item) for item in self.evidence_anchors):
            raise ValueError("task evidence anchors must contain searchable text")
        normalized_page = _normalized_evidence_text(self.page_text)
        if any(
            _normalized_evidence_text(anchor) not in normalized_page
            for anchor in self.evidence_anchors
        ):
            raise ValueError("task evidence anchor is absent from page_text")
        forbidden_fixture_markers = (
            "frozen source excerpt",
            "the source evidence supports the requested answer",
        )
        page_casefold = self.page_text.casefold()
        if any(marker in page_casefold for marker in forbidden_fixture_markers):
            raise ValueError("gold-synthesized placeholder page_text is forbidden")
        if self.mask_url in self.page_text or self.resolved_url in self.page_text:
            raise ValueError("source URLs must not be embedded in policy-facing page_text")

    @property
    def task_id(self) -> str:
        return f"stage1:{self.index:05d}"

    @property
    def private_source_sha256(self) -> str:
        return hashlib.sha256(self.mask_url.encode("utf-8")).hexdigest()

    def server_record(self) -> dict[str, Any]:
        """Return the full verifier record; never pass this to the model."""

        return {
            "index": self.index,
            "question": self.question,
            "targets": list(self.targets),
            "mask_url": self.mask_url,
            "public_url": self.public_url,
            "page_title": self.page_title,
            "page_text": self.page_text,
            "resolved_url": self.resolved_url,
            "retrieved_at": self.retrieved_at,
            "content_sha256": self.content_sha256,
            "extraction_method": self.extraction_method,
            "license_note": self.license_note,
            "evidence_anchors": list(self.evidence_anchors),
            "source_normalization_reason": self.source_normalization_reason,
        }

    def public_record(self) -> dict[str, Any]:
        """Return metadata safe for policy-facing service metadata."""

        return {
            "task_id": self.task_id,
            "index": self.index,
            "public_url": self.public_url,
            "private_source_sha256": self.private_source_sha256,
        }


@dataclass(frozen=True)
class LiteResearcherCoverage:
    """Frozen train/held-out split for the initial RL intake gate."""

    train: tuple[LiteResearcherTask, ...]
    heldout: tuple[LiteResearcherTask, ...]
    manifest_sha256: str
    semantic_audit_sha256: str

    def __post_init__(self) -> None:
        if len(self.train) != 64:
            raise ValueError("LiteResearcher intake requires exactly 64 train rows")
        if len(self.heldout) != 8:
            raise ValueError("LiteResearcher intake requires exactly 8 held-out rows")
        train_indices = [task.index for task in self.train]
        heldout_indices = [task.index for task in self.heldout]
        if len(set(train_indices)) != len(train_indices):
            raise ValueError("train task indices must be unique")
        if len(set(heldout_indices)) != len(heldout_indices):
            raise ValueError("held-out task indices must be unique")
        if set(train_indices) & set(heldout_indices):
            raise ValueError("train and held-out indices must be disjoint")
        public_urls = [task.public_url for task in (*self.train, *self.heldout)]
        if len(set(public_urls)) != len(public_urls):
            raise ValueError("coverage public URLs must be unique")
        if not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a 64-character digest")
        if (
            not isinstance(self.semantic_audit_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.semantic_audit_sha256)
        ):
            raise ValueError("semantic_audit_sha256 must be a lowercase SHA256 digest")

    @property
    def task_count(self) -> int:
        return len(self.train)

    @property
    def heldout_count(self) -> int:
        return len(self.heldout)

    def tasks_for_split(self, split: str) -> tuple[LiteResearcherTask, ...]:
        if split == "train":
            return self.train
        if split in {"test", "heldout"}:
            return self.heldout
        raise ValueError("LiteResearcher split must be train, test, or heldout")

    def task(self, data_idx: int, *, split: str = "train") -> LiteResearcherTask:
        if (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or data_idx < 0
        ):
            raise ValueError("data_idx must be a non-negative integer")
        tasks = self.tasks_for_split(split)
        try:
            return tasks[data_idx]
        except IndexError as exc:
            raise IndexError(
                f"{split} data_idx out of range: {data_idx}"
            ) from exc

    def public_metadata(self) -> dict[str, Any]:
        return {
            "dataset": LITERESEARCHER_DATASET,
            "config": LITERESEARCHER_STAGE,
            "data_revision": LITERESEARCHER_DATA_REVISION,
            "upstream_commit": LITERESEARCHER_UPSTREAM_COMMIT,
            "upstream_stage1_rows": LITERESEARCHER_STAGE1_ROWS,
            "train_count": self.task_count,
            "heldout_count": self.heldout_count,
            "train_indices": [task.index for task in self.train],
            "heldout_indices": [task.index for task in self.heldout],
            "manifest_sha256": self.manifest_sha256,
            "semantic_audit_sha256": self.semantic_audit_sha256,
            "native_episode_contract": "search_visit_answer_single_episode_v1",
            "session_boundaries": 0,
        }


def _canonical_manifest_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _normalized_evidence_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


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


def _load_semantic_audit(
    manifest_path: Path,
    reference: Any,
    tasks: tuple[LiteResearcherTask, ...],
) -> str:
    if not isinstance(reference, Mapping):
        raise ValueError("coverage manifest requires semantic_audit metadata")
    filename = _require_text(reference.get("file"), "semantic audit file")
    if Path(filename).name != filename:
        raise ValueError("semantic audit file must be a sibling filename")
    expected_digest = _require_text(reference.get("sha256"), "semantic audit SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("semantic audit SHA256 must be a lowercase digest")
    audit_path = manifest_path.parent / filename
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("LiteResearcher semantic audit must be an object")
    if audit.get("schema") != "agentmemory_literesearcher_semantic_audit_v1":
        raise ValueError("unsupported LiteResearcher semantic audit schema")
    if audit.get("data_revision") != LITERESEARCHER_DATA_REVISION:
        raise ValueError("semantic audit has the wrong HF data revision")
    unsigned = dict(audit)
    unsigned.pop("audit_sha256", None)
    actual_digest = hashlib.sha256(_canonical_manifest_payload(unsigned)).hexdigest()
    if audit.get("audit_sha256") != actual_digest or expected_digest != actual_digest:
        raise ValueError("semantic audit SHA256 does not match its contents")

    approved = audit.get("approved")
    rejected = audit.get("rejected")
    if not isinstance(approved, list) or len(approved) != 72:
        raise ValueError("semantic audit requires exactly 72 approved rows")
    if not isinstance(rejected, list) or not rejected:
        raise ValueError("semantic audit requires explicit rejected-row evidence")
    if reference.get("approved_count") != len(approved):
        raise ValueError("semantic audit approved_count mismatch")
    if reference.get("rejected_count") != len(rejected):
        raise ValueError("semantic audit rejected_count mismatch")
    if reference.get("source_backed_ratio") != 1.0:
        raise ValueError("LiteResearcher intake must be 100% source-backed")

    approved_by_index: dict[int, Mapping[str, Any]] = {}
    for record in approved:
        if not isinstance(record, Mapping):
            raise ValueError("semantic audit approved row must be an object")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("semantic audit index must be a non-negative integer")
        if index in approved_by_index:
            raise ValueError("semantic audit approved indices must be unique")
        approved_by_index[index] = record
    task_indices = [task.index for task in tasks]
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("coverage task indices must be unique")
    if sorted(approved_by_index) != sorted(task_indices):
        raise ValueError("semantic audit approved indices do not match the manifest")

    for task in tasks:
        record = approved_by_index[task.index]
        if record.get("review_status") != "approved":
            raise ValueError("semantic audit row is not approved")
        if record.get("question_sha256") != _text_sha256(task.question):
            raise ValueError("semantic audit question SHA256 mismatch")
        if record.get("targets_sha256") != _targets_sha256(task.targets):
            raise ValueError("semantic audit targets SHA256 mismatch")
        if record.get("content_sha256") != task.content_sha256:
            raise ValueError("semantic audit content SHA256 mismatch")
        if record.get("resolved_url") != task.resolved_url:
            raise ValueError("semantic audit resolved_url mismatch")
        quote = _require_text(record.get("evidence_quote"), "semantic evidence quote")
        if quote not in task.page_text:
            raise ValueError("semantic evidence quote is absent from page_text")
        normalized_quote = _normalized_evidence_text(quote)
        if not any(
            _normalized_evidence_text(target) in normalized_quote
            for target in task.targets
        ):
            raise ValueError("semantic evidence quote does not contain a target alias")
    return actual_digest


def _task_from_record(raw: Mapping[str, Any]) -> LiteResearcherTask:
    if not isinstance(raw, Mapping):
        raise ValueError("coverage task must be an object")
    targets = raw.get("targets", raw.get("target"))
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, Iterable) or isinstance(targets, (bytes, dict)):
        raise ValueError("coverage task targets must be a list")
    return LiteResearcherTask(
        index=int(raw["index"]),
        question=_require_text(raw["question"], "coverage question"),
        targets=tuple(_require_text(item, "coverage target") for item in targets),
        mask_url=_require_text(raw["mask_url"], "coverage mask_url"),
        public_url=_require_text(raw["public_url"], "coverage public_url"),
        page_title=_require_text(raw["page_title"], "coverage page_title"),
        page_text=_require_text(raw["page_text"], "coverage page_text"),
        resolved_url=_require_text(raw.get("resolved_url"), "coverage resolved_url"),
        retrieved_at=_require_text(raw.get("retrieved_at"), "coverage retrieved_at"),
        content_sha256=_require_text(
            raw.get("content_sha256"), "coverage content_sha256"
        ),
        extraction_method=_require_text(
            raw.get("extraction_method"), "coverage extraction_method"
        ),
        license_note=_require_text(raw.get("license_note"), "coverage license_note"),
        evidence_anchors=_require_text_list(
            raw.get("evidence_anchors"), "coverage evidence_anchors"
        ),
        source_normalization_reason=_require_text(
            raw.get("source_normalization_reason"),
            "coverage source_normalization_reason",
        ),
    )


def load_coverage_manifest(path: str | Path) -> LiteResearcherCoverage:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LiteResearcher coverage manifest must be an object")
    if payload.get("schema") != "agentmemory_literesearcher_coverage_v3":
        raise ValueError("unsupported LiteResearcher coverage schema")
    if (
        payload.get("page_fixture_contract")
        != "source_backed_semantically_reviewed_frozen_text_v1"
    ):
        raise ValueError("coverage manifest is not a source-backed frozen corpus")
    expected_revision = payload.get("data_revision")
    if expected_revision != LITERESEARCHER_DATA_REVISION:
        raise ValueError("coverage manifest has the wrong HF data revision")
    train_raw = payload.get("train")
    heldout_raw = payload.get("heldout")
    if not isinstance(train_raw, list) or not isinstance(heldout_raw, list):
        raise ValueError("coverage manifest train/heldout fields must be lists")
    train = tuple(_task_from_record(item) for item in train_raw)
    heldout = tuple(_task_from_record(item) for item in heldout_raw)
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    digest = hashlib.sha256(_canonical_manifest_payload(unsigned)).hexdigest()
    if payload.get("manifest_sha256") != digest:
        raise ValueError("coverage manifest SHA256 does not match its contents")
    semantic_audit_sha256 = _load_semantic_audit(
        manifest_path,
        payload.get("semantic_audit"),
        (*train, *heldout),
    )
    return LiteResearcherCoverage(
        train=train,
        heldout=heldout,
        manifest_sha256=digest,
        semantic_audit_sha256=semantic_audit_sha256,
    )
