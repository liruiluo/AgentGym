from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LiteResearcherTask:
    """One server-private Stage-1 row plus its frozen page excerpt.

    ``mask_url`` and ``targets`` are verifier inputs.  They are intentionally
    never returned by the policy-facing backend methods.
    """

    index: int
    question: str
    targets: tuple[str, ...]
    mask_url: str
    public_url: str
    page_title: str
    page_text: str

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
        if self.mask_url == self.public_url:
            raise ValueError("server-private mask_url must differ from public_url")

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

    def __post_init__(self) -> None:
        if len(self.train) != 64:
            raise ValueError("LiteResearcher intake requires exactly 64 train rows")
        if not self.heldout:
            raise ValueError("LiteResearcher intake requires a non-empty held-out split")
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

    @property
    def task_count(self) -> int:
        return len(self.train)

    @property
    def heldout_count(self) -> int:
        return len(self.heldout)

    def task(self, data_idx: int) -> LiteResearcherTask:
        if isinstance(data_idx, bool) or not isinstance(data_idx, int):
            raise ValueError("data_idx must be an integer")
        try:
            return self.train[data_idx]
        except IndexError as exc:
            raise IndexError(f"train data_idx out of range: {data_idx}") from exc

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
            "native_episode_contract": "search_visit_answer_single_episode_v1",
            "session_boundaries": 0,
        }


def _canonical_manifest_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _task_from_record(raw: Mapping[str, Any]) -> LiteResearcherTask:
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
    )


def load_coverage_manifest(path: str | Path) -> LiteResearcherCoverage:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LiteResearcher coverage manifest must be an object")
    if payload.get("schema") != "agentmemory_literesearcher_coverage_v1":
        raise ValueError("unsupported LiteResearcher coverage schema")
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
    return LiteResearcherCoverage(train=train, heldout=heldout, manifest_sha256=digest)
