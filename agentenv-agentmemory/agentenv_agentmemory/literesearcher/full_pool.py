from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FULL_POOL_SCHEMA = "agentmemory_literesearcher_full_compatible_pool_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _targets_sha256(targets: tuple[str, ...]) -> str:
    raw = json.dumps(
        list(targets),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FullPoolLiteResearcherTask:
    index: int
    question: str
    targets: tuple[str, ...]
    mask_url: str
    row_identity: str
    parquet_path: str
    physical_row: int
    data_source: str
    upstream_curriculum_stage: int

    @property
    def task_id(self) -> str:
        return f"fullpool:{self.row_identity[:16]}"


class FullPoolLiteResearcherTasks:
    def __init__(
        self,
        tasks: tuple[FullPoolLiteResearcherTask, ...],
        *,
        manifest_sha256: str,
        dataset_revision: str,
        upstream_commit: str,
    ) -> None:
        if not tasks:
            raise ValueError("LiteResearcher full pool must not be empty")
        if [task.index for task in tasks] != list(range(len(tasks))):
            raise ValueError("LiteResearcher full-pool indices must be contiguous")
        self.train = tasks
        self.manifest_sha256 = manifest_sha256
        self.dataset_revision = dataset_revision
        self.upstream_commit = upstream_commit

    @property
    def task_count(self) -> int:
        return len(self.train)

    @property
    def heldout_count(self) -> int:
        return 0

    def tasks_for_split(self, split: str) -> tuple[FullPoolLiteResearcherTask, ...]:
        if split != "train":
            raise ValueError("the frozen LiteResearcher full pool currently supports train only")
        return self.train

    def task(
        self, data_idx: int, *, split: str = "train"
    ) -> FullPoolLiteResearcherTask:
        tasks = self.tasks_for_split(split)
        if isinstance(data_idx, bool) or not isinstance(data_idx, int) or data_idx < 0:
            raise ValueError("data_idx must be a non-negative integer")
        try:
            return tasks[data_idx]
        except IndexError as exc:
            raise IndexError(f"{split} data_idx out of range: {data_idx}") from exc

    def public_metadata(self) -> dict[str, Any]:
        return {
            "dataset": "simplex-ai-inc/LiteResearcher-Data",
            "data_revision": self.dataset_revision,
            "upstream_commit": self.upstream_commit,
            "distribution": "single_frozen_mixed_pool",
            "upstream_curriculum_enabled": False,
            "train_count": self.task_count,
            "heldout_count": 0,
            "manifest_sha256": self.manifest_sha256,
            "native_episode_contract": "search_visit_answer_single_episode_v1",
            "session_boundaries": 0,
        }


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _read_raw_sources(source_root: Path, reports: list[Mapping[str, Any]]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to load the LiteResearcher full pool") from exc

    sources: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        relative = str(report["parquet_relative_path"])
        path = (source_root / relative).resolve()
        if path.parent != source_root.resolve() or not path.is_file():
            raise ValueError(f"invalid LiteResearcher parquet path: {relative}")
        if _sha256(path) != report["parquet_sha256"]:
            raise ValueError(f"LiteResearcher parquet SHA256 mismatch: {relative}")
        table = pq.read_table(
            path,
            columns=["question", "data_source", "reward_model", "extra_info"],
        )
        if table.num_rows != int(report["physical_rows"]):
            raise ValueError(f"LiteResearcher parquet row count mismatch: {relative}")
        sources[relative] = table.to_pylist()
    return sources


def load_full_pool(
    manifest_path: str | Path,
    pool_rows_path: str | Path,
    source_root: str | Path,
) -> FullPoolLiteResearcherTasks:
    manifest_path = Path(manifest_path).expanduser().resolve()
    pool_rows_path = Path(pool_rows_path).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FULL_POOL_SCHEMA:
        raise ValueError("unsupported LiteResearcher full-pool manifest schema")
    artifact = _required_mapping(manifest.get("artifacts"), "artifacts").get(
        "pool_rows.jsonl"
    )
    artifact = _required_mapping(artifact, "pool_rows.jsonl artifact")
    if _sha256(pool_rows_path) != artifact.get("sha256"):
        raise ValueError("LiteResearcher pool_rows.jsonl SHA256 mismatch")
    reports = manifest.get("source_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("LiteResearcher full-pool manifest has no source reports")
    raw_sources = _read_raw_sources(source_root, reports)

    tasks: list[FullPoolLiteResearcherTask] = []
    with pool_rows_path.open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if int(row["pool_index"]) != expected_index:
                raise ValueError("LiteResearcher pool row order is not contiguous")
            relative = str(row["parquet_path"])
            physical_row = int(row["physical_row"])
            try:
                raw = raw_sources[relative][physical_row]
            except (KeyError, IndexError) as exc:
                raise ValueError("LiteResearcher pool row has invalid source identity") from exc
            question = str(raw["question"]).strip()
            reward_model = _required_mapping(raw["reward_model"], "reward_model")
            ground_truth = _required_mapping(
                reward_model.get("ground_truth"), "reward_model.ground_truth"
            )
            raw_targets = ground_truth.get("target")
            if not isinstance(raw_targets, list) or not raw_targets:
                raise ValueError("LiteResearcher target must be a non-empty list")
            targets = tuple(str(item).strip() for item in raw_targets)
            if any(not item for item in targets):
                raise ValueError("LiteResearcher target contains an empty value")
            extra_info = _required_mapping(raw["extra_info"], "extra_info")
            mask_url = str(extra_info.get("mask_url", "")).strip()
            if _text_sha256(question) != row["question_sha256"]:
                raise ValueError("LiteResearcher question differs from the frozen pool row")
            if _targets_sha256(targets) != row["targets_sha256"]:
                raise ValueError("LiteResearcher targets differ from the frozen pool row")
            expected_mask_sha = row.get("mask_url_sha256")
            actual_mask_sha = _text_sha256(mask_url) if mask_url else None
            if actual_mask_sha != expected_mask_sha:
                raise ValueError("LiteResearcher mask_url differs from the frozen pool row")
            tasks.append(
                FullPoolLiteResearcherTask(
                    index=expected_index,
                    question=question,
                    targets=targets,
                    mask_url=mask_url,
                    row_identity=str(row["row_identity"]),
                    parquet_path=relative,
                    physical_row=physical_row,
                    data_source=str(raw["data_source"]),
                    upstream_curriculum_stage=int(row["upstream_curriculum_stage"]),
                )
            )

    expected_count = int(manifest["pool"]["contract_compatible_rows"])
    if len(tasks) != expected_count:
        raise ValueError("LiteResearcher full-pool task count mismatch")
    upstream = _required_mapping(manifest.get("upstream"), "upstream")
    return FullPoolLiteResearcherTasks(
        tuple(tasks),
        manifest_sha256=_sha256(manifest_path),
        dataset_revision=str(upstream["dataset_revision"]),
        upstream_commit=str(upstream["source_commit"]),
    )
