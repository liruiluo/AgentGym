from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import PRODUCTION_PROTOCOL, ProtocolContract

_MANIFEST_KEYS = {"level", "split", "task_id"}
_QUESTION_KEYS = {
    "dataset_revision",
    "level",
    "question",
    "split",
    "task_id",
}


@dataclass(frozen=True)
class GaiaTextTask:
    task_id: str
    level: int
    question: str

    def as_policy_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "question": self.question,
        }


@dataclass(frozen=True)
class GaiaTextDataset:
    tasks: tuple[GaiaTextTask, ...]
    contract: ProtocolContract
    questions_sha256: str

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        questions_path: str | Path,
        *,
        expected_questions_sha256: str,
        contract: ProtocolContract = PRODUCTION_PROTOCOL,
    ) -> GaiaTextDataset:
        manifest_bytes, manifest_rows = _read_jsonl(manifest_path, "manifest")
        _validate_manifest(manifest_bytes, manifest_rows, contract)
        question_bytes, question_rows = _read_jsonl(questions_path, "question")
        expected_questions_sha256 = _sha256(
            expected_questions_sha256, "expected question-file SHA-256"
        )
        observed_questions_sha256 = hashlib.sha256(question_bytes).hexdigest()
        if observed_questions_sha256 != expected_questions_sha256:
            raise ValueError(
                "question-file SHA-256 does not match the staged runtime contract: "
                f"expected {expected_questions_sha256}, got {observed_questions_sha256}"
            )
        tasks = _validate_questions(
            question_bytes,
            question_rows,
            manifest_rows,
            contract,
        )
        return cls(
            tasks=tasks,
            contract=contract,
            questions_sha256=observed_questions_sha256,
        )

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def task(self, data_idx: int) -> GaiaTextTask:
        if isinstance(data_idx, bool) or not isinstance(data_idx, int) or data_idx < 0:
            raise ValueError("data_idx must be a non-negative integer")
        try:
            return self.tasks[data_idx]
        except IndexError as exc:
            raise IndexError(f"GAIA-Text data_idx out of range: {data_idx}") from exc

    def public_metadata(self) -> dict[str, Any]:
        return {
            **self.contract.public_metadata(),
            "questions_sha256": self.questions_sha256,
        }


def _read_jsonl(
    path_value: str | Path, label: str
) -> tuple[bytes, list[dict[str, Any]]]:
    path = Path(path_value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} path must be a real file")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} JSONL must be non-empty and newline terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} JSONL must be UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"{label} JSONL line {line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} JSONL line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise TypeError(f"{label} record {line_number} must be an object")
        rows.append(value)
    return raw, rows


def _validate_manifest(
    raw: bytes,
    rows: list[dict[str, Any]],
    contract: ProtocolContract,
) -> None:
    if len(rows) != contract.task_count:
        raise ValueError(f"manifest must contain exactly {contract.task_count} records")
    task_ids: list[str] = []
    levels: Counter[int] = Counter()
    for line_number, row in enumerate(rows, 1):
        if set(row) != _MANIFEST_KEYS:
            raise ValueError(
                f"manifest record {line_number} has keys {sorted(row)}; "
                f"expected {sorted(_MANIFEST_KEYS)}"
            )
        task_id = _task_id(row["task_id"], f"manifest task_id {line_number}")
        level = row["level"]
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level not in {1, 2, 3}
        ):
            raise ValueError(f"manifest level {line_number} must be integer 1, 2, or 3")
        if row["split"] != contract.split:
            raise ValueError(f"manifest record {line_number} has the wrong split")
        task_ids.append(task_id)
        levels[level] += 1
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("manifest task IDs must be unique")
    if task_ids != sorted(task_ids):
        raise ValueError("manifest records must be sorted by task_id")
    if tuple(sorted(levels.items())) != contract.level_counts:
        raise ValueError(
            "manifest level counts do not match the frozen protocol: "
            f"observed={dict(levels)} expected={dict(contract.level_counts)}"
        )
    canonical = _canonical_jsonl(rows)
    if raw != canonical:
        raise ValueError(
            "manifest must use canonical compact sorted-key JSONL encoding"
        )
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != contract.manifest_sha256:
        raise ValueError(
            "manifest SHA-256 does not match the frozen protocol: "
            f"expected {contract.manifest_sha256}, got {manifest_sha256}"
        )
    id_sha256 = hashlib.sha256(
        "".join(f"{task_id}\n" for task_id in task_ids).encode("utf-8")
    ).hexdigest()
    if id_sha256 != contract.task_ids_sha256:
        raise ValueError(
            "sorted task-ID SHA-256 does not match the frozen protocol: "
            f"expected {contract.task_ids_sha256}, got {id_sha256}"
        )


def _validate_questions(
    raw: bytes,
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    contract: ProtocolContract,
) -> tuple[GaiaTextTask, ...]:
    if len(rows) != contract.task_count:
        raise ValueError(
            f"questions must contain exactly {contract.task_count} records"
        )
    if raw != _canonical_jsonl(rows):
        raise ValueError(
            "questions must use canonical compact sorted-key JSONL encoding"
        )
    tasks: list[GaiaTextTask] = []
    for line_number, (row, manifest) in enumerate(zip(rows, manifest_rows), 1):
        if set(row) != _QUESTION_KEYS:
            raise ValueError(
                f"question record {line_number} has keys {sorted(row)}; "
                f"expected {sorted(_QUESTION_KEYS)}"
            )
        task_id = _task_id(row["task_id"], f"question task_id {line_number}")
        question = _nonempty_text(row["question"], f"question text {line_number}")
        level = row["level"]
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level not in {1, 2, 3}
        ):
            raise ValueError(f"question level {line_number} must be integer 1, 2, or 3")
        if row["dataset_revision"] != contract.dataset_revision:
            raise ValueError(
                f"question record {line_number} has the wrong dataset revision"
            )
        if row["split"] != contract.split:
            raise ValueError(f"question record {line_number} has the wrong split")
        if task_id != manifest["task_id"] or level != manifest["level"]:
            raise ValueError(
                f"question record {line_number} does not match manifest task_id/level"
            )
        tasks.append(
            GaiaTextTask(
                task_id=task_id,
                level=level,
                question=question,
            )
        )
    return tuple(tasks)


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text without NUL bytes")
    return value


def _task_id(value: Any, label: str) -> str:
    value = _nonempty_text(value, label)
    if value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must not contain edge whitespace or line breaks")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value
