from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class SubmissionStore:
    """Publish scorer input only after every frozen manifest ID has an outcome."""

    def __init__(self, task_ids: Iterable[str], output_path: str | Path) -> None:
        ordered = tuple(task_ids)
        if (
            not ordered
            or any(
                not isinstance(task_id, str)
                or not task_id
                or task_id != task_id.strip()
                or "\n" in task_id
                or "\r" in task_id
                for task_id in ordered
            )
            or len(set(ordered)) != len(ordered)
        ):
            raise ValueError("submission task IDs must be non-empty and unique")
        if ordered != tuple(sorted(ordered)):
            raise ValueError("submission task IDs must retain sorted manifest order")
        output = Path(output_path).expanduser()
        parent = output.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("submission parent must be a real existing directory")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"submission output already exists: {output.name}")
        partial = Path(str(output) + ".partial")
        if partial.exists() or partial.is_symlink():
            raise FileExistsError(
                f"submission partial output already exists: {partial.name}"
            )
        self._task_ids = ordered
        self._task_id_set = frozenset(ordered)
        self._output = output
        self._partial = partial
        self._answers: dict[str, str | None] = {}
        self._lock = threading.Lock()

    @property
    def submitted_count(self) -> int:
        with self._lock:
            return len(self._answers)

    def record(self, task_id: str, model_answer: str | None) -> dict[str, Any]:
        if task_id not in self._task_id_set:
            raise KeyError("submission task ID is outside the frozen manifest")
        if model_answer is not None and not isinstance(model_answer, str):
            raise TypeError("model_answer must be a string or null")
        with self._lock:
            if task_id in self._answers:
                raise ValueError(f"task {task_id!r} already has a submission")
            self._answers[task_id] = model_answer
            try:
                payload = self._render()
                _atomic_write(self._partial, payload)
                complete = len(self._answers) == len(self._task_ids)
                if complete:
                    os.replace(self._partial, self._output)
            except Exception:
                del self._answers[task_id]
                raise
            return {
                "schema": "gaia_text_external_submission_receipt_v1",
                "expected_count": len(self._task_ids),
                "submitted_count": len(self._answers),
                "complete": complete,
                "artifact_sha256": hashlib.sha256(payload).hexdigest()
                if complete
                else None,
            }

    def _render(self) -> bytes:
        records = (
            {"task_id": task_id, "model_answer": self._answers[task_id]}
            for task_id in self._task_ids
            if task_id in self._answers
        )
        return b"".join(
            (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            for record in records
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp.unlink(missing_ok=True)
        raise
