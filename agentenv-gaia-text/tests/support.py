from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SYNTHETIC_REVISION = "1" * 40


def canonical_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def synthetic_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(127):
        level = 1 if index < 42 else 2 if index < 108 else 3
        rows.append(
            {
                "level": level,
                "split": "validation",
                "task_id": f"synthetic-task-{index:03d}",
            }
        )
    return rows


def synthetic_question_rows(
    manifest_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_revision": SYNTHETIC_REVISION,
            "level": row["level"],
            "question": f"Synthetic research question for {row['task_id']}?",
            "split": row["split"],
            "task_id": row["task_id"],
        }
        for row in manifest_rows
    ]


def protocol_kwargs(
    rows: Iterable[dict[str, Any]],
    *,
    expected_count: int = 127,
) -> dict[str, Any]:
    materialized = list(rows)
    manifest_bytes = canonical_jsonl(materialized)
    id_bytes = "".join(f"{row['task_id']}\n" for row in materialized).encode()
    return {
        "protocol_id": f"synthetic_gaia_text@{SYNTHETIC_REVISION}",
        "dataset_revision": SYNTHETIC_REVISION,
        "split": "validation",
        "task_count": expected_count,
        "level_counts": ((1, 42), (2, 66), (3, 19)),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "task_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
    }


@dataclass(frozen=True)
class RuntimeFixture:
    root: Path
    manifest: Path
    questions: Path
    questions_sha256: str
    backend: Path
    backend_sha256: str
    predictions: Path
    rows: tuple[dict[str, Any], ...]


def write_runtime_fixture(root: Path) -> RuntimeFixture:
    root.mkdir(parents=True, exist_ok=True)
    rows = synthetic_manifest_rows()
    manifest = root / "manifest.jsonl"
    manifest.write_bytes(canonical_jsonl(rows))
    questions = root / "questions.jsonl"
    question_bytes = canonical_jsonl(synthetic_question_rows(rows))
    questions.write_bytes(question_bytes)
    backend_payload = {
        "documents": [
            {
                "content": "Alpha evidence says the synthetic result is forty two.",
                "title": "Alpha evidence",
                "url": "gaia-text://fixture/alpha",
            },
            {
                "content": (
                    "Beta evidence supplies a second deterministic page. " * 80
                ).strip(),
                "title": "Beta evidence",
                "url": "gaia-text://fixture/beta",
            },
        ],
        "schema": "gaia_text_fixture_backend_v1",
    }
    backend = root / "backend.json"
    backend_bytes = json.dumps(
        backend_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    backend.write_bytes(backend_bytes)
    return RuntimeFixture(
        root=root,
        manifest=manifest,
        questions=questions,
        questions_sha256=hashlib.sha256(question_bytes).hexdigest(),
        backend=backend,
        backend_sha256=hashlib.sha256(backend_bytes).hexdigest(),
        predictions=root / "predictions.jsonl",
        rows=tuple(rows),
    )


class FileWorkspace:
    def __init__(self, root: Path, label: str) -> None:
        self.root = root / label
        self.reset_ids: list[str] = []
        self.actions: list[str] = []
        self.closed = False

    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        assert enabled is True
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, mode=0o700)
        self.reset_ids.append(episode_id)
        self.closed = False

    def apply(self, action: str, *, env_step: int, phase_index: int):
        prefix = "shell_command "
        if not action.startswith(prefix):
            raise ValueError("test workspace accepts only shell_command")
        payload = json.loads(action[len(prefix) :])
        if set(payload) - {"command", "workdir", "timeout_ms"}:
            raise ValueError("unsupported shell argument")
        workdir = payload.get("workdir", ".")
        cwd = (self.root / workdir).resolve()
        if self.root.resolve() not in (cwd, *cwd.parents):
            raise ValueError("workdir escapes the test workspace")
        completed = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", payload["command"]],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=payload.get("timeout_ms", 10_000) / 1000,
        )
        self.actions.append(action)
        output = completed.stdout
        if completed.stderr:
            output += "[stderr]\n" + completed.stderr
        return SimpleNamespace(
            message=f"Exit code: {completed.returncode}\nOutput:\n{output or '<no output>'}",
            op="SHELL_COMMAND",
            workspace_diff={"test_only": True},
        )

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.closed = True
