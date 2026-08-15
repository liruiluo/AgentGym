from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentenv_gaia_text.backend import FixtureBackend, RequestError
from agentenv_gaia_text.submission import SubmissionStore
from support import synthetic_manifest_rows, write_runtime_fixture


def test_fixture_backend_is_external_hashed_and_deterministic(tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path)
    backend = FixtureBackend.load(
        runtime.backend, runtime.backend_sha256, page_chars=80
    )

    first = backend.search(["alpha", "evidence"], top_k=2)
    second = backend.search(["alpha", "evidence"], top_k=2)
    assert first == second
    assert first[0]["url"] == "gaia-text://fixture/alpha"
    page = backend.visit(first[0]["url"], goal="synthetic result", page=1)
    assert "forty two" in page["content"]
    assert set(backend.metadata()) == {
        "backend_contract",
        "asset_sha256",
        "document_count",
        "live_network",
        "failure_mode",
        "page_chars",
    }
    assert str(runtime.backend) not in json.dumps(backend.metadata())

    with pytest.raises(ValueError, match="asset SHA-256"):
        FixtureBackend.load(runtime.backend, "0" * 64)
    with pytest.raises(RequestError, match="outside"):
        backend.visit("file:///private/gold.jsonl")
    with pytest.raises(RequestError, match="non-empty"):
        backend.search([])


def test_submission_is_partial_until_all_manifest_ids_are_present(
    tmp_path: Path,
) -> None:
    rows = synthetic_manifest_rows()
    task_ids = [row["task_id"] for row in rows]
    output = tmp_path / "predictions.jsonl"
    store = SubmissionStore(task_ids, output)

    for task_id in task_ids[:-1]:
        receipt = store.record(task_id, None)
    assert receipt["complete"] is False
    assert receipt["submitted_count"] == 126
    assert not output.exists()
    assert Path(str(output) + ".partial").is_file()
    assert str(tmp_path) not in json.dumps(receipt)

    final = store.record(task_ids[-1], "forty two")
    assert final["complete"] is True
    assert final["submitted_count"] == 127
    assert len(final["artifact_sha256"]) == 64
    assert output.is_file()
    assert not Path(str(output) + ".partial").exists()
    parsed = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(parsed) == 127
    assert [row["task_id"] for row in parsed] == task_ids
    assert all(list(row) == ["task_id", "model_answer"] for row in parsed)
    assert parsed[0]["model_answer"] is None
    assert parsed[-1]["model_answer"] == "forty two"

    with pytest.raises(ValueError, match="already has a submission"):
        store.record(task_ids[-1], "duplicate")
    with pytest.raises(KeyError, match="outside"):
        store.record("synthetic-task-outside", None)


def test_submission_requires_string_or_null_and_fresh_external_target(
    tmp_path: Path,
) -> None:
    task_ids = [row["task_id"] for row in synthetic_manifest_rows()]
    output = tmp_path / "predictions.jsonl"
    store = SubmissionStore(task_ids, output)
    with pytest.raises(TypeError, match="string or null"):
        store.record(task_ids[0], 42)  # type: ignore[arg-type]

    output.write_text("stale")
    with pytest.raises(FileExistsError, match="already exists"):
        SubmissionStore(task_ids, output)
