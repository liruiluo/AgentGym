from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from agentenv_gaia_text.contracts import (
    GAIA_TEXT_PUBLIC_SCOPE,
    GAIA_TEXT_SCORER_REVISION,
    GAIA_TEXT_SCORER_SHA256,
    PRODUCTION_PROTOCOL,
    ProtocolContract,
)
from agentenv_gaia_text.dataset import GaiaTextDataset
from support import (
    canonical_jsonl,
    protocol_kwargs,
    synthetic_manifest_rows,
    synthetic_question_rows,
)


def _write_inputs(root: Path, rows: list[dict], questions: list[dict] | None = None):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    manifest.write_bytes(canonical_jsonl(rows))
    question_path = root / "questions.jsonl"
    question_path.write_bytes(
        canonical_jsonl(questions or synthetic_question_rows(rows))
    )
    return manifest, question_path


def _load(
    manifest: Path,
    questions: Path,
    *,
    contract: ProtocolContract,
    expected_questions_sha256: str | None = None,
) -> GaiaTextDataset:
    expected = (
        expected_questions_sha256 or hashlib.sha256(questions.read_bytes()).hexdigest()
    )
    return GaiaTextDataset.load(
        manifest,
        questions,
        expected_questions_sha256=expected,
        contract=contract,
    )


def test_valid_synthetic_contract_projects_only_public_task_fields(
    tmp_path: Path,
) -> None:
    rows = synthetic_manifest_rows()
    manifest, questions = _write_inputs(tmp_path, rows)
    contract = ProtocolContract(**protocol_kwargs(rows))

    dataset = _load(manifest, questions, contract=contract)

    assert len(dataset) == 127
    assert dataset.task_ids == tuple(row["task_id"] for row in rows)
    assert dataset.task(0).as_policy_record() == {
        "task_id": "synthetic-task-000",
        "level": 1,
        "question": "Synthetic research question for synthetic-task-000?",
    }
    assert set(dataset.public_metadata()) == {
        "public_scope",
        "protocol_id",
        "dataset_revision",
        "split",
        "task_count",
        "level_counts",
        "manifest_sha256",
        "task_ids_sha256",
        "questions_sha256",
    }
    assert dataset.public_metadata()["public_scope"] == GAIA_TEXT_PUBLIC_SCOPE
    assert GAIA_TEXT_PUBLIC_SCOPE == "GAIA-Text-127-attachment-free"
    assert (
        dataset.questions_sha256 == hashlib.sha256(questions.read_bytes()).hexdigest()
    )


def test_production_contract_rejects_synthetic_rows(tmp_path: Path) -> None:
    rows = synthetic_manifest_rows()
    manifest, questions = _write_inputs(tmp_path, rows)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        _load(manifest, questions, contract=PRODUCTION_PROTOCOL)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "exactly 127"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "level": 2}), "level counts"),
    ],
)
def test_count_and_level_checks_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    original = synthetic_manifest_rows()
    rows = [dict(row) for row in original]
    mutation(rows)
    manifest, questions = _write_inputs(tmp_path, rows)
    contract = ProtocolContract(**protocol_kwargs(original))
    with pytest.raises(ValueError, match=message):
        _load(manifest, questions, contract=contract)


def test_manifest_hash_and_sorted_id_hash_are_independent_gates(tmp_path: Path) -> None:
    rows = synthetic_manifest_rows()
    manifest, questions = _write_inputs(tmp_path, rows)
    good = protocol_kwargs(rows)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**{**good, "manifest_sha256": "0" * 64}),
        )
    with pytest.raises(ValueError, match="task-ID SHA-256"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**{**good, "task_ids_sha256": "0" * 64}),
        )


@pytest.mark.parametrize(
    "forbidden_field",
    ["final_answer", "scorer", "file_name", "attachment", "Annotator Metadata"],
)
def test_question_loader_rejects_gold_scorer_attachment_and_annotator_fields(
    tmp_path: Path,
    forbidden_field: str,
) -> None:
    rows = synthetic_manifest_rows()
    question_rows = synthetic_question_rows(rows)
    question_rows[0][forbidden_field] = "must never enter inference"
    manifest, questions = _write_inputs(tmp_path, rows, question_rows)
    with pytest.raises(ValueError, match="question record 1 has keys"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**protocol_kwargs(rows)),
        )


def test_loader_rejects_noncanonical_order_duplicate_and_question_mismatch(
    tmp_path: Path,
) -> None:
    rows = synthetic_manifest_rows()
    swapped = [dict(row) for row in rows]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    manifest, questions = _write_inputs(tmp_path / "order", swapped)
    with pytest.raises(ValueError, match="sorted by task_id"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**protocol_kwargs(rows)),
        )

    duplicate = [dict(row) for row in rows]
    duplicate[1]["task_id"] = duplicate[0]["task_id"]
    manifest, questions = _write_inputs(tmp_path / "duplicate", duplicate)
    with pytest.raises(ValueError, match="unique"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**protocol_kwargs(rows)),
        )

    question_rows = synthetic_question_rows(rows)
    question_rows[0]["level"] = 3
    manifest, questions = _write_inputs(tmp_path / "mismatch", rows, question_rows)
    with pytest.raises(ValueError, match="does not match manifest"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**protocol_kwargs(rows)),
        )


def test_question_level_rejects_boolean_even_when_it_equals_integer_one(
    tmp_path: Path,
) -> None:
    rows = synthetic_manifest_rows()
    question_rows = synthetic_question_rows(rows)
    question_rows[0]["level"] = True
    manifest, questions = _write_inputs(tmp_path, rows, question_rows)
    with pytest.raises(ValueError, match="question level 1"):
        _load(
            manifest,
            questions,
            contract=ProtocolContract(**protocol_kwargs(rows)),
        )


def test_question_file_hash_rejects_tampered_bytes(tmp_path: Path) -> None:
    rows = synthetic_manifest_rows()
    manifest, questions = _write_inputs(tmp_path, rows)
    pinned_sha256 = hashlib.sha256(questions.read_bytes()).hexdigest()
    tampered = synthetic_question_rows(rows)
    tampered[0]["question"] += " tampered"
    questions.write_bytes(canonical_jsonl(tampered))

    with pytest.raises(ValueError, match="question-file SHA-256"):
        _load(
            manifest,
            questions,
            expected_questions_sha256=pinned_sha256,
            contract=ProtocolContract(**protocol_kwargs(rows)),
        )


def test_repository_contains_no_committed_dataset_fixture_rows() -> None:
    package_root = Path(__file__).resolve().parents[1]
    committed_data = [
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".parquet"}
    ]
    assert committed_data == []
    assert (
        PRODUCTION_PROTOCOL.protocol_id == "gaia_text_2023_validation_no_attachment@"
        "682dd723ee1e1697e00360edccf2366dc8418dd9"
    )
    assert (
        PRODUCTION_PROTOCOL.manifest_sha256
        == "06f6da09978555c39f70f2794499012a1d07eb391e01a0f3d498957b09a1fda7"
    )
    assert (
        PRODUCTION_PROTOCOL.task_ids_sha256
        == "57e76233b8b12d8d9ea18639d1d52616449cf521559cd9d103c76ff399a842ad"
    )
    assert PRODUCTION_PROTOCOL.task_count == 127
    assert PRODUCTION_PROTOCOL.level_counts == ((1, 42), (2, 66), (3, 19))
    assert GAIA_TEXT_SCORER_REVISION == "9f133d71362e77b3539f1514f31b9c101a545fec"
    assert (
        GAIA_TEXT_SCORER_SHA256
        == "0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2"
    )
