#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentenv_agentmemory.presentation_randomization import (
    PRESENTATION_LABEL_CONTRACT,
    PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
    reorder_candidate_options,
    split_candidate_block,
    stable_candidate_permutation,
)


SCHEMA = "memoryarena_candidate_order_audit_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that seeded MemoryArena candidate-order variants preserve every "
            "candidate line, all non-candidate text, and every frozen answer."
        )
    )
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.replicas < 1:
        raise SystemExit("--replicas must be positive")
    payload = args.raw_data.read_bytes()
    rows = _decode_jsonl(payload)
    source_answers_sha256 = _answers_sha256(rows)
    permutation_counts: Counter[str] = Counter()
    identity_count = 0
    variant_count = 0
    records: list[dict[str, Any]] = []

    for source_position, row in enumerate(rows):
        _validate_row(row, source_position)
        task_id = row["category"]
        source_row_id = row["id"]
        for replica in range(args.replicas):
            env_uid = f"audit-env{replica}"
            rendered_hashes: list[str] = []
            source_hashes: list[str] = []
            permutations: list[list[int]] = []
            for session_index, question in enumerate(row["questions"]):
                source = split_candidate_block(question)
                permutation = stable_candidate_permutation(
                    base_seed=args.base_seed,
                    env_uid=env_uid,
                    episode_counter=1,
                    task_id=task_id,
                    source_row_id=source_row_id,
                    session_index=session_index,
                    option_titles=source.option_titles,
                )
                rendered = reorder_candidate_options(question, permutation)
                transformed = split_candidate_block(rendered)
                if transformed.prefix != source.prefix or transformed.suffix != source.suffix:
                    raise SystemExit(
                        f"non-candidate text changed at row={source_position} "
                        f"session={session_index} replica={replica}"
                    )
                if transformed.option_endings != source.option_endings:
                    raise SystemExit(
                        f"candidate row boundaries changed at row={source_position} "
                        f"session={session_index} replica={replica}"
                    )
                if sorted(transformed.option_lines) != sorted(source.option_lines):
                    raise SystemExit(
                        f"candidate content changed at row={source_position} "
                        f"session={session_index} replica={replica}"
                    )
                permutation_key = ",".join(str(index) for index in permutation)
                permutation_counts[permutation_key] += 1
                identity_count += permutation == tuple(range(len(source.option_lines)))
                variant_count += 1
                permutations.append(list(permutation))
                source_hashes.append(_sha256_text(question))
                rendered_hashes.append(_sha256_text(rendered))
            records.append(
                {
                    "source_position": source_position,
                    "source_row_id": source_row_id,
                    "task_id": task_id,
                    "replica": replica,
                    "env_uid": env_uid,
                    "episode_counter": 1,
                    "candidate_permutations": permutations,
                    "source_question_sha256s": source_hashes,
                    "rendered_question_sha256s": rendered_hashes,
                }
            )

    if _answers_sha256(rows) != source_answers_sha256:
        raise SystemExit("frozen answers changed during presentation audit")
    output = {
        "schema": SCHEMA,
        "status": "pass",
        "mode": PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
        "label_contract": PRESENTATION_LABEL_CONTRACT,
        "source": {
            "path": str(args.raw_data.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "answer_payload_sha256": source_answers_sha256,
            "row_count": len(rows),
            "session_count": sum(len(row["questions"]) for row in rows),
        },
        "configuration": {
            "base_seed": args.base_seed,
            "replicas": args.replicas,
        },
        "proof": {
            "changes_target_asins": False,
            "changes_answer_attributes": False,
            "changes_non_candidate_text": False,
            "changes_candidate_text": False,
            "changes_candidate_order": True,
            "variant_session_count": variant_count,
            "identity_permutation_count": identity_count,
            "non_identity_permutation_count": variant_count - identity_count,
            "permutation_counts": dict(sorted(permutation_counts.items())),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "rows": len(rows),
                "variant_sessions": variant_count,
                "non_identity": variant_count - identity_count,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _decode_jsonl(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise SystemExit(f"blank JSONL line {line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"JSONL line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise SystemExit("raw dataset is empty")
    return rows


def _validate_row(row: dict[str, Any], source_position: int) -> None:
    if row.get("id") != source_position:
        raise SystemExit(
            f"source identity mismatch at position {source_position}: {row.get('id')!r}"
        )
    if not isinstance(row.get("category"), str) or not row["category"]:
        raise SystemExit(f"invalid category at row {source_position}")
    questions = row.get("questions")
    answers = row.get("answers")
    if not isinstance(questions, list) or len(questions) != 6:
        raise SystemExit(f"row {source_position} must have six questions")
    if not isinstance(answers, list) or len(answers) != 6:
        raise SystemExit(f"row {source_position} must have six answers")
    for session_index, question in enumerate(questions):
        if not isinstance(question, str):
            raise SystemExit(
                f"row {source_position} session {session_index} question is not text"
            )
        split_candidate_block(question)


def _answers_sha256(rows: list[dict[str, Any]]) -> str:
    answers = [row.get("answers") for row in rows]
    payload = json.dumps(
        answers,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
