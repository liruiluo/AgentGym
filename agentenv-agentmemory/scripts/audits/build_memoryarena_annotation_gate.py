#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from agentenv_agentmemory.annotation_gate import (
    ANNOTATION_GATE_MODES,
    AnnotationGateError,
    build_annotation_gate_bindings,
    build_annotation_gate_manifest,
    validate_annotation_gate_manifest,
    write_annotation_gate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate a run-specific MemoryArena annotation gate."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=ANNOTATION_GATE_MODES, required=True)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--domain-data", required=True)
    parser.add_argument("--items-file", required=True)
    parser.add_argument("--attributes-file", required=True)
    parser.add_argument("--lucene-index-manifest", required=True)
    parser.add_argument("--lucene-index-root", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--audit-chains", required=True)
    parser.add_argument("--manual-evidence", required=True)
    parser.add_argument("--memoryarena-root", required=True)
    parser.add_argument("--memoryarena-base-commit", required=True)
    parser.add_argument("--price-seed", type=int, required=True)
    parser.add_argument(
        "--requested-task-ids",
        help="Optional newline-delimited task IDs; defaults to all audited tasks.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_task_ids = _read_task_ids(args.requested_task_ids)
    try:
        bindings = build_annotation_gate_bindings(
            raw_dataset_path=args.raw_data,
            domain_data_path=args.domain_data,
            items_shuffle_path=args.items_file,
            items_ins_v2_path=args.attributes_file,
            lucene_index_manifest_path=args.lucene_index_manifest,
            lucene_index_root=args.lucene_index_root,
            audit_summary_path=args.audit_summary,
            audit_chains_path=args.audit_chains,
            manual_evidence_path=args.manual_evidence,
            memoryarena_repo_path=args.memoryarena_root,
            memoryarena_base_commit=args.memoryarena_base_commit,
            price_seed=args.price_seed,
        )
        manifest = build_annotation_gate_manifest(
            run_id=args.run_id,
            mode=args.mode,
            raw_dataset_path=args.raw_data,
            audit_summary_path=args.audit_summary,
            audit_chains_path=args.audit_chains,
            manual_evidence_path=args.manual_evidence,
            bindings=bindings,
            requested_task_ids=requested_task_ids,
        )
        allowed = manifest["allowed_task_ids"]
        if not allowed:
            raise AnnotationGateError(
                f"Annotation gate mode {args.mode!r} clears zero whole chains."
            )
        output = Path(args.output).expanduser().resolve()
        manifest_sha256 = write_annotation_gate_manifest(manifest, output)
        validate_annotation_gate_manifest(
            output,
            expected_mode=args.mode,
            expected_run_id=args.run_id,
            expected_manifest_sha256=manifest_sha256,
            selected_task_ids=allowed,
            raw_dataset_path=args.raw_data,
            domain_data_path=args.domain_data,
            items_shuffle_path=args.items_file,
            items_ins_v2_path=args.attributes_file,
            lucene_index_manifest_path=args.lucene_index_manifest,
            lucene_index_root=args.lucene_index_root,
            audit_summary_path=args.audit_summary,
            audit_chains_path=args.audit_chains,
            manual_evidence_path=args.manual_evidence,
            memoryarena_repo_path=args.memoryarena_root,
            memoryarena_base_commit=args.memoryarena_base_commit,
            price_seed=args.price_seed,
        )
    except AnnotationGateError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "MEMORYARENA_ANNOTATION_GATE_OK",
        f"mode={args.mode}",
        f"requested={len(manifest['run']['requested_task_ids'])}",
        f"allowed={len(allowed)}",
        f"excluded={len(manifest['excluded_task_ids'])}",
        f"allowed_task_ids_sha256={manifest['allowed_task_ids_sha256']}",
        f"manifest_sha256={manifest_sha256}",
        f"output={output}",
    )


def _read_task_ids(path_value: str | None) -> tuple[str, ...] | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Cannot read requested task IDs: {path}") from exc
    task_ids = tuple(line.strip() for line in lines if line.strip())
    if not task_ids:
        raise SystemExit(f"Requested task ID file is empty: {path}")
    return task_ids


if __name__ == "__main__":
    main()
