from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

RELEASE_REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"
PRIVATE_RUNTIME_DIGEST = "sha256:" + "9" * 64
TASK_ID = "tiny-regression@1"
PRIVATE_CANARY = "OPENMLE_PRIVATE_CANARY_DO_NOT_EXPOSE"
TASK_MARKDOWN = """# OpenMLE-fast task

Task ID: tiny-regression@1
Objective: predict `target` for every row in `data/test.csv`.
Metric: mean absolute error (lower is better).
Write the final file to `/workspace/submission.csv`.
Use only the public files listed below. The episode has 30 total actions.
"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return sha256_bytes(payload)


def tree_sha256(entries: list[dict[str, Any]]) -> str:
    payload = (
        json.dumps(
            sorted(entries, key=lambda item: item["path"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return sha256_bytes(payload)


def create_fixture(
    root: Path,
    *,
    metric_source: str | None = None,
) -> dict[str, Path | str | dict[str, Any]]:
    package_root = root / "runtime"
    package = package_root / "public-tasks" / TASK_ID
    public = package / "data"
    private = package_root / "tasks" / TASK_ID
    metric_dir = private
    archive_root = package_root
    archive_dir = archive_root / "archives"
    provenance_root = package_root / "provenance" / TASK_ID
    episodes_root = root / "episodes"
    audit_root = root / "audit"
    for directory in (
        public,
        private,
        metric_dir,
        archive_dir,
        provenance_root,
        episodes_root,
        audit_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    public_payloads = {
        "description.txt": b"A tiny public regression task.\n",
        "sample_submission.csv": b"id,target\n3,5\n4,6\n",
        "test.csv": b"id,feature\n3,30\n4,40\n",
        "train.csv": b"id,feature,target\n1,10,1\n2,20,2\n",
    }
    for relative, payload in public_payloads.items():
        (public / relative).write_bytes(payload)
    answer = private / "test_answer.csv"
    answer.write_text("id,target\n3,1\n4,2\n", encoding="utf-8")
    (private / "private-canary.txt").write_text(PRIVATE_CANARY, encoding="utf-8")

    metric = metric_dir / "metric.py"
    metric.write_text(
        metric_source
        or """import numpy as np
import pandas as pd

class TinyRegressionMetric:
    def __init__(self):
        self.higher_is_better = False

    def validate_submission(self, pred, truth):
        if list(pred.columns) != list(truth.columns):
            return False, "columns"
        if len(pred) != len(truth):
            return False, "rows"
        if pred.iloc[:, 0].astype(str).tolist() != truth.iloc[:, 0].astype(str).tolist():
            return False, "ids"
        values = pd.to_numeric(pred["target"], errors="coerce").to_numpy()
        return bool(np.isfinite(values).all())

    def evaluate(self, y_true, y_pred):
        truth = pd.to_numeric(y_true["target"]).to_numpy(dtype=float)
        pred = pd.to_numeric(y_pred["target"]).to_numpy(dtype=float)
        return float(np.mean(np.abs(truth - pred)))
""",
        encoding="utf-8",
    )
    archive = archive_dir / f"{TASK_ID}.tar.zst"
    archive.write_bytes(b"frozen-test-archive")
    task_path = package / "TASK.md"
    task_path.write_text(TASK_MARKDOWN, encoding="utf-8")
    task_spec_sha256 = sha256_file(task_path)
    visible_entries = [
        {
            "path": "TASK.md",
            "size": len(TASK_MARKDOWN.encode("utf-8")),
            "sha256": sha256_bytes(TASK_MARKDOWN.encode("utf-8")),
            "mode": 0o444,
        }
    ]
    for relative, payload in sorted(public_payloads.items()):
        visible_entries.append(
            {
                "path": f"data/{relative}",
                "size": len(payload),
                "sha256": sha256_bytes(payload),
                "mode": 0o444,
            }
        )
    for path in package.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    public_tree_sha256 = tree_sha256(visible_entries)
    validator_contract = {
        "audit": "static_ast_success_return_plus_frozen_baseline_evidence_v1",
        "kind": "bool_message_tuple",
        "stringified_baseline_evidence": "(True, 'Submission is valid.')",
        "success": True,
        "success_value": "Submission is valid.",
    }
    normalization = {
        "admission_gap_floor": 0.000004,
        "baseline_score": 4.0,
        "directed_ideal_gap": 4.0,
        "equality_tolerance": 4e-9,
        "higher_is_better": False,
        "ideal_score": 0.0,
        "reward_contract": "openmle_fast_normalized_terminal_reward_v1",
        "reward_eligible": True,
        "reward_formula": "clip(d*(score-baseline)/(d*(ideal-baseline)),-1,1)",
        "scale": 4.0,
    }
    source_family = "TEST:tiny-regression"
    source_family_sha256 = canonical_sha256(
        {"download_ref": "tiny-regression", "release_source_type": "TEST"}
    )
    evidence_sha256 = "7" * 64
    provenance_payloads = {
        "LICENSE": b"CC0-1.0\n",
        "NOTICE": b"Synthetic unit-test task.\n",
    }
    provenance_files: list[dict[str, Any]] = []
    for relative, payload in sorted(provenance_payloads.items()):
        path = provenance_root / relative
        path.write_bytes(payload)
        path.chmod(0o444)
        provenance_files.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": sha256_bytes(payload),
                "mode": 0o444,
            }
        )
    binding = {
        "schema": "openmle_fast_grader_binding_v1",
        "task_id": TASK_ID,
        "archive_sha256": sha256_file(archive),
        "metric_sha256": sha256_file(metric),
        "answer_sha256": sha256_file(answer),
        "public_tree_sha256": public_tree_sha256,
        "task_spec_sha256": task_spec_sha256,
        "evidence_sha256": evidence_sha256,
        "source_family_sha256": source_family_sha256,
        "metric_class": "TinyRegressionMetric",
        "normalization": normalization,
        "validator_contract": validator_contract,
    }
    binding_sha256 = canonical_sha256(binding)
    binding_id = f"openmlefast-grader-{binding_sha256[:24]}"
    identity = {
        "task_id": TASK_ID,
        "archive_sha256": sha256_file(archive),
        "public_tree_sha256": public_tree_sha256,
        "metric_sha256": sha256_file(metric),
        "private_grader_binding_sha256": binding_sha256,
        "task_spec_sha256": task_spec_sha256,
    }
    task = {
        "data_idx": 0,
        "task_id": TASK_ID,
        "source_family": source_family,
        "source_family_sha256": source_family_sha256,
        "source_urls": ["https://example.invalid/source"],
        "license_name_or_permission": "CC0-1.0",
        "archive_relpath": f"archives/{TASK_ID}.tar.zst",
        "archive_sha256": sha256_file(archive),
        "public_task_relpath": f"public-tasks/{TASK_ID}",
        "public_files": visible_entries,
        "public_tree_sha256": public_tree_sha256,
        "package_identity_sha256": canonical_sha256(identity),
        "grader_binding": binding_id,
        "grader_binding_sha256": binding_sha256,
        "baseline_score": 4.0,
        "ideal_score": 0.0,
        "higher_is_better": False,
        "metric_name": "TinyRegressionMetric",
        "metric_direction": "lower",
        "reward_eligible": True,
        "reward_block_reason": None,
        "engineering_gate_member": True,
        "engineering_gate_role": "gate_only",
        "role": "gate_only",
        "split": "train",
        "normalization_contract": "openmle_fast_normalized_terminal_reward_v1",
        "evidence_sha256": evidence_sha256,
        "provenance_relpath": f"provenance/{TASK_ID}",
        "provenance_files": provenance_files,
        "task_spec_sha256": task_spec_sha256,
    }
    task_ids_sha256 = sha256_bytes((TASK_ID + "\n").encode("utf-8"))
    families_sha256 = sha256_bytes((source_family + "\n").encode("utf-8"))
    compact = [
        {
            "task_id": TASK_ID,
            "source_family": source_family,
            "archive_sha256": task["archive_sha256"],
            "public_tree_sha256": public_tree_sha256,
            "package_identity_sha256": task["package_identity_sha256"],
            "reward_eligible": True,
            "engineering_gate_member": True,
        }
    ]
    manifest = {
        "schema": "openmle_fast_public_manifest_v1",
        "contract_version": "openmle_fast_v1",
        "panel_id": "openmle-fast-unit-gate",
        "runtime_id": "openmle-fast-unit-v1",
        "openmle_tasks_revision": RELEASE_REVISION,
        "release_revision": RELEASE_REVISION,
        "role": "gate_only",
        "task_count": 1,
        "source_family_count": 1,
        "max_policy_actions": 30,
        "runtime_contract_sha256": "4" * 64,
        "source_evidence_sha256": "5" * 64,
        "task_id_list_sha256": task_ids_sha256,
        "source_family_list_sha256": families_sha256,
        "compact_panel_sha256": canonical_sha256(compact),
        "records": [task],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    private_manifest = {
        "schema": "openmle_fast_fullpool_private_grader_manifest_v1",
        "contract_version": "openmle_fast_v1",
        "runtime_id": "openmle-fast-unit-v1",
        "openmle_tasks_revision": RELEASE_REVISION,
        "task_count": 1,
        "public_manifest_sha256": {"g64": sha256_file(manifest_path)},
        "records": [
            {
                "private_data_idx": 0,
                "task_id": TASK_ID,
                "split": "train",
                "engineering_gate_member": True,
                "reward_eligible": True,
                "reward_block_reason": None,
                "source_family": source_family,
                "source_family_sha256": source_family_sha256,
                "grader_binding": binding_id,
                "grader_binding_sha256": binding_sha256,
                "grader_binding_payload": binding,
                "archive_sha256": sha256_file(archive),
                "archive_relpath": f"archives/{TASK_ID}.tar.zst",
                "metric_relpath": f"tasks/{TASK_ID}/metric.py",
                "metric_sha256": sha256_file(metric),
                "answer_relpath": f"tasks/{TASK_ID}/test_answer.csv",
                "answer_sha256": sha256_file(answer),
                "public_tree_sha256": public_tree_sha256,
                "task_spec_sha256": task_spec_sha256,
                "package_identity_sha256": canonical_sha256(identity),
                "normalization": normalization,
                "validator_contract": validator_contract,
            }
        ],
    }
    private_manifest_path = root / "private-manifest.json"
    private_manifest_path.write_text(
        json.dumps(private_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    credential = root / "grader.credential"
    credential.write_bytes(os.urandom(32))
    credential.chmod(0o600)
    return {
        "root": root,
        "package_root": package_root,
        "archive_root": archive_root,
        "episodes_root": episodes_root,
        "audit_root": audit_root,
        "manifest": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "private_manifest": private_manifest_path,
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "credential": credential,
        "package": package,
        "task": {
            **task,
            "private_grader_binding_sha256": binding_sha256,
        },
    }


class GraderServiceThread:
    def __init__(self, service: Any) -> None:
        self.service = service
        self.thread = threading.Thread(target=service.serve_forever, daemon=True)

    def __enter__(self) -> Any:
        self.thread.start()
        if not self.service.wait_until_ready(timeout=5.0):
            raise RuntimeError("private grader service did not become ready")
        return self.service

    def __exit__(self, *_exc: object) -> None:
        self.service.shutdown()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise RuntimeError("private grader service did not stop")
