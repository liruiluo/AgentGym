from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


DATASET_REPOSITORY = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
HARNESS_REPOSITORY = "SWE-bench/SWE-bench"
HARNESS_REVISION = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
HARNESS_TAG = "v4.1.0"

POLICY_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
)
FORBIDDEN_POLICY_FIELDS = frozenset(
    {
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "hints_text",
        "eval_script",
        "eval_type",
        "log_parser",
        "grader_logs",
        "parser_state",
    }
)
ARMS = ("native", "amg_memory")
MODEL_LABELS = {
    "native": "qwen35-4b-native",
    "amg_memory": "qwen35-4b-amg-memory",
}
EVALUATION_MAX_POLICY_TURNS = 250

_GIT_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class FrozenDatasetPins:
    repository: str
    revision: str
    split: str
    row_count: int
    canonical_jsonl_sha256: str
    id_ledger_sha256: str


PRODUCTION_DATASET_PINS = FrozenDatasetPins(
    repository=DATASET_REPOSITORY,
    revision=DATASET_REVISION,
    split="test",
    row_count=500,
    canonical_jsonl_sha256=(
        "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb"
    ),
    id_ledger_sha256=(
        "a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9"
    ),
)


def policy_projection(instance: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(instance, Mapping):
        raise TypeError("SWE-bench Verified instance must be a mapping")
    projection = {
        field: require_nonempty_text(instance, field) for field in POLICY_FIELDS
    }
    if _GIT_COMMIT_RE.fullmatch(projection["base_commit"]) is None:
        raise ValueError("base_commit must be a full lowercase Git commit")
    repo_parts = projection["repo"].split("/")
    if len(repo_parts) != 2 or not all(repo_parts):
        raise ValueError("repo must have owner/name form")
    if set(projection) != set(POLICY_FIELDS):
        raise RuntimeError("policy projection field contract drifted")
    if set(projection) & FORBIDDEN_POLICY_FIELDS:
        raise RuntimeError("policy projection contains grader-only fields")
    return projection


def require_nonempty_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be non-empty text")
    return item


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def require_arm(value: str) -> str:
    if value not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}")
    return value
