from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

RESOURCE_CONTRACT_SCHEMA = "mlebench_lite_resource_contract_v2"
RESOURCE_USAGE_KEYS = (
    "execution_time_ms",
    "cpu_time_ms",
    "writable_bytes",
    "writable_inodes",
    "processes_started",
)

DEFAULT_EPISODE_TIMEOUT_MS = 86_400_000
DEFAULT_MAX_SHELL_TIMEOUT_MS = 3_600_000
DEFAULT_MAX_TOTAL_EXECUTION_MS = 72_000_000
DEFAULT_CPU_LIMIT_CORES = 36
DEFAULT_MEMORY_LIMIT_BYTES = 440_000_000_000
DEFAULT_PIDS_LIMIT = 4096
DEFAULT_WRITABLE_BYTES_LIMIT = 500_000_000_000
DEFAULT_WRITABLE_INODES_LIMIT = 2_000_000
DEFAULT_GPU_COUNT = 1
DEFAULT_STEP_RESPONSE_SLACK_MS = 30_000

_NUMERIC_RESOURCE_FIELDS = (
    "max_actions",
    "max_submission_bytes",
    "max_shell_timeout_ms",
    "max_visible_output_bytes",
    "episode_timeout_ms",
    "max_total_execution_ms",
    "cpu_limit_cores",
    "memory_limit_bytes",
    "pids_limit",
    "writable_bytes_limit",
    "writable_inodes_limit",
    "gpu_count",
    "max_step_response_ms",
)
_RESOURCE_CONTRACT_FIELDS = {
    "schema",
    *_NUMERIC_RESOURCE_FIELDS,
    "submission_path",
    "network_disabled",
    "read_only_public_data",
    "process_scope",
    "cgroup_required",
    "isolated_process_group_required",
}


def build_resource_contract(
    *,
    max_actions: int,
    max_submission_bytes: int,
    max_shell_timeout_ms: int,
    max_visible_output_bytes: int,
    submission_path: str,
    episode_timeout_ms: int = DEFAULT_EPISODE_TIMEOUT_MS,
    max_total_execution_ms: int = DEFAULT_MAX_TOTAL_EXECUTION_MS,
    cpu_limit_cores: int = DEFAULT_CPU_LIMIT_CORES,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    pids_limit: int = DEFAULT_PIDS_LIMIT,
    writable_bytes_limit: int = DEFAULT_WRITABLE_BYTES_LIMIT,
    writable_inodes_limit: int = DEFAULT_WRITABLE_INODES_LIMIT,
    gpu_count: int = DEFAULT_GPU_COUNT,
) -> dict[str, Any]:
    values = {
        "max_actions": max_actions,
        "max_submission_bytes": max_submission_bytes,
        "max_shell_timeout_ms": max_shell_timeout_ms,
        "max_visible_output_bytes": max_visible_output_bytes,
        "episode_timeout_ms": episode_timeout_ms,
        "max_total_execution_ms": max_total_execution_ms,
        "cpu_limit_cores": cpu_limit_cores,
        "memory_limit_bytes": memory_limit_bytes,
        "pids_limit": pids_limit,
        "writable_bytes_limit": writable_bytes_limit,
        "writable_inodes_limit": writable_inodes_limit,
        "gpu_count": gpu_count,
        "max_step_response_ms": episode_timeout_ms + DEFAULT_STEP_RESPONSE_SLACK_MS,
    }
    contract = {
        "schema": RESOURCE_CONTRACT_SCHEMA,
        **values,
        "submission_path": submission_path,
        "network_disabled": True,
        "read_only_public_data": True,
        "process_scope": "episode_cgroup_descendants",
        "cgroup_required": True,
        "isolated_process_group_required": True,
    }
    return validate_resource_contract(contract)


def validate_resource_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_CONTRACT_FIELDS:
        raise ValueError("resource contract fields drifted")
    contract = dict(value)
    if contract["schema"] != RESOURCE_CONTRACT_SCHEMA:
        raise ValueError("resource contract schema drifted")
    for label in _NUMERIC_RESOURCE_FIELDS:
        item = contract[label]
        if type(item) is not int or item <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if contract["max_shell_timeout_ms"] > contract["episode_timeout_ms"]:
        raise ValueError("max_shell_timeout_ms exceeds the episode deadline")
    if contract["max_total_execution_ms"] > contract["episode_timeout_ms"]:
        raise ValueError("max_total_execution_ms exceeds the episode deadline")
    if contract["max_submission_bytes"] > contract["writable_bytes_limit"]:
        raise ValueError("submission limit exceeds writable-byte budget")
    if contract["max_step_response_ms"] != (
        contract["episode_timeout_ms"] + DEFAULT_STEP_RESPONSE_SLACK_MS
    ):
        raise ValueError("max_step_response_ms is not canonically derived")
    submission_path = contract["submission_path"]
    if (
        not isinstance(submission_path, str)
        or not submission_path.startswith("/")
        or "\x00" in submission_path
    ):
        raise ValueError("submission_path must be absolute")
    fixed = {
        "network_disabled": True,
        "read_only_public_data": True,
        "process_scope": "episode_cgroup_descendants",
        "cgroup_required": True,
        "isolated_process_group_required": True,
    }
    if any(contract[key] != expected for key, expected in fixed.items()):
        raise ValueError("resource contract isolation fields drifted")
    if any(
        type(contract[key]) is not bool
        for key in (
            "network_disabled",
            "read_only_public_data",
            "cgroup_required",
            "isolated_process_group_required",
        )
    ):
        raise ValueError("resource contract boolean fields drifted")
    return contract


def resource_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes(validate_resource_contract(contract))
    ).hexdigest()


def zero_resource_usage() -> dict[str, int]:
    return {key: 0 for key in RESOURCE_USAGE_KEYS}


def validate_resource_usage(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(RESOURCE_USAGE_KEYS):
        raise ValueError(f"{label} fields drifted")
    result = dict(value)
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise ValueError(f"{label} is invalid")
    return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
