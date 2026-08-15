from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentenv_mlebench_lite.identity import (
    LITE_COMPETITION_IDS,
    SPLIT_SHA256,
    UPSTREAM_COMMIT,
)
from agentenv_mlebench_lite.resources import zero_resource_usage


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


FAKE_RUNNER_SHA256 = sha256_bytes(b"pinned formal sandbox runner")
FAKE_RUNTIME_DIGEST = sha256_bytes(b"pinned formal sandbox runtime")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def write_fixture(root: Path) -> dict[str, Any]:
    upstream_root = root / "upstream"
    split_path = upstream_root / "experiments" / "splits" / "low.txt"
    split_path.parent.mkdir(parents=True)
    split_payload = "\n".join(LITE_COMPETITION_IDS).encode("utf-8")
    assert sha256_bytes(split_payload) == SPLIT_SHA256
    split_path.write_bytes(split_payload)

    data_root = root / "prepared-data"
    tasks: list[dict[str, str]] = []
    for index, competition_id in enumerate(LITE_COMPETITION_IDS):
        public_relative = Path(competition_id) / "prepared" / "public"
        private_relative = Path(competition_id) / "prepared" / "private"
        public_task = data_root / public_relative
        private_task = data_root / private_relative
        public_task.mkdir(parents=True)
        private_task.mkdir(parents=True)
        public_payload = f"row,value\n{index},public-{competition_id}\n"
        (public_task / "train.csv").write_text(public_payload, encoding="utf-8")
        (private_task / "answer.csv").write_text(
            f"row,target\n{index},secret\n", encoding="utf-8"
        )
        public_files = [
            {
                "path": "train.csv",
                "size": len(public_payload.encode("utf-8")),
                "sha256": sha256_bytes(public_payload.encode("utf-8")),
            }
        ]
        tasks.append(
            {
                "competition_id": competition_id,
                "public_relative_path": public_relative.as_posix(),
                "private_relative_path": private_relative.as_posix(),
                "public_files": public_files,
                "public_tree_sha256": canonical_sha256(public_files),
            }
        )

    manifest = {
        "schema": "mlebench_lite_public_manifest_v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "split_sha256": SPLIT_SHA256,
        "tasks": tasks,
    }
    manifest_path = root / "public-manifest.json"
    manifest_payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_payload)
    return {
        "upstream_root": upstream_root,
        "data_root": data_root,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "episodes_root": root / "episodes",
    }


def sandbox_attestation(workspace) -> dict[str, Any]:
    return {
        "schema": "mlebench_lite_sandbox_attestation_v2",
        "runner_sha256": FAKE_RUNNER_SHA256,
        "runtime_digest": FAKE_RUNTIME_DIGEST,
        "resource_contract": dict(workspace.resource_contract),
        "resource_contract_sha256": workspace.resource_contract_sha256,
        "mount_namespace": True,
        "network_disabled": True,
        "non_root": True,
        "read_only_rootfs": True,
        "execution_scope": {
            "scope": "episode_cgroup_descendants",
            "cgroup_enforced": True,
            "isolated_process_group": True,
        },
        "memory_namespace": {
            "path": "/home/workspace/.agent_memory",
            "state": (
                "task_local_rw"
                if workspace.mode == "amg_memory"
                else "absent_and_denied"
            ),
        },
        "mounts": [
            {
                "source": str(workspace.public_root),
                "target": "/home/data",
                "read_only": True,
                "source_tree_sha256": workspace.public_tree_sha256,
            },
            {
                "source": str(workspace.workspace_root),
                "target": "/home/workspace",
                "read_only": False,
            },
            {
                "source": str(workspace.submission_root),
                "target": "/home/submission",
                "read_only": False,
            },
        ],
        "denied_mount_prefixes": ["/host", "/private"],
    }


@dataclass(frozen=True)
class FakeExecutionResult:
    returncode: int = 0
    stdout: str = "sandbox stdout"
    stderr: str = ""
    timed_out: bool = False
    receipt: dict[str, Any] | None = None


class RecordingFormalBackend:
    formal_isolation = True

    def __init__(self) -> None:
        self.attested: list[Any] = []
        self.executed: list[dict[str, Any]] = []
        self.frozen: list[Any] = []
        self.torn_down: list[Any] = []
        self.attestation_override: dict[str, Any] | None = None
        self.resource_usage: dict[str, dict[str, int]] = {}

    def attest(self, workspace):
        self.attested.append(workspace)
        return self.attestation_override or sandbox_attestation(workspace)

    def execute(
        self,
        *,
        workspace,
        command: str,
        timeout_ms: int,
        operation_id: str,
    ):
        self.executed.append(
            {
                "workspace": workspace,
                "command": command,
                "timeout_ms": timeout_ms,
                "operation_id": operation_id,
            }
        )
        if (
            "/private" in command
            or str(workspace.public_root.parent.parent) in command
            or (workspace.mode == "native" and ".agent_memory" in command)
        ):
            return self._result(
                workspace,
                command,
                timeout_ms,
                operation_id=operation_id,
                returncode=1,
                stdout="",
                stderr="not available",
            )
        return self._result(
            workspace,
            command,
            timeout_ms,
            operation_id=operation_id,
        )

    def _result(
        self,
        workspace,
        command: str,
        timeout_ms: int,
        *,
        operation_id: str,
        returncode: int = 0,
        stdout: str = "sandbox stdout",
        stderr: str = "",
        timed_out: bool = False,
        resource_delta: dict[str, int] | None = None,
    ) -> FakeExecutionResult:
        attestation = sandbox_attestation(workspace)
        prior = self.resource_usage.setdefault(
            workspace.episode_id, zero_resource_usage()
        )
        delta = resource_delta or {
            "execution_time_ms": 1,
            "cpu_time_ms": 1,
            "writable_bytes": 0,
            "writable_inodes": 0,
            "processes_started": 1,
        }
        cumulative = {key: prior[key] + delta[key] for key in prior}
        self.resource_usage[workspace.episode_id] = cumulative
        receipt = {
            "schema": "mlebench_lite_sandbox_execution_v2",
            "operation_id": operation_id,
            "runner_sha256": FAKE_RUNNER_SHA256,
            "runtime_digest": FAKE_RUNTIME_DIGEST,
            "resource_contract_sha256": workspace.resource_contract_sha256,
            "mount_attestation_sha256": canonical_sha256(attestation),
            "command_sha256": sha256_bytes(command.encode("utf-8")),
            "timeout_ms": timeout_ms,
            "returncode": returncode,
            "timed_out": timed_out,
            "resource_delta": delta,
            "resource_cumulative": cumulative,
            "containment": {
                "scope": "episode_cgroup_descendants",
                "cgroup_enforced": True,
                "isolated_process_group": True,
                "descendant_process_count": 0,
            },
        }
        return FakeExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            receipt=receipt,
        )

    def freeze_and_reap(self, *, workspace, operation_id):
        self.frozen.append(workspace)
        return {
            "schema": "mlebench_lite_sandbox_freeze_v2",
            "operation_id": operation_id,
            "runner_sha256": FAKE_RUNNER_SHA256,
            "runtime_digest": FAKE_RUNTIME_DIGEST,
            "resource_contract_sha256": workspace.resource_contract_sha256,
            "mount_attestation_sha256": canonical_sha256(
                sandbox_attestation(workspace)
            ),
            "resource_cumulative": self.resource_usage.get(
                workspace.episode_id, zero_resource_usage()
            ),
            "processes_reaped": True,
            "workspace_frozen": True,
            "descendant_process_count": 0,
        }

    def teardown(self, *, workspace, operation_id):
        self.torn_down.append(workspace)
        return {
            "schema": "mlebench_lite_sandbox_teardown_v2",
            "operation_id": operation_id,
            "runner_sha256": FAKE_RUNNER_SHA256,
            "runtime_digest": FAKE_RUNTIME_DIGEST,
            "resource_contract_sha256": workspace.resource_contract_sha256,
            "mount_attestation_sha256": canonical_sha256(
                sandbox_attestation(workspace)
            ),
            "resource_cumulative": self.resource_usage.get(
                workspace.episode_id, zero_resource_usage()
            ),
            "processes_reaped": True,
            "mounts_released": True,
            "descendant_process_count": 0,
            "mount_count": 0,
            "sandbox_present": False,
        }


class UnsafeLocalBackend(RecordingFormalBackend):
    formal_isolation = False
