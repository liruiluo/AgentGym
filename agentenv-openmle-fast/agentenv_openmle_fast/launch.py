from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from . import actions, materializer
from .audit import OpenMLEFastAuditSink
from .dataset import ALLOWED_MANIFEST_ROLES, OpenMLEFastDataset
from .environment import OpenMLEFastEpisodeManager
from .executor import (
    ExternalSandboxRunnerBackend,
    OpenMLEFastExecutor,
    OpenMLEFastResourceLimits,
)
from .grader_client import PrivateGraderClient
from .materializer import OpenMLEFastWorkspaceMaterializer
from .private_grader import PrivateGraderService
from .private_grader_runner import (
    ExternalPrivateGraderRunnerBackend,
    PrivateGraderLimits,
)


def build_manager_from_environment() -> OpenMLEFastEpisodeManager:
    limits = _limits_from_environment()
    manifest = _required_file("OPENMLE_FAST_TASK_MANIFEST")
    package_root = _required_directory("OPENMLE_FAST_PACKAGE_ROOT")
    archive_root = _required_directory("OPENMLE_FAST_ARCHIVE_ROOT")
    episodes_root = _required_directory("OPENMLE_FAST_EPISODES_ROOT", create=True)
    expected_release = _required_revision("OPENMLE_FAST_RELEASE_REVISION")
    manifest_role = _required_manifest_role()
    dataset = OpenMLEFastDataset(
        manifest_path=manifest,
        package_root=package_root,
        archive_root=archive_root,
        expected_manifest_sha256=_required_sha256("OPENMLE_FAST_TASK_MANIFEST_SHA256"),
        expected_release_revision=expected_release,
        expected_role=manifest_role,
    )
    _verify_implementation_digest(
        Path(materializer.__file__),
        _required_sha256("OPENMLE_FAST_MATERIALIZER_SHA256"),
        "materializer",
    )
    _verify_implementation_digest(
        Path(actions.__file__),
        _required_sha256("OPENMLE_FAST_ACTIONS_SHA256"),
        "action parser",
    )
    runner_path = _required_file("OPENMLE_FAST_EXECUTOR_RUNNER")
    runner_sha256 = _required_sha256("OPENMLE_FAST_EXECUTOR_RUNNER_SHA256")
    runtime_digest = _required_runtime_digest("OPENMLE_FAST_EXECUTOR_RUNTIME_DIGEST")
    backend = ExternalSandboxRunnerBackend(
        runner_path=runner_path,
        expected_runner_sha256=runner_sha256,
        expected_runtime_digest=runtime_digest,
        expected_artifact_lock_sha256=_required_sha256(
            "OPENMLE_FAST_RUNTIME_ARTIFACT_LOCK_SHA256"
        ),
        limits=limits,
    )

    grader_timeout = _required_float("OPENMLE_FAST_GRADER_CLIENT_TIMEOUT_SECONDS")
    grader_margin = _required_float("OPENMLE_FAST_GRADER_TIMEOUT_MARGIN_SECONDS")
    client_timeout = _required_float("OPENMLE_FAST_CLIENT_TIMEOUT_SECONDS")
    client_margin = _required_float("OPENMLE_FAST_CLIENT_TIMEOUT_MARGIN_SECONDS")
    _validate_timeout_margins(
        limits=limits,
        grader_timeout=grader_timeout,
        grader_margin=grader_margin,
        client_timeout=client_timeout,
        client_margin=client_margin,
    )
    grader = PrivateGraderClient(
        endpoint=Path(_required_text("OPENMLE_FAST_GRADER_ENDPOINT")),
        credential_path=_required_file("OPENMLE_FAST_GRADER_CREDENTIAL"),
        timeout_seconds=grader_timeout,
    )

    def executor_factory() -> OpenMLEFastExecutor:
        return OpenMLEFastExecutor(limits=limits, backend=backend)

    return OpenMLEFastEpisodeManager(
        dataset=dataset,
        materializer=OpenMLEFastWorkspaceMaterializer(
            episodes_root,
            runner_workspace_parent=backend.metadata.get("workspace_parent"),
            workspace_bytes=limits.workspace_bytes,
            max_files=limits.max_files,
        ),
        executor_factory=executor_factory,
        grader_client=grader,
        limits=limits,
        audit_sink=OpenMLEFastAuditSink(
            _required_directory("OPENMLE_FAST_AUDIT_ROOT", create=True)
        ),
        runtime_metadata={
            "runtime_source": {
                "outer_commit": _required_revision("OPENMLE_FAST_RUNTIME_OUTER_COMMIT"),
                "inner_commit": _required_revision("OPENMLE_FAST_RUNTIME_INNER_COMMIT"),
            },
            "executor_runtime_digest": runtime_digest,
            "max_observation_tokens": _required_integer(
                "OPENMLE_FAST_MAX_OBSERVATION_TOKENS"
            ),
            "implementation_digests": {
                "materializer_sha256": _required_sha256(
                    "OPENMLE_FAST_MATERIALIZER_SHA256"
                ),
                "actions_sha256": _required_sha256("OPENMLE_FAST_ACTIONS_SHA256"),
            },
        },
    )


def build_private_grader_from_environment() -> PrivateGraderService:
    limits = _limits_from_environment()
    runtime_digest = _required_runtime_digest("OPENMLE_FAST_PRIVATE_RUNTIME_DIGEST")
    if _required_integer("OPENMLE_FAST_PRIVATE_CPU_VCPUS") != limits.grader_cpu_vcpus:
        raise RuntimeError("private grader CPU cap differs from frozen v1")
    if (
        _required_integer("OPENMLE_FAST_PRIVATE_MEMORY_BYTES")
        != limits.grader_memory_bytes
    ):
        raise RuntimeError("private grader memory cap differs from frozen v1")
    if (
        _required_integer("OPENMLE_FAST_PRIVATE_MAX_PROCESSES")
        != limits.grader_max_processes
    ):
        raise RuntimeError("private grader PID cap differs from frozen v1")
    if (
        _required_integer("OPENMLE_FAST_PRIVATE_WORKER_WALL_MS")
        != limits.grader_worker_wall_ms
    ):
        raise RuntimeError("private grader worker wall cap differs from frozen v1")
    if (
        _required_integer("OPENMLE_FAST_PRIVATE_TOTAL_WALL_MS")
        != limits.grader_total_wall_ms
    ):
        raise RuntimeError("private grader total wall cap differs from frozen v1")
    if (
        _required_integer("OPENMLE_FAST_PRIVATE_MAX_CONCURRENT_REQUESTS")
        != limits.grader_max_concurrent_requests
    ):
        raise RuntimeError("private grader concurrency differs from frozen v1")
    private_limits = PrivateGraderLimits(
        cpu_vcpus=limits.grader_cpu_vcpus,
        memory_bytes=limits.grader_memory_bytes,
        max_processes=limits.grader_max_processes,
        wall_ms=limits.grader_worker_wall_ms,
        input_bytes=limits.grader_input_bytes,
    )
    backend = ExternalPrivateGraderRunnerBackend(
        runner_path=_required_file("OPENMLE_FAST_PRIVATE_RUNNER"),
        expected_runner_sha256=_required_sha256("OPENMLE_FAST_PRIVATE_RUNNER_SHA256"),
        expected_runtime_digest=runtime_digest,
        expected_artifact_lock_sha256=_required_sha256(
            "OPENMLE_FAST_RUNTIME_ARTIFACT_LOCK_SHA256"
        ),
        limits=private_limits,
    )
    return PrivateGraderService(
        private_manifest_path=_required_file("OPENMLE_FAST_PRIVATE_TASK_MANIFEST"),
        expected_manifest_sha256=_required_sha256(
            "OPENMLE_FAST_PRIVATE_TASK_MANIFEST_SHA256"
        ),
        package_root=_required_directory("OPENMLE_FAST_PRIVATE_PACKAGE_ROOT"),
        archive_root=_required_directory("OPENMLE_FAST_PRIVATE_ARCHIVE_ROOT"),
        expected_release_revision=_required_revision("OPENMLE_FAST_RELEASE_REVISION"),
        expected_runtime_digest=runtime_digest,
        socket_path=Path(_required_text("OPENMLE_FAST_GRADER_ENDPOINT")),
        credential_path=_required_file("OPENMLE_FAST_GRADER_CREDENTIAL"),
        audit_root=_required_directory("OPENMLE_FAST_PRIVATE_AUDIT_ROOT", create=True),
        total_wall_ms=limits.grader_total_wall_ms,
        max_concurrent_requests=limits.grader_max_concurrent_requests,
        backend=backend,
        max_submission_bytes=limits.grader_input_bytes,
    )


def launch() -> None:
    import uvicorn

    from .server import app, configure

    configure(build_manager_from_environment())
    uvicorn.run(
        app,
        host=_required_text("OPENMLE_FAST_HOST"),
        port=_required_integer("OPENMLE_FAST_PORT"),
        log_level=os.environ.get("OPENMLE_FAST_LOG_LEVEL", "info"),
    )


def launch_private_grader() -> None:
    service = build_private_grader_from_environment()
    try:
        service.serve_forever()
    finally:
        service.shutdown()


_LIMIT_ENVIRONMENT = {
    "max_policy_actions": "OPENMLE_FAST_MAX_POLICY_TURNS",
    "cpu_vcpus": "OPENMLE_FAST_CPU_VCPUS",
    "memory_bytes": "OPENMLE_FAST_MEMORY_BYTES",
    "swap_bytes": "OPENMLE_FAST_SWAP_BYTES",
    "workspace_bytes": "OPENMLE_FAST_WORKSPACE_BYTES",
    "tmp_bytes": "OPENMLE_FAST_TMP_BYTES",
    "max_processes": "OPENMLE_FAST_MAX_PROCESSES",
    "max_open_files": "OPENMLE_FAST_MAX_OPEN_FILES",
    "max_files": "OPENMLE_FAST_MAX_FILES",
    "max_file_bytes": "OPENMLE_FAST_MAX_FILE_BYTES",
    "max_submission_bytes": "OPENMLE_FAST_MAX_SUBMISSION_BYTES",
    "shell_wall_ms": "OPENMLE_FAST_SHELL_WALL_MS",
    "managed_runtime_per_action_ms": "OPENMLE_FAST_MANAGED_ACTION_MS",
    "managed_runtime_per_episode_ms": "OPENMLE_FAST_MANAGED_EPISODE_MS",
    "episode_wall_ms": "OPENMLE_FAST_EPISODE_WALL_MS",
    "grader_cpu_vcpus": "OPENMLE_FAST_GRADER_CPU_VCPUS",
    "grader_memory_bytes": "OPENMLE_FAST_GRADER_MEMORY_BYTES",
    "grader_max_processes": "OPENMLE_FAST_GRADER_MAX_PROCESSES",
    "grader_worker_wall_ms": "OPENMLE_FAST_GRADER_WORKER_WALL_MS",
    "grader_total_wall_ms": "OPENMLE_FAST_GRADER_TOTAL_WALL_MS",
    "grader_max_concurrent_requests": "OPENMLE_FAST_GRADER_MAX_CONCURRENT_REQUESTS",
    "grader_input_bytes": "OPENMLE_FAST_GRADER_INPUT_BYTES",
    "raw_output_bytes": "OPENMLE_FAST_RAW_OUTPUT_BYTES",
    "observation_bytes": "OPENMLE_FAST_OBSERVATION_BYTES",
    "observation_head_bytes": "OPENMLE_FAST_OBSERVATION_HEAD_BYTES",
    "observation_tail_bytes": "OPENMLE_FAST_OBSERVATION_TAIL_BYTES",
}


def _required_manifest_role() -> str:
    role = _required_text("OPENMLE_FAST_MANIFEST_ROLE")
    if role not in ALLOWED_MANIFEST_ROLES:
        raise RuntimeError("OpenMLE-fast manifest role is not executable")
    return role


def _limits_from_environment() -> OpenMLEFastResourceLimits:
    values = {
        field: _required_nonnegative_integer(name)
        for field, name in _LIMIT_ENVIRONMENT.items()
    }
    limits = OpenMLEFastResourceLimits(**values)
    if limits != OpenMLEFastResourceLimits.frozen_v1():
        raise RuntimeError(
            "configured OpenMLE-fast limits differ from the frozen v1 contract"
        )
    return limits


def _validate_timeout_margins(
    *,
    limits: OpenMLEFastResourceLimits,
    grader_timeout: float,
    grader_margin: float,
    client_timeout: float,
    client_margin: float,
) -> None:
    values = (grader_timeout, grader_margin, client_timeout, client_margin)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError(
            "OpenMLE-fast timeouts and margins must be finite and positive"
        )
    if grader_timeout <= limits.grader_total_wall_ms / 1000.0 + grader_margin:
        raise RuntimeError(
            "grader client timeout must exceed the total grader wall plus margin"
        )
    if client_timeout <= limits.episode_wall_ms / 1000.0 + client_margin:
        raise RuntimeError(
            "PPO client timeout must exceed the episode-capped step path plus margin"
        )


def _required_text(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value.strip()


def _required_integer(name: str) -> int:
    value = _required_nonnegative_integer(name)
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _required_nonnegative_integer(name: str) -> int:
    raw = _required_text(name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative")
    return value


def _required_float(name: str) -> float:
    raw = _required_text(name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be finite and positive")
    return value


def _required_file(name: str) -> Path:
    path = Path(_required_text(name)).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must name a real file")
    return path.resolve()


def _required_directory(name: str, *, create: bool = False) -> Path:
    path = Path(_required_text(name)).expanduser().absolute()
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{name} must name a real directory")
    return path.resolve()


def _required_sha256(name: str) -> str:
    value = _required_text(name).lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"{name} must be a SHA256 digest")
    return value


def _required_runtime_digest(name: str) -> str:
    value = _required_text(name).lower()
    if (
        not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RuntimeError(f"{name} must be sha256:<64 hex>")
    return value


def _required_revision(name: str) -> str:
    value = _required_text(name).lower()
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"{name} must be a full Git revision")
    return value


def _verify_implementation_digest(path: Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} implementation SHA256 mismatch")
