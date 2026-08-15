from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import load_lite_dataset
from .environment import MLEBenchLiteEpisodeManager, resource_contract_sha256
from .executor import ExternalSandboxRunnerBackend, SandboxExecutor
from .identity import load_official_lite_identity
from .workspace import WorkspaceManager

RUNTIME_CONFIG_SCHEMA = "mlebench_lite_runtime_config_v2"


class MLEBenchLiteConfigError(RuntimeError):
    """The external runtime configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class MLEBenchLiteRuntimeConfig:
    upstream_root: Path
    data_root: Path
    public_manifest_path: Path
    public_manifest_sha256: str
    episodes_root: Path
    handoff_root: Path
    sandbox_runner_path: Path
    sandbox_runner_sha256: str
    sandbox_runtime_digest: str
    sandbox_runner_uid: int
    max_actions: int
    max_submission_bytes: int
    max_shell_timeout_ms: int
    episode_timeout_ms: int
    max_total_execution_ms: int
    cpu_limit_cores: int
    memory_limit_bytes: int
    pids_limit: int
    writable_bytes_limit: int
    writable_inodes_limit: int
    gpu_count: int
    forbidden_roots: tuple[Path, ...]


def load_runtime_config(path: Path) -> MLEBenchLiteRuntimeConfig:
    config_path = _existing_path(Path(path), kind="file", label="runtime config")
    try:
        value = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MLEBenchLiteConfigError("runtime config is not strict JSON") from exc
    expected_keys = {
        "schema",
        "upstream_root",
        "data_root",
        "public_manifest_path",
        "public_manifest_sha256",
        "episodes_root",
        "handoff_root",
        "sandbox_runner_path",
        "sandbox_runner_sha256",
        "sandbox_runtime_digest",
        "sandbox_runner_uid",
        "max_actions",
        "max_submission_bytes",
        "max_shell_timeout_ms",
        "episode_timeout_ms",
        "max_total_execution_ms",
        "cpu_limit_cores",
        "memory_limit_bytes",
        "pids_limit",
        "writable_bytes_limit",
        "writable_inodes_limit",
        "gpu_count",
        "forbidden_roots",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise MLEBenchLiteConfigError("runtime config fields drifted")
    if value["schema"] != RUNTIME_CONFIG_SCHEMA:
        raise MLEBenchLiteConfigError("runtime config schema drifted")

    upstream_root = _configured_path(
        value["upstream_root"], kind="directory", label="upstream root"
    )
    data_root = _configured_path(
        value["data_root"], kind="directory", label="prepared data root"
    )
    public_manifest_path = _configured_path(
        value["public_manifest_path"], kind="file", label="public manifest"
    )
    episodes_root = _configured_output_directory(
        value["episodes_root"], label="episodes root"
    )
    handoff_root = _configured_output_directory(
        value["handoff_root"], label="handoff root"
    )
    sandbox_runner_path = _configured_path(
        value["sandbox_runner_path"], kind="file", label="sandbox runner"
    )
    forbidden_value = value["forbidden_roots"]
    if not isinstance(forbidden_value, list):
        raise MLEBenchLiteConfigError("forbidden_roots must be a list")
    forbidden_roots = tuple(
        _configured_path(item, kind="directory", label="forbidden root")
        for item in forbidden_value
    )
    if len(set(forbidden_roots)) != len(forbidden_roots):
        raise MLEBenchLiteConfigError("forbidden roots contain duplicates")

    _require_sha256(value["public_manifest_sha256"], "public manifest SHA256")
    _require_sha256(value["sandbox_runner_sha256"], "sandbox runner SHA256")
    _require_sha256(value["sandbox_runtime_digest"], "sandbox runtime digest")
    sandbox_runner_uid = _nonnegative_int(
        value["sandbox_runner_uid"], "sandbox_runner_uid"
    )
    if _file_sha256(public_manifest_path) != value["public_manifest_sha256"]:
        raise MLEBenchLiteConfigError("public manifest SHA256 mismatch")
    if _file_sha256(sandbox_runner_path) != value["sandbox_runner_sha256"]:
        raise MLEBenchLiteConfigError("sandbox runner SHA256 mismatch")
    max_actions = _positive_int(value["max_actions"], "max_actions")
    max_submission_bytes = _positive_int(
        value["max_submission_bytes"], "max_submission_bytes"
    )
    max_shell_timeout_ms = _positive_int(
        value["max_shell_timeout_ms"], "max_shell_timeout_ms"
    )
    episode_timeout_ms = _positive_int(
        value["episode_timeout_ms"], "episode_timeout_ms"
    )
    max_total_execution_ms = _positive_int(
        value["max_total_execution_ms"], "max_total_execution_ms"
    )
    cpu_limit_cores = _positive_int(value["cpu_limit_cores"], "cpu_limit_cores")
    memory_limit_bytes = _positive_int(
        value["memory_limit_bytes"], "memory_limit_bytes"
    )
    pids_limit = _positive_int(value["pids_limit"], "pids_limit")
    writable_bytes_limit = _positive_int(
        value["writable_bytes_limit"], "writable_bytes_limit"
    )
    writable_inodes_limit = _positive_int(
        value["writable_inodes_limit"], "writable_inodes_limit"
    )
    gpu_count = _positive_int(value["gpu_count"], "gpu_count")
    if max_shell_timeout_ms > episode_timeout_ms:
        raise MLEBenchLiteConfigError("shell timeout exceeds episode deadline")
    if max_total_execution_ms > episode_timeout_ms:
        raise MLEBenchLiteConfigError("execution budget exceeds episode deadline")
    if max_submission_bytes > writable_bytes_limit:
        raise MLEBenchLiteConfigError("submission exceeds writable-byte budget")

    protected_directories = (
        upstream_root,
        data_root,
        episodes_root,
        handoff_root,
        *forbidden_roots,
    )
    for index, first in enumerate(protected_directories):
        for second in protected_directories[index + 1 :]:
            _require_disjoint(first, second)
    for protected_file in (public_manifest_path, sandbox_runner_path):
        if any(
            _is_relative_to(protected_file, root)
            for root in (data_root, episodes_root, handoff_root)
        ):
            raise MLEBenchLiteConfigError(
                "runtime file overlaps prepared data or episode output"
            )
    return MLEBenchLiteRuntimeConfig(
        upstream_root=upstream_root,
        data_root=data_root,
        public_manifest_path=public_manifest_path,
        public_manifest_sha256=value["public_manifest_sha256"],
        episodes_root=episodes_root,
        handoff_root=handoff_root,
        sandbox_runner_path=sandbox_runner_path,
        sandbox_runner_sha256=value["sandbox_runner_sha256"],
        sandbox_runtime_digest=value["sandbox_runtime_digest"],
        sandbox_runner_uid=sandbox_runner_uid,
        max_actions=max_actions,
        max_submission_bytes=max_submission_bytes,
        max_shell_timeout_ms=max_shell_timeout_ms,
        episode_timeout_ms=episode_timeout_ms,
        max_total_execution_ms=max_total_execution_ms,
        cpu_limit_cores=cpu_limit_cores,
        memory_limit_bytes=memory_limit_bytes,
        pids_limit=pids_limit,
        writable_bytes_limit=writable_bytes_limit,
        writable_inodes_limit=writable_inodes_limit,
        gpu_count=gpu_count,
        forbidden_roots=forbidden_roots,
    )


def build_manager(config: MLEBenchLiteRuntimeConfig) -> MLEBenchLiteEpisodeManager:
    identity = load_official_lite_identity(config.upstream_root)
    dataset = load_lite_dataset(
        identity=identity,
        manifest_path=config.public_manifest_path,
        expected_manifest_sha256=config.public_manifest_sha256,
        data_root=config.data_root,
        forbidden_roots=config.forbidden_roots,
    )
    backend = ExternalSandboxRunnerBackend(
        config.sandbox_runner_path,
        expected_runner_sha256=config.sandbox_runner_sha256,
        expected_runtime_digest=config.sandbox_runtime_digest,
        expected_runner_uid=config.sandbox_runner_uid,
    )
    expected_resource_contract_sha256 = resource_contract_sha256(
        max_actions=config.max_actions,
        max_submission_bytes=config.max_submission_bytes,
        max_shell_timeout_ms=config.max_shell_timeout_ms,
        episode_timeout_ms=config.episode_timeout_ms,
        max_total_execution_ms=config.max_total_execution_ms,
        cpu_limit_cores=config.cpu_limit_cores,
        memory_limit_bytes=config.memory_limit_bytes,
        pids_limit=config.pids_limit,
        writable_bytes_limit=config.writable_bytes_limit,
        writable_inodes_limit=config.writable_inodes_limit,
        gpu_count=config.gpu_count,
    )

    def executor_factory() -> SandboxExecutor:
        return SandboxExecutor(
            backend,
            expected_runner_sha256=config.sandbox_runner_sha256,
            expected_runtime_digest=config.sandbox_runtime_digest,
            expected_resource_contract_sha256=expected_resource_contract_sha256,
        )

    return MLEBenchLiteEpisodeManager(
        dataset=dataset,
        workspace_manager=WorkspaceManager(config.episodes_root, config.handoff_root),
        executor_factory=executor_factory,
        runner_sha256=config.sandbox_runner_sha256,
        runtime_digest=config.sandbox_runtime_digest,
        max_actions=config.max_actions,
        max_submission_bytes=config.max_submission_bytes,
        max_shell_timeout_ms=config.max_shell_timeout_ms,
        episode_timeout_ms=config.episode_timeout_ms,
        max_total_execution_ms=config.max_total_execution_ms,
        cpu_limit_cores=config.cpu_limit_cores,
        memory_limit_bytes=config.memory_limit_bytes,
        pids_limit=config.pids_limit,
        writable_bytes_limit=config.writable_bytes_limit,
        writable_inodes_limit=config.writable_inodes_limit,
        gpu_count=config.gpu_count,
    )


def _configured_path(value: Any, *, kind: str, label: str) -> Path:
    path = _absolute_path(value, label)
    return _existing_path(path, kind=kind, label=label)


def _configured_output_directory(value: Any, *, label: str) -> Path:
    path = _absolute_path(value, label)
    try:
        path.lstat()
        path_exists = True
    except FileNotFoundError:
        path_exists = False
    except OSError as exc:
        raise MLEBenchLiteConfigError(f"{label} is unavailable") from exc
    if path_exists:
        return _existing_path(path, kind="directory", label=label)
    parent = _existing_path(path.parent, kind="directory", label=f"{label} parent")
    candidate = parent / path.name
    _reject_symlink_components(parent, candidate)
    return candidate


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MLEBenchLiteConfigError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise MLEBenchLiteConfigError(f"{label} must be an absolute path")
    return path


def _existing_path(path: Path, *, kind: str, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MLEBenchLiteConfigError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise MLEBenchLiteConfigError(f"{label} must not be a symlink")
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise MLEBenchLiteConfigError(f"{label} must be a regular file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise MLEBenchLiteConfigError(f"{label} must be a directory")
    resolved = path.resolve(strict=True)
    if Path(os.path.abspath(path)) != resolved:
        raise MLEBenchLiteConfigError(f"{label} path contains a symlink")
    return resolved


def _reject_symlink_components(root: Path, target: Path) -> None:
    # Existing configured roots are already resolved and lstat-checked. For a
    # not-yet-created output, walk only below its resolved existing parent.
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise MLEBenchLiteConfigError("configured path escaped its parent") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise MLEBenchLiteConfigError("configured path is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise MLEBenchLiteConfigError("configured path contains a symlink")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MLEBenchLiteConfigError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MLEBenchLiteConfigError(f"{label} must be a nonnegative integer")
    return value


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MLEBenchLiteConfigError(f"{label} must be lowercase SHA256")


def _require_disjoint(first: Path, second: Path) -> None:
    if (
        first == second
        or _is_relative_to(first, second)
        or _is_relative_to(second, first)
    ):
        raise MLEBenchLiteConfigError("configured roots overlap or nest")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MLEBenchLiteConfigError("configured file cannot be hashed") from exc
    return digest.hexdigest()
