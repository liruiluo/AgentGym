from __future__ import annotations

import os
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import ShellSandboxLimits

from .audit import SwesmithEpisodeAuditSink
from .dataset import SwesmithDataset
from .environment import (
    DEFAULT_MAX_OBSERVATION_BYTES,
    DEFAULT_TRAINING_MAX_POLICY_TURNS,
    SwesmithEpisodeManager,
)
from .grader import SwesmithHiddenGrader
from .image_manifest import SwesmithImageManifest
from .profile import OfficialSwesmithProfileResolver
from .provenance import validate_revision_binding
from .sandbox import LinuxNamespaceEpisodeSandbox
from .workspace import SwesmithWorkspaceMaterializer


def build_manager_from_environment() -> SwesmithEpisodeManager:
    dataset = SwesmithDataset(_required_path("SWESMITH_DATASET_MANIFEST", file=True))
    images = SwesmithImageManifest(_required_path("SWESMITH_IMAGE_MANIFEST", file=True))
    source_root = _required_path("SWESMITH_SOURCE_ROOT", file=False)
    dataset_revision = dataset.provenance.upstream_revision
    source_revision = _required_revision("SWESMITH_SOURCE_REVISION")
    validate_revision_binding(
        dataset_revision=dataset_revision,
        source_revision=source_revision,
        image_dataset_revision=images.dataset_revision,
        image_source_revision=images.source_revision,
    )
    profile_resolver = OfficialSwesmithProfileResolver(
        source_root=source_root,
        expected_revision=source_revision,
    )
    materializer = SwesmithWorkspaceMaterializer(
        mirrors_root=_required_path("SWESMITH_MIRRORS_ROOT", file=False),
        episodes_root=_required_path("SWESMITH_EPISODES_ROOT", file=False, create=True),
    )
    max_observation_tokens = _integer("SWESMITH_MAX_OBSERVATION_TOKENS", 8192)
    max_observation_bytes = _integer(
        "SWESMITH_MAX_OBSERVATION_BYTES",
        min(DEFAULT_MAX_OBSERVATION_BYTES, max_observation_tokens - 1),
    )
    if max_observation_bytes >= max_observation_tokens:
        raise ValueError(
            "SWESMITH_MAX_OBSERVATION_BYTES must stay below the token budget"
        )
    limits = _limits_from_environment(max_observation_bytes)
    oci_cache_root = _required_path("SWESMITH_OCI_CACHE_ROOT", file=False)
    rg_binary = _required_path("SWESMITH_RG_BINARY", file=True)
    rg_sha256 = _required_text("SWESMITH_RG_SHA256")
    lease_root_raw = os.environ.get("SWESMITH_UID_LEASE_ROOT")
    lease_root = None if not lease_root_raw else Path(lease_root_raw).expanduser().resolve()
    audit_root_raw = os.environ.get("SWESMITH_AUDIT_ROOT")
    audit_sink = (
        None
        if not audit_root_raw
        else SwesmithEpisodeAuditSink(Path(audit_root_raw).expanduser())
    )
    runtime_source = _runtime_source_from_environment()

    def sandbox_factory(record, profile):
        del record
        image = images.resolve(profile.image)
        return LinuxNamespaceEpisodeSandbox.from_environment(
            limits=limits,
            rg_binary=rg_binary,
            expected_rg_sha256=rg_sha256,
            oci_cache_root=oci_cache_root,
            repo_profile_image=image.image,
            repo_profile_digest=image.digest,
            lease_root=lease_root,
            lease_slots=_integer("SWESMITH_UID_LEASE_SLOTS", 4096),
            run_preflight=True,
        )

    grader_timeout = _integer("SWESMITH_GRADER_TIMEOUT_MS", limits.max_timeout_ms)
    if grader_timeout > limits.max_timeout_ms:
        raise ValueError("SWESMITH_GRADER_TIMEOUT_MS exceeds sandbox max_timeout_ms")
    return SwesmithEpisodeManager(
        dataset=dataset,
        materializer=materializer,
        profile_resolver=profile_resolver,
        sandbox_factory=sandbox_factory,
        grader=SwesmithHiddenGrader(timeout_ms=grader_timeout),
        audit_sink=audit_sink,
        max_steps=_integer(
            "SWESMITH_MAX_STEPS",
            DEFAULT_TRAINING_MAX_POLICY_TURNS,
        ),
        max_observation_bytes=max_observation_bytes,
        runtime_metadata={
            "image_manifest": images.public_metadata(),
            "dataset_revision": dataset_revision,
            "source_revision": source_revision,
            "sandbox_contract": "swesmith_linux_namespace_oci_rootfs_v1",
            "profile_contract": "swesmith_official_repo_profile_v1",
            "max_observation_tokens": max_observation_tokens,
            "max_observation_bytes": max_observation_bytes,
            **runtime_source,
        },
    )


def launch() -> None:
    import uvicorn

    from .server import app, configure

    configure(build_manager_from_environment())
    uvicorn.run(
        app,
        host=os.environ.get("SWESMITH_HOST", "127.0.0.1"),
        port=_integer("SWESMITH_PORT", 8000),
        log_level=os.environ.get("SWESMITH_LOG_LEVEL", "info"),
    )


def _limits_from_environment(max_observation_bytes: int | None = None) -> ShellSandboxLimits:
    gib = 1024**3
    mib = 1024**2
    if max_observation_bytes is None:
        max_observation_bytes = _integer(
            "SWESMITH_MAX_OBSERVATION_BYTES",
            DEFAULT_MAX_OBSERVATION_BYTES,
        )
    if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
        raise ValueError("max_observation_bytes must be a positive integer")
    stdout_bytes = _integer("SWESMITH_STDOUT_BYTES", max_observation_bytes // 2)
    stderr_bytes = _integer(
        "SWESMITH_STDERR_BYTES",
        max_observation_bytes - stdout_bytes,
    )
    if stdout_bytes + stderr_bytes > max_observation_bytes:
        raise ValueError(
            "SWE-smith stdout/stderr caps must fit the combined observation budget"
        )
    return ShellSandboxLimits(
        workspace_bytes=_integer("SWESMITH_WORKSPACE_BYTES", 2 * gib),
        workspace_inodes=_integer("SWESMITH_WORKSPACE_INODES", 250_000),
        max_files=_integer("SWESMITH_MAX_FILES", 200_000),
        max_directories=_integer("SWESMITH_MAX_DIRECTORIES", 50_000),
        max_file_bytes=_integer("SWESMITH_MAX_FILE_BYTES", 256 * mib),
        max_path_chars=_integer("SWESMITH_MAX_PATH_CHARS", 1024),
        default_timeout_ms=_integer("SWESMITH_DEFAULT_TIMEOUT_MS", 120_000),
        max_timeout_ms=_integer("SWESMITH_MAX_TIMEOUT_MS", 900_000),
        cpu_seconds=_integer("SWESMITH_CPU_SECONDS", 900),
        address_space_bytes=_integer("SWESMITH_ADDRESS_SPACE_BYTES", 32 * gib),
        max_processes=_integer("SWESMITH_MAX_PROCESSES", 512),
        max_open_files=_integer("SWESMITH_MAX_OPEN_FILES", 4096),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        tmp_bytes=_integer("SWESMITH_TMP_BYTES", 4 * gib),
        tmp_inodes=_integer("SWESMITH_TMP_INODES", 100_000),
    )


def _required_path(name: str, *, file: bool, create: bool = False) -> Path:
    path = Path(_required_text(name)).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    valid = path.is_file() and not path.is_symlink() if file else path.is_dir() and not path.is_symlink()
    if not valid:
        kind = "file" if file else "directory"
        raise RuntimeError(f"{name} must name a real {kind}: {path}")
    return path


def _required_text(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value.strip()


def _required_revision(name: str) -> str:
    value = _required_text(name).lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{name} must be a full 40-character Git commit")
    return value


def _runtime_source_from_environment() -> dict[str, object]:
    names = ("SWESMITH_RUNTIME_OUTER_COMMIT", "SWESMITH_RUNTIME_INNER_COMMIT")
    present = [bool(os.environ.get(name, "").strip()) for name in names]
    if any(present) and not all(present):
        raise RuntimeError("SWE-smith runtime outer/inner commits must be set together")
    if not all(present):
        return {}
    outer = _required_revision(names[0])
    inner = _required_revision(names[1])
    return {
        "runtime_source": {
            "outer_commit": outer,
            "inner_commit": inner,
            "source_id": f"{outer}_{inner}",
        }
    }


def _integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
