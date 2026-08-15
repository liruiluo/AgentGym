from __future__ import annotations

import os
import stat
from pathlib import Path

from agentenv_agentmemory.workspace_sandbox import ShellSandboxLimits

from .dataset import VerifiedDataset
from .environment import DEFAULT_MAX_OBSERVATION_BYTES, VerifiedEpisodeManager
from .exporter import PredictionStore, SolutionPatchExporter
from .images import VerifiedImageManifest
from .protocol import require_sha256
from .sandbox import (
    SANDBOX_CONTRACT,
    VerifiedLinuxNamespaceEpisodeSandbox,
    resolve_cached_profile_image,
)
from .server import create_http_server
from .testspec import OfficialTestSpecResolver, TESTSPEC_BINDING_CONTRACT
from .workspace import VerifiedWorkspaceMaterializer


ENV_PREFIX = "SWEBENCH_VERIFIED_"


def build_manager_from_environment() -> VerifiedEpisodeManager:
    dataset = VerifiedDataset(required_path("DATASET_MANIFEST", file=True))
    testspec_resolver = OfficialTestSpecResolver(
        source_root=required_path("HARNESS_ROOT", file=False)
    )
    images = VerifiedImageManifest(
        required_path("IMAGE_DIGESTS", file=True),
        expected_manifest_sha256=require_sha256(
            required_text("IMAGE_DIGESTS_SHA256"),
            f"{ENV_PREFIX}IMAGE_DIGESTS_SHA256",
        ),
    )
    materializer = VerifiedWorkspaceMaterializer(
        mirrors_root=required_path("MIRRORS_ROOT", file=False),
        episodes_root=required_path("EPISODES_ROOT", file=False, create=True),
    )
    prediction_store = PredictionStore(
        required_path("PREDICTIONS_ROOT", file=False, create=True),
        instance_ids=dataset.instance_ids,
    )
    max_observation_bytes = integer(
        "MAX_OBSERVATION_BYTES", DEFAULT_MAX_OBSERVATION_BYTES
    )
    max_observation_tokens = integer("MAX_OBSERVATION_TOKENS", 8192)
    if max_observation_bytes >= max_observation_tokens:
        raise ValueError(
            "MAX_OBSERVATION_BYTES must stay below MAX_OBSERVATION_TOKENS"
        )
    limits = limits_from_environment(max_observation_bytes)
    oci_cache_root = required_path("OCI_CACHE_ROOT", file=False)
    rg_binary = required_path("RG_BINARY", file=True)
    rg_sha256 = required_text("RG_SHA256")
    lease_root_raw = os.environ.get(f"{ENV_PREFIX}UID_LEASE_ROOT", "").strip()
    lease_root = (
        None
        if not lease_root_raw
        else Path(lease_root_raw).expanduser().resolve()
    )

    def sandbox_factory(_record, binding):
        digest = images.resolve(binding)
        cache_image = resolve_cached_profile_image(
            oci_cache_root,
            digest=digest,
            allowed_images=images.aliases_for_digest(digest),
        )
        sandbox = VerifiedLinuxNamespaceEpisodeSandbox.from_environment(
            limits=limits,
            rg_binary=rg_binary,
            expected_rg_sha256=rg_sha256,
            oci_cache_root=oci_cache_root,
            repo_profile_image=cache_image,
            repo_profile_digest=digest,
            lease_root=lease_root,
            lease_slots=integer("UID_LEASE_SLOTS", 4096),
            run_preflight=True,
        )
        sandbox.bind_verified_image(
            instance_image_key=binding.instance_image_key,
            digest=digest,
        )
        return sandbox

    return VerifiedEpisodeManager(
        dataset=dataset,
        materializer=materializer,
        testspec_resolver=testspec_resolver,
        sandbox_factory=sandbox_factory,
        exporter=SolutionPatchExporter(),
        prediction_store=prediction_store,
        max_observation_bytes=max_observation_bytes,
        runtime_metadata={
            "testspec_contract": TESTSPEC_BINDING_CONTRACT,
            "sandbox_contract": SANDBOX_CONTRACT,
            "image_manifest": images.public_metadata(),
            "max_observation_tokens": max_observation_tokens,
        },
    )


def launch() -> None:
    manager = build_manager_from_environment()
    server = create_http_server(
        manager,
        host=os.environ.get(f"{ENV_PREFIX}HOST", "127.0.0.1"),
        port=integer("PORT", 8000),
    )
    server.serve_forever()


def limits_from_environment(max_observation_bytes: int) -> ShellSandboxLimits:
    if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
        raise ValueError("max_observation_bytes must be a positive integer")
    gib = 1024**3
    mib = 1024**2
    stdout_bytes = integer("STDOUT_BYTES", max_observation_bytes // 2)
    stderr_bytes = integer(
        "STDERR_BYTES", max_observation_bytes - stdout_bytes
    )
    if stdout_bytes + stderr_bytes > max_observation_bytes:
        raise ValueError("stdout/stderr caps exceed the observation budget")
    return ShellSandboxLimits(
        workspace_bytes=integer("WORKSPACE_BYTES", 2 * gib),
        workspace_inodes=integer("WORKSPACE_INODES", 250_000),
        max_files=integer("MAX_FILES", 200_000),
        max_directories=integer("MAX_DIRECTORIES", 50_000),
        max_file_bytes=integer("MAX_FILE_BYTES", 256 * mib),
        max_path_chars=integer("MAX_PATH_CHARS", 1024),
        default_timeout_ms=integer("DEFAULT_TIMEOUT_MS", 120_000),
        max_timeout_ms=integer("MAX_TIMEOUT_MS", 900_000),
        cpu_seconds=integer("CPU_SECONDS", 900),
        address_space_bytes=integer("ADDRESS_SPACE_BYTES", 32 * gib),
        max_processes=integer("MAX_PROCESSES", 512),
        max_open_files=integer("MAX_OPEN_FILES", 4096),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        tmp_bytes=integer("TMP_BYTES", 4 * gib),
        tmp_inodes=integer("TMP_INODES", 100_000),
    )


def required_path(name: str, *, file: bool, create: bool = False) -> Path:
    path = Path(required_text(name)).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{ENV_PREFIX}{name} must be an absolute path")
    path = Path(os.path.abspath(path))
    if create and path.parent == path:
        raise RuntimeError(f"{ENV_PREFIX}{name} must name a dedicated leaf")

    created = False
    try:
        info = path.lstat()
    except FileNotFoundError:
        if not create:
            kind = "file" if file else "directory"
            raise RuntimeError(
                f"{ENV_PREFIX}{name} must name a real {kind}"
            ) from None
        path.mkdir(parents=True, mode=0o700)
        created = True
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{ENV_PREFIX}{name} is unavailable") from exc

    expected_kind = stat.S_ISREG if file else stat.S_ISDIR
    if stat.S_ISLNK(info.st_mode) or not expected_kind(info.st_mode):
        kind = "file" if file else "directory"
        raise RuntimeError(f"{ENV_PREFIX}{name} must name a real {kind}")
    if create:
        if created:
            os.chmod(path, 0o700, follow_symlinks=False)
        elif stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(f"{ENV_PREFIX}{name} must be a private directory")
    return path.resolve(strict=True)


def required_text(name: str) -> str:
    full_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(full_name)
    if value is None or not value.strip():
        raise RuntimeError(f"required environment variable is unset: {full_name}")
    return value.strip()


def integer(name: str, default: int) -> int:
    full_name = f"{ENV_PREFIX}{name}"
    raw = os.environ.get(full_name)
    if raw is None or not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{full_name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{full_name} must be positive")
    return value


if __name__ == "__main__":
    launch()
