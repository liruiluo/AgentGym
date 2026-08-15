from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentenv_swesmith.sandbox import (
    LinuxNamespaceEpisodeSandbox,
    SwesmithSandboxError,
)


SANDBOX_CONTRACT = "swebench_verified_linux_namespace_oci_rootfs_v1"
_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
MAX_OCI_CACHE_METADATA_BYTES = 1024 * 1024


class VerifiedSandboxError(SwesmithSandboxError):
    pass


class VerifiedLinuxNamespaceEpisodeSandbox(LinuxNamespaceEpisodeSandbox):
    """Verified profile wrapper over the shared namespace/rootfs executor."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        metadata = dict(super().metadata)
        metadata["shared_executor_contract"] = metadata["contract"]
        metadata["contract"] = SANDBOX_CONTRACT
        metadata["policy_root"] = "/testbed"
        requested = getattr(self, "_verified_instance_image_key", None)
        if requested is not None:
            identity = self._oci_rootfs_identity
            if identity is None:
                raise VerifiedSandboxError("Verified image identity is unavailable")
            metadata["verified_instance_image_key"] = requested
            metadata["verified_cache_image_key"] = identity.image
            metadata["verified_image_alias"] = requested != identity.image
        return metadata

    def bind_verified_image(self, *, instance_image_key: str, digest: str) -> None:
        identity = self._oci_rootfs_identity
        if identity is None or identity.digest != digest:
            raise VerifiedSandboxError("Verified image digest binding disagrees")
        self._verified_instance_image_key = instance_image_key

    def preflight(self) -> None:
        self._require_open()
        with tempfile.TemporaryDirectory(
            prefix="swebench-verified-sandbox-preflight-"
        ) as raw:
            workspace = Path(raw) / "workspace"
            workspace.mkdir(mode=0o700)
            workspace.chmod(0o700)
            os.chown(workspace, self.model_uid, self.model_gid)
            result = self._run_namespace(
                workspace,
                command=(
                    'test "$(command -v rg)" = /run/tools/rg && '
                    "rg --version >/dev/null && "
                    "printf SWEBENCH_VERIFIED_SANDBOX_OK > proof && cat proof"
                ),
                workdir=".",
                timeout_ms=min(10_000, self.limits.max_timeout_ms),
            )
            proof = workspace / "proof"
            if (
                result.exit_code != 0
                or result.timed_out
                or result.stdout != b"SWEBENCH_VERIFIED_SANDBOX_OK"
                or result.stderr
                or not proof.is_file()
                or proof.read_bytes() != b"SWEBENCH_VERIFIED_SANDBOX_OK"
            ):
                raise VerifiedSandboxError(
                    "Verified OCI-rootfs sandbox preflight failed"
                )


def resolve_cached_profile_image(
    cache_root: Path | str,
    *,
    digest: str,
    allowed_images: Sequence[str],
) -> str:
    aliases = tuple(allowed_images)
    if (
        not aliases
        or len(set(aliases)) != len(aliases)
        or any(not isinstance(value, str) or not value for value in aliases)
    ):
        raise VerifiedSandboxError("Verified image aliases must be unique text")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise VerifiedSandboxError("Verified image digest must be sha256")
    root = Path(cache_root).expanduser()
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise VerifiedSandboxError("Verified OCI cache root is unavailable") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise VerifiedSandboxError("Verified OCI cache root must be a real directory")
    metadata_path = (
        root
        / f"sha256-{digest.removeprefix('sha256:')}"
        / "metadata.json"
    )
    try:
        info = metadata_path.lstat()
    except OSError as exc:
        raise VerifiedSandboxError("Verified OCI cache metadata is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or info.st_size > MAX_OCI_CACHE_METADATA_BYTES
    ):
        raise VerifiedSandboxError("Verified OCI cache metadata is not a regular file")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerifiedSandboxError("Verified OCI cache metadata is invalid") from exc
    if not isinstance(metadata, Mapping):
        raise VerifiedSandboxError("Verified OCI cache metadata must be an object")
    stored_image = metadata.get("repo_profile_image")
    if stored_image not in aliases:
        raise VerifiedSandboxError(
            "Verified OCI cache image is not an allowed digest alias"
        )
    return stored_image
