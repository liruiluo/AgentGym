from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_CGROUP_CONTROLLERS = ("cpu", "memory", "pids")
_SUPPORTED_CGROUP_VERSIONS = {"v1", "hybrid", "v2"}


def exact_runtime_identity_is_attested(
    metadata: Mapping[str, Any],
    *,
    expected_artifact_lock_sha256: str,
) -> bool:
    """Return whether live cgroup/admission/transitive identity is exact."""

    if not _is_sha256(expected_artifact_lock_sha256):
        return False
    if not versioned_cgroups_are_attested(metadata):
        return False
    active = metadata.get("active_verification")
    if (
        not isinstance(active, Mapping)
        or active.get("admission_stamp_valid") is not True
        or active.get("all_checks_pass") is not True
    ):
        return False
    identity = metadata.get("artifact_identity")
    return bool(
        isinstance(identity, Mapping)
        and identity.get("artifact_lock_sha256") == expected_artifact_lock_sha256
        and identity.get("artifact_lock_expected_sha256")
        == expected_artifact_lock_sha256
    )


def versioned_cgroups_are_attested(metadata: Mapping[str, Any]) -> bool:
    version = metadata.get("cgroup_version")
    if version not in _SUPPORTED_CGROUP_VERSIONS:
        return False
    controller_attestation = metadata.get("cgroup_controller_attestation")
    if (
        not isinstance(controller_attestation, Mapping)
        or controller_attestation.get("version") != version
    ):
        return False
    selected = "v1" if version in {"v1", "hybrid"} else "v2"
    rejected = "v2" if selected == "v1" else "v1"
    return all(
        metadata.get(f"cgroup_{selected}_{controller}") is True
        and metadata.get(f"cgroup_{rejected}_{controller}") is False
        for controller in _CGROUP_CONTROLLERS
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
