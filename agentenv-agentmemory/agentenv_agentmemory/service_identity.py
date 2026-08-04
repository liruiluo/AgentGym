from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping


SERVICE_IDENTITY_SCHEMA = "agentmemory_service_identity_v1"
SERVICE_ROLES = ("formal", "smoke", "intervention_eval")


def decorate_service_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a stable runtime identity without hashing mutable session counts."""

    role = os.environ.get("AGENTMEMORY_SERVICE_ROLE", "formal")
    if role not in SERVICE_ROLES:
        raise RuntimeError(
            "AGENTMEMORY_SERVICE_ROLE must be one of: " + ", ".join(SERVICE_ROLES)
        )
    source_id = os.environ.get("AGENTMEMORY_RUNTIME_SOURCE_ID", "").strip()
    if role in {"smoke", "intervention_eval"} and not source_id:
        raise RuntimeError(
            f"A {role} service requires AGENTMEMORY_RUNTIME_SOURCE_ID so clients "
            "cannot reuse stale code."
        )

    fingerprint_payload = _fingerprint_payload(
        metadata,
        role=role,
        source_id=source_id,
    )
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    service = {
        "schema": SERVICE_IDENTITY_SCHEMA,
        "role": role,
        "runtime_source_id": source_id or None,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "instance_run_id": os.environ.get("AGENTMEMORY_RUN_ID"),
    }
    return {**dict(metadata), "service": service}


def _fingerprint_payload(
    metadata: Mapping[str, Any],
    *,
    role: str,
    source_id: str,
) -> dict[str, Any]:
    backend = metadata.get("backend")
    if not isinstance(backend, Mapping):
        backend = {}
    return {
        "schema": SERVICE_IDENTITY_SCHEMA,
        "role": role,
        "runtime_source_id": source_id,
        "surface": metadata.get("surface"),
        "memoryarena_base_commit": os.environ.get("MEMORYARENA_BASE_COMMIT"),
        "provider": metadata.get("provider"),
        "runtime_inputs": metadata.get("runtime_inputs"),
        "dataset_provenance": metadata.get("dataset_provenance"),
        "annotation_gate_sha256": metadata.get("annotation_gate_sha256"),
        "reward_contract": metadata.get("reward_contract"),
        "ltm_inventory_mode": metadata.get("ltm_inventory_mode"),
        "ltm_transition_notice_mode": metadata.get("ltm_transition_notice_mode"),
        "action_listing_mode": metadata.get("action_listing_mode"),
        "memory_prompt_mode": metadata.get("memory_prompt_mode"),
        "backend": {
            "surface": backend.get("surface"),
            "price_seed": backend.get("price_seed"),
            "product_count": backend.get("product_count"),
            "price_table_sha256": backend.get("price_table_sha256"),
            "upstream_provenance": backend.get("upstream_provenance"),
        },
    }
