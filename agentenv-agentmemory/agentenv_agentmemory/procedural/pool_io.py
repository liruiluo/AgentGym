from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    PRODUCT_POOL_SCHEMA,
    CertifiedProduct,
    ProceduralMemoryDataError,
    ProductPool,
    require_sha256,
)


def load_certified_product_pool(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> ProductPool:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProceduralMemoryDataError(
            f"certified product pool is not a file: {resolved}"
        )
    require_sha256(expected_file_sha256, field="expected_file_sha256")
    payload_bytes = resolved.read_bytes()
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha256 != expected_file_sha256:
        raise ProceduralMemoryDataError(
            "certified product pool SHA256 mismatch: "
            f"expected {expected_file_sha256}, observed {observed_sha256}."
        )
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProceduralMemoryDataError(
            f"certified product pool is not valid UTF-8 JSON: {resolved}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProceduralMemoryDataError("certified product pool must be an object.")
    required = {
        "schema",
        "pool_id",
        "certifier_version",
        "scenario_definition",
        "products_per_cell",
        "products",
        "provenance",
    }
    observed = set(payload)
    if observed != required:
        raise ProceduralMemoryDataError(
            "certified product pool fields mismatch: "
            f"missing={sorted(required - observed)} extra={sorted(observed - required)}."
        )
    if payload["schema"] != PRODUCT_POOL_SCHEMA:
        raise ProceduralMemoryDataError(
            f"unsupported product pool schema {payload['schema']!r}."
        )
    scenario_definition = payload["scenario_definition"]
    provenance = payload["provenance"]
    products = payload["products"]
    if not isinstance(scenario_definition, Mapping) or set(scenario_definition) != {
        "version",
        "sha256",
        "scenario_ids",
    }:
        raise ProceduralMemoryDataError("scenario definition fields mismatch.")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "catalog_sha256",
        "attributes_sha256",
        "price_table_sha256",
        "lucene_index_sha256",
        "source_manifest_sha256",
    }:
        raise ProceduralMemoryDataError("product pool provenance fields mismatch.")
    if not isinstance(products, list):
        raise ProceduralMemoryDataError("product pool products must be a list.")
    return ProductPool(
        pool_id=payload["pool_id"],
        certifier_version=payload["certifier_version"],
        scenario_ids=tuple(scenario_definition["scenario_ids"]),
        products_per_cell=payload["products_per_cell"],
        products=tuple(CertifiedProduct.from_dict(value) for value in products),
        catalog_sha256=provenance["catalog_sha256"],
        attributes_sha256=provenance["attributes_sha256"],
        price_table_sha256=provenance["price_table_sha256"],
        lucene_index_sha256=provenance["lucene_index_sha256"],
        source_manifest_sha256=provenance["source_manifest_sha256"],
        scenario_definition_version=scenario_definition["version"],
        scenario_definition_sha256=scenario_definition["sha256"],
    )


def write_product_pool_manifest(pool: ProductPool, path: str | Path) -> str:
    """Write a canonical pool manifest and return its file SHA256."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        pool.semantic_manifest(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    resolved.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
