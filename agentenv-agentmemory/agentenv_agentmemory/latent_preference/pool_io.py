from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .schema import (
    POOL_SCHEMA,
    CertifiedPreferenceProduct,
    LatentPreferenceDataError,
    PreferenceProductPool,
    PreferenceRecipe,
    require_sha256,
)


def load_preference_product_pool(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PreferenceProductPool:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LatentPreferenceDataError(
            f"certified preference product pool is not a file: {resolved}"
        )
    require_sha256(expected_file_sha256, field="expected_file_sha256")
    payload_bytes = resolved.read_bytes()
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha256 != expected_file_sha256:
        raise LatentPreferenceDataError(
            "certified preference product pool SHA256 mismatch: "
            f"expected {expected_file_sha256}, observed {observed_sha256}."
        )
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LatentPreferenceDataError(
            f"certified preference product pool is not valid UTF-8 JSON: {resolved}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LatentPreferenceDataError("preference product pool must be an object.")
    required = {
        "schema",
        "pool_id",
        "certifier_version",
        "products_per_cell",
        "recipes",
        "products",
        "provenance",
    }
    if set(payload) != required:
        raise LatentPreferenceDataError(
            "preference product pool fields mismatch: "
            f"missing={sorted(required - set(payload))} "
            f"extra={sorted(set(payload) - required)}."
        )
    if payload["schema"] != POOL_SCHEMA:
        raise LatentPreferenceDataError(
            f"unsupported preference pool schema {payload['schema']!r}."
        )
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "catalog_sha256",
        "attributes_sha256",
        "price_table_sha256",
        "lucene_index_sha256",
        "candidate_artifact_sha256",
        "rules_sha256",
        "source_manifest_sha256",
    }:
        raise LatentPreferenceDataError("preference pool provenance fields mismatch.")
    recipes = payload["recipes"]
    products = payload["products"]
    if not isinstance(recipes, list) or not isinstance(products, list):
        raise LatentPreferenceDataError("recipes and products must be JSON lists.")
    return PreferenceProductPool(
        pool_id=payload["pool_id"],
        certifier_version=payload["certifier_version"],
        products_per_cell=payload["products_per_cell"],
        recipes=tuple(PreferenceRecipe.from_dict(value) for value in recipes),
        products=tuple(CertifiedPreferenceProduct.from_dict(value) for value in products),
        catalog_sha256=provenance["catalog_sha256"],
        attributes_sha256=provenance["attributes_sha256"],
        price_table_sha256=provenance["price_table_sha256"],
        lucene_index_sha256=provenance["lucene_index_sha256"],
        candidate_artifact_sha256=provenance["candidate_artifact_sha256"],
        rules_sha256=provenance["rules_sha256"],
        source_manifest_sha256=provenance["source_manifest_sha256"],
    )


def write_preference_product_pool_manifest(
    pool: PreferenceProductPool,
    path: str | Path,
) -> str:
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
