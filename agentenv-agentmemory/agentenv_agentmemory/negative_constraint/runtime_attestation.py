from __future__ import annotations

from pathlib import Path

from ..latent_preference.schema import canonical_sha256
from ..native_webshop_backend import NativeWebShopBackend
from ..procedural import file_sha256, verify_lucene_index_manifest
from .certifier import CERTIFIER_VERSION, source_manifest_for_pool
from .schema import NegativeConstraintProductPool


def attest_negative_constraint_runtime_inputs(
    pool: NegativeConstraintProductPool,
    backend: NativeWebShopBackend,
    *,
    items_file: Path,
    attributes_file: Path,
    search_root: Path,
    lucene_manifest: Path,
) -> None:
    """Fail closed unless the live native runtime matches the certificate."""

    if not pool.native_certified:
        raise RuntimeError("Negative-constraint training refuses a rules-only pool.")
    if pool.certifier_version != CERTIFIER_VERSION:
        raise RuntimeError(
            "Unsupported negative-constraint certifier: "
            f"observed {pool.certifier_version!r}."
        )
    _require_equal_hash(
        "negative source manifest",
        expected=str(pool.source_manifest_sha256),
        observed=canonical_sha256(source_manifest_for_pool(pool)),
    )
    metadata = backend.metadata()
    upstream = metadata.get("upstream_provenance")
    observed_commit = (
        str(upstream.get("memoryarena_commit") or "")
        if isinstance(upstream, dict)
        else ""
    )
    if observed_commit != pool.memoryarena_commit:
        raise RuntimeError(
            "Certified MemoryArena commit mismatch: "
            f"expected {pool.memoryarena_commit}, observed {observed_commit}."
        )
    if metadata.get("price_seed") != pool.price_seed:
        raise RuntimeError(
            "Certified WebShop price seed mismatch: "
            f"expected {pool.price_seed}, observed {metadata.get('price_seed')}."
        )
    _require_equal_hash(
        "product catalog",
        expected=str(pool.catalog_sha256),
        observed=file_sha256(items_file),
    )
    _require_equal_hash(
        "product attributes",
        expected=str(pool.attributes_sha256),
        observed=file_sha256(attributes_file),
    )
    _require_equal_hash(
        "native price table",
        expected=str(pool.price_table_sha256),
        observed=str(metadata.get("price_table_sha256") or ""),
    )
    _require_equal_hash(
        "Lucene index manifest",
        expected=str(pool.lucene_index_sha256),
        observed=file_sha256(lucene_manifest),
    )
    verify_lucene_index_manifest(
        lucene_manifest,
        index_dir=search_root / "indexes-full",
    )

    for candidate in pool.candidates:
        certificate = pool.certificate_for(candidate.asin)
        if not backend.has_product(candidate.asin):
            raise RuntimeError(
                f"Certified negative product disappeared: {candidate.asin}."
            )
        record = backend.product_record(candidate.asin)
        if backend.product_title(candidate.asin) != candidate.title:
            raise RuntimeError(
                f"Certified title no longer matches product {candidate.asin}."
            )
        if str(record.get("product_category") or "") != candidate.product_category:
            raise RuntimeError(
                f"Certified category no longer matches product {candidate.asin}."
            )
        if backend.product_price_cents(candidate.asin) != certificate.price_cents:
            raise RuntimeError(
                f"Certified price no longer matches product {candidate.asin}."
            )
        _require_equal_hash(
            f"catalog record for {candidate.asin}",
            expected=certificate.catalog_record_sha256,
            observed=backend.product_record_sha256(candidate.asin),
        )


def _require_equal_hash(name: str, *, expected: str, observed: str) -> None:
    if expected != observed:
        raise RuntimeError(
            f"Certified {name} SHA256 mismatch: "
            f"expected {expected}, observed {observed}."
        )
