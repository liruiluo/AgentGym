from __future__ import annotations

from pathlib import Path

from ..native_webshop_backend import NativeWebShopBackend
from ..procedural import file_sha256, verify_lucene_index_manifest
from .certifier import CERTIFIER_VERSION, PREFERENCE_RULES_SHA256
from .schema import PreferenceProductPool


TRUSTED_CERTIFIER_RULES_SHA256 = {
    # First fully native-certified 192-product pool used by the v1 surface.
    "native_latent_preference_rules_v2": (
        "1f2aae6b207ae6d2a8c19fd2f621acfb"
        "cc96f342cf05cd30471efbf45f73a10d"
    ),
    CERTIFIER_VERSION: PREFERENCE_RULES_SHA256,
}


def attest_latent_preference_runtime_inputs(
    pool: PreferenceProductPool,
    backend: NativeWebShopBackend,
    *,
    items_file: Path,
    attributes_file: Path,
    search_root: Path,
    lucene_manifest: Path,
) -> None:
    """Fail closed if runtime inputs differ from the certified native pool."""

    expected_rules_sha256 = TRUSTED_CERTIFIER_RULES_SHA256.get(
        pool.certifier_version
    )
    if expected_rules_sha256 is None:
        raise RuntimeError(
            "Certified latent-preference pool used an unsupported certifier: "
            f"observed {pool.certifier_version}."
        )
    _require_equal_hash(
        "preference rules",
        expected=expected_rules_sha256,
        observed=pool.rules_sha256,
    )
    backend_metadata = backend.metadata()
    _require_equal_hash(
        "product catalog",
        expected=pool.catalog_sha256,
        observed=file_sha256(items_file),
    )
    _require_equal_hash(
        "product attributes",
        expected=pool.attributes_sha256,
        observed=file_sha256(attributes_file),
    )
    _require_equal_hash(
        "native price table",
        expected=pool.price_table_sha256,
        observed=str(backend_metadata["price_table_sha256"]),
    )
    _require_equal_hash(
        "Lucene index manifest",
        expected=pool.lucene_index_sha256,
        observed=file_sha256(lucene_manifest),
    )
    verify_lucene_index_manifest(
        lucene_manifest,
        index_dir=search_root / "indexes-full",
    )

    for product in pool.products:
        if not backend.has_product(product.asin):
            raise RuntimeError(
                f"Certified latent-preference product disappeared: {product.asin}."
            )
        if backend.product_title(product.asin) != product.title:
            raise RuntimeError(
                f"Certified title no longer matches native product {product.asin}."
            )
        if backend.product_price_cents(product.asin) != product.price_cents:
            raise RuntimeError(
                f"Certified price no longer matches native product {product.asin}."
            )
        _require_equal_hash(
            f"catalog record for {product.asin}",
            expected=product.catalog_record_sha256,
            observed=backend.product_record_sha256(product.asin),
        )


def _require_equal_hash(name: str, *, expected: str, observed: str) -> None:
    if expected != observed:
        raise RuntimeError(
            f"Certified {name} SHA256 mismatch: "
            f"expected {expected}, observed {observed}."
        )
