from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..latent_preference.schema import (
    canonical_sha256,
    normalize_native_title,
    require_sha256,
)
from ..native_webshop_backend import (
    FROZEN_MEMORYARENA_COMMIT,
    NativeWebShopBackend,
)
from .pool_io import load_negative_constraint_product_pool
from .schema import (
    NegativeConstraintCandidate,
    NegativeConstraintDataError,
    NegativeConstraintNativeCertificate,
    NegativeConstraintProductPool,
)


CERTIFIER_VERSION = "negative_constraint_native_v2"
CERTIFICATION_AUDIT_SCHEMA = "agentmemory_negative_constraint_certification_v2"
SOURCE_MANIFEST_SCHEMA = "agentmemory_negative_constraint_source_manifest_v2"

_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_QUERY_WORD_RE = re.compile(r"\b[\w][\w'&+./-]*\b", flags=re.UNICODE)
_QUERY_SEPARATOR_RE = re.compile(r"\s+(?:[-\u2013\u2014|])\s+|[,;:]")
_QUERY_EDGE_CHARS = " \t,.;:|/\\-\u2013\u2014(){}"
_QUERY_UNSAFE_CHARS = "[]\r\n"

_CANDIDATE_LOCAL_REJECTION_REASONS = frozenset(
    {
        "nonunique_native_normalized_title",
        "native_title_mismatch",
        "native_product_category_mismatch",
        "native_title_api_mismatch",
        "nonpositive_native_price",
        "no_safe_title_derived_search_query",
        "native_search_rank_exceeds_limit",
        "target_absent_from_all_title_derived_first_pages",
        "native_open_url_asin_mismatch",
        "native_buy_now_missing",
        "native_purchase_receipt_missing",
        "native_purchase_asin_mismatch",
        "native_purchase_price_mismatch",
    }
)


class NativeNegativeConstraintPoolCertificationError(NegativeConstraintDataError):
    def __init__(self, message: str, *, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class NativeNegativeConstraintCertificationConfig:
    pool_id: str = "memoryarena_negative_constraint_native_v2"
    max_search_rank: int = 10
    min_search_query_chars: int = 8
    max_search_query_chars: int = 160

    def __post_init__(self) -> None:
        if not isinstance(self.pool_id, str) or not self.pool_id.strip():
            raise NegativeConstraintDataError("native negative pool_id is required.")
        if not 1 <= self.max_search_rank <= 10:
            raise NegativeConstraintDataError(
                "max_search_rank must stay in the native first-page window."
            )
        if not 1 <= self.min_search_query_chars <= self.max_search_query_chars:
            raise NegativeConstraintDataError(
                "invalid native negative search-query length bounds."
            )


def certify_native_negative_constraint_product_pool_with_reselection(
    backend: NativeWebShopBackend,
    *,
    candidate_artifact: str | Path,
    expected_candidate_artifact_sha256: str,
    catalog_sha256: str,
    attributes_sha256: str,
    lucene_index_sha256: str,
    expected_memoryarena_commit: str = FROZEN_MEMORYARENA_COMMIT,
    config: NativeNegativeConstraintCertificationConfig | None = None,
) -> tuple[NegativeConstraintProductPool, dict[str, Any]]:
    """Rebuild the rules pool after candidate-local native rejections.

    Every rebuild uses the same frozen cell ordering and excludes the rejected
    ASIN globally. This keeps each replacement in the original cell/split and
    preserves the pool-wide ASIN/title uniqueness constraints.
    """

    config = config or NativeNegativeConstraintCertificationConfig()
    blocked_asins: set[str] = set()
    selection_rejections: list[dict[str, Any]] = []
    while True:
        rules_pool = load_negative_constraint_product_pool(
            candidate_artifact,
            expected_file_sha256=expected_candidate_artifact_sha256,
            blocked_asins=blocked_asins,
        )
        try:
            pool, audit = certify_native_negative_constraint_product_pool(
                backend,
                rules_pool=rules_pool,
                catalog_sha256=catalog_sha256,
                attributes_sha256=attributes_sha256,
                lucene_index_sha256=lucene_index_sha256,
                expected_memoryarena_commit=expected_memoryarena_commit,
                config=config,
            )
        except NativeNegativeConstraintPoolCertificationError as exc:
            rejected = next(
                (
                    probe
                    for probe in reversed(exc.audit.get("probes", []))
                    if probe.get("status") == "rejected"
                ),
                None,
            )
            reason = (
                str(rejected.get("rejection_reason") or "")
                if isinstance(rejected, Mapping)
                else ""
            )
            asin = (
                str(rejected.get("asin") or "")
                if isinstance(rejected, Mapping)
                else ""
            )
            if (
                reason not in _CANDIDATE_LOCAL_REJECTION_REASONS
                or not asin
                or asin in blocked_asins
            ):
                exc.audit["selection"] = _selection_audit(
                    rules_pool=rules_pool,
                    blocked_asins=blocked_asins,
                    rejections=selection_rejections,
                    retryable_failure=False,
                )
                raise
            if backend.active_session_count() != 0:
                exc.audit["selection"] = _selection_audit(
                    rules_pool=rules_pool,
                    blocked_asins=blocked_asins,
                    rejections=selection_rejections,
                    retryable_failure=False,
                )
                raise
            selection_rejections.append(
                {
                    key: rejected[key]
                    for key in (
                        "asin",
                        "cell",
                        "source_row_sha256",
                        "rejection_reason",
                        "source_title",
                        "native_title",
                        "source_product_category",
                        "native_product_category",
                        "matching_asins",
                        "search_attempts",
                    )
                    if key in rejected
                }
                | {"rejected_rules_pool_sha256": rules_pool.semantic_sha256}
            )
            blocked_asins.add(asin)
            continue

        audit["selection"] = _selection_audit(
            rules_pool=rules_pool,
            blocked_asins=blocked_asins,
            rejections=selection_rejections,
            retryable_failure=None,
        )
        audit["verification"]["deterministic_same_cell_split_reselection"] = True
        return pool, audit


def _selection_audit(
    *,
    rules_pool: NegativeConstraintProductPool,
    blocked_asins: set[str],
    rejections: list[dict[str, Any]],
    retryable_failure: bool | None,
) -> dict[str, Any]:
    return {
        "selection_policy": rules_pool.selection_policy,
        "rebuild_count": len(rejections),
        "blocked_candidate_asins": sorted(blocked_asins),
        "candidate_rejections": list(rejections),
        "final_rules_pool_sha256": rules_pool.semantic_sha256,
        "retryable_failure": retryable_failure,
    }


def source_manifest_for_pool(pool: NegativeConstraintProductPool) -> dict[str, Any]:
    if not pool.native_certified:
        raise NegativeConstraintDataError(
            "rules-only negative pool has no native source manifest."
        )
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "certifier_version": pool.certifier_version,
        "rules_pool_sha256": pool.rules_pool_sha256,
        "candidate_artifact_sha256": pool.candidate_artifact_sha256,
        "memoryarena_commit": pool.memoryarena_commit,
        "catalog_sha256": pool.catalog_sha256,
        "attributes_sha256": pool.attributes_sha256,
        "price_table_sha256": pool.price_table_sha256,
        "lucene_index_sha256": pool.lucene_index_sha256,
        "price_seed": pool.price_seed,
    }


def certify_native_negative_constraint_product_pool(
    backend: NativeWebShopBackend,
    *,
    rules_pool: NegativeConstraintProductPool,
    catalog_sha256: str,
    attributes_sha256: str,
    lucene_index_sha256: str,
    expected_memoryarena_commit: str = FROZEN_MEMORYARENA_COMMIT,
    config: NativeNegativeConstraintCertificationConfig | None = None,
) -> tuple[NegativeConstraintProductPool, dict[str, Any]]:
    """Certify every selected product through the original WebShop runtime."""

    config = config or NativeNegativeConstraintCertificationConfig()
    if rules_pool.native_certified:
        raise NegativeConstraintDataError(
            "negative native certification requires a rules-only source pool."
        )
    for field, value in (
        ("catalog_sha256", catalog_sha256),
        ("attributes_sha256", attributes_sha256),
        ("lucene_index_sha256", lucene_index_sha256),
    ):
        require_sha256(value, field=field)
    if (
        not isinstance(expected_memoryarena_commit, str)
        or len(expected_memoryarena_commit) != 40
        or any(char not in "0123456789abcdef" for char in expected_memoryarena_commit)
    ):
        raise NegativeConstraintDataError(
            "expected_memoryarena_commit must be a full lowercase Git commit."
        )

    metadata = backend.metadata()
    if backend.active_session_count() != 0:
        raise NegativeConstraintDataError(
            "native negative certification requires an idle backend."
        )
    price_table_sha256 = str(metadata.get("price_table_sha256") or "")
    require_sha256(price_table_sha256, field="native price_table_sha256")
    price_seed = metadata.get("price_seed")
    if isinstance(price_seed, bool) or not isinstance(price_seed, int):
        raise NegativeConstraintDataError(
            "native backend metadata is missing an integer price_seed."
        )
    upstream = metadata.get("upstream_provenance")
    observed_commit = (
        str(upstream.get("memoryarena_commit") or "")
        if isinstance(upstream, Mapping)
        else ""
    )
    if observed_commit != expected_memoryarena_commit:
        raise NegativeConstraintDataError(
            "native MemoryArena commit mismatch: "
            f"expected {expected_memoryarena_commit}, observed {observed_commit}."
        )

    rules_pool_sha256 = rules_pool.semantic_sha256
    source_manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "certifier_version": CERTIFIER_VERSION,
        "rules_pool_sha256": rules_pool_sha256,
        "candidate_artifact_sha256": rules_pool.candidate_artifact_sha256,
        "memoryarena_commit": observed_commit,
        "catalog_sha256": catalog_sha256,
        "attributes_sha256": attributes_sha256,
        "price_table_sha256": price_table_sha256,
        "lucene_index_sha256": lucene_index_sha256,
        "price_seed": price_seed,
    }
    source_manifest_sha256 = canonical_sha256(source_manifest)
    audit: dict[str, Any] = {
        "schema": CERTIFICATION_AUDIT_SCHEMA,
        "status": "running",
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "selected_product_count": len(rules_pool.candidates),
        "probes": [],
    }

    normalized_targets = {
        candidate.normalized_title for candidate in rules_pool.candidates
    }
    title_matches: dict[str, set[str]] = {
        title: set() for title in normalized_targets
    }
    scanned = 0
    for raw_asin in backend.product_asins():
        asin = str(raw_asin).upper()
        scanned += 1
        normalized = normalize_native_title(backend.product_title(asin))
        if normalized in title_matches:
            title_matches[normalized].add(asin)
    audit["native_catalog_title_records_scanned"] = scanned

    certificates: list[NegativeConstraintNativeCertificate] = []
    for probe_index, candidate in enumerate(
        sorted(rules_pool.candidates, key=lambda item: item.asin)
    ):
        certificate, detail = _audit_native_candidate(
            backend,
            candidate=candidate,
            title_matches=title_matches[candidate.normalized_title],
            probe_index=probe_index,
            config=config,
        )
        audit["probes"].append(detail)
        if certificate is None:
            audit.update(
                status="failed",
                accepted_product_count=len(certificates),
                failed_product_asin=candidate.asin,
                active_session_count_after_failure=backend.active_session_count(),
            )
            raise NativeNegativeConstraintPoolCertificationError(
                f"native certification failed for selected product {candidate.asin}: "
                f"{detail['rejection_reason']}",
                audit=audit,
            )
        certificates.append(certificate)

    if backend.active_session_count() != 0:
        audit.update(
            status="failed",
            accepted_product_count=len(certificates),
            active_session_count_after_failure=backend.active_session_count(),
        )
        raise NativeNegativeConstraintPoolCertificationError(
            "native certification leaked WebShop sessions.",
            audit=audit,
        )

    pool = replace(
        rules_pool,
        pool_id=config.pool_id,
        native_certified=True,
        certifier_version=CERTIFIER_VERSION,
        memoryarena_commit=observed_commit,
        catalog_sha256=catalog_sha256,
        attributes_sha256=attributes_sha256,
        price_table_sha256=price_table_sha256,
        lucene_index_sha256=lucene_index_sha256,
        source_manifest_sha256=source_manifest_sha256,
        rules_pool_sha256=rules_pool_sha256,
        price_seed=price_seed,
        native_certificates=tuple(certificates),
    )
    if canonical_sha256(source_manifest_for_pool(pool)) != source_manifest_sha256:
        raise AssertionError("negative native source manifest is not reproducible")
    audit.update(
        status="certified",
        accepted_product_count=len(certificates),
        native_pool_semantic_sha256=pool.semantic_sha256,
        active_session_count_after_certification=backend.active_session_count(),
        verification={
            "exact_selected_asin_coverage": True,
            "global_normalized_title_uniqueness": True,
            "title_derived_lucene_rank_at_most_10": True,
            "native_open_verified": True,
            "native_purchase_receipt_verified": True,
            "rules_pool_bound": True,
            "candidate_artifact_bound": True,
            "runtime_inputs_bound": True,
            "human_review_required": False,
            "llm_judge_required": False,
            "training_ready": True,
        },
    )
    return pool, audit


def _audit_native_candidate(
    backend: NativeWebShopBackend,
    *,
    candidate: NegativeConstraintCandidate,
    title_matches: set[str],
    probe_index: int,
    config: NativeNegativeConstraintCertificationConfig,
) -> tuple[NegativeConstraintNativeCertificate | None, dict[str, Any]]:
    token = f"amgnc-cert-{probe_index}-{candidate.asin}"
    detail: dict[str, Any] = {
        "probe_index": probe_index,
        "asin": candidate.asin,
        "cell": "/".join(
            (
                candidate.axis,
                candidate.category_id,
                candidate.attribute_value,
                candidate.split,
            )
        ),
        "source_row_sha256": candidate.source_row_sha256,
        "search_attempts": [],
    }

    def reject(reason: str, **extra: Any):
        detail.update(status="rejected", rejection_reason=reason, **extra)
        return None, detail

    try:
        if title_matches != {candidate.asin}:
            return reject(
                "nonunique_native_normalized_title",
                matching_asins=sorted(title_matches),
            )
        record = backend.product_record(candidate.asin)
        native_title = str(record.get("Title") or "")
        if native_title != candidate.title:
            return reject(
                "native_title_mismatch",
                source_title=candidate.title,
                native_title=native_title,
            )
        native_product_category = str(record.get("product_category") or "")
        if native_product_category != candidate.product_category:
            return reject(
                "native_product_category_mismatch",
                source_product_category=candidate.product_category,
                native_product_category=native_product_category,
            )
        native_api_title = backend.product_title(candidate.asin)
        if native_api_title != candidate.title:
            return reject(
                "native_title_api_mismatch",
                source_title=candidate.title,
                native_title=native_api_title,
            )
        price_cents = backend.product_price_cents(candidate.asin)
        if price_cents <= 0:
            return reject("nonpositive_native_price")
        catalog_record_sha256 = backend.product_record_sha256(candidate.asin)
        require_sha256(
            catalog_record_sha256,
            field="negative native catalog_record_sha256",
        )

        page = backend.open_session(
            token,
            "Native execution certification for a remembered exclusion constraint.",
        )
        if not page.has_search_bar:
            return reject("native_search_bar_missing")
        selected_query: str | None = None
        selected_rank: int | None = None
        selected_results: tuple[str, ...] = ()
        target_seen_outside_limit = False
        queries = _title_derived_search_queries(
            title=candidate.title,
            normalized_title=candidate.normalized_title,
            title_evidence=candidate.title_evidence,
            min_chars=config.min_search_query_chars,
            max_chars=config.max_search_query_chars,
        )
        if not queries:
            return reject("no_safe_title_derived_search_query")
        for query in queries:
            page = backend.step(token, f"search[{query}]")
            result_asins = tuple(
                value.upper()
                for value in page.clickables
                if _ASIN_RE.fullmatch(value.upper())
            )
            try:
                rank = result_asins.index(candidate.asin) + 1
            except ValueError:
                rank = None
            within_limit = rank is not None and rank <= config.max_search_rank
            detail["search_attempts"].append(
                {
                    "query": query,
                    "result_asins": list(result_asins),
                    "target_rank": rank,
                    "within_limit": within_limit,
                }
            )
            if within_limit:
                selected_query = query
                selected_rank = rank
                selected_results = result_asins
                break
            if rank is not None:
                target_seen_outside_limit = True
        if selected_query is None or selected_rank is None:
            return reject(
                "native_search_rank_exceeds_limit"
                if target_seen_outside_limit
                else "target_absent_from_all_title_derived_first_pages"
            )

        page = backend.step(token, f"click[{candidate.asin}]")
        opened_url = page.url
        if candidate.asin.casefold() not in opened_url.casefold():
            return reject("native_open_url_asin_mismatch", opened_url=opened_url)
        buy_now = next(
            (value for value in page.clickables if value.casefold() == "buy now"),
            None,
        )
        if buy_now is None:
            return reject("native_buy_now_missing")
        page = backend.step(token, f"click[{buy_now}]")
        purchase = page.purchase
        if purchase is None:
            return reject("native_purchase_receipt_missing")
        if purchase.asin.upper() != candidate.asin:
            return reject("native_purchase_asin_mismatch")
        if purchase.price_cents != price_cents:
            return reject("native_purchase_price_mismatch")
        purchase_receipt_sha256 = canonical_sha256(
            {
                "asin": purchase.asin.upper(),
                "price_cents": purchase.price_cents,
                "selected_options": dict(sorted(purchase.selected_options.items())),
            }
        )
        detail.update(
            status="accepted",
            rejection_reason=None,
            selected_search_query=selected_query,
            selected_search_rank=selected_rank,
            selected_search_result_asins=list(selected_results),
            opened_url=opened_url,
            price_cents=price_cents,
            catalog_record_sha256=catalog_record_sha256,
            purchase_receipt_sha256=purchase_receipt_sha256,
        )
        return (
            NegativeConstraintNativeCertificate(
                asin=candidate.asin,
                source_row_sha256=candidate.source_row_sha256,
                price_cents=price_cents,
                search_query=selected_query,
                search_rank=selected_rank,
                search_result_asins=selected_results,
                opened_url=opened_url,
                catalog_record_sha256=catalog_record_sha256,
                purchase_receipt_sha256=purchase_receipt_sha256,
            ),
            detail,
        )
    except Exception as exc:
        return reject(
            f"exception_{type(exc).__name__}",
            exception_type=type(exc).__name__,
        )
    finally:
        try:
            backend.close_session(token)
        except Exception:
            pass


def _title_derived_search_queries(
    *,
    title: str,
    normalized_title: str,
    title_evidence: tuple[str, ...],
    min_chars: int,
    max_chars: int,
) -> tuple[str, ...]:
    normalized_evidence = tuple(
        normalize_native_title(value) for value in title_evidence
    )
    queries: list[str] = []
    seen: set[str] = set()

    def add(raw_query: str) -> None:
        query = raw_query.strip(_QUERY_EDGE_CHARS)
        if not min_chars <= len(query) <= max_chars:
            return
        if any(char in query for char in _QUERY_UNSAFE_CHARS):
            return
        if len(_QUERY_WORD_RE.findall(query)) < 3:
            return
        normalized_query = normalize_native_title(query)
        if not normalized_query or normalized_query not in normalized_title:
            return
        if not any(value in normalized_query for value in normalized_evidence):
            return
        if normalized_query in seen:
            return
        seen.add(normalized_query)
        queries.append(query)

    add(title)
    separators = tuple(_QUERY_SEPARATOR_RE.finditer(title))
    for separator in separators:
        add(title[: separator.start()])
    segment_starts = [0, *(match.end() for match in separators)]
    segment_ends = [*(match.start() for match in separators), len(title)]
    for segment_index in range(len(segment_starts)):
        for width in (1, 2, 3):
            end_index = segment_index + width - 1
            if end_index >= len(segment_ends):
                break
            add(title[segment_starts[segment_index] : segment_ends[end_index]])

    words = tuple(_QUERY_WORD_RE.finditer(title))
    folded_title = title.casefold()
    for evidence in title_evidence:
        start = folded_title.find(evidence.casefold())
        if start < 0:
            continue
        end = start + len(evidence)
        containing = [
            index
            for index, word in enumerate(words)
            if word.start() < end and word.end() > start
        ]
        if not containing:
            continue
        first_word = containing[0]
        last_word = containing[-1]
        for trailing_words in (2, 4, 8, 12):
            end_word = min(len(words) - 1, last_word + trailing_words)
            add(title[: words[end_word].end()])
        for context_words in (4, 8, 12):
            start_word = max(0, first_word - context_words)
            end_word = min(len(words) - 1, last_word + context_words)
            add(title[words[start_word].start() : words[end_word].end()])
    for prefix_words in (6, 8, 12, 16, 24, 32):
        if len(words) >= prefix_words:
            add(title[: words[prefix_words - 1].end()])
    return tuple(queries)
