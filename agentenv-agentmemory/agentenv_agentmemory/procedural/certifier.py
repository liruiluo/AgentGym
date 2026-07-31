from __future__ import annotations

import collections
import re
from dataclasses import dataclass, replace
from typing import Any

from ..native_webshop_backend import NativeWebShopBackend
from .schema import (
    SPLITS,
    CertifiedProduct,
    ProceduralMemoryDataError,
    ProductPool,
    canonical_sha256,
    normalize_native_title,
    require_sha256,
)
from .scenarios import (
    SCENARIO_DEFINITION_SHA256,
    SCENARIO_DEFINITION_VERSION,
    SCENARIOS,
    ProductClassification,
    classify_product_record,
    scenario_by_id,
)


CERTIFIER_VERSION = "native_natural_attribute_rules_v4"
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_QUERY_WORD_RE = re.compile(r"\b[\w][\w'&+./-]*\b", flags=re.UNICODE)
_QUERY_SEPARATOR_RE = re.compile(r"\s+(?:[-\u2013\u2014|])\s+|[,;:]")
_QUERY_EDGE_CHARS = " \t,.;:|/\\-\u2013\u2014(){}"
_QUERY_UNSAFE_CHARS = "[]\r\n"


class NativeProductPoolCertificationError(ProceduralMemoryDataError):
    """Certification failure carrying the complete machine-readable audit."""

    def __init__(self, message: str, *, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class NativeCertificationConfig:
    pool_id: str = "memoryarena_natural_order_chains_v4"
    scenario_ids: tuple[str, ...] = tuple(
        scenario.scenario_id for scenario in SCENARIOS
    )
    products_per_cell: int = 4
    probe_cap_per_cell_split: int = 24
    max_search_rank: int = 10
    min_title_chars: int = 8
    max_title_chars: int = 240
    min_search_query_chars: int = 8
    max_search_query_chars: int = 160

    def __post_init__(self) -> None:
        known_order = tuple(
            scenario.scenario_id
            for scenario in SCENARIOS
            if scenario.scenario_id in set(self.scenario_ids)
        )
        if not self.scenario_ids or self.scenario_ids != known_order:
            raise ProceduralMemoryDataError(
                "scenario_ids must be a non-empty canonical ordered subset of the "
                "five built-in scenarios."
            )
        if self.products_per_cell < 1:
            raise ProceduralMemoryDataError("products_per_cell must be positive.")
        if self.probe_cap_per_cell_split < self.products_per_cell:
            raise ProceduralMemoryDataError(
                "probe_cap_per_cell_split must be at least products_per_cell."
            )
        if not 1 <= self.max_search_rank <= 10:
            raise ProceduralMemoryDataError(
                "max_search_rank must stay within the native first-page window 1..10."
            )
        if not 1 <= self.min_title_chars <= self.max_title_chars:
            raise ProceduralMemoryDataError("invalid native title length bounds.")
        if not 1 <= self.min_search_query_chars <= self.max_search_query_chars:
            raise ProceduralMemoryDataError("invalid native search-query length bounds.")
        if self.max_search_query_chars > self.max_title_chars:
            raise ProceduralMemoryDataError(
                "max_search_query_chars cannot exceed max_title_chars."
            )


@dataclass(frozen=True)
class _Candidate:
    asin: str
    title: str
    normalized_title: str
    classification: ProductClassification
    selection_sha256: str
    catalog_title_match_count: int | None = None

    @property
    def base_cell(self) -> tuple[str, str, str]:
        return (
            self.classification.scenario_id,
            self.classification.slot_id,
            self.classification.attribute_value,
        )


@dataclass(frozen=True)
class _NativeEvidence:
    candidate: _Candidate
    price_cents: int
    search_query: str
    search_rank: int
    catalog_record_sha256: str

    def to_product(self, *, split: str) -> CertifiedProduct:
        candidate = self.candidate
        if candidate.catalog_title_match_count != 1:
            raise AssertionError("certified candidate lost catalog title uniqueness")
        return CertifiedProduct.from_classification(
            classification=candidate.classification,
            asin=candidate.asin,
            title=candidate.title,
            split=split,
            price_cents=self.price_cents,
            search_query=self.search_query,
            search_rank=self.search_rank,
            catalog_record_sha256=self.catalog_record_sha256,
            native_title_catalog_match_count=candidate.catalog_title_match_count,
            native_title_globally_unique=True,
        )


def certify_native_product_pool(
    backend: NativeWebShopBackend,
    *,
    catalog_sha256: str,
    attributes_sha256: str,
    lucene_index_sha256: str,
    config: NativeCertificationConfig | None = None,
) -> tuple[ProductPool, dict[str, Any]]:
    """Certify a balanced natural-attribute pool with no human or LLM judge."""

    config = config or NativeCertificationConfig()
    require_sha256(catalog_sha256, field="catalog_sha256")
    require_sha256(attributes_sha256, field="attributes_sha256")
    require_sha256(lucene_index_sha256, field="lucene_index_sha256")
    metadata = backend.metadata()
    price_table_sha256 = metadata.get("price_table_sha256")
    require_sha256(price_table_sha256, field="native price_table_sha256")

    shortlist, scan_counts = _build_shortlist(backend, config=config)
    required_per_base_cell = config.products_per_cell * len(SPLITS)
    accepted_by_base: dict[tuple[str, str, str], list[_NativeEvidence]] = {
        cell: [] for cell in _expected_base_cells(config)
    }
    rejection_counts: collections.Counter[str] = collections.Counter()
    probed_by_base: collections.Counter[tuple[str, str, str]] = collections.Counter()
    probe_details: list[dict[str, Any]] = []
    probe_index = 0
    for base_cell in _expected_base_cells(config):
        for candidate in shortlist[base_cell]:
            if len(accepted_by_base[base_cell]) >= required_per_base_cell:
                break
            probed_by_base[base_cell] += 1
            evidence, detail = _audit_candidate(
                backend,
                candidate=candidate,
                probe_index=probe_index,
                config=config,
                scenario_ids=config.scenario_ids,
            )
            probe_index += 1
            probe_details.append(detail)
            if evidence is None:
                rejection_counts[str(detail["rejection_reason"])] += 1
                continue
            accepted_by_base[base_cell].append(evidence)

    accepted_by_cell: dict[tuple[str, str, str, str], list[CertifiedProduct]] = {
        cell: [] for cell in _expected_cells(config)
    }
    for base_cell in _expected_base_cells(config):
        for rank, evidence in enumerate(accepted_by_base[base_cell]):
            split = SPLITS[rank % len(SPLITS)]
            accepted_by_cell[(*base_cell, split)].append(
                evidence.to_product(split=split)
            )

    source_manifest = {
        "certifier_version": CERTIFIER_VERSION,
        "config": _config_dict(config),
        "scenario_definition": {
            "version": SCENARIO_DEFINITION_VERSION,
            "sha256": SCENARIO_DEFINITION_SHA256,
        },
        "catalog_sha256": catalog_sha256,
        "attributes_sha256": attributes_sha256,
        "price_table_sha256": price_table_sha256,
        "lucene_index_sha256": lucene_index_sha256,
        "backend": _stable_backend_metadata(metadata),
    }
    source_manifest_sha256 = canonical_sha256(source_manifest)
    missing = {
        _cell_name(cell): len(values)
        for cell, values in accepted_by_cell.items()
        if len(values) < config.products_per_cell
    }
    if missing:
        audit = _build_audit(
            status="failed",
            pool=None,
            config=config,
            source_manifest_sha256=source_manifest_sha256,
            metadata=metadata,
            catalog_sha256=catalog_sha256,
            attributes_sha256=attributes_sha256,
            price_table_sha256=price_table_sha256,
            lucene_index_sha256=lucene_index_sha256,
            shortlist=shortlist,
            scan_counts=scan_counts,
            probed_by_base=probed_by_base,
            accepted_by_base=accepted_by_base,
            accepted_by_cell=accepted_by_cell,
            rejection_counts=rejection_counts,
            probe_details=probe_details,
            missing=missing,
        )
        raise NativeProductPoolCertificationError(
            "native certification could not fill every natural-attribute cell: "
            f"{dict(sorted(missing.items()))}; "
            f"rejections={dict(sorted(rejection_counts.items()))}.",
            audit=audit,
        )

    products = tuple(
        product
        for cell in _expected_cells(config)
        for product in accepted_by_cell[cell]
    )
    pool = ProductPool(
        pool_id=config.pool_id,
        certifier_version=CERTIFIER_VERSION,
        scenario_ids=config.scenario_ids,
        products_per_cell=config.products_per_cell,
        products=products,
        catalog_sha256=catalog_sha256,
        attributes_sha256=attributes_sha256,
        price_table_sha256=price_table_sha256,
        lucene_index_sha256=lucene_index_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    audit = _build_audit(
        status="certified",
        pool=pool,
        config=config,
        source_manifest_sha256=source_manifest_sha256,
        metadata=metadata,
        catalog_sha256=catalog_sha256,
        attributes_sha256=attributes_sha256,
        price_table_sha256=price_table_sha256,
        lucene_index_sha256=lucene_index_sha256,
        shortlist=shortlist,
        scan_counts=scan_counts,
        probed_by_base=probed_by_base,
        accepted_by_base=accepted_by_base,
        accepted_by_cell=accepted_by_cell,
        rejection_counts=rejection_counts,
        probe_details=probe_details,
        missing={},
    )
    return pool, audit


def _build_audit(
    *,
    status: str,
    pool: ProductPool | None,
    config: NativeCertificationConfig,
    source_manifest_sha256: str,
    metadata: dict[str, Any],
    catalog_sha256: str,
    attributes_sha256: str,
    price_table_sha256: str,
    lucene_index_sha256: str,
    shortlist: dict[tuple[str, str, str], tuple[_Candidate, ...]],
    scan_counts: dict[str, int],
    probed_by_base: collections.Counter[tuple[str, str, str]],
    accepted_by_base: dict[tuple[str, str, str], list[_NativeEvidence]],
    accepted_by_cell: dict[tuple[str, str, str, str], list[CertifiedProduct]],
    rejection_counts: collections.Counter[str],
    probe_details: list[dict[str, Any]],
    missing: dict[str, int],
) -> dict[str, Any]:
    per_base_cell = {
        _cell_name(cell): {
            "shortlisted": len(shortlist[cell]),
            "probed": probed_by_base[cell],
            "accepted_before_split": len(accepted_by_base[cell]),
            "required_before_split": config.products_per_cell * len(SPLITS),
        }
        for cell in _expected_base_cells(config)
    }
    per_split_cell = {
        _cell_name(cell): {"certified": len(accepted_by_cell[cell])}
        for cell in _expected_cells(config)
    }
    return {
        "schema": "agentmemory_native_natural_attribute_pool_certification_v3",
        "status": status,
        "certifier_version": CERTIFIER_VERSION,
        "pool_id": config.pool_id,
        "product_pool_semantic_sha256": (
            None if pool is None else pool.semantic_sha256
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "contract": _config_dict(config),
        "provenance": {
            "catalog_sha256": catalog_sha256,
            "attributes_sha256": attributes_sha256,
            "price_table_sha256": price_table_sha256,
            "lucene_index_sha256": lucene_index_sha256,
            "backend": _stable_backend_metadata(metadata),
        },
        "counts": {
            **scan_counts,
            "shortlisted": sum(len(values) for values in shortlist.values()),
            "probed": sum(probed_by_base.values()),
            "accepted_before_split": sum(len(values) for values in accepted_by_base.values()),
            "certified": sum(len(values) for values in accepted_by_cell.values()),
            "rejections": dict(sorted(rejection_counts.items())),
            "missing_split_cells": dict(sorted(missing.items())),
            "per_base_cell": per_base_cell,
            "per_split_cell": per_split_cell,
        },
        "candidate_probes": probe_details,
        "verification": {
            "scenario_count": len(config.scenario_ids),
            "slots_per_scenario": 6,
            "natural_attribute_values_per_slot": 2,
            "candidate_count_per_phase": 2,
            "approved_shortlist_uses_exact_asins_internally": True,
            "approved_shortlist_asins_policy_visible": False,
            "policy_visible_product_identity": "complete_native_title",
            "native_title_normalization": "unicode_nfkc_whitespace_casefold_v1",
            "native_title_catalog_match_count_required": 1,
            "native_title_globally_unique": True,
            "native_search_query_title_derived": True,
            "native_search_query_attribute_evidence_required": True,
            "native_search_query_policy_visible_as_separate_id": False,
            "global_catalog_attribute_uniqueness_required": False,
            "global_catalog_attribute_uniqueness_claimed": False,
            "category_and_attribute_from_native_record": True,
            "ambiguous_classification_rejected": True,
            "native_title_exact": True,
            "native_price_exact": True,
            "native_search_first_page": True,
            "native_item_page_opened": True,
            "native_purchase_receipt_exact": True,
            "native_audit_before_split_assignment": True,
            "asin_split_isolated": True,
            "human_review_required": False,
            "llm_judge_required": False,
            "paper_eligible": False,
        },
    }


def _build_shortlist(
    backend: NativeWebShopBackend,
    *,
    config: NativeCertificationConfig,
) -> tuple[dict[tuple[str, str, str], tuple[_Candidate, ...]], dict[str, int]]:
    by_base_cell: dict[tuple[str, str, str], list[_Candidate]] = {
        cell: [] for cell in _expected_base_cells(config)
    }
    counts: collections.Counter[str] = collections.Counter()
    for raw_asin in backend.product_asins():
        counts["native_catalog_products"] += 1
        asin = str(raw_asin).upper()
        if not _ASIN_RE.fullmatch(asin):
            counts["rejected_invalid_asin"] += 1
            continue
        record = backend.product_record(asin)
        title = str(record.get("Title") or "")
        if title != title.strip() or not (
            config.min_title_chars <= len(title) <= config.max_title_chars
        ):
            counts["rejected_title_length_or_whitespace"] += 1
            continue
        if any(char in title for char in "\r\n"):
            counts["rejected_multiline_native_title"] += 1
            continue
        normalized_title = normalize_native_title(title)
        if not normalized_title:
            counts["rejected_empty_normalized_title"] += 1
            continue
        if asin.casefold() in normalized_title:
            counts["rejected_title_contains_internal_asin"] += 1
            continue
        classifications = classify_product_record(
            record,
            scenario_ids=config.scenario_ids,
        )
        if not classifications:
            counts["not_in_natural_attribute_taxonomy"] += 1
            continue
        if len(classifications) != 1:
            counts["rejected_ambiguous_natural_classification"] += 1
            continue
        classification = classifications[0]
        candidate = _Candidate(
            asin=asin,
            title=title,
            normalized_title=normalized_title,
            classification=classification,
            selection_sha256=canonical_sha256(
                {
                    "version": CERTIFIER_VERSION,
                    "pool_id": config.pool_id,
                    "purpose": "balanced_native_probe",
                    "classification_sha256": classification.semantic_sha256,
                    "asin": asin,
                }
            ),
        )
        by_base_cell[candidate.base_cell].append(candidate)
        counts["uniquely_classified"] += 1

    candidate_normalized_titles = {
        candidate.normalized_title
        for values in by_base_cell.values()
        for candidate in values
    }
    matching_catalog_asins: dict[str, set[str]] = {
        title: set() for title in candidate_normalized_titles
    }
    for raw_asin in backend.product_asins():
        counts["catalog_title_identity_records_scanned"] += 1
        catalog_asin = str(raw_asin).upper()
        normalized_title = normalize_native_title(backend.product_title(catalog_asin))
        if normalized_title in matching_catalog_asins:
            matching_catalog_asins[normalized_title].add(catalog_asin)
    counts["candidate_normalized_title_groups"] = len(candidate_normalized_titles)
    counts["duplicate_normalized_title_groups"] = sum(
        len(asins) != 1 for asins in matching_catalog_asins.values()
    )

    max_candidates_per_base = config.probe_cap_per_cell_split * len(SPLITS)
    shortlist: dict[tuple[str, str, str], tuple[_Candidate, ...]] = {}
    for base_cell, values in by_base_cell.items():
        globally_unique: list[_Candidate] = []
        for candidate in values:
            matches = matching_catalog_asins[candidate.normalized_title]
            if matches != {candidate.asin}:
                counts["rejected_nonunique_normalized_title_candidates"] += 1
                continue
            globally_unique.append(
                replace(candidate, catalog_title_match_count=len(matches))
            )
            counts["globally_unique_normalized_title_candidates"] += 1
        ordered = sorted(
            globally_unique,
            key=lambda value: (value.selection_sha256, value.asin),
        )
        counts["shortlist_candidates_before_probe_cap"] += len(ordered)
        if len(ordered) > max_candidates_per_base:
            counts["probe_cap_truncated_candidates"] += (
                len(ordered) - max_candidates_per_base
            )
        shortlist[base_cell] = tuple(ordered[:max_candidates_per_base])
    return shortlist, dict(sorted(counts.items()))


def _audit_candidate(
    backend: NativeWebShopBackend,
    *,
    candidate: _Candidate,
    probe_index: int,
    config: NativeCertificationConfig,
    scenario_ids: tuple[str, ...],
) -> tuple[_NativeEvidence | None, dict[str, Any]]:
    if candidate.catalog_title_match_count != 1:
        raise AssertionError("candidate must pass catalog-wide title uniqueness first")
    token = f"amgpm-cert-{probe_index}-{candidate.asin}"
    detail: dict[str, Any] = {
        "probe_index": probe_index,
        "base_cell": _cell_name(candidate.base_cell),
        "asin": candidate.asin,
        "title": candidate.title,
        "selection_sha256": candidate.selection_sha256,
        "search_attempts": [],
    }

    def reject(reason: str, **extra: Any) -> tuple[None, dict[str, Any]]:
        detail.update(status="rejected", rejection_reason=reason, **extra)
        return None, detail

    try:
        record = backend.product_record(candidate.asin)
        current = classify_product_record(record, scenario_ids=scenario_ids)
        if current != (candidate.classification,):
            return reject("classification_changed_during_probe")
        if str(record.get("Title")) != candidate.title:
            return reject("native_title_mismatch")
        if backend.product_title(candidate.asin) != candidate.title:
            return reject("native_title_api_mismatch")
        price_cents = backend.product_price_cents(candidate.asin)
        if price_cents <= 0:
            return reject("nonpositive_native_price")
        record_sha256 = backend.product_record_sha256(candidate.asin)
        require_sha256(record_sha256, field="native catalog_record_sha256")

        page = backend.open_session(
            token,
            "Native execution certification for a generated shopping task.",
        )
        if not page.has_search_bar:
            return reject("native_search_bar_missing")

        selected_query: str | None = None
        selected_rank: int | None = None
        target_seen_outside_limit = False
        queries = _candidate_search_queries(
            candidate,
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
                search_rank = result_asins.index(candidate.asin) + 1
            except ValueError:
                search_rank = None
            within_limit = search_rank is not None and search_rank <= config.max_search_rank
            detail["search_attempts"].append(
                {
                    "query": query,
                    "result_asins": list(result_asins),
                    "target_rank": search_rank,
                    "within_limit": within_limit,
                }
            )
            if within_limit:
                selected_query = query
                selected_rank = search_rank
                break
            if search_rank is not None:
                target_seen_outside_limit = True
        if selected_query is None or selected_rank is None:
            reason = (
                "native_search_rank_exceeds_limit"
                if target_seen_outside_limit
                else "target_absent_from_all_title_derived_first_pages"
            )
            return reject(reason)

        page = backend.step(token, f"click[{candidate.asin}]")
        buy_now = next(
            (value for value in page.clickables if value.casefold() == "buy now"),
            None,
        )
        if buy_now is None:
            return reject("native_buy_now_missing")
        page = backend.step(token, f"click[{buy_now}]")
        if page.purchase is None:
            return reject("native_purchase_receipt_missing")
        if page.purchase.asin.upper() != candidate.asin:
            return reject("native_purchase_asin_mismatch")
        if page.purchase.price_cents != price_cents:
            return reject("native_purchase_price_mismatch")
        detail.update(
            status="accepted",
            rejection_reason=None,
            selected_search_query=selected_query,
            selected_search_rank=selected_rank,
            price_cents=price_cents,
            catalog_record_sha256=record_sha256,
        )
        return (
            _NativeEvidence(
                candidate=candidate,
                price_cents=price_cents,
                search_query=selected_query,
                search_rank=selected_rank,
                catalog_record_sha256=record_sha256,
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


def _candidate_search_queries(
    candidate: _Candidate,
    *,
    min_chars: int,
    max_chars: int,
) -> tuple[str, ...]:
    """Build bounded natural queries copied contiguously from the visible title."""

    title = candidate.title
    normalized_title = candidate.normalized_title
    attribute_evidence = tuple(
        normalize_native_title(value)
        for value in candidate.classification.attribute_title_evidence
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
        if not any(value in normalized_query for value in attribute_evidence):
            return
        if normalized_query in seen:
            return
        seen.add(normalized_query)
        queries.append(query)

    # Keep the exact listing title when native action syntax and query length allow it.
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
    evidence_spans: list[tuple[int, int]] = []
    for evidence in candidate.classification.attribute_title_evidence:
        start = folded_title.find(evidence.casefold())
        if start >= 0:
            evidence_spans.append((start, start + len(evidence)))
    for evidence_start, evidence_end in evidence_spans:
        containing = [
            index
            for index, word in enumerate(words)
            if word.start() < evidence_end and word.end() > evidence_start
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


def _expected_base_cells(
    config: NativeCertificationConfig,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (scenario_id, slot.slot_id, value.value_id)
        for scenario_id in config.scenario_ids
        for slot in scenario_by_id(scenario_id).slots
        for value in slot.values
    )


def _expected_cells(
    config: NativeCertificationConfig,
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple((*cell, split) for cell in _expected_base_cells(config) for split in SPLITS)


def _cell_name(cell: tuple[str, ...]) -> str:
    return "/".join(cell)


def _config_dict(config: NativeCertificationConfig) -> dict[str, Any]:
    return {
        "scenario_ids": list(config.scenario_ids),
        "scenario_definition_version": SCENARIO_DEFINITION_VERSION,
        "scenario_definition_sha256": SCENARIO_DEFINITION_SHA256,
        "products_per_cell": config.products_per_cell,
        "probe_cap_per_cell_split": config.probe_cap_per_cell_split,
        "max_native_search_rank": config.max_search_rank,
        "title_length_chars": [config.min_title_chars, config.max_title_chars],
        "search_query_length_chars": [
            config.min_search_query_chars,
            config.max_search_query_chars,
        ],
        "classification_source": (
            "native category plus unambiguous natural attribute evidence in title"
        ),
        "policy_visible_product_identity": "complete native title, ASIN hidden",
        "search_query_source": (
            "safe contiguous natural phrase copied from the policy-visible title and "
            "containing its certified attribute evidence"
        ),
        "title_identity_normalization": "Unicode NFKC, whitespace collapse, casefold",
        "catalog_title_match_count_required": 1,
        "split_assignment": (
            "native-audit successes first, then balanced deterministic order, ASIN-disjoint"
        ),
    }


def _stable_backend_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "surface",
            "price_seed",
            "product_count",
            "price_table_sha256",
            "upstream_provenance",
        )
        if key in metadata
    }
