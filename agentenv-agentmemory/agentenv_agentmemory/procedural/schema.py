from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from .scenarios import (
    SCENARIO_DEFINITION_SHA256,
    SCENARIO_DEFINITION_VERSION,
    SCENARIOS,
    ProductClassification,
    ScenarioSpec,
    SlotSpec,
    scenario_by_id,
)


PRODUCT_POOL_SCHEMA = "agentmemory_natural_attribute_product_pool_v3"
TASK_SCHEMA = "agentmemory_natural_attribute_chain_task_v2"
ORBIT_SCHEMA = "agentmemory_natural_attribute_counterfactual_orbit_v2"
PROOF_SCHEMA = "agentmemory_natural_attribute_chain_proof_v2"
SPLITS = ("train", "dev", "test")

_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,159}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProceduralMemoryDataError(ValueError):
    """Raised when generated memory data is not machine-verifiable."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProceduralMemoryDataError(f"{field} must be a lowercase SHA256.")
    return value


def require_id(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ProceduralMemoryDataError(
            f"{field} must match {_ID_RE.pattern!r}, got {value!r}."
        )
    return value


def normalize_native_title(value: str) -> str:
    """Canonicalize a visible catalog title for catalog-wide identity checks."""

    if not isinstance(value, str):
        raise ProceduralMemoryDataError("native product title must be text.")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def classification_payload(
    *,
    scenario_id: str,
    slot_id: str,
    attribute_name: str,
    attribute_value: str,
    attribute_display_name: str,
    native_category: str,
    catalog_query: str,
    product_category: str,
    slot_title_evidence: tuple[str, ...],
    attribute_title_evidence: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "slot_id": slot_id,
        "attribute_name": attribute_name,
        "attribute_value": attribute_value,
        "attribute_display_name": attribute_display_name,
        "native_category": native_category,
        "catalog_query": catalog_query,
        "product_category": product_category,
        "slot_title_evidence": list(slot_title_evidence),
        "attribute_title_evidence": list(attribute_title_evidence),
        "scenario_definition_version": SCENARIO_DEFINITION_VERSION,
        "scenario_definition_sha256": SCENARIO_DEFINITION_SHA256,
    }


@dataclass(frozen=True)
class CertifiedProduct:
    asin: str
    title: str
    scenario_id: str
    slot_id: str
    attribute_name: str
    attribute_value: str
    split: str
    price_cents: int
    search_query: str
    search_rank: int
    catalog_record_sha256: str
    native_category: str
    catalog_query: str
    product_category: str
    slot_title_evidence: tuple[str, ...]
    attribute_title_evidence: tuple[str, ...]
    classification_sha256: str
    native_title_normalized: str
    native_title_catalog_match_count: int
    native_title_globally_unique: bool
    native_search_verified: bool = True
    native_open_verified: bool = True
    native_purchase_verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.asin, str) or not _ASIN_RE.fullmatch(self.asin):
            raise ProceduralMemoryDataError(f"invalid certified ASIN {self.asin!r}.")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ProceduralMemoryDataError(f"certified product {self.asin} has no title.")
        if self.title != self.title.strip():
            raise ProceduralMemoryDataError(
                f"certified product {self.asin} title is not canonical."
            )
        expected_normalized_title = normalize_native_title(self.title)
        if not expected_normalized_title:
            raise ProceduralMemoryDataError(
                f"certified product {self.asin} has an empty normalized title."
            )
        if self.native_title_normalized != expected_normalized_title:
            raise ProceduralMemoryDataError(
                f"certified product {self.asin} normalized title is inconsistent."
            )
        if self.asin.casefold() in self.native_title_normalized:
            raise ProceduralMemoryDataError(
                f"certified product {self.asin} leaks its internal ASIN in the title."
            )
        if (
            isinstance(self.native_title_catalog_match_count, bool)
            or not isinstance(self.native_title_catalog_match_count, int)
            or self.native_title_catalog_match_count != 1
        ):
            raise ProceduralMemoryDataError(
                f"certified product {self.asin} title must match exactly one catalog ASIN."
            )
        if self.native_title_globally_unique is not True:
            raise ProceduralMemoryDataError(
                f"certified product {self.asin} lacks catalog-wide title uniqueness."
            )
        require_id(self.scenario_id, field="product scenario_id")
        require_id(self.slot_id, field="product slot_id")
        require_id(self.attribute_value, field="product attribute_value")
        if self.split not in SPLITS:
            raise ProceduralMemoryDataError(f"invalid product split {self.split!r}.")
        scenario = scenario_by_id(self.scenario_id)
        slot = scenario.slot(self.slot_id)
        if slot.attribute_name != self.attribute_name:
            raise ProceduralMemoryDataError(
                f"product {self.asin} attribute name disagrees with its slot."
            )
        try:
            value = slot.value(self.attribute_value)
        except KeyError as exc:
            raise ProceduralMemoryDataError(str(exc)) from exc
        if isinstance(self.price_cents, bool) or not isinstance(self.price_cents, int):
            raise ProceduralMemoryDataError("price_cents must be an integer.")
        if self.price_cents <= 0:
            raise ProceduralMemoryDataError("price_cents must be positive.")
        if not isinstance(self.search_query, str) or not self.search_query.strip():
            raise ProceduralMemoryDataError("search_query must be non-empty.")
        if self.search_query != self.search_query.strip():
            raise ProceduralMemoryDataError("search_query must be canonical.")
        if any(char in self.search_query for char in "[]\r\n"):
            raise ProceduralMemoryDataError(
                "search_query contains characters unsafe for native WebShop actions."
            )
        normalized_search_query = normalize_native_title(self.search_query)
        if normalized_search_query not in self.native_title_normalized:
            raise ProceduralMemoryDataError(
                "search_query must be a contiguous phrase copied from the "
                "policy-visible native title."
            )
        if len(re.findall(r"\b\w[\w'&+./-]*\b", self.search_query)) < 3:
            raise ProceduralMemoryDataError(
                "search_query must contain at least three natural title words."
            )
        if not any(
            normalize_native_title(evidence) in normalized_search_query
            for evidence in self.attribute_title_evidence
        ):
            raise ProceduralMemoryDataError(
                "search_query must retain the certified natural attribute evidence."
            )
        if isinstance(self.search_rank, bool) or not isinstance(self.search_rank, int):
            raise ProceduralMemoryDataError("search_rank must be an integer.")
        if self.search_rank < 1:
            raise ProceduralMemoryDataError("search_rank must be positive.")
        require_sha256(self.catalog_record_sha256, field="catalog_record_sha256")
        require_sha256(self.classification_sha256, field="classification_sha256")
        if not self.native_category or self.native_category != self.native_category.casefold():
            raise ProceduralMemoryDataError("native_category must be non-empty casefolded text.")
        if not self.slot_title_evidence or not self.attribute_title_evidence:
            raise ProceduralMemoryDataError(
                f"product {self.asin} lacks direct title evidence for its slot or attribute."
            )
        expected_classification = canonical_sha256(
            classification_payload(
                scenario_id=self.scenario_id,
                slot_id=self.slot_id,
                attribute_name=self.attribute_name,
                attribute_value=self.attribute_value,
                attribute_display_name=value.display_name,
                native_category=self.native_category,
                catalog_query=self.catalog_query,
                product_category=self.product_category,
                slot_title_evidence=self.slot_title_evidence,
                attribute_title_evidence=self.attribute_title_evidence,
            )
        )
        if self.classification_sha256 != expected_classification:
            raise ProceduralMemoryDataError(
                f"product {self.asin} classification proof hash is inconsistent."
            )
        if not all(
            (
                self.native_search_verified,
                self.native_open_verified,
                self.native_purchase_verified,
            )
        ):
            raise ProceduralMemoryDataError(
                f"product {self.asin} lacks complete native execution certification."
            )

    @classmethod
    def from_classification(
        cls,
        *,
        classification: ProductClassification,
        asin: str,
        title: str,
        split: str,
        price_cents: int,
        search_query: str,
        search_rank: int,
        catalog_record_sha256: str,
        native_title_catalog_match_count: int,
        native_title_globally_unique: bool,
    ) -> "CertifiedProduct":
        return cls(
            asin=asin,
            title=title,
            scenario_id=classification.scenario_id,
            slot_id=classification.slot_id,
            attribute_name=classification.attribute_name,
            attribute_value=classification.attribute_value,
            split=split,
            price_cents=price_cents,
            search_query=search_query,
            search_rank=search_rank,
            catalog_record_sha256=catalog_record_sha256,
            native_category=classification.native_category,
            catalog_query=classification.catalog_query,
            product_category=classification.product_category,
            slot_title_evidence=classification.slot_title_evidence,
            attribute_title_evidence=classification.attribute_title_evidence,
            classification_sha256=classification.semantic_sha256,
            native_title_normalized=normalize_native_title(title),
            native_title_catalog_match_count=native_title_catalog_match_count,
            native_title_globally_unique=native_title_globally_unique,
        )

    @property
    def attribute_display_name(self) -> str:
        return scenario_by_id(self.scenario_id).slot(self.slot_id).value(
            self.attribute_value
        ).display_name

    def as_dict(self) -> dict[str, Any]:
        return {
            "asin": self.asin,
            "title": self.title,
            "scenario_id": self.scenario_id,
            "slot_id": self.slot_id,
            "attribute_name": self.attribute_name,
            "attribute_value": self.attribute_value,
            "split": self.split,
            "price_cents": self.price_cents,
            "search_query": self.search_query,
            "search_rank": self.search_rank,
            "catalog_record_sha256": self.catalog_record_sha256,
            "title_identity": {
                "normalized_title": self.native_title_normalized,
                "catalog_match_count": self.native_title_catalog_match_count,
                "globally_unique": self.native_title_globally_unique,
            },
            "classification": {
                "native_category": self.native_category,
                "catalog_query": self.catalog_query,
                "product_category": self.product_category,
                "slot_title_evidence": list(self.slot_title_evidence),
                "attribute_title_evidence": list(self.attribute_title_evidence),
                "classification_sha256": self.classification_sha256,
            },
            "native_verification": {
                "search": self.native_search_verified,
                "open": self.native_open_verified,
                "purchase": self.native_purchase_verified,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertifiedProduct":
        required = {
            "asin",
            "title",
            "scenario_id",
            "slot_id",
            "attribute_name",
            "attribute_value",
            "split",
            "price_cents",
            "search_query",
            "search_rank",
            "catalog_record_sha256",
            "title_identity",
            "classification",
            "native_verification",
        }
        _require_exact_keys(payload, required=required, context="certified product")
        classification = payload["classification"]
        title_identity = payload["title_identity"]
        verification = payload["native_verification"]
        if not isinstance(classification, Mapping):
            raise ProceduralMemoryDataError("product classification must be an object.")
        if not isinstance(title_identity, Mapping):
            raise ProceduralMemoryDataError("product title identity must be an object.")
        if not isinstance(verification, Mapping):
            raise ProceduralMemoryDataError("native verification must be an object.")
        _require_exact_keys(
            title_identity,
            required={"normalized_title", "catalog_match_count", "globally_unique"},
            context="product title identity",
        )
        _require_exact_keys(
            classification,
            required={
                "native_category",
                "catalog_query",
                "product_category",
                "slot_title_evidence",
                "attribute_title_evidence",
                "classification_sha256",
            },
            context="product classification",
        )
        _require_exact_keys(
            verification,
            required={"search", "open", "purchase"},
            context="native verification",
        )
        return cls(
            **{
                key: payload[key]
                for key in required
                - {"classification", "title_identity", "native_verification"}
            },
            native_category=classification["native_category"],
            catalog_query=classification["catalog_query"],
            product_category=classification["product_category"],
            slot_title_evidence=tuple(classification["slot_title_evidence"]),
            attribute_title_evidence=tuple(classification["attribute_title_evidence"]),
            classification_sha256=classification["classification_sha256"],
            native_title_normalized=title_identity["normalized_title"],
            native_title_catalog_match_count=title_identity["catalog_match_count"],
            native_title_globally_unique=title_identity["globally_unique"],
            native_search_verified=verification["search"],
            native_open_verified=verification["open"],
            native_purchase_verified=verification["purchase"],
        )


@dataclass(frozen=True)
class ProductPool:
    pool_id: str
    certifier_version: str
    scenario_ids: tuple[str, ...]
    products_per_cell: int
    products: tuple[CertifiedProduct, ...]
    catalog_sha256: str
    attributes_sha256: str
    price_table_sha256: str
    lucene_index_sha256: str
    source_manifest_sha256: str
    scenario_definition_version: str = SCENARIO_DEFINITION_VERSION
    scenario_definition_sha256: str = SCENARIO_DEFINITION_SHA256

    def __post_init__(self) -> None:
        require_id(self.pool_id, field="pool_id")
        require_id(self.certifier_version, field="certifier_version")
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ProceduralMemoryDataError("pool scenario IDs must be non-empty and unique.")
        known_order = tuple(
            scenario.scenario_id
            for scenario in SCENARIOS
            if scenario.scenario_id in set(self.scenario_ids)
        )
        if self.scenario_ids != known_order:
            raise ProceduralMemoryDataError(
                "pool scenario IDs must be a canonical ordered subset of built-in scenarios."
            )
        if self.scenario_definition_version != SCENARIO_DEFINITION_VERSION:
            raise ProceduralMemoryDataError("scenario definition version mismatch.")
        if self.scenario_definition_sha256 != SCENARIO_DEFINITION_SHA256:
            raise ProceduralMemoryDataError("scenario definition SHA256 mismatch.")
        if (
            isinstance(self.products_per_cell, bool)
            or not isinstance(self.products_per_cell, int)
            or self.products_per_cell < 1
        ):
            raise ProceduralMemoryDataError("products_per_cell must be positive.")
        asins = tuple(product.asin for product in self.products)
        if len(set(asins)) != len(asins):
            raise ProceduralMemoryDataError(
                "each certified ASIN must belong to exactly one cell and split."
            )
        normalized_titles = tuple(
            product.native_title_normalized for product in self.products
        )
        if len(set(normalized_titles)) != len(normalized_titles):
            raise ProceduralMemoryDataError(
                "certified pool contains duplicate normalized product titles."
            )
        expected_cells = {
            (scenario_id, slot.slot_id, value.value_id, split)
            for scenario_id in self.scenario_ids
            for slot in scenario_by_id(scenario_id).slots
            for value in slot.values
            for split in SPLITS
        }
        counts = {cell: 0 for cell in expected_cells}
        for product in self.products:
            cell = (
                product.scenario_id,
                product.slot_id,
                product.attribute_value,
                product.split,
            )
            if cell not in counts:
                raise ProceduralMemoryDataError(
                    f"product {product.asin} references a cell outside this pool: {cell}."
                )
            counts[cell] += 1
        wrong = {
            "/".join(cell): count
            for cell, count in counts.items()
            if count != self.products_per_cell
        }
        if wrong:
            raise ProceduralMemoryDataError(
                "every scenario/slot/attribute/split cell must have the declared exact "
                f"product count {self.products_per_cell}: {dict(sorted(wrong.items()))}."
            )
        for field, value in (
            ("catalog_sha256", self.catalog_sha256),
            ("attributes_sha256", self.attributes_sha256),
            ("price_table_sha256", self.price_table_sha256),
            ("lucene_index_sha256", self.lucene_index_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
        ):
            require_sha256(value, field=field)

    @property
    def scenarios(self) -> tuple[ScenarioSpec, ...]:
        return tuple(scenario_by_id(value) for value in self.scenario_ids)

    def products_for_split(
        self,
        scenario_id: str,
        slot_id: str,
        attribute_value: str,
        split: str,
    ) -> tuple[CertifiedProduct, ...]:
        if split not in SPLITS:
            raise ProceduralMemoryDataError(
                f"invalid product split {split!r}; expected one of {SPLITS}."
            )
        values = tuple(
            sorted(
                (
                    product
                    for product in self.products
                    if product.scenario_id == scenario_id
                    and product.slot_id == slot_id
                    and product.attribute_value == attribute_value
                    and product.split == split
                ),
                key=lambda product: product.asin,
            )
        )
        if len(values) != self.products_per_cell:
            raise ProceduralMemoryDataError(
                "pool cell changed after validation: "
                f"{scenario_id}/{slot_id}/{attribute_value}/{split}."
            )
        return values

    def semantic_manifest(self) -> dict[str, Any]:
        return {
            "schema": PRODUCT_POOL_SCHEMA,
            "pool_id": self.pool_id,
            "certifier_version": self.certifier_version,
            "scenario_definition": {
                "version": self.scenario_definition_version,
                "sha256": self.scenario_definition_sha256,
                "scenario_ids": list(self.scenario_ids),
            },
            "products_per_cell": self.products_per_cell,
            "products": [
                product.as_dict()
                for product in sorted(self.products, key=lambda value: value.asin)
            ],
            "provenance": {
                "catalog_sha256": self.catalog_sha256,
                "attributes_sha256": self.attributes_sha256,
                "price_table_sha256": self.price_table_sha256,
                "lucene_index_sha256": self.lucene_index_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
            },
        }

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.semantic_manifest())


@dataclass(frozen=True)
class ProceduralCandidate:
    product: CertifiedProduct

    @property
    def asin(self) -> str:
        return self.product.asin

    @property
    def title(self) -> str:
        return self.product.title

    @property
    def price_cents(self) -> int:
        return self.product.price_cents

    @property
    def attribute_value(self) -> str:
        return self.product.attribute_value

    @property
    def attribute_display_name(self) -> str:
        return self.product.attribute_display_name

    def as_dict(self) -> dict[str, Any]:
        return {"product": self.product.as_dict()}


@dataclass(frozen=True)
class AttributeTransition:
    previous_slot_id: str
    previous_attribute_name: str
    current_attribute_name: str
    pairs: tuple[tuple[str, str], tuple[str, str]]

    def __post_init__(self) -> None:
        require_id(self.previous_slot_id, field="transition previous_slot_id")
        if len(self.pairs) != 2:
            raise ProceduralMemoryDataError("transition must contain exactly two pairs.")
        inputs = tuple(pair[0] for pair in self.pairs)
        outputs = tuple(pair[1] for pair in self.pairs)
        if len(set(inputs)) != 2 or len(set(outputs)) != 2:
            raise ProceduralMemoryDataError("transition must be a complete binary bijection.")
        for value in (*inputs, *outputs):
            require_id(value, field="transition attribute value")
        if not self.previous_attribute_name or not self.current_attribute_name:
            raise ProceduralMemoryDataError("transition attribute names must be non-empty.")

    def resolve(self, previous_value: str) -> str:
        matches = [output for value, output in self.pairs if value == previous_value]
        if len(matches) != 1:
            raise ProceduralMemoryDataError(
                f"transition does not map previous value {previous_value!r} exactly once."
            )
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "previous_slot_id": self.previous_slot_id,
            "previous_attribute_name": self.previous_attribute_name,
            "current_attribute_name": self.current_attribute_name,
            "pairs": [list(pair) for pair in self.pairs],
        }


@dataclass(frozen=True)
class ProceduralPhase:
    phase_index: int
    scenario_id: str
    slot_id: str
    display_name: str
    attribute_name: str
    candidates: tuple[ProceduralCandidate, ProceduralCandidate]
    question: str
    target_asin: str
    root_attribute_value: str | None = None
    transition: AttributeTransition | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.phase_index, bool)
            or not isinstance(self.phase_index, int)
            or not 0 <= self.phase_index < 6
        ):
            raise ProceduralMemoryDataError("phase_index must be an integer in [0, 5].")
        scenario = scenario_by_id(self.scenario_id)
        slot = scenario.slot(self.slot_id)
        if slot.display_name != self.display_name or slot.attribute_name != self.attribute_name:
            raise ProceduralMemoryDataError("phase metadata disagrees with its scenario slot.")
        if len(self.candidates) != 2:
            raise ProceduralMemoryDataError(
                "the memory-focused phase requires exactly two candidates."
            )
        if len({candidate.asin for candidate in self.candidates}) != 2:
            raise ProceduralMemoryDataError("phase candidates must have distinct ASINs.")
        if {candidate.attribute_value for candidate in self.candidates} != set(
            slot.value_ids
        ):
            raise ProceduralMemoryDataError(
                "phase candidates must contain exactly one product for each natural "
                "attribute value."
            )
        if any(
            candidate.product.scenario_id != self.scenario_id
            or candidate.product.slot_id != self.slot_id
            for candidate in self.candidates
        ):
            raise ProceduralMemoryDataError(
                "every phase candidate must belong to the phase scenario and slot."
            )
        if not isinstance(self.question, str) or not self.question.strip():
            raise ProceduralMemoryDataError("phase question must be non-empty.")
        if self.target_asin not in {candidate.asin for candidate in self.candidates}:
            raise ProceduralMemoryDataError(
                "phase target must be one of the two certified candidates."
            )
        if self.phase_index == 0:
            if self.root_attribute_value not in slot.value_ids or self.transition is not None:
                raise ProceduralMemoryDataError(
                    "phase zero requires one natural root value and no transition."
                )
        elif self.root_attribute_value is not None or self.transition is None:
            raise ProceduralMemoryDataError(
                "later phases require one transition and no repeated root value."
            )

    @property
    def target_attribute_value(self) -> str:
        return next(
            candidate.attribute_value
            for candidate in self.candidates
            if candidate.asin == self.target_asin
        )

    def as_dict(self, *, include_target: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase_index": self.phase_index,
            "scenario_id": self.scenario_id,
            "slot_id": self.slot_id,
            "display_name": self.display_name,
            "attribute_name": self.attribute_name,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "question": self.question,
            "root_attribute_value": self.root_attribute_value,
            "transition": self.transition.as_dict() if self.transition else None,
        }
        if include_target:
            payload["target_asin"] = self.target_asin
        return payload


@dataclass(frozen=True)
class ProceduralTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    scenario_id: str
    root_attribute_value: str
    split: str
    budget_cents: int
    phases: tuple[ProceduralPhase, ...]
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        require_id(self.task_id, field="task_id")
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.scenario_id, field="task scenario_id")
        require_id(self.root_attribute_value, field="task root_attribute_value")
        scenario = scenario_by_id(self.scenario_id)
        if self.root_attribute_value not in scenario.slots[0].value_ids:
            raise ProceduralMemoryDataError("task root value is invalid for its first slot.")
        if self.split not in SPLITS:
            raise ProceduralMemoryDataError(f"invalid split {self.split!r}.")
        for field, value in (
            ("orbit_index", self.orbit_index),
            ("semantic_epoch", self.semantic_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProceduralMemoryDataError(f"{field} must be non-negative.")
        if len(self.phases) != 6:
            raise ProceduralMemoryDataError("procedural task must contain six phases.")
        if tuple(phase.phase_index for phase in self.phases) != tuple(range(6)):
            raise ProceduralMemoryDataError(
                "procedural task phases must be ordered exactly from 0 through 5."
            )
        if tuple(phase.slot_id for phase in self.phases) != tuple(
            slot.slot_id for slot in scenario.slots
        ):
            raise ProceduralMemoryDataError("task phase slots disagree with the scenario.")
        if any(phase.scenario_id != self.scenario_id for phase in self.phases):
            raise ProceduralMemoryDataError("task phase scenario mismatch.")
        if self.phases[0].root_attribute_value != self.root_attribute_value:
            raise ProceduralMemoryDataError("task and first phase root values disagree.")
        if self.budget_cents <= 0:
            raise ProceduralMemoryDataError("task budget must be positive.")
        if isinstance(self.generator_seed, bool) or not isinstance(self.generator_seed, int):
            raise ProceduralMemoryDataError("generator_seed must be an integer.")
        require_id(self.generator_version, field="generator_version")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(phase.question for phase in self.phases)

    @property
    def target_asins(self) -> tuple[str, ...]:
        return tuple(phase.target_asin for phase in self.phases)

    @property
    def target_attribute_values(self) -> tuple[str, ...]:
        return tuple(phase.target_attribute_value for phase in self.phases)

    def semantic_manifest(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "root_attribute_value": self.root_attribute_value,
            "split": self.split,
            "budget_cents": self.budget_cents,
            "phases": [phase.as_dict(include_target=False) for phase in self.phases],
        }

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.semantic_manifest())

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        return {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "scenario_id": self.scenario_id,
            "root_attribute_value": self.root_attribute_value,
            "split": self.split,
            "budget_cents": self.budget_cents,
            "phases": [
                phase.as_dict(include_target=include_targets) for phase in self.phases
            ],
            "generator": {
                "version": self.generator_version,
                "seed": self.generator_seed,
                "product_pool_sha256": self.product_pool_sha256,
            },
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class CounterfactualOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    scenario_id: str
    root_attribute_values: tuple[str, str]
    tasks: tuple[ProceduralTask, ProceduralTask]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.scenario_id, field="orbit scenario_id")
        for field, value in (
            ("orbit_index", self.orbit_index),
            ("semantic_epoch", self.semantic_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProceduralMemoryDataError(f"{field} must be non-negative.")
        if len(self.root_attribute_values) != 2 or len(set(self.root_attribute_values)) != 2:
            raise ProceduralMemoryDataError(
                "counterfactual orbit requires exactly two distinct root values."
            )
        expected_roots = set(scenario_by_id(self.scenario_id).slots[0].value_ids)
        if set(self.root_attribute_values) != expected_roots:
            raise ProceduralMemoryDataError("orbit roots must cover both first-slot values.")
        if len(self.tasks) != 2:
            raise ProceduralMemoryDataError("counterfactual orbit requires exactly two tasks.")
        if tuple(task.root_attribute_value for task in self.tasks) != self.root_attribute_values:
            raise ProceduralMemoryDataError("orbit task order must match declared roots.")
        if any(
            task.orbit_id != self.orbit_id
            or task.orbit_index != self.orbit_index
            or task.semantic_epoch != self.semantic_epoch
            or task.scenario_id != self.scenario_id
            for task in self.tasks
        ):
            raise ProceduralMemoryDataError("orbit task metadata mismatch.")

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(
            {
                "scenario_id": self.scenario_id,
                "tasks": [task.semantic_manifest() for task in self.tasks],
            }
        )

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        return {
            "schema": ORBIT_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "scenario_id": self.scenario_id,
            "root_attribute_values": list(self.root_attribute_values),
            "tasks": [
                task.as_dict(include_targets=include_targets) for task in self.tasks
            ],
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class ProceduralMemoryBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    target_attribute_values: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    scenario_id: str
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        require_id(self.task_id, field="bundle task_id")
        require_id(self.orbit_id, field="bundle orbit_id")
        require_id(self.scenario_id, field="bundle scenario_id")
        require_id(self.generator_version, field="bundle generator_version")
        require_sha256(self.proof_sha256, field="bundle proof_sha256")
        require_sha256(self.product_pool_sha256, field="bundle product_pool_sha256")
        if (
            len(self.questions) != 6
            or len(self.target_asins) != 6
            or len(self.target_attribute_values) != 6
        ):
            raise ProceduralMemoryDataError(
                "procedural runtime bundle must contain six questions, targets, and "
                "natural target attributes."
            )
        if self.split not in SPLITS:
            raise ProceduralMemoryDataError(f"invalid bundle split {self.split!r}.")
        if self.budget_cents <= 0:
            raise ProceduralMemoryDataError("bundle budget must be positive.")


def assign_product_split(pool_id: str, product: CertifiedProduct) -> str:
    """Return the frozen pool split; retained as a narrow compatibility helper."""

    require_id(pool_id, field="pool_id")
    return product.split


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ProceduralMemoryDataError(f"{context} must be an object.")
    observed = set(payload)
    if observed != required:
        raise ProceduralMemoryDataError(
            f"{context} fields mismatch: missing={sorted(required - observed)} "
            f"extra={sorted(observed - required)}."
        )
