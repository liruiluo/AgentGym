from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping


SPLITS = ("train", "dev", "test")
PHASE_KINDS = ("evidence", "application")
POOL_SCHEMA = "agentmemory_latent_preference_product_pool_v2"
TASK_SCHEMA = "agentmemory_latent_preference_task_v1"
ORBIT_SCHEMA = "agentmemory_latent_preference_counterfactual_orbit_v1"
PROOF_SCHEMA = "agentmemory_latent_preference_proof_v1"
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LatentPreferenceDataError(ValueError):
    pass


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def normalize_native_title(value: str) -> str:
    if not isinstance(value, str):
        raise LatentPreferenceDataError("native product title must be text.")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def preference_classification_payload(
    *,
    asin: str,
    title: str,
    product_category: str,
    category_title_evidence: tuple[str, ...],
    category_id: str,
    axis: str,
    attribute_value: str,
    title_evidence: tuple[str, ...],
    guard_matches: tuple[str, ...],
    source_candidate_sha256: str,
) -> dict[str, Any]:
    return {
        "asin": asin,
        "title": title,
        "product_category": product_category,
        "category_title_evidence": list(category_title_evidence),
        "category_id": category_id,
        "axis": axis,
        "attribute_value": attribute_value,
        "title_evidence": list(title_evidence),
        "guard_matches": list(guard_matches),
        "source_candidate_sha256": source_candidate_sha256,
    }


def require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LatentPreferenceDataError(f"{field} must be a lowercase SHA256.")
    return value


def require_id(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise LatentPreferenceDataError(
            f"{field} must be a lowercase ASCII identifier."
        )
    return value


def _require_display(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise LatentPreferenceDataError(f"{field} must be one non-empty line.")
    return value


@dataclass(frozen=True)
class PreferenceRecipe:
    recipe_id: str
    axis: str
    axis_display_name: str
    values: tuple[str, str]
    value_display_names: tuple[str, str]
    categories: tuple[str, str, str, str]
    category_display_names: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        require_id(self.recipe_id, field="recipe_id")
        require_id(self.axis, field="recipe axis")
        _require_display(self.axis_display_name, field="axis_display_name")
        if len(self.values) != 2 or len(set(self.values)) != 2:
            raise LatentPreferenceDataError(
                "preference recipe must contain two distinct values."
            )
        for value in self.values:
            require_id(value, field="recipe value")
        if len(self.value_display_names) != 2:
            raise LatentPreferenceDataError(
                "preference recipe must contain two value display names."
            )
        for display in self.value_display_names:
            _require_display(display, field="value display name")
        if len(self.categories) != 4 or len(set(self.categories)) != 4:
            raise LatentPreferenceDataError(
                "preference recipe must contain four distinct categories."
            )
        for category in self.categories:
            require_id(category, field="recipe category")
        if len(self.category_display_names) != 4:
            raise LatentPreferenceDataError(
                "preference recipe must contain four category display names."
            )
        for display in self.category_display_names:
            _require_display(display, field="category display name")

    def category_display_name(self, category_id: str) -> str:
        try:
            return self.category_display_names[self.categories.index(category_id)]
        except ValueError as exc:
            raise KeyError(category_id) from exc

    def value_display_name(self, value_id: str) -> str:
        try:
            return self.value_display_names[self.values.index(value_id)]
        except ValueError as exc:
            raise KeyError(value_id) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "axis": self.axis,
            "axis_display_name": self.axis_display_name,
            "values": list(self.values),
            "value_display_names": list(self.value_display_names),
            "categories": list(self.categories),
            "category_display_names": list(self.category_display_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreferenceRecipe":
        _require_exact_keys(
            payload,
            required={
                "recipe_id",
                "axis",
                "axis_display_name",
                "values",
                "value_display_names",
                "categories",
                "category_display_names",
            },
            context="preference recipe",
        )
        return cls(
            recipe_id=payload["recipe_id"],
            axis=payload["axis"],
            axis_display_name=payload["axis_display_name"],
            values=tuple(payload["values"]),
            value_display_names=tuple(payload["value_display_names"]),
            categories=tuple(payload["categories"]),
            category_display_names=tuple(payload["category_display_names"]),
        )


@dataclass(frozen=True)
class CertifiedPreferenceProduct:
    asin: str
    title: str
    native_title_normalized: str
    price_cents: int
    product_category: str
    category_title_evidence: tuple[str, ...]
    category_id: str
    category_display_name: str
    axis: str
    attribute_value: str
    attribute_display_name: str
    split: str
    search_query: str
    search_rank: int
    catalog_record_sha256: str
    title_evidence: tuple[str, ...]
    guard_matches: tuple[str, ...]
    classification_sha256: str
    source_candidate_sha256: str
    native_title_catalog_match_count: int = 1
    native_title_globally_unique: bool = True
    native_search_verified: bool = True
    native_open_verified: bool = True
    native_purchase_verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.asin, str) or not _ASIN_RE.fullmatch(self.asin):
            raise LatentPreferenceDataError(f"invalid product ASIN {self.asin!r}.")
        if not isinstance(self.title, str) or not 8 <= len(self.title) <= 240:
            raise LatentPreferenceDataError("product title length is outside [8, 240].")
        if self.native_title_normalized != normalize_native_title(self.title):
            raise LatentPreferenceDataError("normalized native title mismatch.")
        if self.asin.casefold() in self.native_title_normalized:
            raise LatentPreferenceDataError("product title leaks its internal ASIN.")
        if (
            isinstance(self.price_cents, bool)
            or not isinstance(self.price_cents, int)
            or self.price_cents <= 0
        ):
            raise LatentPreferenceDataError("product price must be positive cents.")
        _require_display(self.product_category, field="product_category")
        if not self.category_title_evidence or any(
            not isinstance(value, str) or not value.strip()
            for value in self.category_title_evidence
        ):
            raise LatentPreferenceDataError(
                "product category title evidence must be non-empty text."
            )
        if any(
            normalize_native_title(value) not in self.native_title_normalized
            for value in self.category_title_evidence
        ):
            raise LatentPreferenceDataError(
                "product category evidence must occur in its native title."
            )
        require_id(self.category_id, field="product category_id")
        _require_display(self.category_display_name, field="category_display_name")
        require_id(self.axis, field="product axis")
        require_id(self.attribute_value, field="product attribute_value")
        _require_display(self.attribute_display_name, field="attribute_display_name")
        if self.split not in SPLITS:
            raise LatentPreferenceDataError(f"invalid product split {self.split!r}.")
        if (
            not isinstance(self.search_query, str)
            or not self.search_query.strip()
            or "[" in self.search_query
            or "]" in self.search_query
            or "\n" in self.search_query
        ):
            raise LatentPreferenceDataError("product search query is unsafe.")
        if (
            isinstance(self.search_rank, bool)
            or not isinstance(self.search_rank, int)
            or self.search_rank < 1
        ):
            raise LatentPreferenceDataError("product search rank must be positive.")
        normalized_query = normalize_native_title(self.search_query)
        if normalized_query not in self.native_title_normalized:
            raise LatentPreferenceDataError(
                "product search query must be copied contiguously from its title."
            )
        if not self.title_evidence or any(
            not isinstance(value, str) or not value.strip()
            for value in self.title_evidence
        ):
            raise LatentPreferenceDataError("product title evidence must be non-empty text.")
        if not any(
            normalize_native_title(value) in normalized_query
            for value in self.title_evidence
        ):
            raise LatentPreferenceDataError(
                "product search query must retain its certified preference evidence."
            )
        if self.guard_matches != (self.attribute_value,):
            raise LatentPreferenceDataError(
                "product broad attribute guard must have exactly one matching value."
            )
        require_sha256(self.catalog_record_sha256, field="catalog_record_sha256")
        require_sha256(self.source_candidate_sha256, field="source_candidate_sha256")
        expected_classification_sha256 = canonical_sha256(
            preference_classification_payload(
                asin=self.asin,
                title=self.title,
                product_category=self.product_category,
                category_title_evidence=self.category_title_evidence,
                category_id=self.category_id,
                axis=self.axis,
                attribute_value=self.attribute_value,
                title_evidence=self.title_evidence,
                guard_matches=self.guard_matches,
                source_candidate_sha256=self.source_candidate_sha256,
            )
        )
        if self.classification_sha256 != expected_classification_sha256:
            raise LatentPreferenceDataError("product classification proof hash mismatch.")
        if (
            isinstance(self.native_title_catalog_match_count, bool)
            or not isinstance(self.native_title_catalog_match_count, int)
            or self.native_title_catalog_match_count != 1
            or self.native_title_globally_unique is not True
        ):
            raise LatentPreferenceDataError(
                "product title must map to exactly one ASIN in the frozen catalog."
            )
        if not all(
            (
                self.native_search_verified,
                self.native_open_verified,
                self.native_purchase_verified,
            )
        ):
            raise LatentPreferenceDataError(
                "product lacks complete native search/open/purchase certification."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "asin": self.asin,
            "title": self.title,
            "native_title_normalized": self.native_title_normalized,
            "price_cents": self.price_cents,
            "product_category": self.product_category,
            "category_title_evidence": list(self.category_title_evidence),
            "category_id": self.category_id,
            "category_display_name": self.category_display_name,
            "axis": self.axis,
            "attribute_value": self.attribute_value,
            "attribute_display_name": self.attribute_display_name,
            "split": self.split,
            "search_query": self.search_query,
            "search_rank": self.search_rank,
            "catalog_record_sha256": self.catalog_record_sha256,
            "title_evidence": list(self.title_evidence),
            "guard_matches": list(self.guard_matches),
            "classification_sha256": self.classification_sha256,
            "source_candidate_sha256": self.source_candidate_sha256,
            "native_title_catalog_match_count": self.native_title_catalog_match_count,
            "native_title_globally_unique": self.native_title_globally_unique,
            "native_search_verified": self.native_search_verified,
            "native_open_verified": self.native_open_verified,
            "native_purchase_verified": self.native_purchase_verified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertifiedPreferenceProduct":
        _require_exact_keys(
            payload,
            required={
                "asin",
                "title",
                "native_title_normalized",
                "price_cents",
                "product_category",
                "category_title_evidence",
                "category_id",
                "category_display_name",
                "axis",
                "attribute_value",
                "attribute_display_name",
                "split",
                "search_query",
                "search_rank",
                "catalog_record_sha256",
                "title_evidence",
                "guard_matches",
                "classification_sha256",
                "source_candidate_sha256",
                "native_title_catalog_match_count",
                "native_title_globally_unique",
                "native_search_verified",
                "native_open_verified",
                "native_purchase_verified",
            },
            context="certified preference product",
        )
        values = dict(payload)
        values["category_title_evidence"] = tuple(
            values["category_title_evidence"]
        )
        values["title_evidence"] = tuple(values["title_evidence"])
        values["guard_matches"] = tuple(values["guard_matches"])
        return cls(**values)


@dataclass(frozen=True)
class PreferenceProductPool:
    pool_id: str
    certifier_version: str
    products_per_cell: int
    recipes: tuple[PreferenceRecipe, ...]
    products: tuple[CertifiedPreferenceProduct, ...]
    catalog_sha256: str
    attributes_sha256: str
    price_table_sha256: str
    lucene_index_sha256: str
    candidate_artifact_sha256: str
    rules_sha256: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        require_id(self.pool_id, field="pool_id")
        require_id(self.certifier_version, field="certifier_version")
        if (
            isinstance(self.products_per_cell, bool)
            or not isinstance(self.products_per_cell, int)
            or self.products_per_cell < 2
        ):
            raise LatentPreferenceDataError(
                "products_per_cell must be at least two because category schedules "
                "reuse two categories."
            )
        if not self.recipes:
            raise LatentPreferenceDataError("preference pool must contain recipes.")
        recipe_ids = tuple(recipe.recipe_id for recipe in self.recipes)
        if recipe_ids != tuple(sorted(recipe_ids)) or len(set(recipe_ids)) != len(
            recipe_ids
        ):
            raise LatentPreferenceDataError(
                "preference recipes must be unique and sorted by recipe_id."
            )
        expected_cells: set[tuple[str, str, str, str]] = set()
        category_displays: dict[tuple[str, str], str] = {}
        value_displays: dict[tuple[str, str], str] = {}
        for recipe in self.recipes:
            for category, category_display in zip(
                recipe.categories, recipe.category_display_names
            ):
                category_key = (recipe.axis, category)
                prior_category_display = category_displays.setdefault(
                    category_key, category_display
                )
                if prior_category_display != category_display:
                    raise LatentPreferenceDataError(
                        f"inconsistent display names for category {category_key}."
                    )
                for value, value_display in zip(
                    recipe.values, recipe.value_display_names
                ):
                    value_key = (recipe.axis, value)
                    prior_value_display = value_displays.setdefault(
                        value_key, value_display
                    )
                    if prior_value_display != value_display:
                        raise LatentPreferenceDataError(
                            f"inconsistent display names for value {value_key}."
                        )
                    for split in SPLITS:
                        expected_cells.add((recipe.axis, category, value, split))
        asins = tuple(product.asin for product in self.products)
        if len(asins) != len(set(asins)):
            raise LatentPreferenceDataError("certified preference ASINs must be unique.")
        titles = tuple(product.native_title_normalized for product in self.products)
        if len(titles) != len(set(titles)):
            raise LatentPreferenceDataError(
                "certified preference titles must be globally unique."
            )
        product_order = tuple(
            (
                product.axis,
                product.category_id,
                product.attribute_value,
                product.split,
                product.asin,
            )
            for product in self.products
        )
        if product_order != tuple(sorted(product_order)):
            raise LatentPreferenceDataError(
                "certified preference products must use canonical cell/ASIN order."
            )
        counts = {cell: 0 for cell in expected_cells}
        for product in self.products:
            cell = (
                product.axis,
                product.category_id,
                product.attribute_value,
                product.split,
            )
            if cell not in counts:
                raise LatentPreferenceDataError(
                    f"product {product.asin} belongs to undeclared cell {cell}."
                )
            if product.category_display_name != category_displays[
                (product.axis, product.category_id)
            ]:
                raise LatentPreferenceDataError(
                    f"product {product.asin} category display name disagrees with recipe."
                )
            if product.attribute_display_name != value_displays[
                (product.axis, product.attribute_value)
            ]:
                raise LatentPreferenceDataError(
                    f"product {product.asin} value display name disagrees with recipe."
                )
            counts[cell] += 1
        mismatched = {
            "/".join(cell): count
            for cell, count in sorted(counts.items())
            if count != self.products_per_cell
        }
        if mismatched:
            raise LatentPreferenceDataError(
                "preference product cells are not balanced: " + repr(mismatched)
            )
        for field_name in (
            "catalog_sha256",
            "attributes_sha256",
            "price_table_sha256",
            "lucene_index_sha256",
            "candidate_artifact_sha256",
            "rules_sha256",
            "source_manifest_sha256",
        ):
            require_sha256(getattr(self, field_name), field=field_name)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.semantic_manifest())

    def recipe_by_id(self, recipe_id: str) -> PreferenceRecipe:
        for recipe in self.recipes:
            if recipe.recipe_id == recipe_id:
                return recipe
        raise KeyError(recipe_id)

    def products_for(
        self,
        *,
        axis: str,
        category_id: str,
        attribute_value: str,
        split: str,
    ) -> tuple[CertifiedPreferenceProduct, ...]:
        values = tuple(
            product
            for product in self.products
            if product.axis == axis
            and product.category_id == category_id
            and product.attribute_value == attribute_value
            and product.split == split
        )
        if len(values) != self.products_per_cell:
            raise LatentPreferenceDataError(
                f"pool cell {(axis, category_id, attribute_value, split)} is incomplete."
            )
        return values

    def product_by_asin(self, asin: str) -> CertifiedPreferenceProduct:
        for product in self.products:
            if product.asin == asin:
                return product
        raise KeyError(asin)

    def semantic_manifest(self) -> dict[str, Any]:
        return {
            "schema": POOL_SCHEMA,
            "pool_id": self.pool_id,
            "certifier_version": self.certifier_version,
            "products_per_cell": self.products_per_cell,
            "recipes": [recipe.as_dict() for recipe in self.recipes],
            "products": [product.as_dict() for product in self.products],
            "provenance": {
                "catalog_sha256": self.catalog_sha256,
                "attributes_sha256": self.attributes_sha256,
                "price_table_sha256": self.price_table_sha256,
                "lucene_index_sha256": self.lucene_index_sha256,
                "candidate_artifact_sha256": self.candidate_artifact_sha256,
                "rules_sha256": self.rules_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
            },
        }
@dataclass(frozen=True)
class PreferenceCandidate:
    asin: str
    title: str
    price_cents: int
    category_id: str
    category_display_name: str
    axis: str
    attribute_value: str
    attribute_display_name: str
    split: str
    product_pool_sha256: str

    @classmethod
    def from_product(
        cls,
        product: CertifiedPreferenceProduct,
        *,
        product_pool_sha256: str,
    ) -> "PreferenceCandidate":
        return cls(
            asin=product.asin,
            title=product.title,
            price_cents=product.price_cents,
            category_id=product.category_id,
            category_display_name=product.category_display_name,
            axis=product.axis,
            attribute_value=product.attribute_value,
            attribute_display_name=product.attribute_display_name,
            split=product.split,
            product_pool_sha256=product_pool_sha256,
        )

    def __post_init__(self) -> None:
        if not _ASIN_RE.fullmatch(self.asin):
            raise LatentPreferenceDataError("invalid preference candidate ASIN.")
        if not isinstance(self.title, str) or not 8 <= len(self.title) <= 240:
            raise LatentPreferenceDataError("invalid preference candidate title.")
        if (
            isinstance(self.price_cents, bool)
            or not isinstance(self.price_cents, int)
            or self.price_cents <= 0
        ):
            raise LatentPreferenceDataError("candidate price must be positive cents.")
        require_id(self.category_id, field="candidate category_id")
        _require_display(
            self.category_display_name, field="candidate category display name"
        )
        require_id(self.axis, field="candidate axis")
        require_id(self.attribute_value, field="candidate attribute value")
        _require_display(
            self.attribute_display_name, field="candidate attribute display name"
        )
        if self.split not in SPLITS:
            raise LatentPreferenceDataError("invalid preference candidate split.")
        require_sha256(self.product_pool_sha256, field="candidate product pool SHA256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "asin": self.asin,
            "title": self.title,
            "price_cents": self.price_cents,
            "category_id": self.category_id,
            "category_display_name": self.category_display_name,
            "axis": self.axis,
            "attribute_value": self.attribute_value,
            "attribute_display_name": self.attribute_display_name,
            "split": self.split,
            "product_pool_sha256": self.product_pool_sha256,
        }


@dataclass(frozen=True)
class PreferencePhase:
    phase_index: int
    phase_kind: str
    category_id: str
    category_display_name: str
    candidates: tuple[PreferenceCandidate, PreferenceCandidate]
    question: str
    target_asin: str
    confirmed_attribute_value: str | None

    def __post_init__(self) -> None:
        if self.phase_index not in range(6):
            raise LatentPreferenceDataError("preference phase index must be in [0, 6).")
        if self.phase_kind not in PHASE_KINDS:
            raise LatentPreferenceDataError(f"invalid phase kind {self.phase_kind!r}.")
        if len(self.candidates) != 2:
            raise LatentPreferenceDataError("preference phase requires two candidates.")
        if len({candidate.asin for candidate in self.candidates}) != 2:
            raise LatentPreferenceDataError("preference phase candidates must differ.")
        require_id(self.category_id, field="phase category_id")
        _require_display(self.category_display_name, field="phase category display name")
        for candidate in self.candidates:
            if (
                candidate.category_id != self.category_id
                or candidate.category_display_name != self.category_display_name
            ):
                raise LatentPreferenceDataError(
                    "phase candidate category metadata must match the phase."
                )
        if not isinstance(self.question, str) or not self.question.strip():
            raise LatentPreferenceDataError("preference phase question must be non-empty.")
        if self.target_asin not in {candidate.asin for candidate in self.candidates}:
            raise LatentPreferenceDataError("phase target must be one approved candidate.")
        if self.phase_kind == "evidence" and self.confirmed_attribute_value is None:
            raise LatentPreferenceDataError("evidence phase must declare confirmed value.")
        if self.phase_kind == "application" and self.confirmed_attribute_value is not None:
            raise LatentPreferenceDataError(
                "application phase cannot expose a confirmed value."
            )

    def as_dict(self, *, include_target: bool = True) -> dict[str, Any]:
        payload = {
            "phase_index": self.phase_index,
            "phase_kind": self.phase_kind,
            "category_id": self.category_id,
            "category_display_name": self.category_display_name,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "question": self.question,
            "confirmed_attribute_value": self.confirmed_attribute_value,
        }
        if include_target:
            payload["target_asin"] = self.target_asin
        return payload


@dataclass(frozen=True)
class LatentPreferenceTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    user_id: str
    split: str
    preferred_attribute_value: str
    supporting_evidence_count: int
    resolution_step: int
    budget_cents: int
    phases: tuple[PreferencePhase, ...]
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        require_id(self.task_id, field="task_id")
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.recipe_id, field="recipe_id")
        require_id(self.user_id, field="user_id")
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise LatentPreferenceDataError("task orbit_index must be non-negative.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise LatentPreferenceDataError("task semantic_epoch must be non-negative.")
        if self.split not in SPLITS:
            raise LatentPreferenceDataError(f"invalid task split {self.split!r}.")
        if self.supporting_evidence_count not in (1, 2, 3):
            raise LatentPreferenceDataError(
                "supporting_evidence_count must be 1, 2, or 3."
            )
        if self.resolution_step != 1:
            raise LatentPreferenceDataError("same-axis preference resolves at step one.")
        if self.budget_cents <= 0:
            raise LatentPreferenceDataError("task budget must be positive.")
        if len(self.phases) != 6 or tuple(
            phase.phase_index for phase in self.phases
        ) != tuple(range(6)):
            raise LatentPreferenceDataError(
                "preference task phases must be ordered from 0 through 5."
            )
        expected_kinds = (
            ("evidence",) * self.supporting_evidence_count
            + ("application",) * (6 - self.supporting_evidence_count)
        )
        if tuple(phase.phase_kind for phase in self.phases) != expected_kinds:
            raise LatentPreferenceDataError("phase kinds do not match evidence count.")
        require_sha256(self.product_pool_sha256, field="task product pool SHA256")
        if isinstance(self.generator_seed, bool) or not isinstance(
            self.generator_seed, int
        ):
            raise LatentPreferenceDataError("task generator seed must be an integer.")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise LatentPreferenceDataError("task generator version must be non-empty.")

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(phase.question for phase in self.phases)

    @property
    def target_asins(self) -> tuple[str, ...]:
        return tuple(phase.target_asin for phase in self.phases)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_targets=True))

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        payload = {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "recipe_id": self.recipe_id,
            "user_id": self.user_id,
            "split": self.split,
            "supporting_evidence_count": self.supporting_evidence_count,
            "resolution_step": self.resolution_step,
            "budget_cents": self.budget_cents,
            "phases": [
                phase.as_dict(include_target=include_targets) for phase in self.phases
            ],
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }
        if include_targets:
            payload["preferred_attribute_value"] = self.preferred_attribute_value
        return payload


@dataclass(frozen=True)
class LatentPreferenceOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    user_id: str
    preferred_attribute_values: tuple[str, str]
    tasks: tuple[LatentPreferenceTask, LatentPreferenceTask]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.recipe_id, field="orbit recipe_id")
        require_id(self.user_id, field="orbit user_id")
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise LatentPreferenceDataError("orbit_index must be non-negative.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise LatentPreferenceDataError("semantic_epoch must be non-negative.")
        if len(self.preferred_attribute_values) != 2 or len(
            set(self.preferred_attribute_values)
        ) != 2:
            raise LatentPreferenceDataError(
                "counterfactual orbit requires two preference values."
            )
        if len(self.tasks) != 2:
            raise LatentPreferenceDataError("counterfactual orbit requires two tasks.")
        for task in self.tasks:
            if (
                task.orbit_id != self.orbit_id
                or task.orbit_index != self.orbit_index
                or task.semantic_epoch != self.semantic_epoch
                or task.recipe_id != self.recipe_id
                or task.user_id != self.user_id
            ):
                raise LatentPreferenceDataError(
                    "counterfactual task identity disagrees with its orbit."
                )

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_targets=True))

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ORBIT_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "recipe_id": self.recipe_id,
            "user_id": self.user_id,
            "tasks": [task.as_dict(include_targets=include_targets) for task in self.tasks],
        }
        if include_targets:
            payload["preferred_attribute_values"] = list(
                self.preferred_attribute_values
            )
        return payload


@dataclass(frozen=True)
class LatentPreferenceBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    recipe_id: str
    user_id: str
    preference_axis: str
    supporting_evidence_count: int
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise LatentPreferenceDataError(
                "latent preference bundle requires six questions and targets."
            )
        if self.split not in SPLITS:
            raise LatentPreferenceDataError("invalid latent preference bundle split.")
        require_sha256(self.proof_sha256, field="bundle proof SHA256")
        require_sha256(self.product_pool_sha256, field="bundle pool SHA256")


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise LatentPreferenceDataError(f"{context} must be an object.")
    observed = set(payload)
    if observed != required:
        raise LatentPreferenceDataError(
            f"{context} fields mismatch: missing={sorted(required - observed)} "
            f"extra={sorted(observed - required)}."
        )
