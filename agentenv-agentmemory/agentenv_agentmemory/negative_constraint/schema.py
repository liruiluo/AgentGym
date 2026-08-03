from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import (
    canonical_sha256,
    normalize_native_title,
    require_id,
    require_sha256,
)


SPLITS = ("train", "dev", "test")
TASK_SCHEMA = "agentmemory_negative_constraint_task_v1"
ORBIT_SCHEMA = "agentmemory_negative_constraint_counterfactual_orbit_v1"
POOL_SCHEMA = "agentmemory_negative_constraint_rules_pool_v1"
NATIVE_POOL_SCHEMA = "agentmemory_negative_constraint_native_product_pool_v2"
PROOF_SCHEMA = "agentmemory_negative_constraint_proof_v1"


class NegativeConstraintDataError(ValueError):
    """Raised when a negative-constraint task is not machine-verifiable."""


def _require_line(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise NegativeConstraintDataError(f"{field} must be one non-empty line.")
    return value


def _require_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegativeConstraintDataError(f"{field} must be non-empty text.")
    return value


@dataclass(frozen=True)
class NegativeConstraintRecipe:
    recipe_id: str
    axis: str
    axis_display_name: str
    values: tuple[str, str, str]
    value_display_names: tuple[str, str, str]
    categories: tuple[str, str, str, str]
    category_display_names: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        require_id(self.recipe_id, field="negative recipe_id")
        require_id(self.axis, field="negative recipe axis")
        _require_line(self.axis_display_name, field="negative axis display name")
        if len(self.values) != 3 or len(set(self.values)) != 3:
            raise NegativeConstraintDataError(
                "negative recipes require exactly three distinct values."
            )
        if len(self.value_display_names) != 3:
            raise NegativeConstraintDataError(
                "negative recipes require three value display names."
            )
        for value, display in zip(self.values, self.value_display_names):
            require_id(value, field="negative recipe value")
            _require_line(display, field="negative value display name")
        if len(self.categories) not in (3, 4) or len(set(self.categories)) != len(
            self.categories
        ):
            raise NegativeConstraintDataError(
                "negative recipes require three or four distinct categories."
            )
        if len(self.category_display_names) != len(self.categories):
            raise NegativeConstraintDataError(
                "negative recipe category display names must match its categories."
            )
        for category, display in zip(
            self.categories,
            self.category_display_names,
        ):
            require_id(category, field="negative recipe category")
            _require_line(display, field="negative category display name")

    def value_display_name(self, value: str) -> str:
        try:
            return self.value_display_names[self.values.index(value)]
        except ValueError as exc:
            raise KeyError(value) from exc

    def category_display_name(self, category: str) -> str:
        try:
            return self.category_display_names[self.categories.index(category)]
        except ValueError as exc:
            raise KeyError(category) from exc

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


@dataclass(frozen=True)
class NegativeConstraintCandidate:
    asin: str
    title: str
    normalized_title: str
    product_category: str
    category_id: str
    category_display_name: str
    axis: str
    attribute_value: str
    attribute_display_name: str
    title_evidence: tuple[str, ...]
    split: str
    source_classification_sha256: str
    source_row_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.asin, str)
            or len(self.asin) != 10
            or not self.asin.isalnum()
            or self.asin != self.asin.upper()
        ):
            raise NegativeConstraintDataError("invalid candidate ASIN.")
        if not isinstance(self.title, str) or not 8 <= len(self.title) <= 240:
            raise NegativeConstraintDataError("invalid candidate title.")
        if self.normalized_title != normalize_native_title(self.title):
            raise NegativeConstraintDataError("candidate normalized title mismatch.")
        if self.asin.casefold() in self.normalized_title:
            raise NegativeConstraintDataError("candidate title leaks its ASIN.")
        _require_line(self.product_category, field="candidate product category")
        require_id(self.category_id, field="candidate category_id")
        _require_line(
            self.category_display_name,
            field="candidate category display name",
        )
        require_id(self.axis, field="candidate axis")
        require_id(self.attribute_value, field="candidate attribute value")
        _require_line(
            self.attribute_display_name,
            field="candidate attribute display name",
        )
        if (
            not self.title_evidence
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.title_evidence
            )
        ):
            raise NegativeConstraintDataError(
                "candidate title evidence must be non-empty text."
            )
        normalized = self.normalized_title
        if not all(
            normalize_native_title(value) in normalized
            for value in self.title_evidence
        ):
            raise NegativeConstraintDataError(
                "candidate title evidence is absent from its title."
            )
        if self.split not in SPLITS:
            raise NegativeConstraintDataError("invalid candidate split.")
        require_sha256(
            self.source_classification_sha256,
            field="source classification SHA256",
        )
        require_sha256(self.source_row_sha256, field="source row SHA256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "asin": self.asin,
            "title": self.title,
            "normalized_title": self.normalized_title,
            "product_category": self.product_category,
            "category_id": self.category_id,
            "category_display_name": self.category_display_name,
            "axis": self.axis,
            "attribute_value": self.attribute_value,
            "attribute_display_name": self.attribute_display_name,
            "title_evidence": list(self.title_evidence),
            "split": self.split,
            "source_classification_sha256": self.source_classification_sha256,
            "source_row_sha256": self.source_row_sha256,
        }


@dataclass(frozen=True)
class NegativeConstraintNativeCertificate:
    asin: str
    source_row_sha256: str
    price_cents: int
    search_query: str
    search_rank: int
    search_result_asins: tuple[str, ...]
    opened_url: str
    catalog_record_sha256: str
    purchase_receipt_sha256: str
    native_title_catalog_match_count: int = 1
    native_title_globally_unique: bool = True
    native_search_verified: bool = True
    native_open_verified: bool = True
    native_purchase_verified: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.asin, str)
            or len(self.asin) != 10
            or not self.asin.isalnum()
            or self.asin != self.asin.upper()
        ):
            raise NegativeConstraintDataError("invalid native certificate ASIN.")
        require_sha256(self.source_row_sha256, field="certificate source row SHA256")
        if (
            isinstance(self.price_cents, bool)
            or not isinstance(self.price_cents, int)
            or self.price_cents <= 0
        ):
            raise NegativeConstraintDataError(
                "native certificate price must be positive cents."
            )
        _require_line(self.search_query, field="native certificate search query")
        if any(char in self.search_query for char in "[]\r\n"):
            raise NegativeConstraintDataError(
                "native certificate search query contains unsafe action syntax."
            )
        if (
            isinstance(self.search_rank, bool)
            or not isinstance(self.search_rank, int)
            or not 1 <= self.search_rank <= 10
        ):
            raise NegativeConstraintDataError(
                "native certificate search rank must be in [1, 10]."
            )
        if not self.search_result_asins or self.search_rank > len(
            self.search_result_asins
        ):
            raise NegativeConstraintDataError(
                "native certificate search results do not cover the target rank."
            )
        for result_asin in self.search_result_asins:
            if (
                not isinstance(result_asin, str)
                or len(result_asin) != 10
                or not result_asin.isalnum()
                or result_asin != result_asin.upper()
            ):
                raise NegativeConstraintDataError(
                    "native certificate contains an invalid search-result ASIN."
                )
        if self.search_result_asins[self.search_rank - 1] != self.asin:
            raise NegativeConstraintDataError(
                "native certificate target disagrees with its recorded search rank."
            )
        _require_line(self.opened_url, field="native certificate opened URL")
        if self.asin.casefold() not in self.opened_url.casefold():
            raise NegativeConstraintDataError(
                "native certificate opened URL does not identify its ASIN."
            )
        require_sha256(
            self.catalog_record_sha256,
            field="native certificate catalog record SHA256",
        )
        require_sha256(
            self.purchase_receipt_sha256,
            field="native certificate purchase receipt SHA256",
        )
        if self.native_title_catalog_match_count != 1:
            raise NegativeConstraintDataError(
                "native certificate title must map to exactly one catalog ASIN."
            )
        flags = (
            self.native_title_globally_unique,
            self.native_search_verified,
            self.native_open_verified,
            self.native_purchase_verified,
        )
        if any(value is not True for value in flags):
            raise NegativeConstraintDataError(
                "native certificate verification flags must all be true."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "asin": self.asin,
            "source_row_sha256": self.source_row_sha256,
            "price_cents": self.price_cents,
            "search_query": self.search_query,
            "search_rank": self.search_rank,
            "search_result_asins": list(self.search_result_asins),
            "opened_url": self.opened_url,
            "catalog_record_sha256": self.catalog_record_sha256,
            "purchase_receipt_sha256": self.purchase_receipt_sha256,
            "native_title_catalog_match_count": (
                self.native_title_catalog_match_count
            ),
            "native_title_globally_unique": self.native_title_globally_unique,
            "native_search_verified": self.native_search_verified,
            "native_open_verified": self.native_open_verified,
            "native_purchase_verified": self.native_purchase_verified,
        }


@dataclass(frozen=True)
class NegativeConstraintProductPool:
    pool_id: str
    products_per_cell: int
    recipes: tuple[NegativeConstraintRecipe, ...]
    candidates: tuple[NegativeConstraintCandidate, ...]
    candidate_artifact_sha256: str
    split_policy: str
    selection_policy: str
    native_certified: bool = False
    certifier_version: str | None = None
    memoryarena_commit: str | None = None
    catalog_sha256: str | None = None
    attributes_sha256: str | None = None
    price_table_sha256: str | None = None
    lucene_index_sha256: str | None = None
    source_manifest_sha256: str | None = None
    rules_pool_sha256: str | None = None
    price_seed: int | None = None
    native_certificates: tuple[NegativeConstraintNativeCertificate, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.pool_id, field="negative pool_id")
        if self.products_per_cell != 2:
            raise NegativeConstraintDataError(
                "negative pool v1 requires exactly two products per cell."
            )
        if not self.recipes:
            raise NegativeConstraintDataError("negative pool requires recipes.")
        recipe_ids = tuple(recipe.recipe_id for recipe in self.recipes)
        if recipe_ids != tuple(sorted(recipe_ids)) or len(recipe_ids) != len(
            set(recipe_ids)
        ):
            raise NegativeConstraintDataError(
                "negative recipes must be unique and canonically sorted."
            )
        order = tuple(
            (
                item.axis,
                item.category_id,
                item.attribute_value,
                item.split,
                item.asin,
            )
            for item in self.candidates
        )
        if order != tuple(sorted(order)):
            raise NegativeConstraintDataError(
                "negative candidates must use canonical cell/ASIN order."
            )
        asins = tuple(item.asin for item in self.candidates)
        titles = tuple(item.normalized_title for item in self.candidates)
        if len(asins) != len(set(asins)) or len(titles) != len(set(titles)):
            raise NegativeConstraintDataError(
                "selected negative candidates must have unique ASINs and titles."
            )
        expected_cells = {
            (recipe.axis, category, value, split)
            for recipe in self.recipes
            for category in recipe.categories
            for value in recipe.values
            for split in SPLITS
        }
        counts = {cell: 0 for cell in expected_cells}
        recipe_by_axis = {recipe.axis: recipe for recipe in self.recipes}
        if len(recipe_by_axis) != len(self.recipes):
            raise NegativeConstraintDataError(
                "negative pool v1 supports one recipe per axis."
            )
        for item in self.candidates:
            cell = (
                item.axis,
                item.category_id,
                item.attribute_value,
                item.split,
            )
            if cell not in counts:
                raise NegativeConstraintDataError(
                    f"candidate {item.asin} belongs to undeclared cell {cell}."
                )
            recipe = recipe_by_axis[item.axis]
            if item.category_display_name != recipe.category_display_name(
                item.category_id
            ):
                raise NegativeConstraintDataError(
                    "candidate category display disagrees with its recipe."
                )
            if item.attribute_display_name != recipe.value_display_name(
                item.attribute_value
            ):
                raise NegativeConstraintDataError(
                    "candidate value display disagrees with its recipe."
                )
            counts[cell] += 1
        incomplete = {
            cell: count
            for cell, count in counts.items()
            if count != self.products_per_cell
        }
        if incomplete:
            raise NegativeConstraintDataError(
                "negative product cells are incomplete: " + repr(incomplete)
            )
        require_sha256(
            self.candidate_artifact_sha256,
            field="candidate artifact SHA256",
        )
        require_id(self.split_policy, field="negative split policy")
        require_id(self.selection_policy, field="negative selection policy")
        certification_fields = (
            self.certifier_version,
            self.memoryarena_commit,
            self.catalog_sha256,
            self.attributes_sha256,
            self.price_table_sha256,
            self.lucene_index_sha256,
            self.source_manifest_sha256,
            self.rules_pool_sha256,
            self.price_seed,
        )
        if not isinstance(self.native_certified, bool):
            raise NegativeConstraintDataError(
                "negative pool native_certified must be a boolean."
            )
        if not self.native_certified:
            if any(value is not None for value in certification_fields) or (
                self.native_certificates
            ):
                raise NegativeConstraintDataError(
                    "rules-only pool cannot carry native certification fields."
                )
            return

        if not isinstance(self.certifier_version, str):
            raise NegativeConstraintDataError(
                "native pool certifier version must be a string."
            )
        require_id(self.certifier_version, field="negative certifier version")
        if (
            not isinstance(self.memoryarena_commit, str)
            or len(self.memoryarena_commit) != 40
            or any(char not in "0123456789abcdef" for char in self.memoryarena_commit)
        ):
            raise NegativeConstraintDataError(
                "native pool requires a full lowercase MemoryArena commit."
            )
        for field, value in (
            ("catalog SHA256", self.catalog_sha256),
            ("attributes SHA256", self.attributes_sha256),
            ("price table SHA256", self.price_table_sha256),
            ("Lucene index SHA256", self.lucene_index_sha256),
            ("source manifest SHA256", self.source_manifest_sha256),
            ("rules pool SHA256", self.rules_pool_sha256),
        ):
            require_sha256(str(value), field=field)
        if isinstance(self.price_seed, bool) or not isinstance(self.price_seed, int):
            raise NegativeConstraintDataError(
                "native pool price seed must be an integer."
            )
        certificate_order = tuple(item.asin for item in self.native_certificates)
        if certificate_order != tuple(sorted(certificate_order)):
            raise NegativeConstraintDataError(
                "native certificates must use canonical ASIN order."
            )
        if (
            len(certificate_order) != len(asins)
            or len(certificate_order) != len(set(certificate_order))
            or set(certificate_order) != set(asins)
        ):
            raise NegativeConstraintDataError(
                "native certificates must cover every selected ASIN exactly once."
            )
        candidate_by_asin = {item.asin: item for item in self.candidates}
        for certificate in self.native_certificates:
            candidate = candidate_by_asin[certificate.asin]
            if certificate.source_row_sha256 != candidate.source_row_sha256:
                raise NegativeConstraintDataError(
                    "native certificate source row disagrees with its candidate."
                )

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.semantic_manifest())

    def recipe_by_id(self, recipe_id: str) -> NegativeConstraintRecipe:
        for recipe in self.recipes:
            if recipe.recipe_id == recipe_id:
                return recipe
        raise KeyError(recipe_id)

    def candidates_for(
        self,
        *,
        axis: str,
        category_id: str,
        attribute_value: str,
        split: str,
    ) -> tuple[NegativeConstraintCandidate, ...]:
        matches = tuple(
            item
            for item in self.candidates
            if item.axis == axis
            and item.category_id == category_id
            and item.attribute_value == attribute_value
            and item.split == split
        )
        if len(matches) != self.products_per_cell:
            raise NegativeConstraintDataError(
                "negative candidate cell is incomplete."
            )
        return matches

    def certificate_for(self, asin: str) -> NegativeConstraintNativeCertificate:
        for certificate in self.native_certificates:
            if certificate.asin == asin:
                return certificate
        raise KeyError(asin)

    def semantic_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": POOL_SCHEMA,
            "pool_id": self.pool_id,
            "products_per_cell": self.products_per_cell,
            "recipes": [item.as_dict() for item in self.recipes],
            "candidates": [item.as_dict() for item in self.candidates],
            "provenance": {
                "candidate_artifact_sha256": self.candidate_artifact_sha256,
                "split_policy": self.split_policy,
                "selection_policy": self.selection_policy,
                "native_certified": self.native_certified,
            },
        }
        if not self.native_certified:
            return payload
        payload["schema"] = NATIVE_POOL_SCHEMA
        payload["native_certificates"] = [
            item.as_dict() for item in self.native_certificates
        ]
        payload["provenance"].update(
            {
                "certifier_version": self.certifier_version,
                "memoryarena_commit": self.memoryarena_commit,
                "catalog_sha256": self.catalog_sha256,
                "attributes_sha256": self.attributes_sha256,
                "price_table_sha256": self.price_table_sha256,
                "lucene_index_sha256": self.lucene_index_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "rules_pool_sha256": self.rules_pool_sha256,
                "price_seed": self.price_seed,
            }
        )
        return payload


@dataclass(frozen=True)
class NegativeConstraintPhase:
    phase_index: int
    phase_kind: str
    category_id: str
    category_display_name: str
    candidates: tuple[
        NegativeConstraintCandidate,
        NegativeConstraintCandidate,
        NegativeConstraintCandidate,
    ]
    question: str
    target_asin: str
    allowed_attribute_value: str

    def __post_init__(self) -> None:
        if self.phase_index not in range(6):
            raise NegativeConstraintDataError("phase index must be in [0, 6).")
        expected_kind = "constraint_evidence" if self.phase_index == 0 else "application"
        if self.phase_kind != expected_kind:
            raise NegativeConstraintDataError("phase kind disagrees with phase index.")
        require_id(self.category_id, field="negative phase category")
        _require_line(
            self.category_display_name,
            field="negative phase category display",
        )
        if len(self.candidates) != 3:
            raise NegativeConstraintDataError(
                "negative phases require exactly three candidates."
            )
        if len({item.asin for item in self.candidates}) != 3 or len(
            {item.attribute_value for item in self.candidates}
        ) != 3:
            raise NegativeConstraintDataError(
                "negative phase candidates require three ASINs and three values."
            )
        for item in self.candidates:
            if (
                item.category_id != self.category_id
                or item.category_display_name != self.category_display_name
            ):
                raise NegativeConstraintDataError(
                    "negative phase candidate category mismatch."
                )
        _require_text(self.question, field="negative question")
        if self.target_asin not in {item.asin for item in self.candidates}:
            raise NegativeConstraintDataError(
                "negative phase target is outside its candidates."
            )
        target = tuple(
            item for item in self.candidates if item.asin == self.target_asin
        )[0]
        if target.attribute_value != self.allowed_attribute_value:
            raise NegativeConstraintDataError(
                "negative phase target disagrees with allowed value."
            )

    def as_dict(self, *, include_target: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase_index": self.phase_index,
            "phase_kind": self.phase_kind,
            "category_id": self.category_id,
            "category_display_name": self.category_display_name,
            "candidates": [item.as_dict() for item in self.candidates],
            "question": self.question,
            "allowed_attribute_value": self.allowed_attribute_value,
        }
        if include_target:
            payload["target_asin"] = self.target_asin
        return payload


@dataclass(frozen=True)
class NegativeConstraintTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    user_id: str
    split: str
    branch_kind: str
    allowed_attribute_value: str
    forbidden_attribute_values: tuple[str, str]
    canonical_memory_key: str
    canonical_memory_value: str
    canonical_retrieval_query: str
    budget_cents: int
    phases: tuple[NegativeConstraintPhase, ...]
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        for name in ("task_id", "orbit_id", "recipe_id", "user_id"):
            require_id(getattr(self, name), field=name)
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise NegativeConstraintDataError("invalid orbit_index.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise NegativeConstraintDataError("invalid semantic_epoch.")
        if self.split not in SPLITS:
            raise NegativeConstraintDataError("invalid task split.")
        expected_branch = f"allow_{self.allowed_attribute_value}"
        if self.branch_kind != expected_branch:
            raise NegativeConstraintDataError(
                "negative branch kind disagrees with allowed value."
            )
        if (
            len(self.forbidden_attribute_values) != 2
            or len(set(self.forbidden_attribute_values)) != 2
            or self.allowed_attribute_value in self.forbidden_attribute_values
        ):
            raise NegativeConstraintDataError(
                "task requires one allowed and two distinct forbidden values."
            )
        require_id(self.canonical_memory_key, field="canonical memory key")
        _require_line(self.canonical_memory_value, field="canonical memory value")
        _require_line(
            self.canonical_retrieval_query,
            field="canonical retrieval query",
        )
        if (
            isinstance(self.budget_cents, bool)
            or not isinstance(self.budget_cents, int)
            or self.budget_cents <= 0
        ):
            raise NegativeConstraintDataError("invalid task budget.")
        if len(self.phases) != 6 or tuple(
            item.phase_index for item in self.phases
        ) != tuple(range(6)):
            raise NegativeConstraintDataError(
                "negative task requires six ordered phases."
            )
        if any(
            item.allowed_attribute_value != self.allowed_attribute_value
            for item in self.phases
        ):
            raise NegativeConstraintDataError(
                "phase allowed value disagrees with task."
            )
        if isinstance(self.generator_seed, bool) or not isinstance(
            self.generator_seed,
            int,
        ):
            raise NegativeConstraintDataError("generator seed must be an integer.")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise NegativeConstraintDataError("generator version must be non-empty.")
        require_sha256(self.product_pool_sha256, field="product pool SHA256")

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(item.question for item in self.phases)

    @property
    def target_asins(self) -> tuple[str, ...]:
        return tuple(item.target_asin for item in self.phases)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_targets=True))

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        return {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "recipe_id": self.recipe_id,
            "user_id": self.user_id,
            "split": self.split,
            "branch_kind": self.branch_kind,
            "allowed_attribute_value": self.allowed_attribute_value,
            "forbidden_attribute_values": list(self.forbidden_attribute_values),
            "canonical_memory_key": self.canonical_memory_key,
            "canonical_memory_value": self.canonical_memory_value,
            "canonical_retrieval_query": self.canonical_retrieval_query,
            "budget_cents": self.budget_cents,
            "phases": [
                item.as_dict(include_target=include_targets)
                for item in self.phases
            ],
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }


@dataclass(frozen=True)
class NegativeConstraintOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    recipe_id: str
    user_id: str
    tasks: tuple[
        NegativeConstraintTask,
        NegativeConstraintTask,
        NegativeConstraintTask,
    ]

    def __post_init__(self) -> None:
        for name in ("orbit_id", "recipe_id", "user_id"):
            require_id(getattr(self, name), field=name)
        if len(self.tasks) != 3:
            raise NegativeConstraintDataError(
                "negative orbit requires exactly three branches."
            )
        values = tuple(item.allowed_attribute_value for item in self.tasks)
        if len(set(values)) != 3:
            raise NegativeConstraintDataError(
                "negative orbit branches must allow three different values."
            )
        for item in self.tasks:
            if (
                item.orbit_id,
                item.orbit_index,
                item.semantic_epoch,
                item.recipe_id,
                item.user_id,
            ) != (
                self.orbit_id,
                self.orbit_index,
                self.semantic_epoch,
                self.recipe_id,
                self.user_id,
            ):
                raise NegativeConstraintDataError(
                    "negative task identity disagrees with orbit."
                )

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_targets=True))

    def as_dict(self, *, include_targets: bool = True) -> dict[str, Any]:
        return {
            "schema": ORBIT_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "recipe_id": self.recipe_id,
            "user_id": self.user_id,
            "tasks": [
                item.as_dict(include_targets=include_targets)
                for item in self.tasks
            ],
        }


@dataclass(frozen=True)
class NegativeConstraintBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    branch_kind: str
    allowed_attribute_value: str
    forbidden_attribute_values: tuple[str, str]
    canonical_memory_key: str
    canonical_memory_value: str
    canonical_retrieval_query: str
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise NegativeConstraintDataError(
                "negative bundle requires six phases."
            )
        if self.split not in SPLITS:
            raise NegativeConstraintDataError("invalid negative bundle split.")
        require_sha256(self.proof_sha256, field="negative proof SHA256")
        require_sha256(self.product_pool_sha256, field="negative pool SHA256")
