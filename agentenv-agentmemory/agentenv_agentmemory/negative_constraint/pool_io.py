from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..latent_preference.schema import (
    canonical_sha256,
    normalize_native_title,
    require_sha256,
)
from .schema import (
    SPLITS,
    NegativeConstraintCandidate,
    NegativeConstraintDataError,
    NegativeConstraintProductPool,
    NegativeConstraintRecipe,
)


SOURCE_SCHEMA = "agentmemory_latent_preference_rule_candidate_v2"
POOL_ID = "memoryarena_negative_constraint_rules_v1"
SPLIT_POLICY = "asin_hash_80_10_10_v1"
SELECTION_POLICY = "global_unique_cell_hash_first2_v1"
SOURCE_ROW_KEYS = {
    "schema",
    "asin",
    "axis",
    "attribute_value",
    "category_id",
    "classification_sha256",
    "normalized_title",
    "product_category",
    "title",
    "title_evidence",
}
SOURCE_CLASSIFICATION_KEYS = (
    "category_id",
    "axis",
    "attribute_value",
    "asin",
    "title",
    "product_category",
    "title_evidence",
)


NEGATIVE_CONSTRAINT_RECIPES = (
    NegativeConstraintRecipe(
        recipe_id="color.black_gray_red",
        axis="color",
        axis_display_name="color",
        values=("black", "gray", "red"),
        value_display_names=("black", "gray", "red"),
        categories=(
            "area_rug",
            "phone_case",
            "pillowcase",
            "window_curtain",
        ),
        category_display_names=(
            "area rug",
            "phone case",
            "pillowcase",
            "window curtain",
        ),
    ),
    NegativeConstraintRecipe(
        recipe_id="pattern.floral_geometric_solid",
        axis="pattern",
        axis_display_name="pattern",
        values=("floral", "geometric", "solid"),
        value_display_names=("floral", "geometric", "solid"),
        categories=("area_rug", "pillowcase", "window_curtain"),
        category_display_names=(
            "area rug",
            "pillowcase",
            "window curtain",
        ),
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_asin(asin: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_POLICY}:{asin}".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    if bucket < 8:
        return "train"
    return "dev" if bucket == 8 else "test"


def load_negative_constraint_product_pool(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> NegativeConstraintProductPool:
    candidate_path = Path(path)
    if not candidate_path.is_file():
        raise NegativeConstraintDataError(
            f"negative candidate artifact is not a file: {candidate_path}"
        )
    require_sha256(expected_file_sha256, field="expected candidate file SHA256")
    observed_file_sha256 = file_sha256(candidate_path)
    if observed_file_sha256 != expected_file_sha256:
        raise NegativeConstraintDataError(
            "negative candidate artifact SHA256 mismatch: "
            f"expected {expected_file_sha256}, observed {observed_file_sha256}"
        )

    recipes_by_axis = {item.axis: item for item in NEGATIVE_CONSTRAINT_RECIPES}
    selected_cells = {
        (recipe.axis, category, value)
        for recipe in NEGATIVE_CONSTRAINT_RECIPES
        for category in recipe.categories
        for value in recipe.values
    }
    by_asin_axis: dict[
        tuple[str, str],
        tuple[str, NegativeConstraintCandidate],
    ] = {}
    asin_identities: dict[str, tuple[str, str, str]] = {}
    asin_axis_assignments: dict[tuple[str, str], tuple[str, str]] = {}
    row_count = 0
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise NegativeConstraintDataError(
                    f"negative candidate artifact contains blank line {line_number}."
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise NegativeConstraintDataError(
                    f"negative candidate line {line_number} is invalid JSON."
                ) from exc
            if not isinstance(parsed, Mapping):
                raise NegativeConstraintDataError(
                    f"negative candidate line {line_number} is not an object."
                )
            row = dict(parsed)
            _validate_source_row(row, line_number=line_number)
            row_count += 1
            asin = str(row["asin"])
            axis = str(row["axis"])
            category = str(row["category_id"])
            value = str(row["attribute_value"])
            identity = (
                str(row["title"]),
                str(row["normalized_title"]),
                str(row["product_category"]),
            )
            prior_identity = asin_identities.setdefault(asin, identity)
            if prior_identity != identity:
                raise NegativeConstraintDataError(
                    f"candidate ASIN {asin} has conflicting source identities."
                )
            assignment = (category, value)
            prior_assignment = asin_axis_assignments.setdefault(
                (asin, axis),
                assignment,
            )
            if prior_assignment != assignment:
                raise NegativeConstraintDataError(
                    f"candidate ASIN {asin} has conflicting {axis} assignments."
                )
            if (axis, category, value) not in selected_cells:
                continue
            recipe = recipes_by_axis[axis]
            source_row_sha256 = canonical_sha256(row)
            candidate = NegativeConstraintCandidate(
                asin=asin,
                title=str(row["title"]),
                normalized_title=str(row["normalized_title"]),
                product_category=str(row["product_category"]),
                category_id=category,
                category_display_name=recipe.category_display_name(category),
                axis=axis,
                attribute_value=value,
                attribute_display_name=recipe.value_display_name(value),
                title_evidence=tuple(row["title_evidence"]),
                split=split_for_asin(asin),
                source_classification_sha256=str(row["classification_sha256"]),
                source_row_sha256=source_row_sha256,
            )
            selection_sha256 = canonical_sha256(
                {
                    "selection_policy": SELECTION_POLICY,
                    "pool_id": POOL_ID,
                    "candidate_artifact_sha256": observed_file_sha256,
                    "candidate": candidate.as_dict(),
                }
            )
            key = (asin, axis)
            prior = by_asin_axis.get(key)
            if prior is None or selection_sha256 < prior[0]:
                by_asin_axis[key] = (selection_sha256, candidate)
    if row_count == 0:
        raise NegativeConstraintDataError("negative candidate artifact is empty.")

    grouped: dict[
        tuple[str, str, str, str],
        list[tuple[str, NegativeConstraintCandidate]],
    ] = {
        (recipe.axis, category, value, split): []
        for recipe in NEGATIVE_CONSTRAINT_RECIPES
        for category in recipe.categories
        for value in recipe.values
        for split in SPLITS
    }
    for selection_sha256, candidate in by_asin_axis.values():
        cell = (
            candidate.axis,
            candidate.category_id,
            candidate.attribute_value,
            candidate.split,
        )
        if cell in grouped:
            grouped[cell].append((selection_sha256, candidate))

    selected: list[NegativeConstraintCandidate] = []
    selected_asins: set[str] = set()
    selected_titles: set[str] = set()
    for cell, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda item: (item[0], item[1].asin, item[1].source_row_sha256),
        )
        eligible = [
            item
            for item in ordered
            if item[1].asin not in selected_asins
            and item[1].normalized_title not in selected_titles
        ]
        if len(eligible) < 2:
            raise NegativeConstraintDataError(
                f"negative candidate cell {cell} has only {len(eligible)} globally "
                "unique products."
            )
        chosen = [item[1] for item in eligible[:2]]
        selected.extend(chosen)
        selected_asins.update(item.asin for item in chosen)
        selected_titles.update(item.normalized_title for item in chosen)
    selected.sort(
        key=lambda item: (
            item.axis,
            item.category_id,
            item.attribute_value,
            item.split,
            item.asin,
        )
    )
    return NegativeConstraintProductPool(
        pool_id=POOL_ID,
        products_per_cell=2,
        recipes=tuple(sorted(NEGATIVE_CONSTRAINT_RECIPES, key=lambda item: item.recipe_id)),
        candidates=tuple(selected),
        candidate_artifact_sha256=observed_file_sha256,
        split_policy=SPLIT_POLICY,
        selection_policy=SELECTION_POLICY,
        native_certified=False,
    )


def _validate_source_row(row: Mapping[str, Any], *, line_number: int) -> None:
    if set(row) != SOURCE_ROW_KEYS:
        raise NegativeConstraintDataError(
            f"negative candidate line {line_number} fields mismatch: "
            f"missing={sorted(SOURCE_ROW_KEYS - set(row))} "
            f"extra={sorted(set(row) - SOURCE_ROW_KEYS)}"
        )
    if row.get("schema") != SOURCE_SCHEMA:
        raise NegativeConstraintDataError(
            f"negative candidate line {line_number} has unsupported schema."
        )
    for key in (
        "asin",
        "axis",
        "attribute_value",
        "category_id",
        "classification_sha256",
        "normalized_title",
        "product_category",
        "title",
    ):
        if not isinstance(row.get(key), str) or not str(row[key]).strip():
            raise NegativeConstraintDataError(
                f"negative candidate line {line_number} has invalid {key}."
            )
    evidence = row.get("title_evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise NegativeConstraintDataError(
            f"negative candidate line {line_number} has invalid title_evidence."
        )
    if str(row["normalized_title"]) != normalize_native_title(str(row["title"])):
        raise NegativeConstraintDataError(
            f"negative candidate line {line_number} normalized title mismatch."
        )
    require_sha256(
        str(row["classification_sha256"]),
        field=f"candidate line {line_number} classification SHA256",
    )
    observed = canonical_sha256(
        {key: row[key] for key in SOURCE_CLASSIFICATION_KEYS}
    )
    if observed != row["classification_sha256"]:
        raise NegativeConstraintDataError(
            f"negative candidate line {line_number} classification hash mismatch."
        )
