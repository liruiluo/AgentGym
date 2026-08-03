from __future__ import annotations

from typing import Sequence

from .schema import (
    NegativeConstraintCandidate,
    NegativeConstraintDataError,
    NegativeConstraintRecipe,
)


def render_negative_constraint_question(
    *,
    user_id: str,
    phase_index: int,
    recipe: NegativeConstraintRecipe,
    category_id: str,
    candidates: Sequence[NegativeConstraintCandidate],
    budget_cents: int,
    allowed_attribute_value: str,
    forbidden_attribute_values: tuple[str, str],
) -> str:
    if len(candidates) != 3 or len(
        {item.attribute_value for item in candidates}
    ) != 3:
        raise NegativeConstraintDataError(
            "negative questions require three distinct candidate values."
        )
    if set(item.attribute_value for item in candidates) != set(recipe.values):
        raise NegativeConstraintDataError(
            "negative question candidates must cover the recipe values."
        )
    if set(forbidden_attribute_values) != set(recipe.values) - {
        allowed_attribute_value
    }:
        raise NegativeConstraintDataError(
            "negative question forbidden values must leave one allowed value."
        )
    budget = f"${budget_cents / 100:.2f}"
    category = recipe.category_display_name(category_id)
    header = (
        f"Customer profile: {user_id}\n"
        f"Purchase {phase_index + 1} of 6: {category}\n"
        f"Total six-purchase budget: {budget}\n\n"
    )
    titles = "\n".join(
        f"{position}. {candidate.title}"
        for position, candidate in enumerate(candidates, start=1)
    )
    if phase_index == 0:
        forbidden_displays = tuple(
            recipe.value_display_name(value)
            for value in forbidden_attribute_values
        )
        body = (
            "All three approved listings satisfy the current explicit product "
            "requirements. The customer states a standing negative constraint: "
            f'"I will never accept {forbidden_displays[0]} or '
            f'{forbidden_displays[1]} {recipe.axis_display_name}."\n\n'
            f"Approved listings:\n{titles}\n\n"
            "Purchase the only approved listing that violates neither exclusion. "
            "The exclusions remain active for later shopping sessions."
        )
    else:
        body = (
            "All three approved listings satisfy the current explicit product "
            "requirements. The customer says: \"Honor my standing never-accept "
            "constraints from the earlier shopping session.\"\n\n"
            f"Approved listings:\n{titles}\n\n"
            "Purchase exactly one listing that violates none of the stored "
            "negative constraints."
        )
    return header + body
