from __future__ import annotations

from typing import Sequence

from ..latent_preference.schema import PreferenceCandidate, PreferenceRecipe
from .schema import SelectiveMemoryUseDataError


def render_selective_memory_question(
    *,
    user_id: str,
    phase_index: int,
    memory_requirement: str,
    recipe: PreferenceRecipe,
    category_id: str,
    candidates: Sequence[PreferenceCandidate],
    budget_cents: int,
    preferred_attribute_value: str,
) -> str:
    if len(candidates) != 2:
        raise SelectiveMemoryUseDataError(
            "selective-memory question requires two candidates."
        )
    if memory_requirement not in {"memory_required", "memory_not_required"}:
        raise SelectiveMemoryUseDataError("invalid memory requirement.")
    category = recipe.category_display_name(category_id)
    budget = f"${budget_cents / 100:.2f}"
    header = (
        f"Customer: {user_id}\n"
        f"Purchase {phase_index + 1} of 6: {category}\n"
        f"Total six-purchase budget: {budget}\n\n"
    )
    if memory_requirement == "memory_required":
        requirement = (
            f"The customer wants their usual choice for {recipe.axis_display_name}; "
            "this request does not state which value that is."
        )
    else:
        display_value = recipe.value_display_name(preferred_attribute_value)
        requirement = (
            f"For this purchase, the customer's required {recipe.axis_display_name} "
            f"value is {display_value}."
        )
    return (
        header
        + requirement
        + " Both approved listings satisfy every other current requirement.\n\n"
        + "Approved listings:\n"
        + f"1. {candidates[0].title}\n"
        + f"2. {candidates[1].title}\n\n"
        + "Purchase exactly one approved listing."
    )
