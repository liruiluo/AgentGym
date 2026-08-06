from __future__ import annotations

from typing import Sequence

from ..latent_preference.schema import PreferenceCandidate, PreferenceRecipe
from .schema import IntentClarificationDataError


def render_intent_clarification_question(
    *,
    user_id: str,
    phase_index: int,
    recipe: PreferenceRecipe,
    category_id: str,
    candidates: Sequence[PreferenceCandidate],
    budget_cents: int,
) -> str:
    if phase_index not in range(6):
        raise IntentClarificationDataError("phase_index must be in [0, 6).")
    if len(candidates) != 2:
        raise IntentClarificationDataError(
            "intent clarification requires two candidates."
        )
    if len({item.attribute_value for item in candidates}) != 2:
        raise IntentClarificationDataError(
            "the two candidates must represent different preference values."
        )
    category = recipe.category_display_name(category_id)
    budget = f"${budget_cents / 100:.2f}"
    header = (
        f"Customer profile: {user_id}\n"
        f"Purchase {phase_index + 1} of 6: {category}\n"
        f"Total six-purchase budget: {budget}\n\n"
    )
    listings = (
        "Approved listings:\n"
        f"- {candidates[0].title}\n"
        f"- {candidates[1].title}\n\n"
    )
    if phase_index == 0:
        body = (
            "Both approved listings satisfy every stated requirement, but the "
            f"customer has not said which {recipe.axis_display_name} they want. "
            "The current request alone cannot identify a unique listing.\n\n"
            + listings
            + "Before purchasing, ask the customer to clarify this missing "
            "preference with the generic ASK {\"field\":\"...\"} action. "
            "Infer the missing field from the request and candidate attributes; "
            "the action schema does not reveal it. Then purchase "
            "the listing that matches the clarification."
        )
    else:
        body = (
            "Both approved listings satisfy the current explicit requirements. "
            "Use the preference clarified in the first shopping session.\n\n"
            + listings
            + "Retrieve the stored clarification and purchase exactly one matching "
            "listing."
        )
    return header + body
