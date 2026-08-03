from __future__ import annotations

from typing import Sequence

from ..latent_preference.schema import PreferenceCandidate, PreferenceRecipe
from .schema import CompositionalRecallDataError


def render_compositional_question(
    *,
    user_id: str,
    phase_index: int,
    recipe: PreferenceRecipe,
    category_id: str,
    candidates: Sequence[PreferenceCandidate],
    budget_cents: int,
    profile_tokens: tuple[str, str],
    active_profile_token: str,
    profile_directory: tuple[tuple[str, str], tuple[str, str]],
    one_time_attribute_value: str,
) -> str:
    if len(candidates) != 2:
        raise CompositionalRecallDataError(
            "compositional question requires two candidates."
        )
    category = recipe.category_display_name(category_id)
    budget = f"${budget_cents / 100:.2f}"
    header = (
        f"Customer profile: {user_id}\n"
        f"Purchase {phase_index + 1} of 6: {category}\n"
        f"Total six-purchase budget: {budget}\n\n"
    )
    titles = f"1. {candidates[0].title}\n2. {candidates[1].title}"
    if phase_index == 0:
        selected = _candidate_for_value(candidates, one_time_attribute_value)
        position = candidates.index(selected) + 1
        body = (
            f"The customer's active shopping profile token is "
            f"{active_profile_token}. This customer-to-profile link remains relevant "
            "for later purchases.\n\n"
            f"Approved listings:\n{titles}\n\n"
            f"For this purchase only, the customer confirmed approved listing "
            f"{position}. Purchase exactly that listing. This one-time choice does "
            "not define the profile preference."
        )
    elif phase_index == 1:
        directory_lines = []
        for token, value in profile_directory:
            directory_lines.append(
                f"- {token}: {recipe.axis_display_name} is "
                f"{recipe.value_display_name(value)}"
            )
        body = (
            "The current profile directory is:\n"
            + "\n".join(directory_lines)
            + "\n\nThe customer's active profile token is not repeated in this "
            "session. Use the earlier customer-to-profile link.\n\n"
            f"Approved listings:\n{titles}\n\n"
            "Purchase exactly one listing using the preference attached to the "
            "customer's active profile token."
        )
    else:
        body = (
            "Both approved listings satisfy the current explicit requirements. "
            "The customer says: \"Use the current preference linked through my "
            "active shopping profile.\"\n\n"
            f"Approved listings:\n{titles}\n\n"
            "Purchase exactly one listing using the remembered profile link and "
            "profile directory."
        )
    return header + body


def _candidate_for_value(
    candidates: Sequence[PreferenceCandidate],
    value: str,
) -> PreferenceCandidate:
    selected = tuple(item for item in candidates if item.attribute_value == value)
    if len(selected) != 1:
        raise CompositionalRecallDataError(
            "attribute value must identify exactly one candidate."
        )
    return selected[0]
