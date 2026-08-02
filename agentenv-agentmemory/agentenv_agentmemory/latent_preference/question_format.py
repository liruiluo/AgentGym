from __future__ import annotations

from typing import Sequence

from .schema import LatentPreferenceDataError, PreferenceCandidate, PreferenceRecipe


def render_preference_question(
    *,
    user_id: str,
    phase_index: int,
    phase_kind: str,
    supporting_evidence_count: int,
    recipe: PreferenceRecipe,
    category_id: str,
    candidates: Sequence[PreferenceCandidate],
    budget_cents: int,
    confirmed_attribute_value: str | None,
) -> str:
    if len(candidates) != 2:
        raise LatentPreferenceDataError("question renderer requires two candidates.")
    category = recipe.category_display_name(category_id)
    budget = f"${budget_cents / 100:.2f}"
    header = (
        f"Customer profile: {user_id}\n"
        f"Purchase {phase_index + 1} of 6: {category}\n"
        f"Total six-purchase budget: {budget}\n\n"
    )
    if phase_kind == "evidence":
        if confirmed_attribute_value is None:
            raise LatentPreferenceDataError(
                "evidence question requires a confirmed attribute value."
            )
        selected = tuple(
            candidate
            for candidate in candidates
            if candidate.attribute_value == confirmed_attribute_value
        )
        if len(selected) != 1:
            raise LatentPreferenceDataError(
                "confirmed value must identify one evidence candidate."
            )
        if supporting_evidence_count not in (1, 2, 3):
            raise LatentPreferenceDataError(
                "supporting evidence count must be one, two, or three."
            )
        confirmed_position = candidates.index(selected[0]) + 1
        evidence_wording = (
            "Treat this confirmed choice as the customer's current preference evidence."
            if phase_index == 0
            else "Treat this as additional evidence about the same customer preference."
        )
        body = (
            "The customer already compared two approved listings. Both satisfy "
            "the current explicit requirements.\n\n"
            "Approved listings:\n"
            f"1. {candidates[0].title}\n"
            f"2. {candidates[1].title}\n\n"
            f"The customer confirmed approved listing {confirmed_position}. "
            "Purchase exactly that listing. "
            f"{evidence_wording}"
        )
    elif phase_kind == "application":
        if confirmed_attribute_value is not None:
            raise LatentPreferenceDataError(
                "application question cannot reveal the preference value."
            )
        body = (
            "Both approved listings satisfy the current explicit requirements. "
            "The customer says: \"Use what you learned from my earlier choices "
            f"for the {recipe.axis_display_name}.\"\n\n"
            "Approved listings:\n"
            f"- {candidates[0].title}\n"
            f"- {candidates[1].title}\n\n"
            "Purchase exactly one listing using the remembered customer preference."
        )
    else:
        raise LatentPreferenceDataError(f"unknown phase kind {phase_kind!r}.")
    return header + body
