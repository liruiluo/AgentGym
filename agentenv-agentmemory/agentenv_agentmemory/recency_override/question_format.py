from __future__ import annotations

from typing import Sequence

from ..latent_preference.schema import PreferenceCandidate, PreferenceRecipe
from .schema import RecencyOverrideDataError


def render_recency_question(
    *,
    user_id: str,
    phase_index: int,
    phase_kind: str,
    recipe: PreferenceRecipe,
    category_id: str,
    candidates: Sequence[PreferenceCandidate],
    budget_cents: int,
    old_attribute_value: str,
    new_attribute_value: str,
    active_attribute_value: str,
    confirmed_attribute_value: str | None,
) -> str:
    if len(candidates) != 2:
        raise RecencyOverrideDataError("recency question requires two candidates.")
    budget = f"${budget_cents / 100:.2f}"
    category = recipe.category_display_name(category_id)
    header = (
        f"Customer profile: {user_id}\n"
        f"Purchase {phase_index + 1} of 6: {category}\n"
        f"Total six-purchase budget: {budget}\n\n"
    )
    titles = f"1. {candidates[0].title}\n2. {candidates[1].title}"
    if phase_kind == "evidence":
        if confirmed_attribute_value != old_attribute_value:
            raise RecencyOverrideDataError("initial evidence must confirm the old value.")
        selected = tuple(item for item in candidates if item.attribute_value == confirmed_attribute_value)
        if len(selected) != 1:
            raise RecencyOverrideDataError("initial evidence must identify one candidate.")
        position = candidates.index(selected[0]) + 1
        preference_record = _preference_record(recipe, confirmed_attribute_value)
        body = (
            "The customer already compared two approved listings. Both satisfy the "
            "current explicit requirements.\n\n"
            f"Approved listings:\n{titles}\n\n"
            f"The customer confirmed approved listing {position}. Purchase exactly "
            "that listing. This confirmation establishes the following current "
            f"preference record:\n{preference_record}\n\n"
            "Preserve that exact record before the purchase advances the session."
        )
    elif phase_kind == "application":
        if confirmed_attribute_value is not None:
            raise RecencyOverrideDataError("application cannot expose a confirmation.")
        body = (
            "Both approved listings satisfy the current explicit requirements. The "
            "customer says: \"Use the customer's current preference.\"\n\n"
            f"Approved listings:\n{titles}\n\n"
            "Purchase exactly one listing using the current preference after any "
            "earlier customer update."
        )
    elif phase_kind == "override":
        selected = tuple(item for item in candidates if item.attribute_value == confirmed_attribute_value)
        if len(selected) != 1:
            raise RecencyOverrideDataError("override confirmation must identify one candidate.")
        position = candidates.index(selected[0]) + 1
        if active_attribute_value == old_attribute_value:
            if confirmed_attribute_value != old_attribute_value:
                raise RecencyOverrideDataError("stay override must confirm old value.")
            change = "The customer says the earlier preference is still current."
        else:
            if confirmed_attribute_value != new_attribute_value:
                raise RecencyOverrideDataError("flip override must confirm new value.")
            new_display = recipe.value_display_name(new_attribute_value)
            change = (
                f"The customer explicitly changed the preference and now prefers "
                f"{new_display}. Record this as the new current preference."
            )
        preference_record = _preference_record(recipe, confirmed_attribute_value)
        body = (
            f"{change}\n\nApproved listings:\n{titles}\n\n"
            f"The customer confirmed approved listing {position}. Purchase exactly "
            "that listing. After this confirmation, the one current preference record "
            f"must be:\n{preference_record}\n\n"
            "Before purchasing, make sure the persistent record contains exactly this "
            "current value and no conflicting stale value."
        )
    else:
        raise RecencyOverrideDataError(f"unknown recency phase kind {phase_kind!r}.")
    return header + body


def _preference_record(
    recipe: PreferenceRecipe,
    attribute_value: str | None,
) -> str:
    if attribute_value is None:
        raise RecencyOverrideDataError("a visible preference record requires a value.")
    return (
        f"Current preference: {recipe.axis_display_name} = "
        f"{recipe.value_display_name(attribute_value)}"
    )
