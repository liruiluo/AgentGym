from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from .schema import AttributeTransition
from .scenarios import scenario_by_id


QUESTION_FORMAT_VERSION = "natural_customer_approved_shortlist_chain_question_v5"


def render_question(
    *,
    scenario_id: str,
    phase_index: int,
    slot_id: str,
    candidate_rows: Sequence[tuple[str, str, str]],
    budget_cents: int,
    root_attribute_value: str | None,
    transition: AttributeTransition | None,
) -> str:
    """Render the complete policy-visible text for one shopping session."""

    if len(candidate_rows) != 2:
        raise ValueError("the memory-focused question requires exactly two candidates")
    scenario = scenario_by_id(scenario_id)
    slot = scenario.slot(slot_id)
    expected_values = set(slot.value_ids)
    if {row[1] for row in candidate_rows} != expected_values:
        raise ValueError("candidate rows must cover both slot attribute values")

    if phase_index == 0:
        if root_attribute_value not in expected_values or transition is not None:
            raise ValueError("session one requires one root value and no transition")
        root_display = slot.value(root_attribute_value).display_name
        step_rule = (
            f"The customer's starting request is {slot.attribute_name} = "
            f"{root_display}. Buy the {slot.display_name} whose confirmed "
            f"{slot.attribute_name} is exactly {root_display}."
        )
    else:
        if root_attribute_value is not None or transition is None:
            raise ValueError("later sessions require a transition and no root value")
        previous_slot = scenario.slot(transition.previous_slot_id)
        rows = transition_rows(
            scenario_id=scenario_id,
            current_slot_id=slot_id,
            transition=transition,
        )
        step_rule = (
            "The customer supplied this order-specific choice table:\n"
            + "\n".join(f"- {row}" for row in rows)
            + f"\nUse the exact {previous_slot.attribute_name} value from the product "
            "bought in the immediately preceding shopping session. That earlier "
            "purchase is not repeated in "
            f"this session. Buy the matching {slot.display_name}."
        )
        if previous_slot.attribute_name != transition.previous_attribute_name:
            raise ValueError("transition previous attribute metadata mismatch")

    budget = Decimal(budget_cents) / Decimal(100)
    options = "\n".join(
        _render_candidate(slot.attribute_name, title, value_id, display_name)
        for title, value_id, display_name in candidate_rows
    )
    return (
        f"You are shopping for {scenario.order_context}. The customer makes one "
        "purchase in each of six separate webshop sessions.\n\n"
        "*** ORDER RULES ***\n"
        "1. For this session, the customer shared exactly two approved product cards "
        "below. Only those two exact listings are eligible for this order; other "
        "catalog products are not approved alternatives even if they have the same "
        "color, material, flavor, or other attribute. Search using identifying words "
        "copied from the product title on the card, then open the result whose complete "
        "title matches that card.\n"
        "2. The pairing table is this customer's own plan, so both approved products "
        "may be ordinary real-world choices. Do not replace the table with general "
        "product knowledge.\n"
        "3. Before committing a purchase, preserve the exact attribute name and value "
        "printed in that selected card's Confirmed line when a later session will need "
        "it. Copy that field; do not replace it with a generic certified, natural, "
        "normal, or boolean label.\n"
        f"4. The total price of all six products must not exceed ${budget:.2f}.\n"
        "5. Buy the products in the stated session order.\n\n"
        + "-" * 64
        + "\n"
        + f"Session {phase_index + 1} of 6: select {slot.display_name}\n"
        + f"Goal: {step_rule}\n"
        + "Customer-approved product cards:\n"
        + options
    )


def transition_rows(
    *,
    scenario_id: str,
    current_slot_id: str,
    transition: AttributeTransition,
) -> tuple[str, str]:
    scenario = scenario_by_id(scenario_id)
    previous_slot = scenario.slot(transition.previous_slot_id)
    current_slot = scenario.slot(current_slot_id)
    if transition.previous_attribute_name != previous_slot.attribute_name:
        raise ValueError("transition previous attribute name is not canonical")
    if transition.current_attribute_name != current_slot.attribute_name:
        raise ValueError("transition current attribute name is not canonical")
    values = []
    for previous_value, current_value in transition.pairs:
        previous_display = previous_slot.value(previous_value).display_name
        current_display = current_slot.value(current_value).display_name
        values.append(
            f"If the previous {previous_slot.display_name}'s "
            f"{previous_slot.attribute_name} was {previous_display}, choose "
            f"{current_slot.attribute_name} = {current_display}."
        )
    return tuple(values)  # type: ignore[return-value]


def _render_candidate(
    attribute_name: str,
    title: str,
    value_id: str,
    display_name: str,
) -> str:
    del value_id
    return f"- Product: {title}\n  Confirmed {attribute_name}: {display_name}"
