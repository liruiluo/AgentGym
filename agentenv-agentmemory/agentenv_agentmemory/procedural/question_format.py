from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from .schema import AttributeTransition
from .scenarios import scenario_by_id


QUESTION_FORMAT_VERSION = "natural_customer_approved_shortlist_chain_question_v9"


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
            + "\nThe immediately preceding purchase and its attribute value are not "
            "repeated in this session, and prior action messages do not reveal the filename "
            "you chose. As the first action, issue exactly shell_command "
            "{\"command\":\"rg --hidden -n '^Confirmed ' .\",\"workdir\":\".\","
            "\"timeout_ms\":10000} to discover and print saved Confirmed lines; do not guess a "
            "path. A command without --hidden can miss the note. If stdout is empty, retry that "
            "exact command and do nothing else. "
            "Before any catalog search, click, or new file write, confirm that the shell "
            f"output contains the preceding {previous_slot.attribute_name} note. Do not "
            "infer or recreate the previous value from the two table rows. Use the read "
            f"{previous_slot.attribute_name} value and buy "
            f"the matching {slot.display_name}. Then follow this order: search the chosen "
            "approved card's complete Product title without shortening it; open only a result "
            "whose complete visible title exactly matches that card, including size, count, and "
            "pack qualifiers; only on that exact product page copy the card's Confirmed line "
            "verbatim into one new Add File note; after Done!, click[Buy Now]."
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
        "color, material, flavor, or other attribute. Copy the chosen card's complete Product "
        "title verbatim into search[...]; do not shorten it. Open only a result whose complete "
        "visible title exactly equals that card, including size, count, and pack qualifiers.\n"
        "2. The pairing table is this customer's own plan, so both approved products "
        "may be ordinary real-world choices. Do not replace the table with general "
        "product knowledge.\n"
        "3. In sessions 1 through 5, do not write a note before opening the exact selected "
        "listing. Before click[Buy Now], use Add File once with a new path to preserve only "
        "the selected card's complete Confirmed field name and value. Copy the card's entire "
        "Confirmed line verbatim; never invent a `Confirmed ... to buy:` field or replace the "
        "field and value with a product title. Use exactly one content "
        "line beginning with +Confirmed; every Add File content line must begin with +. Copy "
        "that field; do not replace it with a generic certified, natural, normal, or boolean "
        "label. Once the environment "
        "returns Done!, leave that note unchanged and commit the purchase; do not add, "
        "delete, or rewrite the successful note. In the next session, read that same "
        "note's contents before any catalog action or replacement write.\n"
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
