from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

from .native_action_codec import parse_native_bracket_action
from .persistent_workspace import (
    WORKSPACE_TOOL_OPS,
    WorkspaceActionError,
    parse_workspace_action,
)
from .reward_hierarchy import INVALID_ACTION_PENALTY, WRONG_BUY_TERMINAL_FAILURE


MEMORY_ACTION_OPS = frozenset(
    {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
)
WORKSPACE_ACTION_OPS = frozenset(WORKSPACE_TOOL_OPS)
FORMAL_NATIVE_ACTION_OPS = frozenset(
    {
        *MEMORY_ACTION_OPS,
        *WORKSPACE_ACTION_OPS,
        "SEARCH",
        "CLICK",
        "BUY",
        "ASK",
        "CLARIFY",
        "INVALID",
    }
)
_MEMORY_ACTION_RE = re.compile(
    r"\A(ADD|UPDATE|DELETE|RETRIEVE|SUMMARY|FILTER)\s+(\{.*\})\Z",
    re.DOTALL,
)
_ASK_ACTION_RE = re.compile(r"\AASK\s+(\{.*\})\Z", re.DOTALL)


def canonical_tool_op(value: Any) -> str:
    op = str(value).strip().upper()
    if op not in FORMAL_NATIVE_ACTION_OPS - {"INVALID"}:
        raise ValueError(f"Unsupported formal native tool op: {value!r}.")
    return op


def infer_raw_action_op(raw_action: str) -> str:
    text = str(raw_action).strip()
    native = parse_native_bracket_action(text)
    if native is not None:
        # The environment strips the bracket argument and rejects an empty
        # value. Keep the reward ledger on the same INVALID branch instead of
        # classifying ``search[ ]``/``click[\t]`` as executed native actions.
        action_name, argument = native
        if not argument.strip():
            return "INVALID"
        return action_name.upper()
    memory = _MEMORY_ACTION_RE.fullmatch(text)
    if memory is not None:
        try:
            payload = json.loads(memory.group(2))
        except json.JSONDecodeError:
            return "INVALID"
        if not isinstance(payload, dict):
            return "INVALID"
        return memory.group(1).upper()
    try:
        workspace_action = parse_workspace_action(text)
    except WorkspaceActionError:
        return "INVALID"
    if workspace_action is not None:
        return workspace_action.tool_name.upper()
    ask = _ASK_ACTION_RE.fullmatch(text)
    if ask is not None:
        try:
            payload = json.loads(ask.group(1))
        except json.JSONDecodeError:
            return "INVALID"
        if isinstance(payload, dict):
            return "ASK"
    return "INVALID"


def resolve_formal_action_op(
    raw_action: str,
    tool_ops: Sequence[dict[str, Any]],
) -> str:
    """Bind a policy action to the action actually executed by WebShop."""

    inferred = infer_raw_action_op(raw_action)
    if len(tool_ops) > 1:
        raise ValueError("One policy action produced multiple formal tool operations.")
    if not tool_ops:
        return inferred
    tool_op = canonical_tool_op(tool_ops[0].get("op"))
    if inferred == "SEARCH" and tool_op != "SEARCH":
        raise ValueError("A native search action produced a non-SEARCH tool operation.")
    if inferred == "CLICK" and tool_op not in {"CLICK", "BUY"}:
        raise ValueError("A native click action produced a non-CLICK/BUY tool operation.")
    if inferred in MEMORY_ACTION_OPS and tool_op != inferred:
        raise ValueError("A memory action produced a different tool operation.")
    if inferred in WORKSPACE_ACTION_OPS and tool_op != inferred:
        raise ValueError("A workspace action produced a different tool operation.")
    if inferred == "ASK" and tool_op != "CLARIFY":
        raise ValueError("An ASK action produced a non-CLARIFY tool operation.")
    if inferred == "INVALID":
        raise ValueError("An invalid raw action claims a successful tool operation.")
    return tool_op


def build_reward_components(
    *,
    raw_action: str,
    reward: float,
    step: int,
    tool_ops: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Record the existing immediate reward without changing reward semantics."""

    reward = float(reward)
    if not math.isfinite(reward):
        raise ValueError("Formal native reward must be finite.")
    if isinstance(step, bool) or int(step) != step or int(step) <= 0:
        raise ValueError("Formal native reward step must be a positive integer.")
    action_op = resolve_formal_action_op(raw_action, tool_ops)

    if action_op == "BUY":
        event = tool_ops[0]
        purchase_correct = event.get("purchase_correct")
        if not isinstance(purchase_correct, bool):
            raise ValueError("A BUY reward lacks boolean purchase_correct evidence.")
        if not purchase_correct:
            if reward != WRONG_BUY_TERMINAL_FAILURE:
                raise ValueError(
                    "An incorrect formal BUY must retain reward "
                    f"{WRONG_BUY_TERMINAL_FAILURE}."
                )
            values = [("buy_committed_incorrect", reward)]
        elif reward == 1.0:
            values = [("buy_committed_correct", 1.0)]
        elif reward == 2.0 and event.get("terminal") is True:
            values = [
                ("buy_committed_correct", 1.0),
                ("bundle_complete_bonus", 1.0),
            ]
        else:
            raise ValueError(
                "A correct formal BUY must retain reward 1.0, or 2.0 at bundle completion."
            )
    else:
        if action_op == "INVALID":
            if reward != INVALID_ACTION_PENALTY:
                raise ValueError(
                    "An invalid formal action must retain reward "
                    f"{INVALID_ACTION_PENALTY}."
                )
            name = "invalid_action"
        else:
            if reward != 0.0:
                raise ValueError(
                    f"A non-BUY {action_op} action unexpectedly changed reward to {reward}."
                )
            name = f"{action_op.lower()}_transition"
        values = [(name, reward)]

    return [
        {
            "name": name,
            "value": value,
            "op": action_op,
            "step": int(step),
        }
        for name, value in values
    ]
