from __future__ import annotations

import math
from typing import Any, MutableMapping, Sequence


CORRECT_BUY_REWARD = 1.0
MEMORY_PROGRESS_REWARD = 0.05
NONTERMINAL_NOOP_PENALTY = -0.01
NONTERMINAL_NEGATIVE_SHAPING_BUDGET = 0.04
MAX_ROUND_TIMEOUT_FAILURE = -0.05
WRONG_BUY_TERMINAL_FAILURE = -0.10


# Every currently reachable negative component that is not itself a terminal
# outcome must be named here. Unknown negative components fail closed instead
# of silently reintroducing an unbounded failure path.
NONTERMINAL_NEGATIVE_SHAPING_COMPONENTS = frozenset(
    {
        "catalog_search_no_results_noop",
        "catalog_search_repeated_same_query_noop",
        "dependent_memory_ready_answer_instead_of_buy",
        "dependent_memory_ready_answer_no_progress",
        "invalid_action",
        "memory_add_duplicate_visible_product_reference",
        "memory_delete_product_anchor",
        "memory_retrieve_empty_repeat_same_query_noop",
        "memory_retrieve_nonempty_repeat_same_session",
        "memory_retrieve_source_same_session_repeat_noop",
        "memory_update_drops_product_anchor",
        "missing_memory_before_source_purchase",
        "missing_retrieved_memory_before_dependent_purchase",
        "missing_search_before_memoryarena_source_purchase",
        "premature_answer_before_bundle_complete",
        "retrieved_memory_irrelevant_to_dependent_purchase",
        "source_memory_ready_answer_instead_of_buy",
        "source_buy_without_memory_before_close",
        "source_search_after_memory_ready_noop",
    }
)

TERMINAL_NEGATIVE_OUTCOME_COMPONENTS = frozenset(
    {
        "buy_committed_incorrect",
        "max_round_timeout_failure",
    }
)


class RewardHierarchyError(RuntimeError):
    pass


def apply_nonterminal_negative_shaping_budget(
    components: Sequence[MutableMapping[str, Any]],
    *,
    spent: float,
    terminal: bool,
    budget: float = NONTERMINAL_NEGATIVE_SHAPING_BUDGET,
) -> float:
    """Clip all non-terminal negative shaping to one shared session budget."""

    if not math.isfinite(float(spent)) or float(spent) < 0.0:
        raise RewardHierarchyError(f"Invalid negative-shaping spend: {spent!r}.")
    if not math.isfinite(float(budget)) or float(budget) < 0.0:
        raise RewardHierarchyError(f"Invalid negative-shaping budget: {budget!r}.")

    normalized_spent = min(float(spent), float(budget))
    for component in components:
        name = component.get("name")
        value = component.get("value")
        if not isinstance(name, str) or not name:
            raise RewardHierarchyError("Reward component is missing a name.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RewardHierarchyError(
                f"Reward component {name!r} has a non-numeric value."
            )
        requested_value = float(value)
        if not math.isfinite(requested_value):
            raise RewardHierarchyError(
                f"Reward component {name!r} has a non-finite value."
            )
        if requested_value >= 0.0:
            continue

        if name in TERMINAL_NEGATIVE_OUTCOME_COMPONENTS:
            if not terminal:
                raise RewardHierarchyError(
                    f"Terminal outcome component {name!r} appeared on a non-terminal row."
                )
            continue
        if name not in NONTERMINAL_NEGATIVE_SHAPING_COMPONENTS:
            raise RewardHierarchyError(
                f"Unknown non-terminal negative shaping component {name!r}."
            )

        remaining = max(0.0, float(budget) - normalized_spent)
        applied_magnitude = min(-requested_value, remaining)
        normalized_spent += applied_magnitude
        component["requested_value"] = requested_value
        component["value"] = -applied_magnitude
        component["negative_shaping_budget"] = float(budget)
        component["negative_shaping_spent_after"] = normalized_spent
        component["negative_shaping_budget_exhausted"] = math.isclose(
            normalized_spent,
            float(budget),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    return normalized_spent


def bind_max_round_timeout_failure(
    step: MutableMapping[str, Any],
    *,
    max_rounds: int,
    penalty: float = MAX_ROUND_TIMEOUT_FAILURE,
) -> None:
    """Bind an explicit terminal timeout component before suffix credit."""

    if bool(step.get("done")):
        raise RewardHierarchyError("Cannot bind max-round timeout to a done step.")
    if int(max_rounds) <= 0 or int(step.get("task_round", -1)) != int(max_rounds):
        raise RewardHierarchyError(
            "Max-round timeout must bind to the declared final rollout round."
        )
    if not math.isfinite(float(penalty)) or float(penalty) >= 0.0:
        raise RewardHierarchyError("Max-round timeout penalty must be finite and negative.")

    env_info_after = step.get("env_info_after")
    if not isinstance(env_info_after, MutableMapping):
        raise RewardHierarchyError("Timeout step is missing env_info_after.")
    components = env_info_after.get("reward_components")
    if not isinstance(components, list):
        raise RewardHierarchyError("Timeout step is missing its reward ledger.")
    if any(component.get("name") == "max_round_timeout_failure" for component in components):
        raise RewardHierarchyError("Max-round timeout component is already present.")

    current_score = float(step.get("score"))
    ledger_score = sum(float(component["value"]) for component in components)
    if not math.isclose(current_score, ledger_score, rel_tol=1e-9, abs_tol=1e-9):
        raise RewardHierarchyError(
            "Pre-timeout reward ledger does not equal the step score: "
            f"ledger={ledger_score} score={current_score}."
        )

    execution = step.get("action_execution")
    if not isinstance(execution, MutableMapping):
        raise RewardHierarchyError("Timeout step is missing action execution evidence.")
    if execution.get("status") == "executed":
        op = execution.get("executed_action_op")
    else:
        op = execution.get("rejected_action_op") or "INVALID"
    if not isinstance(op, str) or not op:
        raise RewardHierarchyError("Timeout step has no bound action operation.")

    components.append(
        {
            "name": "max_round_timeout_failure",
            "value": float(penalty),
            "op": op,
            "step": int(step["task_round"]),
            "max_rounds": int(max_rounds),
        }
    )
    step["score"] = current_score + float(penalty)
    step["immediate_reward"] = step["score"]
    step["outcome"] = "max_rounds"
    step["max_round_timeout_failure_applied"] = True
