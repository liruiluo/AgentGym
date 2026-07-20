from __future__ import annotations

import math
from typing import Any, MutableMapping, Sequence


CORRECT_BUY_REWARD = 1.0
MEMORY_PROGRESS_REWARD = 0.05
MICRO_ACTION_PENALTY = -0.01
FIRST_VALID_ADD_BONUS = 0.01
FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS = 0.01
NONTERMINAL_NOOP_PENALTY = MICRO_ACTION_PENALTY
INVALID_ACTION_PENALTY = MICRO_ACTION_PENALTY
EXACT_REPEAT_ACTION_PENALTY = MICRO_ACTION_PENALTY
NONTERMINAL_NEGATIVE_SHAPING_BUDGET = 0.04
MAX_ROUND_TIMEOUT_FAILURE = MICRO_ACTION_PENALTY
WRONG_BUY_TERMINAL_FAILURE = MICRO_ACTION_PENALTY


# Every currently reachable negative component that is not itself a terminal
# outcome must be named here. Unknown negative components fail closed instead
# of silently reintroducing an unbounded failure path.
NONTERMINAL_NEGATIVE_SHAPING_COMPONENTS = frozenset(
    {
        "catalog_search_no_results_noop",
        "catalog_search_repeated_same_query_noop",
        "dependent_memory_ready_answer_instead_of_buy",
        "dependent_memory_ready_answer_no_progress",
        "exact_repeated_valid_zero_reward_action",
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
    if type(max_rounds) is not int or max_rounds <= 0:
        raise RewardHierarchyError("Max-round timeout requires a positive integer bound.")
    task_round = step.get("task_round")
    if type(task_round) is not int or task_round != max_rounds:
        raise RewardHierarchyError(
            "Max-round timeout must bind to the declared final rollout round."
        )
    if (
        isinstance(penalty, bool)
        or not isinstance(penalty, (int, float))
        or not math.isfinite(float(penalty))
        or float(penalty) >= 0.0
    ):
        raise RewardHierarchyError("Max-round timeout penalty must be finite and negative.")

    env_info_after = step.get("env_info_after")
    if not isinstance(env_info_after, MutableMapping):
        raise RewardHierarchyError("Timeout step is missing env_info_after.")
    components = env_info_after.get("reward_components")
    if not isinstance(components, list) or not components:
        raise RewardHierarchyError("Timeout step is missing its reward ledger.")
    ledger_op, ledger_score = _validate_timeout_reward_ledger(
        components,
        task_round=task_round,
    )

    score = step.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise RewardHierarchyError("Timeout step score must be finite and numeric.")
    current_score = float(score)
    if not math.isclose(current_score, ledger_score, rel_tol=1e-9, abs_tol=1e-9):
        raise RewardHierarchyError(
            "Pre-timeout reward ledger does not equal the step score: "
            f"ledger={ledger_score} score={current_score}."
        )

    evidence_ops = _collect_timeout_action_ops(
        step,
        components=components,
        task_round=task_round,
    )
    if any(op != ledger_op for op in evidence_ops):
        raise RewardHierarchyError(
            "Timeout action operation disagrees across reward and execution evidence."
        )

    components.append(
        {
            "name": "max_round_timeout_failure",
            "value": float(penalty),
            "op": ledger_op,
            "step": task_round,
            "max_rounds": max_rounds,
        }
    )
    step["score"] = current_score + float(penalty)
    step["immediate_reward"] = step["score"]
    step["outcome"] = "max_rounds"
    step["max_round_timeout_failure_applied"] = True


def _validate_timeout_reward_ledger(
    components: Sequence[MutableMapping[str, Any]],
    *,
    task_round: int,
) -> tuple[str, float]:
    ledger_op: str | None = None
    ledger_score = 0.0
    for index, component in enumerate(components):
        if not isinstance(component, MutableMapping):
            raise RewardHierarchyError(
                f"Timeout reward component {index} is not an object."
            )
        name = component.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RewardHierarchyError(
                f"Timeout reward component {index} is missing a name."
            )
        if name == "max_round_timeout_failure":
            raise RewardHierarchyError("Max-round timeout component is already present.")
        component_op = _canonical_timeout_op(
            component.get("op"),
            source=f"reward component {index}",
        )
        if ledger_op is None:
            ledger_op = component_op
        elif component_op != ledger_op:
            raise RewardHierarchyError(
                "Timeout reward ledger contains multiple action operations."
            )
        _require_timeout_step(
            component.get("step"),
            task_round=task_round,
            source=f"reward component {index}",
        )
        value = component.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RewardHierarchyError(
                f"Timeout reward component {index} has a non-finite value."
            )
        ledger_score += float(value)

    if ledger_op is None:
        raise RewardHierarchyError("Timeout step is missing its reward ledger.")
    return ledger_op, ledger_score


def _collect_timeout_action_ops(
    step: MutableMapping[str, Any],
    *,
    components: Sequence[MutableMapping[str, Any]],
    task_round: int,
) -> list[str]:
    env_info_after = step["env_info_after"]
    tool_ops = env_info_after.get("tool_ops")
    if not isinstance(tool_ops, list):
        raise RewardHierarchyError("Timeout step is missing its tool-operation ledger.")

    evidence_ops: list[str] = []
    if len(tool_ops) > 1:
        raise RewardHierarchyError(
            "Timeout step contains multiple same-action tool operations."
        )
    if tool_ops:
        tool_op = tool_ops[0]
        if not isinstance(tool_op, MutableMapping):
            raise RewardHierarchyError("Timeout tool operation is not an object.")
        _require_timeout_step(
            tool_op.get("step"),
            task_round=task_round,
            source="tool operation",
        )
        evidence_ops.append(
            _canonical_timeout_op(tool_op.get("op"), source="tool operation")
        )

    raw_action_values = [
        component["raw_action"]
        for component in components
        if "raw_action" in component
    ]
    if any(
        not isinstance(action, str) or not action.strip()
        for action in raw_action_values
    ):
        raise RewardHierarchyError("Timeout reward ledger contains an invalid raw action.")
    raw_actions = set(raw_action_values)
    if len(raw_actions) > 1:
        raise RewardHierarchyError(
            "Timeout reward ledger contains multiple raw policy actions."
        )
    if raw_actions:
        from .formal_native_contract import resolve_formal_action_op

        try:
            resolved_op = resolve_formal_action_op(
                next(iter(raw_actions)),
                tool_ops,
            )
        except (TypeError, ValueError) as exc:
            raise RewardHierarchyError(
                "Timeout raw action cannot be bound to its tool operation."
            ) from exc
        evidence_ops.append(resolved_op)

    execution = step.get("action_execution")
    if execution is not None:
        if not isinstance(execution, MutableMapping):
            raise RewardHierarchyError("Timeout action execution evidence is not an object.")
        if "task_round" in execution:
            _require_timeout_step(
                execution.get("task_round"),
                task_round=task_round,
                source="action execution",
            )
        status = execution.get("status")
        if status == "executed":
            execution_op = execution.get("executed_action_op")
        elif status in {"rejected", "server_rejected"}:
            execution_op = execution.get("rejected_action_op") or "INVALID"
        else:
            raise RewardHierarchyError("Timeout action execution status is invalid.")
        evidence_ops.append(
            _canonical_timeout_op(execution_op, source="action execution")
        )

    if not evidence_ops:
        raise RewardHierarchyError("Timeout step has no bound action operation evidence.")
    return evidence_ops


def _canonical_timeout_op(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RewardHierarchyError(f"Timeout {source} is missing an action operation.")
    canonical = value.strip().upper()
    if value != canonical:
        raise RewardHierarchyError(
            f"Timeout {source} action operation is not canonical: {value!r}."
        )
    return canonical


def _require_timeout_step(value: Any, *, task_round: int, source: str) -> None:
    if type(value) is not int or value != task_round:
        raise RewardHierarchyError(
            f"Timeout {source} is not bound to rollout round {task_round}."
        )
