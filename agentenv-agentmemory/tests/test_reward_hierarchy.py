from __future__ import annotations

import math
import unittest

from agentenv_agentmemory.reward_hierarchy import (
    EXACT_REPEAT_ACTION_PENALTY,
    INVALID_ACTION_PENALTY,
    MAX_ROUND_TIMEOUT_FAILURE,
    RewardHierarchyError,
    WRONG_BUY_TERMINAL_FAILURE,
    bind_max_round_timeout_failure,
)


MAX_ROUNDS = 30


def make_timeout_step(
    op: str,
    *,
    reward: float = 0.0,
    tool_op: bool = True,
    raw_action: str | None = None,
    action_execution: dict[str, object] | None = None,
) -> dict[str, object]:
    component: dict[str, object] = {
        "name": f"{op.lower()}_transition",
        "value": reward,
        "op": op,
        "step": MAX_ROUNDS,
    }
    if raw_action is not None:
        component["raw_action"] = raw_action
    return {
        "done": False,
        "task_round": MAX_ROUNDS,
        "score": reward,
        "env_info_after": {
            "reward_components": [component],
            "tool_ops": ([{"op": op, "step": MAX_ROUNDS}] if tool_op else []),
        },
        "action_execution": action_execution,
    }


class MaxRoundTimeoutBindingTests(unittest.TestCase):
    def test_all_failure_components_use_the_micro_penalty(self) -> None:
        self.assertEqual(INVALID_ACTION_PENALTY, -0.01)
        self.assertEqual(EXACT_REPEAT_ACTION_PENALTY, -0.01)
        self.assertEqual(MAX_ROUND_TIMEOUT_FAILURE, -0.01)
        self.assertEqual(WRONG_BUY_TERMINAL_FAILURE, -0.01)

    def assert_timeout_bound(self, step: dict[str, object], *, op: str) -> None:
        before_score = float(step["score"])

        bind_max_round_timeout_failure(step, max_rounds=MAX_ROUNDS)

        components = step["env_info_after"]["reward_components"]
        timeout = components[-1]
        self.assertEqual(timeout["name"], "max_round_timeout_failure")
        self.assertEqual(timeout["value"], MAX_ROUND_TIMEOUT_FAILURE)
        self.assertEqual(timeout["op"], op)
        self.assertEqual(timeout["step"], MAX_ROUNDS)
        self.assertEqual(timeout["max_rounds"], MAX_ROUNDS)
        self.assertTrue(
            math.isclose(
                sum(float(component["value"]) for component in components),
                float(step["score"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(step["score"], before_score + MAX_ROUND_TIMEOUT_FAILURE)
        self.assertEqual(step["immediate_reward"], step["score"])
        self.assertEqual(step["outcome"], "max_rounds")
        self.assertIs(step["max_round_timeout_failure_applied"], True)
        self.assertIs(step["done"], False)
        self.assertNotIn("trajectory_terminal", step)

    def test_binds_native_search_and_click(self) -> None:
        for op in ("SEARCH", "CLICK"):
            with self.subTest(op=op):
                self.assert_timeout_bound(make_timeout_step(op), op=op)

    def test_binds_nonterminal_correct_buy(self) -> None:
        step = make_timeout_step(
            "BUY",
            reward=1.0,
            raw_action="click[Buy Now]",
        )
        self.assert_timeout_bound(step, op="BUY")

    def test_binds_each_memory_operation(self) -> None:
        for op in ("ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"):
            with self.subTest(op=op):
                self.assert_timeout_bound(make_timeout_step(op), op=op)

    def test_binds_invalid_action_from_raw_action_without_tool_op(self) -> None:
        step = make_timeout_step(
            "INVALID",
            reward=-0.01,
            tool_op=False,
            raw_action="not a valid action",
        )
        self.assert_timeout_bound(step, op="INVALID")

    def test_keeps_legacy_action_execution_as_optional_evidence(self) -> None:
        execution = {
            "task_round": MAX_ROUNDS,
            "status": "executed",
            "executed_action_op": "SEARCH",
        }
        step = make_timeout_step(
            "SEARCH",
            tool_op=False,
            action_execution=execution,
        )
        self.assert_timeout_bound(step, op="SEARCH")

    def assert_rejected_without_mutation(
        self,
        step: dict[str, object],
        message: str,
    ) -> None:
        components = step.get("env_info_after", {}).get("reward_components")
        before = (
            [dict(component) for component in components]
            if isinstance(components, list)
            else None
        )
        with self.assertRaisesRegex(RewardHierarchyError, message):
            bind_max_round_timeout_failure(step, max_rounds=MAX_ROUNDS)
        if before is not None:
            self.assertEqual(components, before)
        self.assertNotIn("max_round_timeout_failure_applied", step)

    def test_rejects_missing_or_empty_reward_ledger(self) -> None:
        for components in (None, []):
            with self.subTest(components=components):
                step = make_timeout_step("SEARCH")
                if components is None:
                    del step["env_info_after"]["reward_components"]
                else:
                    step["env_info_after"]["reward_components"] = components
                self.assert_rejected_without_mutation(step, "missing its reward ledger")

    def test_rejects_missing_operation(self) -> None:
        step = make_timeout_step("SEARCH")
        del step["env_info_after"]["reward_components"][0]["op"]
        self.assert_rejected_without_mutation(step, "missing an action operation")

    def test_rejects_non_string_raw_action_without_mutation(self) -> None:
        step = make_timeout_step("INVALID", tool_op=False)
        step["env_info_after"]["reward_components"][0]["raw_action"] = []
        self.assert_rejected_without_mutation(step, "invalid raw action")

    def test_rejects_missing_or_mismatched_step(self) -> None:
        cases = (
            ("component", None),
            ("component", 29),
            ("tool", None),
            ("tool", 29),
        )
        for target, value in cases:
            with self.subTest(target=target, value=value):
                step = make_timeout_step("SEARCH")
                record = (
                    step["env_info_after"]["reward_components"][0]
                    if target == "component"
                    else step["env_info_after"]["tool_ops"][0]
                )
                if value is None:
                    del record["step"]
                else:
                    record["step"] = value
                self.assert_rejected_without_mutation(step, "not bound to rollout round")

    def test_rejects_reward_sum_mismatch(self) -> None:
        step = make_timeout_step("SEARCH")
        step["score"] = 1.0
        self.assert_rejected_without_mutation(step, "does not equal the step score")

    def test_rejects_operation_mismatch(self) -> None:
        step = make_timeout_step("SEARCH")
        step["env_info_after"]["tool_ops"][0]["op"] = "CLICK"
        self.assert_rejected_without_mutation(step, "disagrees across")


if __name__ == "__main__":
    unittest.main()
