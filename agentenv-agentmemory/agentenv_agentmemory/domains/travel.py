from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..runtime.domain import DomainContract, DomainTransition
from .memoryarena_dataset import (
    MemoryArenaDatasetProvenance,
    verify_memoryarena_dataset_provenance,
)


TRAVEL_CONTRACT_MODES = ("failfast", "paper_eval")
TRAVEL_SURFACES = {
    "failfast": "memoryarena_travel_planner_failfast_one_action_v3",
    "paper_eval": "memoryarena_travel_planner_paper_eval_one_action_v3",
}
TRAVEL_DOMAIN_ID = "travel_planner"
TRAVEL_DATASET_CONFIG = "group_travel_planner"
TRAVEL_FROZEN_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
TRAVEL_PAPER_GROUP_COUNT = 270
TRAVEL_PAPER_PHASE_COUNT = 1869
TRAVEL_MAX_STEPS_PER_PHASE = 30
TRAVEL_TOOL_OPS = (
    "FlightSearch",
    "RestaurantSearch",
    "AccommodationSearch",
    "AttractionSearch",
    "DistanceMatrix",
    "CitySearch",
)
TRAVEL_UPSTREAM_RELATIVE_PATHS = (
    "run_travel.py",
    "agent/travel_planner.py",
    "env/__init__.py",
    "env/env_systems/__init__.py",
    "env/env_systems/base_env.py",
    "env/env_systems/travel_env.py",
    "env/env_systems/travel_planner_env/__init__.py",
    "env/env_systems/travel_planner_env/combination.py",
    "env/env_systems/travel_planner_env/data_loader.py",
    "env/env_systems/travel_planner_env/eval.py",
    "env/env_systems/travel_planner_env/prompts.py",
    "env/env_systems/travel_planner_env/tool_executor.py",
    "env/env_systems/travel_planner_env/tool_schemas.py",
    "env/env_systems/travel_planner_env/tools/accommodations.py",
    "env/env_systems/travel_planner_env/tools/attractions.py",
    "env/env_systems/travel_planner_env/tools/cities.py",
    "env/env_systems/travel_planner_env/tools/distance_matrix.py",
    "env/env_systems/travel_planner_env/tools/flights.py",
    "env/env_systems/travel_planner_env/tools/func.py",
    "env/env_systems/travel_planner_env/tools/__init__.py",
    "env/env_systems/travel_planner_env/tools/restaurants.py",
)
TRAVEL_DATABASE_ASSET_SPECS = {
    "flights": (
        "flights/clean_Flights_2022.csv",
        "8dafdb0e3f8b79ce599a1e612a772865295bc226b46e5fb278368f7255b11cee",
    ),
    "restaurants": (
        "restaurants/clean_restaurant_2022.csv",
        "a38ef00a3398b1ef78306a9b1851c491723d27899c6a1d4a3d6fe042929c0f26",
    ),
    "accommodations": (
        "accommodations/clean_accommodations_2022.csv",
        "7e70cc8cb573e824267be47153a3cf7934c839f01c955fb523d18be405921d86",
    ),
    "attractions": (
        "attractions/attractions.csv",
        "8007fa851ba5b3adb205fbd298169c95116b297cef7ccd21dbea26ecce2605d8",
    ),
    "distance_matrix": (
        "googleDistanceMatrix/distance.csv",
        "7dc1a26dc7877ae96ce63e74be0d9660bbdef96ffccaffce441128e3a4864a27",
    ),
    "cities": (
        "background/citySet_with_states.txt",
        "a3d18b5c692857cd561cbb4ad8d221ac3eb6c47af7c787a3489e3ffa184ca4d6",
    ),
}
_ACTION_RE = re.compile(
    r"\A(" + "|".join((*TRAVEL_TOOL_OPS, "SUBMIT_PLAN")) + r")\s+(\{.*\})\Z",
    flags=re.DOTALL,
)
_NAME_RE = re.compile(r"\AI am (\w+)\.")


def _travel_contract(mode: str) -> DomainContract:
    if mode == "failfast":
        mode_prompt = (
            "This is the explicitly named fail-fast training variant. A correct "
            "traveler plan earns +1 and advances exactly once. An incorrect plan "
            "earns 0, does not advance, and ends the episode immediately while "
            "preserving rewards earned for earlier travelers. Exhausting a "
            "traveler's 30 native actions without a plan has the same failed "
            "terminal outcome."
        )
    elif mode == "paper_eval":
        mode_prompt = (
            "This is the paper-evaluation continuation variant. Every submitted "
            "traveler plan is recorded and advances exactly once, including an "
            "incorrect plan. A correct plan earns +1 and an incorrect plan earns "
            "0. Exhausting a traveler's 30 native actions records an empty, "
            "incorrect plan and advances. Completing the group emits the official "
            "Travel PS/SPS/SR contribution ledger."
        )
    else:  # pragma: no cover - callers validate before construction.
        raise ValueError(f"unsupported Travel contract mode: {mode}")
    return DomainContract(
        contract_id=(
            f"memoryarena_travel_planner_{mode}_one_action_v3_20260722"
        ),
        system_prompt=(
            "You are operating the MemoryArena Travel Planner domain. Each episode "
            "contains a fixed base traveler followed by sequential traveler phases. "
            + mode_prompt
            + " Use one native JSON action at a time to inspect the frozen local "
            "travel database. Submit the current traveler's complete plan with "
            "SUBMIT_PLAN. The server privately evaluates the plan and never exposes "
            "answer labels or verifier reasoning. This adapter executes exactly one "
            "action per policy turn and does not claim parity with upstream turns "
            "that batch multiple tool calls. The submitted plan must start with the "
            "current traveler's actual name, not the literal word Name. For example, "
            "Eric's header is === Eric's Plan ===. Then contain one Day N: block per "
            "trip day. Every day block must contain all seven slots in this order, "
            "one per line: Current City, Transportation, Breakfast, Attraction, "
            "Lunch, Dinner, Accommodation. Use - when a slot does not apply, and "
            "separate multiple attractions with semicolons. Do not include prices, "
            "ratings, cuisines, comments, or explanations in the plan."
        ),
        native_action_descriptions=(
            'FlightSearch {"origin": "...", "destination": "...", "date": "YYYY-MM-DD"}',
            'RestaurantSearch {"city": "..."}',
            'AccommodationSearch {"city": "..."}',
            'AttractionSearch {"city": "..."}',
            'DistanceMatrix {"origin": "...", "destination": "...", "mode": "driving|taxi"}',
            'CitySearch {"state": "..."}',
            'SUBMIT_PLAN {"plan": "=== Eric\'s Plan ===\\nDay 1: ..."}',
        ),
        max_steps=TRAVEL_MAX_STEPS_PER_PHASE,
    )


TRAVEL_CONTRACTS = {
    mode: _travel_contract(mode) for mode in TRAVEL_CONTRACT_MODES
}


@dataclass(frozen=True)
class TravelPhase:
    round_index: int
    name: str
    query: str
    ground_truth_plans: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TravelTask:
    task_id: str
    source_id: Any
    base_person: dict[str, Any]
    phases: tuple[TravelPhase, ...]


TravelJudge = Callable[[str, str, Sequence[dict[str, Any]]], bool]


class TravelPaperEvaluator:
    """Evaluate complete predictions with the attested upstream paper helpers."""

    def __init__(
        self,
        *,
        tasks,
        paper_eval_module,
        paper_parser_module,
        dataset_scope: str,
    ) -> None:
        self.tasks = tuple(tasks)
        self.tasks_by_source_id = {task.source_id: task for task in self.tasks}
        if len(self.tasks_by_source_id) != len(self.tasks):
            raise ValueError("Travel source ids must be unique for paper evaluation")
        self.paper_eval_module = paper_eval_module
        self.paper_parser_module = paper_parser_module
        self.dataset_scope = str(dataset_scope)
        self.ground_truth = {
            (task.source_id, phase.round_index): list(phase.ground_truth_plans)
            for task in self.tasks
            for phase in task.phases
        }
        self.base_plans = {
            task.source_id: task.base_person["daily_plans"] for task in self.tasks
        }

    def parse_prediction(self, plan_text: str) -> list[dict[str, Any]]:
        parsed = self.paper_parser_module.parse_plan_text(plan_text)
        if parsed is None:
            return []
        if not isinstance(parsed, list) or any(
            not isinstance(day, dict) for day in parsed
        ):
            raise TypeError("MemoryArena Travel paper parser returned invalid plan data")
        return [dict(day) for day in parsed]

    def evaluate_group(
        self,
        source_id: Any,
        predictions: Mapping[int, Sequence[dict[str, Any]]],
    ) -> dict[str, Any]:
        try:
            task = self.tasks_by_source_id[source_id]
        except KeyError as exc:
            raise ValueError(f"Unknown Travel source id for paper evaluation: {source_id}") from exc
        expected_rounds = {phase.round_index for phase in task.phases}
        observed_rounds = set(predictions)
        if observed_rounds != expected_rounds:
            raise ValueError(
                "Travel group paper evaluation requires exactly one plan per "
                f"traveler: source_id={source_id!r} "
                f"missing={sorted(expected_rounds - observed_rounds)} "
                f"extra={sorted(observed_rounds - expected_rounds)}"
            )

        full_pass_people = 0
        person_constraint_rates = []
        constraint_people = 0
        for phase in task.phases:
            prediction = predictions[phase.round_index]
            if not isinstance(prediction, (list, tuple)) or any(
                not isinstance(day, dict) for day in prediction
            ):
                raise TypeError(
                    "Travel paper predictions must be structured daily-plan lists"
                )
            truth = self.ground_truth[(source_id, phase.round_index)]
            constraint_slots = self.paper_eval_module.find_constraint_slots(
                self.ground_truth,
                self.base_plans,
                source_id,
                phase.round_index,
            )
            constraint_passed = 0
            constraint_total = 0
            for truth_day in truth:
                day_index = truth_day.get("days") or truth_day.get("day")
                for slot in self.paper_eval_module.SLOTS:
                    if (day_index, slot) not in constraint_slots:
                        continue
                    constraint_total += 1
                    if self.paper_eval_module.check_slot_pass(
                        truth,
                        prediction,
                        day_index,
                        slot,
                    ):
                        constraint_passed += 1
            if constraint_total:
                constraint_people += 1
                person_constraint_rates.append(
                    constraint_passed / constraint_total
                )
            if self.paper_eval_module.check_person_full_pass(truth, prediction):
                full_pass_people += 1

        total_people = len(task.phases)
        group_success = full_pass_people == total_people
        group_constraint_rate = (
            sum(person_constraint_rates) / len(person_constraint_rates)
            if person_constraint_rates
            else None
        )
        return {
            "metric_contract": "memoryarena_travel_eval_py_ps_sps_sr_v1",
            "dataset_scope": self.dataset_scope,
            "source_id": source_id,
            "complete": True,
            "full_pass_people": full_pass_people,
            "total_people": total_people,
            "group_success": group_success,
            "group_constraint_rate": group_constraint_rate,
            "constraint_people": constraint_people,
            "online_reward_is_separate": True,
        }

    def evaluate(
        self,
        predictions: Mapping[tuple[Any, int], Sequence[dict[str, Any]]],
    ) -> dict[str, Any]:
        expected_keys = {
            (task.source_id, phase.round_index)
            for task in self.tasks
            for phase in task.phases
        }
        observed_keys = set(predictions)
        missing = expected_keys.difference(observed_keys)
        extra = observed_keys.difference(expected_keys)
        if missing or extra:
            raise ValueError(
                "Travel paper evaluation requires exactly one plan per traveler: "
                f"missing={len(missing)} extra={len(extra)}"
            )

        parsed = {}
        for key, plan in predictions.items():
            if not isinstance(plan, (list, tuple)) or any(
                not isinstance(day, dict) for day in plan
            ):
                raise TypeError(
                    "Travel paper predictions must be structured daily-plan lists"
                )
            parsed[key] = [dict(day) for day in plan]

        contributions = []
        for task in self.tasks:
            contributions.append(
                self.evaluate_group(
                    task.source_id,
                    {
                        phase.round_index: parsed[(task.source_id, phase.round_index)]
                        for phase in task.phases
                    },
                )
            )

        full_pass_count = sum(item["full_pass_people"] for item in contributions)
        person_count = sum(item["total_people"] for item in contributions)
        group_pass_count = sum(bool(item["group_success"]) for item in contributions)
        group_constraint_rates = [
            item["group_constraint_rate"]
            for item in contributions
            if item["group_constraint_rate"] is not None
        ]
        group_count = len(self.tasks)
        ps = 100.0 * full_pass_count / person_count if person_count else 0.0
        sps = (
            100.0 * sum(group_constraint_rates) / len(group_constraint_rates)
            if group_constraint_rates
            else 0.0
        )
        sr = 100.0 * group_pass_count / group_count if group_count else 0.0
        return {
            "metric_contract": "memoryarena_travel_eval_py_ps_sps_sr_v1",
            "ps": ps,
            "sps": sps,
            "sr": sr,
            "full_pass_people": full_pass_count,
            "total_people": person_count,
            "successful_groups": group_pass_count,
            "total_groups": group_count,
            "groups_with_constraint_slots": len(group_constraint_rates),
            "slots": list(self.paper_eval_module.SLOTS),
            "online_reward_is_separate": True,
        }


class TravelPlannerFactory:
    domain_id = TRAVEL_DOMAIN_ID

    def __init__(
        self,
        *,
        contract_mode: str,
        tasks_path: str | Path,
        memoryarena_root: str | Path | None = None,
        database_path: str | Path | None = None,
        tool_executor: Any | None = None,
        judge: TravelJudge | None = None,
        expected_memoryarena_commit: str | None = None,
        dataset_provenance: MemoryArenaDatasetProvenance,
    ) -> None:
        if contract_mode not in TRAVEL_CONTRACT_MODES:
            raise ValueError(
                "Travel contract_mode must be one of: "
                + ", ".join(TRAVEL_CONTRACT_MODES)
            )
        self.contract_mode = contract_mode
        self.surface = TRAVEL_SURFACES[contract_mode]
        self.tasks_path = Path(tasks_path).expanduser().resolve()
        verify_memoryarena_dataset_provenance(
            self.tasks_path,
            expected_config=TRAVEL_DATASET_CONFIG,
            provenance=dataset_provenance,
        )
        injected_test = dataset_provenance.mode == "injected_test_fixture"
        if (
            not injected_test
            and expected_memoryarena_commit != TRAVEL_FROZEN_MEMORYARENA_COMMIT
        ):
            raise RuntimeError(
                "production Travel requires frozen MemoryArena commit "
                f"{TRAVEL_FROZEN_MEMORYARENA_COMMIT}"
            )
        self.dataset_provenance = dataset_provenance
        self.tasks = load_travel_tasks(self.tasks_path)
        self.paper_dataset_scope = (
            "memoryarena_group_travel_planner_frozen270"
            if not injected_test
            else "injected_test_fixture"
        )
        self.task_count = len(self.tasks)
        self.phase_count = sum(len(task.phases) for task in self.tasks)
        if (
            self.task_count != dataset_provenance.record_count
            or self.phase_count != dataset_provenance.phase_count
        ):
            raise RuntimeError(
                "Loaded Travel tasks differ from dataset provenance"
            )
        self.paper_panel_complete = (
            dataset_provenance.mode == "frozen_public_hf_dataset"
            and self.task_count == TRAVEL_PAPER_GROUP_COUNT
            and self.phase_count == TRAVEL_PAPER_PHASE_COUNT
        )
        self.max_phase_count = max(len(task.phases) for task in self.tasks)
        self.contract = replace(
            TRAVEL_CONTRACTS[contract_mode],
            max_steps=TRAVEL_MAX_STEPS_PER_PHASE * self.max_phase_count,
        )
        self.dataset_sha256 = self.dataset_provenance.sha256
        self.memoryarena_root = (
            Path(memoryarena_root).expanduser().resolve()
            if memoryarena_root is not None
            else None
        )
        if self.memoryarena_root is None and not injected_test:
            raise RuntimeError("production Travel requires memoryarena_root")
        self.upstream_provenance = (
            attest_travel_upstream(
                self.memoryarena_root,
                expected_commit=expected_memoryarena_commit,
            )
            if self.memoryarena_root is not None
            else {"mode": "injected_test_fixture"}
        )
        if not injected_test and (tool_executor is not None or judge is not None):
            raise RuntimeError(
                "production Travel cannot use injected tool or judge implementations"
            )
        upstream_module = None
        paper_eval_module = None
        paper_parser_module = None
        if tool_executor is None or judge is None:
            if self.memoryarena_root is None:
                raise RuntimeError(
                    "memoryarena_root is required when Travel tools or judge are not injected"
                )
            (
                upstream_module,
                executor_type,
                paper_eval_module,
                paper_parser_module,
            ) = _load_upstream(self.memoryarena_root)
            if tool_executor is None:
                if database_path is None:
                    raise RuntimeError(
                        "Travel database_path is required for all six native assets"
                    )
                db_path = Path(database_path).expanduser().resolve()
                self.database_provenance = attest_travel_database(db_path)
                tool_executor = executor_type(db_path=str(db_path))
            if judge is None:
                judge = _build_upstream_judge(upstream_module)
        else:
            self.database_provenance = {"mode": "injected_test_tool_executor"}
        if (
            contract_mode == "paper_eval"
            and upstream_module is None
            and self.memoryarena_root is not None
        ):
            (
                upstream_module,
                _,
                paper_eval_module,
                paper_parser_module,
            ) = _load_upstream(self.memoryarena_root)
        if not hasattr(self, "database_provenance"):
            self.database_provenance = {"mode": "injected_test_tool_executor"}
        self.tool_executor = tool_executor
        self.judge = judge
        self.paper_evaluator = (
            TravelPaperEvaluator(
                tasks=self.tasks,
                paper_eval_module=paper_eval_module,
                paper_parser_module=paper_parser_module,
                dataset_scope=self.paper_dataset_scope,
            )
            if (
                contract_mode == "paper_eval"
                and upstream_module is not None
                and paper_eval_module is not None
                and paper_parser_module is not None
            )
            else None
        )

    def create(self, env_uid: str):
        return TravelPlannerDriver(
            contract_mode=self.contract_mode,
            tasks=self.tasks,
            tool_executor=self.tool_executor,
            judge=self.judge,
            env_uid=env_uid,
            contract=self.contract,
            paper_evaluator=self.paper_evaluator,
        )

    def metadata(self) -> dict[str, Any]:
        paper_evaluation_available = (
            self.contract_mode == "paper_eval" and self.paper_evaluator is not None
        )
        paper_column_eligible = (
            paper_evaluation_available and self.paper_panel_complete
        )
        return {
            "source": "MemoryArena",
            "contract_mode": self.contract_mode,
            "semantic_variant": (
                "paper_metric_evaluation_continue_on_incorrect_one_action_v1"
                if self.contract_mode == "paper_eval"
                else "ordered_traveler_failfast_training_one_action_v1"
            ),
            "dataset_sha256": self.dataset_sha256,
            "task_count": self.task_count,
            "phase_count": self.phase_count,
            "dataset_provenance": self.dataset_provenance.metadata(),
            "database_provenance": self.database_provenance,
            "native_tool_ops": list(TRAVEL_TOOL_OPS),
            "online_reward_judge": {
                "id": "memoryarena_travel_env_slot_similarity_v1",
                "slots": [
                    "current_city",
                    "transportation",
                    "breakfast",
                    "attraction",
                    "lunch",
                    "dinner",
                    "accommodation",
                ],
            },
            "paper_evaluation": {
                "id": "memoryarena_travel_eval_py_ps_sps_sr_v1",
                "dataset_scope": self.paper_dataset_scope,
                "metrics": ["PS", "SPS", "SR"],
                "slots": [
                    "breakfast",
                    "lunch",
                    "dinner",
                    "accommodation",
                    "transportation",
                    "attraction",
                ],
                "available": paper_evaluation_available,
                "canonical_semantics": self.contract_mode == "paper_eval",
                "paper_panel_complete": self.paper_panel_complete,
                "paper_column_eligible": paper_column_eligible,
                "separate_from_online_reward": True,
            },
            "reward_contract": (
                "correct_plan_plus_1; incorrect_plan_0; always_advance"
                if self.contract_mode == "paper_eval"
                else "correct_plan_plus_1; incorrect_plan_0_terminal_no_advance"
            ),
            "wrong_submission_semantics": (
                "continue_to_next_traveler"
                if self.contract_mode == "paper_eval"
                else "terminal_failure_without_phase_advance"
            ),
            "phase_timeout_semantics": (
                "record_empty_incorrect_plan_and_advance"
                if self.contract_mode == "paper_eval"
                else "terminal_failure_without_phase_advance"
            ),
            "policy_action_granularity": "one_native_action_per_policy_turn",
            "action_granularity": {
                "variant": "one_action_v3",
                "policy_actions_per_turn": 1,
                "upstream_batched_model_turn_parity": False,
                "upstream_model_turn_may_batch_tool_calls": True,
            },
            "max_actions_per_phase": TRAVEL_MAX_STEPS_PER_PHASE,
            "phase_action_budget": {
                "limit": TRAVEL_MAX_STEPS_PER_PHASE,
                "counts": ["native", "invalid"],
                "memory_actions_consume_budget": False,
                "upstream_batched_model_turn_parity": False,
            },
            "upstream_agent_turn_budget": 30,
            "native_agent_turn_budget_equivalent": False,
            "batch_tool_call_parity_mode": {
                "available": False,
                "reason": (
                    "the shared AMG rollout protocol records exactly one action "
                    "per policy turn; native batch parity requires a separate "
                    "ordered multi-tool-call action and evidence schema"
                ),
            },
            "max_phase_count": self.max_phase_count,
            "max_episode_steps": self.contract.max_steps,
            "upstream_provenance": self.upstream_provenance,
        }

    def evaluate_paper_predictions(
        self,
        predictions: Mapping[tuple[Any, int], Sequence[dict[str, Any]]],
    ) -> dict[str, Any]:
        if self.contract_mode != "paper_eval" or self.paper_evaluator is None:
            raise RuntimeError(
                "paper evaluation requires an attested MemoryArena runtime"
            )
        return self.paper_evaluator.evaluate(predictions)

    def parse_submitted_plan(self, plan: str, name: str) -> list[dict[str, Any]]:
        if self.contract_mode != "paper_eval" or self.paper_evaluator is None:
            raise RuntimeError(
                "plan parsing requires an attested MemoryArena runtime"
            )
        return self.paper_evaluator.parse_prediction(plan)


class TravelPlannerDriver:
    domain_id = TRAVEL_DOMAIN_ID

    def __init__(
        self,
        *,
        contract_mode: str,
        tasks: Sequence[TravelTask],
        tool_executor: Any,
        judge: TravelJudge,
        env_uid: str,
        contract: DomainContract | None = None,
        paper_evaluator: TravelPaperEvaluator | None = None,
    ) -> None:
        if contract_mode not in TRAVEL_CONTRACT_MODES:
            raise ValueError(f"unsupported Travel contract mode: {contract_mode}")
        if not tasks:
            raise ValueError("TravelPlannerDriver requires tasks")
        if contract_mode == "failfast" and paper_evaluator is not None:
            raise ValueError("Travel failfast mode cannot attach a paper evaluator")
        self.contract_mode = contract_mode
        self.surface = TRAVEL_SURFACES[contract_mode]
        self.tasks = tuple(tasks)
        self.tool_executor = tool_executor
        self.judge = judge
        self.env_uid = env_uid
        self.contract = contract or replace(
            TRAVEL_CONTRACTS[contract_mode],
            max_steps=(
                TRAVEL_MAX_STEPS_PER_PHASE
                * max(len(task.phases) for task in self.tasks)
            ),
        )
        self.paper_evaluator = paper_evaluator
        self.task: TravelTask | None = None
        self.data_idx = 0
        self.phase_index = 0
        self.phase_step_count = 0
        self.phase_results: list[bool] = []
        self.paper_predictions: dict[int, list[dict[str, Any]]] = {}
        self.done = False
        self.status = "idle"

    def reset(self, data_idx: int) -> DomainTransition:
        self.data_idx, self.task = self._select_task(data_idx)
        self.phase_index = 0
        self.phase_step_count = 0
        self.phase_results = []
        self.paper_predictions = {}
        self.done = False
        self.status = "active"
        return self._transition(self._render_phase())

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.task is None:
            raise RuntimeError("Travel driver must be reset before step")
        if self.done:
            return self._transition("The travel episode is already complete.", done=True)
        self.phase_step_count += 1
        parsed = _parse_action(action)
        if parsed is None:
            transition = self._invalid(
                action,
                env_step,
                "invalid Travel action grammar",
            )
            return self._apply_phase_limit(transition, env_step)
        op, payload = parsed
        if op == "SUBMIT_PLAN":
            transition = self._submit_plan(payload, action, env_step)
            return self._apply_phase_limit(transition, env_step)
        transition = self._tool_step(op, payload, action, env_step)
        return self._apply_phase_limit(transition, env_step)

    def close(self) -> None:
        self.status = "closed"
        self.done = True

    def _tool_step(
        self,
        op: str,
        payload: dict[str, Any],
        raw_action: str,
        env_step: int,
    ) -> DomainTransition:
        try:
            result = str(self.tool_executor.execute(op, payload))
        except Exception as exc:
            return self._infra_error(op, env_step, exc)
        upstream_error = result.startswith("Error executing ") or result.startswith(
            "Error: Unknown tool"
        )
        component = {
            "name": "travel_tool_transition",
            "value": 0.0,
            "op": op,
            "step": env_step,
        }
        return self._transition(
            f"Tool result ({op}):\n{result}\n\n{self._render_phase()}",
            action_execution={
                "op": op,
                "status": "executed_with_error" if upstream_error else "executed",
                "step": env_step,
                "arguments": dict(payload),
            },
            tool_ops=(
                {
                    "op": op,
                    "step": env_step,
                    "arguments": dict(payload),
                    "result_length": len(result),
                    "upstream_error_result": upstream_error,
                },
            ),
            reward_components=(component,),
            domain_evidence={
                "task_id": self.task.task_id,
                "tool_result_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
            },
        )

    def _submit_plan(
        self,
        payload: dict[str, Any],
        raw_action: str,
        env_step: int,
    ) -> DomainTransition:
        if set(payload) != {"plan"}:
            return self._invalid(raw_action, env_step, "SUBMIT_PLAN expects only plan")
        plan = payload.get("plan")
        if not isinstance(plan, str) or not plan.strip():
            return self._invalid(raw_action, env_step, "plan must be a non-empty string")
        phase = self._current_phase()
        phase_before = self.phase_index
        final_phase = phase_before + 1 == len(self.task.phases)
        paper_evaluation = None
        next_paper_predictions = dict(self.paper_predictions)
        try:
            judgement = self.judge(plan, phase.name, phase.ground_truth_plans)
            passed = _validate_travel_judgement(judgement)
            if self.contract_mode == "paper_eval":
                paper_prediction = (
                    self.paper_evaluator.parse_prediction(plan)
                    if self.paper_evaluator is not None
                    else []
                )
                next_paper_predictions[phase.round_index] = paper_prediction
            if final_phase and self.paper_evaluator is not None:
                paper_evaluation = self.paper_evaluator.evaluate_group(
                    self.task.source_id,
                    next_paper_predictions,
                )
        except Exception as exc:
            return self._infra_error("SUBMIT_PLAN", env_step, exc)
        plan_sha256 = hashlib.sha256(plan.encode("utf-8")).hexdigest()
        completed_phase_step_count = self.phase_step_count
        self.phase_results.append(passed)

        if self.contract_mode == "paper_eval":
            self.paper_predictions = next_paper_predictions
            self.phase_index += 1
            self.phase_step_count = 0
            terminal = self.phase_index == len(self.task.phases)
            phase_advanced = True
            episode_success = terminal and all(self.phase_results)
            self.done = terminal
            self.status = (
                "success"
                if episode_success
                else "completed_with_errors"
                if terminal
                else "active"
            )
            observation = (
                "All traveler phases have been evaluated."
                if terminal
                else "The submitted plan was evaluated. The next traveler phase is ready.\n\n"
                + self._render_phase()
            )
        elif passed:
            self.phase_index += 1
            self.phase_step_count = 0
            terminal = self.phase_index == len(self.task.phases)
            phase_advanced = True
            episode_success = terminal
            self.done = terminal
            self.status = "success" if terminal else "active"
            observation = (
                "All traveler phases were completed successfully."
                if terminal
                else "The submitted plan was correct. The next traveler phase is ready.\n\n"
                + self._render_phase()
            )
        else:
            terminal = True
            phase_advanced = False
            episode_success = False
            self.done = True
            self.status = "failed_on_incorrect_plan"
            observation = (
                "The submitted plan was incorrect and the fail-fast travel episode ended."
            )

        tool_op = {
            "op": "SUBMIT_PLAN",
            "step": env_step,
            "committed": True,
            "submission_correct": passed,
            "phase_index": phase_before,
            "phase_advanced": phase_advanced,
            "terminal": terminal,
            "plan_sha256": plan_sha256,
        }
        component = {
            "name": "travel_plan_correct" if passed else "travel_plan_incorrect",
            "value": 1.0 if passed else 0.0,
            "op": "SUBMIT_PLAN",
            "step": env_step,
        }
        transition_event = {
            "type": (
                "travel_phase_completed"
                if phase_advanced
                else "travel_phase_failed"
            ),
            "phase_advanced": phase_advanced,
            "plan_sha256": plan_sha256,
        }
        if phase_advanced:
            transition_event.update(
                {
                    "completed_round_index": phase.round_index,
                    "completed_phase_step_count": completed_phase_step_count,
                    "phase_completion_reason": "submitted_plan",
                }
            )
        else:
            transition_event.update(
                {
                    "failed_round_index": phase.round_index,
                    "failed_phase_step_count": completed_phase_step_count,
                    "failure_reason": "incorrect_submitted_plan",
                }
            )
        return self._transition(
            observation,
            reward=1.0 if passed else 0.0,
            done=terminal,
            status=self.status,
            episode_success=episode_success,
            action_execution={
                "op": "SUBMIT_PLAN",
                "status": "committed_correct" if passed else "committed_incorrect",
                "step": env_step,
                "committed": True,
                "submission_correct": passed,
                "phase_advanced": phase_advanced,
                "terminal": terminal,
            },
            tool_ops=(tool_op,),
            reward_components=(component,),
            domain_evidence={
                "task_id": self.task.task_id,
                "judge_id": "memoryarena_travel_slot_similarity_v1",
                "phase_advanced": phase_advanced,
                "transition_event": transition_event,
                **(
                    {"paper_evaluation": paper_evaluation}
                    if paper_evaluation is not None
                    else {}
                ),
            },
        )

    def _apply_phase_limit(
        self,
        transition: DomainTransition,
        env_step: int,
    ) -> DomainTransition:
        if transition.done or transition.sample_excluded:
            return transition
        if self.phase_step_count < TRAVEL_MAX_STEPS_PER_PHASE:
            return transition

        task = self._require_task()
        completed_phase_index = self.phase_index
        completed_round_index = self._current_phase().round_index
        completed_phase_step_count = self.phase_step_count
        paper_evaluation = None
        if self.contract_mode == "paper_eval":
            next_paper_predictions = dict(self.paper_predictions)
            next_paper_predictions[completed_round_index] = []
            final_phase = self.phase_index + 1 == len(task.phases)
            try:
                paper_evaluation = (
                    self.paper_evaluator.evaluate_group(
                        task.source_id,
                        next_paper_predictions,
                    )
                    if final_phase and self.paper_evaluator is not None
                    else None
                )
            except Exception as exc:
                return self._infra_error("PAPER_EVALUATION", env_step, exc)
            self.paper_predictions = next_paper_predictions
            self.phase_index += 1
            self.phase_step_count = 0
            terminal = self.phase_index == len(task.phases)
            phase_advanced = True
            self.done = terminal
            self.status = "completed_with_errors" if terminal else "active"
        else:
            terminal = True
            phase_advanced = False
            self.done = True
            self.status = "failed_on_phase_timeout"
        self.phase_results.append(False)
        timeout_component = {
            "name": "travel_phase_step_limit",
            "value": 0.0,
            "op": "PHASE_TIMEOUT",
            "step": env_step,
        }
        observation = (
            "The current traveler exhausted the AMG one-action variant's "
            "30-action planning budget."
        )
        if self.contract_mode == "paper_eval":
            observation += " An empty plan was recorded as incorrect."
            if terminal:
                observation += " All traveler phases have been evaluated."
            else:
                observation += " The next traveler phase is ready.\n\n" + self._render_phase()
        else:
            observation += " The fail-fast travel episode ended without advancing."
        execution = dict(transition.action_execution)
        execution.update(
            {
                "phase_timeout": True,
                "phase_index": completed_phase_index,
                "phase_advanced": phase_advanced,
                "terminal": terminal,
            }
        )
        return self._transition(
            observation,
            reward=0.0,
            done=terminal,
            status=self.status,
            episode_success=False,
            action_execution=execution,
            tool_ops=(
                *transition.tool_ops,
                {
                    "op": "PHASE_TIMEOUT",
                    "step": env_step,
                    "phase_index": completed_phase_index,
                    "phase_advanced": phase_advanced,
                    "terminal": terminal,
                },
            ),
            reward_components=(*transition.reward_components, timeout_component),
            domain_evidence={
                **{
                    key: value
                    for key, value in transition.domain_evidence.items()
                    if key not in {"transition_event", "paper_evaluation"}
                },
                "phase_advanced": phase_advanced,
                "transition_event": _travel_timeout_event(
                    round_index=completed_round_index,
                    phase_step_count=completed_phase_step_count,
                    phase_advanced=phase_advanced,
                ),
                **(
                    {"paper_evaluation": paper_evaluation}
                    if paper_evaluation is not None
                    else {}
                ),
            },
        )

    def _select_task(self, dataset_position: int) -> tuple[int, TravelTask]:
        if isinstance(dataset_position, bool) or not isinstance(dataset_position, int):
            raise TypeError("Travel dataset position must be an integer")
        if not 0 <= dataset_position < len(self.tasks):
            raise IndexError(
                "Travel dataset position out of range: "
                f"{dataset_position} not in [0, {len(self.tasks)})"
            )
        return dataset_position, self.tasks[dataset_position]

    def _invalid(self, raw_action: str, env_step: int, message: str) -> DomainTransition:
        component = {
            "name": "invalid_action",
            "value": 0.0,
            "op": "INVALID",
            "step": env_step,
        }
        return self._transition(
            f"Invalid action: {message}\n\n{self._render_phase()}",
            reward=0.0,
            action_execution={
                "op": "INVALID",
                "status": "invalid",
                "step": env_step,
                "attempted_action_sha256": hashlib.sha256(
                    raw_action.encode("utf-8")
                ).hexdigest(),
            },
            reward_components=(component,),
            domain_evidence={"task_id": self.task.task_id},
        )

    def _infra_error(self, op: str, env_step: int, exc: Exception) -> DomainTransition:
        self.done = True
        self.status = "infra_error"
        component = {
            "name": "infrastructure_error_excluded",
            "value": 0.0,
            "op": op,
            "step": env_step,
            "error_type": type(exc).__name__,
        }
        return self._transition(
            "The travel environment encountered an infrastructure error.",
            done=True,
            status=self.status,
            action_execution={
                "op": op,
                "status": "error",
                "step": env_step,
            },
            tool_ops=(
                {
                    "op": "INFRA_ERROR",
                    "attempted_op": op,
                    "step": env_step,
                    "sample_excluded": True,
                    "error_type": type(exc).__name__,
                },
            ),
            reward_components=(component,),
            domain_evidence={"task_id": self.task.task_id},
            sample_excluded=True,
        )

    def _render_phase(self) -> str:
        task = self._require_task()
        phase = self._current_phase()
        sections = [
            (
                "Task family: travel_planner\n"
                f"Progress: {self.phase_index}/{len(task.phases)}"
            )
        ]
        if self.phase_index == 0:
            base = task.base_person
            sections.append(
                "Fixed base traveler (visible in phase 1 only):\n"
                f"Name: {base['name']}\n"
                f"Query: {base['query']}\n"
                f"Plan:\n{_format_plan(base['name'], base['daily_plans'])}"
            )
        sections.append(
            f"Current traveler: {phase.name}\n"
            f"Query: {phase.query}"
        )
        return "\n\n".join(sections)

    def _current_phase(self) -> TravelPhase:
        task = self._require_task()
        if self.phase_index >= len(task.phases):
            return task.phases[-1]
        return task.phases[self.phase_index]

    def _require_task(self) -> TravelTask:
        if self.task is None:
            raise RuntimeError("Travel driver must be reset before use")
        return self.task

    def _transition(
        self,
        observation: str,
        *,
        reward: float = 0.0,
        done: bool | None = None,
        status: str | None = None,
        episode_success: bool = False,
        action_execution=None,
        tool_ops=(),
        reward_components=(),
        domain_evidence=None,
        sample_excluded: bool = False,
    ) -> DomainTransition:
        task = self._require_task()
        phase = self._current_phase()
        active_round_index = None if self.done else phase.round_index
        evidence = {
            "task_id": task.task_id,
            "contract_mode": self.contract_mode,
            "dataset_position": self.data_idx,
            "source_id": task.source_id,
            "round_index": active_round_index,
            "active_round_index": active_round_index,
            "phase_step_count": self.phase_step_count,
            "max_actions_per_phase": TRAVEL_MAX_STEPS_PER_PHASE,
            **(domain_evidence or {}),
        }
        return DomainTransition(
            observation=observation,
            reward=reward,
            done=self.done if done is None else done,
            status=self.status if status is None else status,
            phase_index=self.phase_index,
            phase_count=len(task.phases),
            episode_success=episode_success,
            action_execution=action_execution or {},
            tool_ops=tool_ops,
            reward_components=reward_components,
            domain_evidence=evidence,
            sample_excluded=sample_excluded,
        )


def load_travel_tasks(path: Path) -> tuple[TravelTask, ...]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank Travel JSONL row at line {line_number}")
            payload = json.loads(line)
            tasks.append(_parse_task(payload, line_number))
    if not tasks:
        raise ValueError("Travel task file is empty")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Travel task ids must be unique")
    return tuple(tasks)


def _parse_task(payload: dict[str, Any], line_number: int) -> TravelTask:
    if not isinstance(payload, dict):
        raise ValueError(f"Travel row {line_number} must be an object")
    base = payload.get("base_person")
    questions = payload.get("questions")
    answers = payload.get("answers")
    if not isinstance(base, dict) or not isinstance(questions, list) or not isinstance(answers, list):
        raise ValueError(f"Travel row {line_number} has invalid base/questions/answers")
    if not questions or len(questions) != len(answers):
        raise ValueError(f"Travel row {line_number} has misaligned questions and answers")
    for field in ("name", "query", "daily_plans"):
        if field not in base:
            raise ValueError(f"Travel row {line_number} base person is missing {field}")
    answers_by_round: dict[int, Any] = {}
    for index, answer in enumerate(answers, start=1):
        answer_round = (
            int(answer.get("round_idx", index))
            if isinstance(answer, dict)
            else index
        )
        if answer_round in answers_by_round:
            raise ValueError(
                f"Travel row {line_number} has duplicate answer round {answer_round}"
            )
        answers_by_round[answer_round] = answer

    phases = []
    question_rounds = set()
    for index, question in enumerate(questions, start=1):
        query, name, round_index = _normalize_question(question, index)
        if round_index in question_rounds:
            raise ValueError(
                f"Travel row {line_number} has duplicate question round {round_index}"
            )
        question_rounds.add(round_index)
        try:
            answer = answers_by_round[round_index]
        except KeyError as exc:
            raise ValueError(
                f"Travel row {line_number} has no answer for round {round_index}"
            ) from exc
        plans, answer_name, answer_round = _normalize_answer(answer, name, round_index)
        if answer_name and answer_name != name:
            raise ValueError(f"Travel row {line_number} phase {index} name mismatch")
        if answer_round is not None and answer_round != round_index:
            raise ValueError(f"Travel row {line_number} phase {index} round mismatch")
        phases.append(
            TravelPhase(
                round_index=round_index,
                name=name,
                query=query,
                ground_truth_plans=tuple(dict(plan) for plan in plans),
            )
        )
    extra_answer_rounds = set(answers_by_round).difference(question_rounds)
    if extra_answer_rounds:
        raise ValueError(
            f"Travel row {line_number} has answers without questions: "
            f"{sorted(extra_answer_rounds)}"
        )
    source_id = payload.get("id", line_number)
    return TravelTask(
        task_id=str(source_id),
        source_id=source_id,
        base_person={
            "name": str(base["name"]),
            "query": str(base["query"]),
            "daily_plans": [dict(plan) for plan in base["daily_plans"]],
        },
        phases=tuple(phases),
    )


def _normalize_question(question: Any, index: int) -> tuple[str, str, int]:
    if isinstance(question, dict):
        query = question.get("query")
        name = question.get("name")
        round_index = int(question.get("round_idx", index))
    else:
        query = question
        name = None
        round_index = index
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Travel question {index} must contain text")
    if not name:
        match = _NAME_RE.search(query)
        name = match.group(1).strip() if match else "Person"
    return query.strip(), str(name).strip(), round_index


def _normalize_answer(answer: Any, default_name: str, default_round: int):
    if isinstance(answer, dict):
        plans = answer.get("daily_plans")
        name = str(answer.get("name", default_name))
        round_index = int(answer.get("round_idx", default_round))
    else:
        plans = answer
        name = default_name
        round_index = default_round
    if not isinstance(plans, list) or not plans or any(not isinstance(item, dict) for item in plans):
        raise ValueError("Travel answer must contain a non-empty daily plan list")
    return plans, name, round_index


def _parse_action(action: str) -> tuple[str, dict[str, Any]] | None:
    match = _ACTION_RE.fullmatch(action.strip())
    if match is None:
        return None
    try:
        payload = json.loads(match.group(2))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return match.group(1), payload


def _validate_travel_judgement(value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError("Travel judge must return a boolean")
    return value


def _travel_timeout_event(
    *,
    round_index: int,
    phase_step_count: int,
    phase_advanced: bool,
) -> dict[str, Any]:
    event = {
        "type": (
            "travel_phase_completed" if phase_advanced else "travel_phase_failed"
        ),
        "phase_advanced": phase_advanced,
        "plan_sha256": None,
    }
    if phase_advanced:
        event.update(
            {
                "completed_round_index": round_index,
                "completed_phase_step_count": phase_step_count,
                "phase_completion_reason": "one_action_variant_limit",
            }
        )
    else:
        event.update(
            {
                "failed_round_index": round_index,
                "failed_phase_step_count": phase_step_count,
                "failure_reason": "one_action_variant_limit",
            }
        )
    return event


def _format_plan(name: str, plans: Sequence[dict[str, Any]]) -> str:
    lines = [f"=== {name}'s Plan ==="]
    slots = (
        ("Current City", "current_city"),
        ("Transportation", "transportation"),
        ("Breakfast", "breakfast"),
        ("Attraction", "attraction"),
        ("Lunch", "lunch"),
        ("Dinner", "dinner"),
        ("Accommodation", "accommodation"),
    )
    for plan in plans:
        day = plan.get("days", plan.get("day"))
        lines.append(f"Day {day}:")
        lines.extend(f"{label}: {plan.get(key, '-')}" for label, key in slots)
        lines.append("")
    return "\n".join(lines)


def _load_upstream(memoryarena_root: Path):
    if not memoryarena_root.exists():
        raise RuntimeError(f"MemoryArena root does not exist: {memoryarena_root}")
    root_text = str(memoryarena_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    travel_module = importlib.import_module("env.env_systems.travel_env")
    executor_module = importlib.import_module(
        "env.env_systems.travel_planner_env.tool_executor"
    )
    paper_eval_module = importlib.import_module(
        "env.env_systems.travel_planner_env.eval"
    )
    paper_parser_module = importlib.import_module(
        "env.env_systems.travel_planner_env.combination"
    )
    _require_module_under_root(travel_module, memoryarena_root)
    _require_module_under_root(executor_module, memoryarena_root)
    _require_module_under_root(paper_eval_module, memoryarena_root)
    _require_module_under_root(paper_parser_module, memoryarena_root)
    return (
        travel_module,
        executor_module.ToolExecutor,
        paper_eval_module,
        paper_parser_module,
    )


def _build_upstream_judge(module) -> TravelJudge:
    def judge(plan_text: str, name: str, ground_truth_plans) -> bool:
        parsed = module._parse_person_plan_from_result(plan_text, name)
        return bool(
            module.TravelPlannerEnvironment._evaluate_slots(
                None,
                parsed,
                ground_truth_plans,
            )
        )

    return judge


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest_travel_database(
    database_path: str | Path,
    *,
    injected_test_assets: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every Travel tool reads its frozen canonical asset."""

    root = Path(database_path).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Travel database root is not a directory: {root}")
    specs = dict(injected_test_assets or TRAVEL_DATABASE_ASSET_SPECS)
    if set(specs) != set(TRAVEL_DATABASE_ASSET_SPECS):
        raise RuntimeError(
            "Travel database manifest must cover exactly six native asset classes"
        )
    files = {}
    for asset_class, (relative_path, expected_sha256) in specs.items():
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"Missing Travel database asset {asset_class}: {path}"
            )
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Travel database asset SHA256 mismatch for {asset_class}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        files[asset_class] = {
            "relative_path": relative_path,
            "sha256": observed_sha256,
            "size_bytes": path.stat().st_size,
        }
    bundle_sha256 = hashlib.sha256(
        json.dumps(
            files,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "mode": (
            "injected_test_database_manifest"
            if injected_test_assets is not None
            else "frozen_memoryarena_travel_database_manifest"
        ),
        "manifest_schema": "memoryarena_travel_database_assets_v1",
        "asset_count": len(files),
        "assets": files,
        "manifest_sha256": bundle_sha256,
    }


def attest_travel_upstream(
    memoryarena_root: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Fail closed if the Travel implementation differs from its git commit."""

    root = memoryarena_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"MemoryArena root is not a git worktree: {root}")
    commit = _git(root, "rev-parse", "HEAD").strip()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "MemoryArena commit mismatch for Travel: "
            f"expected {expected_commit}, observed {commit}"
        )
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *TRAVEL_UPSTREAM_RELATIVE_PATHS,
    )
    if status.strip():
        raise RuntimeError(
            "MemoryArena Travel source is not pristine at the pinned commit:\n"
            + status.rstrip()
        )
    source_sha256 = {}
    for relative_path in TRAVEL_UPSTREAM_RELATIVE_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing MemoryArena Travel source file: {path}")
        source_sha256[relative_path] = _sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(
            source_sha256,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "mode": "pinned_pristine_upstream",
        "memoryarena_commit": commit,
        "source_files_sha256": source_sha256,
        "source_bundle_sha256": digest,
    }


def _git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        *args,
    ]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"Cannot attest MemoryArena Travel source at {root}: {stderr.strip()}"
        ) from exc


def _require_module_under_root(module: Any, root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"Imported module {module.__name__!r} has no source path")
    path = Path(module_file).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported {module.__name__} from the wrong MemoryArena root: {path}"
        ) from exc
