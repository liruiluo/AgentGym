from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..runtime.domain import DomainContract, DomainTransition


TRAVEL_SURFACE = "memoryarena_travel_planner_v3"
TRAVEL_DOMAIN_ID = "travel_planner"
TRAVEL_TOOL_OPS = (
    "FlightSearch",
    "RestaurantSearch",
    "AccommodationSearch",
    "AttractionSearch",
    "DistanceMatrix",
    "CitySearch",
)
TRAVEL_UPSTREAM_RELATIVE_PATHS = (
    "env/env_systems/travel_env.py",
    "env/env_systems/travel_planner_env/data_loader.py",
    "env/env_systems/travel_planner_env/eval.py",
    "env/env_systems/travel_planner_env/prompts.py",
    "env/env_systems/travel_planner_env/tool_executor.py",
    "env/env_systems/travel_planner_env/tool_schemas.py",
)
_ACTION_RE = re.compile(
    r"\A(" + "|".join((*TRAVEL_TOOL_OPS, "SUBMIT_PLAN")) + r")\s+(\{.*\})\Z",
    flags=re.DOTALL,
)
_NAME_RE = re.compile(r"\bI am\s+([^\.\n]+)", flags=re.IGNORECASE)


TRAVEL_CONTRACT = DomainContract(
    contract_id="memoryarena_travel_planner_v3_20260721",
    system_prompt=(
        "You are operating the MemoryArena Travel Planner domain. Each episode "
        "contains a fixed base traveler followed by sequential traveler phases. "
        "Use one native JSON action at a time to inspect the frozen local travel "
        "database. Submit the current traveler's complete plan with SUBMIT_PLAN. "
        "The server privately evaluates the plan and never exposes answer labels. "
        "Reply with brief reasoning followed by exactly one Action line."
    ),
    native_action_descriptions=(
        'FlightSearch {"origin": "...", "destination": "...", "date": "YYYY-MM-DD"}',
        'RestaurantSearch {"city": "..."}',
        'AccommodationSearch {"city": "..."}',
        'AttractionSearch {"city": "..."}',
        'DistanceMatrix {"origin": "...", "destination": "...", "mode": "driving|taxi"}',
        'CitySearch {"state": "..."}',
        'SUBMIT_PLAN {"plan": "=== Name\'s Plan ===\\nDay 1: ..."}',
    ),
    max_steps=30,
)


@dataclass(frozen=True)
class TravelPhase:
    round_index: int
    name: str
    query: str
    ground_truth_plans: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TravelTask:
    task_id: str
    base_person: dict[str, Any]
    phases: tuple[TravelPhase, ...]


TravelJudge = Callable[[str, str, Sequence[dict[str, Any]]], bool]


class TravelPlannerFactory:
    domain_id = TRAVEL_DOMAIN_ID
    surface = TRAVEL_SURFACE
    contract = TRAVEL_CONTRACT

    def __init__(
        self,
        *,
        tasks_path: str | Path,
        memoryarena_root: str | Path | None = None,
        database_path: str | Path | None = None,
        tool_executor: Any | None = None,
        judge: TravelJudge | None = None,
        expected_memoryarena_commit: str | None = None,
    ) -> None:
        self.tasks_path = Path(tasks_path).expanduser().resolve()
        self.tasks = load_travel_tasks(self.tasks_path)
        self.task_count = len(self.tasks)
        self.dataset_sha256 = _sha256_file(self.tasks_path)
        self.memoryarena_root = (
            Path(memoryarena_root).expanduser().resolve()
            if memoryarena_root is not None
            else None
        )
        self.upstream_provenance = (
            attest_travel_upstream(
                self.memoryarena_root,
                expected_commit=expected_memoryarena_commit,
            )
            if self.memoryarena_root is not None
            else {"mode": "injected_test_double"}
        )
        if tool_executor is None or judge is None:
            if self.memoryarena_root is None:
                raise RuntimeError(
                    "memoryarena_root is required when Travel tools or judge are not injected"
                )
            upstream_module, executor_type = _load_upstream(self.memoryarena_root)
            if tool_executor is None:
                db_path = (
                    Path(database_path).expanduser().resolve()
                    if database_path is not None
                    else self.memoryarena_root
                    / "env/env_systems/travel_planner_env/database"
                )
                tool_executor = executor_type(db_path=str(db_path))
            if judge is None:
                judge = _build_upstream_judge(upstream_module)
        self.tool_executor = tool_executor
        self.judge = judge

    def create(self, env_uid: str):
        return TravelPlannerDriver(
            tasks=self.tasks,
            tool_executor=self.tool_executor,
            judge=self.judge,
            env_uid=env_uid,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "MemoryArena",
            "dataset_sha256": self.dataset_sha256,
            "native_tool_ops": list(TRAVEL_TOOL_OPS),
            "judge": "memoryarena_travel_slot_similarity_v1",
            "wrong_submission_semantics": "continue_to_next_traveler",
            "upstream_provenance": self.upstream_provenance,
        }


class TravelPlannerDriver:
    domain_id = TRAVEL_DOMAIN_ID
    surface = TRAVEL_SURFACE
    contract = TRAVEL_CONTRACT

    def __init__(
        self,
        *,
        tasks: Sequence[TravelTask],
        tool_executor: Any,
        judge: TravelJudge,
        env_uid: str,
    ) -> None:
        if not tasks:
            raise ValueError("TravelPlannerDriver requires tasks")
        self.tasks = tuple(tasks)
        self.tool_executor = tool_executor
        self.judge = judge
        self.env_uid = env_uid
        self.task: TravelTask | None = None
        self.data_idx = 0
        self.phase_index = 0
        self.phase_results: list[bool] = []
        self.done = False
        self.status = "idle"

    def reset(self, data_idx: int) -> DomainTransition:
        self.data_idx = int(data_idx) % len(self.tasks)
        self.task = self.tasks[self.data_idx]
        self.phase_index = 0
        self.phase_results = []
        self.done = False
        self.status = "active"
        return self._transition(self._render_phase())

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.task is None:
            raise RuntimeError("Travel driver must be reset before step")
        if self.done:
            return self._transition("The travel episode is already complete.", done=True)
        parsed = _parse_action(action)
        if parsed is None:
            return self._invalid(action, env_step, "invalid Travel action grammar")
        op, payload = parsed
        if op == "SUBMIT_PLAN":
            return self._submit_plan(payload, action, env_step)
        return self._tool_step(op, payload, action, env_step)

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
        if result.startswith("Error executing ") or result.startswith("Error: Unknown tool"):
            return self._invalid(raw_action, env_step, result)
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
                "status": "executed",
                "step": env_step,
                "arguments": dict(payload),
            },
            tool_ops=(
                {
                    "op": op,
                    "step": env_step,
                    "arguments": dict(payload),
                    "result_length": len(result),
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
        try:
            passed = bool(self.judge(plan, phase.name, phase.ground_truth_plans))
        except Exception as exc:
            return self._infra_error("SUBMIT_PLAN", env_step, exc)
        tool_op = {
            "op": "SUBMIT_PLAN",
            "step": env_step,
            "committed": True,
            "submission_correct": passed,
            "phase_index": self.phase_index,
            "terminal": self.phase_index + 1 == len(self.task.phases),
            "plan_sha256": hashlib.sha256(plan.encode("utf-8")).hexdigest(),
        }
        self.phase_results.append(passed)
        self.phase_index += 1
        final = self.phase_index == len(self.task.phases)
        self.done = final
        episode_success = final and all(self.phase_results)
        self.status = (
            "success"
            if episode_success
            else "completed_with_errors"
            if final
            else "active"
        )
        tool_op.update(
            {
                "phase_advanced": True,
                "terminal": final,
            }
        )
        component = {
            "name": "travel_plan_correct" if passed else "travel_plan_incorrect",
            "value": 1.0 if passed else 0.0,
            "op": "SUBMIT_PLAN",
            "step": env_step,
        }
        observation = (
            "All traveler phases have been evaluated."
            if final
            else "The submitted plan was evaluated. The next traveler phase is ready.\n\n"
            + self._render_phase()
        )
        return self._transition(
            observation,
            reward=1.0 if passed else 0.0,
            done=final,
            status=self.status,
            episode_success=episode_success,
            action_execution={
                "op": "SUBMIT_PLAN",
                "status": "committed_correct" if passed else "committed_incorrect",
                "step": env_step,
            },
            tool_ops=(tool_op,),
            reward_components=(component,),
            domain_evidence={
                "task_id": self.task.task_id,
                "judge_id": "memoryarena_travel_slot_similarity_v1",
            },
        )

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
        sections.append(
            "Native Travel actions:\n"
            + "\n".join(f"- {item}" for item in self.contract.native_action_descriptions)
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
            domain_evidence=domain_evidence or {"task_id": task.task_id},
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
    phases = []
    for index, (question, answer) in enumerate(zip(questions, answers), start=1):
        query, name, round_index = _normalize_question(question, index)
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
    return TravelTask(
        task_id=str(payload.get("id", line_number)),
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
        name = match.group(1).strip() if match else f"Traveler{index}"
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
    _require_module_under_root(travel_module, memoryarena_root)
    _require_module_under_root(executor_module, memoryarena_root)
    return travel_module, executor_module.ToolExecutor


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
