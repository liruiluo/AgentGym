from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from ..runtime.domain import DomainContract, DomainTransition


SCIWORLD_DOMAIN_ID = "sciworld"
SCIWORLD_CONDUCTIVITY_SURFACE = "sciworld_conductivity_memory_v1"
SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE = "sciworld_lab_notebook_longhorizon_v1"
SCIWORLD_SURFACES = {
    "conductivity_memory": SCIWORLD_CONDUCTIVITY_SURFACE,
    "lab_notebook_longhorizon": SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
}
SCIWORLD_BACKENDS = ("fixture", "scienceworld")

_SCI_ACTION_RE = re.compile(r"\ASCI_ACTION\s+(\{.*\})\Z", flags=re.DOTALL)
_ANSWER_RE = re.compile(r"\AANSWER\s+(\{.*\})\Z", flags=re.DOTALL)


@dataclass(frozen=True)
class SciWorldSurfaceConfig:
    surface: str
    contract_id: str
    system_prompt: str
    max_steps: int
    fixture_phase_count: int


def _surface_configs() -> dict[str, SciWorldSurfaceConfig]:
    common = (
        "You are operating a SciWorld lab through AgentMemoryGym. Run one "
        "executable lab action at a time. Private target facts and future tasks "
        "are never exposed. The environment does not write lab notes, summarize "
        "history, or maintain a helpful rolling transcript for you. If a task "
        "spans multiple phases, use your own memory actions to keep any fact, "
        "procedure, or notebook entry you will need later. If your memory is "
        "insufficient, run another visible experiment instead of inventing a result."
    )
    return {
        SCIWORLD_CONDUCTIVITY_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_CONDUCTIVITY_SURFACE,
            contract_id="sciworld_conductivity_memory_v1_20260803",
            system_prompt=(
                common
                + " This surface focuses on remembering experimental conductivity "
                "results for unknown materials and using them in later phases."
            ),
            max_steps=64,
            fixture_phase_count=2,
        ),
        SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
            contract_id="sciworld_lab_notebook_longhorizon_v1_20260803",
            system_prompt=(
                common
                + " This long-horizon surface is intended to exceed raw-context "
                "comfort unless the policy maintains its own external lab notebook "
                "with ADD/UPDATE/RETRIEVE and policy-authored context control."
            ),
            max_steps=512,
            fixture_phase_count=12,
        ),
    }


SCIWORLD_SURFACE_CONFIGS = _surface_configs()


def contract_for_surface(surface: str) -> DomainContract:
    try:
        config = SCIWORLD_SURFACE_CONFIGS[surface]
    except KeyError as exc:
        raise ValueError(f"unsupported SciWorld surface: {surface!r}") from exc
    return DomainContract(
        contract_id=config.contract_id,
        system_prompt=config.system_prompt,
        native_action_descriptions=(
            'SCI_ACTION {"action": "<native SciWorld text command>"}',
            'ANSWER {"answer": "..."}',
        ),
        max_steps=config.max_steps,
    )


@dataclass(frozen=True)
class SciWorldFixtureTask:
    task_id: str
    unknown_material: str
    property_value: Literal["conductive", "nonconductive"]
    distractor_material: str


FIXTURE_TASKS = (
    SciWorldFixtureTask(
        task_id="conductivity_fixture_000",
        unknown_material="unknown sample alpha",
        property_value="conductive",
        distractor_material="rubber strip beta",
    ),
    SciWorldFixtureTask(
        task_id="conductivity_fixture_001",
        unknown_material="unknown sample gamma",
        property_value="nonconductive",
        distractor_material="copper wire delta",
    ),
)


class SciWorldMemoryFactory:
    domain_id = SCIWORLD_DOMAIN_ID

    def __init__(
        self,
        *,
        surface: str = SCIWORLD_CONDUCTIVITY_SURFACE,
        backend: str = "scienceworld",
        task_count: int | None = None,
    ) -> None:
        if surface not in SCIWORLD_SURFACE_CONFIGS:
            raise ValueError(f"unsupported SciWorld surface: {surface!r}")
        if backend not in SCIWORLD_BACKENDS:
            raise ValueError(
                "SciWorld backend must be one of: " + ", ".join(SCIWORLD_BACKENDS)
            )
        if backend == "scienceworld":
            _require_scienceworld_dependency()
        self.surface = surface
        self.backend = backend
        self.contract = contract_for_surface(surface)
        self._task_count = task_count or (
            len(FIXTURE_TASKS)
            if backend == "fixture"
            else _scienceworld_task_count_hint(surface)
        )

    @property
    def task_count(self) -> int:
        return self._task_count

    def create(self, env_uid: str):
        if self.backend == "fixture":
            return SciWorldFixtureDriver(
                env_uid=env_uid,
                surface=self.surface,
                contract=self.contract,
                phase_count=SCIWORLD_SURFACE_CONFIGS[self.surface].fixture_phase_count,
            )
        return ScienceWorldNativeDriver(
            env_uid=env_uid,
            surface=self.surface,
            contract=self.contract,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "allenai/ScienceWorld",
            "domain_family": "scientific_experiment_lab",
            "backend": self.backend,
            "memory_management": "policy_managed_external_notebook",
            "history_policy": "no_harness_recent_n_no_environment_summary",
            "harness_summarizes_history": False,
            "manual_recent_n_window": None,
            "requires_scienceworld_dependency": self.backend == "scienceworld",
            "surfaces": dict(SCIWORLD_SURFACES),
        }


class SciWorldFixtureDriver:
    domain_id = SCIWORLD_DOMAIN_ID

    def __init__(
        self,
        *,
        env_uid: str,
        surface: str,
        contract: DomainContract,
        phase_count: int,
    ) -> None:
        self.env_uid = env_uid
        self.surface = surface
        self.contract = contract
        self.phase_count = phase_count
        self.phase_index = 0
        self.task = FIXTURE_TASKS[0]
        self.closed = False
        self.tested = False

    def reset(self, data_idx: int) -> DomainTransition:
        self.task = FIXTURE_TASKS[int(data_idx) % len(FIXTURE_TASKS)]
        self.phase_index = 0
        self.closed = False
        self.tested = False
        return self._transition(self._source_observation(), status="active")

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.closed:
            raise RuntimeError("SciWorld fixture driver is closed")
        try:
            parsed = _parse_native_action(action)
        except ValueError as exc:
            return self._invalid(env_step, action, str(exc))
        if parsed is None:
            return self._invalid(env_step, action, "expected SCI_ACTION or ANSWER")
        op, payload = parsed
        if op == "SCI_ACTION":
            return self._step_sci_action(payload, env_step, action)
        return self._step_answer(payload, env_step, action)

    def close(self) -> None:
        self.closed = True

    def _step_sci_action(
        self,
        payload: dict[str, Any],
        env_step: int,
        raw_action: str,
    ) -> DomainTransition:
        command = _require_payload_text(payload, "action")
        lowered = command.lower()
        if self.phase_index == 0 and "conduct" in lowered:
            self.tested = True
            observation = (
                "You build a simple circuit with the unknown material. "
                f"Result: {self.task.unknown_material} is {self.task.property_value}. "
                "The lab result is visible now, but it will not be automatically "
                "shown after the phase changes."
            )
            return self._transition(
                observation,
                action_execution={
                    "op": "SCI_ACTION",
                    "status": "executed",
                    "step": env_step,
                    "command": command,
                },
                tool_ops=(
                    {
                        "op": "SCI_ACTION",
                        "step": env_step,
                        "command": command,
                    },
                ),
                domain_evidence={
                    "task_id": self.task.task_id,
                    "experiment_observed": True,
                },
            )
        observation = (
            "The lab action executes, but it does not resolve the current memory-"
            "dependent question."
        )
        return self._transition(
            observation,
            action_execution={
                "op": "SCI_ACTION",
                "status": "executed",
                "step": env_step,
                "command": command,
            },
            tool_ops=(
                {
                    "op": "SCI_ACTION",
                    "step": env_step,
                    "command": command,
                },
            ),
        )

    def _step_answer(
        self,
        payload: dict[str, Any],
        env_step: int,
        raw_action: str,
    ) -> DomainTransition:
        answer = _require_payload_text(payload, "answer").lower()
        if self.phase_index == 0:
            correct = self.task.property_value in answer
            if not correct:
                return self._terminal_failure(env_step, raw_action, answer)
            self.phase_index = 1
            self.tested = False
            return self._transition(
                self._dependent_observation(),
                reward=1.0,
                status="active",
                action_execution={
                    "op": "ANSWER",
                    "status": "correct",
                    "step": env_step,
                    "answer": answer,
                    "phase_advanced": True,
                },
                tool_ops=(
                    {
                        "op": "ANSWER",
                        "step": env_step,
                        "answer": answer,
                        "correct": True,
                    },
                ),
                reward_components=(
                    {
                        "name": "sciworld_phase_answer_correct",
                        "value": 1.0,
                        "op": "ANSWER",
                        "step": env_step,
                    },
                ),
                domain_evidence={
                    "task_id": self.task.task_id,
                    "hidden_property": self.task.property_value,
                    "phase_advanced": True,
                },
            )
        target = self.task.unknown_material if self.task.property_value == "conductive" else self.task.distractor_material
        correct = target.lower() in answer
        if not correct:
            return self._terminal_failure(env_step, raw_action, answer)
        self.phase_index = self.phase_count
        return self._transition(
            "The final circuit works. Episode complete.",
            reward=1.0,
            done=True,
            status="success",
            episode_success=True,
            action_execution={
                "op": "ANSWER",
                "status": "correct",
                "step": env_step,
                "answer": answer,
            },
            tool_ops=(
                {
                    "op": "ANSWER",
                    "step": env_step,
                    "answer": answer,
                    "correct": True,
                },
            ),
            reward_components=(
                {
                    "name": "sciworld_final_answer_correct",
                    "value": 1.0,
                    "op": "ANSWER",
                    "step": env_step,
                },
            ),
            domain_evidence={
                "task_id": self.task.task_id,
                "target_material": target,
                "memory_dependency": "prior_conductivity_result",
            },
        )

    def _invalid(
        self,
        env_step: int,
        raw_action: str,
        reason: str,
    ) -> DomainTransition:
        return self._transition(
            f"Invalid SciWorld action: {reason}.",
            reward=-0.01,
            action_execution={
                "op": "INVALID",
                "status": "invalid",
                "step": env_step,
                "submitted_action": raw_action,
                "error": reason,
            },
            reward_components=(
                {
                    "name": "invalid_action",
                    "value": -0.01,
                    "op": "INVALID",
                    "step": env_step,
                },
            ),
        )

    def _terminal_failure(
        self, env_step: int, raw_action: str, answer: str) -> DomainTransition:
        return self._transition(
            "Incorrect answer. Episode terminated without revealing the target.",
            reward=0.0,
            done=True,
            status="failed",
            episode_success=False,
            action_execution={
                "op": "ANSWER",
                "status": "incorrect",
                "step": env_step,
                "submitted_action": raw_action,
                "answer": answer,
            },
            reward_components=(
                {
                    "name": "sciworld_answer_incorrect",
                    "value": 0.0,
                    "op": "ANSWER",
                    "step": env_step,
                },
            ),
            domain_evidence={"task_id": self.task.task_id},
        )

    def _source_observation(self) -> str:
        return (
            "SciWorld conductivity source phase. You are in a lab with "
            f"{self.task.unknown_material}. Determine whether it conducts "
            "electricity. Native lab action example: "
            'SCI_ACTION {"action": "test conductivity of the unknown sample"}. '
            "When ready, submit ANSWER with the observed property."
        )

    def _dependent_observation(self) -> str:
        return (
            "SciWorld dependent phase. Build a working circuit using one material. "
            f"Candidate materials: {self.task.unknown_material}; "
            f"{self.task.distractor_material}. The prior lab result is not repeated "
            "in this observation. Use your own stored memory if you made one, or run "
            "a visible experiment before answering."
        )

    def _transition(
        self,
        observation: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        status: str = "active",
        episode_success: bool = False,
        action_execution=None,
        tool_ops=(),
        reward_components=(),
        domain_evidence=None,
    ) -> DomainTransition:
        evidence = {
            "task_id": self.task.task_id,
            "surface": self.surface,
            "backend": "fixture",
            "history_policy": "no_harness_recent_n_no_environment_summary",
        }
        if domain_evidence:
            evidence.update(domain_evidence)
        return DomainTransition(
            observation=(
                f"SciWorld task {self.task.task_id}. Phase "
                f"{min(self.phase_index, self.phase_count)}/{self.phase_count}. "
                + observation
            ),
            reward=reward,
            done=done,
            status=status,
            phase_index=min(self.phase_index, self.phase_count),
            phase_count=self.phase_count,
            episode_success=episode_success,
            action_execution=action_execution or {},
            tool_ops=tool_ops,
            reward_components=reward_components,
            domain_evidence=evidence,
        )


class ScienceWorldNativeDriver:
    domain_id = SCIWORLD_DOMAIN_ID

    def __init__(self, *, env_uid: str, surface: str, contract: DomainContract) -> None:
        self.env_uid = env_uid
        self.surface = surface
        self.contract = contract
        module = _require_scienceworld_dependency()
        self.env = module.ScienceWorldEnv()
        self.closed = False
        self.phase_index = 0
        self.task_name = ""
        self.variation_idx = 0

    def reset(self, data_idx: int) -> DomainTransition:
        tasks = list(getattr(self.env, "tasks", {}).values())
        if not tasks:
            raise RuntimeError("ScienceWorldEnv exposes no tasks")
        self.task_name = tasks[int(data_idx) % len(tasks)]
        self.variation_idx = 0
        self.env.load(self.task_name, self.variation_idx)
        observation, _reward, done, info = self.env.step("look around")
        return self._transition(
            observation,
            done=bool(done),
            status="success" if done else "active",
            domain_evidence={"scienceworld_info": dict(info)},
        )

    def step(self, action: str, env_step: int) -> DomainTransition:
        try:
            parsed = _parse_native_action(action)
        except ValueError as exc:
            parsed = None
            parse_error = str(exc)
        else:
            parse_error = "expected SCI_ACTION JSON"
        if parsed is None or parsed[0] != "SCI_ACTION":
            return DomainTransition(
                observation=f"Native ScienceWorld backend expects SCI_ACTION JSON: {parse_error}.",
                reward=-0.01,
                done=False,
                status="active",
                phase_index=self.phase_index,
                phase_count=None,
                episode_success=False,
                action_execution={
                    "op": "INVALID",
                    "status": "invalid",
                    "step": env_step,
                    "submitted_action": action,
                },
                reward_components=(
                    {
                        "name": "invalid_action",
                        "value": -0.01,
                        "op": "INVALID",
                        "step": env_step,
                    },
                ),
                domain_evidence={"backend": "scienceworld"},
            )
        command = _require_payload_text(parsed[1], "action")
        observation, reward, done, info = self.env.step(command)
        score = float(info.get("score", reward)) if isinstance(info, dict) else float(reward)
        if done:
            self.phase_index = 1
        return self._transition(
            observation,
            reward=score,
            done=bool(done),
            status="success" if done else "active",
            episode_success=bool(done and score > 0.0),
            action_execution={
                "op": "SCI_ACTION",
                "status": "executed",
                "step": env_step,
                "command": command,
            },
            tool_ops=(
                {
                    "op": "SCI_ACTION",
                    "step": env_step,
                    "command": command,
                },
            ),
            reward_components=(
                {
                    "name": "scienceworld_score_delta_or_score",
                    "value": score,
                    "op": "SCI_ACTION",
                    "step": env_step,
                },
            ),
            domain_evidence={"scienceworld_info": dict(info) if isinstance(info, dict) else {}},
        )

    def close(self) -> None:
        if not self.closed:
            self.env.close()
            self.closed = True

    def _transition(self, observation: str, **kwargs) -> DomainTransition:
        evidence = {
            "surface": self.surface,
            "backend": "scienceworld",
            "task_name": self.task_name,
            "variation_idx": self.variation_idx,
            "history_policy": "no_harness_recent_n_no_environment_summary",
        }
        evidence.update(kwargs.pop("domain_evidence", {}) or {})
        return DomainTransition(
            observation=observation,
            reward=float(kwargs.pop("reward", 0.0)),
            done=bool(kwargs.pop("done", False)),
            status=kwargs.pop("status", "active"),
            phase_index=self.phase_index,
            phase_count=1,
            episode_success=bool(kwargs.pop("episode_success", False)),
            action_execution=kwargs.pop("action_execution", {}),
            tool_ops=kwargs.pop("tool_ops", ()),
            reward_components=kwargs.pop("reward_components", ()),
            domain_evidence=evidence,
        )


def _parse_native_action(action: str) -> tuple[str, dict[str, Any]] | None:
    text = action.strip()
    for op, regex in (("SCI_ACTION", _SCI_ACTION_RE), ("ANSWER", _ANSWER_RE)):
        match = regex.fullmatch(text)
        if match is None:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{op} payload must be valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{op} payload must be a JSON object")
        return op, payload
    return None


def _require_payload_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_scienceworld_dependency():
    try:
        import scienceworld  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "SciWorld backend requires the optional 'scienceworld' package and "
            "Java runtime. Use AGENTMEMORY_SCIWORLD_BACKEND=fixture for static "
            "AgentMemoryGym contract tests, or install/verify SciWorld before a "
            "native smoke."
        ) from exc
    return scienceworld


def _scienceworld_task_count_hint(surface: str) -> int:
    # Real task enumeration is deferred to the backend process because importing
    # ScienceWorld can start JVM-heavy setup. The exact count is checked in the
    # native smoke; this hint only keeps the v3 factory contract positive.
    if surface == SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE:
        return 1
    return 30
