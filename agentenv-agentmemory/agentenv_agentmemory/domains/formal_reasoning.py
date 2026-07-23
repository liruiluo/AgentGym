from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, Union

from ..runtime.domain import DomainContract, DomainTransition
from .memoryarena_dataset import (
    MemoryArenaDatasetProvenance,
    verify_memoryarena_dataset_provenance,
)


FORMAL_REASONING_DOMAINS = ("math", "phys")
FORMAL_REASONING_SURFACES = {
    "math": "memoryarena_formal_reasoning_math_failfast_v3",
    "phys": "memoryarena_formal_reasoning_phys_failfast_v3",
}
FORMAL_REASONING_DOMAIN_IDS = {
    "math": "formal_reasoning_math",
    "phys": "formal_reasoning_phys",
}
FROZEN_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
FORMAL_REASONING_RUNTIME_IMPORT_RELATIVE_PATHS = (
    "env/__init__.py",
    "env/env_client.py",
    "env/env_systems/__init__.py",
    "env/env_systems/base_env.py",
    "env/env_systems/webshop_env.py",
    "env/env_systems/browsecomp_plus_env.py",
    "env/env_systems/math_env.py",
    "env/env_systems/travel_env.py",
    "env/env_systems/formal_reasoning_env/llm_backend.py",
)
FORMAL_REASONING_REFERENCE_RELATIVE_PATHS = (
    "agent/math.py",
    "env/env_systems/formal_reasoning_env/eval.py",
    "run_math.py",
)
FORMAL_REASONING_UPSTREAM_RELATIVE_PATHS = (
    *FORMAL_REASONING_RUNTIME_IMPORT_RELATIVE_PATHS,
    *FORMAL_REASONING_REFERENCE_RELATIVE_PATHS,
)
FORMAL_REASONING_PRISTINE_GIT_SCOPES = ("env", "agent/math.py", "run_math.py")
_FORMAL_JUDGE_PROMPT_TEMPLATE = {
    "query_mode": "none_matching_run_math.py",
    "system": (
        "You are a helpful assistant that judges the equivalence of two "
        "mathematical expressions."
    ),
    "user": (
        "\n"
        "            You are a math expert. \n"
        "            Determine if these two expressions are mathematically "
        "equivalent answer for the given question:\n"
        "            Question: {query}\n"
        "            Expression 1: {action}\n"
        "            Expression 2: {ground_truth}\n"
        "\n"
        '            Respond only with "yes" or "no". '
    ),
}
FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    json.dumps(
        _FORMAL_JUDGE_PROMPT_TEMPLATE,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _contract(domain: str) -> DomainContract:
    label = "mathematics" if domain == "math" else "physics"
    return DomainContract(
        contract_id=f"memoryarena_formal_reasoning_{domain}_failfast_v3_20260721",
        system_prompt=(
            f"You are operating the MemoryArena formal-reasoning {label} domain. "
            "An episode contains sequential questions from one paper. The current "
            "question and its published background are visible; answer text is "
            "privately evaluated by the original MemoryArena judge. A correct answer "
            "advances to the next question and earns +1; an incorrect answer ends the "
            "episode immediately. Submit one final answer for the current question."
        ),
        native_action_descriptions=("<final answer text>",),
        max_steps=64,
    )


FORMAL_REASONING_CONTRACTS = {
    domain: _contract(domain) for domain in FORMAL_REASONING_DOMAINS
}


@dataclass(frozen=True)
class FormalReasoningPhase:
    question: str
    answer: str
    background: str


@dataclass(frozen=True)
class FormalReasoningTask:
    task_id: str
    paper_name: str
    phases: tuple[FormalReasoningPhase, ...]


FormalReasoningJudge = Callable[[str, str], Union[bool, tuple[bool, str]]]


class FormalReasoningFactory:
    def __init__(
        self,
        *,
        domain: str,
        tasks_path: str | Path,
        dataset_provenance: MemoryArenaDatasetProvenance,
        memoryarena_root: str | Path | None = None,
        judge: FormalReasoningJudge | None = None,
        judge_config: dict[str, Any] | None = None,
        expected_memoryarena_commit: str | None = None,
    ) -> None:
        if domain not in FORMAL_REASONING_DOMAINS:
            raise ValueError(
                "formal-reasoning domain must be one of: "
                + ", ".join(FORMAL_REASONING_DOMAINS)
            )
        self.domain = domain
        self.domain_id = FORMAL_REASONING_DOMAIN_IDS[domain]
        self.surface = FORMAL_REASONING_SURFACES[domain]
        self.contract = FORMAL_REASONING_CONTRACTS[domain]
        self.tasks_path = Path(tasks_path).expanduser().resolve()
        if (
            judge is not None
            and isinstance(dataset_provenance, MemoryArenaDatasetProvenance)
            and dataset_provenance.mode != "injected_test_fixture"
        ):
            raise RuntimeError(
                "an injected formal judge requires explicit injected-test "
                "dataset provenance"
            )
        verify_memoryarena_dataset_provenance(
            self.tasks_path,
            expected_config=self.domain_id,
            provenance=dataset_provenance,
        )
        self.dataset_provenance = dataset_provenance
        self.tasks = load_formal_reasoning_tasks(self.tasks_path)
        self.task_count = len(self.tasks)
        self.dataset_sha256 = _sha256_file(self.tasks_path)
        self.phase_count = sum(len(task.phases) for task in self.tasks)
        if (
            self.dataset_sha256 != dataset_provenance.sha256
            or self.task_count != dataset_provenance.record_count
            or self.phase_count != dataset_provenance.phase_count
        ):
            raise RuntimeError(
                "Loaded formal-reasoning tasks differ from their dataset provenance"
            )
        self.memoryarena_root = (
            Path(memoryarena_root).expanduser().resolve()
            if memoryarena_root is not None
            else None
        )
        if self.memoryarena_root is not None and expected_memoryarena_commit is None:
            raise RuntimeError(
                "expected_memoryarena_commit is required for an upstream formal judge"
            )
        self.upstream_provenance = (
            attest_formal_reasoning_upstream(
                self.memoryarena_root,
                expected_commit=expected_memoryarena_commit,
            )
            if self.memoryarena_root is not None
            else {"mode": "injected_test_double"}
        )
        if judge is None:
            if dataset_provenance.mode != "frozen_public_hf_dataset":
                raise RuntimeError(
                    "An upstream formal judge requires frozen public dataset provenance"
                )
            if self.memoryarena_root is None:
                raise RuntimeError(
                    "memoryarena_root is required when a formal-reasoning judge "
                    "is not injected"
                )
            normalized_judge_config = _normalize_upstream_judge_config(judge_config)
            judge = build_upstream_formal_reasoning_judge(
                self.memoryarena_root,
                config=normalized_judge_config,
            )
            self.judge_provenance = _judge_provenance(
                normalized_judge_config,
                mode="upstream_memoryarena_judge",
            )
        else:
            self.judge_provenance = _injected_judge_provenance()
        self.judge = judge

    def create(self, env_uid: str):
        return FormalReasoningDriver(
            domain=self.domain,
            tasks=self.tasks,
            judge=self.judge,
            env_uid=env_uid,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "MemoryArena",
            "dataset_config": self.domain_id,
            "dataset_sha256": self.dataset_sha256,
            "task_count": self.task_count,
            "phase_count": self.phase_count,
            "dataset_provenance": self.dataset_provenance.metadata(),
            "judge": "memoryarena_llm_math_equivalence_v1",
            "judge_provenance": self.judge_provenance,
            "semantic_variant": "ordered_subtask_failfast_v1",
            "phase_transition": "advance_on_correct; terminal_on_incorrect",
            "episode_success": "all_questions_correct",
            "upstream_provenance": self.upstream_provenance,
        }


class FormalReasoningDriver:
    def __init__(
        self,
        *,
        domain: str,
        tasks: Sequence[FormalReasoningTask],
        judge: FormalReasoningJudge,
        env_uid: str,
    ) -> None:
        if domain not in FORMAL_REASONING_DOMAINS:
            raise ValueError(f"unsupported formal-reasoning domain: {domain}")
        if not tasks:
            raise ValueError("FormalReasoningDriver requires tasks")
        self.domain = domain
        self.domain_id = FORMAL_REASONING_DOMAIN_IDS[domain]
        self.surface = FORMAL_REASONING_SURFACES[domain]
        self.contract = FORMAL_REASONING_CONTRACTS[domain]
        self.tasks = tuple(tasks)
        self.judge = judge
        self.env_uid = env_uid
        self.data_idx = 0
        self.task: FormalReasoningTask | None = None
        self.phase_index = 0
        self.phase_results: list[bool] = []
        self.done = False
        self.status = "idle"

    def reset(self, data_idx: int) -> DomainTransition:
        index = int(data_idx)
        if index < 0 or index >= len(self.tasks):
            raise IndexError(
                f"formal-reasoning data index {index} is outside "
                f"[0, {len(self.tasks)})"
            )
        self.data_idx = index
        self.task = self.tasks[index]
        self.phase_index = 0
        self.phase_results = []
        self.done = False
        self.status = "active"
        return self._transition(self._render_phase())

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.task is None:
            raise RuntimeError("formal-reasoning driver must be reset before step")
        if self.done:
            return self._transition(
                "The formal-reasoning episode is already complete.",
                done=True,
            )
        answer = action.strip()
        if not answer:
            return self._invalid(action, env_step, "answer must be non-empty")
        phase = self._current_phase()
        try:
            judged = self.judge(answer, phase.answer)
            if isinstance(judged, tuple):
                passed, judge_output = judged
            else:
                passed, judge_output = judged, None
            passed = bool(passed)
        except Exception as exc:
            return self._infra_error(answer, env_step, exc)

        answer_sha256 = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        self.phase_results.append(passed)
        if passed:
            self.phase_index += 1
        final = passed and self.phase_index == len(self.task.phases)
        self.done = final or not passed
        episode_success = final
        self.status = (
            "success"
            if episode_success
            else "failed_on_incorrect_answer"
            if not passed
            else "active"
        )
        component = {
            "name": (
                "formal_reasoning_answer_correct"
                if passed
                else "formal_reasoning_answer_incorrect"
            ),
            "value": 1.0 if passed else 0.0,
            "op": "ANSWER",
            "step": env_step,
        }
        tool_op = {
            "op": "ANSWER",
            "step": env_step,
            "committed": True,
            "submission_correct": passed,
            "phase_index": self.phase_index - 1 if passed else self.phase_index,
            "phase_advanced": passed,
            "terminal": self.done,
            "answer_sha256": answer_sha256,
        }
        evidence = {
            "task_id": self.task.task_id,
            "paper_name": self.task.paper_name,
            "judge_id": "memoryarena_llm_math_equivalence_v1",
            "answer_sha256": answer_sha256,
            "ground_truth_sha256": hashlib.sha256(
                phase.answer.encode("utf-8")
            ).hexdigest(),
            "correct_count": sum(self.phase_results),
            "phase_results": list(self.phase_results),
        }
        if judge_output is not None:
            evidence["judge_output_sha256"] = hashlib.sha256(
                str(judge_output).encode("utf-8")
            ).hexdigest()
        observation = (
            "All formal-reasoning questions have been evaluated."
            if final
            else "The submitted answer ended the formal-reasoning episode."
            if not passed
            else "The submitted answer was evaluated. The next question is ready.\n\n"
            + self._render_phase()
        )
        return self._transition(
            observation,
            reward=1.0 if passed else 0.0,
            done=self.done,
            status=self.status,
            episode_success=episode_success,
            action_execution={
                "op": "ANSWER",
                "status": "committed_correct" if passed else "committed_incorrect",
                "step": env_step,
            },
            tool_ops=(tool_op,),
            reward_components=(component,),
            domain_evidence=evidence,
        )

    def close(self) -> None:
        self.status = "closed"
        self.done = True

    def _render_phase(self) -> str:
        phase = self._current_phase()
        background = phase.background if phase.background.strip() else "No information provided."
        return (
            "### BACKGROUND:\n"
            f"{background}\n"
            "### PROBLEM:\n"
            f"{phase.question}"
        )

    def _current_phase(self) -> FormalReasoningPhase:
        task = self._require_task()
        if self.phase_index >= len(task.phases):
            return task.phases[-1]
        return task.phases[self.phase_index]

    def _require_task(self) -> FormalReasoningTask:
        if self.task is None:
            raise RuntimeError("formal-reasoning driver must be reset before use")
        return self.task

    def _infra_error(
        self,
        answer: str,
        env_step: int,
        exc: Exception,
    ) -> DomainTransition:
        self.done = True
        self.status = "infra_error"
        component = {
            "name": "infrastructure_error_excluded",
            "value": 0.0,
            "op": "ANSWER",
            "step": env_step,
            "error_type": type(exc).__name__,
        }
        return self._transition(
            "The formal-reasoning judge encountered an infrastructure error.",
            done=True,
            status=self.status,
            action_execution={
                "op": "ANSWER",
                "status": "error",
                "step": env_step,
            },
            tool_ops=(
                {
                    "op": "INFRA_ERROR",
                    "attempted_op": "ANSWER",
                    "step": env_step,
                    "sample_excluded": True,
                    "error_type": type(exc).__name__,
                    "answer_sha256": hashlib.sha256(
                        answer.encode("utf-8")
                    ).hexdigest(),
                },
            ),
            reward_components=(component,),
            domain_evidence={
                "task_id": self._require_task().task_id,
                "paper_name": self._require_task().paper_name,
            },
            sample_excluded=True,
        )

    def _invalid(
        self,
        raw_action: str,
        env_step: int,
        message: str,
    ) -> DomainTransition:
        return self._transition(
            f"Invalid action: {message}\n\n{self._render_phase()}",
            action_execution={
                "op": "INVALID",
                "status": "invalid",
                "step": env_step,
                "attempted_action_sha256": hashlib.sha256(
                    raw_action.encode("utf-8")
                ).hexdigest(),
            },
            reward_components=(
                {
                    "name": "invalid_action",
                    "value": 0.0,
                    "op": "INVALID",
                    "step": env_step,
                },
            ),
            domain_evidence={
                "task_id": self._require_task().task_id,
                "paper_name": self._require_task().paper_name,
                "invalid_reason": message,
            },
        )

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
            domain_evidence=domain_evidence
            or {"task_id": task.task_id, "paper_name": task.paper_name},
            sample_excluded=sample_excluded,
        )


def load_formal_reasoning_tasks(path: Path) -> tuple[FormalReasoningTask, ...]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"blank formal-reasoning JSONL row at line {line_number}"
                )
            payload = json.loads(line)
            tasks.append(_parse_task(payload, line_number))
    if not tasks:
        raise ValueError("formal-reasoning task file is empty")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("formal-reasoning task ids must be unique")
    return tuple(tasks)


def _parse_task(payload: Any, line_number: int) -> FormalReasoningTask:
    if not isinstance(payload, dict):
        raise ValueError(f"formal-reasoning row {line_number} must be an object")
    questions = payload.get("questions")
    answers = payload.get("answers")
    backgrounds = payload.get("backgrounds")
    if not all(isinstance(items, list) for items in (questions, answers, backgrounds)):
        raise ValueError(
            f"formal-reasoning row {line_number} has invalid questions/answers/backgrounds"
        )
    if not questions or not (len(questions) == len(answers) == len(backgrounds)):
        raise ValueError(
            f"formal-reasoning row {line_number} has misaligned phase arrays"
        )
    if any(
        not isinstance(value, str)
        for items in (questions, answers, backgrounds)
        for value in items
    ):
        raise ValueError(
            f"formal-reasoning row {line_number} phase values must be strings"
        )
    if any(not value.strip() for value in questions):
        raise ValueError(
            f"formal-reasoning row {line_number} questions must be non-empty"
        )
    if any(not value.strip() for value in answers):
        raise ValueError(
            f"formal-reasoning row {line_number} answers must be non-empty"
        )
    paper_name = payload.get("paper_name")
    if not isinstance(paper_name, str) or not paper_name.strip():
        raise ValueError(
            f"formal-reasoning row {line_number} requires a paper_name"
        )
    phases = tuple(
        FormalReasoningPhase(question=question, answer=answer, background=background)
        for question, answer, background in zip(questions, answers, backgrounds)
    )
    return FormalReasoningTask(
        task_id=str(payload.get("id", line_number - 1)),
        paper_name=paper_name,
        phases=phases,
    )


def build_upstream_formal_reasoning_judge(
    memoryarena_root: Path,
    *,
    config: dict[str, Any],
) -> FormalReasoningJudge:
    module = _load_upstream_math_module(memoryarena_root)
    environment = module.MathEnvironment(config=dict(config))

    def judge(answer: str, ground_truth: str):
        # run_math.py does not pass the query into MathEnvironment.step, so the
        # original judge receives query=None. Keep that behavior exactly.
        return environment.judge(answer, ground_truth)

    return judge


def _normalize_upstream_judge_config(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise RuntimeError("formal-reasoning judge_config must be an explicit mapping")
    required = ("backend", "model_name", "base_url", "temperature", "max_tokens")
    missing = [key for key in required if key not in config]
    if missing:
        raise RuntimeError(
            "formal-reasoning judge_config lacks explicit fields: " + ", ".join(missing)
        )
    backend = str(config["backend"]).strip().lower()
    model_name = str(config["model_name"]).strip()
    base_url = str(config["base_url"]).strip().rstrip("/")
    if not backend or not model_name or not base_url:
        raise RuntimeError("formal-reasoning judge configuration cannot contain blanks")
    try:
        temperature = float(config["temperature"])
        max_tokens = int(config["max_tokens"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("formal-reasoning judge numeric configuration is invalid") from exc
    if not math.isfinite(temperature):
        raise RuntimeError("formal-reasoning judge temperature must be finite")
    if isinstance(config["max_tokens"], bool) or max_tokens < 1:
        raise RuntimeError("formal-reasoning judge max_tokens must be positive")
    return {
        "backend": backend,
        "model_name": model_name,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _judge_provenance(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    endpoint_sha256 = hashlib.sha256(config["base_url"].encode("utf-8")).hexdigest()
    public_config = {
        "mode": mode,
        "backend": config["backend"],
        "model": config["model_name"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "endpoint_sha256": endpoint_sha256,
        "prompt_template_sha256": FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    public_config["config_sha256"] = hashlib.sha256(
        json.dumps(
            public_config,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return public_config


def _injected_judge_provenance() -> dict[str, Any]:
    payload = {
        "mode": "injected_test_double",
        "backend": "injected_test_double",
        "model": "injected_test_double",
        "temperature": None,
        "max_tokens": None,
        "endpoint_sha256": None,
        "prompt_template_sha256": FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    payload["config_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _load_upstream_math_module(memoryarena_root: Path):
    root = memoryarena_root.expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"MemoryArena root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("env.env_systems.math_env")
    for module_name in (
        "env",
        "env.env_systems",
        "env.env_systems.base_env",
        "env.env_systems.math_env",
        "env.env_systems.formal_reasoning_env.llm_backend",
    ):
        imported = sys.modules.get(module_name)
        if imported is None:
            raise RuntimeError(
                f"Expected MemoryArena runtime module was not imported: {module_name}"
            )
        _require_module_under_root(imported, root)
    return module


def attest_formal_reasoning_upstream(
    memoryarena_root: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Fail closed if the formal-reasoning implementation is not pristine."""

    root = memoryarena_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"MemoryArena root is not a git worktree: {root}")
    commit = _git(root, "rev-parse", "HEAD").strip()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "MemoryArena commit mismatch for formal reasoning: "
            f"expected {expected_commit}, observed {commit}"
        )
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        *FORMAL_REASONING_PRISTINE_GIT_SCOPES,
    )
    if status.strip():
        raise RuntimeError(
            "MemoryArena formal-reasoning source is not pristine at the pinned commit:\n"
            + status.rstrip()
        )
    tracked_env_files = set(
        _git(root, "ls-files", "--", "env").splitlines()
    )
    unexpected_python_files = sorted(
        str(path.relative_to(root))
        for path in (root / "env").rglob("*.py")
        if str(path.relative_to(root)) not in tracked_env_files
        and "__pycache__" not in path.parts
    )
    if unexpected_python_files:
        raise RuntimeError(
            "MemoryArena env contains untracked or ignored Python source that can "
            "alter imports:\n" + "\n".join(unexpected_python_files)
        )
    selected_source_sha256 = {}
    for relative_path in FORMAL_REASONING_UPSTREAM_RELATIVE_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"Missing MemoryArena formal-reasoning source file: {path}"
            )
        selected_source_sha256[relative_path] = _sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(
            selected_source_sha256,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "mode": "pinned_pristine_upstream_scopes",
        "memoryarena_commit": commit,
        "pristine_git_scopes": list(FORMAL_REASONING_PRISTINE_GIT_SCOPES),
        "env_git_tree_oid": _git(root, "rev-parse", f"{commit}:env").strip(),
        "runtime_import_entry_files_sha256": {
            path: selected_source_sha256[path]
            for path in FORMAL_REASONING_RUNTIME_IMPORT_RELATIVE_PATHS
        },
        "reference_entrypoint_files_sha256": {
            path: selected_source_sha256[path]
            for path in FORMAL_REASONING_REFERENCE_RELATIVE_PATHS
        },
        "selected_files_bundle_sha256": digest,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "Cannot attest MemoryArena formal-reasoning source at "
            f"{root}: {stderr.strip()}"
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
