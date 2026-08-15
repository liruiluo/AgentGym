from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backend import BackendError, FixtureBackend, RequestError
from .contracts import EvaluationArm
from .dataset import GaiaTextDataset, GaiaTextTask
from .submission import SubmissionStore

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL
)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_WORKSPACE_PREFIXES = ("shell_command ", "apply_patch\n")
_PAIRED_RUNTIME_SCHEMA = "gaia_text_paired_runtime_contract_v1"
_DOMAIN_ACTION_CONTRACT = "shared_search_visit_answer_v1"
_ANSWER_EXTRACTION_CONTRACT = "single_trimmed_answer_tag_v1"
_REWARD_CONTRACT = "external_official_scoring_zero_online_reward_v1"
_SUBMISSION_CONTRACT = "gaia_task_id_model_answer_jsonl_v1"
_LOGGER = logging.getLogger(__name__)


class Workspace(Protocol):
    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None: ...

    def apply(self, action: str, *, env_step: int, phase_index: int): ...

    def close(self) -> None: ...


WorkspaceFactory = Callable[[int, str, int], Workspace]


@dataclass
class _Episode:
    env_id: int
    episode_index: int = 0
    task: GaiaTextTask | None = None
    step_count: int = 0
    backend_call_count: int = 0
    workspace_action_count: int = 0
    done: bool = False
    status: str = "unbound"
    workspace: Workspace | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    closed: bool = False
    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


class GaiaTextEpisodeManager:
    """One action dispatcher shared by native and AMG-memory evaluation arms."""

    def __init__(
        self,
        dataset: GaiaTextDataset,
        backend: FixtureBackend,
        submissions: SubmissionStore,
        *,
        arm: EvaluationArm | str,
        workspace_factory: WorkspaceFactory | None = None,
        workspace_runtime: Mapping[str, Any] | None = None,
        max_policy_steps: int = 40,
    ) -> None:
        self.arm = EvaluationArm(arm)
        if self.arm is EvaluationArm.NATIVE and workspace_factory is not None:
            raise ValueError("native arm must not receive a workspace factory")
        if self.arm is EvaluationArm.AMG_MEMORY and workspace_factory is None:
            raise ValueError("amg_memory arm requires a workspace factory")
        if type(max_policy_steps) is not int or max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be a positive integer")
        self.dataset = dataset
        self.backend = backend
        self.submissions = submissions
        self._workspace_factory = workspace_factory
        self._workspace_runtime = deepcopy(dict(workspace_runtime or {}))
        self.max_policy_steps = max_policy_steps
        self._next_id = 0
        self._episodes: dict[int, _Episode] = {}
        self._claimed_task_ids: set[str] = set()
        self._lock = threading.RLock()
        self._paired_runtime_contract = self._build_paired_runtime_contract()
        self._paired_runtime_contract_sha256 = _canonical_sha256(
            self._paired_runtime_contract
        )

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            episodes = tuple(self._episodes.values())
        active_environment_count = 0
        active_workspace_count = 0
        for episode in episodes:
            with episode.lock:
                if not episode.closed:
                    active_environment_count += 1
                    if episode.workspace is not None:
                        active_workspace_count += 1
        workspace_available = self.arm is EvaluationArm.AMG_MEMORY
        return {
            **self.dataset.public_metadata(),
            "domain_id": "gaia_text",
            "surface": "gaia_text_thin_http_adapter_v1",
            "arm": self.arm.value,
            "task_count": len(self.dataset),
            "backend": self.backend.metadata(),
            "reward_contract": _REWARD_CONTRACT,
            "submission_contract": _SUBMISSION_CONTRACT,
            "domain_action_contract": _DOMAIN_ACTION_CONTRACT,
            "answer_extraction_contract": _ANSWER_EXTRACTION_CONTRACT,
            "paired_runtime_contract": deepcopy(self._paired_runtime_contract),
            "paired_runtime_contract_sha256": self._paired_runtime_contract_sha256,
            "workspace_available": workspace_available,
            "workspace_contract": (
                "codex_shell_command_apply_patch_v1"
                if workspace_available
                else "disabled"
            ),
            "workspace_lifetime": ("clean_per_task" if workspace_available else "none"),
            "workspace_runtime": (
                deepcopy(self._workspace_runtime) if workspace_available else {}
            ),
            "compaction_available": workspace_available,
            "compaction_contract": (
                "task_neutral_client_replace_messages_v1"
                if workspace_available
                else "disabled"
            ),
            "compaction_calls_server": False,
            "compaction_calls_backend": False,
            "max_policy_steps": self.max_policy_steps,
            "max_policy_steps_enforced_by": "shared_policy_runner",
            "server_native_action_safety_cap": self.max_policy_steps,
            "active_environment_count": active_environment_count,
            "active_workspace_count": active_workspace_count,
        }

    def _build_paired_runtime_contract(self) -> dict[str, Any]:
        dataset = self.dataset.public_metadata()
        backend = self.backend.metadata()
        return {
            "schema": _PAIRED_RUNTIME_SCHEMA,
            "protocol_id": dataset["protocol_id"],
            "dataset_revision": dataset["dataset_revision"],
            "split": dataset["split"],
            "task_count": dataset["task_count"],
            "level_counts": deepcopy(dataset["level_counts"]),
            "manifest_sha256": dataset["manifest_sha256"],
            "task_ids_sha256": dataset["task_ids_sha256"],
            "questions_sha256": dataset["questions_sha256"],
            "backend_contract": backend["backend_contract"],
            "backend_asset_sha256": backend["asset_sha256"],
            "visit_page_chars": backend["page_chars"],
            "max_policy_steps": self.max_policy_steps,
            "domain_action_contract": _DOMAIN_ACTION_CONTRACT,
            "answer_extraction_contract": _ANSWER_EXTRACTION_CONTRACT,
            "reward_contract": _REWARD_CONTRACT,
            "submission_contract": _SUBMISSION_CONTRACT,
        }

    def create(self) -> dict[str, Any]:
        with self._lock:
            env_id = self._next_id
            self._next_id += 1
            episode = _Episode(env_id=env_id)
            episode.payload = self._payload(
                episode,
                include_id=True,
                observation="GAIA-Text environment created; reset with an explicit data_idx.",
                domain_action="create",
            )
            self._episodes[env_id] = episode
            return deepcopy(episode.payload)

    def reset(self, env_id: int, data_idx: int) -> dict[str, Any]:
        task = self.dataset.task(data_idx)
        episode = self._require(env_id)
        with episode.lock:
            self._ensure_open(episode)
            if episode.task is not None and not episode.done:
                raise RuntimeError("cannot reset an active unfinished GAIA-Text task")
            with self._lock:
                if task.task_id in self._claimed_task_ids:
                    raise RuntimeError(
                        f"GAIA-Text task {task.task_id!r} is already claimed"
                    )
                self._claimed_task_ids.add(task.task_id)
            next_episode_index = episode.episode_index + 1
            workspace = None
            try:
                if self.arm is EvaluationArm.AMG_MEMORY:
                    factory = self._workspace_factory
                    if factory is None:  # pragma: no cover - constructor invariant.
                        raise RuntimeError("memory workspace factory disappeared")
                    workspace = factory(env_id, task.task_id, next_episode_index)
                    workspace.reset_episode(
                        "gaia-text:"
                        f"env-{env_id}:episode-{next_episode_index}:task-{task.task_id}",
                        enabled=True,
                    )
                previous_workspace = episode.workspace
                if previous_workspace is not None:
                    previous_workspace.close()
            except BaseException:
                try:
                    _best_effort_close(workspace)
                finally:
                    with self._lock:
                        self._claimed_task_ids.discard(task.task_id)
                raise
            episode.episode_index = next_episode_index
            episode.task = task
            episode.step_count = 0
            episode.backend_call_count = 0
            episode.workspace_action_count = 0
            episode.done = False
            episode.status = "active"
            episode.workspace = workspace
            observation = json.dumps(
                task.as_policy_record(), ensure_ascii=False, separators=(",", ":")
            )
            episode.payload = self._payload(
                episode,
                include_id=False,
                observation=observation,
                domain_action="reset",
            )
            return deepcopy(episode.payload)

    def step(self, env_id: int, action: str) -> dict[str, Any]:
        episode = self._require(env_id)
        with episode.lock:
            self._ensure_open(episode)
            self._ensure_bound(episode)
            if episode.done:
                raise RuntimeError("GAIA-Text task is already terminal")
            if episode.step_count >= self.max_policy_steps:
                raise RuntimeError(
                    "server native-action safety cap reached; finalize the shared policy horizon"
                )
            if not isinstance(action, str):
                raise TypeError("policy action must be text")
            episode.step_count += 1
            try:
                parsed_tool = _parse_tool_call(action)
                answer_matches = list(_ANSWER_RE.finditer(action))
                if parsed_tool is not None and answer_matches:
                    raise RequestError(
                        "one policy action cannot contain both tool_call and answer"
                    )
                if len(answer_matches) > 1:
                    raise RequestError(
                        "one policy action cannot contain multiple answers"
                    )
                if parsed_tool is not None:
                    return self._domain_tool(episode, action, parsed_tool)
                if answer_matches:
                    answer = answer_matches[0].group(1).strip()
                    return self._answer(episode, action, answer)
                if action.strip().startswith(_WORKSPACE_PREFIXES):
                    return self._workspace(episode, action)
                raise RequestError(
                    "expected one search/visit tool_call, one answer, or one workspace action"
                )
            except RequestError as exc:
                return self._ordinary(
                    episode,
                    observation=f"Invalid policy action: {exc}",
                    status="invalid_action",
                    domain_action="invalid",
                    action_submission={"raw_policy_output": action},
                    wrapper_evidence={"invalid_action": True},
                )
            except (TypeError, ValueError, KeyError) as exc:
                return self._ordinary(
                    episode,
                    observation=f"Invalid policy action: {exc}",
                    status="invalid_action",
                    domain_action="invalid",
                    action_submission={"raw_policy_output": action},
                    wrapper_evidence={"invalid_action": True},
                )
            except BackendError as exc:
                return self._terminal_failure(episode, action, type(exc).__name__)

    def finalize_horizon(self, env_id: int) -> dict[str, Any]:
        episode = self._require(env_id)
        with episode.lock:
            self._ensure_open(episode)
            self._ensure_bound(episode)
            if episode.done:
                raise RuntimeError("GAIA-Text task is already terminal")
            task = _task(episode)
            receipt = self.submissions.record(task.task_id, None)
            episode.done = True
            episode.status = "policy_horizon"
            episode.payload = self._payload(
                episode,
                include_id=False,
                observation="Policy-turn budget exhausted; null answer submitted externally.",
                domain_action="horizon",
                action_submission={"control_action": "horizon"},
                submission_receipt=receipt,
                wrapper_evidence={"outcome": "max_policy_steps"},
            )
            return deepcopy(episode.payload)

    def close(self, env_id: int) -> bool:
        episode = self._require(env_id)
        with episode.lock:
            self._ensure_open(episode)
            if episode.task is not None and not episode.done:
                raise RuntimeError(
                    "cannot close an unfinished GAIA-Text task; finalize its horizon first"
                )
            if episode.workspace is not None:
                episode.workspace.close()
                episode.workspace = None
            with self._lock:
                if self._episodes.get(env_id) is not episode:
                    raise RuntimeError(
                        "GAIA-Text environment registry changed unexpectedly"
                    )
                episode.closed = True
                del self._episodes[env_id]
            return True

    def _domain_tool(
        self,
        episode: _Episode,
        raw_action: str,
        parsed: tuple[str, dict[str, Any]],
    ) -> dict[str, Any]:
        name, arguments = parsed
        if name == "search":
            if set(arguments) - {"query", "top_k"} or "query" not in arguments:
                raise RequestError("search arguments require query and optional top_k")
            result = self.backend.search(
                arguments["query"], top_k=arguments.get("top_k", 5)
            )
            episode.backend_call_count += 1
            observation = json.dumps(
                {"tool": "search", "results": result},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        elif name == "visit":
            if set(arguments) - {"url", "goal", "page"} or "url" not in arguments:
                raise RequestError("visit arguments require url and optional goal/page")
            result = self.backend.visit(
                arguments["url"],
                goal=arguments.get("goal", ""),
                page=arguments.get("page", 1),
            )
            episode.backend_call_count += 1
            observation = json.dumps(
                {"tool": "visit", "page": result},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            raise RequestError("only search and visit domain tools are available")
        return self._ordinary(
            episode,
            observation=observation,
            status="active",
            domain_action=name,
            action_submission={
                "raw_policy_output": raw_action,
                "tool": name,
                "arguments": arguments,
            },
            wrapper_evidence={"native_environment_call_count": 1},
        )

    def _answer(
        self, episode: _Episode, raw_action: str, answer: str
    ) -> dict[str, Any]:
        task = _task(episode)
        receipt = self.submissions.record(task.task_id, answer)
        episode.done = True
        episode.status = "answer_submitted"
        episode.payload = self._payload(
            episode,
            include_id=False,
            observation="Final answer submitted for external official scoring.",
            domain_action="answer",
            action_submission={"raw_policy_output": raw_action, "kind": "answer"},
            submission_receipt=receipt,
            wrapper_evidence={"terminal_answer_only": True},
        )
        return deepcopy(episode.payload)

    def _workspace(self, episode: _Episode, raw_action: str) -> dict[str, Any]:
        if self.arm is EvaluationArm.NATIVE or episode.workspace is None:
            raise RequestError("workspace actions are unavailable in the native arm")
        result = episode.workspace.apply(
            raw_action,
            env_step=episode.step_count,
            phase_index=0,
        )
        if result is None:
            raise RequestError("workspace action did not match a supported tool form")
        episode.workspace_action_count += 1
        return self._ordinary(
            episode,
            observation=str(getattr(result, "message", result)),
            status="active",
            domain_action="workspace",
            action_submission={
                "raw_policy_output": raw_action,
                "kind": "workspace",
                "op": str(getattr(result, "op", "WORKSPACE")),
            },
            wrapper_evidence={
                "workspace_op": str(getattr(result, "op", "WORKSPACE")),
                "native_environment_call_count": 0,
            },
        )

    def _ordinary(
        self,
        episode: _Episode,
        *,
        observation: str,
        status: str,
        domain_action: str,
        action_submission: Mapping[str, Any],
        wrapper_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        episode.status = status
        episode.payload = self._payload(
            episode,
            include_id=False,
            observation=observation,
            domain_action=domain_action,
            action_submission=action_submission,
            wrapper_evidence=wrapper_evidence,
        )
        return deepcopy(episode.payload)

    def _terminal_failure(
        self, episode: _Episode, raw_action: str, error_type: str
    ) -> dict[str, Any]:
        task = _task(episode)
        receipt = self.submissions.record(task.task_id, None)
        episode.done = True
        episode.status = "environment_error"
        episode.payload = self._payload(
            episode,
            include_id=False,
            observation="Verified search/browse backend failed closed.",
            domain_action="environment_error",
            action_submission={"raw_policy_output": raw_action},
            submission_receipt=receipt,
            wrapper_evidence={"backend_error": error_type},
        )
        return deepcopy(episode.payload)

    def _payload(
        self,
        episode: _Episode,
        *,
        include_id: bool,
        observation: str,
        domain_action: str,
        action_submission: Mapping[str, Any] | None = None,
        submission_receipt: Mapping[str, Any] | None = None,
        wrapper_evidence: Mapping[str, Any] | None = None,
        sample_excluded: bool = False,
    ) -> dict[str, Any]:
        task = episode.task
        info = {
            "schema": "gaia_text_public_episode_v1",
            "status": episode.status,
            "task_id": None if task is None else task.task_id,
            "level": None if task is None else task.level,
            "step": episode.step_count,
            "backend_call_count": episode.backend_call_count,
            "workspace_action_count": episode.workspace_action_count,
            "domain_action": domain_action,
            "sample_excluded": sample_excluded,
            "action_submission": (
                None if action_submission is None else dict(action_submission)
            ),
            "submission_receipt": (
                None if submission_receipt is None else dict(submission_receipt)
            ),
            "wrapper_evidence": dict(wrapper_evidence or {}),
        }
        payload: dict[str, Any] = {
            "observation": observation,
            "reward": 0.0,
            "done": episode.done,
            "info": info,
        }
        if include_id:
            payload["id"] = episode.env_id
        return payload

    def _require(self, env_id: int) -> _Episode:
        if isinstance(env_id, bool) or not isinstance(env_id, int):
            raise KeyError("GAIA-Text environment ID must be an integer")
        with self._lock:
            try:
                return self._episodes[env_id]
            except KeyError as exc:
                raise KeyError(f"unknown GAIA-Text environment id {env_id}") from exc

    @staticmethod
    def _ensure_open(episode: _Episode) -> None:
        if episode.closed:
            raise KeyError(f"unknown GAIA-Text environment id {episode.env_id}")

    @staticmethod
    def _ensure_bound(episode: _Episode) -> None:
        if episode.task is None:
            raise RuntimeError("GAIA-Text environment must be reset before use")


def _parse_tool_call(raw: str) -> tuple[str, dict[str, Any]] | None:
    matches = list(_TOOL_CALL_RE.finditer(raw))
    if not matches:
        return None
    if len(matches) > 1:
        raise RequestError("one policy action cannot contain multiple tool calls")
    try:
        payload = json.loads(matches[0].group(1))
    except json.JSONDecodeError as exc:
        raise RequestError("tool_call must contain a JSON object") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"name", "arguments"}:
        raise RequestError("tool_call requires exactly name and arguments")
    name = payload["name"]
    arguments = payload["arguments"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(arguments, Mapping)
    ):
        raise RequestError("tool_call requires a non-empty name and object arguments")
    return name.strip().casefold(), dict(arguments)


def _task(episode: _Episode) -> GaiaTextTask:
    if episode.task is None:  # pragma: no cover - bound callers own invariant.
        raise RuntimeError("GAIA-Text episode is unbound")
    return episode.task


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _best_effort_close(workspace: Workspace | None) -> None:
    if workspace is None:
        return
    try:
        workspace.close()
    except OSError:
        _LOGGER.warning("failed to clean a rejected GAIA-Text workspace", exc_info=True)
