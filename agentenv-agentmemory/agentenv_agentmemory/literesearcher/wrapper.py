from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol

from .backend import FrozenLiteResearchBackend, LiteResearchBackendError
from .contracts import LiteResearcherCoverage


LITERESEARCHER_SURFACE = "agentmemory_literesearcher_stage1_rag_only_v1"
_REPLACE_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
_APPEND_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_COMPACTION_RE = re.compile(
    r"<context_compaction>\s*(.*?)\s*</context_compaction>",
    re.IGNORECASE | re.DOTALL,
)


class WorkspaceAdapter(Protocol):
    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        ...

    def apply(self, action: str, *, env_step: int, phase_index: int):
        ...

    def close(self) -> None:
        ...


def _normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def _parse_tool_call(raw: str) -> tuple[str, dict[str, Any]] | None:
    match = _TOOL_CALL_RE.search(raw)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("tool_call must contain a JSON object") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("tool_call payload must be an object")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, Mapping):
        raise ValueError("tool_call requires a name and object arguments")
    return name.strip().lower(), dict(arguments)


class LiteResearcherWrapper:
    """HTTP-shaped wrapper for one continuous LiteResearcher episode.

    There are no hand-authored sessions.  Every search, visit, workspace action,
    answer, and policy-authored context compaction consumes one ordinary RL
    policy step.  ``context_token_count`` is supplied by the rollout adapter;
    this wrapper only decides whether the next row must be a compaction row.
    """

    def __init__(
        self,
        coverage: LiteResearcherCoverage,
        backend: FrozenLiteResearchBackend,
        *,
        workspace: WorkspaceAdapter | None = None,
        workspace_factory: Callable[[int], WorkspaceAdapter] | None = None,
        max_policy_steps: int = 40,
        model_context_tokens: int = 32_768,
        max_response_tokens: int = 2_048,
        compaction_margin_tokens: int = 256,
    ) -> None:
        if backend.coverage is not coverage:
            raise ValueError("backend and wrapper must share the exact coverage object")
        if workspace is not None and workspace_factory is not None:
            raise ValueError("provide workspace or workspace_factory, not both")
        values = (
            max_policy_steps,
            model_context_tokens,
            max_response_tokens,
            compaction_margin_tokens,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("LiteResearcher limits must be positive integers")
        if max_response_tokens + compaction_margin_tokens >= model_context_tokens:
            raise ValueError("response budget plus compaction margin must fit context")
        self.coverage = coverage
        self.backend = backend
        self._shared_workspace = workspace
        self._workspace_factory = workspace_factory
        self._workspaces: dict[int, WorkspaceAdapter] = {}
        self.max_policy_steps = max_policy_steps
        self.model_context_tokens = model_context_tokens
        self.max_response_tokens = max_response_tokens
        self.compaction_margin_tokens = compaction_margin_tokens
        self._next_id = 0
        self._episodes: dict[int, dict[str, Any]] = {}

    def metadata(self) -> dict[str, Any]:
        metadata = self.coverage.public_metadata()
        metadata.update(
            {
                "surface": LITERESEARCHER_SURFACE,
                "backend": self.backend.metadata(),
                "max_policy_steps": self.max_policy_steps,
                "model_context_tokens": self.model_context_tokens,
                "max_response_tokens": self.max_response_tokens,
                "compaction_margin_tokens": self.compaction_margin_tokens,
                "compaction_contract": "policy_authored_replace_messages_v1",
                "compaction_counts_as_env_step": True,
                "compaction_calls_backend": False,
                "reward_contract": "terminal_answer_only_binary_v1",
                "judge_fallback": "forbidden_fail_closed",
                "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
                "workspace_memory_reward": 0.0,
            }
        )
        return metadata

    def create(self, *, data_idx: int = 0) -> dict[str, Any]:
        env_id = self._next_id
        self._next_id += 1
        episode = self._new_episode(env_id, data_idx)
        self._episodes[env_id] = episode
        return self._payload(env_id, episode, include_id=True)

    def reset(self, env_id: int, data_idx: int = 0) -> dict[str, Any]:
        if env_id not in self._episodes:
            raise KeyError(f"Unknown LiteResearcher environment id {env_id}")
        episode = self._new_episode(env_id, data_idx)
        self._episodes[env_id] = episode
        return self._payload(env_id, episode, include_id=False)

    def close(self, env_id: int) -> bool:
        episode = self._episodes.pop(env_id, None)
        if episode is None:
            raise KeyError(f"Unknown LiteResearcher environment id {env_id}")
        workspace = episode.get("workspace")
        if workspace is not None:
            workspace.close()
            self._workspaces.pop(env_id, None)
        return True

    def detail(self, env_id: int) -> dict[str, Any]:
        return deepcopy(self._require(env_id)["payload"])

    def step(
        self,
        env_id: int,
        action: str,
        *,
        context_token_count: int | None = None,
    ) -> dict[str, Any]:
        episode = self._require(env_id)
        if episode["done"]:
            return deepcopy(episode["payload"])
        episode["step_count"] += 1
        step = episode["step_count"]
        if step > self.max_policy_steps:
            return self._finish(
                env_id,
                episode,
                reward=0.0,
                status="max_policy_steps_exhausted",
                outcome="terminal_failure",
                sample_excluded=False,
                observation="Maximum policy-step budget reached.",
                action_submission={"raw_policy_output": str(action)},
                transition=self._append_transition(episode, "Maximum policy-step budget reached."),
                wrapper_evidence={"step": step, "max_policy_steps": self.max_policy_steps},
            )

        count = self._validate_context_count(context_token_count)
        if count is not None:
            episode["last_context_token_count"] = count
        compaction_required = self._compaction_required(count)
        episode["compaction_required"] = compaction_required
        if compaction_required:
            compaction = _COMPACTION_RE.search(str(action))
            if compaction is None:
                return self._ordinary_result(
                    env_id,
                    episode,
                    observation=(
                        "Context capacity is exhausted. Submit one policy-authored "
                        "<context_compaction>summary</context_compaction> row before "
                        "another search, visit, answer, or workspace action."
                    ),
                    status="compaction_required",
                    action_submission={"raw_policy_output": str(action)},
                    transition=self._append_transition(
                        episode,
                        "Context compaction required before the next domain action.",
                    ),
                    wrapper_evidence={
                        "step": step,
                        "compaction_required": True,
                        "backend_call_count": 0,
                    },
                )
            try:
                return self._apply_compaction(
                    env_id, episode, str(action), compaction.group(1)
                )
            except (TypeError, ValueError) as exc:
                return self._ordinary_result(
                    env_id,
                    episode,
                    observation=f"Invalid policy compaction: {exc}",
                    status="invalid_compaction",
                    action_submission={"raw_policy_output": str(action)},
                    transition=self._append_transition(
                        episode, f"Invalid policy compaction: {exc}"
                    ),
                    wrapper_evidence={"step": step, "invalid_compaction": True},
                )
        if _COMPACTION_RE.search(str(action)) is not None:
            return self._ordinary_result(
                env_id,
                episode,
                observation="Compaction is accepted only when context pressure requires it.",
                status="invalid_compaction",
                action_submission={"raw_policy_output": str(action)},
                transition=self._append_transition(episode, "Invalid compaction row."),
                wrapper_evidence={"step": step, "compaction_required": False},
            )

        try:
            parsed_tool = _parse_tool_call(str(action))
            answer_match = _ANSWER_RE.search(str(action))
            if parsed_tool is not None and answer_match is not None:
                raise ValueError("one policy row cannot contain both tool_call and answer")
            if parsed_tool is not None:
                return self._apply_domain_tool(env_id, episode, str(action), parsed_tool)
            if answer_match is not None:
                return self._apply_answer(env_id, episode, str(action), answer_match.group(1))
            if str(action).startswith("shell_command") or str(action).startswith("apply_patch\n"):
                return self._apply_workspace(env_id, episode, str(action))
            raise ValueError(
                "expected one search/visit tool_call, one answer, or one workspace action"
            )
        except LiteResearchBackendError as exc:
            return self._finish(
                env_id,
                episode,
                reward=0.0,
                status="environment_error",
                outcome="environment_error",
                sample_excluded=True,
                observation="Frozen research backend failed; episode excluded.",
                action_submission={"raw_policy_output": str(action)},
                transition=self._append_transition(episode, "Frozen research backend failed."),
                wrapper_evidence={"step": step, "backend_error": type(exc).__name__},
            )
        except (TypeError, ValueError, KeyError) as exc:
            return self._ordinary_result(
                env_id,
                episode,
                observation=f"Invalid policy action: {exc}",
                status="invalid_action",
                action_submission={"raw_policy_output": str(action)},
                transition=self._append_transition(episode, f"Invalid policy action: {exc}"),
                wrapper_evidence={"step": step, "invalid_action": True},
            )

    def _new_episode(self, env_id: int, data_idx: int) -> dict[str, Any]:
        task = self.coverage.task(data_idx)
        old_episode = self._episodes.get(env_id)
        workspace = None if old_episode is None else old_episode.get("workspace")
        if workspace is not None:
            workspace.close()
            self._workspaces.pop(env_id, None)
        if self._shared_workspace is not None:
            if env_id != 0 and self._episodes:
                raise RuntimeError(
                    "a single injected workspace cannot back multiple LiteResearcher episodes"
                )
            workspace = self._shared_workspace
        elif self._workspace_factory is not None:
            workspace = self._workspace_factory(env_id)
            self._workspaces[env_id] = workspace
        else:
            workspace = None
        if workspace is not None:
            workspace.reset_episode(f"literesearcher:env{env_id}:episode", enabled=True)
        episode = {
            "env_id": env_id,
            "task": task,
            "step_count": 0,
            "done": False,
            "status": "active",
            "outcome": "continue",
            "sample_excluded": False,
            "context_epoch": 0,
            "compaction_count": 0,
            "compaction_required": False,
            "last_context_token_count": None,
            "backend_call_count": 0,
            "visited_urls": [],
            "payload": {},
            "workspace": workspace,
        }
        episode["payload"] = self._payload(env_id, episode, include_id=True)
        return episode

    def _require(self, env_id: int) -> dict[str, Any]:
        try:
            return self._episodes[env_id]
        except KeyError as exc:
            raise KeyError(f"Unknown LiteResearcher environment id {env_id}") from exc

    def _validate_context_count(self, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("context_token_count must be a non-negative integer")
        return value

    def _compaction_required(self, count: int | None) -> bool:
        if count is None:
            return False
        return (
            count + self.max_response_tokens + self.compaction_margin_tokens
            >= self.model_context_tokens
        )

    def _append_transition(self, episode: dict[str, Any], observation: str) -> dict[str, Any]:
        return {
            "schema": _APPEND_SCHEMA,
            "operation": "append_messages",
            "context_epoch_before": episode["context_epoch"],
            "context_epoch_after": episode["context_epoch"],
            "messages": [{"role": "tool", "content": observation}],
        }

    def _apply_compaction(
        self,
        env_id: int,
        episode: dict[str, Any],
        raw_action: str,
        summary: str,
    ) -> dict[str, Any]:
        summary = summary.strip()
        private_urls = {task.mask_url for task in (*self.coverage.train, *self.coverage.heldout)}
        if not summary:
            raise ValueError("policy-authored compaction summary must not be empty")
        if any(url in summary for url in private_urls):
            raise ValueError("compaction summary contains a server-private URL")
        before = episode["context_epoch"]
        after = before + 1
        episode["context_epoch"] = after
        episode["compaction_count"] += 1
        episode["compaction_required"] = False
        transition = {
            "schema": _REPLACE_SCHEMA,
            "operation": "replace_messages",
            "context_epoch_before": before,
            "context_epoch_after": after,
            "continuity_id": episode["task"].task_id,
            "workspace_path": ".agent_memory",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Continue the same LiteResearcher question with the frozen "
                        "search/visit tools."
                    ),
                },
                {"role": "user", "content": episode["task"].question},
                {"role": "assistant", "content": summary},
            ],
            "policy_authored": True,
        }
        evidence = {
            "step": episode["step_count"],
            "event": "context_compaction",
            "policy_authored": True,
            "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            "summary_chars": len(summary),
            "native_environment_call_count": 0,
            "context_epoch_before": before,
            "context_epoch_after": after,
        }
        return self._ordinary_result(
            env_id,
            episode,
            observation="Context replaced by the policy-authored summary; continue research.",
            status="context_compacted",
            action_submission={
                "raw_policy_output": raw_action,
                "kind": "context_compaction",
                "policy_authored": True,
            },
            transition=transition,
            wrapper_evidence=evidence,
        )

    def _apply_domain_tool(
        self,
        env_id: int,
        episode: dict[str, Any],
        raw_action: str,
        parsed: tuple[str, dict[str, Any]],
    ) -> dict[str, Any]:
        name, arguments = parsed
        if name == "search":
            query = arguments.get("query")
            result = self.backend.search(query)
            observation = json.dumps({"tool": "search", "results": result}, ensure_ascii=False)
            episode["backend_call_count"] += 1
            return self._ordinary_result(
                env_id,
                episode,
                observation=observation,
                status="active",
                action_submission={
                    "raw_policy_output": raw_action,
                    "tool": name,
                    "arguments": arguments,
                },
                transition=self._append_transition(episode, observation),
                wrapper_evidence={
                    "step": episode["step_count"],
                    "native_environment_call_count": 1,
                },
            )
        if name == "visit":
            urls = arguments.get("url")
            if isinstance(urls, str):
                urls = [urls]
            if not isinstance(urls, list) or not urls or any(
                not isinstance(url, str) for url in urls
            ):
                raise ValueError("visit arguments require a non-empty url string or list")
            visited = [self.backend.visit(url, goal=str(arguments.get("goal", ""))) for url in urls]
            episode["backend_call_count"] += len(visited)
            for item in visited:
                if item["url"] not in episode["visited_urls"]:
                    episode["visited_urls"].append(item["url"])
            observation = json.dumps({"tool": "visit", "pages": visited}, ensure_ascii=False)
            return self._ordinary_result(
                env_id,
                episode,
                observation=observation,
                status="active",
                action_submission={
                    "raw_policy_output": raw_action,
                    "tool": name,
                    "arguments": arguments,
                },
                transition=self._append_transition(episode, observation),
                wrapper_evidence={
                    "step": episode["step_count"],
                    "native_environment_call_count": len(visited),
                },
            )
        raise ValueError("only search and visit are available in the LiteResearcher gate")

    def _apply_answer(
        self,
        env_id: int,
        episode: dict[str, Any],
        raw_action: str,
        answer: str,
    ) -> dict[str, Any]:
        normalized = _normalize_answer(answer)
        targets = {_normalize_answer(target) for target in episode["task"].targets}
        correct = bool(normalized) and normalized in targets
        return self._finish(
            env_id,
            episode,
            reward=1.0 if correct else 0.0,
            status="success" if correct else "terminal_failure",
            outcome="success" if correct else "terminal_failure",
            sample_excluded=False,
            observation="Final answer accepted." if correct else "Final answer was not accepted.",
            action_submission={"raw_policy_output": raw_action, "kind": "answer"},
            transition=self._append_transition(
                episode,
                "Final answer accepted." if correct else "Final answer was not accepted.",
            ),
            wrapper_evidence={
                "step": episode["step_count"],
                "terminal_answer_only": True,
                "visited_url_count": len(episode["visited_urls"]),
                "answer_correct": correct,
            },
        )

    def _apply_workspace(
        self, env_id: int, episode: dict[str, Any], raw_action: str
    ) -> dict[str, Any]:
        workspace = episode.get("workspace")
        if workspace is None:
            raise ValueError("workspace tools are unavailable in this intake instance")
        result = workspace.apply(raw_action, env_step=episode["step_count"], phase_index=0)
        message = str(getattr(result, "message", result))
        op = str(getattr(result, "op", "WORKSPACE")).upper()
        return self._ordinary_result(
            env_id,
            episode,
            observation=message,
            status="active",
            action_submission={"raw_policy_output": raw_action, "kind": "workspace", "op": op},
            transition=self._append_transition(episode, message),
            wrapper_evidence={
                "step": episode["step_count"],
                "workspace_op": op,
                "workspace_reward": 0.0,
                "native_environment_call_count": 0,
            },
        )

    def _ordinary_result(
        self,
        env_id: int,
        episode: dict[str, Any],
        *,
        observation: str,
        status: str,
        action_submission: Mapping[str, Any],
        transition: Mapping[str, Any],
        wrapper_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = dict(wrapper_evidence)
        episode["status"] = status
        if episode["step_count"] >= self.max_policy_steps:
            episode.update(
                {
                    "done": True,
                    "status": "max_policy_steps_exhausted",
                    "outcome": "terminal_failure",
                    "sample_excluded": False,
                    "reward": 0.0,
                }
            )
            evidence.update(
                {
                    "max_policy_steps_exhausted": True,
                    "max_policy_steps": self.max_policy_steps,
                }
            )
        episode["payload"] = self._payload(
            env_id,
            episode,
            include_id=False,
            observation=observation,
            action_submission=action_submission,
            context_transition=transition,
            wrapper_evidence=evidence,
        )
        return deepcopy(episode["payload"])

    def _finish(
        self,
        env_id: int,
        episode: dict[str, Any],
        *,
        reward: float,
        status: str,
        outcome: str,
        sample_excluded: bool,
        observation: str,
        action_submission: Mapping[str, Any],
        transition: Mapping[str, Any],
        wrapper_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        episode.update(
            {
                "done": True,
                "status": status,
                "outcome": outcome,
                "sample_excluded": sample_excluded,
                "reward": float(reward),
            }
        )
        episode["payload"] = self._payload(
            env_id,
            episode,
            include_id=False,
            observation=observation,
            action_submission=action_submission,
            context_transition=transition,
            wrapper_evidence=wrapper_evidence,
        )
        return deepcopy(episode["payload"])

    def _payload(
        self,
        env_id: int,
        episode: dict[str, Any],
        *,
        include_id: bool,
        observation: str | None = None,
        action_submission: Mapping[str, Any] | None = None,
        context_transition: Mapping[str, Any] | None = None,
        wrapper_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        task = episode["task"]
        info = {
            "formal_schema_version": "agentmemory_formal_step_v3",
            "domain_id": "literesearcher",
            "surface": LITERESEARCHER_SURFACE,
            "task_id": task.task_id,
            "data_idx": task.index,
            "status": episode["status"],
            "phase_index": 0,
            "phase_count": None,
            "episode_success": episode["outcome"] == "success",
            "sample_excluded": bool(episode["sample_excluded"]),
            "action_submission": deepcopy(dict(action_submission or {})),
            "context_transition": deepcopy(
                dict(context_transition or self._append_transition(episode, ""))
            ),
            "wrapper_evidence": deepcopy(dict(wrapper_evidence or {})),
            "native_environment_call_count": int(episode["backend_call_count"]),
            "compaction_count": int(episode["compaction_count"]),
            "context_epoch": int(episode["context_epoch"]),
            "compaction_required": bool(episode["compaction_required"]),
            "workspace_tools": ["shell_command", "apply_patch"],
            "reward_contract": "terminal_answer_only_binary_v1",
        }
        payload = {
            "observation": task.question if observation is None else str(observation),
            "reward": float(episode.get("reward", 0.0)),
            "done": bool(episode["done"]),
            "info": info,
        }
        if include_id:
            payload["id"] = env_id
        return payload
