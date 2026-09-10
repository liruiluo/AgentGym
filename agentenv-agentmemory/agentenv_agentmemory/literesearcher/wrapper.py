from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol

from .backend import LiteResearchBackendError
from ..persistent_workspace import WORKSPACE_TOOL_NAMES
from .judge import (
    LiteResearchJudge,
    LiteResearchJudgeInfrastructureError,
    NormalizedExactLiteResearchJudge,
    UPSTREAM_LLM_JUDGE_CONTRACT,
)


LITERESEARCHER_SURFACE = "agentmemory_literesearcher_stage1_rag_only_v1"
LITERESEARCHER_FULLPOOL_SURFACE = (
    "agentmemory_literesearcher_fullpool_upstream_hybrid_v1"
)
_APPEND_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_NATIVE_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_NATIVE_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_receipt_v1"
)
FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_read_receipt_v1"
)
FILESYSTEM_CHECKPOINT_PATH = ".agent_memory/CONTINUATION.md"
FILESYSTEM_CHECKPOINT_MAX_BYTES = 8 * 1024


class WorkspaceAdapter(Protocol):
    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        ...

    def apply(self, action: str, *, env_step: int, phase_index: int):
        ...

    def snapshot(self) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...


def _normalize_xml_parameter_name(raw: str) -> str:
    name = raw.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in {"'", '"'}:
        name = name[1:-1].strip()
    return name.lower()


def _parse_xml_parameters(body: str) -> dict[str, str]:
    parameters: dict[str, str] = {}
    for key, value in _NATIVE_PARAMETER_RE.findall(body):
        normalized_key = _normalize_xml_parameter_name(key)
        if not normalized_key:
            raise ValueError("tool_call parameter name must be non-empty")
        if normalized_key in parameters:
            raise ValueError(f"tool_call repeats parameter {normalized_key!r}")
        parameters[normalized_key] = value.strip()
    return parameters


def _parse_workspace_xml_call(
    *, name: str, body: str, parameters: Mapping[str, str]
) -> dict[str, Any]:
    if name == "apply_patch":
        # Qwen may place a patch directly in the function body, while the
        # schema-guided form uses <parameter=patch>.  Both map to the one
        # existing canonical workspace adapter; the adapter remains the strict
        # parser and transactional executor.
        if body.lstrip().startswith("*** Begin Patch"):
            patch = body.strip()
        else:
            if set(parameters) != {"patch"}:
                raise ValueError("apply_patch requires exactly one patch parameter")
            patch = parameters["patch"].strip()
        if not patch:
            raise ValueError("apply_patch patch must be non-empty")
        return {"patch": patch}

    if name != "shell_command":
        raise ValueError(f"unsupported workspace tool {name!r}")
    if body.lstrip().startswith("{"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("shell_command body must be valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("shell_command body must be a JSON object")
        return dict(payload)
    allowed = {"command", "workdir", "timeout_ms"}
    if "command" not in parameters or not set(parameters) <= allowed:
        raise ValueError(
            "shell_command requires command, with optional workdir and timeout_ms"
        )
    payload: dict[str, Any] = {
        "command": _parse_native_string(parameters["command"], name="command")
    }
    if "workdir" in parameters:
        payload["workdir"] = _parse_native_string(
            parameters["workdir"], name="workdir"
        )
    if "timeout_ms" in parameters:
        try:
            payload["timeout_ms"] = int(parameters["timeout_ms"])
        except ValueError as exc:
            raise ValueError(
                "shell_command timeout_ms must be a positive integer"
            ) from exc
        if payload["timeout_ms"] < 1:
            raise ValueError("shell_command timeout_ms must be a positive integer")
    return payload


def _parse_tool_call(raw: str) -> tuple[str, dict[str, Any]] | None:
    native_match = _NATIVE_TOOL_CALL_RE.search(raw)
    if native_match is not None:
        name = native_match.group(1).strip().lower()
        body = native_match.group(2).strip()
        parameters = _parse_xml_parameters(body)
        if name == "search":
            if set(parameters) != {"query"}:
                raise ValueError("search requires exactly one query parameter")
            try:
                query = json.loads(parameters["query"])
            except json.JSONDecodeError as exc:
                raise ValueError("search query parameter must be a JSON array") from exc
            if (
                not isinstance(query, list)
                or not query
                or any(not isinstance(item, str) or not item.strip() for item in query)
            ):
                raise ValueError("search query parameter must be a non-empty string array")
            return name, {"query": query}
        if name == "visit":
            required = {"url", "goal"}
            if not required <= parameters.keys() or not set(parameters) <= required | {"page"}:
                raise ValueError("visit requires url and goal, with optional page")
            arguments: dict[str, Any] = {
                "url": _parse_native_string(parameters["url"], name="url"),
                "goal": _parse_native_string(parameters["goal"], name="goal"),
            }
            if "page" in parameters:
                try:
                    page = int(parameters["page"])
                except ValueError as exc:
                    raise ValueError("visit page must be a positive integer") from exc
                if page < 1:
                    raise ValueError("visit page must be a positive integer")
                arguments["page"] = page
            return name, arguments
        if name in WORKSPACE_TOOL_NAMES:
            if native_match.group(0).strip() != raw.strip():
                raise ValueError("workspace tool_call must be the complete policy output")
            return name, _parse_workspace_xml_call(
                name=name, body=body, parameters=parameters
            )
        raise ValueError(
            "only search, visit, shell_command, and apply_patch are available "
            "in LiteResearcher"
        )

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
    normalized_name = name.strip().lower()
    if (
        normalized_name in WORKSPACE_TOOL_NAMES
        and match.group(0).strip() != raw.strip()
    ):
        raise ValueError("workspace tool_call must be the complete policy output")
    return normalized_name, dict(arguments)


def _canonical_workspace_action(
    name: str, arguments: Mapping[str, Any]
) -> str:
    if name == "shell_command":
        return "shell_command " + json.dumps(
            dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if name == "apply_patch":
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("apply_patch patch must be a non-empty string")
        return "apply_patch\n" + patch.strip()
    raise ValueError(f"unsupported workspace tool {name!r}")

def _parse_native_string(raw: str, *, name: str) -> str:
    value: Any = raw.strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = value
    if not isinstance(decoded, str) or not decoded.strip():
        raise ValueError(f"visit {name} must be a non-empty string")
    return decoded.strip()


def _workspace_action_completed(result: Any) -> bool:
    tool_op = getattr(result, "tool_op", None)
    if not isinstance(tool_op, Mapping) or tool_op.get("status") != "executed":
        return False
    action_kind = str(getattr(result, "op", "")).lower()
    if action_kind == "shell_command":
        exit_code = tool_op.get("exit_code")
        return bool(
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code == 0
            and tool_op.get("timed_out") is False
        )
    return action_kind == "apply_patch"


def _workspace_changed_paths(workspace_diff: Any) -> tuple[str, ...]:
    if not isinstance(workspace_diff, Mapping):
        return ()
    paths: set[str] = set()
    for item in workspace_diff.get("added", ()):
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            paths.add(str(item["path"]))
    for item in workspace_diff.get("modified", ()):
        if not isinstance(item, Mapping):
            continue
        after = item.get("after")
        path = after.get("path") if isinstance(after, Mapping) else item.get("path")
        if isinstance(path, str):
            paths.add(path)
    for item in workspace_diff.get("deleted", ()):
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            paths.add(str(item["path"]))
    return tuple(sorted(paths))


def _checkpoint_entry_size(entry: Any) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("bytes", entry.get("size"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _explicit_regular_checkpoint_entry(entry: Any) -> bool:
    if not isinstance(entry, Mapping) or entry.get("kind") != "file":
        return False
    size = _checkpoint_entry_size(entry)
    digest = entry.get("sha256")
    return bool(
        size is not None
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def _filesystem_checkpoint_receipt(
    *,
    result: Any,
    workspace_snapshot: Mapping[str, Any],
) -> dict[str, Any] | None:
    if result is None:
        return None
    workspace_diff = getattr(result, "workspace_diff", None)
    diff = workspace_diff if isinstance(workspace_diff, Mapping) else {}
    changed_entry: Mapping[str, Any] | None = None
    for item in diff.get("added", ()):
        if (
            isinstance(item, Mapping)
            and item.get("path") == FILESYSTEM_CHECKPOINT_PATH
            and _explicit_regular_checkpoint_entry(item)
        ):
            changed_entry = item
            break
    if changed_entry is None:
        for item in diff.get("modified", ()):
            if not isinstance(item, Mapping):
                continue
            before = item.get("before")
            after = item.get("after")
            if (
                isinstance(before, Mapping)
                and isinstance(after, Mapping)
                and before.get("path") == FILESYSTEM_CHECKPOINT_PATH
                and after.get("path") == FILESYSTEM_CHECKPOINT_PATH
                and _explicit_regular_checkpoint_entry(before)
                and _explicit_regular_checkpoint_entry(after)
                and before.get("sha256") != after.get("sha256")
            ):
                changed_entry = after
                break
    snapshot_entry: Mapping[str, Any] | None = None
    files = workspace_snapshot.get("files", ())
    if isinstance(files, (list, tuple)):
        for item in files:
            if isinstance(item, Mapping) and item.get("path") == FILESYSTEM_CHECKPOINT_PATH:
                snapshot_entry = item
                break
    size = _checkpoint_entry_size(snapshot_entry)
    digest = (
        snapshot_entry.get("sha256")
        if isinstance(snapshot_entry, Mapping)
        else None
    )
    regular_file = _explicit_regular_checkpoint_entry(snapshot_entry)
    changed = bool(
        changed_entry is not None
        and regular_file
        and _checkpoint_entry_size(changed_entry) == size
        and changed_entry.get("sha256") == digest
    )
    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "action_kind": str(getattr(result, "op", "")).lower(),
        "action_completed": _workspace_action_completed(result),
        "changed": changed,
        "exists": snapshot_entry is not None,
        "regular_file": regular_file,
        "size_bytes": size if isinstance(size, int) and not isinstance(size, bool) else None,
        "sha256": digest if isinstance(digest, str) else None,
    }


def _filesystem_checkpoint_read_receipt(
    *,
    result: Any,
    checkpoint_receipt: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if result is None or not isinstance(checkpoint_receipt, Mapping):
        return None
    tool_op = getattr(result, "tool_op", None)
    stdout = tool_op.get("stdout") if isinstance(tool_op, Mapping) else None
    payload = stdout.encode("utf-8") if isinstance(stdout, str) else None
    size = checkpoint_receipt.get("size_bytes")
    digest = checkpoint_receipt.get("sha256")
    observed = bool(
        str(getattr(result, "op", "")).lower() == "shell_command"
        and _workspace_action_completed(result)
        and isinstance(tool_op, Mapping)
        and tool_op.get("stdout_truncated") is False
        and checkpoint_receipt.get("changed") is False
        and checkpoint_receipt.get("exists") is True
        and checkpoint_receipt.get("regular_file") is True
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < size <= FILESYSTEM_CHECKPOINT_MAX_BYTES
        and isinstance(digest, str)
        and payload is not None
        and len(payload) == size
        and hashlib.sha256(payload).hexdigest() == digest
    )
    return {
        "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "observed": observed,
        "size_bytes": size if isinstance(size, int) and not isinstance(size, bool) else None,
        "sha256": digest if isinstance(digest, str) else None,
    }


class LiteResearcherWrapper:
    """HTTP-shaped wrapper for one continuous LiteResearcher episode.

    There are no hand-authored sessions. Search, visit, workspace actions, and
    answers are ordinary native actions. The task-neutral client owns the exact
    token-pressure decision; its checkpoint write still calls this server as a
    normal workspace action before the client replaces old messages.
    """

    def __init__(
        self,
        task_source,
        backend,
        *,
        workspace: WorkspaceAdapter | None = None,
        workspace_factory: Callable[[int], WorkspaceAdapter] | None = None,
        workspace_runtime_metadata: Mapping[str, Any] | None = None,
        max_policy_steps: int = 40,
        split: str = "train",
        surface: str = LITERESEARCHER_SURFACE,
        judge: LiteResearchJudge | None = None,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        if backend.tasks_source is not task_source:
            raise ValueError("backend and wrapper must share the exact task source")
        if backend.split != split:
            raise ValueError("backend and wrapper must use the same LiteResearcher split")
        if workspace is not None and workspace_factory is not None:
            raise ValueError("provide workspace or workspace_factory, not both")
        if type(max_policy_steps) is not int or max_policy_steps <= 0:
            raise ValueError("LiteResearcher max_policy_steps must be positive")
        if surface not in {LITERESEARCHER_SURFACE, LITERESEARCHER_FULLPOOL_SURFACE}:
            raise ValueError("unsupported LiteResearcher surface")
        self.judge = judge or NormalizedExactLiteResearchJudge()
        if (
            surface == LITERESEARCHER_FULLPOOL_SURFACE
            and self.judge.contract_id != UPSTREAM_LLM_JUDGE_CONTRACT
        ):
            raise ValueError(
                "LiteResearcher full pool requires the upstream LLM judge "
                "with EM fallback"
            )
        self.task_source = task_source
        self.backend = backend
        self.split = split
        self.surface = surface
        self.reward_contract = (
            "terminal_upstream_llm_judge_binary_v1"
            if surface == LITERESEARCHER_FULLPOOL_SURFACE
            else "terminal_normalized_exact_binary_v1"
        )
        self.tasks = task_source.tasks_for_split(split)
        self._shared_workspace = workspace
        self._workspace_factory = workspace_factory
        self._workspace_runtime_metadata = dict(workspace_runtime_metadata or {})
        self._workspaces: dict[int, WorkspaceAdapter] = {}
        self.max_policy_steps = max_policy_steps
        if isinstance(invalid_action_penalty, bool) or not isinstance(
            invalid_action_penalty, (int, float)
        ):
            raise TypeError("LiteResearcher invalid-action penalty must be numeric")
        self.invalid_action_penalty = float(invalid_action_penalty)
        if self.invalid_action_penalty > 0.0:
            raise ValueError("LiteResearcher invalid-action penalty must be non-positive")
        self._next_id = 0
        self._episodes: dict[int, dict[str, Any]] = {}

    def metadata(self) -> dict[str, Any]:
        metadata = self.task_source.public_metadata()
        judge_metadata = self.judge.metadata()
        metadata.update(
            {
                "surface": self.surface,
                "domain_id": "literesearcher",
                "backend": self.backend.metadata(),
                "split": self.split,
                "task_count": len(self.tasks),
                "active_environment_count": len(self._episodes),
                "active_workspace_count": len(self._workspaces),
                "max_policy_steps": self.max_policy_steps,
                "max_policy_steps_enforced_by": "shared_policy_runner",
                "server_native_action_safety_cap": self.max_policy_steps,
                "compaction_contract": (
                    "policy_filesystem_checkpoint_then_client_replace_v2"
                ),
                "compaction_counts_as_env_step": True,
                "compaction_calls_backend": True,
                "reward_contract": self.reward_contract,
                "judge": judge_metadata,
                "judge_fallback": judge_metadata["fallback"],
                "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
                "policy_workspace_tool_contract": "qwen3_xml_function_call_v1",
                "workspace_memory_reward": 0.0,
                "recoverable_invalid_action_reward": self.invalid_action_penalty,
                "workspace_runtime": deepcopy(self._workspace_runtime_metadata),
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

    def observation(self, env_id: int) -> str:
        return str(self._require(env_id)["payload"]["observation"])

    def step(self, env_id: int, action: str) -> dict[str, Any]:
        episode = self._require(env_id)
        if episode["done"]:
            return deepcopy(episode["payload"])
        # Reward is transition-local.  Reset it before every recoverable action so
        # an invalid-action penalty cannot leak into the next valid tool call.
        episode["reward"] = 0.0
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

        try:
            parsed_tool = _parse_tool_call(str(action))
            answer_match = _ANSWER_RE.search(str(action))
            if parsed_tool is not None and answer_match is not None:
                raise ValueError("one policy row cannot contain both tool_call and answer")
            if parsed_tool is not None:
                name, arguments = parsed_tool
                if name in WORKSPACE_TOOL_NAMES:
                    return self._apply_workspace(
                        env_id,
                        episode,
                        str(action),
                        execution_action=_canonical_workspace_action(name, arguments),
                        policy_format="qwen3_xml",
                    )
                return self._apply_domain_tool(env_id, episode, str(action), parsed_tool)
            if answer_match is not None:
                return self._apply_answer(env_id, episode, str(action), answer_match.group(1))
            if str(action).startswith("shell_command") or str(action).startswith("apply_patch\n"):
                return self._apply_workspace(env_id, episode, str(action))
            raise ValueError(
                "expected one search/visit tool_call, one answer, or one workspace action"
            )
        except LiteResearchJudgeInfrastructureError as exc:
            return self._finish(
                env_id,
                episode,
                reward=0.0,
                status="environment_error",
                outcome="environment_error",
                sample_excluded=True,
                observation="Semantic judge unavailable; episode excluded.",
                action_submission={"raw_policy_output": str(action), "kind": "answer"},
                transition=self._append_transition(
                    episode, "Semantic judge unavailable; episode excluded."
                ),
                wrapper_evidence={
                    "step": step,
                    "judge_method": "infrastructure_fault",
                    "judge_attempts": exc.attempts,
                    "judge_latency_seconds": exc.latency_seconds,
                    "judge_primary_model": exc.primary_model,
                    "judge_fallback_used": False,
                    "judge_failure_reason": exc.reason,
                    "judge_infrastructure_fault": True,
                },
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
            episode["reward"] = self.invalid_action_penalty
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
        task = self.task_source.task(data_idx, split=self.split)
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
            "data_idx": data_idx,
            "task": task,
            "step_count": 0,
            "done": False,
            "status": "active",
            "outcome": "continue",
            "sample_excluded": False,
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

    def _append_transition(self, episode: dict[str, Any], observation: str) -> dict[str, Any]:
        return {
            "schema": _APPEND_SCHEMA,
            "operation": "append_messages",
            "context_epoch_before": 0,
            "context_epoch_after": 0,
            "messages": [{"role": "tool", "content": observation}],
        }

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
            result = self.backend.search(query, mask_url=episode["task"].mask_url)
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
            url = arguments.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError("visit arguments require exactly one non-empty url string")
            page = arguments.get("page", 1)
            visited = self.backend.visit(
                url,
                goal=str(arguments.get("goal", "")),
                page=page,
            )
            episode["backend_call_count"] += 1
            if visited["url"] not in episode["visited_urls"]:
                episode["visited_urls"].append(visited["url"])
            observation = json.dumps({"tool": "visit", "page": visited}, ensure_ascii=False)
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
                    "visit_page": visited["page"],
                    "visit_page_count": visited["page_count"],
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
        judgment = self.judge.judge(
            episode["task"].question,
            episode["task"].targets,
            answer,
        )
        correct = judgment.correct
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
                "judge_method": judgment.method,
                "judge_attempts": judgment.attempts,
                "judge_latency_seconds": judgment.latency_seconds,
                "judge_primary_model": judgment.primary_model,
                "judge_fallback_used": judgment.fallback_reason is not None,
                "judge_fallback_reason": judgment.fallback_reason,
            },
        )

    def _apply_workspace(
        self,
        env_id: int,
        episode: dict[str, Any],
        raw_action: str,
        *,
        execution_action: str | None = None,
        policy_format: str = "codex_canonical",
    ) -> dict[str, Any]:
        workspace = episode.get("workspace")
        if workspace is None:
            raise ValueError("workspace tools are unavailable in this intake instance")
        result = workspace.apply(
            raw_action if execution_action is None else execution_action,
            env_step=episode["step_count"],
            phase_index=0,
        )
        message = str(getattr(result, "message", result))
        op = str(getattr(result, "op", "WORKSPACE")).upper()
        checkpoint_receipt = _filesystem_checkpoint_receipt(
            result=result,
            workspace_snapshot=workspace.snapshot(),
        )
        checkpoint_read_receipt = _filesystem_checkpoint_read_receipt(
            result=result,
            checkpoint_receipt=checkpoint_receipt,
        )
        changed_paths = _workspace_changed_paths(
            getattr(result, "workspace_diff", None)
        )
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
                "workspace_policy_format": policy_format,
                "workspace_reward": 0.0,
                "native_environment_call_count": 0,
                "workspace_action_completed": _workspace_action_completed(result),
                "workspace_changed_paths": list(changed_paths),
                "filesystem_checkpoint": checkpoint_receipt,
                "filesystem_checkpoint_read": checkpoint_read_receipt,
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
            "surface": self.surface,
            "task_id": task.task_id,
            "data_idx": episode["data_idx"],
            "source_data_idx": task.index,
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
            "workspace_tools": ["shell_command", "apply_patch"],
            "reward_contract": self.reward_contract,
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
