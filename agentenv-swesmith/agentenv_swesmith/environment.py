from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol

from agentenv_agentmemory.workspace_patch import (
    WorkspacePatchError,
    apply_workspace_patch_touched_transaction,
    parse_workspace_patch,
)

from .actions import (
    UPSTREAM_SUBMISSION_SENTINEL,
    ParsedPolicyAction,
    parse_policy_action,
)
from .audit import SwesmithEpisodeAuditSink
from .dataset import SwesmithDataset, SwesmithRecord
from .grader import SwesmithGradeResult, SwesmithHiddenGrader
from .profile import SwesmithProfileBinding
from .sandbox import (
    LinuxNamespaceEpisodeSandbox,
    SwesmithSandboxError,
    snapshot_workspace_tree,
)
from .workspace import SwesmithWorkspace, SwesmithWorkspaceMaterializer


EPISODE_SCHEMA = "agentmemory_swesmith_native_episode_v1"
DEFAULT_TRAINING_MAX_POLICY_TURNS = 75
UPSTREAM_REFERENCE_MAX_POLICY_TURNS = 250
UPSTREAM_AGENT_REPOSITORY = "SWE-agent/mini-swe-agent"
UPSTREAM_AGENT_REVISION = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
REWARD_CONTRACT = "explicit_submission_full_resolution_binary_v2"
SUBMISSION_CONTRACT = "upstream_shell_output_sentinel_v1"
HORIZON_CONTRACT = "unified_policy_step_no_submission_failure_v2"
DEFAULT_MAX_OBSERVATION_BYTES = 6144
ACTOR_CREDIT_SCHEMA = "task_neutral_actor_credit_v1"
ACTION_PROGRESS_SCHEMA = "swesmith_action_progress_v1"
MAX_VISIBLE_CHANGED_PATHS_BYTES = 1024


class ProfileResolver(Protocol):
    def resolve(self, instance: Mapping[str, Any]) -> SwesmithProfileBinding: ...


class SandboxFactory(Protocol):
    def __call__(
        self,
        record: SwesmithRecord,
        profile: SwesmithProfileBinding,
    ) -> LinuxNamespaceEpisodeSandbox: ...


@dataclass(frozen=True)
class EpisodeStep:
    observation: str
    reward: float
    done: bool
    info: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "state": self.observation,
            "reward": self.reward,
            "done": self.done,
            "info": dict(self.info),
        }


@dataclass
class _Episode:
    slot_id: int
    audit_id: str
    started_at: str
    record: SwesmithRecord
    profile: SwesmithProfileBinding
    workspace: SwesmithWorkspace
    sandbox: LinuxNamespaceEpisodeSandbox
    observation: str
    initial_observation: str
    step_count: int = 0
    done: bool = False
    reward: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    grade: SwesmithGradeResult | None = None


@dataclass
class _Slot:
    episode: _Episode | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _actor_credit(positive_eligible: bool, basis: str) -> dict[str, Any]:
    return {
        "schema": ACTOR_CREDIT_SCHEMA,
        "positive_eligible": positive_eligible,
        "basis": basis,
    }


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shell_action_progress(
    action: ParsedPolicyAction,
    *,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    workspace_diff = result.get("workspace_diff")
    if not isinstance(workspace_diff, Mapping):
        raise RuntimeError("SWE-smith shell result lost its workspace diff")
    action_payload = {
        "kind": action.kind,
        "arguments": dict(action.arguments or {}),
    }
    result_payload = {
        key: result.get(key)
        for key in (
            "exit_code",
            "timed_out",
            "stdout",
            "stderr",
            "stdout_truncated",
            "stderr_truncated",
            "termination_reason",
        )
    }
    result_payload.update(
        {
            "before_tree_sha256": workspace_diff.get("before_tree_sha256"),
            "after_tree_sha256": workspace_diff.get("after_tree_sha256"),
        }
    )
    return {
        "schema": ACTION_PROGRESS_SCHEMA,
        "action_fingerprint": _stable_json_sha256(action_payload),
        "result_fingerprint": _stable_json_sha256(result_payload),
        "workspace_changed": bool(workspace_diff.get("changed_paths")),
    }


class SwesmithEpisodeManager:
    """Own one persistent repository workspace per native SWE-smith episode."""

    def __init__(
        self,
        *,
        dataset: SwesmithDataset,
        materializer: SwesmithWorkspaceMaterializer,
        profile_resolver: ProfileResolver,
        sandbox_factory: SandboxFactory,
        grader: SwesmithHiddenGrader,
        audit_sink: SwesmithEpisodeAuditSink | None = None,
        max_steps: int = DEFAULT_TRAINING_MAX_POLICY_TURNS,
        max_observation_bytes: int = DEFAULT_MAX_OBSERVATION_BYTES,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("SWE-smith max_steps must be a positive integer")
        if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
            raise ValueError(
                "SWE-smith max_observation_bytes must be a positive integer"
            )
        self.dataset = dataset
        self.materializer = materializer
        self.profile_resolver = profile_resolver
        self.sandbox_factory = sandbox_factory
        self.grader = grader
        self.audit_sink = audit_sink
        self.max_steps = max_steps
        self.max_observation_bytes = max_observation_bytes
        self.runtime_metadata = dict(runtime_metadata or {})
        self._slots: dict[int, _Slot] = {}
        self._next_slot = 0
        self._slots_lock = threading.Lock()

    def create(self) -> int:
        with self._slots_lock:
            slot_id = self._next_slot
            self._next_slot += 1
            self._slots[slot_id] = _Slot()
        return slot_id

    def reset(self, slot_id: int, data_idx: int) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            self._close_episode(slot, close_reason="reset")
            record = self.dataset[data_idx]
            profile = self.profile_resolver.resolve(record.instance)
            sandbox: LinuxNamespaceEpisodeSandbox | None = None
            workspace: SwesmithWorkspace | None = None
            try:
                sandbox = self.sandbox_factory(record, profile)
                workspace = self.materializer.materialize(
                    record.instance,
                    test_paths=profile.all_test_paths,
                    model_uid=sandbox.model_uid,
                    model_gid=sandbox.model_gid,
                )
                initial_snapshot = sandbox.attach_workspace(workspace.policy_root)
                observation = _initial_observation(record.problem_statement)
                episode = _Episode(
                    slot_id=slot_id,
                    audit_id=uuid.uuid4().hex,
                    started_at=_utc_now(),
                    record=record,
                    profile=profile,
                    workspace=workspace,
                    sandbox=sandbox,
                    observation=observation,
                    initial_observation=observation,
                )
                episode.evidence.append(
                    {
                        "event": "reset",
                        "slot_id": slot_id,
                        "audit_id": episode.audit_id,
                        "data_idx": record.data_idx,
                        "physical_index": record.physical_index,
                        "instance_id": record.instance_id,
                        "dataset_shard_sha256": record.shard_sha256,
                        "profile": profile.as_private_metadata(),
                        "workspace_contract": workspace.contract,
                        "workspace_initial": initial_snapshot.as_summary(),
                        "sandbox": dict(sandbox.metadata),
                        "observation": observation,
                    }
                )
                slot.episode = episode
                return self._public_step(episode, action_kind="reset")
            except Exception:
                if workspace is not None:
                    self.materializer.close(workspace)
                if sandbox is not None:
                    sandbox.close()
                raise

    def step(self, slot_id: int, raw_output: str) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            episode = self._episode(slot)
            if episode.done:
                raise RuntimeError("SWE-smith episode is already terminal")
            episode.step_count += 1
            observation_before = episode.observation
            action = parse_policy_action(raw_output)
            evidence: dict[str, Any] = {
                "event": "policy_step",
                "step": episode.step_count,
                "observation_before": observation_before,
                "action": action.as_evidence(),
            }
            public_action_kind = action.kind
            if action.kind == "parser_error":
                episode.observation = _parser_error_observation(action)
                evidence["result"] = {"parser_error": action.error}
                actor_credit = _actor_credit(False, "parser_rejected")
            elif action.kind == "shell_command":
                actor_credit = self._run_shell(episode, action, evidence)
                if actor_credit["basis"] == "terminal_submission":
                    public_action_kind = "final"
            elif action.kind == "apply_patch":
                actor_credit = self._run_patch(episode, action, evidence)
            elif action.kind == "final":
                self._grade_terminal(episode, evidence, termination_reason="final")
                actor_credit = _actor_credit(True, "terminal_submission")
            else:  # pragma: no cover - action parser owns the closed kind set.
                raise RuntimeError(f"unsupported SWE-smith action kind: {action.kind}")
            evidence["actor_credit"] = dict(actor_credit)
            evidence["observation_after"] = episode.observation
            episode.evidence.append(evidence)

            if not episode.done and episode.step_count >= self.max_steps:
                horizon: dict[str, Any] = {
                    "event": "horizon_exhaustion",
                    "step": episode.step_count,
                    "observation_before": episode.observation,
                    "action": {"kind": "horizon"},
                }
                self._terminate_without_submission(
                    episode,
                    horizon,
                    termination_reason="max_steps",
                )
                horizon["observation_after"] = episode.observation
                episode.evidence.append(horizon)
            return self._public_step(
                episode,
                action_kind=public_action_kind,
                actor_credit=actor_credit,
                action_progress=evidence.get("action_progress"),
            )

    def observation(self, slot_id: int) -> str:
        slot = self._slot(slot_id)
        with slot.lock:
            return self._episode(slot).observation

    def finalize_horizon(self, slot_id: int) -> EpisodeStep:
        """Fail the episode when the unified policy budget expires.

        The rollout owns the unified policy-step counter because model-authored
        compactions consume steps without calling this native environment. This
        control call therefore does not increment the native action counter or
        invoke the hidden grader.  This matches the upstream agent's
        ``LimitsExceeded`` path: an episode without an explicit submission is
        a terminal failure, even when the current workspace would pass.
        """

        slot = self._slot(slot_id)
        with slot.lock:
            episode = self._episode(slot)
            if episode.done:
                raise RuntimeError("SWE-smith episode is already terminal")
            horizon: dict[str, Any] = {
                "event": "horizon_exhaustion",
                "step": episode.step_count,
                "observation_before": episode.observation,
                "action": {"kind": "policy_turn_horizon"},
            }
            self._terminate_without_submission(
                episode,
                horizon,
                termination_reason="policy_turn_horizon",
            )
            horizon["observation_after"] = episode.observation
            episode.evidence.append(horizon)
            return self._public_step(episode, action_kind="policy_turn_horizon")

    def detail(self, slot_id: int) -> dict[str, Any]:
        slot = self._slot(slot_id)
        with slot.lock:
            episode = self._episode(slot)
            return {
                "schema": EPISODE_SCHEMA,
                "slot_id": episode.slot_id,
                "audit_id": episode.audit_id,
                "started_at": episode.started_at,
                "data_idx": episode.record.data_idx,
                "physical_index": episode.record.physical_index,
                "instance_id": episode.record.instance_id,
                "step_count": episode.step_count,
                "done": episode.done,
                "reward": episode.reward,
                "workspace": {
                    "episode_root": str(episode.workspace.episode_root),
                    "policy_root": str(episode.workspace.policy_root),
                    "model_uid": episode.sandbox.model_uid,
                    "model_gid": episode.sandbox.model_gid,
                },
                "evidence": list(episode.evidence),
                "grade": (
                    None if episode.grade is None else episode.grade.as_private_dict()
                ),
            }

    def close(self, slot_id: int) -> dict[str, Any]:
        slot = self._slot(slot_id)
        with slot.lock:
            self._close_episode(slot, close_reason="client_close")
        with self._slots_lock:
            self._slots.pop(slot_id, None)
        return {"closed": True, "id": slot_id}

    def metadata(self) -> dict[str, Any]:
        provenance = self.dataset.provenance
        slot_count, active_count = self._active_counts()
        metadata = {
            "schema": EPISODE_SCHEMA,
            "task_count": len(self.dataset),
            "dataset_id": provenance.dataset_id,
            "dataset_role": provenance.role,
            "upstream_repository": provenance.upstream_repository,
            "dataset_revision": provenance.upstream_revision,
            # Compatibility alias; new launchers should use dataset_revision.
            "upstream_revision": provenance.upstream_revision,
            "dataset_manifest_sha256": provenance.manifest_sha256,
            "selection_mode": provenance.selection_mode,
            "max_steps": self.max_steps,
            "configured_max_policy_turns": self.max_steps,
            "training_max_policy_turns": DEFAULT_TRAINING_MAX_POLICY_TURNS,
            "upstream_reference_max_policy_turns": (
                UPSTREAM_REFERENCE_MAX_POLICY_TURNS
            ),
            "upstream_agent_repository": UPSTREAM_AGENT_REPOSITORY,
            "upstream_agent_revision": UPSTREAM_AGENT_REVISION,
            "tool_contract": "codex_shell_command_apply_patch_v1",
            "tool_serialization": "qwen35_native_single_function_v1",
            "observation_contract": "bounded_combined_shell_output_v1",
            "max_observation_bytes": self.max_observation_bytes,
            "reward_contract": REWARD_CONTRACT,
            "context_contract": "one_native_issue_continuous_episode_v1",
            "memory_contract": (
                "policy_compaction_plus_optional_durable_filesystem_v1"
            ),
            "submission_contract": SUBMISSION_CONTRACT,
            "horizon_contract": HORIZON_CONTRACT,
            "active_slot_count": slot_count,
            "active_environment_count": active_count,
            "active_workspace_count": active_count,
            "private_audit_contract": (
                "agentmemory_swesmith_private_episode_audit_v1"
                if self.audit_sink is not None
                else "disabled"
            ),
        }
        metadata.update(self.runtime_metadata)
        return metadata

    def _active_counts(self) -> tuple[int, int]:
        with self._slots_lock:
            slots = list(self._slots.values())
        active_count = 0
        for slot in slots:
            with slot.lock:
                active_count += int(slot.episode is not None)
        return len(slots), active_count

    def _run_shell(
        self,
        episode: _Episode,
        action: ParsedPolicyAction,
        evidence: dict[str, Any],
    ) -> Mapping[str, Any]:
        assert action.arguments is not None
        timeout_ms = action.arguments.get(
            "timeout_ms", episode.sandbox.limits.default_timeout_ms
        )
        try:
            execution = episode.sandbox.run(
                command=str(action.arguments["command"]),
                workdir=str(action.arguments["workdir"]),
                timeout_ms=int(timeout_ms),
            )
        except SwesmithSandboxError as exc:
            poisoned = episode.sandbox.poisoned_reason
            evidence["result"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "poisoned": poisoned,
            }
            episode.observation = f"shell_command failed: {exc}"
            if poisoned is not None:
                episode.done = True
                episode.reward = 0.0
            return _actor_credit(False, "executor_rejected")
        result = execution.result
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        evidence["result"] = {
            "exit_code": result.exit_code,
            "elapsed_ms": result.elapsed_ms,
            "timed_out": result.timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "observation_output_budget_bytes": self.max_observation_bytes,
            "observation_output_truncated": bool(
                result.stdout_truncated or result.stderr_truncated
            ),
            "termination_reason": result.termination_reason,
            "workspace_diff": execution.workspace_diff.as_dict(),
        }
        evidence["action_progress"] = _shell_action_progress(
            action,
            result=evidence["result"],
        )
        episode.observation = _shell_observation(
            exit_code=result.exit_code,
            elapsed_ms=result.elapsed_ms,
            timed_out=result.timed_out,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            changed_paths=execution.workspace_diff.changed_paths,
            max_observation_bytes=self.max_observation_bytes,
        )
        submission = _submission_from_shell_result(
            stdout=stdout,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
        )
        if submission is not None:
            evidence["submission"] = {
                "sentinel": UPSTREAM_SUBMISSION_SENTINEL,
                "output_bytes": len(submission.encode("utf-8")),
                "output_sha256": hashlib.sha256(
                    submission.encode("utf-8")
                ).hexdigest(),
            }
            self._grade_terminal(
                episode,
                evidence,
                termination_reason="submission_sentinel",
            )
            return _actor_credit(True, "terminal_submission")
        return _actor_credit(True, "shell_executed")

    def _run_patch(
        self,
        episode: _Episode,
        action: ParsedPolicyAction,
        evidence: dict[str, Any],
    ) -> Mapping[str, Any]:
        assert action.patch is not None
        try:
            operations = parse_workspace_patch(action.patch)
            result = apply_workspace_patch_touched_transaction(
                episode.workspace.policy_root,
                operations,
                normalize_path=_normalize_patch_path,
                validate_tree=lambda root: snapshot_workspace_tree(
                    root, episode.sandbox.limits
                ),
            )
            diff = episode.sandbox.refresh_after_host_mutation()
        except (WorkspacePatchError, SwesmithSandboxError) as exc:
            evidence["result"] = {"error": f"{type(exc).__name__}: {exc}"}
            episode.observation = f"apply_patch failed: {exc}"
            if episode.sandbox.poisoned_reason is not None:
                episode.done = True
                episode.reward = 0.0
            return _actor_credit(False, "executor_rejected")
        evidence["result"] = {
            "changed_paths": list(result.changed_paths),
            "added_paths": list(result.added_paths),
            "updated_paths": list(result.updated_paths),
            "deleted_paths": list(result.deleted_paths),
            "workspace_diff": diff.as_dict(),
        }
        episode.observation = (
            "apply_patch succeeded. Changed paths: "
            + (", ".join(result.changed_paths) if result.changed_paths else "none")
        )
        if not diff.changed_paths:
            return _actor_credit(False, "no_workspace_change")
        return _actor_credit(True, "workspace_changed")

    def _grade_terminal(
        self,
        episode: _Episode,
        evidence: dict[str, Any],
        *,
        termination_reason: str,
    ) -> None:
        grade = self.grader.grade(
            instance=episode.record.instance,
            profile=episode.profile,
            workspace=episode.workspace,
            sandbox=episode.sandbox,
        )
        episode.grade = grade
        episode.done = True
        episode.reward = grade.reward
        episode.observation = (
            "Submission accepted and graded. "
            + ("The issue is resolved." if grade.resolved else "The issue is not resolved.")
        )
        evidence["termination_reason"] = termination_reason
        evidence["terminal_grade"] = {
            "reward": grade.reward,
            "resolved": grade.resolved,
            "grader_error": grade.error,
        }

    def _terminate_without_submission(
        self,
        episode: _Episode,
        evidence: dict[str, Any],
        *,
        termination_reason: str,
    ) -> None:
        episode.done = True
        episode.reward = 0.0
        episode.observation = (
            "Episode ended without the explicit submission sentinel. "
            "The workspace was not graded."
        )
        evidence["termination_reason"] = termination_reason
        evidence["terminal_grade"] = {
            "reward": 0.0,
            "resolved": False,
            "grader_error": None,
            "graded": False,
        }

    def _public_step(
        self,
        episode: _Episode,
        *,
        action_kind: str,
        actor_credit: Mapping[str, Any] | None = None,
        action_progress: Mapping[str, Any] | None = None,
    ) -> EpisodeStep:
        info: dict[str, Any] = {
            "schema": EPISODE_SCHEMA,
            "step": episode.step_count,
            "action_kind": action_kind,
            "terminal": episode.done,
            "episode_success": bool(
                episode.done
                and episode.grade is not None
                and episode.grade.resolved
            ),
        }
        if actor_credit is not None:
            info["actor_credit"] = dict(actor_credit)
        if action_progress is not None:
            info["action_progress"] = dict(action_progress)
        return EpisodeStep(
            observation=episode.observation,
            reward=episode.reward,
            done=episode.done,
            info=info,
        )

    def _slot(self, slot_id: int) -> _Slot:
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            raise KeyError("SWE-smith slot id must be an integer")
        with self._slots_lock:
            try:
                return self._slots[slot_id]
            except KeyError as exc:
                raise KeyError(f"unknown SWE-smith slot id: {slot_id}") from exc

    @staticmethod
    def _episode(slot: _Slot) -> _Episode:
        if slot.episode is None:
            raise RuntimeError("SWE-smith slot must be reset before use")
        return slot.episode

    def _close_episode(self, slot: _Slot, *, close_reason: str) -> None:
        episode = slot.episode
        slot.episode = None
        if episode is None:
            return
        try:
            if self.audit_sink is not None:
                self.audit_sink.write(
                    audit_id=episode.audit_id,
                    payload={
                        "episode_schema": EPISODE_SCHEMA,
                        "closed_at": _utc_now(),
                        "close_reason": close_reason,
                        "slot_id": episode.slot_id,
                        "started_at": episode.started_at,
                        "data_idx": episode.record.data_idx,
                        "physical_index": episode.record.physical_index,
                        "instance_id": episode.record.instance_id,
                        "dataset_shard_sha256": episode.record.shard_sha256,
                        "problem_statement": episode.record.problem_statement,
                        "initial_observation": episode.initial_observation,
                        "step_count": episode.step_count,
                        "done": episode.done,
                        "reward": episode.reward,
                        "runtime_metadata": dict(self.runtime_metadata),
                        "evidence": list(episode.evidence),
                        "grade": (
                            None
                            if episode.grade is None
                            else episode.grade.as_private_dict()
                        ),
                    },
                )
        finally:
            try:
                episode.sandbox.close()
            finally:
                self.materializer.close(episode.workspace)


def _normalize_patch_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise WorkspacePatchError("apply_patch path must be non-empty text")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspacePatchError(f"apply_patch path escapes the workspace: {raw!r}")
    return str(path)


def _submission_from_shell_result(
    *, stdout: str, exit_code: int, timed_out: bool
) -> str | None:
    if timed_out or exit_code != 0:
        return None
    lines = stdout.lstrip().splitlines(keepends=True)
    if not lines or lines[0].strip() != UPSTREAM_SUBMISSION_SENTINEL:
        return None
    return "".join(lines[1:])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_observation(problem_statement: str) -> str:
    return (
        "Repair the persistent repository in /testbed for this issue. Use exactly one "
        "action per turn. Use shell_command for inspection, editing, and tests. Its exact "
        "form is one line such as shell_command "
        '{"command":"ls","workdir":"."}. Start at byte zero and output only the '
        "action, without XML, prose, Markdown, or a <think> tag. apply_patch is optional "
        "and starts with the literal line apply_patch followed by one complete "
        "*** Begin Patch ... *** End Patch payload. Keep edits localized: never paste or "
        "rewrite an entire existing file in one action. Use a small patch around the "
        "changed lines or a bounded shell command, and stay below the response limit. "
        "The workspace persists for the whole episode and has no .git directory. After a "
        "source path changed and relevant tests ran, submit with exactly "
        'shell_command {"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",'
        '"workdir":"."}. The successful command must print the upstream submission '
        "sentinel as its first stdout line; then the current persistent workspace receives "
        "one official grade. Any plain text is invalid. Reaching the turn limit without "
        "that sentinel ends the episode with reward 0 and does not grade the workspace.\n\n"
        "Issue:\n"
        + problem_statement.strip()
        + "\n\nBegin with a real shell action in the exact one-line form above."
    )


def _parser_error_observation(action: ParsedPolicyAction) -> str:
    return (
        f"Invalid action syntax: {action.error}. Start at byte zero and retry exactly "
        'like shell_command {"command":"pwd","workdir":"."} on one line. '
        "For a patch, start with the literal line apply_patch, then one complete "
        "*** Begin Patch ... *** End Patch payload. Output only one action, with no XML "
        "tags, reasoning, Markdown, second action, or surrounding text. Keep the edit "
        "localized instead of pasting or rewriting an entire existing file. After editing "
        "and testing, submit with exactly shell_command "
        '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}.'
    )


def _shell_observation(
    *,
    exit_code: int,
    elapsed_ms: int,
    timed_out: bool,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
    changed_paths: tuple[str, ...],
    max_observation_bytes: int = DEFAULT_MAX_OBSERVATION_BYTES,
) -> str:
    if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
        raise ValueError("max_observation_bytes must be a positive integer")
    # Bound both command output and the workspace-diff receipt.  The complete
    # diff remains in the private audit; the policy only needs a compact signal
    # that its workspace changed and a few paths for orientation.
    stdout_limit = max(1, max_observation_bytes // 2)
    stderr_limit = max(1, max_observation_bytes - stdout_limit)
    visible_stdout, stdout_was_bounded = _bound_shell_output(
        stdout,
        limit=stdout_limit,
        truncated=stdout_truncated,
        label="stdout",
    )
    visible_stderr, stderr_was_bounded = _bound_shell_output(
        stderr,
        limit=stderr_limit,
        truncated=stderr_truncated,
        label="stderr",
    )
    return (
        f"shell_command exit_code={exit_code} elapsed_ms={elapsed_ms} "
        f"timed_out={str(timed_out).lower()} "
        f"stdout_truncated={str(stdout_truncated).lower()} "
        f"stderr_truncated={str(stderr_truncated).lower()} "
        f"visible_output_truncated={str(stdout_was_bounded or stderr_was_bounded).lower()} "
        f"visible_output_budget_bytes={max_observation_bytes}\n"
        "stdout:\n"
        + (visible_stdout if visible_stdout else "<empty>")
        + "\nstderr:\n"
        + (visible_stderr if visible_stderr else "<empty>")
        + "\nworkspace changed paths: "
        + _changed_paths_observation(changed_paths)
    )


def _changed_paths_observation(
    changed_paths: tuple[str, ...],
    *,
    max_bytes: int = MAX_VISIBLE_CHANGED_PATHS_BYTES,
) -> str:
    """Expose a bounded diff summary while keeping exact paths private."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not changed_paths:
        return "none"

    encoded_paths = [path.encode("utf-8", errors="replace") for path in changed_paths]
    digest = hashlib.sha256(b"\n".join(encoded_paths)).hexdigest()
    prefix = f"{len(changed_paths)} paths; sha256={digest}; sample="
    suffix = "; sample_truncated=true"
    # Keep the summary itself bounded even when one path is unusually long.
    budget = max(0, max_bytes - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8")))
    selected: list[str] = []
    used = 2  # the surrounding brackets
    candidates = list(dict.fromkeys((changed_paths[0], changed_paths[-1])))
    for path in candidates:
        rendered = path.replace("\\", "/")
        token = rendered if not selected else ", " + rendered
        token_bytes = len(token.encode("utf-8", errors="replace"))
        if used + token_bytes > budget:
            continue
        selected.append(rendered)
        used += token_bytes
    sample = "[" + ", ".join(selected) + "]"
    omitted = len(changed_paths) - len(selected)
    result = prefix + sample
    if omitted:
        result += f"; omitted={omitted}"
    if len(result.encode("utf-8", errors="replace")) > max_bytes:
        result = (prefix + suffix).encode("utf-8", errors="replace")[:max_bytes].decode(
            "utf-8", errors="ignore"
        )
    return result


def _bound_shell_output(
    text: str,
    *,
    limit: int,
    truncated: bool,
    label: str,
) -> tuple[str, bool]:
    """Keep visible tool output bounded while retaining both ends of a log."""

    raw = text.encode("utf-8", errors="replace")
    marker = f"\n[{label} truncated: visible output budget reached]\n".encode(
        "utf-8"
    )
    if not truncated and len(raw) <= limit:
        return text, False
    if limit <= len(marker):
        return marker[:limit].decode("utf-8", errors="ignore"), True
    payload_limit = limit - len(marker)
    head_limit = (payload_limit + 1) // 2
    tail_limit = payload_limit - head_limit
    head = raw[:head_limit].decode("utf-8", errors="ignore")
    tail = raw[-tail_limit:].decode("utf-8", errors="ignore") if tail_limit else ""
    return head + marker.decode("utf-8") + tail, True
