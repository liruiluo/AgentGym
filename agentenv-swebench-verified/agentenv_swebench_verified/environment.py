from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol

from agentenv_agentmemory.workspace_patch import (
    WorkspacePatchError,
    apply_workspace_patch_touched_transaction,
    parse_workspace_patch,
)
from agentenv_swesmith.actions import ParsedPolicyAction, parse_policy_action
from agentenv_swesmith.sandbox import (
    SwesmithSandboxError,
    snapshot_workspace_tree,
)

from .dataset import VerifiedDataset, VerifiedRecord
from .exporter import (
    MAX_MODEL_PATCH_BYTES,
    PATCH_EXPORT_CONTRACT,
    PredictionStore,
    RunCapabilityMismatch,
    SolutionPatchExporter,
    validate_run_id,
)
from .protocol import (
    ARMS,
    EVALUATION_MAX_POLICY_TURNS,
    FORBIDDEN_POLICY_FIELDS,
    MODEL_LABELS,
    POLICY_FIELDS,
    require_arm,
)
from .sandbox import VerifiedLinuxNamespaceEpisodeSandbox
from .testspec import VerifiedTestSpecBinding
from .workspace import VerifiedWorkspace, VerifiedWorkspaceMaterializer


EPISODE_SCHEMA = "swebench_verified_external_patch_episode_v1"
DEFAULT_MAX_OBSERVATION_BYTES = 6144
OBSERVATION_CONTRACT = "bounded_policy_observation_v1"
ACTOR_CREDIT_SCHEMA = "task_neutral_actor_credit_v1"
ACTION_PROGRESS_SCHEMA = "swebench_verified_action_progress_v1"
RUN_CAPABILITY_CONTRACT = "caller_supplied_run_bearer_first_claim_v1"
_RUN_CAPABILITY_RE = re.compile(r"\A[A-Za-z0-9_-]{43,128}\Z")


class TestSpecResolver(Protocol):
    def resolve(self, instance: Mapping[str, Any]) -> VerifiedTestSpecBinding: ...


class SandboxFactory(Protocol):
    def __call__(
        self,
        record: VerifiedRecord,
        binding: VerifiedTestSpecBinding,
    ) -> VerifiedLinuxNamespaceEpisodeSandbox: ...


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
    record: VerifiedRecord
    binding: VerifiedTestSpecBinding
    workspace: VerifiedWorkspace
    sandbox: VerifiedLinuxNamespaceEpisodeSandbox
    observation: str
    step_count: int = 0
    done: bool = False
    prediction: dict[str, str] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Slot:
    arm: str
    run_id: str
    capability: str = field(repr=False)
    episode: _Episode | None = None
    closed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class VerifiedEpisodeManager:
    """Own persistent policy workspaces and export patches for external grading."""

    def __init__(
        self,
        *,
        dataset: VerifiedDataset,
        materializer: VerifiedWorkspaceMaterializer,
        testspec_resolver: TestSpecResolver,
        sandbox_factory: SandboxFactory,
        exporter: SolutionPatchExporter,
        prediction_store: PredictionStore,
        max_native_actions: int = EVALUATION_MAX_POLICY_TURNS,
        max_observation_bytes: int = DEFAULT_MAX_OBSERVATION_BYTES,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if type(max_native_actions) is not int or max_native_actions <= 0:
            raise ValueError("max_native_actions must be a positive integer")
        if max_native_actions != EVALUATION_MAX_POLICY_TURNS:
            raise ValueError(
                "Verified external evaluation requires the frozen 250-turn cap"
            )
        if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
            raise ValueError("max_observation_bytes must be a positive integer")
        if prediction_store.instance_ids != dataset.instance_ids:
            raise ValueError("prediction store task order disagrees with the dataset")
        self.dataset = dataset
        self.materializer = materializer
        self.testspec_resolver = testspec_resolver
        self.sandbox_factory = sandbox_factory
        self.exporter = exporter
        self.prediction_store = prediction_store
        self.max_native_actions = max_native_actions
        self.max_observation_bytes = max_observation_bytes
        self.runtime_metadata = dict(runtime_metadata or {})
        self._slots: dict[int, _Slot] = {}
        self._next_slot = 0
        self._slots_lock = threading.Lock()

    def create(self, *, arm: str, run_id: str, run_capability: str) -> int:
        normalized_arm = require_arm(arm)
        normalized_run = validate_run_id(run_id)
        normalized_capability = require_run_capability(run_capability)
        capability_digest = hashlib.sha256(
            normalized_capability.encode("ascii")
        ).digest()
        with self._slots_lock:
            try:
                self.prediction_store.claim_run(
                    arm=normalized_arm,
                    run_id=normalized_run,
                    capability_digest=capability_digest,
                )
            except RunCapabilityMismatch as exc:
                raise PermissionError("run authorization failed") from exc
            slot_id = self._next_slot
            self._next_slot += 1
            self._slots[slot_id] = _Slot(
                normalized_arm,
                normalized_run,
                secrets.token_urlsafe(32),
            )
        return slot_id

    def capability(self, slot_id: int) -> str:
        return self._slot(slot_id).capability

    def authorize(
        self,
        slot_id: int,
        capability: str,
        *,
        arm: str | None = None,
        run_id: str | None = None,
    ) -> None:
        try:
            slot = self._slot(slot_id)
        except KeyError as exc:
            raise PermissionError("slot authorization failed") from exc
        checks = [
            isinstance(capability, str)
            and hmac.compare_digest(slot.capability, capability)
        ]
        if arm is not None:
            checks.append(
                isinstance(arm, str) and hmac.compare_digest(slot.arm, arm)
            )
        if run_id is not None:
            checks.append(
                isinstance(run_id, str)
                and hmac.compare_digest(slot.run_id, run_id)
            )
        if not all(checks):
            raise PermissionError("slot authorization failed")

    def reset(self, slot_id: int, data_idx: int) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            self._require_slot_open(slot)
            self._close_episode(slot, export_if_needed=True)
            record = self.dataset[data_idx]
            binding = self.testspec_resolver.resolve(record.private_instance())
            sandbox: VerifiedLinuxNamespaceEpisodeSandbox | None = None
            workspace: VerifiedWorkspace | None = None
            try:
                sandbox = self.sandbox_factory(record, binding)
                workspace = self.materializer.materialize(
                    record.policy_instance,
                    model_uid=sandbox.model_uid,
                    model_gid=sandbox.model_gid,
                )
                initial = sandbox.attach_workspace(workspace.policy_root)
                observation = initial_observation(record.problem_statement)
                episode = _Episode(
                    record=record,
                    binding=binding,
                    workspace=workspace,
                    sandbox=sandbox,
                    observation=observation,
                )
                episode.evidence.append(
                    {
                        "event": "reset",
                        "data_idx": data_idx,
                        "instance_id": record.instance_id,
                        "workspace_tree_sha256": initial.tree_sha256,
                        "sandbox_contract": sandbox.metadata["contract"],
                    }
                )
                slot.episode = episode
                return self._public_step(slot, episode, action_kind="reset")
            except Exception:
                if workspace is not None:
                    self.materializer.close(workspace)
                if sandbox is not None:
                    sandbox.close()
                raise

    def step(self, slot_id: int, raw_output: str) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            self._require_slot_open(slot)
            episode = self._episode(slot)
            if episode.done:
                raise RuntimeError("Verified episode is already terminal")
            episode.step_count += 1
            action = parse_policy_action(raw_output)
            evidence: dict[str, Any] = {
                "event": "native_action",
                "step": episode.step_count,
                "action_kind": action.kind,
            }
            if action.kind == "parser_error":
                episode.observation = parser_error_observation(action)
                actor_credit = build_actor_credit(False, "parser_rejected")
            elif action.kind == "shell_command":
                actor_credit = self._run_shell(episode, action, evidence)
            elif action.kind == "apply_patch":
                actor_credit = self._run_patch(episode, action, evidence)
            elif action.kind == "final":
                self._export_terminal(slot, episode, reason="final")
                actor_credit = build_actor_credit(True, "terminal_submission")
            else:  # pragma: no cover
                raise RuntimeError(f"unsupported action kind: {action.kind}")
            evidence["actor_credit"] = dict(actor_credit)
            episode.evidence.append(evidence)
            if not episode.done and episode.step_count >= self.max_native_actions:
                self._export_terminal(slot, episode, reason="native_action_cap")
            return self._public_step(
                slot,
                episode,
                action_kind=action.kind,
                actor_credit=actor_credit,
                action_progress=evidence.get("action_progress"),
            )

    def observation(self, slot_id: int) -> str:
        slot = self._slot(slot_id)
        with slot.lock:
            self._require_slot_open(slot)
            return self._episode(slot).observation

    def finalize_horizon(self, slot_id: int) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            self._require_slot_open(slot)
            episode = self._episode(slot)
            if episode.done:
                raise RuntimeError("Verified episode is already terminal")
            self._export_terminal(slot, episode, reason="unified_policy_horizon")
            return self._public_step(
                slot, episode, action_kind="unified_policy_horizon"
            )

    def prediction(self, slot_id: int) -> dict[str, str]:
        slot = self._slot(slot_id)
        with slot.lock:
            self._require_slot_open(slot)
            episode = self._episode(slot)
            if episode.prediction is None:
                raise RuntimeError("prediction is unavailable before terminal export")
            return dict(episode.prediction)

    def assemble_predictions(self, *, arm: str, run_id: str) -> dict[str, Any]:
        path = self.prediction_store.assemble(arm=arm, run_id=run_id)
        digest = hashlib.sha256()
        row_count = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                row_count += chunk.count(b"\n")
        return {
            "assembled": True,
            "arm": require_arm(arm),
            "run_id": validate_run_id(run_id),
            "model_name_or_path": MODEL_LABELS[arm],
            "row_count": row_count,
            "sha256": digest.hexdigest(),
        }

    def close(self, slot_id: int) -> dict[str, Any]:
        slot = self._slot(slot_id)
        try:
            with slot.lock:
                self._require_slot_open(slot)
                slot.closed = True
                self._close_episode(slot, export_if_needed=True)
        finally:
            with self._slots_lock:
                self._slots.pop(slot_id, None)
        return {"closed": True, "id": slot_id}

    def metadata(self) -> dict[str, Any]:
        with self._slots_lock:
            slots = tuple(self._slots.values())
        active = sum(slot.episode is not None for slot in slots)
        metadata = {
            "schema": EPISODE_SCHEMA,
            "task_count": len(self.dataset),
            "full_benchmark_task_count": 500,
            "supported_arms": list(ARMS),
            "model_labels": dict(MODEL_LABELS),
            "policy_visible_fields": list(POLICY_FIELDS),
            "denied_grader_fields": sorted(FORBIDDEN_POLICY_FIELDS),
            "tool_contract": "codex_shell_command_apply_patch_v1",
            "tool_serialization": "qwen35_native_single_function_v1",
            "observation_contract": OBSERVATION_CONTRACT,
            "max_observation_bytes": self.max_observation_bytes,
            "evaluation_max_policy_turns": EVALUATION_MAX_POLICY_TURNS,
            "max_native_actions": self.max_native_actions,
            "compaction_consumes_policy_turn": True,
            "compaction_consumes_native_call": False,
            "run_capability_contract": RUN_CAPABILITY_CONTRACT,
            "reward_contract": "external_official_grading_only",
            "patch_export_contract": PATCH_EXPORT_CONTRACT,
            "max_model_patch_bytes": MAX_MODEL_PATCH_BYTES,
            "prediction_contract": self.prediction_store.public_metadata(),
            "dataset": self.dataset.provenance.public_metadata(),
            "active_slot_count": len(slots),
            "active_workspace_count": active,
            "official_grading_inside_adapter": False,
        }
        metadata.update(self.runtime_metadata)
        return metadata

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
            episode.observation = f"shell_command failed: {exc}"
            evidence["error"] = type(exc).__name__
            if episode.sandbox.poisoned_reason is not None:
                episode.observation += " The workspace was sealed for export."
            return build_actor_credit(False, "executor_rejected")
        result = execution.result
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        result_evidence = {
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "termination_reason": result.termination_reason,
            "workspace_diff": execution.workspace_diff.as_dict(),
        }
        evidence["action_progress"] = shell_action_progress(
            action, result=result_evidence
        )
        episode.observation = shell_observation(
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
        return build_actor_credit(True, "shell_executed")

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
                normalize_path=normalize_patch_path,
                validate_tree=lambda root: snapshot_workspace_tree(
                    root, episode.sandbox.limits
                ),
            )
            diff = episode.sandbox.refresh_after_host_mutation()
        except (WorkspacePatchError, SwesmithSandboxError) as exc:
            episode.observation = f"apply_patch failed: {exc}"
            evidence["error"] = type(exc).__name__
            return build_actor_credit(False, "executor_rejected")
        evidence["changed_paths"] = list(result.changed_paths)
        episode.observation = (
            "apply_patch succeeded. Changed paths: "
            + (", ".join(result.changed_paths) if result.changed_paths else "none")
        )
        if not diff.changed_paths:
            return build_actor_credit(False, "no_workspace_change")
        return build_actor_credit(True, "workspace_changed")

    def _export_terminal(
        self, slot: _Slot, episode: _Episode, *, reason: str
    ) -> None:
        row = self.exporter.prediction_row(episode.workspace, arm=slot.arm)
        self.prediction_store.write(
            arm=slot.arm,
            run_id=slot.run_id,
            data_idx=episode.record.data_idx,
            row=row,
        )
        episode.prediction = row
        episode.done = True
        episode.observation = (
            "Submission exported. Official SWE-bench grading remains external."
        )
        episode.evidence.append({"event": "export", "reason": reason})

    def _public_step(
        self,
        slot: _Slot,
        episode: _Episode,
        *,
        action_kind: str,
        actor_credit: Mapping[str, Any] | None = None,
        action_progress: Mapping[str, Any] | None = None,
    ) -> EpisodeStep:
        episode.observation = bound_policy_observation(
            episode.observation,
            limit=self.max_observation_bytes,
        )
        info: dict[str, Any] = {
            "schema": EPISODE_SCHEMA,
            "step": episode.step_count,
            "action_kind": action_kind,
            "terminal": episode.done,
            "external_grading_required": episode.done,
            "arm": slot.arm,
        }
        if actor_credit is not None:
            info["actor_credit"] = dict(actor_credit)
        if action_progress is not None:
            info["action_progress"] = dict(action_progress)
        return EpisodeStep(
            observation=episode.observation,
            reward=0.0,
            done=episode.done,
            info=info,
        )

    def _slot(self, slot_id: int) -> _Slot:
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            raise KeyError("slot id must be an integer")
        with self._slots_lock:
            try:
                return self._slots[slot_id]
            except KeyError as exc:
                raise KeyError(f"unknown Verified slot id: {slot_id}") from exc

    @staticmethod
    def _episode(slot: _Slot) -> _Episode:
        if slot.episode is None:
            raise RuntimeError("Verified slot must be reset before use")
        return slot.episode

    @staticmethod
    def _require_slot_open(slot: _Slot) -> None:
        if slot.closed:
            raise RuntimeError("Verified slot is closed")

    def _close_episode(self, slot: _Slot, *, export_if_needed: bool) -> None:
        episode = slot.episode
        slot.episode = None
        if episode is None:
            return
        try:
            if export_if_needed and not episode.done:
                self._export_terminal(slot, episode, reason="lifecycle_close")
        finally:
            try:
                episode.sandbox.close()
            finally:
                self.materializer.close(episode.workspace)


def initial_observation(problem_statement: str) -> str:
    return (
        "Repair the persistent repository in /testbed for this issue. Use exactly one "
        "action per turn. Inspect, edit, and test with shell_command, or use one "
        "bounded apply_patch payload. The exact shell form is "
        'shell_command {"command":"ls","workdir":"."}. The workspace has no '
        ".git directory. Submit a plain final response only when ready to export the "
        "repository diff. Official grading is external.\n\nIssue:\n"
        + problem_statement.strip()
    )


def build_actor_credit(positive_eligible: bool, basis: str) -> dict[str, Any]:
    return {
        "schema": ACTOR_CREDIT_SCHEMA,
        "positive_eligible": positive_eligible,
        "basis": basis,
    }


def require_run_capability(value: str) -> str:
    if not isinstance(value, str) or _RUN_CAPABILITY_RE.fullmatch(value) is None:
        raise PermissionError(
            "run capability must be an unpredictable URL-safe bearer"
        )
    return value


def stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shell_action_progress(
    action: ParsedPolicyAction, *, result: Mapping[str, Any]
) -> dict[str, Any]:
    workspace_diff = result.get("workspace_diff")
    if not isinstance(workspace_diff, Mapping):
        raise RuntimeError("shell result lost its workspace diff")
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
        "action_fingerprint": stable_json_sha256(action_payload),
        "result_fingerprint": stable_json_sha256(result_payload),
        "workspace_changed": bool(workspace_diff.get("changed_paths")),
    }


def normalize_patch_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise WorkspacePatchError("apply_patch path must be non-empty text")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspacePatchError(f"apply_patch path escapes the workspace: {raw!r}")
    return str(path)


def parser_error_observation(action: ParsedPolicyAction) -> str:
    return (
        f"Invalid action syntax: {action.error}. Retry exactly like "
        'shell_command {"command":"pwd","workdir":"."} on one line, or use '
        "one complete apply_patch payload. Output only one action with no XML, "
        "reasoning, Markdown, second action, or surrounding text."
    )


def shell_observation(
    *,
    exit_code: int,
    elapsed_ms: int,
    timed_out: bool,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
    changed_paths: tuple[str, ...],
    max_observation_bytes: int,
) -> str:
    stdout_limit = max(1, max_observation_bytes // 2)
    stderr_limit = max(1, max_observation_bytes - stdout_limit)
    visible_stdout, stdout_bounded = bound_output(
        stdout,
        limit=stdout_limit,
        already_truncated=stdout_truncated,
        label="stdout",
    )
    visible_stderr, stderr_bounded = bound_output(
        stderr,
        limit=stderr_limit,
        already_truncated=stderr_truncated,
        label="stderr",
    )

    def render(visible_output_truncated: bool) -> str:
        return (
            f"shell_command exit_code={exit_code} elapsed_ms={elapsed_ms} "
            f"timed_out={str(timed_out).lower()} "
            f"stdout_truncated={str(stdout_truncated).lower()} "
            f"stderr_truncated={str(stderr_truncated).lower()} "
            "visible_output_truncated="
            f"{str(visible_output_truncated).lower()} "
            f"visible_output_budget_bytes={max_observation_bytes}\n"
            "stdout:\n"
            + (visible_stdout if visible_stdout else "<empty>")
            + "\nstderr:\n"
            + (visible_stderr if visible_stderr else "<empty>")
            + "\nworkspace changed paths: "
            + changed_paths_observation(changed_paths)
        )

    streams_bounded = stdout_bounded or stderr_bounded
    observation = render(streams_bounded)
    if len(observation.encode("utf-8", errors="replace")) > max_observation_bytes:
        observation = render(True)
    return bound_policy_observation(observation, limit=max_observation_bytes)


def bound_policy_observation(text: str, *, limit: int) -> str:
    if not isinstance(text, str):
        raise TypeError("policy observation must be text")
    if type(limit) is not int or limit <= 0:
        raise ValueError("policy observation limit must be a positive integer")
    visible, _bounded = bound_output(
        text,
        limit=limit,
        already_truncated=False,
        label="observation",
    )
    return visible


def bound_output(
    text: str,
    *,
    limit: int,
    already_truncated: bool,
    label: str,
) -> tuple[str, bool]:
    raw = text.encode("utf-8", errors="replace")
    if not already_truncated and len(raw) <= limit:
        return text, False
    marker = f"\n[{label} truncated: visible output budget reached]\n".encode()
    if len(marker) >= limit:
        return marker[:limit].decode("utf-8", errors="ignore"), True
    remaining = limit - len(marker)
    head_size = (remaining + 1) // 2
    tail_size = remaining - head_size
    head = raw[:head_size].decode("utf-8", errors="ignore")
    tail = raw[-tail_size:].decode("utf-8", errors="ignore") if tail_size else ""
    return head + marker.decode() + tail, True


def changed_paths_observation(changed_paths: tuple[str, ...]) -> str:
    if not changed_paths:
        return "none"
    digest = hashlib.sha256("\n".join(changed_paths).encode("utf-8")).hexdigest()
    sample = list(dict.fromkeys((changed_paths[0], changed_paths[-1])))
    return f"{len(changed_paths)} paths; sha256={digest}; sample={sample!r}"
