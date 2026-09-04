from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .actions import ParsedPolicyAction, parse_policy_action
from .audit import AUDIT_CONTRACT, OpenMLEFastAuditSink
from .bounded_text import bound_text
from .dataset import OpenMLEFastDataset, OpenMLEFastRecord
from .deadline import DeadlineExceeded, MonotonicDeadline
from .executor import (
    BackendExecution,
    ExecutionReceipt,
    OpenMLEFastExecutionDeadlineExceeded,
    OpenMLEFastExecutor,
    OpenMLEFastResourceLimits,
)
from .grader_protocol import GradeResult
from .materializer import (
    OpenMLEFastWorkspace,
    OpenMLEFastWorkspaceMaterializer,
)

PUBLIC_SERVICE_SCHEMA = "openmle_fast_public_metadata_v1"
EPISODE_SCHEMA = "openmle_fast_episode_v1"
ACTION_CONTRACT = "openmle_fast_three_tool_qwen_xml_v1"
HORIZON_CONTRACT = "openmle_fast_global_30_action_v1"
CLEANUP_CONTRACT = "openmle_fast_owned_resource_cleanup_v1"
OBSERVATION_CONTRACT = "openmle_fast_bounded_observation_v1"
RECOVERABLE_INVALID_ACTION_REWARD = -0.01
FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_receipt_v1"
)
FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_read_receipt_v1"
)
FILESYSTEM_CHECKPOINT_PATH = ".agent_memory/CONTINUATION.md"
FILESYSTEM_CHECKPOINT_MAX_BYTES = 8 * 1024
FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE = (
    f"`{FILESYSTEM_CHECKPOINT_PATH}` is a single-boundary handoff slot, not "
    "cumulative memory. Every later context boundary overwrites this same file. "
    "Store evidence that must survive multiple context boundaries in other ordinary "
    "workspace files before the boundary, and list those paths in the checkpoint."
)
BOUNDARY_CONTRACTS = {
    "actions": ACTION_CONTRACT,
    "workspace": "openmle_fast_public_workspace_v1",
    "executor": "openmle_fast_executor_v1",
    "grader": "openmle_fast_authenticated_private_ipc_v1",
    "observation": OBSERVATION_CONTRACT,
    "horizon": HORIZON_CONTRACT,
    "cleanup": CLEANUP_CONTRACT,
    "audit": AUDIT_CONTRACT,
}
VERIFIER_CONTRACTS = {
    "action": BOUNDARY_CONTRACTS["actions"],
    "observation": BOUNDARY_CONTRACTS["observation"],
    "horizon": BOUNDARY_CONTRACTS["horizon"],
    "workspace": BOUNDARY_CONTRACTS["workspace"],
    "executor": BOUNDARY_CONTRACTS["executor"],
    "grader_boundary": BOUNDARY_CONTRACTS["grader"],
    "cleanup": BOUNDARY_CONTRACTS["cleanup"],
}
POLICY_PROMPT = """You are solving one OpenMLE-fast task in an isolated /workspace with exactly 30 total policy actions.
Use exactly one Qwen XML function call per response. Output no reasoning, explanation, Markdown fence, action-number prefix, bare JSON, or text before or after the function call. Encode every function argument with the literal Qwen form `<parameter=ARGUMENT_NAME>VALUE</parameter>`. Copy each argument name exactly from the selected function's schema and close every argument with `</parameter>`. Put reflection that must survive context replacement into a workspace file through a valid action.

<tools>
{"type": "function", "function": {"name": "shell_command", "description": "Run one networkless shell command in the episode-private persistent workspace.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "workdir": {"type": "string"}, "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 20000}}, "required": ["command"]}}}
{"type": "function", "function": {"name": "apply_patch", "description": "Apply one patch to files in the episode-private persistent workspace.", "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}}}
{"type": "function", "function": {"name": "submit", "description": "Submit workspace-relative submission.csv once to the protected private grader and terminate the episode.", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For a shell action, use this complete form:

<tool_call>
<function=shell_command>
<parameter=command>
cat TASK.md
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
20000
</parameter>
</function>
</tool_call>

For creating or replacing `train.py`, prefer one shell_command with `printf`; put each Python source line in one shell-quoted `printf` argument and redirect only that bounded command to `train.py`. Do not use `python -c`, `python3 -c`, a heredoc, or `bash -c`.

For a workspace file edit, put the complete patch inside the patch parameter:

<tool_call>
<function=apply_patch>
<parameter=patch>
*** Begin Patch
*** Add File: train.py
+print("ok")
*** End Patch
</parameter>
</function>
</tool_call>

For final submission, use this complete form:

<tool_call>
<function=submit>
</function>
</tool_call>

Function names are limited to shell_command, apply_patch, and submit. For shell_command, command must be non-empty, optional workdir must be exactly `.`, and optional timeout_ms must be an integer from 1 through 20000. The command already runs from /workspace; use workspace-relative paths. For apply_patch, use only workspace-relative paths. Never use `/workspace/` in an apply_patch file path. Use `*** Begin Patch`, `*** Add File:` or `*** Update File:`, and `*** End Patch`. Every added file line starts with `+`. Never emit two function calls or prose with a call. A parser error still consumes an action; after one, emit only one corrected complete function call.

Dependencies are already installed. No network access is available. Never run `pip`, `pip3`, `conda`, `apt`, `ssh`, `curl`, `wget`, or `chmod`, and do not inspect package versions. For shell work prefer `cat`, `head`, `tail`, `ls`, `grep`, `cut`, `sort`, `wc`, `mkdir`, `printf`, and `python train.py`.

Use the first turns efficiently. A useful first response is exactly:
<tool_call>
<function=shell_command>
<parameter=command>
cat TASK.md; head -3 data/train.csv; head -3 data/test.csv; head -3 data/sample_submission.csv
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
20000
</parameter>
</function>
</tool_call>
On a later turn, create workspace-relative `train.py` with one shell_command and `printf`. The script must load public labelled training data, use only public labelled training data to make one deterministic local validation split (a holdout or small cross-validation), print one explicit measured metric line in the form `validation_<metric>=<finite_value>`, fit a bounded candidate with deterministic seeds and `n_jobs=1` where supported, and write workspace-relative `submission.csv` in the required schema. On the following turn, run `python train.py` with one shell_command. Do not spend separate early turns on more row previews, dataset summaries, package checks, or training-set-only metrics.

Work as an iterative ML engineer. Training-set metrics do not count as local validation. The printed validation value must come from executed code, not a placeholder. Keep each run within the managed 15000 ms runtime; do not use broad sweeps, nested parallelism, or large grid searches. After a traceback or measured validation result, modify `train.py` and run it again. The environment exposes no repeatable private score or free validation oracle.

Do not write the continuation note before the first measured validation unless an explicit context-compaction request requires it. At that request, use one shell_command function call to overwrite `.agent_memory/CONTINUATION.md`. Include the last measured validation metric or exact failure, `code_path: train.py`, and a `next_action` that changes `train.py` before rerunning; `python train.py` alone is not enough. When compaction finishes, after a continuation marker, read it exactly once and perform that edit; do not re-inspect the task or repeat the note without new evidence. """ + FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE + """

Reading, editing, running, writing or reading memory, compaction, and submit all consume the same 30-action budget. Every observation reports completed and remaining actions. TASK.md and data are read-only. When a measured local validation and `submission.csv` exist, iterate only while enough actions remain and submit no later than action 27. `submit` grades against the protected private data exactly once; the first submit is terminal, and there is no automatic submission at the action limit; action 30 ends in failure if it is not submit.
If an observation reports a parser error, respond next with only one corrected complete Qwen XML function call. Never describe the correction.
"""
POLICY_PROMPT_SHA256 = hashlib.sha256(POLICY_PROMPT.encode("utf-8")).hexdigest()


class GraderClient(Protocol):
    def grade(
        self,
        *,
        request_id: str,
        episode_id: str,
        task_id: str,
        grader_binding_sha256: str,
        package_identity_sha256: str,
        baseline_score: float,
        ideal_score: float,
        higher_is_better: bool,
        submission: bytes,
        deadline: MonotonicDeadline,
    ) -> GradeResult: ...


@dataclass(frozen=True)
class EpisodeStep:
    observation: str
    reward: float | None
    done: bool
    info: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "state": self.observation,
            "reward": self.reward,
            "done": self.done,
            "truncated": self.info.get("truncated", False),
            "info": dict(self.info),
        }


@dataclass
class EpisodeCounters:
    action_count: int = 0
    execution_action_count: int = 0
    execution_attempt_count: int = 0
    execution_completed_count: int = 0
    nested_subprocess_count: int = 0
    fit_count: int = 0
    grading_count: int = 0
    managed_runtime_wall_seconds: float = 0.0
    raw_output_bytes: int = 0

    def add_execution(self, receipt: ExecutionReceipt) -> dict[str, Any]:
        delta = {
            "action_count": 0,
            "execution_action_count": receipt.execution_action_delta,
            "execution_attempt_count": receipt.execution_attempt_delta,
            "execution_completed_count": receipt.execution_completed_delta,
            "nested_subprocess_count": receipt.nested_subprocess_delta,
            "fit_count": receipt.fit_delta,
            "grading_count": 0,
            "managed_runtime_wall_seconds": (receipt.managed_runtime_wall_seconds),
            "raw_output_bytes": receipt.raw_output_bytes,
        }
        self.execution_action_count += receipt.execution_action_delta
        self.execution_attempt_count += receipt.execution_attempt_delta
        self.execution_completed_count += receipt.execution_completed_delta
        self.nested_subprocess_count += receipt.nested_subprocess_delta
        self.fit_count += receipt.fit_delta
        self.managed_runtime_wall_seconds += delta["managed_runtime_wall_seconds"]
        self.raw_output_bytes += receipt.raw_output_bytes
        if self.execution_completed_count > self.execution_attempt_count:
            raise RuntimeError("execution completed counter exceeds attempts")
        return delta

    def add_deadline_execution(self, backend: BackendExecution) -> dict[str, Any]:
        delta = {
            "action_count": 0,
            "execution_action_count": int(backend.execution_attempt_delta > 0),
            "execution_attempt_count": backend.execution_attempt_delta,
            "execution_completed_count": backend.execution_completed_delta,
            "nested_subprocess_count": backend.nested_subprocess_delta,
            "fit_count": backend.fit_delta,
            "grading_count": 0,
            "managed_runtime_wall_seconds": backend.managed_runtime_wall_seconds,
            "raw_output_bytes": len(backend.stdout) + len(backend.stderr),
        }
        self.execution_action_count += delta["execution_action_count"]
        self.execution_attempt_count += delta["execution_attempt_count"]
        self.execution_completed_count += delta["execution_completed_count"]
        self.nested_subprocess_count += delta["nested_subprocess_count"]
        self.fit_count += delta["fit_count"]
        self.managed_runtime_wall_seconds += delta["managed_runtime_wall_seconds"]
        self.raw_output_bytes += delta["raw_output_bytes"]
        if self.execution_completed_count > self.execution_attempt_count:
            raise RuntimeError("execution completed counter exceeds attempts")
        return delta

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Episode:
    slot_id: int
    episode_id: str
    record: OpenMLEFastRecord | None
    workspace: OpenMLEFastWorkspace | None
    executor: OpenMLEFastExecutor | None
    observation: str
    started_monotonic: float
    counters: EpisodeCounters = field(default_factory=EpisodeCounters)
    done: bool = False
    truncated: bool = False
    reward: float | None = 0.0
    terminal_reason: str | None = None
    grade: GradeResult | None = None
    last_execution: ExecutionReceipt | None = None
    freeze_receipt: Mapping[str, Any] | None = None
    teardown_receipt: Mapping[str, Any] | None = None
    fit_counter_coverage: str = "not_observed"


@dataclass
class _Slot:
    episode: _Episode | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class OpenMLEFastEpisodeManager:
    def __init__(
        self,
        *,
        dataset: OpenMLEFastDataset,
        materializer: OpenMLEFastWorkspaceMaterializer,
        executor_factory: Callable[[], OpenMLEFastExecutor],
        grader_client: GraderClient,
        limits: OpenMLEFastResourceLimits,
        runtime_metadata: Mapping[str, Any],
        audit_sink: OpenMLEFastAuditSink | None = None,
    ) -> None:
        if limits.max_policy_actions != 30:
            raise ValueError("OpenMLE-fast requires the frozen 30-action limit")
        self.dataset = dataset
        self.materializer = materializer
        self.executor_factory = executor_factory
        self.grader_client = grader_client
        self.limits = limits
        self.runtime_metadata = _validate_runtime_metadata(runtime_metadata)
        self.max_observation_tokens = int(
            self.runtime_metadata.get(
                "max_observation_tokens", limits.observation_bytes
            )
        )
        self.audit_sink = audit_sink
        probe = executor_factory()
        self.executor_metadata = dict(probe.metadata)
        self._slots: dict[int, _Slot] = {}
        self._closed_slots: set[int] = set()
        self._next_slot = 0
        self._slots_lock = threading.Lock()

    def create(self) -> int:
        with self._slots_lock:
            slot_id = self._next_slot
            self._next_slot += 1
            self._slots[slot_id] = _Slot()
        return slot_id

    def reconcile_orphans(self) -> int:
        with self._slots_lock:
            if self._slots:
                raise RuntimeError(
                    "cannot reconcile episode roots while slots are active"
                )
        return self.materializer.reconcile_orphans()

    def close_all(self) -> dict[str, Any]:
        with self._slots_lock:
            slot_ids = tuple(self._slots)
        receipts = [self.close(slot_id) for slot_id in slot_ids]
        failed = sum(1 for receipt in receipts if receipt.get("closed") is not True)
        return {
            "schema": "openmle_fast_close_all_receipt_v1",
            "requested": len(slot_ids),
            "closed": len(slot_ids) - failed,
            "failed": failed,
            "cleanup_contract": CLEANUP_CONTRACT,
        }

    def reset(self, slot_id: int, data_idx: int) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            started_monotonic = time.monotonic()
            try:
                self._close_episode(slot)
            except Exception:  # noqa: BLE001 - cleanup remains retryable
                episode = self._episode(slot)
                self._infrastructure_terminal(episode, "cleanup_infrastructure_fault")
                return self._public_step(
                    episode,
                    action_kind="reset",
                    action_status="infrastructure_fault",
                )
            record = self.dataset[data_idx]
            episode = _Episode(
                slot_id=slot_id,
                episode_id=uuid.uuid4().hex,
                record=record,
                workspace=None,
                executor=None,
                observation=(
                    "Environment infrastructure fault; this sample must be rescheduled."
                ),
                started_monotonic=started_monotonic,
            )
            # Adopt the cleanup handle before constructing any later component.
            # A failed cleanup remains reachable through the slot for bounded retry.
            slot.episode = episode
            try:
                episode.workspace = self.materializer.materialize(record)
                observation = _bound_text(
                    record.task_markdown,
                    self.limits,
                    self.max_observation_tokens,
                )
                executor = self.executor_factory()
                episode.observation = observation
                episode.executor = executor
            except Exception:  # noqa: BLE001 - reset faults become truncations
                episode.done = True
                episode.truncated = True
                episode.reward = None
                episode.terminal_reason = "reset_infrastructure_fault"
                if episode.workspace is not None and episode.executor is None:
                    try:
                        self.materializer.close(episode.workspace)
                    except Exception:  # noqa: BLE001 - retain handle for close retry
                        pass
                    else:
                        episode.workspace = None
            episode.observation = _with_action_budget_status(
                episode.observation,
                episode.counters.action_count,
                self.limits,
                self.max_observation_tokens,
            )
            return self._public_step(
                episode, action_kind="reset", action_status="reset"
            )

    def step(self, slot_id: int, raw_policy_output: str) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            episode = self._episode(slot)
            if episode.done:
                raise RuntimeError("OpenMLE-fast episode is already terminal")
            # Reward is transition-local.  Clear the previous nonterminal
            # reward before classifying this action so a parser penalty cannot
            # leak into the next valid tool call.
            episode.reward = 0.0
            episode.counters.action_count += 1
            delta = _zero_delta()
            delta["action_count"] = 1
            episode_deadline = _episode_deadline(episode, self.limits)
            if episode_deadline.expired():
                self._policy_terminal(episode, "episode_wall_limit")
                episode.observation = (
                    "The reset-to-terminal episode wall expired before this action "
                    "could be parsed or executed."
                )
                episode.observation = _with_action_budget_status(
                    episode.observation,
                    episode.counters.action_count,
                    self.limits,
                    self.max_observation_tokens,
                )
                return self._public_step(
                    episode,
                    action_kind="deadline_rejected",
                    action_status="policy_violation",
                    counter_delta=delta,
                )
            action = parse_policy_action(raw_policy_output)
            action_status = "parser_error"
            if action.kind == "parser_error":
                episode.reward = RECOVERABLE_INVALID_ACTION_REWARD
                episode.observation = _bound_text(
                    "Action rejected by the exact three-tool parser. "
                    + str(action.error),
                    self.limits,
                    self.max_observation_tokens,
                )
            elif action.kind == "submit":
                grading_before = episode.counters.grading_count
                action_status = self._submit(episode, episode_deadline)
                delta["grading_count"] = episode.counters.grading_count - grading_before
            else:
                action_status, execution_delta = self._execute(
                    episode,
                    action,
                    episode_deadline,
                )
                for key, value in execution_delta.items():
                    delta[key] += value

            if not episode.done and episode_deadline.expired():
                self._policy_terminal(episode, "episode_wall_limit")
            if (
                not episode.done
                and episode.counters.managed_runtime_wall_seconds
                >= self.limits.managed_runtime_per_episode_ms / 1000.0
            ):
                self._policy_terminal(episode, "managed_runtime_limit")
            if (
                not episode.done
                and episode.counters.action_count >= self.limits.max_policy_actions
            ):
                self._policy_terminal(episode, "action_budget_exhausted")
                episode.observation = _bound_text(
                    episode.observation
                    + "\nThe executed action was action 30; the global action budget is exhausted.",
                    self.limits,
                    self.max_observation_tokens,
                )
            episode.observation = _with_action_budget_status(
                episode.observation,
                episode.counters.action_count,
                self.limits,
                self.max_observation_tokens,
            )
            return self._public_step(
                episode,
                action_kind=action.kind,
                action_status=action_status,
                counter_delta=delta,
            )

    def observation(self, slot_id: int) -> str:
        slot = self._slot(slot_id)
        with slot.lock:
            return self._episode(slot).observation

    def finalize_horizon(self, slot_id: int) -> EpisodeStep:
        slot = self._slot(slot_id)
        with slot.lock:
            episode = self._episode(slot)
            if episode.done:
                raise RuntimeError("OpenMLE-fast episode is already terminal")
            deadline = _episode_deadline(episode, self.limits)
            if deadline.expired():
                self._policy_terminal(episode, "episode_wall_limit")
                episode.observation = (
                    "The reset-to-terminal episode wall expired before horizon "
                    "finalization."
                )
            else:
                self._policy_terminal(episode, "action_budget_exhausted")
                episode.observation = _bound_text(
                    episode.observation
                    + "\nThe policy horizon ended. No extra policy action or grade was created.",
                    self.limits,
                    self.max_observation_tokens,
                )
            return self._public_step(
                episode,
                action_kind="policy_horizon",
                action_status="terminal",
                counter_delta=_zero_delta(),
            )

    def close(self, slot_id: int) -> dict[str, Any]:
        with self._slots_lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                if slot_id in self._closed_slots:
                    return {
                        "schema": "openmle_fast_cleanup_receipt_v1",
                        "closed": False,
                        "already_closed": True,
                        "workspace_removed": False,
                        "retryable": False,
                        "failure_class": None,
                        "cleanup_contract": CLEANUP_CONTRACT,
                    }
                raise KeyError(f"unknown OpenMLE-fast slot: {slot_id}")
        with slot.lock:
            had_workspace = bool(
                slot.episode is not None and slot.episode.workspace is not None
            )
            try:
                self._close_episode(slot)
            except Exception:  # noqa: BLE001 - retain the handle for retry
                return {
                    "schema": "openmle_fast_cleanup_receipt_v1",
                    "closed": False,
                    "already_closed": False,
                    "workspace_removed": False,
                    "retryable": True,
                    "failure_class": "cleanup_infrastructure_fault",
                    "cleanup_contract": CLEANUP_CONTRACT,
                }
        with self._slots_lock:
            self._slots.pop(slot_id, None)
            self._closed_slots.add(slot_id)
        return {
            "schema": "openmle_fast_cleanup_receipt_v1",
            "closed": True,
            "already_closed": False,
            "workspace_removed": had_workspace,
            "cleanup_contract": CLEANUP_CONTRACT,
            "retryable": False,
            "failure_class": None,
        }

    def metadata(self) -> dict[str, Any]:
        with self._slots_lock:
            slots = list(self._slots.values())
        active_workspaces = sum(
            1
            for slot in slots
            if slot.episode is not None and slot.episode.workspace is not None
        )
        active_environments = sum(1 for slot in slots if slot.episode is not None)
        backend = self.executor_metadata.get("backend", {})
        provenance = self.dataset.provenance
        maximum_request_wall_seconds = (
            max(
                self.limits.shell_wall_ms,
                self.limits.grader_total_wall_ms,
            )
            / 1000.0
        )
        return {
            "schema": PUBLIC_SERVICE_SCHEMA,
            "domain_id": "openmle_fast",
            "contract_version": "openmle_fast_v1",
            "panel_id": provenance.panel_id,
            "release_revision": provenance.release_revision,
            "openmle_tasks_revision": provenance.release_revision,
            "manifest_sha256": provenance.manifest_sha256,
            "task_manifest_sha256": provenance.manifest_sha256,
            "task_id_list_sha256": provenance.task_id_list_sha256,
            "compact_panel_sha256": provenance.compact_panel_sha256,
            "role": provenance.role,
            "task_count": len(self.dataset),
            "policy_prompt_sha256": POLICY_PROMPT_SHA256,
            "max_policy_actions": self.limits.max_policy_actions,
            "observation_max_bytes": self.limits.observation_bytes,
            "resource_limits": self.limits.as_dict(),
            "boundary_contracts": dict(BOUNDARY_CONTRACTS),
            "contracts": dict(VERIFIER_CONTRACTS),
            "limits": {
                "max_policy_actions": self.limits.max_policy_actions,
                "max_request_wall_seconds": maximum_request_wall_seconds,
            },
            "runtime_source": dict(self.runtime_metadata["runtime_source"]),
            "executor_runtime_digest": self.runtime_metadata["executor_runtime_digest"],
            "max_observation_tokens": self.runtime_metadata.get(
                "max_observation_tokens"
            ),
            "recoverable_invalid_action_reward": RECOVERABLE_INVALID_ACTION_REWARD,
            "implementation_digests": dict(
                self.runtime_metadata.get("implementation_digests", {})
            ),
            "audit_enabled": self.audit_sink is not None,
            "executor_coverage": {
                "formal_eligible": bool(
                    isinstance(backend, Mapping)
                    and backend.get("formal_eligible") is True
                ),
                "backend_contract": (
                    backend.get("contract") if isinstance(backend, Mapping) else None
                ),
                "execution_counter_coverage": (
                    backend.get("execution_counter_coverage")
                    if isinstance(backend, Mapping)
                    else None
                ),
                "fit_counter_coverage": (
                    backend.get("fit_counter_coverage")
                    if isinstance(backend, Mapping)
                    else None
                ),
            },
            "active_counts": {
                "slots": len(slots),
                "episodes": active_environments,
                "workspaces": active_workspaces,
            },
            "active_slot_count": len(slots),
            "active_environment_count": active_environments,
            "active_workspace_count": active_workspaces,
        }

    def _execute(
        self,
        episode: _Episode,
        action: ParsedPolicyAction,
        deadline: MonotonicDeadline,
    ) -> tuple[str, dict[str, Any]]:
        if episode.workspace is None or episode.executor is None:
            self._infrastructure_terminal(episode, "executor_infrastructure_fault")
            return "infrastructure_fault", _zero_delta()
        managed_runtime_budget_ms = max(
            0,
            self.limits.managed_runtime_per_episode_ms
            - math.ceil(episode.counters.managed_runtime_wall_seconds * 1000.0),
        )
        if action.kind == "shell_command" and managed_runtime_budget_ms == 0:
            self._policy_terminal(episode, "managed_runtime_limit")
            episode.observation = "The cumulative managed-runtime budget is exhausted."
            return "policy_violation", _zero_delta()
        try:
            if episode.executor.runner_owns_workspace_lifecycle:
                self.materializer.mark_adopted_by_runner(episode.workspace)
            receipt = episode.executor.execute(
                episode.workspace.policy_root,
                action,
                deadline=deadline,
                managed_runtime_budget_ms=(
                    managed_runtime_budget_ms
                    if action.kind == "shell_command"
                    else None
                ),
            )
            episode.last_execution = receipt
            episode.fit_counter_coverage = receipt.fit_counter_coverage
            delta = episode.counters.add_execution(receipt)
        except OpenMLEFastExecutionDeadlineExceeded as exc:
            episode.fit_counter_coverage = exc.backend.fit_counter_coverage
            delta = episode.counters.add_deadline_execution(exc.backend)
            reason = "episode_wall_limit" if deadline.expired() else "wall_timeout"
            self._policy_terminal(episode, reason)
            episode.observation = "The action exceeded its monotonic wall deadline."
            return "policy_violation", delta
        except DeadlineExceeded:
            reason = "episode_wall_limit" if deadline.expired() else "wall_timeout"
            self._policy_terminal(episode, reason)
            episode.observation = "The action exceeded its monotonic wall deadline."
            return "policy_violation", _zero_delta()
        except Exception:  # noqa: BLE001 - executor faults become truncations
            self._infrastructure_terminal(episode, "executor_infrastructure_fault")
            return "infrastructure_fault", _zero_delta()
        if receipt.infrastructure_fault:
            self._infrastructure_terminal(episode, "executor_infrastructure_fault")
        elif receipt.policy_terminal:
            self._policy_terminal(
                episode, receipt.failure_class or "policy_resource_violation"
            )
        else:
            episode.observation = _execution_observation(
                receipt, self.limits, self.max_observation_tokens
            )
        return receipt.status, delta

    def _submit(
        self,
        episode: _Episode,
        deadline: MonotonicDeadline,
    ) -> str:
        if (
            episode.workspace is None
            or episode.executor is None
            or episode.record is None
        ):
            self._infrastructure_terminal(episode, "grader_infrastructure_fault")
            return "infrastructure_fault"
        try:
            deadline.check()
            freeze = episode.executor.freeze_for_grading(
                episode.workspace.policy_root,
                deadline=deadline,
            )
            episode.freeze_receipt = freeze.as_dict()
        except DeadlineExceeded:
            self._policy_terminal(episode, "episode_wall_limit")
            episode.observation = (
                "The episode wall expired while freezing the workspace."
            )
            return "policy_violation"
        except Exception:  # noqa: BLE001 - classify deadline-limited freeze faults
            try:
                remaining_ms = deadline.remaining_milliseconds()
            except DeadlineExceeded:
                remaining_ms = 0
            if remaining_ms <= self.limits.grader_total_wall_ms:
                self._policy_terminal(episode, "episode_wall_limit")
                episode.observation = (
                    "The episode no longer had enough wall time to freeze the "
                    "workspace and complete private grading."
                )
                return "policy_violation"
            self._infrastructure_terminal(episode, "sandbox_freeze_fault")
            return "infrastructure_fault"
        submission = b""
        safe_submission = False
        try:
            submission = _read_submission(
                episode.workspace.policy_root,
                self.limits.max_submission_bytes,
                deadline=deadline,
            )
            safe_submission = True
        except DeadlineExceeded:
            self._policy_terminal(episode, "episode_wall_limit")
            episode.observation = (
                "The episode wall expired while reading the submission."
            )
            return "policy_violation"
        except (OSError, ValueError):
            submission = b""
        episode.counters.grading_count += 1
        if episode.counters.grading_count != 1:
            raise RuntimeError("OpenMLE-fast grade count exceeded one")
        try:
            result = self.grader_client.grade(
                request_id=uuid.uuid4().hex,
                episode_id=episode.episode_id,
                task_id=episode.record.task_id,
                grader_binding_sha256=episode.record.grader_binding_sha256,
                package_identity_sha256=episode.record.package_identity_sha256,
                baseline_score=episode.record.baseline_score,
                ideal_score=episode.record.ideal_score,
                higher_is_better=episode.record.higher_is_better,
                submission=submission,
                deadline=deadline,
            )
            deadline.check()
        except DeadlineExceeded:
            self._policy_terminal(episode, "episode_wall_limit")
            episode.observation = "The episode wall expired during private grading."
            return "policy_violation"
        except Exception:  # noqa: BLE001 - grader faults become truncations
            if deadline.expired():
                self._policy_terminal(episode, "episode_wall_limit")
                episode.observation = "The episode wall expired during private grading."
                return "policy_violation"
            self._infrastructure_terminal(episode, "grader_infrastructure_fault")
            return "infrastructure_fault"
        episode.grade = result
        episode.done = True
        if not _grade_matches_record(result, episode.record):
            self._infrastructure_terminal(episode, "grader_binding_fault")
            return "infrastructure_fault"
        if result.classification == "infrastructure_fault":
            episode.truncated = True
            episode.reward = None
            episode.terminal_reason = "grader_infrastructure_fault"
            episode.observation = (
                "Private grading infrastructure fault; this sample must be rescheduled."
            )
            return "infrastructure_fault"
        if not safe_submission or not result.submission_valid:
            episode.reward = -1.0
            episode.terminal_reason = "invalid_submission"
            episode.observation = "Submission is invalid. The episode is terminal."
            return "invalid_submission"
        episode.reward = result.normalized_reward
        episode.terminal_reason = result.terminal_reason
        episode.observation = (
            "Submission graded. "
            f"native_score={result.native_score!r} "
            f"normalized_reward={result.normalized_reward!r}. The episode is terminal."
        )
        return "graded"

    def _policy_terminal(self, episode: _Episode, reason: str) -> None:
        episode.done = True
        episode.truncated = False
        episode.reward = -1.0
        episode.terminal_reason = reason

    def _infrastructure_terminal(self, episode: _Episode, reason: str) -> None:
        episode.done = True
        episode.truncated = True
        episode.reward = None
        episode.terminal_reason = reason
        episode.observation = (
            "Environment infrastructure fault; this sample must be rescheduled."
        )

    def _public_step(
        self,
        episode: _Episode,
        *,
        action_kind: str,
        action_status: str,
        counter_delta: Mapping[str, Any] | None = None,
    ) -> EpisodeStep:
        record = episode.record
        grade = None if episode.grade is None else episode.grade.public_payload()
        current_execution = (
            episode.last_execution
            if (
                episode.last_execution is not None
                and action_kind in {"shell_command", "apply_patch"}
                and episode.last_execution.action_kind == action_kind
            )
            else None
        )
        execution = None if current_execution is None else current_execution.as_dict()
        if current_execution is not None and episode.workspace is not None:
            action_completed = bool(
                action_status == "completed"
                and (
                    current_execution.action_kind != "shell_command"
                    or (
                        isinstance(current_execution.exit_code, int)
                        and not isinstance(current_execution.exit_code, bool)
                        and current_execution.exit_code == 0
                        and current_execution.timed_out is False
                    )
                )
            )
            checkpoint_receipt = _filesystem_checkpoint_receipt(
                episode.workspace.policy_root,
                current_execution,
                action_completed=action_completed,
            )
            execution["filesystem_checkpoint"] = checkpoint_receipt
            execution["filesystem_checkpoint_read"] = (
                _filesystem_checkpoint_read_receipt(
                    checkpoint_receipt,
                    current_execution,
                    action_completed=action_completed,
                )
            )
        info = {
            "schema": EPISODE_SCHEMA,
            "episode_id": episode.episode_id,
            "data_idx": None if record is None else record.data_idx,
            "task_id": None if record is None else record.task_id,
            "source_family": None if record is None else record.source_family,
            "public_tree_sha256": (
                None if record is None else record.public_tree_sha256
            ),
            "manifest_sha256": self.dataset.provenance.manifest_sha256,
            "task_manifest_sha256": self.dataset.provenance.manifest_sha256,
            "release_revision": self.dataset.provenance.release_revision,
            "manifest_role": self.dataset.provenance.role,
            "archive_sha256": (None if record is None else record.archive_sha256),
            "package_identity_sha256": (
                None if record is None else record.package_identity_sha256
            ),
            "task_spec_sha256": (None if record is None else record.task_spec_sha256),
            "grader_binding_sha256": (
                None if record is None else record.grader_binding_sha256
            ),
            "runtime_source": dict(self.runtime_metadata["runtime_source"]),
            "executor_runtime_digest": self.runtime_metadata["executor_runtime_digest"],
            "implementation_digests": dict(
                self.runtime_metadata.get("implementation_digests", {})
            ),
            "boundary_contracts": dict(BOUNDARY_CONTRACTS),
            "action_kind": action_kind,
            "action_status": action_status,
            "terminal": episode.done,
            "truncated": episode.truncated,
            "terminal_reason": episode.terminal_reason,
            "runtime_success": bool(
                not episode.truncated
                and episode.grade is not None
                and episode.grade.runtime_success
            ),
            "episode_success": bool(
                not episode.truncated
                and episode.grade is not None
                and episode.grade.submission_valid
                and episode.grade.improved_over_baseline
            ),
            "counters": episode.counters.as_dict(),
            "counter_delta": dict(counter_delta or _zero_delta()),
            "fit_counter_coverage": episode.fit_counter_coverage,
            "execution": execution,
            "sandbox_freeze": episode.freeze_receipt,
            "sandbox_teardown": episode.teardown_receipt,
            "grade": grade,
            "audit_digest": None,
            "unaudited_evidence_sha256": None,
        }
        step = EpisodeStep(
            observation=_bound_text(
                episode.observation,
                self.limits,
                self.max_observation_tokens,
            ),
            reward=episode.reward,
            done=episode.done,
            info=info,
        )
        if self.audit_sink is not None:
            try:
                audit_digest = self.audit_sink.emit(
                    event=action_kind,
                    episode_id=episode.episode_id,
                    payload=step.as_dict(),
                )
            except Exception:  # noqa: BLE001 - audit failures truncate coherently
                unaudited = hashlib.sha256(
                    json.dumps(
                        step.as_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                self._infrastructure_terminal(episode, "audit_infrastructure_fault")
                info.update(
                    {
                        "action_status": "infrastructure_fault",
                        "terminal": True,
                        "truncated": True,
                        "terminal_reason": "audit_infrastructure_fault",
                        "runtime_success": False,
                        "episode_success": False,
                        "unaudited_evidence_sha256": unaudited,
                    }
                )
                return EpisodeStep(
                    observation=_bound_text(
                        episode.observation,
                        self.limits,
                        self.max_observation_tokens,
                    ),
                    reward=None,
                    done=True,
                    info=info,
                )
            info["audit_digest"] = audit_digest
            step = EpisodeStep(
                observation=step.observation,
                reward=step.reward,
                done=step.done,
                info=info,
            )
        return step

    def _slot(self, slot_id: int) -> _Slot:
        if type(slot_id) is not int:
            raise TypeError("OpenMLE-fast slot id must be an integer")
        with self._slots_lock:
            try:
                return self._slots[slot_id]
            except KeyError as exc:
                raise KeyError(f"unknown OpenMLE-fast slot: {slot_id}") from exc

    @staticmethod
    def _episode(slot: _Slot) -> _Episode:
        if slot.episode is None:
            raise RuntimeError("OpenMLE-fast slot has not been reset")
        return slot.episode

    def _close_episode(self, slot: _Slot) -> None:
        episode = slot.episode
        if episode is None:
            return
        if episode.workspace is not None and episode.executor is not None:
            runner_owns_mount = episode.executor.runner_owns_workspace_lifecycle
            adopted = self.materializer.is_adopted_by_runner(episode.workspace)
            if not runner_owns_mount or adopted:
                teardown = episode.executor.close(episode.workspace.policy_root)
                episode.teardown_receipt = teardown.as_dict()
        if episode.workspace is not None:
            self.materializer.close(episode.workspace)
        slot.episode = None

    def _testing_policy_root(self, slot_id: int) -> Path:
        slot = self._slot(slot_id)
        episode = self._episode(slot)
        if episode.workspace is None:
            raise RuntimeError("test episode has no workspace")
        return episode.workspace.policy_root


def _read_submission(
    workspace: Path,
    maximum: int,
    *,
    deadline: MonotonicDeadline,
) -> bytes:
    deadline.check()
    root_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        root_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW
    root_descriptor = os.open(workspace, root_flags)
    try:
        flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open("submission.csv", flags, dir_fd=root_descriptor)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("submission is not an independent regular file")
            if info.st_size > maximum:
                raise ValueError("submission exceeds the input cap")
            allocated = getattr(info, "st_blocks", 0) * 512
            if info.st_size > 0 and allocated < info.st_size:
                raise ValueError("sparse submissions are not accepted")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                chunks: list[bytes] = []
                remaining = maximum + 1
                while remaining:
                    deadline.check()
                    chunk = handle.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
            after = os.fstat(descriptor)
            identity_before = (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_nlink,
                info.st_size,
                getattr(info, "st_blocks", 0),
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                getattr(after, "st_blocks", 0),
            )
            if identity_after != identity_before or len(payload) != info.st_size:
                raise ValueError("submission changed while it was being read")
        finally:
            os.close(descriptor)
    finally:
        os.close(root_descriptor)
    if len(payload) > maximum:
        raise ValueError("submission exceeds the input cap")
    deadline.check()
    return payload


def _filesystem_checkpoint_receipt(
    root: Path,
    execution: ExecutionReceipt,
    *,
    action_completed: bool,
) -> dict[str, Any]:
    exists, regular_file, size_bytes, digest, _payload = (
        _read_filesystem_checkpoint_beneath(root)
    )
    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "action_kind": execution.action_kind,
        "action_completed": bool(action_completed),
        "changed": FILESYSTEM_CHECKPOINT_PATH in execution.changed_paths,
        "exists": exists,
        "regular_file": regular_file,
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def _filesystem_checkpoint_read_receipt(
    checkpoint_receipt: Mapping[str, Any],
    execution: ExecutionReceipt,
    *,
    action_completed: bool,
) -> dict[str, Any]:
    payload = execution.stdout.encode("utf-8")
    size = checkpoint_receipt.get("size_bytes")
    digest = checkpoint_receipt.get("sha256")
    observed = bool(
        execution.action_kind == "shell_command"
        and action_completed
        and execution.visible_output_truncated is False
        and checkpoint_receipt.get("changed") is False
        and checkpoint_receipt.get("exists") is True
        and checkpoint_receipt.get("regular_file") is True
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < size <= FILESYSTEM_CHECKPOINT_MAX_BYTES
        and isinstance(digest, str)
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


def _read_filesystem_checkpoint_beneath(
    root: Path,
) -> tuple[bool, bool, int | None, str | None, bytes | None]:
    """Read the fixed checkpoint through stable directory/file descriptors.

    The receipt fails closed on symlinks, hard links, path replacement, in-place
    mutation, special files, and files above the 8 KiB contract.  Returning the
    payload only to this module lets the endpoint attest an exact stdout read
    without exposing checkpoint contents as free wrapper state.
    """

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC

    root_fd: int | None = None
    memory_fd: int | None = None
    checkpoint_fd: int | None = None
    try:
        root_fd = os.open(root, directory_flags)
        try:
            memory_fd = os.open(".agent_memory", directory_flags, dir_fd=root_fd)
        except OSError:
            return False, False, None, None, None
        try:
            path_before = os.stat(
                "CONTINUATION.md", dir_fd=memory_fd, follow_symlinks=False
            )
        except OSError:
            return False, False, None, None, None
        try:
            checkpoint_fd = os.open(
                "CONTINUATION.md", file_flags, dir_fd=memory_fd
            )
        except OSError:
            return True, False, None, None, None

        before = os.fstat(checkpoint_fd)
        if not _same_file_identity(path_before, before):
            return True, False, None, None, None
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return True, False, None, None, None
        size = int(before.st_size)
        payload: bytes | None = None
        if size <= FILESYSTEM_CHECKPOINT_MAX_BYTES:
            chunks: list[bytes] = []
            remaining = FILESYSTEM_CHECKPOINT_MAX_BYTES + 1
            while remaining > 0:
                chunk = os.read(checkpoint_fd, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)

        after = os.fstat(checkpoint_fd)
        try:
            path_after = os.stat(
                "CONTINUATION.md", dir_fd=memory_fd, follow_symlinks=False
            )
        except OSError:
            return True, False, None, None, None
        if not (
            _same_file_identity(before, after)
            and _same_file_identity(before, path_after)
        ):
            return True, False, None, None, None
        if size > FILESYSTEM_CHECKPOINT_MAX_BYTES:
            return True, True, size, None, None
        if payload is None or len(payload) != size:
            return True, False, None, None, None
        return True, True, size, hashlib.sha256(payload).hexdigest(), payload
    except OSError:
        return False, False, None, None, None
    finally:
        for descriptor in (checkpoint_fd, memory_fd, root_fd):
            if descriptor is not None:
                os.close(descriptor)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size")
    if any(getattr(left, name) != getattr(right, name) for name in fields):
        return False
    for name in ("st_mtime_ns", "st_ctime_ns", "st_blocks"):
        if hasattr(left, name) and hasattr(right, name):
            if getattr(left, name) != getattr(right, name):
                return False
    return True


def _execution_observation(
    receipt: ExecutionReceipt,
    limits: OpenMLEFastResourceLimits,
    max_tokens: int,
) -> str:
    parts = [
        f"action_status={receipt.status}",
        f"exit_code={receipt.exit_code!r}",
        f"timed_out={receipt.timed_out}",
    ]
    if receipt.failure_class:
        parts.append(f"failure_class={receipt.failure_class}")
    if receipt.stdout:
        parts.append("stdout:\n" + receipt.stdout)
    if receipt.stderr:
        parts.append("stderr:\n" + receipt.stderr)
    if receipt.changed_paths:
        parts.append("changed_paths=" + ",".join(receipt.changed_paths))
    return _bound_text("\n".join(parts), limits, max_tokens)


def _with_action_budget_status(
    observation: str,
    action_count: int,
    limits: OpenMLEFastResourceLimits,
    max_tokens: int,
) -> str:
    remaining = max(0, limits.max_policy_actions - action_count)
    separator = "" if observation.endswith("\n") else "\n"
    return _bound_text(
        observation
        + separator
        + (
            f"[OpenMLE action budget: action {action_count} completed; "
            f"{remaining} actions remain.]"
        ),
        limits,
        max_tokens,
    )


def _bound_text(
    value: str,
    limits: OpenMLEFastResourceLimits,
    max_tokens: int | None = None,
) -> str:
    return bound_text(
        value,
        max_bytes=limits.observation_bytes,
        max_tokens=max_tokens,
        marker="\n...[observation truncated]...\n",
    )


def _episode_deadline(
    episode: _Episode,
    limits: OpenMLEFastResourceLimits,
) -> MonotonicDeadline:
    return MonotonicDeadline(
        episode.started_monotonic + limits.episode_wall_ms / 1000.0
    )


def _grade_matches_record(
    result: GradeResult,
    record: OpenMLEFastRecord,
) -> bool:
    if (
        result.task_id != record.task_id
        or result.grader_binding_sha256 != record.grader_binding_sha256
        or result.package_identity_sha256 != record.package_identity_sha256
        or result.baseline_score != record.baseline_score
        or result.ideal_score != record.ideal_score
        or result.higher_is_better != record.higher_is_better
    ):
        return False
    if result.classification != "graded":
        return True
    if result.native_score is None or result.normalized_reward is None:
        return False
    direction = 1.0 if record.higher_is_better else -1.0
    scale = max(1.0, abs(record.baseline_score), abs(record.ideal_score))
    tolerance = 1e-9 * scale
    gap = direction * (record.ideal_score - record.baseline_score)
    delta_raw = direction * (result.native_score - record.baseline_score)
    delta = 0.0 if abs(delta_raw) <= tolerance else delta_raw
    expected_reward = max(-1.0, min(1.0, delta / gap))
    return math.isclose(
        result.normalized_reward, expected_reward, rel_tol=0.0, abs_tol=1e-12
    ) and result.improved_over_baseline == (delta_raw > tolerance)


def _zero_delta() -> dict[str, Any]:
    return {
        "action_count": 0,
        "execution_action_count": 0,
        "execution_attempt_count": 0,
        "execution_completed_count": 0,
        "nested_subprocess_count": 0,
        "fit_count": 0,
        "grading_count": 0,
        "managed_runtime_wall_seconds": 0.0,
        "raw_output_bytes": 0,
    }


def _validate_runtime_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime metadata must be a mapping")
    source = value.get("runtime_source")
    if not isinstance(source, Mapping) or set(source) != {
        "outer_commit",
        "inner_commit",
    }:
        raise ValueError("runtime source must contain exact outer/inner commits")
    normalized_source: dict[str, str] = {}
    for key in ("outer_commit", "inner_commit"):
        revision = source[key]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(
                character not in "0123456789abcdef" for character in revision.lower()
            )
        ):
            raise ValueError(f"runtime {key} must be a full Git revision")
        normalized_source[key] = revision.lower()
    digest = value.get("executor_runtime_digest")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:].lower())
    ):
        raise ValueError("executor runtime digest must be sha256:<64 hex>")
    result: dict[str, Any] = {
        "runtime_source": normalized_source,
        "executor_runtime_digest": digest.lower(),
    }
    max_observation_tokens = value.get("max_observation_tokens")
    if max_observation_tokens is not None:
        if type(max_observation_tokens) is not int or max_observation_tokens <= 0:
            raise ValueError("max_observation_tokens must be a positive integer")
        result["max_observation_tokens"] = max_observation_tokens
    implementation_digests = value.get("implementation_digests")
    if implementation_digests is not None:
        if not isinstance(implementation_digests, Mapping) or set(
            implementation_digests
        ) != {"materializer_sha256", "actions_sha256"}:
            raise ValueError("implementation digests are incomplete")
        normalized_digests: dict[str, str] = {}
        for key, implementation_digest in implementation_digests.items():
            if (
                not isinstance(implementation_digest, str)
                or len(implementation_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in implementation_digest.lower()
                )
            ):
                raise ValueError(f"implementation digest {key} is invalid")
            normalized_digests[str(key)] = implementation_digest.lower()
        result["implementation_digests"] = normalized_digests
    return result
