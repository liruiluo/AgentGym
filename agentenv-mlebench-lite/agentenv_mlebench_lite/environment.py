from __future__ import annotations

import copy
import csv
import errno
import hashlib
import hmac
import io
import json
import os
import secrets
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import PolicyAction, parse_policy_action
from .dataset import MLEBenchLiteDataset
from .executor import BackendExecution, MLEBenchLiteExecutorError, SandboxExecutor
from .resources import (
    DEFAULT_CPU_LIMIT_CORES,
    DEFAULT_EPISODE_TIMEOUT_MS,
    DEFAULT_GPU_COUNT,
    DEFAULT_MAX_SHELL_TIMEOUT_MS,
    DEFAULT_MAX_TOTAL_EXECUTION_MS,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_WRITABLE_BYTES_LIMIT,
    DEFAULT_WRITABLE_INODES_LIMIT,
    RESOURCE_USAGE_KEYS,
    build_resource_contract,
    zero_resource_usage,
)
from .resources import (
    resource_contract_sha256 as _resource_contract_sha256,
)
from .workspace import (
    MODE_AMG_MEMORY,
    MODES,
    SUBMISSION_PATH,
    EpisodeWorkspace,
    HandoffStaging,
    MLEBenchLitePolicyPathError,
    MLEBenchLiteWorkspaceError,
    MLEBenchLiteWorkspaceRollbackError,
    PendingCreationCleanup,
    WorkspaceManager,
)

METADATA_SCHEMA = "mlebench_lite_metadata_v2"
COMPACTION_RECEIPT_SCHEMA = "mlebench_lite_compaction_receipt_v2"
MAX_COMPACTION_BYTES = 8192
MAX_VISIBLE_OUTPUT_BYTES = 65_536


class ActionSequenceError(RuntimeError):
    """The client attempted a stale, reused, or malformed policy action."""


@dataclass(frozen=True)
class EpisodeStep:
    observation: str
    reward: float
    done: bool
    info: Mapping[str, Any]


@dataclass
class _Counters:
    action_count: int = 0
    native_action_count: int = 0
    execution_count: int = 0
    grading_count: int = 0
    resources: dict[str, int] = field(default_factory=zero_resource_usage)

    def public(self) -> dict[str, int]:
        return {
            "action_count": self.action_count,
            "native_action_count": self.native_action_count,
            "execution_count": self.execution_count,
            "grading_count": self.grading_count,
            **self.resources,
        }


@dataclass
class _Episode:
    workspace: EpisodeWorkspace
    executor: SandboxExecutor
    observation: str
    started_ns: int
    deadline_ns: int
    counters: _Counters = field(default_factory=_Counters)
    done: bool = False
    sandbox_cleaned: bool = False
    cleaned: bool = False
    teardown_receipt: Mapping[str, Any] | None = None
    host_handoff_manifest: Mapping[str, Any] | None = None
    handoff_staging: dict[Path, HandoffStaging] = field(default_factory=dict)


@dataclass(frozen=True)
class _CachedAction:
    payload_sha256: str
    step: EpisodeStep | None = None
    error_kind: str | None = None


@dataclass
class _Slot:
    mode: str
    capability_token: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    episode: _Episode | None = None
    pending_creation: PendingCreationCleanup | None = None
    closed: bool = False
    action_cache: dict[str, _CachedAction] = field(default_factory=dict)


ExecutorFactory = Callable[[], SandboxExecutor]


class MLEBenchLiteEpisodeManager:
    """Server-side lifecycle with replay-safe, atomic policy actions."""

    def __init__(
        self,
        *,
        dataset: MLEBenchLiteDataset,
        workspace_manager: WorkspaceManager,
        executor_factory: ExecutorFactory,
        runner_sha256: str,
        runtime_digest: str,
        max_actions: int = 30,
        max_submission_bytes: int = 100_000_000,
        max_shell_timeout_ms: int = DEFAULT_MAX_SHELL_TIMEOUT_MS,
        episode_timeout_ms: int = DEFAULT_EPISODE_TIMEOUT_MS,
        max_total_execution_ms: int = DEFAULT_MAX_TOTAL_EXECUTION_MS,
        cpu_limit_cores: int = DEFAULT_CPU_LIMIT_CORES,
        memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
        pids_limit: int = DEFAULT_PIDS_LIMIT,
        writable_bytes_limit: int = DEFAULT_WRITABLE_BYTES_LIMIT,
        writable_inodes_limit: int = DEFAULT_WRITABLE_INODES_LIMIT,
        gpu_count: int = DEFAULT_GPU_COUNT,
    ) -> None:
        self.resource_contract = build_resource_contract(
            max_actions=max_actions,
            max_submission_bytes=max_submission_bytes,
            max_shell_timeout_ms=max_shell_timeout_ms,
            max_visible_output_bytes=MAX_VISIBLE_OUTPUT_BYTES,
            submission_path=SUBMISSION_PATH,
            episode_timeout_ms=episode_timeout_ms,
            max_total_execution_ms=max_total_execution_ms,
            cpu_limit_cores=cpu_limit_cores,
            memory_limit_bytes=memory_limit_bytes,
            pids_limit=pids_limit,
            writable_bytes_limit=writable_bytes_limit,
            writable_inodes_limit=writable_inodes_limit,
            gpu_count=gpu_count,
        )
        self.resource_contract_sha256 = _resource_contract_sha256(
            self.resource_contract
        )
        self.dataset = dataset
        self.workspace_manager = workspace_manager
        self.executor_factory = executor_factory
        self.runner_sha256 = runner_sha256
        self.runtime_digest = runtime_digest
        self.max_actions = max_actions
        self.max_submission_bytes = max_submission_bytes
        self.max_shell_timeout_ms = max_shell_timeout_ms
        self.episode_timeout_ms = episode_timeout_ms
        self.max_total_execution_ms = max_total_execution_ms
        self._slots: dict[int, _Slot] = {}
        self._slots_lock = threading.Lock()
        self._next_slot = 0

    def metadata(self) -> dict[str, Any]:
        identity = self.dataset.identity
        return {
            "schema": METADATA_SCHEMA,
            "upstream_commit": identity.upstream_commit,
            "split_sha256": identity.split_sha256,
            "competition_ids": list(identity.competition_ids),
            "task_count": len(self.dataset),
            "public_manifest_sha256": self.dataset.public_manifest_sha256,
            "runner_sha256": self.runner_sha256,
            "runtime_digest": self.runtime_digest,
            "submission_path": SUBMISSION_PATH,
            "modes": list(MODES),
            "resource_contract": dict(self.resource_contract),
            "resource_contract_sha256": self.resource_contract_sha256,
        }

    def create(self, *, mode: str) -> int:
        if mode not in MODES:
            raise ValueError("unsupported evaluation mode")
        with self._slots_lock:
            slot_id = self._next_slot
            self._next_slot += 1
            self._slots[slot_id] = _Slot(
                mode=mode,
                capability_token=secrets.token_hex(32),
            )
            return slot_id

    def capability_token(self, slot_id: int) -> str:
        return self._slot(slot_id).capability_token

    def reset(
        self,
        slot_id: int,
        data_idx: int,
        *,
        capability_token: str | None = None,
    ) -> EpisodeStep:
        slot = self._slot(slot_id, capability_token)
        if (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or data_idx < 0
            or data_idx >= len(self.dataset)
        ):
            raise IndexError("data index must be an integer")
        record = self.dataset[data_idx]
        with slot.lock:
            if slot.closed:
                raise KeyError("unknown environment slot")
            if slot.episode is not None:
                slot.episode.done = True
                self._cleanup_episode(slot.episode)
                slot.episode = None
            self._cleanup_pending_creation(slot)
            slot.action_cache.clear()
            executor = self.executor_factory()
            if not isinstance(executor, SandboxExecutor):
                raise MLEBenchLiteExecutorError(
                    "executor factory returned an invalid executor"
                )
            executor.validate_resource_contract(
                self.resource_contract,
                self.resource_contract_sha256,
            )
            try:
                workspace = self.workspace_manager.create(
                    record,
                    slot.mode,
                    resource_contract=self.resource_contract,
                    resource_contract_sha256=self.resource_contract_sha256,
                )
            except MLEBenchLiteWorkspaceRollbackError as exc:
                slot.pending_creation = exc.pending_cleanup
                raise
            started = time.monotonic_ns()
            episode = _Episode(
                workspace=workspace,
                executor=executor,
                observation="",
                started_ns=started,
                deadline_ns=started + self.episode_timeout_ms * 1_000_000,
                done=True,
            )
            slot.episode = episode
            try:
                executor.preflight(workspace)
            except Exception as setup_exc:
                try:
                    self._cleanup_episode(episode)
                except Exception as cleanup_exc:
                    raise cleanup_exc from setup_exc
                slot.episode = None
                raise
            observation = (
                f"Competition: {record.competition_id}. Public data is mounted at "
                f"/home/data. Work in /home/workspace and place the final CSV at "
                f"{SUBMISSION_PATH}."
            )
            episode.done = False
            episode.observation = observation
            return self._step_output(episode, observation=observation)

    def step(
        self,
        slot_id: int,
        raw_policy_output: str,
        *,
        control: str | None = None,
        expected_action_count: int | None = None,
        action_id: str | None = None,
        capability_token: str | None = None,
    ) -> EpisodeStep:
        slot = self._slot(slot_id, capability_token)
        canonical_action_id = _canonical_uuid4(action_id or str(uuid.uuid4()))
        payload_sha256 = _action_payload_sha256(
            raw_policy_output,
            control=control,
            expected_action_count=expected_action_count,
        )
        with slot.lock:
            if slot.closed:
                raise KeyError("unknown environment slot")
            cached = slot.action_cache.get(canonical_action_id)
            if cached is not None:
                if not hmac.compare_digest(cached.payload_sha256, payload_sha256):
                    raise ActionSequenceError("action id payload changed")
                if cached.error_kind == "sequence":
                    raise ActionSequenceError("action sequence rejected")
                if cached.error_kind == "episode":
                    raise RuntimeError("episode is unavailable")
                assert cached.step is not None
                return copy.deepcopy(cached.step)
            try:
                result = self._step_once(
                    slot,
                    raw_policy_output,
                    control=control,
                    expected_action_count=expected_action_count,
                    action_id=canonical_action_id,
                )
            except ActionSequenceError:
                slot.action_cache[canonical_action_id] = _CachedAction(
                    payload_sha256=payload_sha256,
                    error_kind="sequence",
                )
                raise
            except RuntimeError:
                slot.action_cache[canonical_action_id] = _CachedAction(
                    payload_sha256=payload_sha256,
                    error_kind="episode",
                )
                raise
            slot.action_cache[canonical_action_id] = _CachedAction(
                payload_sha256=payload_sha256,
                step=copy.deepcopy(result),
            )
            return result

    def close(
        self,
        slot_id: int,
        *,
        capability_token: str | None = None,
    ) -> None:
        slot = self._slot(slot_id, capability_token)
        with slot.lock:
            if slot.closed:
                raise KeyError("unknown environment slot")
            slot.closed = True
            try:
                if slot.episode is not None:
                    slot.episode.done = True
                    self._cleanup_episode(slot.episode)
                self._cleanup_pending_creation(slot)
                slot.episode = None
            except Exception:
                slot.closed = False
                raise
            with self._slots_lock:
                self._slots.pop(slot_id, None)

    def close_all(self) -> None:
        with self._slots_lock:
            slot_ids = tuple(self._slots)
        failures: list[Exception] = []
        for slot_id in slot_ids:
            try:
                self.close(slot_id)
            except Exception as exc:  # noqa: BLE001 - attempt every slot
                failures.append(exc)
        if failures:
            raise RuntimeError(
                "one or more MLE-bench Lite slots failed cleanup"
            ) from failures[0]

    def host_submission_path(
        self,
        slot_id: int,
        *,
        capability_token: str | None = None,
    ) -> Path:
        """Reopen and verify the protected host handoff on every lookup."""

        slot = self._slot(slot_id, capability_token)
        with slot.lock:
            if slot.closed:
                raise KeyError("unknown environment slot")
            episode = self._episode(slot)
            if episode.host_handoff_manifest is None:
                raise RuntimeError("slot has no host submission handoff")
            try:
                return self.workspace_manager.verify_handoff_submission(
                    episode.workspace,
                    episode.host_handoff_manifest,
                )
            except MLEBenchLiteWorkspaceError as exc:
                raise RuntimeError("host submission handoff is unavailable") from exc

    def _step_once(
        self,
        slot: _Slot,
        raw_policy_output: str,
        *,
        control: str | None,
        expected_action_count: int | None,
        action_id: str,
    ) -> EpisodeStep:
        episode = self._episode(slot)
        if episode.done:
            raise RuntimeError("episode is already terminal")
        before = episode.counters.public()
        if expected_action_count is not None and (
            isinstance(expected_action_count, bool)
            or not isinstance(expected_action_count, int)
            or expected_action_count != before["action_count"]
        ):
            raise ActionSequenceError("stale policy action sequence")
        episode.counters.action_count += 1
        delta = _zero_delta()
        delta["action_count"] = 1
        info: dict[str, Any] = {}
        try:
            if time.monotonic_ns() >= episode.deadline_ns:
                raise MLEBenchLiteExecutorError("episode deadline elapsed")
            if control is not None:
                observation, action_kind = self._control_step(
                    episode,
                    raw_policy_output,
                    control=control,
                    before=before,
                    delta=delta,
                    info=info,
                )
            else:
                episode.counters.native_action_count += 1
                delta["native_action_count"] = 1
                action = parse_policy_action(raw_policy_output)
                action_kind = action.kind
                observation = self._dispatch(
                    episode,
                    action,
                    action_id=action_id,
                    delta=delta,
                    info=info,
                )
            if not episode.done and episode.counters.action_count >= self.max_actions:
                episode.done = True
                observation = "Action budget exhausted."
                info["terminal_reason"] = "action_budget_exhausted"
                self._cleanup_episode(episode)
        except (MLEBenchLiteExecutorError, MLEBenchLiteWorkspaceError, OSError):
            return self._terminalize_infrastructure_fault(
                episode,
                delta=delta,
                info=info,
            )
        episode.observation = observation
        info.update(
            {
                "action_kind": action_kind,
                "counters": episode.counters.public(),
                "counter_delta": delta,
            }
        )
        return EpisodeStep(
            observation=observation,
            reward=0.0,
            done=episode.done,
            info=info,
        )

    def _control_step(
        self,
        episode: _Episode,
        raw_policy_output: str,
        *,
        control: str,
        before: Mapping[str, int],
        delta: dict[str, int],
        info: dict[str, Any],
    ) -> tuple[str, str]:
        accepted = (
            control == "compaction"
            and episode.workspace.mode == MODE_AMG_MEMORY
            and isinstance(raw_policy_output, str)
        )
        if accepted:
            try:
                accepted = (
                    len(raw_policy_output.encode("utf-8")) <= MAX_COMPACTION_BYTES
                )
            except UnicodeEncodeError:
                accepted = False
        info["control_receipt"] = {
            "schema": COMPACTION_RECEIPT_SCHEMA,
            "action_count_before": before["action_count"],
            "action_count_after": episode.counters.action_count,
            "counter_delta": dict(delta),
            "accepted": accepted,
        }
        if not accepted:
            return "Action is unavailable.", "control_rejected"
        return episode.observation, "compaction"

    def _dispatch(
        self,
        episode: _Episode,
        action: PolicyAction,
        *,
        action_id: str,
        delta: dict[str, int],
        info: dict[str, Any],
    ) -> str:
        if action.kind == "parser_error":
            return "Action could not be parsed."
        if action.kind == "inspect":
            return self._inspect(episode.workspace, action)
        if action.kind == "edit":
            return self._edit(episode.workspace, action)
        if action.kind == "shell":
            episode.counters.execution_count += 1
            delta["execution_count"] = 1
            return self._shell(
                episode,
                action,
                action_id=action_id,
                delta=delta,
            )
        if action.kind == "submit":
            receipt = self._submit(episode)
            if receipt is None:
                return "Submission is unavailable."
            episode.done = True
            info["terminal_receipt"] = receipt
            info["terminal_reason"] = "submission_handoff"
            return "Submission handed off."
        raise RuntimeError("unreachable action kind")

    @staticmethod
    def _inspect(workspace: EpisodeWorkspace, action: PolicyAction) -> str:
        assert action.path is not None
        try:
            payload = workspace.read_policy_file(
                action.path,
                offset=action.offset,
                max_bytes=action.max_bytes,
            )
            return payload.decode("utf-8", errors="replace")
        except MLEBenchLitePolicyPathError:
            return "Path is unavailable."

    @staticmethod
    def _edit(workspace: EpisodeWorkspace, action: PolicyAction) -> str:
        assert action.path is not None and action.content is not None
        try:
            workspace.atomic_write_policy_file(
                action.path,
                action.content.encode("utf-8"),
            )
            return "Edit completed."
        except (UnicodeEncodeError, MLEBenchLitePolicyPathError):
            return "Path is unavailable."

    def _shell(
        self,
        episode: _Episode,
        action: PolicyAction,
        *,
        action_id: str,
        delta: dict[str, int],
    ) -> str:
        assert action.command is not None
        now_ns = time.monotonic_ns()
        remaining_wall_ms = max(0, (episode.deadline_ns - now_ns) // 1_000_000)
        remaining_execution_ms = max(
            0,
            self.max_total_execution_ms
            - episode.counters.resources["execution_time_ms"],
        )
        timeout_ms = min(
            action.timeout_ms,
            self.max_shell_timeout_ms,
            remaining_wall_ms,
            remaining_execution_ms,
        )
        if timeout_ms <= 0:
            raise MLEBenchLiteExecutorError("execution budget exhausted")
        result = episode.executor.run(
            episode.workspace,
            action.command,
            timeout_ms=int(timeout_ms),
            operation_id=action_id,
        )
        receipt = result.receipt
        resource_delta = dict(receipt["resource_delta"])
        resource_cumulative = dict(receipt["resource_cumulative"])
        for key in RESOURCE_USAGE_KEYS:
            delta[key] = resource_delta[key]
        episode.counters.resources = resource_cumulative
        if (
            resource_cumulative["execution_time_ms"] > self.max_total_execution_ms
            or resource_cumulative["writable_bytes"]
            > self.resource_contract["writable_bytes_limit"]
            or resource_cumulative["writable_inodes"]
            > self.resource_contract["writable_inodes_limit"]
        ):
            raise MLEBenchLiteExecutorError("sandbox resource limit exceeded")
        return _visible_execution(result, episode.workspace)

    def _submit(self, episode: _Episode) -> dict[str, str] | None:
        path = episode.workspace.submission_path
        if not _safe_submission_metadata(path, self.max_submission_bytes):
            return None
        try:
            candidate = _read_submission(path, self.max_submission_bytes)
            _validate_public_csv(candidate)
        except (MLEBenchLitePolicyPathError, ValueError):
            return None
        staging: HandoffStaging | None = None
        try:
            freeze_receipt = episode.executor.freeze_and_reap(episode.workspace)
            payload = _read_submission(path, self.max_submission_bytes)
            if payload != candidate:
                raise MLEBenchLiteWorkspaceError("submission changed during freeze")
            submission_sha256 = hashlib.sha256(payload).hexdigest()
            try:
                staging = self.workspace_manager.stage_submission(
                    episode.workspace,
                    payload,
                    submission_sha256,
                )
            finally:
                self._sync_episode_staging(episode)
            teardown_receipt = self._cleanup_episode(
                episode,
                preserve_staging=True,
            )
            manifest: dict[str, Any] = {
                "schema": "mlebench_lite_host_handoff_v2",
                "episode_id": episode.workspace.episode_id,
                "mode": episode.workspace.mode,
                "competition_id": episode.workspace.competition_id,
                "submission_file": "submission.csv",
                "submission_sha256": submission_sha256,
                "runner_sha256": self.runner_sha256,
                "runtime_digest": self.runtime_digest,
                "resource_contract_sha256": self.resource_contract_sha256,
                "freeze_receipt": dict(freeze_receipt),
                "teardown_receipt": dict(teardown_receipt),
            }
            self.workspace_manager.publish_submission(staging, manifest)
            self._sync_episode_staging(episode)
            episode.host_handoff_manifest = manifest
            episode.cleaned = episode.sandbox_cleaned and not episode.handoff_staging
        except (
            MLEBenchLiteExecutorError,
            MLEBenchLiteWorkspaceError,
            OSError,
        ):
            self._sync_episode_staging(episode)
            if staging is not None and staging.directory in episode.handoff_staging:
                try:
                    self.workspace_manager.discard_staging(staging)
                except MLEBenchLiteWorkspaceError as cleanup_exc:
                    raise MLEBenchLiteWorkspaceError(
                        "submission staging rollback failed"
                    ) from cleanup_exc
                finally:
                    self._sync_episode_staging(episode)
            raise
        return {
            "competition_id": episode.workspace.competition_id,
            "submission_path": SUBMISSION_PATH,
            "submission_sha256": submission_sha256,
        }

    def _cleanup_episode(
        self,
        episode: _Episode,
        *,
        preserve_staging: bool = False,
    ) -> Mapping[str, Any]:
        if episode.cleaned:
            assert episode.teardown_receipt is not None
            return episode.teardown_receipt
        failures: list[Exception] = []
        if not episode.sandbox_cleaned:
            if episode.teardown_receipt is None:
                try:
                    episode.teardown_receipt = dict(
                        episode.executor.teardown(episode.workspace)
                    )
                except (MLEBenchLiteExecutorError, MLEBenchLiteWorkspaceError) as exc:
                    failures.append(exc)
            if episode.teardown_receipt is not None:
                try:
                    self.workspace_manager.remove(episode.workspace)
                except MLEBenchLiteWorkspaceError as exc:
                    failures.append(exc)
                else:
                    episode.sandbox_cleaned = True
        self._sync_episode_staging(episode)
        if not preserve_staging:
            for staging in tuple(episode.handoff_staging.values()):
                try:
                    self.workspace_manager.discard_staging(staging)
                except MLEBenchLiteWorkspaceError as exc:
                    failures.append(exc)
            self._sync_episode_staging(episode)
        episode.cleaned = episode.sandbox_cleaned and not episode.handoff_staging
        if failures:
            raise failures[0]
        if episode.teardown_receipt is None:
            raise MLEBenchLiteExecutorError("sandbox teardown receipt is unavailable")
        return episode.teardown_receipt

    def _sync_episode_staging(self, episode: _Episode) -> None:
        episode.handoff_staging = {
            staging.directory: staging
            for staging in self.workspace_manager.tracked_staging(
                episode.workspace.episode_id
            )
        }

    def _cleanup_pending_creation(self, slot: _Slot) -> None:
        if slot.pending_creation is None:
            return
        self.workspace_manager.cleanup_pending_creation(slot.pending_creation)
        slot.pending_creation = None

    def _terminalize_infrastructure_fault(
        self,
        episode: _Episode,
        *,
        delta: dict[str, int],
        info: dict[str, Any],
    ) -> EpisodeStep:
        episode.done = True
        episode.observation = "Episode terminated."
        try:
            self._cleanup_episode(episode)
        except (MLEBenchLiteExecutorError, MLEBenchLiteWorkspaceError):
            # Keep the terminal episode attached to its slot so reset/close can
            # retry the same idempotent teardown and workspace removal.
            pass
        public_info = {
            "action_kind": "infrastructure_terminal",
            "terminal_reason": "infrastructure_failure",
            "counters": episode.counters.public(),
            "counter_delta": dict(delta),
        }
        if "control_receipt" in info:
            public_info["control_receipt"] = info["control_receipt"]
        return EpisodeStep(
            observation="Episode terminated.",
            reward=0.0,
            done=True,
            info=public_info,
        )

    def _slot(
        self,
        slot_id: int,
        capability_token: str | None = None,
    ) -> _Slot:
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            raise KeyError("unknown environment slot")
        try:
            slot = self._slots[slot_id]
        except KeyError as exc:
            raise KeyError("unknown environment slot") from exc
        if capability_token is not None and (
            not isinstance(capability_token, str)
            or not hmac.compare_digest(slot.capability_token, capability_token)
        ):
            raise KeyError("unknown environment slot")
        return slot

    @staticmethod
    def _episode(slot: _Slot) -> _Episode:
        if slot.pending_creation is not None:
            raise RuntimeError("environment slot cleanup is pending")
        if slot.episode is None:
            raise RuntimeError("environment slot has not been reset")
        return slot.episode

    @staticmethod
    def _step_output(episode: _Episode, *, observation: str) -> EpisodeStep:
        return EpisodeStep(
            observation=observation,
            reward=0.0,
            done=episode.done,
            info={"counters": episode.counters.public()},
        )

    def _testing_workspace(self, slot_id: int) -> EpisodeWorkspace:
        return self._episode(self._slot(slot_id)).workspace


def _zero_delta() -> dict[str, int]:
    return {
        "action_count": 0,
        "native_action_count": 0,
        "execution_count": 0,
        "grading_count": 0,
        **zero_resource_usage(),
    }


def _visible_execution(result: BackendExecution, workspace: EpisodeWorkspace) -> str:
    text = result.stdout
    if result.stderr:
        text = f"{text}\n{result.stderr}" if text else result.stderr
    for path in (
        workspace.episode_root,
        workspace.workspace_root,
        workspace.submission_root,
        workspace.public_root,
    ):
        text = text.replace(str(path), "[unavailable]")
    payload = text.encode("utf-8", errors="replace")
    truncated = len(payload) > MAX_VISIBLE_OUTPUT_BYTES
    payload = payload[:MAX_VISIBLE_OUTPUT_BYTES]
    header = (
        f"[execution returncode={result.returncode} "
        f"timed_out={str(result.timed_out).lower()} "
        f"truncated={str(truncated).lower()}]"
    )
    body = payload.decode("utf-8", errors="replace")
    return f"{header}\n{body}" if body else header


def _safe_submission_metadata(path: Path, max_bytes: int) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
            return False
        raise MLEBenchLiteWorkspaceError("submission storage is unavailable") from exc
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and 0 < metadata.st_size <= max_bytes
    )


def _read_submission(path: Path, max_bytes: int) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise OSError("unsafe submission")
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) != before.st_size or stable_before != stable_after:
            raise OSError("submission changed after freeze")
        return bytes(payload)
    except OSError as exc:
        if exc.errno in {None, errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EISDIR}:
            raise MLEBenchLitePolicyPathError("submission path is unavailable") from exc
        raise MLEBenchLiteWorkspaceError("submission storage is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_public_csv(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(rows)
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError("submission is not structurally valid CSV") from exc
    if not header or any(not column.strip() for column in header):
        raise ValueError("submission header is empty")
    row_count = 0
    try:
        for row in rows:
            row_count += 1
            if len(row) != len(header):
                raise ValueError("submission row width differs from its header")
    except csv.Error as exc:
        raise ValueError("submission is not structurally valid CSV") from exc
    if row_count == 0:
        raise ValueError("submission has no data rows")


def resource_contract_sha256(
    *,
    max_actions: int,
    max_submission_bytes: int,
    max_shell_timeout_ms: int,
    episode_timeout_ms: int = DEFAULT_EPISODE_TIMEOUT_MS,
    max_total_execution_ms: int = DEFAULT_MAX_TOTAL_EXECUTION_MS,
    cpu_limit_cores: int = DEFAULT_CPU_LIMIT_CORES,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    pids_limit: int = DEFAULT_PIDS_LIMIT,
    writable_bytes_limit: int = DEFAULT_WRITABLE_BYTES_LIMIT,
    writable_inodes_limit: int = DEFAULT_WRITABLE_INODES_LIMIT,
    gpu_count: int = DEFAULT_GPU_COUNT,
) -> str:
    return _resource_contract_sha256(
        build_resource_contract(
            max_actions=max_actions,
            max_submission_bytes=max_submission_bytes,
            max_shell_timeout_ms=max_shell_timeout_ms,
            max_visible_output_bytes=MAX_VISIBLE_OUTPUT_BYTES,
            submission_path=SUBMISSION_PATH,
            episode_timeout_ms=episode_timeout_ms,
            max_total_execution_ms=max_total_execution_ms,
            cpu_limit_cores=cpu_limit_cores,
            memory_limit_bytes=memory_limit_bytes,
            pids_limit=pids_limit,
            writable_bytes_limit=writable_bytes_limit,
            writable_inodes_limit=writable_inodes_limit,
            gpu_count=gpu_count,
        )
    )


def _canonical_uuid4(value: Any) -> str:
    if not isinstance(value, str):
        raise ActionSequenceError("action id must be a UUID4")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ActionSequenceError("action id must be a UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ActionSequenceError("action id must be a canonical UUID4")
    return value


def _action_payload_sha256(
    raw_policy_output: str,
    *,
    control: str | None,
    expected_action_count: int | None,
) -> str:
    value = {
        "action": raw_policy_output,
        "control": control,
        "expected_action_count": expected_action_count,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
