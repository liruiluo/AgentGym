from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

from .env_wrapper import NATURAL_FILESYSTEM_PROMPT_MODE
from .filesystem_webshop_env import (
    PROCEDURAL_FILESYSTEM_SURFACE,
    ProceduralFilesystemWebShopEnv,
    build_filesystem_reward_contract,
    validate_positive_task_reward_scale,
)
from .persistent_workspace import (
    WORKSPACE_CAUSAL_ARMS,
    WORKSPACE_TOOL_CONTRACT,
    WORKSPACE_TOOL_OPS,
    WorkspaceLimits,
)
from .procedural_wrapper import ProceduralAgentMemoryWrapper
from .workspace_sandbox import LinuxNamespaceShellSandbox


SOURCE_PAIRING_XOR_LSB = "xor_lsb_within_orbit_v1"
SOURCE_PAIRING_XOR_DISTRACTOR_CONDITION = (
    "xor_distractor_condition_within_orbit_v1"
)
SOURCE_PAIRING_XOR_PREFERENCE_COORDINATE = (
    "xor_preference_coordinate_within_factorial_v1"
)
SOURCE_PAIRING_CYCLIC_NEXT = "cyclic_next_within_orbit_v1"
SOURCE_PAIRING_CONTRACTS = frozenset(
    {
        SOURCE_PAIRING_XOR_LSB,
        SOURCE_PAIRING_XOR_DISTRACTOR_CONDITION,
        SOURCE_PAIRING_XOR_PREFERENCE_COORDINATE,
        SOURCE_PAIRING_CYCLIC_NEXT,
    }
)
WORKSPACE_PROMPT_FAMILY_NATURAL = "natural_attribute_chain_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_LATENT_PREFERENCE = "latent_preference_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_RECENCY = "recency_override_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_COMPOSITIONAL = "compositional_recall_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_NEGATIVE = "negative_constraint_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_DISTRACTOR = "distractor_robustness_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_INTENT = "intent_clarification_filesystem_v2"
WORKSPACE_PROMPT_FAMILY_SELECTIVE = "selective_memory_use_filesystem_v2"


def resolve_workspace_source_data_idx(
    data_idx: int,
    *,
    source_pairing: str,
    tasks_per_orbit: int,
) -> int:
    """Resolve the exact counterfactual source without crossing an orbit."""
    if isinstance(data_idx, bool) or not isinstance(data_idx, int) or data_idx < 0:
        raise ValueError("workspace source data_idx must be a non-negative integer")
    if (
        isinstance(tasks_per_orbit, bool)
        or not isinstance(tasks_per_orbit, int)
        or tasks_per_orbit < 2
    ):
        raise ValueError("workspace tasks_per_orbit must be an integer >= 2")
    if source_pairing not in SOURCE_PAIRING_CONTRACTS:
        raise ValueError(f"unsupported workspace source pairing: {source_pairing!r}")

    orbit_start = data_idx - (data_idx % tasks_per_orbit)
    offset = data_idx - orbit_start
    if source_pairing in {
        SOURCE_PAIRING_XOR_LSB,
        SOURCE_PAIRING_XOR_DISTRACTOR_CONDITION,
    }:
        if tasks_per_orbit % 2:
            raise ValueError("xor_lsb pairing requires an even tasks_per_orbit")
        source_offset = offset ^ 1
    elif source_pairing == SOURCE_PAIRING_XOR_PREFERENCE_COORDINATE:
        if tasks_per_orbit != 4:
            raise ValueError(
                "preference-coordinate pairing requires exactly four tasks per orbit"
            )
        source_offset = offset ^ 2
    else:
        source_offset = (offset + 1) % tasks_per_orbit
    source_data_idx = orbit_start + source_offset
    if source_data_idx == data_idx:
        raise ValueError("workspace source pairing resolved to the target itself")
    return source_data_idx


class FilesystemAgentMemoryWrapperMixin:
    """Shared Codex-workspace runtime and authenticated intervention control."""

    workspace_intervention_boundary_index = 1
    workspace_causal_arms = ("correct", "blank", "swapped", "no_workspace")
    workspace_source_pairing = SOURCE_PAIRING_XOR_LSB
    workspace_tasks_per_orbit = 2
    workspace_prompt_family = WORKSPACE_PROMPT_FAMILY_NATURAL
    workspace_intervention_source_state = "policy_authored_workspace_only"
    workspace_seed_contract = "none"
    workspace_evaluation_contract = "directional_counterfactual_separation_v1"

    def _initialize_filesystem_runtime(self) -> None:
        if self.memory_prompt_mode != NATURAL_FILESYSTEM_PROMPT_MODE:
            raise RuntimeError(
                "The filesystem-v2 surface requires memory prompt mode "
                f"{NATURAL_FILESYSTEM_PROMPT_MODE!r}."
            )
        configured_shaping = (
            float(self.reward_contract["first_valid_add_reward"]),
            float(self.reward_contract["first_valid_later_session_retrieve_reward"]),
        )
        if any(value != 0.0 for value in configured_shaping):
            raise RuntimeError(
                "The filesystem-v2 surface refuses dedicated write/read reward shaping."
            )
        raw_positive_reward_scale = os.environ.get(
            "AGENTMEMORY_WEBSHOP_POSITIVE_TASK_REWARD_SCALE", "1"
        )
        try:
            positive_reward_scale = validate_positive_task_reward_scale(
                float(raw_positive_reward_scale)
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "AGENTMEMORY_WEBSHOP_POSITIVE_TASK_REWARD_SCALE must be a "
                "finite number > 0"
            ) from exc
        self.positive_task_reward_scale = positive_reward_scale
        self.reward_contract = build_filesystem_reward_contract(
            positive_reward_scale
        )
        provider_metadata = self.provider.metadata()
        if (
            provider_metadata.get("tasks_per_orbit")
            != self.workspace_tasks_per_orbit
        ):
            raise RuntimeError(
                "The filesystem-v2 source-pairing orbit disagrees with its provider."
            )
        if (
            not isinstance(self.workspace_prompt_family, str)
            or not self.workspace_prompt_family
        ):
            raise RuntimeError("The filesystem-v2 prompt family is invalid.")
        try:
            resolve_workspace_source_data_idx(
                0,
                source_pairing=self.workspace_source_pairing,
                tasks_per_orbit=self.workspace_tasks_per_orbit,
            )
        except ValueError as exc:
            raise RuntimeError(
                "The filesystem-v2 source-pairing contract is invalid."
            ) from exc
        root_parent = os.environ.get("AGENTMEMORY_WORKSPACE_ROOT_PARENT")
        self.workspace_root_parent = (
            None if not root_parent else Path(root_parent).expanduser().resolve()
        )
        self.workspace_limits = WorkspaceLimits()
        rg_binary = os.environ.get("AGENTMEMORY_WORKSPACE_RG_BINARY")
        if not rg_binary:
            raise RuntimeError(
                "The Codex workspace surface requires AGENTMEMORY_WORKSPACE_RG_BINARY "
                "to point to a pinned static ripgrep executable."
            )
        rg_sha256 = os.environ.get("AGENTMEMORY_WORKSPACE_RG_SHA256")
        if not rg_sha256:
            raise RuntimeError(
                "The Codex workspace surface requires AGENTMEMORY_WORKSPACE_RG_SHA256 "
                "to freeze the exact ripgrep executable."
            )
        self.shell_sandbox = LinuxNamespaceShellSandbox.from_environment(
            limits=self.workspace_limits.shell_limits(),
            rg_binary=Path(rg_binary),
            expected_rg_sha256=rg_sha256,
        )
        service_role = os.environ.get("AGENTMEMORY_SERVICE_ROLE", "formal")
        intervention_token = os.environ.get(
            "AGENTMEMORY_WORKSPACE_INTERVENTION_TOKEN"
        )
        if service_role == "intervention_eval":
            if not intervention_token or len(intervention_token) < 32:
                raise RuntimeError(
                    "The intervention_eval filesystem service requires a private "
                    "workspace intervention token."
                )
            self._workspace_intervention_token = intervention_token
        else:
            if intervention_token:
                raise RuntimeError(
                    "Workspace intervention control is forbidden outside the "
                    "intervention_eval service role."
                )
            self._workspace_intervention_token = None

    def _environment_configuration(self) -> dict[str, Any]:
        return {
            "workspace_root_parent": self.workspace_root_parent,
            "workspace_limits": self.workspace_limits,
            "shell_sandbox": self.shell_sandbox,
            "positive_task_reward_scale": self.positive_task_reward_scale,
        }

    def filesystem_checkpoint_commit(
        self,
        env_id: int,
        *,
        session_index: int,
        step_count: int,
        size_bytes: int,
        sha256: str,
    ) -> dict[str, Any]:
        """Commit a verified WebShop checkpoint without consuming a policy step."""

        environment = self.require_env(env_id)
        with self.require_lock(env_id):
            observation, info = environment.commit_filesystem_checkpoint(
                expected_session_index=session_index,
                expected_step_count=step_count,
                expected_size_bytes=size_bytes,
                expected_sha256=sha256,
            )
            payload = {
                "id": env_id,
                "observation": observation,
                "reward": 0.0,
                "done": False,
                "info": info,
            }
            self.info[env_id] = payload
            return payload

    def workspace_intervention(
        self,
        env_id: int,
        *,
        arm: str,
        source_env_id: int | None,
        token: str,
    ) -> dict[str, Any]:
        expected_token = self._workspace_intervention_token
        if expected_token is None or not secrets.compare_digest(token, expected_token):
            raise PermissionError("invalid workspace intervention control token")
        if arm not in self.workspace_causal_arms:
            raise ValueError(
                "workspace intervention arm must be one of: "
                + ", ".join(self.workspace_causal_arms)
            )
        target = self.require_env(env_id)
        source = None if source_env_id is None else self.require_env(source_env_id)
        lock_ids = sorted({env_id, *(() if source_env_id is None else (source_env_id,))})
        locks = [self.require_lock(identifier) for identifier in lock_ids]
        for lock in locks:
            lock.acquire()
        try:
            self._validate_intervention_boundary(target, label="target")
            state = None
            source_label = None
            if arm == "correct":
                if source_env_id not in (None, env_id):
                    raise ValueError(
                        "correct intervention may only use the target workspace"
                    )
                state = target.workspace.export_state()
                source_label = f"target_env:{env_id}:data_idx:{target.data_idx}"
            elif arm in {"swapped", "stale"}:
                if source is None or source_env_id == env_id:
                    raise ValueError(
                        f"{arm} intervention requires a distinct source environment"
                    )
                self._validate_intervention_boundary(source, label="source")
                if not self._is_paired_intervention_source(
                    target,
                    source,
                    arm=arm,
                ):
                    raise ValueError(
                        f"{arm} intervention source must be the target's exact "
                        "counterfactual pair"
                    )
                state = source.workspace.export_state()
                source_label = (
                    f"paired_env:{source_env_id}:data_idx:{source.data_idx}"
                )
            elif source_env_id is not None:
                raise ValueError(f"{arm} intervention must not name a source environment")

            if state is not None and (
                state["file_count"] == 0 and state["directory_count"] == 0
            ):
                raise ValueError(
                    f"{arm} intervention source workspace is empty; causal gate is ineligible"
                )
            observation, info = target.install_workspace_causal_intervention(
                arm,
                state=state,
                source_label=source_label,
            )
            payload = {
                "id": env_id,
                "observation": observation,
                "reward": 0.0,
                "done": False,
                "info": info,
            }
            self.info[env_id] = payload
            return payload
        finally:
            for lock in reversed(locks):
                lock.release()

    def workspace_export(
        self,
        env_id: int,
        *,
        token: str,
    ) -> dict[str, Any]:
        expected_token = self._workspace_intervention_token
        if expected_token is None or not secrets.compare_digest(token, expected_token):
            raise PermissionError("invalid workspace intervention control token")
        environment = self.require_env(env_id)
        lock = self.require_lock(env_id)
        with lock:
            self._validate_intervention_boundary(environment, label="export")
            state = environment.workspace.export_state()
            provenance = environment.workspace.provenance_summary
            return {
                "schema": "agentmemory_workspace_authenticated_export_v1",
                "id": env_id,
                "data_idx": environment.data_idx,
                "workspace_state": state,
                "policy_authored": provenance["policy_authored"],
                "contains_harness_seed": provenance["contains_harness_seed"],
                "workspace_provenance": provenance,
                "hidden_answer_injection": False,
            }

    def _validate_intervention_boundary(self, environment, *, label: str) -> None:
        if (
            environment.done
            or environment.status != "active"
            or environment.current_session_index
            != self.workspace_intervention_boundary_index
        ):
            raise ValueError(
                f"{label} environment is not at the frozen session-"
                f"{self.workspace_intervention_boundary_index} boundary"
            )

    @classmethod
    def _is_paired_intervention_source(cls, target, source, *, arm: str) -> bool:
        del arm
        return source.data_idx == resolve_workspace_source_data_idx(
            target.data_idx,
            source_pairing=cls.workspace_source_pairing,
            tasks_per_orbit=cls.workspace_tasks_per_orbit,
        )

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        for key in (
            "ltm_inventory_mode",
            "ltm_transition_notice_mode",
            "ltm_inventory_key_max_chars",
            "ltm_inventory_key_format",
        ):
            metadata.pop(key, None)
        metadata.update(
            {
                "surface": self.surface,
                "reward_contract": dict(self.reward_contract),
                "memory_management": "policy_managed_persistent_workspace",
                "workspace_surface": "codex_workspace_v2",
                "workspace_tool_contract": WORKSPACE_TOOL_CONTRACT,
                "workspace_tool_ops": list(WORKSPACE_TOOL_OPS),
                "workspace_persistence": "episode_across_sessions",
                "source_pairing": self.workspace_source_pairing,
                "tasks_per_orbit": self.workspace_tasks_per_orbit,
                "workspace_prompt_family": self.workspace_prompt_family,
                "workspace_seed_contract": self.workspace_seed_contract,
                "workspace_evaluation_contract": (
                    self.workspace_evaluation_contract
                ),
                "workspace_episode_isolation": True,
                "workspace_shell_enabled": True,
                "workspace_apply_patch_enabled": True,
                "workspace_host_path_exposed": False,
                "workspace_sandbox": dict(self.shell_sandbox.metadata),
                "workspace_limits": self.workspace_limits.as_metadata(),
                "workspace_intervention_control": {
                    "enabled": self._workspace_intervention_token is not None,
                    "contract": (
                        "authenticated_session_boundary_counterfactual_copy_v1"
                    ),
                    "allowed_arms": list(self.workspace_causal_arms),
                    "boundary_session_index": (
                        self.workspace_intervention_boundary_index
                    ),
                    "source_state": self.workspace_intervention_source_state,
                    "authenticated_export": True,
                    "hidden_answer_injection": False,
                    "token_sha256": (
                        None
                        if self._workspace_intervention_token is None
                        else hashlib.sha256(
                            self._workspace_intervention_token.encode("utf-8")
                        ).hexdigest()
                    ),
                },
            }
        )
        return metadata


class ProceduralFilesystemAgentMemoryWrapper(
    FilesystemAgentMemoryWrapperMixin,
    ProceduralAgentMemoryWrapper,
):
    """HTTP wrapper for the natural-chain persistent-workspace surface."""

    surface = PROCEDURAL_FILESYSTEM_SURFACE
    environment_type = ProceduralFilesystemWebShopEnv

    def __init__(self) -> None:
        super().__init__()
        self._initialize_filesystem_runtime()
