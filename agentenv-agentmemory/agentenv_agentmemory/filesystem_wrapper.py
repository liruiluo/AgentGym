from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

from .env_wrapper import NATURAL_FILESYSTEM_PROMPT_MODE
from .filesystem_webshop_env import (
    FILESYSTEM_REWARD_CONTRACT,
    PROCEDURAL_FILESYSTEM_SURFACE,
    ProceduralFilesystemWebShopEnv,
)
from .persistent_workspace import (
    WORKSPACE_CAUSAL_ARMS,
    WORKSPACE_TOOL_CONTRACT,
    WORKSPACE_TOOL_OPS,
    WorkspaceLimits,
)
from .procedural_wrapper import ProceduralAgentMemoryWrapper
from .workspace_sandbox import LinuxNamespaceShellSandbox


class ProceduralFilesystemAgentMemoryWrapper(ProceduralAgentMemoryWrapper):
    """HTTP wrapper for the natural-chain persistent-workspace surface."""

    surface = PROCEDURAL_FILESYSTEM_SURFACE
    environment_type = ProceduralFilesystemWebShopEnv

    def __init__(self) -> None:
        super().__init__()
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
        self.reward_contract = dict(FILESYSTEM_REWARD_CONTRACT)
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
        }

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
        if arm not in WORKSPACE_CAUSAL_ARMS:
            raise ValueError(
                "workspace intervention arm must be one of: "
                + ", ".join(WORKSPACE_CAUSAL_ARMS)
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
            elif arm == "swapped":
                if source is None or source_env_id == env_id:
                    raise ValueError(
                        "swapped intervention requires a distinct source environment"
                    )
                self._validate_intervention_boundary(source, label="source")
                if target.data_idx // 2 != source.data_idx // 2 or (
                    target.data_idx ^ 1
                ) != source.data_idx:
                    raise ValueError(
                        "swapped intervention source must be the target's exact "
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
            return {
                "schema": "agentmemory_workspace_authenticated_export_v1",
                "id": env_id,
                "data_idx": environment.data_idx,
                "workspace_state": state,
                "policy_authored": True,
                "hidden_answer_injection": False,
            }

    @staticmethod
    def _validate_intervention_boundary(environment, *, label: str) -> None:
        if (
            environment.done
            or environment.status != "active"
            or environment.current_session_index != 1
        ):
            raise ValueError(
                f"{label} environment is not at the frozen first-session boundary"
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
                "reward_contract": dict(FILESYSTEM_REWARD_CONTRACT),
                "memory_management": "policy_managed_persistent_workspace",
                "workspace_surface": "codex_workspace_v2",
                "workspace_tool_contract": WORKSPACE_TOOL_CONTRACT,
                "workspace_tool_ops": list(WORKSPACE_TOOL_OPS),
                "workspace_persistence": "episode_across_sessions",
                "workspace_episode_isolation": True,
                "workspace_shell_enabled": True,
                "workspace_apply_patch_enabled": True,
                "workspace_host_path_exposed": False,
                "workspace_sandbox": dict(self.shell_sandbox.metadata),
                "workspace_limits": self.workspace_limits.as_metadata(),
                "workspace_intervention_control": {
                    "enabled": self._workspace_intervention_token is not None,
                    "contract": "authenticated_first_boundary_counterfactual_copy_v1",
                    "allowed_arms": list(WORKSPACE_CAUSAL_ARMS),
                    "boundary_session_index": 1,
                    "source_state": "policy_authored_workspace_only",
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
