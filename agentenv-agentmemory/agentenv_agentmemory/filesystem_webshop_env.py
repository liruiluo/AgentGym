from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .memoryarena_webshop_env import (
    InvalidNativeAction,
    MemoryArenaWebShopEnv,
    ParsedAction,
    _render_native_actions,
    parse_mixed_action,
)
from .native_webshop_backend import NativeWebShopBackend
from .persistent_workspace import (
    PersistentWorkspace,
    WORKSPACE_TOOL_OPS,
    WorkspaceActionError,
    WorkspaceLimits,
    parse_workspace_action,
)
from .procedural import VerifiedProceduralBundleProvider
from .procedural_webshop_env import _sanitized_evidence_tool_op
from .reward_hierarchy import WRONG_BUY_TERMINAL_FAILURE


PROCEDURAL_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
)
RAW_INTERMEDIATE_BUY_REWARD = 1.0
RAW_FINAL_BUY_REWARD = 2.0
RAW_MAXIMUM_POSITIVE_TRAJECTORY_REWARD = 7.0


def validate_positive_task_reward_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("positive task reward scale must be a finite number > 0")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("positive task reward scale must be a finite number > 0")
    return scale


def build_filesystem_reward_contract(
    positive_task_reward_scale: float = 1.0,
) -> dict[str, Any]:
    scale = validate_positive_task_reward_scale(positive_task_reward_scale)
    return {
        "schema": "agentmemory_webshop_reward_contract_v3",
        "contract_id": "native_buy_positive_scale_zero_codex_workspace_tool_reward_v1",
        "workspace_action_reward": 0.0,
        "shell_command_reward": 0.0,
        "apply_patch_reward": 0.0,
        "memory_specific_shaping": "none",
        "wrong_buy_terminal_reward": WRONG_BUY_TERMINAL_FAILURE,
        "positive_task_reward_scale": scale,
        "raw_correct_intermediate_buy_reward": RAW_INTERMEDIATE_BUY_REWARD,
        "raw_correct_final_buy_reward": RAW_FINAL_BUY_REWARD,
        "raw_maximum_positive_trajectory_reward": (
            RAW_MAXIMUM_POSITIVE_TRAJECTORY_REWARD
        ),
        "correct_intermediate_buy_reward": RAW_INTERMEDIATE_BUY_REWARD * scale,
        "correct_final_buy_reward": RAW_FINAL_BUY_REWARD * scale,
        "maximum_positive_trajectory_reward": (
            RAW_MAXIMUM_POSITIVE_TRAJECTORY_REWARD * scale
        ),
    }


FILESYSTEM_REWARD_CONTRACT = build_filesystem_reward_contract()


class PersistentWorkspaceWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop plus an episode-scoped general-purpose workspace."""

    workspace_intervention_boundary_index = 1

    def __init__(
        self,
        *,
        bundles,
        backend: NativeWebShopBackend,
        env_uid: str | None = None,
        shell_sandbox,
        workspace_root_parent: Path | None = None,
        workspace_limits: WorkspaceLimits | None = None,
        positive_task_reward_scale: float = 1.0,
    ) -> None:
        self.positive_task_reward_scale = validate_positive_task_reward_scale(
            positive_task_reward_scale
        )
        self._last_raw_task_reward: float | None = None
        super().__init__(
            bundles=bundles,
            backend=backend,
            env_uid=env_uid,
            first_valid_add_reward=0.0,
            first_valid_later_session_retrieve_reward=0.0,
            ltm_inventory_mode="hidden",
            ltm_transition_notice_mode="none",
            action_listing_mode="separate",
            retrieve_policy="standard",
        )
        self.workspace = PersistentWorkspace(
            workspace_id=self.env_uid,
            shell_sandbox=shell_sandbox,
            root_parent=workspace_root_parent,
            limits=workspace_limits or WorkspaceLimits(),
        )
        self._workspace_enabled = True

    def set_workspace_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("workspace enabled flag must be boolean")
        self._workspace_enabled = enabled

    def install_workspace_causal_intervention(
        self,
        arm: str,
        *,
        state: dict[str, Any] | None = None,
        source_label: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self.done or self.status != "active":
            raise RuntimeError(
                "workspace intervention requires an active nonterminal episode"
            )
        if (
            self.current_session_index
            != self.workspace_intervention_boundary_index
        ):
            raise RuntimeError(
                "workspace intervention is frozen at session-"
                f"{self.workspace_intervention_boundary_index} boundary"
            )
        self.workspace.install_causal_intervention(
            arm,
            state=state,
            source_label=source_label,
        )
        self.last_tool_ops = []
        self.last_reward_components = []
        return self.render_observation(), self.build_info()

    def reset(self, seed: int | None = None, data_idx: int = 0):
        episode_id = f"{self.env_uid}:episode:{self.episode_counter + 1}"
        self.workspace.reset_episode(episode_id, enabled=self._workspace_enabled)
        try:
            return super().reset(seed=seed, data_idx=data_idx)
        except Exception:
            self.workspace.close()
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.workspace.close()

    def _parse_action(self, action: str) -> ParsedAction:
        try:
            workspace_action = parse_workspace_action(action)
        except WorkspaceActionError as exc:
            raise InvalidNativeAction(str(exc)) from exc
        if workspace_action is not None:
            return ParsedAction(
                op=workspace_action.tool_name.upper(),
                raw_action=action.strip(),
                # ParsedAction uses a non-None payload to route non-WebShop
                # actions through _step_memory. apply_patch carries native
                # patch text rather than JSON arguments, so an empty mapping
                # is the routing marker for that tool.
                payload=(
                    {}
                    if workspace_action.arguments is None
                    else dict(workspace_action.arguments)
                ),
            )
        try:
            parsed = parse_mixed_action(action)
        except InvalidNativeAction as exc:
            raise InvalidNativeAction(
                "Expected one native search[...] / click[...] action, or one canonical "
                "workspace action: shell_command {JSON} with the literal prefix and one "
                "space, or apply_patch followed by a newline patch. Bare JSON, markdown "
                "code fences, and explanations are invalid."
            ) from exc
        if parsed.payload is not None:
            raise InvalidNativeAction(
                "This filesystem-v2 surface does not expose ADD, RETRIEVE, or other memory-specific tools."
            )
        return parsed

    def _step_memory(self, parsed: ParsedAction) -> tuple[str, float, bool]:
        try:
            result = self.workspace.apply(
                parsed.raw_action,
                env_step=self.step_count,
                phase_index=self.current_session_index,
            )
        except WorkspaceActionError as exc:
            raise InvalidNativeAction(str(exc)) from exc
        if result is None:
            raise InvalidNativeAction("Expected one Codex workspace tool action.")
        self.last_tool_ops = [result.tool_op]
        self._append_trace(parsed.raw_action, result.message)
        return self.render_observation(result.message), 0.0, False

    def step(self, action: str):
        self._last_raw_task_reward = None
        observation, reward, terminated, truncated, info = super().step(action)
        raw_task_reward = (
            float(reward)
            if self._last_raw_task_reward is None
            else self._last_raw_task_reward
        )
        step_scale = (
            self.positive_task_reward_scale if raw_task_reward > 0.0 else 1.0
        )
        info = dict(info)
        info.update(
            {
                "raw_task_reward": raw_task_reward,
                "reward_scale": step_scale,
                "training_reward": float(reward),
            }
        )
        return observation, float(reward), terminated, truncated, info

    def _apply_session_action_shaping(
        self,
        *,
        parsed: ParsedAction,
        raw_action: str,
        reward: float,
        done: bool,
    ) -> float:
        del parsed, raw_action, done
        raw_reward = float(reward)
        self._last_raw_task_reward = raw_reward
        if raw_reward <= 0.0:
            return raw_reward

        scale = self.positive_task_reward_scale
        for component in self.last_reward_components:
            raw_value = float(component["value"])
            if raw_value <= 0.0:
                continue
            training_value = raw_value * scale
            component.update(
                {
                    "raw_task_reward": raw_value,
                    "reward_scale": scale,
                    "training_reward": training_value,
                    "value": training_value,
                }
            )
        return raw_reward * scale

    def render_observation(self, prefix: str | None = None) -> str:
        self._require_bundle()
        page = self._require_page()
        sections = []
        if prefix:
            sections.append(prefix.strip())
        sections.extend(
            [
                f"Task family: bundled_shopping\nProgress: {self.current_session_index}/6",
                page.observation.strip(),
                _render_native_actions(page),
                _render_session_trace(self.session_trace),
                self.workspace.render_contract(),
            ]
        )
        return "\n\n".join(section for section in sections if section)

    def build_info(self) -> dict[str, Any]:
        info = super().build_info()
        snapshot = self.workspace.snapshot()
        workspace_ops = [
            dict(item)
            for item in self.last_tool_ops
            if str(item.get("op", "")).upper() in WORKSPACE_TOOL_OPS
        ]
        for key in (
            "ltm_inventory_mode",
            "ltm_transition_notice_mode",
            "retrieve_policy",
            "ltm_inventory_count",
            "ltm_inventory_key_max_chars",
            "ltm_inventory_key_format",
            "memory_state_diff",
        ):
            info.pop(key, None)
        info.update(
            {
                "memory_ops": [],
                "workspace_ops": workspace_ops,
                "memory_management": "policy_managed_persistent_workspace",
                "workspace_surface": "codex_workspace_v2",
                "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
                "workspace_tool_ops": list(WORKSPACE_TOOL_OPS),
                "workspace_persistence": "episode_across_sessions",
                "workspace_episode_isolation": True,
                "workspace_host_path_exposed": False,
                "workspace_shell_enabled": self.workspace.enabled,
                "workspace_apply_patch_enabled": self.workspace.enabled,
                "workspace_intervention": (
                    "enabled" if self.workspace.enabled else "no_workspace"
                ),
                "workspace_causal_arm": self.workspace.causal_arm,
                "workspace_control_event": self.workspace.control_event,
                "workspace_snapshot": snapshot,
                "workspace_seed_manifest": self.workspace.seed_manifest,
                "workspace_provenance": self.workspace.provenance_summary,
                "workspace_audit_event_count": len(self.workspace.audit_events),
                "workspace_latest_event": workspace_ops[0] if workspace_ops else None,
                "reward_contract": self.reward_contract(),
            }
        )
        return info

    def reward_contract(self) -> dict[str, Any]:
        return build_filesystem_reward_contract(self.positive_task_reward_scale)


class ProceduralFilesystemWebShopEnv(PersistentWorkspaceWebShopEnv):
    """Verified natural-chain data on the filesystem-v2 action surface."""

    surface = PROCEDURAL_FILESYSTEM_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedProceduralBundleProvider,
        backend: NativeWebShopBackend,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        super().__init__(
            bundles=(provider.get(0),),
            backend=backend,
            **kwargs,
        )

    def _bundle_for_data_idx(self, data_idx: int):
        return self.provider.get(data_idx)

    def build_info(self) -> dict[str, Any]:
        info = super().build_info()
        bundle = self._require_bundle()
        info.pop("task_id", None)
        info.pop("purchase_history", None)
        if any(item.get("op") == "BUY" for item in self.last_tool_ops):
            info["session_trace"] = []
        info["tool_ops"] = [
            _sanitized_evidence_tool_op(item) for item in info["tool_ops"]
        ]
        info["workspace_ops"] = [
            item
            for item in info["tool_ops"]
            if str(item.get("op", "")).upper() in WORKSPACE_TOOL_OPS
        ]
        info.update(
            {
                "task_family": "procedural_natural_attribute_chain_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "scenario_id": bundle.scenario_id,
                "candidate_count_per_phase": 2,
                "purchase_eligibility_scope": "current_phase_two_approved_listings",
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "global_catalog_attribute_uniqueness_claimed": False,
                "global_catalog_normalized_title_uniqueness_claimed": True,
                "memory_dependency": "previous_purchased_natural_attribute",
                "memory_mechanism": "policy_authored_workspace_files",
                "paper_eligible": False,
            }
        )
        return info


def _render_session_trace(trace: list[str]) -> str:
    lines = ["Current-session action trace:"]
    lines.extend(f"- S{index}: {item}" for index, item in enumerate(trace))
    if not trace:
        lines.append("<empty>")
    return "\n".join(lines)
