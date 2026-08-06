from __future__ import annotations

import math
from typing import Any

from .memory_state import MemoryEntry
from .filesystem_webshop_env import PersistentWorkspaceWebShopEnv
from .memoryarena_webshop_env import (
    MEMORY_TOOL_OPS,
    MemoryArenaWebShopEnv,
    ParsedAction,
)
from .native_webshop_backend import NativeWebShopBackend
from .procedural_webshop_env import _sanitized_evidence_tool_op
from .reward_hierarchy import MICRO_ACTION_PENALTY
from .selective_memory_use import VerifiedSelectiveMemoryUseBundleProvider


SELECTIVE_MEMORY_USE_SURFACE = (
    "agentmemory_webshop_selective_memory_use_top1_train_v1"
)
SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_selective_memory_use_filesystem_v2"
)


class SelectiveMemoryUseWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime that rewards using memory only when instrumental."""

    surface = SELECTIVE_MEMORY_USE_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedSelectiveMemoryUseBundleProvider,
        backend: NativeWebShopBackend,
        unnecessary_memory_action_penalty: float = MICRO_ACTION_PENALTY,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        for reward_name in (
            "first_valid_add_reward",
            "first_valid_later_session_retrieve_reward",
        ):
            reward = float(kwargs.setdefault(reward_name, 0.0))
            if reward != 0.0:
                raise ValueError(
                    "selective memory use forbids positive memory-action shaping."
                )
        if (
            isinstance(unnecessary_memory_action_penalty, bool)
            or not isinstance(unnecessary_memory_action_penalty, (int, float))
            or not math.isfinite(float(unnecessary_memory_action_penalty))
            or float(unnecessary_memory_action_penalty) >= 0.0
        ):
            raise ValueError(
                "unnecessary_memory_action_penalty must be finite and negative."
            )
        self.unnecessary_memory_action_penalty = float(
            unnecessary_memory_action_penalty
        )
        if kwargs.setdefault("ltm_inventory_mode", "hidden") != "hidden":
            raise ValueError("selective memory use requires hidden LTM inventory.")
        if kwargs.setdefault("retrieve_policy", "query_top1") != "query_top1":
            raise ValueError("selective memory use requires query_top1 retrieval.")
        super().__init__(
            bundles=(provider.get(0),),
            backend=backend,
            **kwargs,
        )

    def _bundle_for_data_idx(self, data_idx: int):
        return self.provider.get(data_idx)

    def reset(self, seed: int | None = None, data_idx: int = 0):
        super().reset(seed=seed, data_idx=data_idx)
        bundle = self._require_bundle()
        item = bundle.initial_memory
        memory_id = f"mem_{self.memory_id_counter:04d}"
        self.memory_id_counter += 1
        self.long_term_memory[memory_id] = MemoryEntry(
            memory_id=memory_id,
            key=item.key,
            value=item.value,
            created_step=0,
            updated_step=0,
        )
        return self.render_observation(), self.build_info()

    def _apply_session_action_shaping(
        self,
        *,
        parsed: ParsedAction,
        raw_action: str,
        reward: float,
        done: bool,
    ) -> float:
        if done or float(reward) != 0.0 or parsed.op not in MEMORY_TOOL_OPS:
            return super()._apply_session_action_shaping(
                parsed=parsed,
                raw_action=raw_action,
                reward=reward,
                done=done,
            )
        bundle = self._require_bundle()
        if bundle.memory_requirement == "memory_required" and parsed.op == "RETRIEVE":
            return super()._apply_session_action_shaping(
                parsed=parsed,
                raw_action=raw_action,
                reward=reward,
                done=done,
            )

        occurrence = self.valid_zero_reward_action_counts_this_session.get(
            raw_action,
            0,
        ) + 1
        self.valid_zero_reward_action_counts_this_session[raw_action] = occurrence
        self.last_reward_components.append(
            {
                "name": "memory_action_not_required",
                "value": self.unnecessary_memory_action_penalty,
                "op": parsed.op,
                "step": self.step_count,
                "raw_action": raw_action,
                "session_index": self.current_session_index,
                "occurrence": occurrence,
                "memory_requirement": bundle.memory_requirement,
            }
        )
        return float(reward) + self.unnecessary_memory_action_penalty

    def reward_contract(self) -> dict[str, Any]:
        contract = super().reward_contract()
        contract.update(
            {
                "selective_memory_use": True,
                "positive_memory_action_shaping_allowed": False,
                "unnecessary_memory_action_penalty": (
                    self.unnecessary_memory_action_penalty
                ),
                "required_branch_instrumental_retrieve_reward": 0.0,
            }
        )
        return contract

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
        info["memory_ops"] = [
            item
            for item in info["tool_ops"]
            if item.get("op") in MEMORY_TOOL_OPS
        ]
        info.update(
            {
                "task_family": "procedural_selective_memory_use_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "memory_requirement": bundle.memory_requirement,
                "preloaded_memory_count": 1,
                "preloaded_memory_values_visible": False,
                "preloaded_memory_diff_visible": False,
                "candidate_count_per_phase": 2,
                "purchase_eligibility_scope": (
                    "current_phase_two_approved_listings"
                ),
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_dependency": "selective_use_or_abstain_query_top1",
                "factorial_pairing": True,
                "paper_eligible": False,
            }
        )
        return info


class SelectiveMemoryUseFilesystemWebShopEnv(PersistentWorkspaceWebShopEnv):
    """Selective-memory control with a branch-conditioned ordinary profile file."""

    surface = SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE
    workspace_intervention_boundary_index = 1

    def __init__(
        self,
        *,
        provider: VerifiedSelectiveMemoryUseBundleProvider,
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

    def reset(self, seed: int | None = None, data_idx: int = 0):
        observation, info = super().reset(seed=seed, data_idx=data_idx)
        bundle = self._require_bundle()
        state_label = (
            "current" if bundle.memory_requirement == "memory_required" else "stale"
        )
        self.workspace.install_seed_files(
            {
                ".agent_memory/profile.md": (
                    f"Profile preference: {bundle.initial_memory.value}\n"
                )
            },
            source_label=(
                "selective_memory_branch_conditioned_profile_"
                f"{state_label}_v1"
            ),
        )
        return self.render_observation(), self.build_info()

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
            if item.get("op") in {"SHELL_COMMAND", "APPLY_PATCH"}
        ]
        info.update(
            {
                "task_family": "procedural_selective_memory_use_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "memory_requirement": bundle.memory_requirement,
                "preloaded_memory_count": 1,
                "preloaded_memory_values_visible": False,
                "preloaded_memory_diff_visible": False,
                "candidate_count_per_phase": 2,
                "purchase_eligibility_scope": (
                    "current_phase_two_approved_listings"
                ),
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_dependency": "selective_use_or_abstain_workspace_read",
                "factorial_pairing": True,
                "memory_mechanism": "harness_seeded_workspace_profile",
                "paper_eligible": False,
            }
        )
        return info
