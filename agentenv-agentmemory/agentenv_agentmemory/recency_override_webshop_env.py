from __future__ import annotations

from typing import Any

from .memoryarena_webshop_env import MemoryArenaWebShopEnv
from .filesystem_webshop_env import PersistentWorkspaceWebShopEnv
from .native_webshop_backend import NativeWebShopBackend
from .procedural_webshop_env import _sanitized_evidence_tool_op
from .recency_override import VerifiedRecencyOverrideBundleProvider


RECENCY_OVERRIDE_SURFACE = "agentmemory_webshop_recency_override_train_v1"
RECENCY_OVERRIDE_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_recency_override_filesystem_v2"
)


class RecencyOverrideWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime backed by verified recency-override tasks."""

    surface = RECENCY_OVERRIDE_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedRecencyOverrideBundleProvider,
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
        info["memory_ops"] = [
            item
            for item in info["tool_ops"]
            if item.get("op") in {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
        ]
        # Branch labels and old/new values remain audit metadata, not policy
        # observations.  The visible application questions are identical across
        # stay/flip after the override phase; only the hidden target changes.
        info.update(
            {
                "task_family": "procedural_recency_override_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "recipe_id": bundle.recipe_id,
                "user_id": bundle.user_id,
                "candidate_count_per_phase": 2,
                "override_phase_index": 2,
                "canonical_memory_key": bundle.canonical_memory_key,
                "override_mode": bundle.override_mode,
                "branch_kind": bundle.branch_kind,
                "purchase_eligibility_scope": "current_phase_two_approved_listings",
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "global_catalog_attribute_uniqueness_claimed": False,
                "global_catalog_normalized_title_uniqueness_claimed": True,
                "memory_dependency": "latest_preference_overrides_conflicting_history",
                "counterfactual_pairing": True,
                "application_observation_identity_after_override": True,
                "application_target_flip_after_override": True,
                "paper_eligible": False,
            }
        )
        return info


class RecencyOverrideFilesystemWebShopEnv(PersistentWorkspaceWebShopEnv):
    """Verified recency tasks on the shared Codex filesystem-v2 surface."""

    surface = RECENCY_OVERRIDE_FILESYSTEM_SURFACE
    workspace_intervention_boundary_index = 3

    def __init__(
        self,
        *,
        provider: VerifiedRecencyOverrideBundleProvider,
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
            if item.get("op") in {"SHELL_COMMAND", "APPLY_PATCH"}
        ]
        info.update(
            {
                "task_family": "procedural_recency_override_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "recipe_id": bundle.recipe_id,
                "user_id": bundle.user_id,
                "candidate_count_per_phase": 2,
                "override_phase_index": 2,
                "canonical_memory_key": bundle.canonical_memory_key,
                "override_mode": bundle.override_mode,
                "branch_kind": bundle.branch_kind,
                "purchase_eligibility_scope": (
                    "current_phase_two_approved_listings"
                ),
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "global_catalog_attribute_uniqueness_claimed": False,
                "global_catalog_normalized_title_uniqueness_claimed": True,
                "memory_dependency": (
                    "latest_preference_overrides_conflicting_history"
                ),
                "memory_mechanism": "policy_authored_workspace_files",
                "counterfactual_pairing": True,
                "application_observation_identity_after_override": True,
                "application_target_flip_after_override": True,
                "paper_eligible": False,
            }
        )
        return info
