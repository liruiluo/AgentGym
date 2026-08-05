from __future__ import annotations

from typing import Any

from .filesystem_webshop_env import PersistentWorkspaceWebShopEnv
from .memoryarena_webshop_env import MemoryArenaWebShopEnv
from .native_webshop_backend import NativeWebShopBackend
from .negative_constraint import VerifiedNegativeConstraintBundleProvider
from .procedural_webshop_env import _sanitized_evidence_tool_op


NEGATIVE_CONSTRAINT_SURFACE = (
    "agentmemory_webshop_negative_constraint_top1_train_v1"
)
NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_negative_constraint_filesystem_v2"
)


class NegativeConstraintWebShopEnv(MemoryArenaWebShopEnv):
    """Three-candidate WebShop runtime for remembered exclusion constraints."""

    surface = NEGATIVE_CONSTRAINT_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedNegativeConstraintBundleProvider,
        backend: NativeWebShopBackend,
        allow_rules_only: bool = False,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        if not provider.generator.pool.native_certified and not allow_rules_only:
            raise ValueError(
                "negative-constraint runtime refuses a rules-only product pool."
            )
        if kwargs.setdefault("ltm_inventory_mode", "hidden") != "hidden":
            raise ValueError("negative constraint requires hidden LTM inventory.")
        if kwargs.setdefault("retrieve_policy", "query_top1") != "query_top1":
            raise ValueError("negative constraint requires query_top1 retrieval.")
        super().__init__(bundles=(provider.get(0),), backend=backend, **kwargs)

    def _bundle_for_data_idx(self, data_idx: int):
        return self.provider.get(data_idx)

    def build_info(self) -> dict[str, Any]:
        info = super().build_info()
        bundle = self._require_bundle()
        native_certified = self.provider.generator.pool.native_certified
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
            if item.get("op")
            in {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
        ]
        info.update(
            {
                "task_family": "procedural_negative_constraint_shopping",
                "source": "agentmemory_programmatic_rules_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "candidate_count_per_phase": 3,
                "distinct_attribute_values_per_phase": 3,
                "counterfactual_branch_count": 3,
                "retrieve_policy": "query_top1",
                "ltm_inventory_mode": "hidden",
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "rules_only": not native_certified,
                "native_certified": native_certified,
                "training_ready": native_certified,
                "paper_eligible": False,
            }
        )
        return info


class NegativeConstraintFilesystemWebShopEnv(PersistentWorkspaceWebShopEnv):
    """Standing exclusions remembered through policy-authored ordinary files."""

    surface = NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE
    workspace_intervention_boundary_index = 1

    def __init__(
        self,
        *,
        provider: VerifiedNegativeConstraintBundleProvider,
        backend: NativeWebShopBackend,
        allow_rules_only: bool = False,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        if not provider.generator.pool.native_certified and not allow_rules_only:
            raise ValueError(
                "negative-constraint runtime refuses a rules-only product pool."
            )
        super().__init__(bundles=(provider.get(0),), backend=backend, **kwargs)

    def _bundle_for_data_idx(self, data_idx: int):
        return self.provider.get(data_idx)

    def build_info(self) -> dict[str, Any]:
        info = super().build_info()
        bundle = self._require_bundle()
        native_certified = self.provider.generator.pool.native_certified
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
                "task_family": "procedural_negative_constraint_shopping",
                "source": "agentmemory_programmatic_rules_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "candidate_count_per_phase": 3,
                "distinct_attribute_values_per_phase": 3,
                "counterfactual_branch_count": 3,
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_dependency": "standing_never_accept_exclusions",
                "memory_mechanism": "policy_authored_workspace_files",
                "rules_only": not native_certified,
                "native_certified": native_certified,
                "training_ready": native_certified,
                "paper_eligible": False,
            }
        )
        return info
