from __future__ import annotations

from typing import Any

from .compositional_recall import VerifiedCompositionalRecallBundleProvider
from .memoryarena_webshop_env import MemoryArenaWebShopEnv
from .native_webshop_backend import NativeWebShopBackend
from .procedural_webshop_env import _sanitized_evidence_tool_op


COMPOSITIONAL_RECALL_SURFACE = (
    "agentmemory_webshop_compositional_recall_top1_train_v1"
)


class CompositionalRecallWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime requiring two sequential query-based recalls."""

    surface = COMPOSITIONAL_RECALL_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedCompositionalRecallBundleProvider,
        backend: NativeWebShopBackend,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        if kwargs.setdefault("ltm_inventory_mode", "hidden") != "hidden":
            raise ValueError("compositional recall requires hidden LTM inventory.")
        if kwargs.setdefault("retrieve_policy", "query_top1") != "query_top1":
            raise ValueError("compositional recall requires query_top1 retrieval.")
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
            if item.get("op")
            in {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
        ]
        info.update(
            {
                "task_family": "procedural_compositional_recall_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "candidate_count_per_phase": 2,
                "canonical_memory_count": 2,
                "required_sequential_retrievals": 2,
                "leave_one_memory_out_certified": True,
                "purchase_eligibility_scope": (
                    "current_phase_two_approved_listings"
                ),
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_dependency": "customer_to_profile_to_attribute",
                "factorial_counterfactual": True,
                "paper_eligible": False,
            }
        )
        return info
