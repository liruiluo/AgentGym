from __future__ import annotations

from typing import Any

from .memoryarena_webshop_env import MemoryArenaWebShopEnv
from .native_webshop_backend import NativeWebShopBackend
from .procedural import VerifiedProceduralBundleProvider


PROCEDURAL_SURFACE = "agentmemory_webshop_procedural_natural_chain_train_v1"


class ProceduralMemoryWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime backed by verified on-demand memory tasks."""

    surface = PROCEDURAL_SURFACE

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
        info["tool_ops"] = [_public_tool_op(item) for item in info["tool_ops"]]
        info["memory_ops"] = [
            item
            for item in info["tool_ops"]
            if item.get("op") in {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
        ]
        info.update(
            {
                "task_family": "procedural_natural_attribute_chain_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "scenario_id": bundle.scenario_id,
                "candidate_count_per_phase": 2,
                "purchase_eligibility_scope": "current_phase_two_approved_listings",
                "policy_visible_product_identity": "complete_native_title",
                "asin_policy_visible": False,
                "purchase_receipt_asin_verification": True,
                "global_catalog_attribute_uniqueness_claimed": False,
                "global_catalog_normalized_title_uniqueness_claimed": True,
                "memory_dependency": "previous_purchased_natural_attribute",
                "paper_eligible": False,
            }
        )
        return info


def _public_tool_op(event: dict[str, Any]) -> dict[str, Any]:
    """Remove purchase state that could replace cross-session memory."""

    if event.get("op") != "BUY":
        return dict(event)
    allowed = {
        "op",
        "committed",
        "terminal",
        "session_advanced",
        "step",
        "session_index",
    }
    return {key: value for key, value in event.items() if key in allowed}
