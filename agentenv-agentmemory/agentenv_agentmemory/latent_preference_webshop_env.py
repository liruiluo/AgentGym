from __future__ import annotations

from typing import Any

from .latent_preference import VerifiedLatentPreferenceBundleProvider
from .memoryarena_webshop_env import MemoryArenaWebShopEnv
from .native_webshop_backend import NativeWebShopBackend


LATENT_PREFERENCE_SURFACE = "agentmemory_webshop_latent_preference_train_v1"


class LatentPreferenceWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime backed by verified hidden-preference tasks."""

    surface = LATENT_PREFERENCE_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedLatentPreferenceBundleProvider,
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
            if item.get("op")
            in {"ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER"}
        ]
        info.update(
            {
                "task_family": "procedural_latent_user_preference_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "recipe_id": bundle.recipe_id,
                "user_id": bundle.user_id,
                "preference_axis": bundle.preference_axis,
                "supporting_evidence_count": bundle.supporting_evidence_count,
                "preference_resolution_step": 1,
                "candidate_count_per_phase": 2,
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
                    "historical_confirmed_choice_on_named_preference_axis"
                ),
                "counterfactual_pairing": True,
                "paper_eligible": False,
            }
        )
        return info


def _public_tool_op(event: dict[str, Any]) -> dict[str, Any]:
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
