from __future__ import annotations

from typing import Any

from .distractor_robustness import VerifiedDistractorRobustnessBundleProvider
from .filesystem_webshop_env import PersistentWorkspaceWebShopEnv
from .memory_state import MemoryEntry
from .memoryarena_webshop_env import MemoryArenaWebShopEnv
from .native_webshop_backend import NativeWebShopBackend
from .procedural_webshop_env import _sanitized_evidence_tool_op


DISTRACTOR_ROBUSTNESS_SURFACE = (
    "agentmemory_webshop_distractor_robustness_top1_train_v1"
)
DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_distractor_robustness_filesystem_v2"
)


class DistractorRobustnessWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime with hidden distractors and query-only top1 recall."""

    surface = DISTRACTOR_ROBUSTNESS_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedDistractorRobustnessBundleProvider,
        backend: NativeWebShopBackend,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        if kwargs.setdefault("ltm_inventory_mode", "hidden") != "hidden":
            raise ValueError("distractor robustness requires hidden LTM inventory.")
        if kwargs.setdefault("retrieve_policy", "query_top1") != "query_top1":
            raise ValueError("distractor robustness requires query_top1 retrieval.")
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
        for item in bundle.initial_memories:
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
                "task_family": "procedural_distractor_robustness_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "preloaded_distractor_count": len(bundle.initial_memories),
                "preloaded_memory_values_visible": False,
                "preloaded_memory_diff_visible": False,
                "correct_memory_preloaded": False,
                "correct_memory_policy_authored_after_evidence": True,
                "candidate_count_per_phase": 2,
                "purchase_eligibility_scope": (
                    "current_phase_two_approved_listings"
                ),
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_dependency": "selective_query_top1_under_distractors",
                "counterfactual_pairing": False,
                "answer_preserving_robustness_pairing": True,
                "paper_eligible": False,
            }
        )
        return info


class DistractorRobustnessFilesystemWebShopEnv(PersistentWorkspaceWebShopEnv):
    """Selective retrieval with harness-seeded ordinary workspace files."""

    surface = DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE
    workspace_intervention_boundary_index = 1

    def __init__(
        self,
        *,
        provider: VerifiedDistractorRobustnessBundleProvider,
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
        super().reset(seed=seed, data_idx=data_idx)
        bundle = self._require_bundle()
        if bundle.initial_memories:
            self.workspace.install_seed_files(
                {
                    f".agent_memory/inbox/profile-{index:02d}.md": (
                        "# Saved profile note\n\n" + item.value.strip() + "\n"
                    )
                    for index, item in enumerate(bundle.initial_memories, start=1)
                },
                source_label=f"distractor_task:{bundle.task_id}",
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
                "task_family": "procedural_distractor_robustness_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "branch_kind": bundle.branch_kind,
                "preloaded_distractor_count": len(bundle.initial_memories),
                "preloaded_workspace_values_visible": False,
                "preloaded_workspace_diff_visible": False,
                "correct_memory_preloaded": False,
                "correct_memory_policy_authored_after_evidence": True,
                "candidate_count_per_phase": 2,
                "purchase_eligibility_scope": (
                    "current_phase_two_approved_listings"
                ),
                "task_prompt_product_identity": "complete_native_title",
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_dependency": (
                    "selective_filesystem_search_under_seeded_distractors"
                ),
                "memory_mechanism": "ordinary_workspace_files",
                "counterfactual_pairing": False,
                "answer_preserving_robustness_pairing": True,
                "paper_eligible": False,
            }
        )
        return info
