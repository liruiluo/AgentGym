from __future__ import annotations

import json
import re
from typing import Any

from .formal_native_contract import build_reward_components
from .filesystem_webshop_env import PersistentWorkspaceWebShopEnv
from .intent_clarification import VerifiedIntentClarificationBundleProvider
from .memoryarena_webshop_env import (
    InvalidNativeAction,
    MemoryArenaWebShopEnv,
    ParsedAction,
)
from .native_webshop_backend import NativeWebShopBackend
from .procedural_webshop_env import _sanitized_evidence_tool_op
from .reward_hierarchy import INVALID_ACTION_PENALTY


INTENT_CLARIFICATION_SURFACE = (
    "agentmemory_webshop_intent_clarification_train_v1"
)
INTENT_CLARIFICATION_FILESYSTEM_SURFACE = (
    "agentmemory_webshop_intent_clarification_filesystem_v2"
)
_ASK_RE = re.compile(r"\AASK\s+(\{.*\})\Z", re.DOTALL)


def _parse_ask_object(action_text: str) -> dict[str, Any]:
    match = _ASK_RE.fullmatch(action_text)
    if match is None:
        raise InvalidNativeAction(
            'ASK requires exactly one JSON object, for example ASK {"field":"color"}.'
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise InvalidNativeAction(f"ASK payload is invalid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise InvalidNativeAction("ASK payload must be a JSON object.")
    return payload


class IntentClarificationWebShopEnv(MemoryArenaWebShopEnv):
    """Native WebShop runtime with one first-session clarification action."""

    surface = INTENT_CLARIFICATION_SURFACE

    def __init__(
        self,
        *,
        provider: VerifiedIntentClarificationBundleProvider,
        backend: NativeWebShopBackend,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        self.clarification_received = False
        self.clarification_request_count = 0
        if kwargs.setdefault("ltm_inventory_mode", "hidden") != "hidden":
            raise ValueError("intent clarification requires hidden LTM inventory.")
        if kwargs.setdefault("retrieve_policy", "query_top1") != "query_top1":
            raise ValueError("intent clarification requires query_top1 retrieval.")
        super().__init__(bundles=(provider.get(0),), backend=backend, **kwargs)

    def _bundle_for_data_idx(self, data_idx: int):
        return self.provider.get(data_idx)

    def reset(self, seed: int | None = None, data_idx: int = 0):
        self.clarification_received = False
        self.clarification_request_count = 0
        return super().reset(seed=seed, data_idx=data_idx)

    def step(self, action: str):
        action_text = action.strip() if isinstance(action, str) else repr(action)
        if action_text.startswith("ASK"):
            return self._step_ask(action_text)
        if (
            not self.clarification_received
            and self.current_session_index == 0
            and action_text.casefold() == "click[buy now]"
        ):
            return self._invalid_clarification_action(
                action_text,
                "The customer must clarify the requested field before purchase.",
                attempted_op="CLICK",
            )
        return super().step(action)

    def render_observation(self, prefix: str | None = None) -> str:
        observation = super().render_observation(prefix)
        if self.current_session_index == 0 and not self.clarification_received:
            observation += (
                "\n\nIntent clarification action:\n"
                '- ASK {"field":"..."} (infer the missing field from the request)'
            )
        return observation

    def build_info(self) -> dict[str, Any]:
        info = super().build_info()
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
                "task_family": "procedural_intent_clarification_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "candidate_count_per_phase": 2,
                "ask_allowed_session": 0,
                "ask_max_success_count": 1,
                "ask_completed": self.clarification_received,
                "ask_success_count": self.clarification_request_count,
                "clarification_result_event": "CLARIFY",
                "purchase_before_clarification_allowed": False,
                "retrieve_policy": "query_top1",
                "ltm_inventory_mode": "hidden",
                "counterfactual_pre_ask_observation_identity": True,
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "training_ready": True,
                "paper_eligible": False,
            }
        )
        return info

    def _step_ask(self, action_text: str):
        if self.done:
            return (
                self.render_terminal_observation("Episode is already done."),
                0.0,
                True,
                False,
                self.build_info(),
            )
        try:
            payload = _parse_ask_object(action_text)
        except InvalidNativeAction as exc:
            return self._invalid_clarification_action(
                action_text,
                str(exc),
                attempted_op="INVALID",
            )
        bundle = self._require_bundle()
        if set(payload) != {"field"}:
            return self._invalid_clarification_action(
                action_text,
                "ASK expects exactly the field property.",
            )
        if payload.get("field") != bundle.clarification_field:
            return self._invalid_clarification_action(
                action_text,
                "ASK requested a field that cannot resolve the current ambiguity.",
            )
        if self.current_session_index != 0:
            return self._invalid_clarification_action(
                action_text,
                "ASK is available only in the first shopping session.",
            )
        if self.clarification_received:
            return self._invalid_clarification_action(
                action_text,
                "The clarification has already been provided.",
            )

        self._prepare_custom_action()
        self.clarification_received = True
        self.clarification_request_count = 1
        message = f"CLARIFY: {bundle.clarification_answer}"
        self.last_tool_ops = [
            {
                "op": "CLARIFY",
                "request_op": "ASK",
                "field": bundle.clarification_field,
                "clarification_received": True,
                "step": self.step_count,
                "session_index": self.current_session_index,
            }
        ]
        self._append_trace(action_text, message)
        self.last_reward_components = build_reward_components(
            raw_action=action_text,
            reward=0.0,
            step=self.step_count,
            tool_ops=self.last_tool_ops,
        )
        return self.render_observation(message), 0.0, False, False, self.build_info()

    def _invalid_clarification_action(
        self,
        action_text: str,
        error: str,
        *,
        attempted_op: str = "ASK",
    ):
        if self.done:
            return (
                self.render_terminal_observation("Episode is already done."),
                0.0,
                True,
                False,
                self.build_info(),
            )
        self._prepare_custom_action()
        message = f"Invalid action: {error}"
        self._append_trace(action_text, message)
        self.last_reward_components = [
            {
                "name": "invalid_action",
                "value": INVALID_ACTION_PENALTY,
                "op": attempted_op,
                "step": self.step_count,
                "raw_action": action_text,
                "error": error,
            }
        ]
        return (
            self.render_observation(message),
            float(INVALID_ACTION_PENALTY),
            False,
            False,
            self.build_info(),
        )

    def _prepare_custom_action(self) -> None:
        self.step_count += 1
        self.last_tool_ops = []
        self.last_reward_components = []
        self.last_memory_diff = {"added": [], "updated": [], "deleted": []}


class IntentClarificationFilesystemWebShopEnv(PersistentWorkspaceWebShopEnv):
    """Intent clarification plus the shared Codex-style workspace contract."""

    surface = INTENT_CLARIFICATION_FILESYSTEM_SURFACE
    workspace_intervention_boundary_index = 1

    def __init__(
        self,
        *,
        provider: VerifiedIntentClarificationBundleProvider,
        backend: NativeWebShopBackend,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        self.clarification_received = False
        self.clarification_request_count = 0
        super().__init__(
            bundles=(provider.get(0),),
            backend=backend,
            **kwargs,
        )

    def _bundle_for_data_idx(self, data_idx: int):
        return self.provider.get(data_idx)

    def reset(self, seed: int | None = None, data_idx: int = 0):
        self.clarification_received = False
        self.clarification_request_count = 0
        return super().reset(seed=seed, data_idx=data_idx)

    def _parse_action(self, action: str) -> ParsedAction:
        action_text = action.strip() if isinstance(action, str) else repr(action)
        if (
            not self.clarification_received
            and self.current_session_index == 0
            and action_text.casefold() == "click[buy now]"
        ):
            raise InvalidNativeAction(
                "The customer must clarify the requested field before purchase."
            )
        if action_text.startswith("ASK"):
            payload = _parse_ask_object(action_text)
            return ParsedAction(op="ASK", raw_action=action_text, payload=payload)
        return super()._parse_action(action)

    def _step_memory(self, parsed: ParsedAction) -> tuple[str, float, bool]:
        if parsed.op == "ASK":
            return self._step_ask(parsed.raw_action)
        return super()._step_memory(parsed)

    def _step_ask(self, action_text: str) -> tuple[str, float, bool]:
        payload = _parse_ask_object(action_text)
        bundle = self._require_bundle()
        if set(payload) != {"field"}:
            raise InvalidNativeAction("ASK expects exactly the field property.")
        if payload.get("field") != bundle.clarification_field:
            raise InvalidNativeAction(
                "ASK requested a field that cannot resolve the current ambiguity."
            )
        if self.current_session_index != 0:
            raise InvalidNativeAction("ASK is available only in the first shopping session.")
        if self.clarification_received:
            raise InvalidNativeAction("The clarification has already been provided.")
        self.clarification_received = True
        self.clarification_request_count = 1
        message = f"CLARIFY: {bundle.clarification_answer}"
        self.last_tool_ops = [
            {
                "op": "CLARIFY",
                "request_op": "ASK",
                "field": bundle.clarification_field,
                "clarification_received": True,
                "step": self.step_count,
                "session_index": self.current_session_index,
            }
        ]
        self._append_trace(action_text, message)
        self.last_reward_components = build_reward_components(
            raw_action=action_text,
            reward=0.0,
            step=self.step_count,
            tool_ops=self.last_tool_ops,
        )
        return self.render_observation(message), 0.0, False

    def render_observation(self, prefix: str | None = None) -> str:
        observation = super().render_observation(prefix)
        if self.current_session_index == 0 and not self.clarification_received:
            observation += (
                "\n\nIntent clarification action:\n"
                '- ASK {"field":"..."} (infer the missing field from the request)'
            )
        return observation

    def build_info(self) -> dict[str, Any]:
        info = super().build_info()
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
                "task_family": "procedural_intent_clarification_shopping",
                "source": "agentmemory_programmatic_generator",
                "surface": self.surface,
                "candidate_count_per_phase": 2,
                "ask_allowed_session": 0,
                "ask_max_success_count": 1,
                "ask_completed": self.clarification_received,
                "ask_success_count": self.clarification_request_count,
                "clarification_result_event": "CLARIFY",
                "purchase_before_clarification_allowed": False,
                "counterfactual_pre_ask_observation_identity": True,
                "target_asin_in_task_prompt": False,
                "native_search_result_asin_handles_visible": True,
                "native_click_action_uses_asin_handle": True,
                "purchase_receipt_asin_verification": True,
                "memory_mechanism": "policy_authored_workspace_files",
                "training_ready": True,
                "paper_eligible": False,
            }
        )
        return info
