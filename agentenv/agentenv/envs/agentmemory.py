from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

import requests

from agentenv.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
    format_code_as_action_prompt,
    format_function_call_prompt,
    parse_python_code_comments,
)
from agentenv.controller.types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    StepOutput,
)

AGENTMEMORY_FUNCTION_DESCRIPTION = [
    {
        "name": "search",
        "description": "Use the original WebShop search bar with policy-chosen keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "WebShop search keywords."},
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "click",
        "description": "Click one value currently exposed by the original WebShop page.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Current clickable value."},
            },
            "required": ["item"],
        },
    },
    {
        "name": "add",
        "description": "Store exactly the provided key and value in hidden long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key."},
                "value": {"type": "string", "description": "Memory value."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "update",
        "description": "Update an existing long-term memory by memory_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id such as mem_0000."},
                "key": {"type": "string", "description": "Optional replacement memory key."},
                "value": {"type": "string", "description": "New memory value."},
            },
            "required": ["memory_id", "value"],
        },
    },
    {
        "name": "delete",
        "description": "Delete an obsolete or wrong long-term memory by memory_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id such as mem_0000."},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "retrieve",
        "description": "Match a query against text previously stored with add and expose matching memories as active context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Retrieval query."},
                "top_k": {"type": "integer", "description": "Number of memories to retrieve."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "summary",
        "description": "Replace active context with policy-authored text grounded in visible context IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Summary text."},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Visible S*/C* context IDs used for the summary.",
                },
            },
            "required": ["text", "source_ids"],
        },
    },
    {
        "name": "filter",
        "description": "Keep or drop already visible short-term context IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "keep_ids": {"type": "array", "items": {"type": "string"}, "description": "Context IDs to keep, such as C0 or S0."},
                "drop_ids": {"type": "array", "items": {"type": "string"}, "description": "Context IDs to drop, such as C0 or S0."},
                "scope": {"type": "string", "description": "active, session, or all."},
            },
        },
    },
]

AGENTMEMORY_ASK_FUNCTION_DESCRIPTION = {
    "name": "ask",
    "description": (
        "Ask once for the single clarification field exposed by the current "
        "intent-clarification task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "The ambiguity-resolving field named by the task.",
            },
        },
        "required": ["field"],
        "additionalProperties": False,
    },
}

FUNCTION_TO_ACTION = {
    "search": "search",
    "click": "click",
    "add": "ADD",
    "update": "UPDATE",
    "delete": "DELETE",
    "retrieve": "RETRIEVE",
    "summary": "SUMMARY",
    "filter": "FILTER",
    "ask": "ASK",
}

MEMORY_ACTION_NAMES = ("ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER")
CLARIFICATION_ACTION_NAMES = ("ASK",)
NATIVE_ACTION_NAMES = ("search", "click")
JSON_ACTION_NAMES = (*MEMORY_ACTION_NAMES, *CLARIFICATION_ACTION_NAMES)
ACTION_NAMES = (*NATIVE_ACTION_NAMES, *JSON_ACTION_NAMES)
NATIVE_ACTION_RE = re.compile(r"\A(search|click)\[([^\[\]\r\n]+)\]\Z")
JSON_ACTION_RE = re.compile(
    r"\A(" + "|".join(JSON_ACTION_NAMES) + r")\s+(\{.*\})\Z",
    flags=re.DOTALL,
)
FORMAL_SCHEMA_V3 = "agentmemory_formal_step_v3"
WEBSHOP_V2_SURFACE = "memoryarena_webshop_native_v1"
PROCEDURAL_WEBSHOP_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_train_v1"
)
LATENT_PREFERENCE_WEBSHOP_SURFACE = (
    "agentmemory_webshop_latent_preference_train_v1"
)
RECENCY_OVERRIDE_WEBSHOP_SURFACE = (
    "agentmemory_webshop_recency_override_train_v1"
)
DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE = (
    "agentmemory_webshop_distractor_robustness_top1_train_v1"
)
COMPOSITIONAL_RECALL_WEBSHOP_SURFACE = (
    "agentmemory_webshop_compositional_recall_top1_train_v1"
)
INTENT_CLARIFICATION_WEBSHOP_SURFACE = (
    "agentmemory_webshop_intent_clarification_train_v1"
)
SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE = (
    "agentmemory_webshop_selective_memory_use_top1_train_v1"
)
PREFERENCE_WEBSHOP_SURFACES = frozenset(
    {LATENT_PREFERENCE_WEBSHOP_SURFACE, RECENCY_OVERRIDE_WEBSHOP_SURFACE}
)
QUERY_TOP1_WEBSHOP_SURFACES = frozenset(
    {
        DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE,
        COMPOSITIONAL_RECALL_WEBSHOP_SURFACE,
        INTENT_CLARIFICATION_WEBSHOP_SURFACE,
        SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE,
    }
)
LATENT_PREFERENCE_SOP_WEBSHOP_SURFACES = frozenset(
    {
        *PREFERENCE_WEBSHOP_SURFACES,
        DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE,
        COMPOSITIONAL_RECALL_WEBSHOP_SURFACE,
        INTENT_CLARIFICATION_WEBSHOP_SURFACE,
    }
)
PROGRAMMATIC_WEBSHOP_SURFACES = frozenset(
    {
        PROCEDURAL_WEBSHOP_SURFACE,
        SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE,
        *LATENT_PREFERENCE_SOP_WEBSHOP_SURFACES,
    }
)
LATENT_PREFERENCE_PROMPT_MODE = "latent_preference_sop"
SELECTIVE_MEMORY_PROMPT_MODE = "selective_memory_sop"
MEMORY_PROMPT_MODES = (
    "legacy",
    "neutral",
    "neutral_horizon",
    "neutral_horizon_responsibility",
    LATENT_PREFERENCE_PROMPT_MODE,
    SELECTIVE_MEMORY_PROMPT_MODE,
)
NEUTRAL_HORIZON_CONTEXT = (
    "This episode has six sequential shopping sessions. Later-session compatibility "
    "constraints may refer to products purchased in earlier sessions."
)
CROSS_SESSION_MEMORY_RESPONSIBILITY = (
    "Across shopping sessions, you are responsible for preserving and accessing any "
    "facts needed for later decisions."
)
LATENT_PREFERENCE_SOP = (
    "Each early evidence session may show which approved listing a customer "
    "confirmed. Treat each confirmed choice as preference evidence. Preserve the "
    "confirmed listing and its visible distinguishing attributes. After a confirmed "
    "choice is visible, use ADD before click[Buy Now] to store that evidence or "
    "create a customer-profile memory containing the customer, preference axis, and "
    "inferred value. When a customer-profile memory already exists, first retrieve "
    "its exact memory_id and use UPDATE to incorporate additional evidence without "
    "discarding prior support. Do not assume a fixed number of examples is always "
    "sufficient; infer a preference only when the visible confirmed choices support "
    "it. At the start of every later shopping session, use RETRIEVE to expose the "
    "relevant confirmed-choice evidence or customer profile. In later application "
    "sessions, apply the retrieved preference when choosing between approved "
    "listings. The environment does not perform these memory actions for you, and it "
    "does not reject an otherwise correct purchase when ADD was skipped."
)
SELECTIVE_MEMORY_SOP = (
    "First decide whether the current request already states every attribute needed "
    "to choose between its approved listings. When the current request is complete, "
    "follow it directly: explicit current requirements override profile history, and "
    "you should not ADD or RETRIEVE merely by habit. When the current request omits "
    "the customer's profile preference, use RETRIEVE to expose the saved current "
    "profile before choosing. Store new memory only when the episode provides new "
    "information that a later session will actually need."
)
QUERY_TOP1_RETRIEVAL_CONTRACT = (
    "On this surface, RETRIEVE requires exactly query:string and returns exactly "
    "one highest-ranked matching memory. memory_id and top_k are forbidden."
)
INTENT_CLARIFICATION_CONTRACT = (
    "In the first shopping session, the request is intentionally ambiguous. Use "
    'ASK {"field":"..."} exactly once with the ambiguity-resolving field named '
    "by the task. The environment returns a CLARIFY observation; store that "
    "clarification before the first purchase and retrieve it in later sessions."
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

# In thinking mode the model reasons inside a <think>...</think> block before the
# action. Depending on where the opening tag was emitted -- inside the response,
# or already in the chat-template generation prompt -- the captured text is either
# "<think>reasoning</think>\naction" or "reasoning</think>\naction". Either way,
# only the text after the final </think> is the action to execute.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)


def build_v3_conversation_start(
    metadata: Mapping[str, Any],
) -> tuple[ConversationMessage, ConversationMessage]:
    system_prompt = _required_metadata_text(metadata, "system_prompt")
    return (
        ConversationMessage({"from": "human", "loss": None, "value": system_prompt}),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )


def _required_metadata_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"AgentMemoryGym metadata requires non-empty {key}")
    return value.strip()


def _validate_v3_metadata(metadata: Mapping[str, Any]) -> None:
    for key in (
        "surface",
        "domain_id",
        "contract_id",
        "contract_sha256",
        "system_prompt_sha256",
    ):
        _required_metadata_text(metadata, key)
    contract_sha256 = str(metadata["contract_sha256"])
    if _SHA256_RE.fullmatch(contract_sha256) is None:
        raise RuntimeError(
            "AgentMemoryGym v3 metadata contract_sha256 must be lowercase SHA-256"
        )
    prompt_sha256 = str(metadata["system_prompt_sha256"])
    if _SHA256_RE.fullmatch(prompt_sha256) is None:
        raise RuntimeError(
            "AgentMemoryGym v3 metadata system_prompt_sha256 must be lowercase SHA-256"
        )
    system_prompt = _required_metadata_text(metadata, "system_prompt")
    observed_prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    if observed_prompt_sha256 != prompt_sha256:
        raise RuntimeError(
            "AgentMemoryGym v3 metadata system_prompt_sha256 does not match system_prompt"
        )
    native_actions = metadata.get("native_action_descriptions")
    if not isinstance(native_actions, list) or not native_actions or any(
        not isinstance(item, str) or not item.strip() for item in native_actions
    ):
        raise RuntimeError(
            "AgentMemoryGym v3 metadata requires native_action_descriptions"
        )
    if any(item.strip() not in system_prompt for item in native_actions):
        raise RuntimeError(
            "AgentMemoryGym v3 system_prompt must contain every native action form"
        )
    build_v3_conversation_start(metadata)


def _validate_programmatic_metadata(
    metadata: Mapping[str, Any],
    *,
    provider_schema: str,
    tasks_per_orbit: int = 2,
    seed_epoch_boundary_field: str = (
        "counterfactual_pair_never_crosses_seed_epoch"
    ),
) -> Mapping[str, Any]:
    if metadata.get("source") != "agentmemory_programmatic_generator":
        raise RuntimeError("Procedural AgentMemoryGym source metadata is invalid")
    if metadata.get("paper_eligible") is not False:
        raise RuntimeError("Procedural AgentMemoryGym must not be paper-eligible")
    provider = metadata.get("provider")
    if not isinstance(provider, Mapping):
        raise RuntimeError("Procedural AgentMemoryGym metadata requires provider")
    if provider.get("schema") != provider_schema:
        raise RuntimeError("Programmatic AgentMemoryGym provider schema is unsupported")
    if provider.get("tasks_per_orbit") != tasks_per_orbit:
        raise RuntimeError(
            "Programmatic AgentMemoryGym tasks_per_orbit metadata is invalid"
        )
    if provider.get("candidate_count_per_phase") != 2:
        raise RuntimeError("Procedural AgentMemoryGym requires two candidates per phase")
    if provider.get("phase_count_per_task") != 6:
        raise RuntimeError("Procedural AgentMemoryGym requires six phases per task")
    if provider.get("human_review_required") is not False:
        raise RuntimeError("Procedural AgentMemoryGym must not require human review")
    if provider.get("llm_judge_required") is not False:
        raise RuntimeError("Procedural AgentMemoryGym must not require an LLM judge")
    if provider.get("task_prompt_product_identity") != "complete_native_title":
        raise RuntimeError(
            "Procedural AgentMemoryGym task prompt product identity is invalid"
        )
    if provider.get("target_asin_in_task_prompt") is not False:
        raise RuntimeError(
            "Procedural AgentMemoryGym task prompt must not reveal the target ASIN"
        )
    if provider.get("native_search_result_asin_handles_visible") is not True:
        raise RuntimeError(
            "Procedural AgentMemoryGym must preserve native search-result ASIN handles"
        )
    if provider.get("native_click_action_uses_asin_handle") is not True:
        raise RuntimeError(
            "Procedural AgentMemoryGym must preserve native click[ASIN] actions"
        )
    provider_mode = provider.get("provider_mode")
    if provider_mode not in {"fixed_window", "reseeded_stream"}:
        raise RuntimeError("Procedural AgentMemoryGym provider mode is unsupported")
    task_count = metadata.get("task_count")
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
        or task_count % tasks_per_orbit
        or provider.get("task_count") != task_count
    ):
        raise RuntimeError("Procedural AgentMemoryGym task_count metadata is invalid")
    if metadata.get("provider_mode") != provider_mode:
        raise RuntimeError("Procedural AgentMemoryGym provider mode metadata disagrees")
    accepted_index_domain = provider.get("accepted_index_domain")
    if metadata.get("accepted_index_domain") != accepted_index_domain:
        raise RuntimeError(
            "Procedural AgentMemoryGym accepted index domain metadata disagrees"
        )
    semantic_period_orbits = provider.get("semantic_period_orbits")
    semantic_period_tasks = provider.get("semantic_period_tasks")
    if (
        isinstance(semantic_period_orbits, bool)
        or not isinstance(semantic_period_orbits, int)
        or semantic_period_orbits <= 0
        or isinstance(semantic_period_tasks, bool)
        or not isinstance(semantic_period_tasks, int)
        or semantic_period_tasks != semantic_period_orbits * tasks_per_orbit
    ):
        raise RuntimeError(
            "Procedural AgentMemoryGym semantic period metadata is invalid"
        )
    stream = provider.get("reseeded_stream")
    if provider_mode == "reseeded_stream":
        if accepted_index_domain != "all_nonnegative_integers":
            raise RuntimeError(
                "Procedural AgentMemoryGym stream must accept all non-negative indices"
            )
        if not isinstance(stream, Mapping):
            raise RuntimeError(
                "Procedural AgentMemoryGym stream metadata is missing"
            )
        expected_stream_values = {
            "tasks_per_seed_epoch": semantic_period_tasks,
            "orbits_per_seed_epoch": semantic_period_orbits,
            seed_epoch_boundary_field: True,
            "seed_epoch_zero_uses_base_seed": True,
            "collision_free_within_complete_seed_epoch": True,
            "semantic_uniqueness_guaranteed_through_task_index": (
                semantic_period_tasks - 1
            ),
            "cross_seed_epoch_semantic_uniqueness_guaranteed": False,
        }
        if any(stream.get(key) != value for key, value in expected_stream_values.items()):
            raise RuntimeError(
                "Procedural AgentMemoryGym stream epoch metadata is inconsistent"
            )
    elif stream is not None:
        raise RuntimeError(
            "Procedural AgentMemoryGym fixed windows must not expose stream metadata"
        )
    return provider


def _validate_procedural_metadata(metadata: Mapping[str, Any]) -> None:
    _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_natural_chain_provider_v4",
    )


def _validate_latent_preference_metadata(metadata: Mapping[str, Any]) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_latent_preference_provider_v1",
    )
    expected = {
        "supporting_evidence_counts": [1, 2, 3],
        "resolution_step": 1,
        "preference_hypothesis": "one_value_on_one_natural_attribute_axis",
        "counterfactual_pairing": True,
        "application_observation_identity": True,
        "application_target_flip": True,
        "purchase_receipt_asin_verification": True,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Latent-preference AgentMemoryGym provider metadata is inconsistent: "
            + ", ".join(mismatches)
        )


def _validate_recency_override_metadata(metadata: Mapping[str, Any]) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_recency_override_provider_v1",
    )
    expected = {
        "phase_schedule": [
            "evidence",
            "application",
            "override",
            "application",
            "application",
            "application",
        ],
        "override_phase_index": 2,
        "canonical_memory_key": "user_preference",
        "counterfactual_pairing": True,
        "stay_branch": "old preference remains active",
        "flip_branch": "new preference replaces old canonical state",
        "update_contract": "UPDATE same memory_id or DELETE old then ADD new",
        "application_observation_identity": True,
        "application_target_flip": True,
        "purchase_receipt_asin_verification": True,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Recency-override AgentMemoryGym provider metadata is inconsistent: "
            + ", ".join(mismatches)
        )


def _validate_distractor_robustness_metadata(
    metadata: Mapping[str, Any],
) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_distractor_robustness_provider_v1",
    )
    expected = {
        "counterfactual_pairing": True,
        "branch_order": ["clean", "distracted"],
        "correct_memory_preloaded": False,
        "correct_memory_policy_authored_after_evidence": True,
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "initial_memory_inventory_visible": False,
        "strict_top1_certified": True,
        "purchase_receipt_asin_verification": True,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Distractor-robustness AgentMemoryGym provider metadata is "
            "inconsistent: " + ", ".join(mismatches)
        )


def _validate_compositional_recall_metadata(
    metadata: Mapping[str, Any],
) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_compositional_recall_provider_v1",
        tasks_per_orbit=4,
        seed_epoch_boundary_field="factorial_orbit_never_crosses_seed_epoch",
    )
    expected = {
        "factorial_coordinates": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "canonical_memory_count": 2,
        "retrieve_policy": "query_top1",
        "required_sequential_retrievals": 2,
        "memory_id_lookup_allowed": False,
        "ltm_inventory_visible": False,
        "leave_one_memory_out_certified": True,
        "purchase_receipt_asin_verification": True,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Compositional-recall AgentMemoryGym provider metadata is "
            "inconsistent: " + ", ".join(mismatches)
        )


def _validate_intent_clarification_metadata(
    metadata: Mapping[str, Any],
) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_intent_clarification_provider_v1",
    )
    expected = {
        "counterfactual_pairing": True,
        "pre_ask_observation_identity": True,
        "all_targets_flip_after_clarification": True,
        "required_action": "ASK",
        "clarification_event": "CLARIFY",
        "ask_allowed_session": 0,
        "max_successful_asks": 1,
        "purchase_before_clarification_allowed": False,
        "canonical_memory_count": 1,
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "ltm_inventory_visible": False,
        "training_ready": True,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Intent-clarification AgentMemoryGym provider metadata is "
            "inconsistent: " + ", ".join(mismatches)
        )


def _validate_selective_memory_use_metadata(
    metadata: Mapping[str, Any],
) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_selective_memory_use_provider_v1",
        tasks_per_orbit=4,
        seed_epoch_boundary_field="factorial_orbit_never_crosses_seed_epoch",
    )
    expected = {
        "memory_required_fraction": 0.5,
        "memory_not_required_fraction": 0.5,
        "required_branch_seeded_memory_state": "current",
        "not_required_branch_seeded_memory_state": "stale_opposite",
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "ltm_inventory_visible": False,
        "memory_action_positive_shaping_allowed": False,
        "unnecessary_memory_action_penalty": -0.01,
        "memory_required_without_memory_counterfactually_ambiguous": True,
        "memory_not_required_current_request_explicit": True,
        "purchase_receipt_asin_verification": True,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Selective-memory-use AgentMemoryGym provider metadata is "
            "inconsistent: " + ", ".join(mismatches)
        )


def build_procedural_conversation_start(
    action_format: ActionFormat,
    memory_prompt_mode: str,
    *,
    surface: str = PROCEDURAL_WEBSHOP_SURFACE,
) -> tuple[ConversationMessage, ConversationMessage]:
    if memory_prompt_mode not in MEMORY_PROMPT_MODES:
        raise ValueError(
            "memory_prompt_mode must be one of: "
            + ", ".join(MEMORY_PROMPT_MODES)
            + "."
        )
    interface = (
        "You are operating a programmatically generated AgentMemoryGym WebShop "
        "training task with six separate shopping sessions. Native shopping "
        "actions are search[keywords] and click[current clickable value]; "
        "click[Buy Now] commits the current product. A successful purchase clears "
        "the native page and short-term S*/C* context before the next session. "
        "ADD stores the provided key/value verbatim in hidden long-term memory. "
        "RETRIEVE exposes matching stored memories as C* context. SUMMARY and "
        "FILTER operate only on visible S*/C* items. Long-term memory persists "
        "between sessions but remains hidden until RETRIEVE. A failed purchase "
        "ends the episode without revealing the expected answer."
    )
    if memory_prompt_mode in {"neutral_horizon", "neutral_horizon_responsibility"}:
        interface += " " + NEUTRAL_HORIZON_CONTEXT
    if memory_prompt_mode == "neutral_horizon_responsibility":
        interface += " " + CROSS_SESSION_MEMORY_RESPONSIBILITY
    if memory_prompt_mode == LATENT_PREFERENCE_PROMPT_MODE:
        interface += " " + LATENT_PREFERENCE_SOP
    elif memory_prompt_mode == SELECTIVE_MEMORY_PROMPT_MODE:
        interface += " " + SELECTIVE_MEMORY_SOP
    elif memory_prompt_mode == "legacy":
        interface += (
            " Preserve and retrieve any visible product attribute that a later "
            "customer rule needs; the environment does not perform memory actions "
            "for you and does not require a particular key or schema."
        )
    if surface in QUERY_TOP1_WEBSHOP_SURFACES:
        interface += " " + QUERY_TOP1_RETRIEVAL_CONTRACT
    if surface == INTENT_CLARIFICATION_WEBSHOP_SURFACE:
        interface += " " + INTENT_CLARIFICATION_CONTRACT

    function_descriptions = list(AGENTMEMORY_FUNCTION_DESCRIPTION)
    if surface in QUERY_TOP1_WEBSHOP_SURFACES:
        function_descriptions = [
            (
                {
                    **description,
                    "description": (
                        "Match a query against text previously stored with add and "
                        "expose exactly the highest-ranked memory as active context."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Retrieval query.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                }
                if description["name"] == "retrieve"
                else description
            )
            for description in function_descriptions
        ]
    if surface == INTENT_CLARIFICATION_WEBSHOP_SURFACE:
        function_descriptions.append(AGENTMEMORY_ASK_FUNCTION_DESCRIPTION)

    if action_format is ActionFormat.REACT:
        prompt = (
            interface
            + " Reply in exactly this format:\n\nThought:\nbrief reasoning\n\n"
            "Action:\n<exactly one native bracket action or uppercase "
            "memory-tool JSON action>"
        )
    elif action_format is ActionFormat.FUNCTION_CALLING:
        prompt = (
            interface
            + " Invoke exactly one available function.\n\n"
            + format_function_call_prompt(function_descriptions)
        )
    elif action_format is ActionFormat.CODE_AS_ACTION:
        prompt = (
            interface
            + " Write Python code to call exactly one available function.\n\n"
            + format_code_as_action_prompt(function_descriptions)
        )
    else:  # pragma: no cover - ActionFormat is closed over the three modes above.
        raise ValueError(f"Unsupported AgentMemoryGym action format: {action_format}")
    return (
        ConversationMessage({"from": "human", "loss": None, "value": prompt}),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )


def strip_think_prefix(text: str) -> str:
    """Drop a leading chain-of-thought block, returning just the action portion.

    Everything up to and including the last </think> is reasoning and is removed.
    Text with no </think> (legacy no-thinking replies) is returned unchanged, so
    this is a safe no-op when thinking is disabled.
    """
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if matches:
        return text[matches[-1].end():]
    return text


class AgentMemoryAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym on the original MemoryArena WebShop surface. Native shopping actions are search[keywords] and click[current clickable value]; click[Buy Now] commits the current product. ADD stores the provided key/value verbatim in hidden long-term memory. RETRIEVE matches its query against text previously stored with ADD and exposes matches as C* context. SUMMARY and FILTER operate only on visible S*/C* items. A committed purchase that advances the session clears the native page state and S*/C* context; long-term memory remains hidden until RETRIEVE. Once you have selected the current product, use ADD before click[Buy Now] to save one concise memory containing its identity and visible compatibility-relevant attributes. At the start of every later shopping session, use RETRIEVE to expose the relevant prior-purchase memories before choosing a compatible product. The environment does not perform these memory actions for you and does not reject an otherwise correct purchase when ADD was skipped. A purchase that fails verification ends the episode without revealing the verifier reason. Reply in exactly this format:\n\nThought:\nbrief reasoning\n\nAction:\n<exactly one native bracket action or uppercase memory-tool JSON action>",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym on the original MemoryArena WebShop surface. Long-term memory is hidden unless you retrieve it. Before committing the selected product, use add to save its identity and visible compatibility-relevant attributes; at the start of every later shopping session, use retrieve to expose the relevant prior-purchase memories before choosing. The environment does not enforce add-before-buy. Invoke exactly one available function.\n\n"
                    + format_function_call_prompt(AGENTMEMORY_FUNCTION_DESCRIPTION),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym on the original MemoryArena WebShop surface. Long-term memory is hidden unless you retrieve it. Before committing the selected product, use add to save its identity and visible compatibility-relevant attributes; at the start of every later shopping session, use retrieve to expose the relevant prior-purchase memories before choosing. The environment does not enforce add-before-buy. Write Python code to call exactly one available function.\n\n"
                    + format_code_as_action_prompt(AGENTMEMORY_FUNCTION_DESCRIPTION),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }
    neutral_conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym on the original MemoryArena WebShop surface. Native shopping actions are search[keywords] and click[current clickable value]; click[Buy Now] commits the current product. ADD stores the provided key/value verbatim in hidden long-term memory. RETRIEVE matches its query against text previously stored with ADD and exposes matches as C* context. SUMMARY and FILTER operate only on visible S*/C* items. A committed purchase that advances the session clears the native page state and S*/C* context; long-term memory remains hidden until RETRIEVE. A purchase that fails verification ends the episode without revealing the verifier reason. Reply in exactly this format:\n\nThought:\nbrief reasoning\n\nAction:\n<exactly one native bracket action or uppercase memory-tool JSON action>",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym on the original MemoryArena WebShop surface. Long-term memory persists across shopping sessions and is hidden unless retrieve exposes it. The available functions define the browser and memory interfaces. Invoke exactly one available function.\n\n"
                    + format_function_call_prompt(AGENTMEMORY_FUNCTION_DESCRIPTION),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym on the original MemoryArena WebShop surface. Long-term memory persists across shopping sessions and is hidden unless retrieve exposes it. The available functions define the browser and memory interfaces. Write Python code to call exactly one available function.\n\n"
                    + format_code_as_action_prompt(AGENTMEMORY_FUNCTION_DESCRIPTION),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }
    neutral_horizon_conversation_start_dict = {
        action_format: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": prompt[0]["value"] + " " + NEUTRAL_HORIZON_CONTEXT,
                }
            ),
            prompt[1],
        )
        for action_format, prompt in neutral_conversation_start_dict.items()
    }
    neutral_horizon_responsibility_conversation_start_dict = {
        action_format: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": prompt[0]["value"]
                    + " "
                    + CROSS_SESSION_MEMORY_RESPONSIBILITY,
                }
            ),
            prompt[1],
        )
        for action_format, prompt in neutral_horizon_conversation_start_dict.items()
    }

    @classmethod
    def conversation_start_for_mode(
        cls,
        action_format: ActionFormat,
        memory_prompt_mode: str,
    ) -> tuple[ConversationMessage, ConversationMessage]:
        if memory_prompt_mode not in MEMORY_PROMPT_MODES:
            raise ValueError(
                "memory_prompt_mode must be one of: "
                + ", ".join(MEMORY_PROMPT_MODES)
                + "."
            )
        if memory_prompt_mode == "neutral_horizon_responsibility":
            prompts = cls.neutral_horizon_responsibility_conversation_start_dict
        elif memory_prompt_mode == "neutral_horizon":
            prompts = cls.neutral_horizon_conversation_start_dict
        elif memory_prompt_mode == "neutral":
            prompts = cls.neutral_conversation_start_dict
        else:
            prompts = cls.conversation_start_dict
        return prompts[action_format]

    @staticmethod
    def parse_react(text: str) -> ActionWithTought:
        # Thinking mode emits "<think>reasoning</think>\n<action>". Separate the
        # reasoning first: a stray "Action:"/"search["/"click[" inside the
        # reasoning must not be mistaken for the executable action. The removed
        # reasoning is kept as the thought. For legacy no-thinking replies
        # strip_think_prefix is a no-op, so behaviour is unchanged.
        action_text = strip_think_prefix(text)
        think_thought = ""
        if action_text != text:
            removed = text[: len(text) - len(action_text)]
            think_thought = re.sub(r"</?think\s*>", "", removed, flags=re.IGNORECASE).strip()

        if "Action:" not in action_text:
            bare_action = extract_bare_env_action(action_text)
            if bare_action:
                return ActionWithTought(thought=think_thought, action=bare_action)
        parsed = BaseAdapter.parse_react(action_text)
        if parsed.action:
            try:
                action_name, arguments = parse_env_action(parsed.action)
                return ActionWithTought(
                    thought=parsed.thought or think_thought,
                    action=format_action(action_name, arguments),
                )
            except Exception:
                return ActionWithTought(
                    thought=parsed.thought or think_thought,
                    action="",
                )
        bare_action = extract_bare_env_action(action_text)
        if bare_action:
            return ActionWithTought(thought=parsed.thought or think_thought, action=bare_action)
        return ActionWithTought(thought=parsed.thought or think_thought, action="")

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        fn_call = json.loads("{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False)
        thought = fn_call["thought"]
        function_name = fn_call["function_name"]
        arguments = fn_call["arguments"]
        action_name = FUNCTION_TO_ACTION.get(str(function_name).lower())
        if action_name is None:
            raise ValueError(f"Invalid function name: {function_name}")
        return ActionWithTought(thought=thought, action=format_action(action_name, arguments))

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        action_name, arguments = parse_env_action(action_with_thought.action)
        function_name = action_name.lower()
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": function_name,
                "arguments": arguments,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        code = extract_python_code_blocks(text)
        action = eval(code, {}, build_code_action_functions())
        thought = parse_python_code_comments(code)
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        action_name, arguments = parse_env_action(action_with_thought.action)
        function_name = action_name.lower()
        return f"```python\n# {action_with_thought.thought}\n{function_name}(**{repr(arguments)})\n```"


class AgentMemoryEnvClient(BaseEnvClient):
    adapter_cls = AgentMemoryAdapter
    requires_ephemeral_context = True
    rollout_context_policy = "latest_observation_only"
    is_v3 = False

    def __init__(
        self,
        env_server_base: str,
        data_len: int | None,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.metadata = self.get_metadata()
        self.surface = _required_metadata_text(self.metadata, "surface")
        self.formal_schema_version = self.metadata.get("formal_schema_version")
        self.domain_id = self.metadata.get("domain_id")
        self.system_prompt = self.metadata.get("system_prompt")
        self.contract_id = self.metadata.get("contract_id")
        self.contract_sha256 = self.metadata.get("contract_sha256")
        self.system_prompt_sha256 = self.metadata.get("system_prompt_sha256")
        self.memory_prompt_mode: str | None = None
        self.is_v3 = self.formal_schema_version == FORMAL_SCHEMA_V3
        self.is_procedural = self.surface in PROGRAMMATIC_WEBSHOP_SURFACES
        self.is_latent_preference = (
            self.surface == LATENT_PREFERENCE_WEBSHOP_SURFACE
        )
        self.is_recency_override = (
            self.surface == RECENCY_OVERRIDE_WEBSHOP_SURFACE
        )
        self.is_distractor_robustness = (
            self.surface == DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE
        )
        self.is_compositional_recall = (
            self.surface == COMPOSITIONAL_RECALL_WEBSHOP_SURFACE
        )
        self.is_intent_clarification = (
            self.surface == INTENT_CLARIFICATION_WEBSHOP_SURFACE
        )
        self.is_selective_memory_use = (
            self.surface == SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE
        )
        self.is_preference_memory = self.surface in PREFERENCE_WEBSHOP_SURFACES
        self.requires_latent_preference_sop = (
            self.surface in LATENT_PREFERENCE_SOP_WEBSHOP_SURFACES
        )
        self.requires_selective_memory_sop = self.is_selective_memory_use
        if self.is_v3:
            _validate_v3_metadata(self.metadata)
            if self.action_format is not ActionFormat.REACT:
                raise RuntimeError(
                    "AgentMemoryGym v3 domains currently require action_format='react'"
                )
        elif self.is_procedural:
            if self.is_latent_preference:
                _validate_latent_preference_metadata(self.metadata)
            elif self.is_recency_override:
                _validate_recency_override_metadata(self.metadata)
            elif self.is_distractor_robustness:
                _validate_distractor_robustness_metadata(self.metadata)
            elif self.is_compositional_recall:
                _validate_compositional_recall_metadata(self.metadata)
            elif self.is_intent_clarification:
                _validate_intent_clarification_metadata(self.metadata)
            elif self.is_selective_memory_use:
                _validate_selective_memory_use_metadata(self.metadata)
            else:
                _validate_procedural_metadata(self.metadata)
            memory_prompt_mode = self.metadata.get("memory_prompt_mode", "legacy")
            if memory_prompt_mode not in MEMORY_PROMPT_MODES:
                raise RuntimeError(
                    "AgentMemoryGym procedural metadata has unsupported "
                    f"memory_prompt_mode: {memory_prompt_mode!r}"
                )
            if self.requires_latent_preference_sop and (
                memory_prompt_mode != LATENT_PREFERENCE_PROMPT_MODE
            ):
                raise RuntimeError(
                    "AgentMemoryGym surface requires "
                    f"memory_prompt_mode={LATENT_PREFERENCE_PROMPT_MODE!r}"
                )
            if self.requires_selective_memory_sop and (
                memory_prompt_mode != SELECTIVE_MEMORY_PROMPT_MODE
            ):
                raise RuntimeError(
                    "AgentMemoryGym selective-memory-use surface requires "
                    f"memory_prompt_mode={SELECTIVE_MEMORY_PROMPT_MODE!r}"
                )
            if (
                not self.requires_latent_preference_sop
                and not self.requires_selective_memory_sop
                and memory_prompt_mode in {
                    LATENT_PREFERENCE_PROMPT_MODE,
                    SELECTIVE_MEMORY_PROMPT_MODE,
                }
            ):
                raise RuntimeError(
                    "AgentMemoryGym natural-chain surface cannot use the "
                    "specialized memory prompt"
                )
            self.memory_prompt_mode = memory_prompt_mode
        elif self.surface != WEBSHOP_V2_SURFACE:
            raise RuntimeError(
                "AgentMemoryGym client received an unsupported legacy surface without "
                f"the v3 schema: {self.surface!r}"
            )
        else:
            memory_prompt_mode = self.metadata.get("memory_prompt_mode", "legacy")
            if memory_prompt_mode not in MEMORY_PROMPT_MODES:
                raise RuntimeError(
                    "AgentMemoryGym WebShop metadata has unsupported "
                    f"memory_prompt_mode: {memory_prompt_mode!r}"
                )
            self.memory_prompt_mode = memory_prompt_mode
        self.data_len = data_len if data_len is not None else int(self.metadata["task_count"])
        response = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if response.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {response}")
        created = response.json()
        self.env_id = created["id"]
        self.info = {
            "observation": created["observation"],
            "reward": created["reward"],
            "done": created["done"],
            "env_info": created.get("info", {}),
            "metadata": self.metadata,
        }
        self.last_action_submission: dict[str, str] | None = None
        self.conversation_start = (
            build_v3_conversation_start(self.metadata)
            if self.is_v3
            else build_procedural_conversation_start(
                self.action_format,
                self.memory_prompt_mode,
                surface=self.surface,
            )
            if self.is_procedural
            else self.adapter_cls.conversation_start_for_mode(
                self.action_format,
                self.memory_prompt_mode,
            )
        )

    def __len__(self):
        return self.data_len

    def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        response = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        assert response.status_code == 200
        return response.json()

    def get_metadata(self) -> dict[str, Any]:
        response = requests.get(f"{self.env_server_base}/metadata", timeout=self.timeout)
        if response.status_code != 200:
            raise requests.RequestException(
                f"Failed to fetch AgentMemoryGym metadata: {response}"
            )
        return response.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        raw_policy_output = action
        parser_status = "server_native_v3"
        if action.endswith("</s>"):
            action = action[:-4]
        if self.is_v3:
            # V3 native grammars are domain-owned. Preserve the exact sampled text
            # so the server records and judges the same action PPO generated.
            parsed_action = action
        else:
            try:
                parsed_action = self.adapter_cls.action_parser(
                    action,
                    self.action_format,
                )
                parser_status = "adapter_parsed"
            except Exception:
                # WebShop remains server-authoritative for invalid sampled text.
                parsed_action = action
                parser_status = "raw_fallback"
            if not isinstance(parsed_action, str) or not parsed_action.strip():
                parsed_action = action
                parser_status = "raw_fallback"
        response = self.post("step", {"action": parsed_action})
        self.last_action_submission = {
            "raw_policy_output": raw_policy_output,
            "submitted_action": parsed_action,
            "parser_status": parser_status,
        }
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
            "env_info": response.get("info", {}),
            "metadata": self.metadata,
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    @property
    def sample_excluded(self) -> bool:
        return bool(self.info.get("env_info", {}).get("sample_excluded", False))

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self.post("reset", {"data_idx": idx})
        self.last_action_submission = None
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
            "env_info": response.get("info", {}),
            "metadata": self.metadata,
        }
        return response

    def close(self):
        return self.post("close", {})


class AgentMemoryTask(BaseTask):
    env_client_cls = AgentMemoryEnvClient
    env_name = "AgentMemoryGym"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        *args,
        n_clients: int = 1,
        **kwargs,
    ) -> None:
        del args, kwargs
        super().__init__(client_args=client_args, n_clients=n_clients)


def format_action(action_name: str, arguments: dict[str, Any]) -> str:
    if action_name == "search":
        return f"search[{_require_function_text(arguments, 'keywords')}]"
    if action_name == "click":
        return f"click[{_require_function_text(arguments, 'item')}]"
    if action_name not in JSON_ACTION_NAMES:
        raise ValueError(f"Unsupported AgentMemoryGym action: {action_name}")
    return f"{action_name} {json.dumps(arguments, ensure_ascii=False)}"


def extract_bare_env_action(text: str) -> str:
    # Remove any leading <think>...</think> reasoning so only the action remains.
    cleaned = strip_think_prefix(text).strip()
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    # The entire post-thinking remainder must be exactly one action. This keeps
    # multi-line memory JSON intact while rejecting extra prefix/suffix text.
    try:
        action_name, arguments = parse_env_action(cleaned)
        return format_action(action_name, arguments)
    except Exception:
        return ""


def parse_env_action(action: str) -> tuple[str, dict[str, Any]]:
    cleaned = action.strip()
    native_match = NATIVE_ACTION_RE.fullmatch(cleaned)
    if native_match is not None:
        argument = native_match.group(2).strip()
        key = "keywords" if native_match.group(1) == "search" else "item"
        return native_match.group(1), {key: argument}
    json_match = JSON_ACTION_RE.fullmatch(cleaned)
    if json_match is None:
        raise ValueError(
            "Expected one native bracket action or uppercase JSON tool action."
        )
    payload = json.loads(json_match.group(2))
    if not isinstance(payload, dict):
        raise ValueError("JSON tool payload must be a JSON object.")
    return json_match.group(1), payload


def build_code_action_functions() -> dict[str, Any]:
    def make_function(action_name: str):
        def code_action(**kwargs):
            return format_action(action_name, kwargs)

        return code_action

    return {function_name: make_function(action_name) for function_name, action_name in FUNCTION_TO_ACTION.items()}


def _require_function_text(arguments: dict[str, Any], key: str) -> str:
    if set(arguments) != {key}:
        raise ValueError(f"Expected exactly one function argument: {key}")
    value = arguments[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Function argument {key} must be a non-empty string.")
    if any(char in value for char in "[]\r\n"):
        raise ValueError(f"Function argument {key} contains an invalid bracket or newline.")
    return value.strip()
