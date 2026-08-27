from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

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
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_PRESERVE,
    CONTEXT_OPERATION_REPLACE,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from .filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_MAX_BYTES,
    FILESYSTEM_CHECKPOINT_PATH,
    build_filesystem_checkpoint_read_retry_observation,
    build_filesystem_checkpoint_read_receipt,
    build_filesystem_checkpoint_receipt,
    build_filesystem_checkpoint_retry_observation,
    build_post_checkpoint_context,
    checkpoint_retry_ceiling_tokens,
    filesystem_checkpoint_action_completed,
    filesystem_checkpoint_failure_reason,
    filesystem_checkpoint_framing_sha256,
    filesystem_checkpoint_read_observed,
    filesystem_checkpoint_read_failure_reason,
    filesystem_workspace_action_request_sha256,
    filesystem_checkpoint_write_succeeded,
)
from .webshop_handoff import WEBSHOP_SESSION_HANDOFF_REQUEST

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

AGENTMEMORY_FILESYSTEM_FUNCTION_DESCRIPTION = [
    *AGENTMEMORY_FUNCTION_DESCRIPTION[:2],
    {
        "name": "shell_command",
        "description": (
            "Run one shell command inside the private, networkless, "
            "resource-bounded episode workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional workspace-relative working directory.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional wall-clock timeout in milliseconds.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply one Codex-style *** Begin Patch ... *** End Patch patch to "
            "workspace-relative files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "The complete multiline Codex patch payload.",
                },
            },
            "required": ["patch"],
            "additionalProperties": False,
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
FILESYSTEM_FUNCTION_TO_ACTION = {
    "search": "search",
    "click": "click",
    "shell_command": "shell_command",
    "apply_patch": "apply_patch",
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
FILESYSTEM_ACTION_NAMES = ("shell_command", "apply_patch")
FILESYSTEM_JSON_ACTION_RE = re.compile(
    r"\Ashell_command\s+(\{.*\})\Z",
    flags=re.DOTALL,
)
FILESYSTEM_ASK_ACTION_RE = re.compile(
    r"\AASK\s+(\{.*\})\Z",
    flags=re.DOTALL,
)
FILESYSTEM_APPLY_PATCH_PREFIX = "apply_patch\n"


def _optional_transition_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("transition counters must not be boolean")
    return int(value)


def _copy_policy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return normalized


def _reported_webshop_session(info: Mapping[str, Any], fallback: int) -> int:
    value = info.get("current_subtask_index")
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise ValueError("current_subtask_index must not be boolean")
    result = int(value)
    if result < 0:
        raise ValueError("current_subtask_index must be non-negative")
    return result


def _validate_webshop_session_advance(
    info: Mapping[str, Any],
    *,
    before: int,
    after: int,
    done: bool,
) -> bool:
    if after not in {before, before + 1}:
        raise RuntimeError(
            "WebShop session index must stay fixed or advance by exactly one"
        )
    tool_ops = info.get("tool_ops", ())
    if not isinstance(tool_ops, Sequence) or isinstance(tool_ops, (str, bytes)):
        raise RuntimeError("WebShop transition evidence must contain a tool_ops list")
    advances = [
        item
        for item in tool_ops
        if isinstance(item, Mapping) and item.get("session_advanced") is True
    ]
    advanced = after == before + 1
    if advanced:
        if len(advances) != 1:
            raise RuntimeError(
                "WebShop session advance requires exactly one authoritative BUY record"
            )
        buy = advances[0]
        if (
            str(buy.get("op", "")).upper() != "BUY"
            or buy.get("committed") is not True
            or buy.get("purchase_correct") is not True
        ):
            raise RuntimeError("WebShop session advance has invalid BUY evidence")
        if not done and info.get("session_trace") != []:
            raise RuntimeError(
                "WebShop session advance must clear the native session trace"
            )
    elif advances:
        raise RuntimeError(
            "WebShop tool evidence claims a session advance without an index change"
        )
    return advanced


FORMAL_SCHEMA_V3 = "agentmemory_formal_step_v3"
WEBSHOP_V2_SURFACE = "memoryarena_webshop_native_v1"
PROCEDURAL_WEBSHOP_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_train_v1"
)
PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
)
LATENT_PREFERENCE_WEBSHOP_SURFACE = (
    "agentmemory_webshop_latent_preference_train_v1"
)
LATENT_PREFERENCE_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_latent_preference_filesystem_v2"
)
RECENCY_OVERRIDE_WEBSHOP_SURFACE = (
    "agentmemory_webshop_recency_override_train_v1"
)
RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_recency_override_filesystem_v2"
)
DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE = (
    "agentmemory_webshop_distractor_robustness_top1_train_v1"
)
DISTRACTOR_ROBUSTNESS_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_distractor_robustness_filesystem_v2"
)
COMPOSITIONAL_RECALL_WEBSHOP_SURFACE = (
    "agentmemory_webshop_compositional_recall_top1_train_v1"
)
COMPOSITIONAL_RECALL_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_compositional_recall_filesystem_v2"
)
INTENT_CLARIFICATION_WEBSHOP_SURFACE = (
    "agentmemory_webshop_intent_clarification_train_v1"
)
INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_intent_clarification_filesystem_v2"
)
SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE = (
    "agentmemory_webshop_selective_memory_use_top1_train_v1"
)
SELECTIVE_MEMORY_USE_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_selective_memory_use_filesystem_v2"
)
NEGATIVE_CONSTRAINT_WEBSHOP_SURFACE = (
    "agentmemory_webshop_negative_constraint_top1_train_v1"
)
NEGATIVE_CONSTRAINT_FILESYSTEM_WEBSHOP_SURFACE = (
    "agentmemory_webshop_negative_constraint_filesystem_v2"
)
PREFERENCE_WEBSHOP_SURFACES = frozenset(
    {
        LATENT_PREFERENCE_WEBSHOP_SURFACE,
        LATENT_PREFERENCE_FILESYSTEM_WEBSHOP_SURFACE,
        RECENCY_OVERRIDE_WEBSHOP_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
LATENT_PREFERENCE_WEBSHOP_SURFACES = frozenset(
    {
        LATENT_PREFERENCE_WEBSHOP_SURFACE,
        LATENT_PREFERENCE_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
INTENT_CLARIFICATION_WEBSHOP_SURFACES = frozenset(
    {
        INTENT_CLARIFICATION_WEBSHOP_SURFACE,
        INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
SELECTIVE_MEMORY_USE_WEBSHOP_SURFACES = frozenset(
    {
        SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE,
        SELECTIVE_MEMORY_USE_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
RECENCY_OVERRIDE_WEBSHOP_SURFACES = frozenset(
    {
        RECENCY_OVERRIDE_WEBSHOP_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
FILESYSTEM_WEBSHOP_SURFACES = frozenset(
    {
        PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE,
        DISTRACTOR_ROBUSTNESS_FILESYSTEM_WEBSHOP_SURFACE,
        COMPOSITIONAL_RECALL_FILESYSTEM_WEBSHOP_SURFACE,
        LATENT_PREFERENCE_FILESYSTEM_WEBSHOP_SURFACE,
        INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE,
        SELECTIVE_MEMORY_USE_FILESYSTEM_WEBSHOP_SURFACE,
        NEGATIVE_CONSTRAINT_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACES = frozenset(
    {
        DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE,
        DISTRACTOR_ROBUSTNESS_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
COMPOSITIONAL_RECALL_WEBSHOP_SURFACES = frozenset(
    {
        COMPOSITIONAL_RECALL_WEBSHOP_SURFACE,
        COMPOSITIONAL_RECALL_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
NEGATIVE_CONSTRAINT_WEBSHOP_SURFACES = frozenset(
    {
        NEGATIVE_CONSTRAINT_WEBSHOP_SURFACE,
        NEGATIVE_CONSTRAINT_FILESYSTEM_WEBSHOP_SURFACE,
    }
)
QUERY_TOP1_WEBSHOP_SURFACES = frozenset(
    {
        DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE,
        COMPOSITIONAL_RECALL_WEBSHOP_SURFACE,
        INTENT_CLARIFICATION_WEBSHOP_SURFACE,
        SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE,
        NEGATIVE_CONSTRAINT_WEBSHOP_SURFACE,
    }
)
LATENT_PREFERENCE_SOP_WEBSHOP_SURFACES = frozenset(
    {
        LATENT_PREFERENCE_WEBSHOP_SURFACE,
        RECENCY_OVERRIDE_WEBSHOP_SURFACE,
        DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACE,
        COMPOSITIONAL_RECALL_WEBSHOP_SURFACE,
        INTENT_CLARIFICATION_WEBSHOP_SURFACE,
        NEGATIVE_CONSTRAINT_WEBSHOP_SURFACE,
    }
)
PROGRAMMATIC_WEBSHOP_SURFACES = frozenset(
    {
        PROCEDURAL_WEBSHOP_SURFACE,
        *FILESYSTEM_WEBSHOP_SURFACES,
        SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE,
        *LATENT_PREFERENCE_SOP_WEBSHOP_SURFACES,
    }
)
LATENT_PREFERENCE_PROMPT_MODE = "latent_preference_sop"
SELECTIVE_MEMORY_PROMPT_MODE = "selective_memory_sop"
NATURAL_FILESYSTEM_PROMPT_MODE = "natural_filesystem"
FILESYSTEM_SURFACE_CONTRACTS = {
    PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_natural_chain_provider_v4",
        "tasks_per_orbit": 2,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_lsb_within_orbit_v1",
        "boundary_session_index": 1,
        "prompt_family": "natural_attribute_chain_filesystem_v2",
        "source_state": "policy_authored_workspace_only",
        "seed_contract": "none",
        "evaluation_contract": "directional_counterfactual_separation_v1",
    },
    LATENT_PREFERENCE_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_latent_preference_provider_v1",
        "tasks_per_orbit": 2,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_lsb_within_orbit_v1",
        "boundary_session_index": 1,
        "prompt_family": "latent_preference_filesystem_v2",
        "source_state": "policy_authored_workspace_only",
        "seed_contract": "none",
        "evaluation_contract": "directional_counterfactual_separation_v1",
    },
    RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_recency_override_provider_v1",
        "tasks_per_orbit": 2,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_lsb_within_orbit_v1",
        "boundary_session_index": 3,
        "prompt_family": "recency_override_filesystem_v2",
        "source_state": "policy_authored_workspace_only",
        "seed_contract": "none",
        "evaluation_contract": "directional_counterfactual_separation_v1",
    },
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_distractor_robustness_provider_v1",
        "tasks_per_orbit": 2,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_distractor_condition_within_orbit_v1",
        "boundary_session_index": 1,
        "prompt_family": "distractor_robustness_filesystem_v2",
        "source_state": "policy_authored_current_record_plus_branch_distractors",
        "seed_contract": "branch_conditioned_ordinary_profile_files_v1",
        "evaluation_contract": "paired_distractor_robustness_v1",
    },
    COMPOSITIONAL_RECALL_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_compositional_recall_provider_v1",
        "tasks_per_orbit": 4,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_lsb_within_orbit_v1",
        "boundary_session_index": 2,
        "prompt_family": "compositional_recall_filesystem_v2",
        "source_state": "policy_authored_workspace_only",
        "seed_contract": "none",
        "evaluation_contract": "directional_counterfactual_separation_v1",
    },
    INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_intent_clarification_provider_v1",
        "tasks_per_orbit": 2,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_lsb_within_orbit_v1",
        "boundary_session_index": 1,
        "prompt_family": "intent_clarification_filesystem_v2",
        "source_state": "policy_authored_workspace_only",
        "seed_contract": "none",
        "evaluation_contract": "directional_counterfactual_separation_v1",
    },
    SELECTIVE_MEMORY_USE_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_selective_memory_use_provider_v1",
        "tasks_per_orbit": 4,
        "candidate_count_per_phase": 2,
        "source_pairing": "xor_preference_coordinate_within_factorial_v1",
        "boundary_session_index": 1,
        "prompt_family": "selective_memory_use_filesystem_v2",
        "source_state": "harness_seeded_branch_profile_with_optional_policy_edits",
        "seed_contract": "branch_conditioned_initial_profile_files_v1",
        "evaluation_contract": (
            "selective_required_separation_not_required_invariance_v1"
        ),
    },
    NEGATIVE_CONSTRAINT_FILESYSTEM_WEBSHOP_SURFACE: {
        "provider_schema": "agentmemory_verified_negative_constraint_provider_v1",
        "tasks_per_orbit": 3,
        "candidate_count_per_phase": 3,
        "source_pairing": "cyclic_next_within_orbit_v1",
        "boundary_session_index": 1,
        "prompt_family": "negative_constraint_filesystem_v2",
        "source_state": "policy_authored_workspace_only",
        "seed_contract": "none",
        "evaluation_contract": "directional_counterfactual_separation_v1",
    },
}
FILESYSTEM_WORKSPACE_LIMIT_FIELDS = frozenset(
    {
        "max_path_chars",
        "max_files",
        "max_directories",
        "max_file_bytes",
        "max_total_bytes",
        "max_command_chars",
        "max_patch_bytes",
        "default_timeout_ms",
        "max_timeout_ms",
        "cpu_seconds",
        "address_space_bytes",
        "max_processes",
        "max_open_files",
        "stdout_bytes",
        "stderr_bytes",
        "tmp_bytes",
        "tmp_inodes",
    }
)
FILESYSTEM_SANDBOX_FIELDS = {
    "contract": "linux_namespace_chroot_tmpfs_v1",
    "formal_eligible": True,
    "network": "new_namespace_no_routes",
    "rootfs": "minimal_read_only_system_roots",
    "workspace_mount": "bounded_tmpfs_copy_in_copy_out",
    "shell": "bash_no_profile_no_rc",
    "ripgrep_path": "/tools/rg",
    "ripgrep_revalidation": "stat_fingerprint_before_each_command",
    "model_identity": "exclusive_leased_high_uid_per_command",
    "rlimit_nproc_scope": "host_uid_lease_per_concurrent_command",
    "uid_lease_slots": 4096,
    "no_new_privileges": True,
    "capability_bounding_set": "empty",
    "process_namespace": True,
    "mount_namespace": True,
    "ipc_namespace": True,
    "uts_namespace": True,
}
FILESYSTEM_SANDBOX_BOOLEAN_FIELDS = frozenset(
    {
        "formal_eligible",
        "no_new_privileges",
        "process_namespace",
        "mount_namespace",
        "ipc_namespace",
        "uts_namespace",
    }
)
FILESYSTEM_SANDBOX_RESOURCE_FIELDS = frozenset(
    {
        "workspace_bytes",
        "workspace_inodes",
        "max_files",
        "max_directories",
        "max_file_bytes",
        "max_path_chars",
        "default_timeout_ms",
        "max_timeout_ms",
        "cpu_seconds",
        "address_space_bytes",
        "max_processes",
        "max_open_files",
        "stdout_bytes",
        "stderr_bytes",
        "tmp_bytes",
        "tmp_inodes",
    }
)
FILESYSTEM_SANDBOX_SHARED_LIMIT_FIELDS = frozenset(
    FILESYSTEM_SANDBOX_RESOURCE_FIELDS - {"workspace_bytes", "workspace_inodes"}
)
FILESYSTEM_SANDBOX_FINGERPRINT_FIELDS = frozenset(
    {"device", "inode", "mode", "size", "mtime_ns", "ctime_ns"}
)
MEMORY_PROMPT_MODES = (
    "legacy",
    "neutral",
    "neutral_horizon",
    "neutral_horizon_responsibility",
    LATENT_PREFERENCE_PROMPT_MODE,
    SELECTIVE_MEMORY_PROMPT_MODE,
    NATURAL_FILESYSTEM_PROMPT_MODE,
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
    candidate_count_per_phase: int = 2,
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
    if provider.get("candidate_count_per_phase") != candidate_count_per_phase:
        raise RuntimeError(
            "Procedural AgentMemoryGym candidate_count_per_phase metadata is invalid"
        )
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


def _validate_filesystem_sandbox_metadata(metadata: Mapping[str, Any]) -> None:
    limits = metadata.get("workspace_limits")
    if (
        not isinstance(limits, Mapping)
        or not FILESYSTEM_WORKSPACE_LIMIT_FIELDS.issubset(limits)
        or any(
            type(limits[name]) is not int or limits[name] <= 0
            for name in FILESYSTEM_WORKSPACE_LIMIT_FIELDS
        )
    ):
        raise RuntimeError(
            "Filesystem AgentMemoryGym workspace_limits are incomplete or invalid"
        )
    if limits["default_timeout_ms"] > limits["max_timeout_ms"]:
        raise RuntimeError(
            "Filesystem AgentMemoryGym default timeout exceeds its maximum"
        )
    if limits["max_file_bytes"] > limits["max_total_bytes"]:
        raise RuntimeError(
            "Filesystem AgentMemoryGym file limit exceeds workspace capacity"
        )

    sandbox = metadata.get("workspace_sandbox")
    if not isinstance(sandbox, Mapping):
        raise RuntimeError("Filesystem AgentMemoryGym requires workspace_sandbox")
    mismatches = []
    for field, expected in FILESYSTEM_SANDBOX_FIELDS.items():
        observed = sandbox.get(field)
        if field in FILESYSTEM_SANDBOX_BOOLEAN_FIELDS:
            matches = type(observed) is bool and observed is expected
        elif field == "uid_lease_slots":
            matches = type(observed) is int and observed == expected
        else:
            matches = observed == expected
        if not matches:
            mismatches.append(field)
    if mismatches:
        raise RuntimeError(
            "Filesystem AgentMemoryGym workspace_sandbox is inconsistent: "
            + ", ".join(sorted(mismatches))
        )

    observed_sha256 = sandbox.get("ripgrep_sha256")
    expected_sha256 = sandbox.get("ripgrep_expected_sha256")
    if (
        not isinstance(observed_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", observed_sha256) is None
        or expected_sha256 != observed_sha256
    ):
        raise RuntimeError(
            "Filesystem AgentMemoryGym workspace_sandbox has an invalid ripgrep pin"
        )
    version = sandbox.get("ripgrep_version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(
            "Filesystem AgentMemoryGym workspace_sandbox lacks a ripgrep version"
        )
    fingerprint = sandbox.get("ripgrep_startup_fingerprint")
    if (
        not isinstance(fingerprint, Mapping)
        or not FILESYSTEM_SANDBOX_FINGERPRINT_FIELDS.issubset(fingerprint)
        or any(
            type(fingerprint[name]) is not int or fingerprint[name] < 0
            for name in FILESYSTEM_SANDBOX_FINGERPRINT_FIELDS
        )
        or any(
            fingerprint[name] <= 0
            for name in FILESYSTEM_SANDBOX_FINGERPRINT_FIELDS - {"device"}
        )
    ):
        raise RuntimeError(
            "Filesystem AgentMemoryGym workspace_sandbox has an invalid ripgrep fingerprint"
        )

    resources = sandbox.get("resource_limits")
    if (
        not isinstance(resources, Mapping)
        or not FILESYSTEM_SANDBOX_RESOURCE_FIELDS.issubset(resources)
        or any(
            type(resources[name]) is not int or resources[name] <= 0
            for name in FILESYSTEM_SANDBOX_RESOURCE_FIELDS
        )
    ):
        raise RuntimeError(
            "Filesystem AgentMemoryGym workspace_sandbox resource_limits are invalid"
        )
    for name in FILESYSTEM_SANDBOX_SHARED_LIMIT_FIELDS:
        if resources[name] != limits[name]:
            raise RuntimeError(
                f"Filesystem AgentMemoryGym sandbox/workspace limit mismatch: {name}"
            )
    if resources["workspace_bytes"] != limits["max_total_bytes"]:
        raise RuntimeError(
            "Filesystem AgentMemoryGym sandbox workspace_bytes mismatch"
        )
    if (
        resources["workspace_inodes"]
        != limits["max_files"] + limits["max_directories"] + 1
    ):
        raise RuntimeError(
            "Filesystem AgentMemoryGym sandbox workspace_inodes mismatch"
        )


def _validate_filesystem_metadata(metadata: Mapping[str, Any]) -> None:
    surface = metadata.get("surface")
    contract = FILESYSTEM_SURFACE_CONTRACTS.get(surface)
    if contract is None:
        raise RuntimeError("Unsupported filesystem AgentMemoryGym surface")
    expected = {
        "memory_prompt_mode": NATURAL_FILESYSTEM_PROMPT_MODE,
        "memory_management": "policy_managed_persistent_workspace",
        "workspace_surface": "codex_workspace_v2",
        "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
        "workspace_persistence": "episode_across_sessions",
        "workspace_episode_isolation": True,
        "workspace_shell_enabled": True,
        "workspace_apply_patch_enabled": True,
        "workspace_host_path_exposed": False,
        "source_pairing": contract["source_pairing"],
        "tasks_per_orbit": contract["tasks_per_orbit"],
        "workspace_prompt_family": contract["prompt_family"],
        "workspace_seed_contract": contract["seed_contract"],
        "workspace_evaluation_contract": contract["evaluation_contract"],
    }
    mismatches = []
    for key, expected_value in expected.items():
        observed = metadata.get(key)
        if (
            key == "workspace_seed_contract"
            and expected_value == "none"
            and observed is None
        ):
            observed = expected_value
        if type(expected_value) is bool:
            matches = type(observed) is bool and observed is expected_value
        else:
            matches = observed == expected_value
        if not matches:
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            "Filesystem AgentMemoryGym metadata is inconsistent: "
            + ", ".join(mismatches)
        )
    observed_ops = metadata.get("workspace_tool_ops")
    if not isinstance(observed_ops, (list, tuple)) or {
        str(value) for value in observed_ops
    } != {"SHELL_COMMAND", "APPLY_PATCH"}:
        raise RuntimeError(
            "Filesystem AgentMemoryGym must expose exactly shell_command/apply_patch"
        )
    reward_contract = metadata.get("reward_contract")
    if not isinstance(reward_contract, Mapping):
        raise RuntimeError("Filesystem AgentMemoryGym metadata requires reward_contract")
    for field in (
        "workspace_action_reward",
        "shell_command_reward",
        "apply_patch_reward",
    ):
        value = reward_contract.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 0.0:
            raise RuntimeError(
                f"Filesystem AgentMemoryGym requires zero {field}"
            )
    if reward_contract.get("memory_specific_shaping") != "none":
        raise RuntimeError(
            "Filesystem AgentMemoryGym must disable memory-specific shaping"
        )
    _validate_filesystem_sandbox_metadata(metadata)
    provider = metadata.get("provider")
    if not isinstance(provider, Mapping):
        raise RuntimeError("Filesystem AgentMemoryGym metadata requires provider")
    provider_expected = {
        "schema": contract["provider_schema"],
        "tasks_per_orbit": contract["tasks_per_orbit"],
        "candidate_count_per_phase": contract["candidate_count_per_phase"],
    }
    provider_mismatches = [
        key
        for key, expected_value in provider_expected.items()
        if provider.get(key) != expected_value
    ]
    if provider_mismatches:
        raise RuntimeError(
            "Filesystem AgentMemoryGym provider contract is inconsistent: "
            + ", ".join(f"provider {key}" for key in provider_mismatches)
        )
    control = metadata.get("workspace_intervention_control")
    expected_arms = ["correct", "blank", "swapped", "no_workspace"]
    if surface == DISTRACTOR_ROBUSTNESS_FILESYSTEM_WEBSHOP_SURFACE:
        expected_arms = ["correct", "blank", "no_workspace"]
    elif surface == RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE:
        expected_arms.insert(3, "stale")
    if (
        not isinstance(control, Mapping)
        or control.get("contract")
        != "authenticated_session_boundary_counterfactual_copy_v1"
        or control.get("allowed_arms") != expected_arms
        or control.get("boundary_session_index")
        != contract["boundary_session_index"]
        or control.get("source_state") != contract["source_state"]
        or control.get("authenticated_export") is not True
        or control.get("hidden_answer_injection") is not False
    ):
        raise RuntimeError(
            "Filesystem AgentMemoryGym intervention boundary contract is invalid"
        )
    forbidden_legacy_fields = {
        "ltm_inventory_mode",
        "ltm_transition_notice_mode",
        "ltm_inventory_key_max_chars",
        "ltm_inventory_key_format",
    }
    leaked = sorted(forbidden_legacy_fields.intersection(metadata))
    if leaked:
        raise RuntimeError(
            "Filesystem AgentMemoryGym leaks legacy LTM metadata: "
            + ", ".join(leaked)
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
        "factorial_coordinates": [
            ["token_a", "identity"],
            ["token_a", "swapped"],
            ["token_b", "identity"],
            ["token_b", "swapped"],
        ],
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


def _validate_negative_constraint_metadata(
    metadata: Mapping[str, Any],
) -> None:
    provider = _validate_programmatic_metadata(
        metadata,
        provider_schema="agentmemory_verified_negative_constraint_provider_v1",
        tasks_per_orbit=3,
        candidate_count_per_phase=3,
        seed_epoch_boundary_field=(
            "counterfactual_orbit_never_crosses_seed_epoch"
        ),
    )
    expected = {
        "distinct_values_per_phase": 3,
        "counterfactual_branches": 3,
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "initial_memory_inventory_visible": False,
        "purchase_receipt_asin_verification": True,
        "rules_only": False,
        "native_certified": True,
        "training_ready": True,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if provider.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Negative-constraint AgentMemoryGym provider metadata is inconsistent: "
            + ", ".join(mismatches)
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


def build_filesystem_conversation_start(
    action_format: ActionFormat,
    *,
    surface: str,
) -> tuple[ConversationMessage, ConversationMessage]:
    if surface not in FILESYSTEM_WEBSHOP_SURFACES:
        raise ValueError(f"Unsupported filesystem AgentMemoryGym surface: {surface!r}")
    interface = (
        "You are operating a programmatically generated AgentMemoryGym WebShop "
        "task with six shopping sessions and a private persistent workspace. "
        "Native shopping actions are search[keywords] and click[current clickable "
        "value]; click[Buy Now] commits the current product. A failed purchase ends "
        "the episode without revealing the expected answer. The workspace persists "
        "across shopping sessions within this episode and is reset between episodes. "
        "Use shell_command with one JSON object containing command and optional workdir "
        "and timeout_ms to inspect or manipulate files. Use apply_patch followed by a "
        "multiline *** Begin Patch ... *** End Patch payload for precise file edits. "
        "The shell is networkless and resource-bounded; paths stay inside the workspace. "
        "Workspace actions have zero task reward and are optional outside a context "
        "boundary. At each shopping-session boundary, use one normal shell_command "
        "or apply_patch action to overwrite .agent_memory/CONTINUATION.md; only a "
        "verified non-empty write lets the wrapper remove old messages. Then read "
        "that file through a normal action in the new context. Other workspace files "
        "remain available for voluntary notes. There is no host-path access or "
        "dedicated memory API."
    )
    if surface == LATENT_PREFERENCE_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            " Each confirmed choice is preference evidence. Preserve confirmed "
            "preference evidence in an ordinary workspace file, including the "
            "customer-profile memory, preference axis, and inferred value. In later "
            "application sessions, inspect the file and apply the retrieved preference."
        )
    elif surface == RECENCY_OVERRIDE_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            " Maintain the user's current confirmed preference in ordinary workspace "
            "files. When the user explicitly changes that preference, update the "
            "existing current-state record so it contains the new value and no "
            "conflicting stale value. In later application sessions, inspect the "
            "workspace and use the current recorded value rather than an older one."
        )
    elif surface == DISTRACTOR_ROBUSTNESS_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            " The workspace may start with harness-seeded ordinary profile notes. "
            "Treat those as untrusted background records. In session 0, save the exact "
            "authoritative Current preference line in a new ordinary file without "
            "rewriting seeded notes. In later sessions, inspect file contents and use "
            "the policy-authored current record for the active customer and attribute "
            "axis; ignore stale, wrong-customer, and wrong-axis distractors."
        )
    elif surface == COMPOSITIONAL_RECALL_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            " In session 0, save the visible customer-to-profile-token link in an "
            "ordinary file. In session 1, save the visible profile-token-to-attribute "
            "directory in another ordinary file. In every later session, inspect both "
            "records and compose customer -> active profile token -> attribute before "
            "choosing. Do not infer either hop from the current product choices."
        )
    elif surface == NEGATIVE_CONSTRAINT_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            " In session 0, save both visible standing never-accept exclusions and "
            "their attribute axis in an ordinary file. In every later session, inspect "
            "that record and reject every listing that violates either exclusion. Do "
            "not replace the exclusions with only the currently allowed product."
        )
    elif surface == INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            ' In the first shopping session the request is intentionally ambiguous. '
            'Use ASK {"field":"..."} exactly once; ASK is available only in the first '
            "shopping session. The environment returns a CLARIFY observation. Store "
            "the clarification in an ordinary workspace file before the first purchase "
            "and inspect it in later sessions."
        )
    elif surface == SELECTIVE_MEMORY_USE_FILESYSTEM_WEBSHOP_SURFACE:
        interface += (
            " The workspace may start with one branch-conditioned ordinary profile file. "
            "First decide whether the current request already states every attribute needed. "
            "Do not read the profile merely by habit; read the profile when the current "
            "request omits the preference, and follow explicit current requirements directly."
        )
    if action_format is ActionFormat.REACT:
        prompt = (
            interface
            + " Reply in exactly this format:\n\nThought:\nbrief reasoning\n\n"
            "Action:\n<exactly one native bracket action, shell_command JSON action, "
            "or apply_patch newline action>"
        )
        if surface == INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE:
            prompt = prompt.replace(
                "or apply_patch newline action>",
                'or ASK {"field":"..."}, or apply_patch newline action>',
            )
    elif action_format is ActionFormat.FUNCTION_CALLING:
        function_descriptions = list(AGENTMEMORY_FILESYSTEM_FUNCTION_DESCRIPTION)
        if surface == INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE:
            function_descriptions.append(AGENTMEMORY_ASK_FUNCTION_DESCRIPTION)
        prompt = (
            interface
            + " Invoke exactly one available function.\n\n"
            + format_function_call_prompt(function_descriptions)
        )
    elif action_format is ActionFormat.CODE_AS_ACTION:
        function_descriptions = list(AGENTMEMORY_FILESYSTEM_FUNCTION_DESCRIPTION)
        if surface == INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE:
            function_descriptions.append(AGENTMEMORY_ASK_FUNCTION_DESCRIPTION)
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
    if surface in FILESYSTEM_WEBSHOP_SURFACES:
        if memory_prompt_mode != NATURAL_FILESYSTEM_PROMPT_MODE:
            raise ValueError(
                "The filesystem surface requires memory_prompt_mode="
                f"{NATURAL_FILESYSTEM_PROMPT_MODE!r}."
            )
        return build_filesystem_conversation_start(
            action_format,
            surface=surface,
        )
    if memory_prompt_mode == NATURAL_FILESYSTEM_PROMPT_MODE:
        raise ValueError(
            "natural_filesystem prompt mode is only valid for a filesystem surface."
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


def format_filesystem_action(action_name: str, arguments: dict[str, Any]) -> str:
    if action_name == "search":
        return f"search[{_require_function_text(arguments, 'keywords')}]"
    if action_name == "click":
        return f"click[{_require_function_text(arguments, 'item')}]"
    if action_name == "shell_command":
        if not isinstance(arguments, dict):
            raise ValueError("shell_command arguments must be an object.")
        return "shell_command " + json.dumps(arguments, ensure_ascii=False)
    if action_name == "apply_patch":
        if not isinstance(arguments, dict) or set(arguments) != {"patch"}:
            raise ValueError("apply_patch requires exactly patch:string.")
        patch = arguments["patch"]
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("apply_patch patch must be a non-empty string.")
        return FILESYSTEM_APPLY_PATCH_PREFIX + patch.strip()
    if action_name == "ask":
        if not isinstance(arguments, dict) or set(arguments) != {"field"}:
            raise ValueError("ask requires exactly field:string.")
        return "ASK " + json.dumps(
            {"field": _require_function_text(arguments, "field")},
            ensure_ascii=False,
        )
    raise ValueError(f"Unsupported filesystem AgentMemoryGym action: {action_name}")


def parse_filesystem_env_action(
    action: str,
    *,
    allow_ask: bool = False,
) -> tuple[str, dict[str, Any]]:
    cleaned = action.strip()
    native_match = NATIVE_ACTION_RE.fullmatch(cleaned)
    if native_match is not None:
        argument = native_match.group(2).strip()
        key = "keywords" if native_match.group(1) == "search" else "item"
        return native_match.group(1), {key: argument}
    json_match = FILESYSTEM_JSON_ACTION_RE.fullmatch(cleaned)
    if json_match is not None:
        payload = json.loads(json_match.group(1))
        if not isinstance(payload, dict):
            raise ValueError("shell_command payload must be a JSON object.")
        return "shell_command", payload
    if allow_ask:
        ask_match = FILESYSTEM_ASK_ACTION_RE.fullmatch(cleaned)
        if ask_match is not None:
            payload = json.loads(ask_match.group(1))
            if not isinstance(payload, dict):
                raise ValueError("ASK payload must be a JSON object.")
            return "ask", payload
    if cleaned.startswith(FILESYSTEM_APPLY_PATCH_PREFIX):
        patch = cleaned[len(FILESYSTEM_APPLY_PATCH_PREFIX) :]
        if not patch.strip():
            raise ValueError("apply_patch patch must be non-empty.")
        return "apply_patch", {"patch": patch}
    raise ValueError(
        "Expected one native bracket action, shell_command JSON action, or "
        "apply_patch newline action."
    )


def extract_bare_filesystem_action(text: str, *, allow_ask: bool = False) -> str:
    cleaned = strip_think_prefix(text).strip()
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    try:
        action_name, arguments = parse_filesystem_env_action(
            cleaned,
            allow_ask=allow_ask,
        )
        return format_filesystem_action(action_name, arguments)
    except Exception:
        return ""


def parse_filesystem_code_action(code: str, *, allow_ask: bool = False) -> str:
    """Parse one literal tool call without executing policy-authored Python."""

    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError("Filesystem code action must be valid Python syntax.") from exc
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
        raise ValueError("Filesystem code action must contain exactly one function call.")
    call = module.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("Filesystem code action must call one registered function.")
    if call.args:
        raise ValueError("Filesystem code action accepts keyword arguments only.")
    function_name = call.func.id
    action_name = FILESYSTEM_FUNCTION_TO_ACTION.get(function_name.lower())
    if action_name is None and allow_ask and function_name.lower() == "ask":
        action_name = "ask"
    if action_name is None:
        raise ValueError(f"Invalid filesystem function name: {function_name}")
    arguments: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("Filesystem code action does not accept **kwargs expansion.")
        if keyword.arg in arguments:
            raise ValueError(
                f"Filesystem code action repeats argument: {keyword.arg}"
            )
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Filesystem code action arguments must be Python literals."
            ) from exc
    try:
        json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Filesystem code action arguments must be JSON-compatible literals."
        ) from exc
    return format_filesystem_action(action_name, arguments)


class FilesystemAgentMemoryAdapter(AgentMemoryAdapter):
    """Surface-local parser that leaves legacy memory actions unchanged elsewhere."""

    allow_ask = False

    @classmethod
    def parse_react(cls, text: str) -> ActionWithTought:
        action_text = strip_think_prefix(text)
        think_thought = ""
        if action_text != text:
            removed = text[: len(text) - len(action_text)]
            think_thought = re.sub(
                r"</?think\s*>", "", removed, flags=re.IGNORECASE
            ).strip()
        if "Action:" not in action_text:
            bare_action = extract_bare_filesystem_action(
                action_text,
                allow_ask=cls.allow_ask,
            )
            if bare_action:
                return ActionWithTought(thought=think_thought, action=bare_action)
        parsed = BaseAdapter.parse_react(action_text)
        if parsed.action:
            try:
                action_name, arguments = parse_filesystem_env_action(
                    parsed.action,
                    allow_ask=cls.allow_ask,
                )
                return ActionWithTought(
                    thought=parsed.thought or think_thought,
                    action=format_filesystem_action(action_name, arguments),
                )
            except Exception:
                return ActionWithTought(
                    thought=parsed.thought or think_thought,
                    action="",
                )
        bare_action = extract_bare_filesystem_action(
            action_text,
            allow_ask=cls.allow_ask,
        )
        if bare_action:
            return ActionWithTought(
                thought=parsed.thought or think_thought,
                action=bare_action,
            )
        return ActionWithTought(thought=parsed.thought or think_thought, action="")

    @classmethod
    def parse_function_calling(cls, text: str) -> ActionWithTought:
        fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}",
            strict=False,
        )
        function_name = fn_call["function_name"]
        action_name = FILESYSTEM_FUNCTION_TO_ACTION.get(str(function_name).lower())
        if action_name is None and cls.allow_ask and str(function_name).lower() == "ask":
            action_name = "ask"
        if action_name is None:
            raise ValueError(f"Invalid filesystem function name: {function_name}")
        return ActionWithTought(
            thought=fn_call["thought"],
            action=format_filesystem_action(action_name, fn_call["arguments"]),
        )

    @classmethod
    def to_function_calling(cls, action_with_thought: ActionWithTought) -> str:
        action_name, arguments = parse_filesystem_env_action(
            action_with_thought.action,
            allow_ask=cls.allow_ask,
        )
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": action_name.lower(),
                "arguments": arguments,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def parse_code_as_action(cls, text: str) -> ActionWithTought:
        code = extract_python_code_blocks(text)
        action = parse_filesystem_code_action(code, allow_ask=cls.allow_ask)
        return ActionWithTought(
            thought=parse_python_code_comments(code),
            action=action,
        )

    @classmethod
    def to_code_as_action(cls, action_with_thought: ActionWithTought) -> str:
        action_name, arguments = parse_filesystem_env_action(
            action_with_thought.action,
            allow_ask=cls.allow_ask,
        )
        return (
            f"```python\n# {action_with_thought.thought}\n"
            f"{action_name.lower()}(**{repr(arguments)})\n```"
        )


class IntentClarificationFilesystemAgentMemoryAdapter(FilesystemAgentMemoryAdapter):
    """Filesystem adapter with the intent surface's single explicit ASK action."""

    allow_ask = True


def _workspace_event_matches_current_action(
    latest_event: Any,
    tool_ops: Any,
    *,
    native_step_after: Any,
    submitted_action: Any,
) -> bool:
    """Bind endpoint workspace evidence to this exact dispatched policy action."""

    if (
        not isinstance(latest_event, Mapping)
        or not isinstance(tool_ops, Sequence)
        or isinstance(tool_ops, (str, bytes))
        or not tool_ops
        or not isinstance(tool_ops[-1], Mapping)
        or isinstance(native_step_after, bool)
        or not isinstance(native_step_after, int)
    ):
        return False
    latest_tool_op = tool_ops[-1]
    event_step = latest_event.get("step")
    tool_step = latest_tool_op.get("step")
    if (
        isinstance(event_step, bool)
        or not isinstance(event_step, int)
        or event_step != tool_step
        or event_step != native_step_after
        or str(latest_event.get("op", "")).upper()
        != str(latest_tool_op.get("op", "")).upper()
    ):
        return False
    event_id = latest_event.get("event_id")
    tool_event_id = latest_tool_op.get("event_id")
    event_request_sha256 = latest_event.get("request_sha256")
    tool_request_sha256 = latest_tool_op.get("request_sha256")
    expected_request_sha256 = filesystem_workspace_action_request_sha256(
        submitted_action
    )
    return bool(
        isinstance(event_id, int)
        and not isinstance(event_id, bool)
        and event_id == tool_event_id
        and isinstance(event_request_sha256, str)
        and len(event_request_sha256) == 64
        and all(char in "0123456789abcdef" for char in event_request_sha256)
        and event_request_sha256 == tool_request_sha256
        and event_request_sha256 == expected_request_sha256
    )


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
        self._policy_system_prompt: str | None = None
        self.memory_prompt_mode: str | None = None
        self.is_v3 = self.formal_schema_version == FORMAL_SCHEMA_V3
        self.is_procedural = self.surface in PROGRAMMATIC_WEBSHOP_SURFACES
        self.is_filesystem = self.surface in FILESYSTEM_WEBSHOP_SURFACES
        if self.surface == INTENT_CLARIFICATION_FILESYSTEM_WEBSHOP_SURFACE:
            self.adapter_cls = IntentClarificationFilesystemAgentMemoryAdapter
        elif self.is_filesystem:
            self.adapter_cls = FilesystemAgentMemoryAdapter
        else:
            self.adapter_cls = AgentMemoryAdapter
        self.is_latent_preference = self.surface in LATENT_PREFERENCE_WEBSHOP_SURFACES
        self.is_recency_override = self.surface in RECENCY_OVERRIDE_WEBSHOP_SURFACES
        self.is_distractor_robustness = (
            self.surface in DISTRACTOR_ROBUSTNESS_WEBSHOP_SURFACES
        )
        self.is_compositional_recall = (
            self.surface in COMPOSITIONAL_RECALL_WEBSHOP_SURFACES
        )
        self.is_intent_clarification = self.surface in INTENT_CLARIFICATION_WEBSHOP_SURFACES
        self.is_selective_memory_use = self.surface in SELECTIVE_MEMORY_USE_WEBSHOP_SURFACES
        self.is_negative_constraint = (
            self.surface in NEGATIVE_CONSTRAINT_WEBSHOP_SURFACES
        )
        self.is_preference_memory = self.surface in PREFERENCE_WEBSHOP_SURFACES
        self.requires_latent_preference_sop = (
            self.surface in LATENT_PREFERENCE_SOP_WEBSHOP_SURFACES
        )
        self.requires_selective_memory_sop = (
            self.surface == SELECTIVE_MEMORY_USE_WEBSHOP_SURFACE
        )
        if self.is_v3:
            _validate_v3_metadata(self.metadata)
            if self.action_format is not ActionFormat.REACT:
                raise RuntimeError(
                    "AgentMemoryGym v3 domains currently require action_format='react'"
                )
        elif self.is_procedural:
            if self.is_filesystem:
                _validate_filesystem_metadata(self.metadata)
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
            elif self.is_negative_constraint:
                _validate_negative_constraint_metadata(self.metadata)
            elif self.surface == PROCEDURAL_FILESYSTEM_WEBSHOP_SURFACE:
                _validate_procedural_metadata(self.metadata)
            elif not self.is_filesystem:
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
            if self.is_filesystem and (
                memory_prompt_mode != NATURAL_FILESYSTEM_PROMPT_MODE
            ):
                raise RuntimeError(
                    "AgentMemoryGym filesystem surface requires "
                    f"memory_prompt_mode={NATURAL_FILESYSTEM_PROMPT_MODE!r}"
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
                and not self.is_filesystem
                and memory_prompt_mode in {
                    LATENT_PREFERENCE_PROMPT_MODE,
                    SELECTIVE_MEMORY_PROMPT_MODE,
                    NATURAL_FILESYSTEM_PROMPT_MODE,
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
            if memory_prompt_mode == NATURAL_FILESYSTEM_PROMPT_MODE:
                raise RuntimeError(
                    "natural_filesystem prompt mode is only valid for the "
                    "persistent-workspace surface"
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
        self._reset_policy_transition_state(created.get("info", {}))
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

    def configure_policy_system_prompt(self, prompt: str) -> None:
        """Bind the exact formal prompt resolved by the training adapter."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("AgentMemory policy system prompt must not be empty")
        normalized = prompt.strip()
        server_prompt = getattr(self, "system_prompt", None)
        if self.is_v3 and isinstance(server_prompt, str):
            if normalized != server_prompt.strip():
                raise ValueError(
                    "AgentMemory v3 policy prompt differs from server metadata"
                )
        expected_sha256 = getattr(self, "system_prompt_sha256", None)
        if isinstance(expected_sha256, str):
            observed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if observed != expected_sha256:
                raise ValueError(
                    "AgentMemory policy prompt SHA-256 differs from server metadata"
                )
        self._policy_system_prompt = normalized

    def policy_framing(self) -> list[dict[str, str]]:
        if self._policy_system_prompt is None:
            raise RuntimeError(
                "AgentMemory formal policy system prompt was not configured"
            )
        return [{"role": "system", "content": self._policy_system_prompt}]

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = _copy_policy_messages(messages)
        if not normalized or normalized[-1]["role"] != "user":
            raise ValueError(
                "AgentMemory initial policy context must end with an observation"
            )
        observation = str(self.observe())
        if normalized[-1]["content"] != observation:
            raise ValueError(
                "AgentMemory initial policy context does not end with the current observation"
            )
        return self.policy_framing() + [
            {"role": "user", "content": observation}
        ]

    def _reset_policy_transition_state(self, env_info: Mapping[str, Any]) -> None:
        self._policy_step_count = 0
        self._native_call_count = 0
        self._context_epoch = 0
        self._session_epoch = _optional_transition_int(
            env_info.get("current_subtask_index")
        ) or 0
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._pending_session_handoff: dict[str, Any] | None = None
        self._pending_checkpoint_read: dict[str, Any] | None = None
        self._selected_policy_control: str | None = None

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = _copy_policy_messages(messages)
        if initial:
            framing = list(normalized)
            if (
                framing
                and framing[-1]["role"] == "user"
                and framing[-1]["content"] == str(self.observe())
            ):
                framing.pop()
            if not framing:
                raise ValueError("AgentMemory policy framing must not be empty")
            if framing != self.policy_framing():
                raise ValueError(
                    "AgentMemory initial policy framing differs from the formal prompt"
                )
            self._immutable_policy_context = framing
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if self._pending_checkpoint_read is not None:
            return None
        if self._pending_session_handoff is None:
            return None
        return WEBSHOP_SESSION_HANDOFF_REQUEST

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        self._selected_policy_control = None
        if self._pending_checkpoint_read is not None:
            return None
        if self._pending_session_handoff is None:
            return None
        if not self._policy_context_bound or self._current_policy_context is None:
            raise RuntimeError(
                "WebShop session handoff requires a bound policy context"
            )
        if pressure is None:
            raise RuntimeError(
                "WebShop session handoff requires task-neutral token pressure"
            )
        if pressure.candidate_prompt_tokens > pressure.effective_prompt_capacity:
            raise RuntimeError(
                "WebShop session handoff request exceeds the policy prompt capacity"
            )
        if checkpoint_retry_ceiling_tokens(
            pressure,
            control_request=WEBSHOP_SESSION_HANDOFF_REQUEST,
        ) > pressure.effective_prompt_capacity:
            raise RuntimeError(
                "WebShop session handoff does not leave room for one bounded "
                "failed checkpoint attempt and retry"
            )
        self._selected_policy_control = "webshop_session_handoff"
        return WEBSHOP_SESSION_HANDOFF_REQUEST

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
        # A few offline adapters construct the client with ``__new__`` to
        # exercise action parsing without opening an environment server.  Keep
        # that legacy fixture contract compatible with the lifecycle state.
        if not hasattr(self, "_policy_step_count"):
            self._reset_policy_transition_state({})
        if not hasattr(self, "is_filesystem"):
            self.is_filesystem = False
        if self._selected_policy_control == "webshop_session_handoff":
            return self._complete_session_handoff(action)
        if (
            self._pending_session_handoff is not None
            and self._pending_checkpoint_read is None
        ):
            raise RuntimeError(
                "WebShop session handoff is pending; prepare the wrapper control "
                "turn before submitting another native action"
            )
        return self._step_native_policy_action(action)

    def _step_native_policy_action(self, action: str) -> StepOutput:
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
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        session_before = self._session_epoch
        checkpoint_read_pending_before = self._pending_checkpoint_read
        response = self.post("step", {"action": parsed_action})
        self._native_call_count += 1
        self._policy_step_count += 1
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
        response_env_info = response.get("info", {})
        if not isinstance(response_env_info, Mapping):
            response_env_info = {}
        else:
            response_env_info = dict(response_env_info)
        self.info["env_info"] = response_env_info
        session_after = session_before
        session_advanced = False
        if self.is_filesystem:
            session_after = _reported_webshop_session(
                response_env_info, session_before
            )
            session_advanced = _validate_webshop_session_advance(
                response_env_info,
                before=session_before,
                after=session_after,
                done=bool(response["done"]),
            )
        self._session_epoch = session_after

        context_transition = build_task_neutral_context_transition(
            CONTEXT_OPERATION_APPEND
        )
        native_wrapper_evidence = {
            "event": "native_action",
            "session_advanced": session_advanced,
            "native_call_count_before": native_before,
            "native_call_count_after": self._native_call_count,
            "policy_step_before": policy_before,
            "policy_step_after": self._policy_step_count,
            "context_epoch_before": context_before,
            "context_epoch_after": self._context_epoch,
            "session_epoch_before": session_before,
            "session_epoch_after": self._session_epoch,
            "successor_context_policy": self.rollout_context_policy,
        }
        read_receipt = None
        if self.is_filesystem:
            latest_event = response_env_info.get("workspace_latest_event")
            tool_ops = response_env_info.get("tool_ops")
            event_is_current = _workspace_event_matches_current_action(
                latest_event,
                tool_ops,
                native_step_after=self._native_call_count,
                submitted_action=parsed_action,
            )
            if event_is_current:
                action_kind = str(
                    latest_event.get("tool_name", latest_event.get("op", ""))
                ).lower()
                action_completed = filesystem_checkpoint_action_completed(
                    action_kind,
                    latest_event,
                )
                checkpoint_receipt = build_filesystem_checkpoint_receipt(
                    action_kind=action_kind,
                    action_completed=action_completed,
                    workspace_diff=latest_event.get("workspace_diff"),
                    workspace_snapshot=response_env_info.get("workspace_snapshot"),
                )
                native_wrapper_evidence[
                    "filesystem_checkpoint"
                ] = checkpoint_receipt
                response_env_info["action_kind"] = action_kind
                response_env_info["filesystem_checkpoint"] = checkpoint_receipt
                read_receipt = build_filesystem_checkpoint_read_receipt(
                    checkpoint_receipt,
                    action_kind=action_kind,
                    action_completed=action_completed,
                    stdout=latest_event.get("stdout"),
                )
                native_wrapper_evidence[
                    "filesystem_checkpoint_read"
                ] = read_receipt
                response_env_info["filesystem_checkpoint_read"] = read_receipt
                if filesystem_checkpoint_read_observed(read_receipt):
                    native_wrapper_evidence.update(
                        {
                            "memory_event": "read",
                            "document_read_observed": True,
                        }
                    )
        checkpoint_read_satisfied = False
        read_failure_reason = None
        if checkpoint_read_pending_before is not None:
            read_failure_reason = filesystem_checkpoint_read_failure_reason(
                read_receipt,
                checkpoint_read_pending_before,
            )
            checkpoint_read_satisfied = read_failure_reason is None
            native_wrapper_evidence.update(
                {
                    "checkpoint_read_required": True,
                    "checkpoint_read_satisfied": checkpoint_read_satisfied,
                    "checkpoint_read_retry_pending": bool(
                        not checkpoint_read_satisfied and not bool(response["done"])
                    ),
                    "checkpoint_read_failure_reason": read_failure_reason,
                    "checkpoint_read_expected_size_bytes": (
                        checkpoint_read_pending_before.get("size_bytes")
                    ),
                    "checkpoint_read_expected_sha256": (
                        checkpoint_read_pending_before.get("sha256")
                    ),
                }
            )
            if checkpoint_read_satisfied or bool(response["done"]):
                self._pending_checkpoint_read = None

        policy_observation = (
            build_filesystem_checkpoint_read_retry_observation(
                read_failure_reason or "checkpoint_read_not_observed"
            )
            if checkpoint_read_pending_before is not None
            and not checkpoint_read_satisfied
            and not bool(response["done"])
            else response["observation"]
        )
        if self.is_filesystem and session_advanced:
            self._pending_session_handoff = {
                "fresh_observation": str(response["observation"]),
                "session_before": session_before,
                "session_after": session_after,
                "native_call_count": self._native_call_count,
            }
        if self._policy_context_bound and not bool(response["done"]):
            if self._selected_policy_control == "webshop_session_handoff":
                # The ordinary filesystem action used for a pending handoff must
                # not clear history before its receipt is checked below.
                context_transition = build_task_neutral_context_transition(
                    CONTEXT_OPERATION_PRESERVE
                )
            elif (
                checkpoint_read_pending_before is not None
                and not checkpoint_read_satisfied
            ):
                context_transition = build_task_neutral_context_transition(
                    CONTEXT_OPERATION_APPEND
                )
            elif self.is_filesystem and session_advanced:
                context_transition = build_task_neutral_context_transition(
                    CONTEXT_OPERATION_PRESERVE
                )
            else:
                context_transition = build_task_neutral_context_transition(
                    CONTEXT_OPERATION_REPLACE,
                    messages=self._fresh_policy_context(
                        str(response["observation"]),
                    ),
                )
        return StepOutput(
            state=policy_observation,
            reward=response["reward"],
            done=response["done"],
            info=build_task_neutral_transition_info(
                env_info=response_env_info,
                action_submission=self.last_action_submission,
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=session_before,
                session_epoch_after=self._session_epoch,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                context_transition=context_transition,
                wrapper_evidence={
                    **native_wrapper_evidence,
                    "raw_history_cleared": (
                        context_transition["operation"]
                        == CONTEXT_OPERATION_REPLACE
                    ),
                },
            ),
        )

    def _complete_session_handoff(self, action: str) -> StepOutput:
        pending = self._pending_session_handoff
        if pending is None:
            raise RuntimeError("WebShop session handoff was selected without a boundary")
        if self._immutable_policy_context is None:
            raise RuntimeError("WebShop session handoff lost its immutable framing")

        native_output = self._step_native_policy_action(action)
        self._selected_policy_control = None
        info = dict(native_output.info)
        env_info = info.get("env_info", {})
        latest_event = (
            env_info.get("workspace_latest_event")
            if isinstance(env_info, Mapping)
            else None
        )
        tool_ops = env_info.get("tool_ops") if isinstance(env_info, Mapping) else None
        checkpoint_receipt = None
        if _workspace_event_matches_current_action(
            latest_event,
            tool_ops,
            native_step_after=info.get("native_step_after"),
            submitted_action=(
                info.get("action_submission", {}).get("submitted_action")
                if isinstance(info.get("action_submission"), Mapping)
                else None
            ),
        ):
            action_kind = str(
                latest_event.get("tool_name", latest_event.get("op", ""))
            ).lower()
            checkpoint_receipt = build_filesystem_checkpoint_receipt(
                action_kind=action_kind,
                action_completed=filesystem_checkpoint_action_completed(
                    action_kind,
                    latest_event,
                ),
                workspace_diff=latest_event.get("workspace_diff"),
                workspace_snapshot=env_info.get("workspace_snapshot"),
            )
        persisted = filesystem_checkpoint_write_succeeded(checkpoint_receipt)
        session_stable = self._session_epoch == int(pending["session_after"])
        checkpoint_failure_reason = (
            filesystem_checkpoint_failure_reason(checkpoint_receipt)
            if session_stable
            else "unexpected_session_advance"
        )
        replace_context = bool(persisted and session_stable and not native_output.done)
        retry_pending = bool(
            self._pending_session_handoff is not None
            and not replace_context
            and not native_output.done
        )
        policy_observation = (
            build_filesystem_checkpoint_retry_observation(
                checkpoint_failure_reason or "unknown_checkpoint_failure"
            )
            if retry_pending
            else native_output.state
        )

        context_transition = None
        checkpoint_framing_sha256 = None
        if replace_context:
            framing = self._fresh_policy_context(str(pending["fresh_observation"]))
            checkpoint_framing_sha256 = filesystem_checkpoint_framing_sha256(
                framing
            )
            replacement = build_post_checkpoint_context(framing, checkpoint_receipt)
            self._context_epoch += 1
            self._pending_session_handoff = None
            self._pending_checkpoint_read = dict(checkpoint_receipt)
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=replacement,
            )
        elif native_output.done:
            self._pending_session_handoff = None
            self._pending_checkpoint_read = None

        return StepOutput(
            state=policy_observation,
            reward=native_output.reward,
            done=native_output.done,
            info=build_task_neutral_transition_info(
                env_info=env_info if isinstance(env_info, Mapping) else {},
                action_submission=info.get(
                    "action_submission", {"raw_policy_output": action}
                ),
                native_step_before=info.get("native_step_before"),
                native_step_after=info.get("native_step_after"),
                native_call_count_before=info.get("native_call_count_before"),
                native_call_count_after=info.get("native_call_count_after"),
                context_epoch_before=info.get("context_epoch_before"),
                context_epoch_after=self._context_epoch,
                session_epoch_before=info.get("session_epoch_before"),
                session_epoch_after=info.get("session_epoch_after"),
                policy_step_before=info.get("policy_step_before"),
                policy_step_after=info.get("policy_step_after"),
                context_transition=context_transition,
                wrapper_evidence={
                    "event": "webshop_session_handoff",
                    "session_before": pending["session_before"],
                    "session_after": pending["session_after"],
                    "native_call_count_before": info.get(
                        "native_call_count_before"
                    ),
                    "native_call_count_after": info.get("native_call_count_after"),
                    "policy_step_before": info.get("policy_step_before"),
                    "policy_step_after": info.get("policy_step_after"),
                    "context_epoch_before": info.get("context_epoch_before"),
                    "context_epoch_after": self._context_epoch,
                    "session_epoch_before": info.get("session_epoch_before"),
                    "session_epoch_after": info.get("session_epoch_after"),
                    "continuation_path": FILESYSTEM_CHECKPOINT_PATH,
                    "continuation_max_bytes": FILESYSTEM_CHECKPOINT_MAX_BYTES,
                    "continuation_persisted": persisted,
                    "checkpoint_receipt": checkpoint_receipt,
                    "checkpoint_failure_reason": checkpoint_failure_reason,
                    "context_replaced": replace_context,
                    "retry_pending": retry_pending,
                    "checkpoint_retry_observation_bounded": retry_pending,
                    "raw_history_cleared": replace_context,
                    "preserved_policy_output": replace_context,
                    "preserved_native_observation": replace_context,
                    "checkpoint_action_in_successor_context": False,
                    "checkpoint_observation_in_successor_context": False,
                    "checkpoint_content_in_successor_context": False,
                    "checkpoint_framing_sha256": checkpoint_framing_sha256,
                    "checkpoint_read_required_after": replace_context,
                    "native_wrapper_evidence": (
                        dict(info.get("wrapper_evidence", {}))
                        if isinstance(info.get("wrapper_evidence"), Mapping)
                        else {}
                    ),
                },
            ),
        )

    def _fresh_policy_context(self, observation: str) -> list[dict[str, str]]:
        framing = self._immutable_policy_context
        if framing is None:
            raise RuntimeError("AgentMemory policy context was not initialized")
        messages = deepcopy(framing)
        messages.append({"role": "user", "content": str(observation)})
        return messages

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
        response_env_info = response.get("info", {})
        if not isinstance(response_env_info, Mapping):
            response_env_info = {}
        self._reset_policy_transition_state(response_env_info)
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
