"""WebShop session-boundary checkpoint request.

The request is domain-specific, but the response is an ordinary executable
workspace action handled by the WebShop environment.  Receipt validation and
context replacement live in the task-neutral filesystem-checkpoint helper.
"""

from __future__ import annotations

from .filesystem_checkpoint import FILESYSTEM_CHECKPOINT_REQUEST


WEBSHOP_QWEN_XML_CHECKPOINT_GUIDANCE = (
    " Use the system prompt's Qwen XML shell_command form; run "
    "`mkdir -p .agent_memory` before overwriting "
    "`.agent_memory/CONTINUATION.md`."
)


WEBSHOP_SESSION_HANDOFF_REQUEST = (
    FILESYSTEM_CHECKPOINT_REQUEST
    + WEBSHOP_QWEN_XML_CHECKPOINT_GUIDANCE
    + " For this shopping task, preserve completed purchases and their visible "
    "compatibility-relevant attributes, the current customer requirement, and "
    "the next shopping action. Overwrite the checkpoint even if you also keep "
    "other voluntary notes."
)


WEBSHOP_CONTEXT_COMPACTION_REQUEST = (
    FILESYSTEM_CHECKPOINT_REQUEST
    + WEBSHOP_QWEN_XML_CHECKPOINT_GUIDANCE
    + " For this shopping task, preserve the current session requirement, "
    "decisive product evidence and selections, completed purchases, remaining "
    "budget, and the next shopping action. Overwrite the checkpoint even if "
    "you also keep other voluntary notes."
)
