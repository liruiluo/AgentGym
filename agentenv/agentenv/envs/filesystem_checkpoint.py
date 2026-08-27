"""Task-neutral filesystem checkpoint contract for context replacement.

Environment wrappers own when a checkpoint is requested.  The sampled write is
still an ordinary policy action sent through ``env.step``; this module only
normalizes the evidence required before a wrapper may clear old messages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_receipt_v1"
)
FILESYSTEM_CHECKPOINT_PATH = ".agent_memory/CONTINUATION.md"
FILESYSTEM_CHECKPOINT_MAX_BYTES = 8 * 1024
FILESYSTEM_CHECKPOINT_ACTION_KINDS = frozenset({"shell_command", "apply_patch"})

FILESYSTEM_CHECKPOINT_REQUEST = (
    "The conversation is nearing a context boundary. Use this turn for exactly "
    "one normal executable shell_command or apply_patch action that creates or "
    f"overwrites `{FILESYSTEM_CHECKPOINT_PATH}`. Write a non-empty continuation "
    f"snapshot of at most {FILESYSTEM_CHECKPOINT_MAX_BYTES} bytes containing only "
    "the objective, decisive evidence/state, relevant workspace paths, and the "
    "next concrete action. Do not emit prose outside the action. This action is "
    "executed normally and consumes one policy-action step. Earlier messages are "
    "removed only after the environment verifies this exact file write; a failed "
    "write keeps the current context and can be retried. This bounded checkpoint "
    "is only a cross-context working-state snapshot, not a replacement for task "
    "artifacts or longer-lived evidence notes; continue to create, read, and update "
    "other workspace files whenever they are useful."
)

FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER = (
    "Earlier conversation was removed after the continuation snapshot write "
    f"succeeded. The workspace persists, but `{FILESYSTEM_CHECKPOINT_PATH}` was "
    "not copied into this prompt. Use the next normal action to read that file, "
    "then continue from its evidence and next action. Other workspace files remain "
    "available and may still be read or updated normally."
)


def _maximum_policy_turn_growth(pressure: Any) -> int:
    return (
        int(pressure.max_response_tokens)
        + int(pressure.max_observation_tokens)
        + int(pressure.action_observation_envelope_tokens)
    )


def checkpoint_retry_trigger_tokens(pressure: Any) -> int:
    """Legacy projection that reserves room for a growing failed retry."""

    return (
        int(pressure.action_prompt_tokens)
        + 2 * _maximum_policy_turn_growth(pressure)
    )


def checkpoint_bounded_retry_trigger_tokens(pressure: Any) -> int:
    """Project one ordinary turn before a bounded control retry is required.

    A failed control turn is restored to its exact pre-control context, so it
    does not need another response-plus-observation reserve.  Taking the larger
    of the ordinary and candidate render remains conservative when a chat
    template shortens generation-only history while appending the control.
    """

    return max(
        int(pressure.action_prompt_tokens),
        int(pressure.candidate_prompt_tokens),
    ) + _maximum_policy_turn_growth(pressure)


def build_filesystem_checkpoint_receipt(
    *,
    action_kind: str,
    action_completed: bool,
    workspace_diff: Mapping[str, Any] | None,
    workspace_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize an attested workspace diff into the shared checkpoint schema.

    A pre-existing checkpoint is deliberately insufficient: ``changed`` is true
    only when the current ordinary action added or modified the exact fixed path.
    This prevents a read or unrelated tool call from authorizing context removal.
    """

    path = FILESYSTEM_CHECKPOINT_PATH
    diff = workspace_diff if isinstance(workspace_diff, Mapping) else {}
    changed_entry: Mapping[str, Any] | None = None

    added = diff.get("added", ())
    if _is_mapping_sequence(added):
        for item in added:
            if item.get("path") == path:
                changed_entry = item
                break

    modified = diff.get("modified", ())
    if changed_entry is None and _is_mapping_sequence(modified):
        for item in modified:
            after = item.get("after")
            if isinstance(after, Mapping) and after.get("path") == path:
                changed_entry = after
                break
            # Some workspace implementations expose modified entries directly.
            if item.get("path") == path:
                changed_entry = item
                break

    snapshot_entry: Mapping[str, Any] | None = None
    snapshot = workspace_snapshot if isinstance(workspace_snapshot, Mapping) else {}
    files = snapshot.get("files", ())
    if _is_mapping_sequence(files):
        for item in files:
            if item.get("path") == path:
                snapshot_entry = item
                break

    entry = changed_entry if changed_entry is not None else snapshot_entry
    size = _entry_size(entry)
    sha256 = entry.get("sha256") if isinstance(entry, Mapping) else None
    regular_file = bool(
        isinstance(entry, Mapping)
        and entry.get("kind", "file") == "file"
    )
    deleted = diff.get("deleted", ())
    deleted_path = bool(
        _is_mapping_sequence(deleted)
        and any(item.get("path") == path for item in deleted)
    )
    exists = bool(entry is not None and not deleted_path)

    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": path,
        "action_kind": str(action_kind).lower(),
        "action_completed": bool(action_completed),
        "changed": changed_entry is not None,
        "exists": exists,
        "regular_file": regular_file,
        "size_bytes": size,
        "sha256": sha256,
    }


def normalize_filesystem_checkpoint_receipt(value: Any) -> dict[str, Any] | None:
    """Validate a server receipt; ``None`` means the action produced no receipt."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError("filesystem checkpoint receipt must be a mapping")
    expected = {
        "schema",
        "path",
        "action_kind",
        "action_completed",
        "changed",
        "exists",
        "regular_file",
        "size_bytes",
        "sha256",
    }
    if set(value) != expected:
        raise RuntimeError("filesystem checkpoint receipt schema drifted")
    if value.get("schema") != FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA:
        raise RuntimeError("filesystem checkpoint receipt version drifted")
    if value.get("path") != FILESYSTEM_CHECKPOINT_PATH:
        raise RuntimeError("filesystem checkpoint receipt reports the wrong path")
    action_kind = value.get("action_kind")
    if not isinstance(action_kind, str):
        raise RuntimeError("filesystem checkpoint action kind must be text")
    for key in ("action_completed", "changed", "exists", "regular_file"):
        if type(value.get(key)) is not bool:
            raise RuntimeError(f"filesystem checkpoint {key} must be boolean")
    size = value.get("size_bytes")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        raise RuntimeError("filesystem checkpoint size_bytes is invalid")
    sha256 = value.get("sha256")
    if sha256 is not None and (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise RuntimeError("filesystem checkpoint sha256 is invalid")
    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "action_kind": action_kind.lower(),
        "action_completed": value["action_completed"],
        "changed": value["changed"],
        "exists": value["exists"],
        "regular_file": value["regular_file"],
        "size_bytes": size,
        "sha256": sha256,
    }


def filesystem_checkpoint_write_succeeded(value: Any) -> bool:
    receipt = normalize_filesystem_checkpoint_receipt(value)
    if receipt is None:
        return False
    size = receipt["size_bytes"]
    return bool(
        receipt["action_kind"] in FILESYSTEM_CHECKPOINT_ACTION_KINDS
        and receipt["action_completed"]
        and receipt["changed"]
        and receipt["exists"]
        and receipt["regular_file"]
        and isinstance(size, int)
        and 0 < size <= FILESYSTEM_CHECKPOINT_MAX_BYTES
        and receipt["sha256"] is not None
    )


def filesystem_checkpoint_failure_reason(value: Any) -> str | None:
    receipt = normalize_filesystem_checkpoint_receipt(value)
    if receipt is None:
        return "missing_receipt"
    if receipt["action_kind"] not in FILESYSTEM_CHECKPOINT_ACTION_KINDS:
        return "wrong_action_kind"
    if not receipt["action_completed"]:
        return "action_not_completed"
    if not receipt["changed"]:
        return "checkpoint_not_changed"
    if not receipt["exists"]:
        return "checkpoint_missing"
    if not receipt["regular_file"]:
        return "checkpoint_not_regular_file"
    size = receipt["size_bytes"]
    if not isinstance(size, int) or size <= 0:
        return "checkpoint_empty"
    if size > FILESYSTEM_CHECKPOINT_MAX_BYTES:
        return "checkpoint_too_large"
    if receipt["sha256"] is None:
        return "checkpoint_digest_missing"
    return None


def _is_mapping_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, Mapping) for item in value)
    )


def _entry_size(entry: Mapping[str, Any] | None) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("bytes", entry.get("size"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
