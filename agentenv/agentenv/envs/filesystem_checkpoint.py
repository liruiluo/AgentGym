"""Task-neutral filesystem checkpoint contract for context replacement.

Environment wrappers own when a checkpoint is requested.  The sampled write is
still an ordinary policy action sent through ``env.step``; this module only
normalizes the evidence required before a wrapper may clear old messages.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA_V1 = (
    "agentmemory_filesystem_checkpoint_receipt_v1"
)
FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_receipt_v2"
)
FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_read_receipt_v1"
)
FILESYSTEM_CHECKPOINT_PATH = ".agent_memory/CONTINUATION.md"
FILESYSTEM_CHECKPOINT_MAX_BYTES = 8 * 1024
FILESYSTEM_CHECKPOINT_ACTION_KINDS = frozenset({"shell_command", "apply_patch"})
# A failed checkpoint attempt is surfaced through a fixed, task-neutral message
# rather than the route's potentially large native observation. UTF-8 bytes are
# a conservative upper bound for the tokenizer used by the formal runtime.
FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS = 256
FILESYSTEM_CHECKPOINT_REQUEST_TOKEN_SLACK = 32
_FILESYSTEM_SHELL_ACTION_RE = re.compile(
    r"\Ashell_command\s+(\{.*\})\Z", re.DOTALL
)
_FILESYSTEM_APPLY_PATCH_PREFIX = "apply_patch\n"
_FILESYSTEM_CHECKPOINT_HEREDOC_RE = re.compile(
    r"\A"
    r"(?:mkdir\s+-p\s+\.agent_memory\s+&&\s+)?"
    r"cat\s*>\s*\.agent_memory/CONTINUATION\.md\s+"
    r"<<'(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)'\n"
    r"(?P<body>.*)\n(?P=delimiter)\n?\Z",
    re.DOTALL,
)
_FILESYSTEM_CHECKPOINT_PRINTF_TOKEN = r"[A-Za-z0-9._:/+=-]+"
_FILESYSTEM_CHECKPOINT_PRINTF_RE = re.compile(
    r"\A[ \t]*(?:mkdir[ \t]+-p[ \t]+\.agent_memory[ \t]+&&[ \t]+)?"
    r"printf[ \t]+'%s(?:\\n|\n)'[ \t]+"
    rf"(?P<arguments>{_FILESYSTEM_CHECKPOINT_PRINTF_TOKEN}"
    rf"(?:[ \t]+{_FILESYSTEM_CHECKPOINT_PRINTF_TOKEN})*)"
    r"[ \t]+>[ \t]+\.agent_memory/CONTINUATION\.md[ \t]*\Z"
)

FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE = (
    f"`{FILESYSTEM_CHECKPOINT_PATH}` is a single-boundary handoff slot, not "
    "cumulative memory. Every later context boundary overwrites this same file. "
    "Store evidence that must survive multiple context boundaries in other ordinary "
    "workspace files before the boundary, and list those paths in the checkpoint."
)

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
    "other workspace files whenever they are useful. "
    + FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE
)

FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER = (
    "Earlier conversation was removed after the continuation snapshot write "
    f"succeeded. The workspace persists, but `{FILESYSTEM_CHECKPOINT_PATH}` was "
    "not copied into this prompt. Use the next normal action to read that file, "
    "then continue from its evidence and next action. Other workspace files remain "
    "available and may still be read or updated normally."
)

FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION = (
    'shell_command {"command":"mkdir -p .agent_memory && cat > '
    ".agent_memory/CONTINUATION.md <<'AGENT_MEMORY_EOF'\\n"
    "objective: CURRENT_OBJECTIVE\\n"
    "decisive_evidence: DECISIVE_EVIDENCE\\n"
    "workspace_paths: RELEVANT_PATHS\\n"
    "next_action: NEXT_ACTION\\n"
    'AGENT_MEMORY_EOF"}'
)
FILESYSTEM_BARE_CHECKPOINT_WRITE_GUIDANCE = (
    "\n\nFor this bare/Codex action surface, output exactly the one "
    "shell_command action below and no other text. Replace the uppercase "
    "placeholders with concise current state; do not change the command shape "
    "or add optional JSON fields:\n\n"
    + FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION
)
FILESYSTEM_BARE_CHECKPOINT_READ_ACTION = (
    'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}'
)
FILESYSTEM_BARE_CHECKPOINT_CONTINUATION_MARKER = (
    FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER
    + " The required checkpoint read must happen before any dependent task "
    "action. Output exactly this action and no other text:\n\n"
    + FILESYSTEM_BARE_CHECKPOINT_READ_ACTION
)

FILESYSTEM_QWEN_CHECKPOINT_WRITE_ACTION = (
    "<tool_call>\n"
    "<function=shell_command>\n"
    "<parameter=command>\n"
    "mkdir -p .agent_memory && cat > .agent_memory/CONTINUATION.md "
    "<<'AGENT_MEMORY_EOF'\n"
    "objective: CURRENT_OBJECTIVE\n"
    "decisive_evidence: DECISIVE_EVIDENCE\n"
    "workspace_paths: RELEVANT_PATHS\n"
    "next_action: NEXT_ACTION\n"
    "AGENT_MEMORY_EOF\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
FILESYSTEM_QWEN_CHECKPOINT_WRITE_GUIDANCE = (
    "\n\nOutput exactly the one Qwen XML shell_command call below and no other "
    "text. Replace the uppercase placeholders with concise current state; do "
    "not change the command shape or add optional parameters:\n\n"
    + FILESYSTEM_QWEN_CHECKPOINT_WRITE_ACTION
)
FILESYSTEM_QWEN_CHECKPOINT_READ_ACTION = (
    "<tool_call>\n"
    "<function=shell_command>\n"
    "<parameter=command>\n"
    "cat .agent_memory/CONTINUATION.md\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
FILESYSTEM_QWEN_CHECKPOINT_CONTINUATION_MARKER = (
    FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER
    + " The required checkpoint read must happen before any dependent task "
    "action. Output exactly this Qwen XML action and no other text:\n\n"
    + FILESYSTEM_QWEN_CHECKPOINT_READ_ACTION
)


def checkpoint_retry_trigger_tokens(
    pressure: Any,
    *,
    control_request: str,
) -> int:
    """Project the latest safe point for a checkpoint with one failed retry.

    The first term reserves one ordinary action/observation before the wrapper
    asks for a checkpoint. The remaining terms reserve the checkpoint request,
    its sampled action, the bounded failure observation, and a second request.
    In particular, the second observation must *not* use the route-wide bound:
    LiteResearcher needs a large bound for Visit pages, while a checkpoint
    failure is deliberately reduced to a small task-neutral message.
    """

    if not isinstance(control_request, str) or not control_request.strip():
        raise ValueError("filesystem checkpoint control request must be nonempty")
    request_upper_bound = (
        len(control_request.encode("utf-8"))
        + FILESYSTEM_CHECKPOINT_REQUEST_TOKEN_SLACK
    )
    ordinary_turn = (
        int(pressure.max_response_tokens)
        + int(pressure.max_observation_tokens)
        + int(pressure.action_observation_envelope_tokens)
    )
    failed_checkpoint_turn = (
        int(pressure.max_response_tokens)
        + FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS
        + int(pressure.action_observation_envelope_tokens)
    )
    return (
        int(pressure.action_prompt_tokens)
        + ordinary_turn
        + failed_checkpoint_turn
        + 2 * request_upper_bound
    )


def checkpoint_retry_ceiling_tokens(
    pressure: Any,
    *,
    control_request: str,
) -> int:
    """Return the prompt ceiling needed once a checkpoint is already due.

    Event-driven boundaries such as a WebShop session transition cannot move the
    checkpoint request earlier.  At that point the candidate prompt already
    contains the first request, so reserve only its sampled response, the bounded
    failure observation, and one retry request.
    """

    if not isinstance(control_request, str) or not control_request.strip():
        raise ValueError("filesystem checkpoint control request must be nonempty")
    request_upper_bound = (
        len(control_request.encode("utf-8"))
        + FILESYSTEM_CHECKPOINT_REQUEST_TOKEN_SLACK
    )
    failed_checkpoint_turn = (
        int(pressure.max_response_tokens)
        + FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS
        + int(pressure.action_observation_envelope_tokens)
    )
    return (
        int(pressure.candidate_prompt_tokens)
        + failed_checkpoint_turn
        + request_upper_bound
    )


def build_filesystem_checkpoint_retry_observation(reason: str) -> str:
    """Return the bounded model-visible result of a failed checkpoint write."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("filesystem checkpoint failure reason must be nonempty")
    observation = (
        f"Filesystem checkpoint was not accepted ({reason}). The earlier context "
        f"is still present. Retry now with exactly one Qwen XML shell_command or "
        f"apply_patch function call that overwrites `{FILESYSTEM_CHECKPOINT_PATH}` with 1 to "
        f"{FILESYSTEM_CHECKPOINT_MAX_BYTES} bytes."
    )
    if (
        len(observation.encode("utf-8"))
        > FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS
    ):
        raise AssertionError(
            "filesystem checkpoint retry observation exceeded its bound"
        )
    return observation


def build_filesystem_checkpoint_write_retry_context(
    messages: Sequence[Mapping[str, str]],
    reason: str,
) -> list[dict[str, str]]:
    """Rebuild a stable prompt after a failed checkpoint-write action.

    ``messages`` is the complete policy context captured immediately before the
    first checkpoint request. The sampled failed action and native observation
    remain in the PPO trajectory ledger, but neither is copied into the next
    prompt. Reusing the same captured context on every retry preserves all
    pre-boundary task evidence without letting malformed writes grow the prompt.
    """

    replacement = _normalize_checkpoint_framing(messages)
    retry = build_filesystem_checkpoint_retry_observation(reason)
    if replacement[-1]["role"] == "user":
        replacement[-1]["content"] = (
            f"{replacement[-1]['content']}\n\n{retry}"
        )
    else:
        replacement.append({"role": "user", "content": retry})
    return replacement


def build_filesystem_checkpoint_read_retry_observation(reason: str) -> str:
    """Return a bounded reminder while the verified continuation is unread."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("filesystem checkpoint read failure reason must be nonempty")
    observation = (
        f"Checkpoint read failed ({reason}). Next, use one shell_command to output "
        f"the complete unchanged `{FILESYSTEM_CHECKPOINT_PATH}` and no other "
        "stdout. The file remains on disk; reading it is still required."
    )
    if (
        len(observation.encode("utf-8"))
        > FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS
    ):
        raise AssertionError(
            "filesystem checkpoint read retry observation exceeded its bound"
        )
    return observation


def filesystem_workspace_action_request_sha256(action: Any) -> str | None:
    """Hash the canonical endpoint tool input for one workspace action.

    ``PersistentWorkspace`` hashes parsed tool input rather than raw policy text.
    Recomputing that exact digest client-side binds an endpoint event to the
    action dispatched on this turn instead of merely comparing two endpoint
    copies that could both be stale.
    """

    if not isinstance(action, str):
        return None
    text = action.strip()
    shell_match = _FILESYSTEM_SHELL_ACTION_RE.fullmatch(text)
    if shell_match is not None:
        try:
            payload = json.loads(shell_match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if text.startswith(_FILESYSTEM_APPLY_PATCH_PREFIX):
        patch_text = text[len(_FILESYSTEM_APPLY_PATCH_PREFIX) :]
        return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    return None


def _filesystem_checkpoint_exact_printf_payload(command: str) -> bytes | None:
    """Parse the bounded command-only ``printf`` checkpoint shape.

    The OpenMLE boundary prompt uses one safe token per checkpoint field.  Keep
    this parser task-neutral: accept one or more shell-safe literal tokens, but
    reject substitutions, extra commands, alternate redirections, and free-form
    shell syntax.  Both a literal ``\n`` format and a quoted physical newline
    are equivalent for POSIX ``printf``.
    """

    match = _FILESYSTEM_CHECKPOINT_PRINTF_RE.fullmatch(command)
    if match is None:
        return None
    arguments = re.split(r"[ \t]+", match.group("arguments"))
    try:
        return ("\n".join(arguments) + "\n").encode("utf-8")
    except UnicodeEncodeError:
        return None


def filesystem_checkpoint_exact_shell_payload(action: Any) -> bytes | None:
    """Extract bytes from an exact checkpoint shell action shown to the policy.

    A content-only workspace diff cannot distinguish a fresh overwrite from a
    stale file when both byte streams are identical.  The recognized commands
    are deliberately restricted to the two command-only shapes emitted by the
    shared boundary guidance: an anchored heredoc, or a bounded safe-token
    ``printf``.  The caller separately binds execution evidence to this action.
    """

    if not isinstance(action, str):
        return None
    shell_match = _FILESYSTEM_SHELL_ACTION_RE.fullmatch(action.strip())
    if shell_match is None:
        return None
    try:
        arguments = json.loads(shell_match.group(1))
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(arguments, dict)
        or "command" not in arguments
        or not set(arguments) <= {"command", "workdir", "timeout_ms"}
        or arguments.get("workdir", ".") != "."
        or not isinstance(arguments.get("command"), str)
    ):
        return None
    command = arguments["command"]
    match = _FILESYSTEM_CHECKPOINT_HEREDOC_RE.fullmatch(command)
    if match is not None:
        body = match.group("body")
        # A delimiter line inside the captured body would terminate the shell
        # heredoc early and turn the remaining text into executable commands.
        # Reject it rather than treating the final matching delimiter as proof
        # that the action was one canonical checkpoint write.
        if match.group("delimiter") in body.splitlines():
            return None
        try:
            return (body + "\n").encode("utf-8")
        except UnicodeEncodeError:
            return None
    return _filesystem_checkpoint_exact_printf_payload(command)


def _normalize_checkpoint_framing(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Normalize immutable task framing exactly as the successor builder does."""

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"checkpoint context message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"checkpoint context message {index} has invalid role: {role!r}"
            )
        if not isinstance(content, str):
            raise TypeError(
                f"checkpoint context message {index} content must be text"
            )
        normalized_message = {"role": role, "content": content}
        if role == "tool":
            name = message.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"checkpoint context message {index} tool result must carry "
                    "a nonempty name"
                )
            normalized_message["name"] = name
        normalized.append(normalized_message)
    if not normalized:
        raise ValueError("post-checkpoint context must preserve task framing")
    if normalized[-1]["role"] == "user":
        normalized[-1]["content"] = normalized[-1]["content"].rstrip()
    return normalized


def filesystem_checkpoint_framing_sha256(
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Bind a replacement to the immutable pre-episode task framing."""

    normalized = _normalize_checkpoint_framing(messages)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def filesystem_checkpoint_action_completed(
    action_kind: str,
    execution: Mapping[str, Any] | None,
) -> bool:
    """Return whether an endpoint-attested workspace action actually succeeded."""

    if not isinstance(execution, Mapping) or execution.get("status") != "executed":
        return False
    normalized_kind = str(action_kind).lower()
    if normalized_kind == "shell_command":
        exit_code = execution.get("exit_code")
        return bool(
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code == 0
            and execution.get("timed_out") is False
        )
    return normalized_kind == "apply_patch"


def build_post_checkpoint_context(
    messages: Sequence[Mapping[str, str]],
    receipt_value: Any,
    *,
    continuation_marker: str = FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER,
) -> list[dict[str, str]]:
    """Return a fresh context that names, but never injects, the checkpoint.

    The sampled write remains in the trajectory ledger for PPO credit.  It is
    deliberately absent from the successor prompt because shell/apply-patch
    payloads contain the file body and would otherwise become a free hidden
    read.  The policy must recover the bytes with a later ordinary action.
    """

    receipt = normalize_filesystem_checkpoint_receipt(receipt_value)
    if not filesystem_checkpoint_write_succeeded(receipt):
        raise ValueError("post-checkpoint context requires a successful receipt")
    if not isinstance(continuation_marker, str) or not continuation_marker.strip():
        raise ValueError("post-checkpoint continuation marker must be nonempty")
    normalized = _normalize_checkpoint_framing(messages)

    marker = (
        f"{continuation_marker.strip()} "
        f"Verified receipt: size_bytes={receipt['size_bytes']}, "
        f"sha256={receipt['sha256']}."
    )
    if normalized[-1]["role"] == "user":
        normalized[-1]["content"] = (
            f"{normalized[-1]['content']}\n\n{marker}"
        )
    else:
        normalized.append({"role": "user", "content": marker})
    return normalized


def build_post_checkpoint_read_retry_context(
    messages: Sequence[Mapping[str, str]],
    receipt_value: Any,
    reason: str,
    *,
    continuation_marker: str = FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER,
) -> list[dict[str, str]]:
    """Rebuild the bounded post-checkpoint prompt after a failed required read.

    The failed policy action and its native observation remain in the trajectory
    ledger, but neither is copied into the next prompt.  Rebuilding from the
    same trusted framing and verified checkpoint receipt makes repeated failed
    reads constant-size while still requiring a later ordinary filesystem read.
    """

    replacement = build_post_checkpoint_context(
        messages,
        receipt_value,
        continuation_marker=continuation_marker,
    )
    retry = build_filesystem_checkpoint_read_retry_observation(reason)
    if replacement[-1]["role"] == "user":
        replacement[-1]["content"] = (
            f"{replacement[-1]['content']}\n\n{retry}"
        )
    else:
        replacement.append({"role": "user", "content": retry})
    return replacement


def build_filesystem_checkpoint_receipt(
    *,
    action_kind: str,
    action_completed: bool,
    submitted_action: str | None = None,
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
            if item.get("path") == path and _is_explicit_regular_file_entry(item):
                changed_entry = item
                break

    modified = diff.get("modified", ())
    if changed_entry is None and _is_mapping_sequence(modified):
        for item in modified:
            before = item.get("before")
            after = item.get("after")
            if (
                isinstance(before, Mapping)
                and isinstance(after, Mapping)
                and before.get("path") == path
                and after.get("path") == path
                and _is_explicit_regular_file_entry(before)
                and _is_explicit_regular_file_entry(after)
                and before.get("sha256") != after.get("sha256")
            ):
                changed_entry = after
                break

    snapshot_entry: Mapping[str, Any] | None = None
    snapshot = workspace_snapshot if isinstance(workspace_snapshot, Mapping) else {}
    files = snapshot.get("files", ())
    if _is_mapping_sequence(files):
        for item in files:
            if item.get("path") == path:
                snapshot_entry = item
                break

    size = _entry_size(snapshot_entry)
    sha256 = (
        snapshot_entry.get("sha256")
        if isinstance(snapshot_entry, Mapping)
        else None
    )
    regular_file = _is_explicit_regular_file_entry(snapshot_entry)
    deleted = diff.get("deleted", ())
    deleted_path = bool(
        _is_mapping_sequence(deleted)
        and any(item.get("path") == path for item in deleted)
    )
    exists = bool(snapshot_entry is not None and not deleted_path)
    content_changed = bool(
        changed_entry is not None
        and regular_file
        and _entry_size(changed_entry) == size
        and changed_entry.get("sha256") == sha256
    )
    idempotent_payload = (
        filesystem_checkpoint_exact_shell_payload(submitted_action)
        if not content_changed
        else None
    )
    idempotent_overwrite = bool(
        str(action_kind).lower() == "shell_command"
        and action_completed
        and exists
        and regular_file
        and idempotent_payload is not None
        and len(idempotent_payload) == size
        and hashlib.sha256(idempotent_payload).hexdigest() == sha256
    )
    write_observed = bool(content_changed or idempotent_overwrite)

    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": path,
        "action_kind": str(action_kind).lower(),
        "action_completed": bool(action_completed),
        "changed": content_changed,
        "idempotent_overwrite": idempotent_overwrite,
        "write_observed": write_observed,
        "exists": exists,
        "regular_file": regular_file,
        "size_bytes": size,
        "sha256": sha256,
    }


def build_filesystem_checkpoint_read_receipt(
    checkpoint_receipt: Any,
    *,
    action_kind: str,
    action_completed: bool,
    stdout: str | bytes | None,
) -> dict[str, Any]:
    """Attest an exact, non-mutating read of the checkpoint through stdout.

    The caller supplies endpoint-produced execution evidence, not policy text.
    Exact byte equality avoids crediting an echoed filename or a partial read.
    """

    receipt = normalize_filesystem_checkpoint_receipt(checkpoint_receipt)
    payload = (
        stdout.encode("utf-8")
        if isinstance(stdout, str)
        else (bytes(stdout) if isinstance(stdout, bytes) else None)
    )
    observed = False
    if (
        receipt is not None
        and str(action_kind).lower() == "shell_command"
        and bool(action_completed)
        and receipt["exists"]
        and receipt["regular_file"]
        and not receipt.get("write_observed", receipt["changed"])
        and isinstance(receipt["size_bytes"], int)
        and receipt["size_bytes"] > 0
        and receipt["sha256"] is not None
        and payload is not None
        and len(payload) == receipt["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == receipt["sha256"]
    ):
        observed = True
    return {
        "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "observed": observed,
        "size_bytes": receipt["size_bytes"] if receipt is not None else None,
        "sha256": receipt["sha256"] if receipt is not None else None,
    }


def filesystem_checkpoint_read_observed(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if set(value) != {"schema", "path", "observed", "size_bytes", "sha256"}:
        return False
    if value.get("schema") != FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA:
        return False
    if value.get("path") != FILESYSTEM_CHECKPOINT_PATH:
        return False
    if type(value.get("observed")) is not bool:
        return False
    size = value.get("size_bytes")
    digest = value.get("sha256")
    return bool(
        value["observed"]
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < size <= FILESYSTEM_CHECKPOINT_MAX_BYTES
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def filesystem_checkpoint_read_matches(
    read_value: Any,
    checkpoint_value: Any,
) -> bool:
    """Require an endpoint-attested full read of the saved checkpoint identity."""

    if not filesystem_checkpoint_read_observed(read_value):
        return False
    checkpoint = normalize_filesystem_checkpoint_receipt(checkpoint_value)
    if not filesystem_checkpoint_write_succeeded(checkpoint):
        return False
    return bool(
        read_value.get("size_bytes") == checkpoint["size_bytes"]
        and read_value.get("sha256") == checkpoint["sha256"]
    )


def filesystem_checkpoint_read_failure_reason(
    read_value: Any,
    checkpoint_value: Any,
) -> str | None:
    """Classify why the required post-boundary checkpoint read did not pass."""

    if not filesystem_checkpoint_read_observed(read_value):
        return "checkpoint_read_not_observed"
    if not filesystem_checkpoint_read_matches(read_value, checkpoint_value):
        return "checkpoint_read_identity_mismatch"
    return None


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
    schema = value.get("schema")
    if schema == FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA:
        expected = expected | {"idempotent_overwrite", "write_observed"}
    elif schema != FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA_V1:
        raise RuntimeError("filesystem checkpoint receipt version drifted")
    if set(value) != expected:
        raise RuntimeError("filesystem checkpoint receipt schema drifted")
    if value.get("path") != FILESYSTEM_CHECKPOINT_PATH:
        raise RuntimeError("filesystem checkpoint receipt reports the wrong path")
    action_kind = value.get("action_kind")
    if not isinstance(action_kind, str):
        raise RuntimeError("filesystem checkpoint action kind must be text")
    for key in ("action_completed", "changed", "exists", "regular_file"):
        if type(value.get(key)) is not bool:
            raise RuntimeError(f"filesystem checkpoint {key} must be boolean")
    if schema == FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA:
        for key in ("idempotent_overwrite", "write_observed"):
            if type(value.get(key)) is not bool:
                raise RuntimeError(f"filesystem checkpoint {key} must be boolean")
        if value["changed"] and value["idempotent_overwrite"]:
            raise RuntimeError("filesystem checkpoint write evidence is ambiguous")
        if value["write_observed"] != bool(
            value["changed"] or value["idempotent_overwrite"]
        ):
            raise RuntimeError("filesystem checkpoint write evidence is inconsistent")
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
    normalized = {
        "schema": schema,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "action_kind": action_kind.lower(),
        "action_completed": value["action_completed"],
        "changed": value["changed"],
        "exists": value["exists"],
        "regular_file": value["regular_file"],
        "size_bytes": size,
        "sha256": sha256,
    }
    if schema == FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA:
        normalized.update(
            {
                "idempotent_overwrite": value["idempotent_overwrite"],
                "write_observed": value["write_observed"],
            }
        )
    return normalized


def bind_filesystem_checkpoint_receipt_to_submitted_action(
    value: Any,
    *,
    submitted_action: str | None,
) -> dict[str, Any] | None:
    """Bind a server receipt to the exact current checkpoint write action.

    Legacy endpoints report content diffs only, so a successful overwrite with
    identical bytes appears unchanged. Upgrade that receipt only when the
    current canonical heredoc payload exactly matches the attested file
    identity. A pre-existing file or unrelated action remains insufficient.
    """

    receipt = normalize_filesystem_checkpoint_receipt(value)
    if receipt is None:
        return None

    payload = (
        filesystem_checkpoint_exact_shell_payload(submitted_action)
        if not receipt["changed"] and receipt["action_kind"] == "shell_command"
        else None
    )
    if receipt["schema"] == FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA:
        if not receipt["idempotent_overwrite"]:
            return receipt
        size = receipt["size_bytes"]
        digest = receipt["sha256"]
        current_action_matches = bool(
            receipt["action_completed"]
            and receipt["exists"]
            and receipt["regular_file"]
            and payload is not None
            and isinstance(size, int)
            and not isinstance(size, bool)
            and len(payload) == size
            and isinstance(digest, str)
            and hashlib.sha256(payload).hexdigest() == digest
        )
        if current_action_matches:
            return receipt
        return {
            **receipt,
            "idempotent_overwrite": False,
            "write_observed": bool(receipt["changed"]),
        }

    size = receipt["size_bytes"]
    digest = receipt["sha256"]
    idempotent_overwrite = bool(
        receipt["action_completed"]
        and receipt["exists"]
        and receipt["regular_file"]
        and payload is not None
        and isinstance(size, int)
        and not isinstance(size, bool)
        and len(payload) == size
        and isinstance(digest, str)
        and hashlib.sha256(payload).hexdigest() == digest
    )
    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": receipt["path"],
        "action_kind": receipt["action_kind"],
        "action_completed": receipt["action_completed"],
        "changed": receipt["changed"],
        "idempotent_overwrite": idempotent_overwrite,
        "write_observed": bool(receipt["changed"] or idempotent_overwrite),
        "exists": receipt["exists"],
        "regular_file": receipt["regular_file"],
        "size_bytes": size,
        "sha256": digest,
    }


def filesystem_checkpoint_write_succeeded(value: Any) -> bool:
    receipt = normalize_filesystem_checkpoint_receipt(value)
    if receipt is None:
        return False
    size = receipt["size_bytes"]
    write_observed = bool(receipt.get("write_observed", receipt["changed"]))
    return bool(
        receipt["action_kind"] in FILESYSTEM_CHECKPOINT_ACTION_KINDS
        and receipt["action_completed"]
        and write_observed
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
    if not receipt.get("write_observed", receipt["changed"]):
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


def _is_explicit_regular_file_entry(entry: Any) -> bool:
    if not isinstance(entry, Mapping) or entry.get("kind") != "file":
        return False
    size = _entry_size(entry)
    digest = entry.get("sha256")
    return bool(
        size is not None
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )
