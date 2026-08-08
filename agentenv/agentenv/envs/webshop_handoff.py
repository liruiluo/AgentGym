from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import PurePosixPath
from typing import Any


WEBSHOP_HANDOFF_PARSE_SCHEMA = "agentmemory_webshop_handoff_parse_v1"
WEBSHOP_SESSION_HANDOFF_REQUEST = (
    "The native shopping session has just reset. Write only a locator for an "
    "external workspace note you already created: either its workspace-relative "
    "path or a generic command for finding and reading it. Do not include "
    "shopping facts, product choices, progress, previous actions or observations, "
    "or any other task state. If no note locator is needed, state only that no "
    "locator is available. Your response will be preserved verbatim and will not "
    "be sent to the shopping environment."
)
_READONLY_COMMANDS = frozenset(
    {"cat", "rg", "grep", "find", "ls", "head", "tail", "sed"}
)
_NO_LOCATOR_RE = re.compile(
    r"^(?:none|no[- ]locator(?: (?:is )?(?:available|needed))?)[.!。]?$",
    re.IGNORECASE,
)


def parse_webshop_session_handoff(content: str) -> dict[str, Any]:
    """Fail closed while retaining raw policy output outside this parser."""

    if not isinstance(content, str):
        return _invalid(str(content), "not_text")
    raw = content
    value = raw.strip()
    if not value:
        return _invalid(raw, "empty")
    if "\n" in value or "\r" in value:
        return _invalid(raw, "multiline")
    if any(ord(char) < 32 and char != "\t" for char in value):
        return _invalid(raw, "control_character")
    if _NO_LOCATOR_RE.fullmatch(value):
        return _result(raw, valid=True, kind="no_locator", forwarded=None)

    command_ok, command_reason = _parse_readonly_command(value)
    if command_ok:
        return _result(
            raw,
            valid=True,
            kind="readonly_discovery_command",
            forwarded=value,
        )
    path_ok, path_reason = _parse_relative_path(value)
    if path_ok:
        return _result(
            raw,
            valid=True,
            kind="workspace_relative_path",
            forwarded=value,
        )
    reason = command_reason if command_reason != "not_readonly_command" else path_reason
    return _invalid(raw, reason or "invalid_locator")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result(
    raw: str, *, valid: bool, kind: str, forwarded: str | None, reason: str | None = None
) -> dict[str, Any]:
    return {
        "schema_version": WEBSHOP_HANDOFF_PARSE_SCHEMA,
        "raw_content_sha256": _sha256(raw),
        "valid": valid,
        "kind": kind,
        "forwarded_content": forwarded,
        "forwarded_content_sha256": None if forwarded is None else _sha256(forwarded),
        "rejection_reason": reason,
    }


def _invalid(raw: str, reason: str) -> dict[str, Any]:
    return _result(raw, valid=False, kind="invalid", forwarded=None, reason=reason)


def _forbidden_path(token: str) -> bool:
    lowered = token.lower()
    if (
        lowered.startswith(("file://", "file:", "~", "/"))
        or re.match(r"^[a-z]:[\\/]", lowered)
        or "\\" in token
    ):
        return True
    return ".." in re.split(r"[/\\]", token)


def _parse_readonly_command(value: str) -> tuple[bool, str]:
    if any(operator in value for operator in (";", "|", "&", ">", "<", "`")):
        return False, "shell_operator"
    if "$(" in value or "${" in value:
        return False, "shell_expansion"
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError:
        return False, "invalid_shell_quoting"
    if not tokens or tokens[0] not in _READONLY_COMMANDS:
        return False, "not_readonly_command"
    if any(_forbidden_path(token) for token in tokens[1:]):
        return False, "non_workspace_path"
    if tokens[0] == "sed" and any(
        token in {"-i", "--in-place"} or token.startswith("-i")
        for token in tokens[1:]
    ):
        return False, "write_command"
    if tokens[0] == "find" and any(
        token in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        for token in tokens[1:]
    ):
        return False, "write_or_child_process"
    return True, ""


def _parse_relative_path(value: str) -> tuple[bool, str]:
    if not value or any(char.isspace() for char in value):
        return False, "not_single_relative_path"
    if any(operator in value for operator in (";", "|", "&", ">", "<", "`")):
        return False, "shell_operator"
    if _forbidden_path(value) or "://" in value:
        return False, "non_workspace_path"
    try:
        path = PurePosixPath(value)
    except (TypeError, ValueError):
        return False, "invalid_relative_path"
    if path.is_absolute() or not path.parts or path.parts == (".",):
        return False, "invalid_relative_path"
    if any(part in {"", ".", ".."} for part in path.parts):
        return False, "invalid_relative_path"
    return True, ""
