from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MAX_INSPECT_BYTES = 65_536
MAX_EDIT_BYTES = 1_048_576
MAX_COMMAND_BYTES = 32_768
MAX_SHELL_TIMEOUT_MS = 86_400_000


@dataclass(frozen=True)
class PolicyAction:
    kind: str
    path: str | None = None
    offset: int = 0
    max_bytes: int = 4096
    content: str | None = None
    command: str | None = None
    timeout_ms: int = 3_600_000


def parse_policy_action(raw: str) -> PolicyAction:
    if not isinstance(raw, str) or raw != raw.strip() or not raw:
        return PolicyAction("parser_error")
    if raw == "submit":
        return PolicyAction("submit")
    if "\n" in raw or "\r" in raw or " " not in raw:
        return PolicyAction("parser_error")
    kind, encoded = raw.split(" ", 1)
    if kind not in {"inspect", "edit", "shell"}:
        return PolicyAction("parser_error")
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return PolicyAction("parser_error")
    if not isinstance(value, dict):
        return PolicyAction("parser_error")
    if kind == "inspect":
        if not {"path"} <= set(value) <= {"path", "offset", "max_bytes"}:
            return PolicyAction("parser_error")
        path = value.get("path")
        offset = value.get("offset", 0)
        max_bytes = value.get("max_bytes", 4096)
        if (
            not isinstance(path, str)
            or not path
            or _utf8_size(path) is None
            or not _plain_int(offset, minimum=0)
            or not _plain_int(max_bytes, minimum=1, maximum=MAX_INSPECT_BYTES)
        ):
            return PolicyAction("parser_error")
        return PolicyAction("inspect", path=path, offset=offset, max_bytes=max_bytes)
    if kind == "edit":
        if set(value) != {"path", "content"}:
            return PolicyAction("parser_error")
        path = value.get("path")
        content = value.get("content")
        if (
            not isinstance(path, str)
            or not path
            or _utf8_size(path) is None
            or not isinstance(content, str)
            or (_utf8_size(content) is None)
            or _utf8_size(content) > MAX_EDIT_BYTES
        ):
            return PolicyAction("parser_error")
        return PolicyAction("edit", path=path, content=content)
    if set(value) - {"command", "timeout_ms"} or "command" not in value:
        return PolicyAction("parser_error")
    command = value.get("command")
    timeout_ms = value.get("timeout_ms", 3_600_000)
    if (
        not isinstance(command, str)
        or not command
        or (_utf8_size(command) is None)
        or _utf8_size(command) > MAX_COMMAND_BYTES
        or not _plain_int(timeout_ms, minimum=1, maximum=MAX_SHELL_TIMEOUT_MS)
    ):
        return PolicyAction("parser_error")
    return PolicyAction("shell", command=command, timeout_ms=timeout_ms)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _plain_int(value: Any, *, minimum: int, maximum: int | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return False
    return maximum is None or value <= maximum


def _utf8_size(value: str) -> int | None:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return None
