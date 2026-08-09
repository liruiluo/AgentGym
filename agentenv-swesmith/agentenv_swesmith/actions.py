from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping


ActionKind = Literal["shell_command", "apply_patch", "final", "parser_error"]

_SHELL_PREFIX = "shell_command "
_PATCH_PREFIX = "apply_patch\n"
_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"
_THINK_START_RE = re.compile(r"\A<think\s*>", re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_TOOLISH_PREFIX_RE = re.compile(
    r"\A(?:Action\s*:\s*)?(?:shell[\s_-]*command|apply[\s_-]*patch)\b",
    re.IGNORECASE,
)
_FENCED_TOOLISH_RE = re.compile(
    r"\A```(?:json|python|bash|shell)?\s*\n?"
    r"(?:shell[\s_-]*command|apply[\s_-]*patch)\b",
    re.IGNORECASE,
)
_EMBEDDED_SHELL_ATTEMPT_RE = re.compile(
    r"\bshell_command\s+\{",
    re.IGNORECASE,
)
_EMBEDDED_PATCH_ATTEMPT_RE = re.compile(
    r"(?:<tool_call>\s*)?apply_patch\s*\r?\n\s*\*\*\* Begin Patch",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedPolicyAction:
    kind: ActionKind
    raw_output: str
    action_text: str
    thought: str = ""
    arguments: Mapping[str, Any] | None = None
    patch: str | None = None
    final_response: str | None = None
    error: str | None = None
    tool_hint: str | None = None

    @property
    def is_tool(self) -> bool:
        return self.kind in {"shell_command", "apply_patch"}

    @property
    def terminates_episode(self) -> bool:
        return self.kind == "final"

    def as_evidence(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "raw_output": self.raw_output,
            "action_text": self.action_text,
            "thought": self.thought,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "patch": self.patch,
            "final_response": self.final_response,
            "error": self.error,
            "tool_hint": self.tool_hint,
        }


class _DuplicateJsonKey(ValueError):
    pass


def parse_policy_action(raw_output: str) -> ParsedPolicyAction:
    """Classify one sampled SWE-smith turn without executing policy text.

    A plain, non-empty response is a terminal submission. Anything that looks
    like an attempted workspace tool call is kept on the non-terminal parser
    error path unless it exactly matches the canonical tool grammar.
    """

    if not isinstance(raw_output, str):
        raise TypeError(
            f"SWE-smith policy output must be text, got {type(raw_output).__name__}"
        )
    thought, action_text, framing_error = _split_thinking(raw_output)
    if framing_error is not None:
        return _parser_error(
            raw_output,
            action_text,
            thought,
            framing_error,
        )
    action_text = _strip_single_eos(action_text).strip()
    if not action_text:
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "policy output is empty after removing optional thinking text",
        )

    if action_text.startswith(_SHELL_PREFIX):
        return _parse_shell_command(raw_output, action_text, thought)
    if action_text.startswith(_PATCH_PREFIX):
        return _parse_apply_patch(raw_output, action_text, thought)

    tool_hint = _infer_toolish_attempt(action_text)
    if tool_hint is not None:
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "attempted tool call does not use the canonical SWE-smith grammar",
            tool_hint=tool_hint,
        )
    return ParsedPolicyAction(
        kind="final",
        raw_output=raw_output,
        action_text=action_text,
        thought=thought,
        final_response=action_text,
    )


def _parse_shell_command(
    raw_output: str,
    action_text: str,
    thought: str,
) -> ParsedPolicyAction:
    payload_text = action_text[len(_SHELL_PREFIX) :]
    try:
        payload = json.loads(payload_text, object_pairs_hook=_unique_json_object)
    except (_DuplicateJsonKey, json.JSONDecodeError) as exc:
        return _parser_error(
            raw_output,
            action_text,
            thought,
            f"shell_command payload must be one valid JSON object: {exc}",
            tool_hint="shell_command",
        )
    if not isinstance(payload, dict):
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "shell_command payload must be a JSON object",
            tool_hint="shell_command",
        )
    allowed = {"command", "workdir", "timeout_ms"}
    unexpected = sorted(set(payload) - allowed)
    missing = sorted({"command"} - set(payload))
    if unexpected or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "shell_command arguments are invalid: " + "; ".join(detail),
            tool_hint="shell_command",
        )
    command = payload["command"]
    if not isinstance(command, str) or not command or "\x00" in command:
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "shell_command command must be a non-empty string without NUL bytes",
            tool_hint="shell_command",
        )
    workdir = payload.get("workdir", ".")
    if not isinstance(workdir, str) or not workdir or "\x00" in workdir:
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "shell_command workdir must be a non-empty string without NUL bytes",
            tool_hint="shell_command",
        )
    timeout_ms = payload.get("timeout_ms")
    if timeout_ms is not None and (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or timeout_ms <= 0
    ):
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "shell_command timeout_ms must be a positive integer",
            tool_hint="shell_command",
        )
    normalized: dict[str, Any] = {"command": command, "workdir": workdir}
    if timeout_ms is not None:
        normalized["timeout_ms"] = timeout_ms
    return ParsedPolicyAction(
        kind="shell_command",
        raw_output=raw_output,
        action_text=action_text,
        thought=thought,
        arguments=normalized,
        tool_hint="shell_command",
    )


def _parse_apply_patch(
    raw_output: str,
    action_text: str,
    thought: str,
) -> ParsedPolicyAction:
    patch = action_text[len(_PATCH_PREFIX) :]
    lines = patch.splitlines()
    if (
        not lines
        or lines[0] != _PATCH_BEGIN
        or lines[-1] != _PATCH_END
        or len(lines) < 3
    ):
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "apply_patch requires one complete *** Begin Patch ... *** End Patch payload",
            tool_hint="apply_patch",
        )
    if any(line == _PATCH_BEGIN for line in lines[1:]) or any(
        line == _PATCH_END for line in lines[:-1]
    ):
        return _parser_error(
            raw_output,
            action_text,
            thought,
            "apply_patch payload contains nested or trailing patch delimiters",
            tool_hint="apply_patch",
        )
    return ParsedPolicyAction(
        kind="apply_patch",
        raw_output=raw_output,
        action_text=action_text,
        thought=thought,
        patch=patch,
        tool_hint="apply_patch",
    )


def _split_thinking(raw_output: str) -> tuple[str, str, str | None]:
    text = raw_output.strip()
    start = _THINK_START_RE.match(text)
    if start is None:
        if _THINK_END_RE.search(text):
            return "", text, "policy output contains an unmatched </think> tag"
        return "", text, None
    end = _THINK_END_RE.search(text, start.end())
    if end is None:
        return (
            text[start.end() :].strip(),
            "",
            "policy output contains an unclosed <think> block",
        )
    thought = text[start.end() : end.start()].strip()
    action_text = text[end.end() :].strip()
    if _THINK_START_RE.search(action_text) or _THINK_END_RE.search(action_text):
        return thought, action_text, "policy output contains multiple thinking blocks"
    return thought, action_text, None


def _strip_single_eos(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("</s>"):
        return stripped[:-4].rstrip()
    return text


def _infer_toolish_attempt(text: str) -> str | None:
    if _TOOLISH_PREFIX_RE.match(text) or _FENCED_TOOLISH_RE.match(text):
        lowered = text.lower()
        return "apply_patch" if "apply" in lowered[:80] else "shell_command"
    embedded_patch = _EMBEDDED_PATCH_ATTEMPT_RE.search(text)
    embedded_shell = _EMBEDDED_SHELL_ATTEMPT_RE.search(text)
    if embedded_patch is not None and (
        embedded_shell is None or embedded_patch.start() < embedded_shell.start()
    ):
        return "apply_patch"
    if embedded_shell is not None:
        return "shell_command"
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            candidate = payload.get("function_name", payload.get("name"))
            if isinstance(candidate, str):
                normalized = candidate.lower().replace("-", "_").replace(" ", "_")
                if normalized in {"shell_command", "apply_patch"}:
                    return normalized
    return None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKey(f"duplicate key {key!r}")
        value[key] = item
    return value


def _parser_error(
    raw_output: str,
    action_text: str,
    thought: str,
    error: str,
    *,
    tool_hint: str | None = None,
) -> ParsedPolicyAction:
    return ParsedPolicyAction(
        kind="parser_error",
        raw_output=raw_output,
        action_text=action_text,
        thought=thought,
        error=error,
        tool_hint=tool_hint,
    )
