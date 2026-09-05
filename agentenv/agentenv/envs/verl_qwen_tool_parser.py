"""Thin text adapter around the pinned veRL Qwen3 tool parser.

Environment clients receive decoded policy text rather than token ids. Reuse
veRL's parser implementation without moving environment parsing into the shared
AgentLoop or maintaining a second XML grammar here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


QWEN_SINGLE_TOOL_CALL_CONTRACT = (
    "Every policy response must contain exactly one Qwen XML `<tool_call>` "
    "envelope and nothing else. Start with `<tool_call>` and end with "
    "`</tool_call>`; do not emit reasoning, prose, Markdown fences, `<think>`, "
    "bare JSON, or bare tool syntax."
)

# The endpoint-side legacy parsers intentionally remain capable of reading their
# canonical internal actions. Policy-facing clients replace malformed/non-XML
# model output with this value so those legacy parsers cannot accidentally execute
# a bare action. The sentinel is deliberately outside every environment grammar.
QWEN_INVALID_ACTION_SENTINEL = "__AMG_INVALID_QWEN_TOOL_CALL__"

_LEGACY_MARKER_IN_JSON_STRING_RE = re.compile(
    r"(?i)(?:"
    r"(?<![A-Za-z0-9_])(?P<bracket>search|click)\["
    r"|(?<![A-Za-z0-9_])(?P<object>shell_command|ask|submit)\s+\{"
    r"|(?P<patch>`apply_patch`|(?<![A-Za-z0-9_`])apply_patch(?!`))"
    r"(?=(?:\s+followed|\s*\\n\*\*\* Begin Patch))"
    r")"
)


def _inert_exact_value(value: Any) -> tuple[str, str]:
    """Encode one record value reversibly without exposing legacy call syntax."""

    if isinstance(value, str):
        kind = "string"
        payload = value
    else:
        kind = "canonical JSON"
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    encoded = json.dumps(payload, ensure_ascii=False)

    def escape_marker(match: re.Match[str]) -> str:
        matched = match.group(0)
        if match.group("bracket") is not None:
            return matched[:-1] + "\\u005b"
        if match.group("object") is not None:
            return matched[:-1] + "\\u007b"
        underscore = matched.find("_")
        if underscore < 0:
            raise AssertionError("apply_patch marker lost its underscore")
        return matched[:underscore] + "\\u005f" + matched[underscore + 1 :]

    encoded = _LEGACY_MARKER_IN_JSON_STRING_RE.sub(escape_marker, encoded)
    # A prior record can contain model-authored XML-looking text. Keep it exact
    # but non-callable in the next policy prompt for the same reason as legacy
    # endpoint syntax. JSON decoding restores both characters losslessly.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    return kind, encoded


def describe_inert_qwen_function_record(
    name: str, arguments: Mapping[str, Any]
) -> str:
    """Render an exact, non-callable record using only the Qwen function vocabulary.

    Values are JSON-string encoded so a reader can recover them exactly. Legacy
    action-looking substrings inside those values use JSON unicode escapes; this
    keeps the projection idempotent and prevents opaque stdout/checkpoint payloads
    from becoming a second executable-looking grammar.
    """

    if not arguments:
        return f"an inert record of the Qwen XML `{name}` function with no parameters"
    rendered_arguments = []
    for key, value in arguments.items():
        kind, encoded = _inert_exact_value(value)
        rendered_arguments.append(
            f"`{key}` exact {kind} value encoded as a JSON string: {encoded}"
        )
    return (
        f"an inert record of the Qwen XML `{name}` function; "
        + "; ".join(rendered_arguments)
    )


@dataclass(frozen=True)
class ParsedQwenToolCall:
    name: str
    arguments: Mapping[str, Any]
    parser_name: str = "qwen3_coder"


def parse_single_qwen3_tool_call(
    raw_output: str,
    *,
    tool_schemas: Sequence[Mapping[str, Any]],
) -> ParsedQwenToolCall | None:
    """Parse exactly one call with veRL's native Qwen3 semantics.

    The prompt still asks for a bare XML call, but execution follows the pinned
    upstream parser: harmless surrounding assistant content and a truncated
    final parameter closing tag are tolerated. We retain AMG's one-action,
    complete-envelope, and no-think safety boundaries, plus strict parameter
    structure and route validation.
    """

    if not isinstance(raw_output, str):
        raise TypeError("Qwen policy output must be text")
    text = raw_output.strip()
    if text.endswith("</s>"):
        text = text[:-4].rstrip()
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        return None
    if "<think>" in text or "</think>" in text:
        return None

    # These imports are intentionally lazy. Standalone environment utilities
    # that never opt into Qwen XML do not acquire a veRL import dependency; the
    # integrated AMG runtime pins and attests the exact veRL source revision.
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema

    parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer=None)
    schemas = [
        OpenAIFunctionToolSchema.model_validate(dict(item)) for item in tool_schemas
    ]

    # veRL currently exposes token-based async extraction publicly. The
    # environment adapter already receives decoded text, so reuse its complete
    # envelope regex and conversion helper without re-tokenizing or copying the
    # XML grammar. Unlike the previous ``fullmatch``, ``search`` preserves the
    # upstream behavior for harmless assistant content around one complete call.
    try:
        envelope = parser.tool_call_complete_regex.search(text)
        if envelope is None:
            return None
        envelope_body = envelope.group(1).strip()
        function_match = parser.tool_call_function_regex.fullmatch(envelope_body)
        if function_match is None or function_match.group(1) is None:
            return None
        raw_function_call = function_match.group(1)
        separator = raw_function_call.find(">")
        if separator < 0:
            return None
        parameter_body = raw_function_call[separator + 1 :]
        parameter_matches = list(
            parser.tool_call_parameter_regex.finditer(parameter_body)
        )
        cursor = 0
        parameter_names: set[str] = set()
        truncated_parameter_count = 0
        for parameter_index, parameter_match in enumerate(parameter_matches):
            if parameter_body[cursor : parameter_match.start()].strip():
                return None
            match_text = parameter_match.group(1)
            if match_text is None:
                # Qwen3 and veRL intentionally accept a missing closing tag on
                # the final parameter. Keep that upstream-native tolerance, but
                # only at the end of the single function body.
                match_text = parameter_match.group(2)
                truncated_parameter_count += 1
                if (
                    match_text is None
                    or parameter_index != len(parameter_matches) - 1
                    or parameter_body[parameter_match.end() :].strip()
                ):
                    return None
            name_separator = match_text.find(">")
            if name_separator < 0:
                return None
            parameter_name = match_text[:name_separator]
            parameter_value = match_text[name_separator + 1 :]
            # An unclosed earlier parameter can make the regex consume a later
            # parameter as plain text. Do not let nested parameter markup become
            # command payload.
            if (
                "<parameter=" in parameter_value
                or "</parameter>" in parameter_value
            ):
                return None
            if not parameter_name or parameter_name in parameter_names:
                return None
            parameter_names.add(parameter_name)
            cursor = parameter_match.end()
        if parameter_body[cursor:].strip() or truncated_parameter_count > 1:
            return None
        function_call = parser._parse_xml_function_call(raw_function_call, schemas)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    if function_call is None:
        return None
    try:
        arguments = json.loads(function_call.arguments)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(arguments, dict):
        return None
    return ParsedQwenToolCall(
        name=function_call.name,
        arguments=arguments,
    )
