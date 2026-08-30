"""Thin text adapter around the pinned veRL Qwen3 tool parser.

Environment clients receive decoded policy text rather than token ids. Reuse
veRL's parser implementation without moving environment parsing into the shared
AgentLoop or maintaining a second XML grammar here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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
    """Parse one envelope with veRL's native Qwen3 parser.

    The outer envelope remains strict so the no-prose, one-action AMG contract
    is unchanged. Inside that envelope, the pinned upstream parser owns the
    tolerant Qwen XML semantics used by veRL's ``ToolAgentLoop``.
    """

    if not isinstance(raw_output, str):
        raise TypeError("Qwen policy output must be text")
    text = raw_output.strip()
    if not text.startswith("<tool_call>") or not text.endswith("</tool_call>"):
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
    # environment adapter already receives decoded text, so use the same
    # parser's text helpers rather than re-tokenizing or copying its grammar.
    raw_calls = parser._get_function_calls(text)
    if len(raw_calls) != 1:
        return None
    function_call = parser._parse_xml_function_call(raw_calls[0], schemas)
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
