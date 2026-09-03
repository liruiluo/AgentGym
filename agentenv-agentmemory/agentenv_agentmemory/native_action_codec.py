from __future__ import annotations

import re
from typing import Any


_NATIVE_ACTION_PREFIX_RE = re.compile(r"\A(search|click)\[")


def parse_native_bracket_action(text: str) -> tuple[str, str] | None:
    """Parse one native WebShop action with balanced brackets in its argument.

    Product titles can legitimately contain bracketed text such as ``[2022]``.
    The outermost brackets delimit the action argument; balanced brackets inside
    that argument are data.  Newlines, trailing text, and unbalanced input are
    rejected so one policy row can still encode exactly one native action.
    """

    if not isinstance(text, str) or "\n" in text or "\r" in text:
        return None
    prefix = _NATIVE_ACTION_PREFIX_RE.match(text)
    if prefix is None:
        return None

    depth = 1
    argument_start = prefix.end()
    for index in range(argument_start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                if index != len(text) - 1:
                    return None
                return prefix.group(1), text[argument_start:index]
            if depth < 0:
                return None
    return None


def parse_memoryarena_native_action(action: Any) -> tuple[Any, str | None]:
    """Compatibility parser for pinned MemoryArena's ``WebAgentTextEnv``.

    MemoryArena's original regex greedily splits at an inner closing bracket.
    Preserve its fallback contract for malformed/non-native input while fixing
    valid native actions whose product titles contain balanced brackets.
    """

    if not isinstance(action, str):
        return action, None
    parsed = parse_native_bracket_action(action)
    if parsed is None:
        return action, None
    return parsed
