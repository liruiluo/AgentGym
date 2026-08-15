from __future__ import annotations


def bound_text(
    value: str,
    *,
    max_bytes: int,
    max_tokens: int | None = None,
    marker: str = "\n...[observation truncated]...\n",
) -> str:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("text byte cap must be a positive integer")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("text token cap must be a positive integer")
    cap = min(max_bytes, max_tokens) if max_tokens is not None else max_bytes
    payload = value.encode("utf-8", errors="replace")
    normalized = payload.decode("utf-8")
    if len(payload) <= cap:
        return normalized
    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) >= cap:
        return _prefix(payload, cap).decode("utf-8")
    available = cap - len(marker_bytes)
    head = _prefix(payload, available // 2)
    tail = _suffix(payload, available - len(head))
    bounded = head + marker_bytes + tail
    if len(bounded) > cap:
        raise AssertionError("bounded UTF-8 text exceeded its cap")
    return bounded.decode("utf-8")


def _prefix(payload: bytes, maximum: int) -> bytes:
    chunk = payload[:maximum]
    while chunk:
        try:
            chunk.decode("utf-8")
            return chunk
        except UnicodeDecodeError as exc:
            chunk = chunk[: exc.start]
    return b""


def _suffix(payload: bytes, maximum: int) -> bytes:
    if maximum <= 0:
        return b""
    chunk = payload[-maximum:]
    while chunk and chunk[0] & 0b1100_0000 == 0b1000_0000:
        chunk = chunk[1:]
    return chunk
