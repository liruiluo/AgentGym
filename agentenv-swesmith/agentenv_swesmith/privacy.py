from __future__ import annotations

import hmac
import os


def private_detail_authorized(provided: str | None) -> bool:
    expected = os.environ.get("SWESMITH_DETAIL_TOKEN")
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided, expected)
