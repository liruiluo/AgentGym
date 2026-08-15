from __future__ import annotations

import math
import time
from dataclasses import dataclass


class DeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class MonotonicDeadline:
    """One absolute monotonic deadline shared by every stage of an operation."""

    expires_at: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, (int, float))
            or not math.isfinite(self.expires_at)
        ):
            raise ValueError("monotonic deadline must be finite")

    @classmethod
    def after_ms(
        cls,
        duration_ms: int,
        *,
        cap: MonotonicDeadline | None = None,
    ) -> MonotonicDeadline:
        if type(duration_ms) is not int or duration_ms <= 0:
            raise ValueError("deadline duration must be a positive integer")
        expires_at = time.monotonic() + duration_ms / 1000.0
        if cap is not None:
            expires_at = min(expires_at, cap.expires_at)
        return cls(expires_at)

    def check(self) -> None:
        if time.monotonic() >= self.expires_at:
            raise DeadlineExceeded("monotonic deadline expired")

    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def remaining_seconds(self) -> float:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceeded("monotonic deadline expired")
        return remaining

    def remaining_milliseconds(self) -> int:
        remaining = int(self.remaining_seconds() * 1000.0)
        if remaining <= 0:
            raise DeadlineExceeded("less than one millisecond remains")
        return remaining
