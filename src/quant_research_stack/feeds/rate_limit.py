from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Sliding-window rate limiter: at most ``max_calls`` per ``period_seconds``.

    Deterministic and unit-testable: the monotonic clock and the sleep function
    are injectable. In production the defaults (``time.monotonic`` / ``time.sleep``)
    enforce a real wall-clock budget — Massive.com's free tier allows 5 REST calls
    per minute, so a ``RateLimiter(5, 60.0)`` keeps the client inside that envelope.
    """

    max_calls: int
    period_seconds: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _calls: list[float] = field(default_factory=list, init=False, repr=False)

    def acquire(self) -> float:
        """Block until a call slot is free. Returns the seconds slept (0.0 if none)."""
        if self.max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        now = self.clock()
        self._purge(now)
        slept = 0.0
        if len(self._calls) >= self.max_calls:
            wait = self._calls[0] + self.period_seconds - now
            if wait > 0:
                self.sleep(wait)
                slept = wait
                now = self.clock()
                self._purge(now)
        self._calls.append(now)
        return slept

    def _purge(self, now: float) -> None:
        cutoff = now - self.period_seconds
        self._calls = [t for t in self._calls if t > cutoff]
