from __future__ import annotations

import pytest

from quant_research_stack.feeds.rate_limit import RateLimiter


class _FakeClock:
    """Deterministic monotonic clock; sleep advances time."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _limiter(max_calls: int, period: float) -> tuple[RateLimiter, _FakeClock]:
    clock = _FakeClock()
    rl = RateLimiter(max_calls=max_calls, period_seconds=period, clock=clock.now, sleep=clock.sleep)
    return rl, clock


def test_calls_under_limit_never_sleep() -> None:
    rl, _clock = _limiter(5, 60.0)
    slept = [rl.acquire() for _ in range(5)]
    assert slept == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_sixth_call_in_window_sleeps_until_oldest_expires() -> None:
    rl, clock = _limiter(5, 60.0)
    for _ in range(5):
        rl.acquire()  # all at t=0
    slept = rl.acquire()  # 6th call must wait the full 60s window
    assert slept == pytest.approx(60.0)
    assert clock.now() == pytest.approx(60.0)


def test_window_slides_so_old_calls_do_not_count() -> None:
    rl, clock = _limiter(5, 60.0)
    for _ in range(5):
        rl.acquire()
    clock.t = 61.0  # all five fall out of the window
    assert rl.acquire() == 0.0  # no sleep needed


def test_partial_wait_when_some_calls_still_in_window() -> None:
    rl, clock = _limiter(2, 60.0)
    rl.acquire()  # t=0
    clock.t = 10.0
    rl.acquire()  # t=10
    # third call: oldest (t=0) expires at t=60, now=10 -> wait 50
    slept = rl.acquire()
    assert slept == pytest.approx(50.0)


def test_invalid_max_calls_raises() -> None:
    rl, _clock = _limiter(0, 60.0)
    with pytest.raises(ValueError):
        rl.acquire()
