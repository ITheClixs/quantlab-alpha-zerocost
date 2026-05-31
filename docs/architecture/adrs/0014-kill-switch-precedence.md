# ADR 0014: Kill-switch precedence — file flag is the first risk gate

## Status
Accepted, 2026-05-20.

## Context
Multiple risk checks could raise on a given signal: drawdown breach, exposure cap,
feed gap, etc. If the file-flag kill check runs after another check, an operator
who touches `KILL_TRADING` could see the daemon halt for the *other* reason — not
for their override. Worse, if the other check has a bug and never raises, the
operator's override is silently defeated.

## Decision
The kill-flag check is the *first* call in `RiskGate.evaluate()`, before all other
gates. Its precedence is enforced by:

1. A unit test asserting the check order at module-level.
2. A code comment in `risk.py` documenting the invariant.
3. The `RiskGate` class running checks via an explicit list (`_GATES`) where the
   first element is always `kill_flag_check`.

## Consequences
Operator intent is always honored before any algorithmic check fires. The cost is
that the kill flag is checked on every signal, but the check is a single `stat()`
call (sub-millisecond) so the overhead is negligible.
