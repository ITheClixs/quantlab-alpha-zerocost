from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_research_stack.execution.audit import AuditLog
from quant_research_stack.execution.types import ExecutionTicket, S1Signal
from quant_research_stack.governor.signal_schema import GovernorVerdict


class SignalIngestor:
    """Tails S1 predictions Parquet + S2 verdicts JSONL; emits ExecutionTickets."""

    def __init__(
        self,
        preds_dir: Path,
        verdicts_dir: Path,
        poll_interval_s: float,
        pair_window_s: int,
        audit: AuditLog,
    ) -> None:
        self._preds_dir = Path(preds_dir)
        self._verdicts_dir = Path(verdicts_dir)
        self._poll = float(poll_interval_s)
        self._pair_window = int(pair_window_s)
        self._audit = audit
        self._stop = False
        self._seen_preds: set[str] = set()
        self._seen_verdicts: dict[str, GovernorVerdict] = {}
        self._pending_preds: dict[str, tuple[S1Signal, float]] = {}

    def stop(self) -> None:
        self._stop = True

    async def stream(self) -> AsyncIterator[ExecutionTicket]:
        while not self._stop:
            loop = asyncio.get_running_loop()
            self._scan_predictions(loop.time())
            self._scan_verdicts()
            for sig_id, (signal, first_seen) in list(self._pending_preds.items()):
                verdict = self._seen_verdicts.get(sig_id)
                if verdict is not None:
                    self._pending_preds.pop(sig_id, None)
                    self._audit.append("signal_ingested", {"signal_id": sig_id, "symbol": signal.symbol})
                    self._audit.append("verdict_received", {"signal_id": sig_id, "decision": verdict.decision.value})
                    yield ExecutionTicket(
                        signal=signal,
                        primary_verdict=verdict,
                        tier3_verdict=None,
                        ingested_at=datetime.now(UTC),
                    )
                elif loop.time() - first_seen > self._pair_window:
                    self._pending_preds.pop(sig_id, None)
                    self._audit.append(
                        "verdict_timeout",
                        {"signal_id": sig_id, "waited_seconds": self._pair_window},
                    )
            await asyncio.sleep(self._poll)

    def _scan_predictions(self, now_monotonic: float) -> None:
        if not self._preds_dir.exists():
            return
        for path in sorted(self._preds_dir.glob("*.parquet")):
            try:
                df = pl.read_parquet(path)
            except Exception:
                continue
            for row in df.iter_rows(named=True):
                sig_id = row.get("signal_id")
                if not sig_id or sig_id in self._seen_preds:
                    continue
                try:
                    signal = S1Signal(
                        signal_id=sig_id,
                        symbol=row["symbol"],
                        predicted_score=float(row["predicted_score"]),
                        confidence=float(row["confidence"]),
                        horizon_minutes=int(row["horizon_minutes"]),
                        ts_utc=datetime.fromisoformat(row["ts_utc"]),
                    )
                except Exception:
                    self._audit.append("signal_parse_error", {"signal_id": sig_id})
                    self._seen_preds.add(sig_id)
                    continue
                self._seen_preds.add(sig_id)
                self._pending_preds[sig_id] = (signal, now_monotonic)

    def _scan_verdicts(self) -> None:
        if not self._verdicts_dir.exists():
            return
        for path in sorted(self._verdicts_dir.glob("*.jsonl")):
            try:
                text = path.read_text()
            except Exception:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    verdict = GovernorVerdict.model_validate(payload)
                except Exception:
                    self._audit.append("verdict_parse_error", {"raw": line[:200]})
                    continue
                self._seen_verdicts[verdict.signal_id] = verdict
