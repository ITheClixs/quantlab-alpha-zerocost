from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReconReport:
    cycles: int
    n_rebalances: int
    funding_pnl: float
    basis_samples: list[float]
    equity_start: float
    equity_end: float

    def basis_mean_pct(self) -> float:
        return (sum(self.basis_samples) / len(self.basis_samples) * 100.0) if self.basis_samples else 0.0

    def basis_max_abs_pct(self) -> float:
        return (max(abs(b) for b in self.basis_samples) * 100.0) if self.basis_samples else 0.0

    def render(self) -> str:
        eq_delta = self.equity_end - self.equity_start
        return "\n".join([
            "# Funding-Carry Paper Sim — Live-vs-Model Reconciliation",
            "",
            "**Observation-only.** Strategy verdict: **DO_NOT_ADVANCE**. Not validation, "
            "not a step toward live (CLAUDE.md §7, §11).",
            "",
            f"- cycles: {self.cycles}  |  rebalances: {self.n_rebalances}",
            f"- funding P&L collected: {self.funding_pnl:.2f} USD",
            f"- equity: {self.equity_start:.2f} -> {self.equity_end:.2f} "
            f"(delta {eq_delta:+.2f})",
            f"- live basis mean: {self.basis_mean_pct():.4f}%  |  "
            f"max |basis|: {self.basis_max_abs_pct():.4f}%  "
            f"(backtest daily-close model: ~0% mean, <0.1% p95)",
            "",
            "Funding/basis are REAL (public mainnet); fills are simulated (FillModel). "
            "Compare live funding/basis above against the backtest cost-and-tail models in "
            "`reports/signal_research/funding_carry_v1/funding_carry_realism_results.md`.",
        ])
