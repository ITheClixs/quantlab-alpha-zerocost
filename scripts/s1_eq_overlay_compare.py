"""Compare S1-EQ holdout result against the JS-trained stack applied to the
same engineered features (sanity overlay — spec §5.15, §6.4-9)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_sharpe(run_dir: Path) -> float:
    metrics_path = run_dir / "holdout_metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    return float(payload.get("holdout_sharpe") or payload.get("net_annualized_sharpe") or 0.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s1-eq-run", required=True)
    p.add_argument("--js-overlay-run", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    s1_eq_sharpe = _read_sharpe(Path(args.s1_eq_run))
    js_sharpe = _read_sharpe(Path(args.js_overlay_run))
    payload = {
        "s1_eq_run": str(args.s1_eq_run),
        "js_overlay_run": str(args.js_overlay_run),
        "s1_eq_sharpe": s1_eq_sharpe,
        "js_overlay_sharpe": js_sharpe,
        "s1_eq_beats_js": s1_eq_sharpe > js_sharpe,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
