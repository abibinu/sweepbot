"""
sweepbot/engine/metrics.py
---------------------------
Compute performance metrics from a list of TradeResults.

Standard metrics (SDD §7):
  - trade_count, win_rate, avg_r, max_drawdown
  - bootstrap 95% CI on avg_r

New v0.2 metrics:
  - commission_drag: total raw R - total net R
  - late_entry_drag: win rate comparison (before 19:00 IST vs after)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from sweepbot.engine.fills import TradeResult


@dataclass
class MetricsResult:
    """All computed metrics for a backtest run."""
    # Core
    trade_count: int
    win_count: int
    win_rate: float             # fraction [0, 1]
    avg_gross_r: float
    avg_net_r: float
    total_gross_r: float
    total_net_r: float
    max_drawdown_r: float       # Maximum peak-to-trough drawdown in R

    # Commission drag
    commission_drag: float      # total_gross_r - total_net_r

    # Confidence interval on avg_net_r (bootstrap)
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95

    # Late entry drag
    early_trade_count: int = 0
    early_win_rate: float = float("nan")
    late_trade_count: int = 0
    late_win_rate: float = float("nan")
    late_entry_drag_win_rate: float = float("nan")  # early - late win rate

    # Per-day equity curve (for drawdown chart)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def loss_count(self) -> int:
        return self.trade_count - self.win_count

    def summary_dict(self) -> dict:
        return {
            "Trades": self.trade_count,
            "Wins": self.win_count,
            "Win Rate": f"{self.win_rate:.1%}",
            "Avg Gross R": f"{self.avg_gross_r:+.3f}",
            "Avg Net R": f"{self.avg_net_r:+.3f}",
            "Total Net R": f"{self.total_net_r:+.3f}",
            "Max Drawdown": f"{self.max_drawdown_r:.2f}R",
            "Commission Drag": f"{self.commission_drag:.3f}R",
            f"95% CI Net R": f"[{self.ci_lower:+.3f}, {self.ci_upper:+.3f}]",
            "Early Entries": self.early_trade_count,
            "Early Win Rate": f"{self.early_win_rate:.1%}" if not math.isnan(self.early_win_rate) else "N/A",
            "Late Entries": self.late_trade_count,
            "Late Win Rate": f"{self.late_win_rate:.1%}" if not math.isnan(self.late_win_rate) else "N/A",
            "Late Entry Drag": f"{self.late_entry_drag_win_rate:+.1%}" if not math.isnan(self.late_entry_drag_win_rate) else "N/A",
        }


def _max_drawdown_r(net_r_series: list[float]) -> float:
    """Compute max peak-to-trough drawdown in R units from cumulative R series."""
    if not net_r_series:
        return 0.0
    cumulative = np.cumsum(net_r_series)
    peak = cumulative[0]
    max_dd = 0.0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval on the mean.
    Returns (lower, upper) at the given CI level.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_metrics(results: list[TradeResult], ci_level: float = 0.95) -> MetricsResult:
    """
    Compute all metrics from a list of TradeResults.

    Parameters
    ----------
    results  : List of TradeResult objects (all trades in the backtest).
    ci_level : Bootstrap CI level (default 0.95).

    Returns
    -------
    MetricsResult with all computed metrics.
    """
    if not results:
        # Return zero-result metrics
        return MetricsResult(
            trade_count=0, win_count=0, win_rate=0.0,
            avg_gross_r=0.0, avg_net_r=0.0,
            total_gross_r=0.0, total_net_r=0.0,
            max_drawdown_r=0.0,
            commission_drag=0.0,
            ci_lower=float("nan"), ci_upper=float("nan"),
        )

    gross_rs = [r.gross_r for r in results]
    net_rs = [r.net_r for r in results]

    total_gross = sum(gross_rs)
    total_net = sum(net_rs)
    avg_gross = total_gross / len(gross_rs)
    avg_net = total_net / len(net_rs)
    wins = sum(1 for r in results if r.net_r > 0)
    win_rate = wins / len(results)
    max_dd = _max_drawdown_r(net_rs)
    comm_drag = total_gross - total_net
    ci_lo, ci_hi = _bootstrap_ci(net_rs, ci_level=ci_level)

    # Late entry drag (SDD §7 new v0.2 metric)
    early = [r for r in results if r.is_early_entry]
    late = [r for r in results if not r.is_early_entry]

    early_wr = sum(1 for r in early if r.net_r > 0) / len(early) if early else float("nan")
    late_wr = sum(1 for r in late if r.net_r > 0) / len(late) if late else float("nan")

    if not math.isnan(early_wr) and not math.isnan(late_wr):
        drag = early_wr - late_wr
    else:
        drag = float("nan")

    equity_curve = list(np.cumsum(net_rs))

    return MetricsResult(
        trade_count=len(results),
        win_count=wins,
        win_rate=win_rate,
        avg_gross_r=avg_gross,
        avg_net_r=avg_net,
        total_gross_r=total_gross,
        total_net_r=total_net,
        max_drawdown_r=max_dd,
        commission_drag=comm_drag,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=ci_level,
        early_trade_count=len(early),
        early_win_rate=early_wr,
        late_trade_count=len(late),
        late_win_rate=late_wr,
        late_entry_drag_win_rate=drag,
        equity_curve=equity_curve,
    )
