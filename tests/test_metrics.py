"""
tests/test_metrics.py
---------------------
Unit tests for engine/metrics.py (SDD §7 metrics calculation).
Verifies:
  - Standard metrics (trades, win rate, avg/total gross & net R)
  - Max drawdown calculation
  - Commission drag (gross R - net R)
  - Bootstrap 95% confidence interval
  - Late entry drag win rate
  - Zero/empty trades edge case
"""
import math
import datetime as dt
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.engine.fills import ExitReason, TradeResult
from sweepbot.engine.metrics import compute_metrics, _max_drawdown_r, _bootstrap_ci
from sweepbot.risk import OrderSpec
from sweepbot.strategy import Direction, Signal

TZ_IST = ZoneInfo("Asia/Kolkata")


def _make_dummy_result(
    gross_r: float,
    comm_r: float = 0.03,
    is_early: bool = True,
) -> TradeResult:
    sig = Signal(
        direction=Direction.LONG,
        level_name="PDL",
        level_price=1.0900,
        pattern="one_candle",
        extreme=1.0880,
        entry_bar_time=pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST),
    )
    order = OrderSpec(
        signal=sig,
        direction=Direction.LONG,
        entry_price=1.0900,
        stop_price=1.0880,
        target_price=1.0940,
        risk_pips=20.0,
        lot_size=0.25,
        account_balance=10_000.0,
        risk_amount_usd=50.0,
        commission_usd=1.50,
    )
    return TradeResult(
        order=order,
        exit_price=1.0940 if gross_r > 0 else 1.0880,
        exit_time=pd.Timestamp("2025-06-10 19:00:00", tz=TZ_IST),
        exit_reason=ExitReason.TARGET if gross_r > 0 else ExitReason.STOP,
        gross_r=gross_r,
        commission_r=comm_r,
        net_r=gross_r - comm_r,
        entry_time=pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST),
        entry_bar_ist=pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST),
        is_early_entry=is_early,
    )


class TestMaxDrawdown:
    def test_drawdown_computation(self):
        # Sequence: +2, -1, -1, +2, -2, -1
        # Cumulative: 2, 1, 0, 2, 0, -1
        # Peak: 2 -> drops to 0 (dd=2) -> recovers to 2 -> drops to -1 (dd=3)
        series = [2.0, -1.0, -1.0, 2.0, -2.0, -1.0]
        dd = _max_drawdown_r(series)
        assert dd == pytest.approx(3.0)

    def test_empty_drawdown(self):
        assert _max_drawdown_r([]) == 0.0


class TestBootstrapCI:
    def test_ci_bounds(self):
        values = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.05]
        lo, hi = _bootstrap_ci(values, n_bootstrap=1000, ci_level=0.95)
        assert lo < sum(values) / len(values) < hi

    def test_insufficient_samples(self):
        lo, hi = _bootstrap_ci([1.0])
        assert math.isnan(lo) and math.isnan(hi)


class TestComputeMetrics:
    def test_empty_results(self):
        m = compute_metrics([])
        assert m.trade_count == 0
        assert m.win_rate == 0.0
        assert m.avg_net_r == 0.0
        assert m.total_net_r == 0.0

    def test_standard_metrics_and_drag(self):
        # 4 trades: 2 wins (+2.0 gross, -0.05 comm = +1.95 net), 2 losses (-1.0 gross, -0.05 comm = -1.05 net)
        # Early: 2 wins
        # Late: 2 losses
        t1 = _make_dummy_result(gross_r=2.0, comm_r=0.05, is_early=True)
        t2 = _make_dummy_result(gross_r=2.0, comm_r=0.05, is_early=True)
        t3 = _make_dummy_result(gross_r=-1.0, comm_r=0.05, is_early=False)
        t4 = _make_dummy_result(gross_r=-1.0, comm_r=0.05, is_early=False)

        results = [t1, t2, t3, t4]
        m = compute_metrics(results)

        assert m.trade_count == 4
        assert m.win_count == 2
        assert m.loss_count == 2
        assert m.win_rate == pytest.approx(0.50)

        assert m.total_gross_r == pytest.approx(2.0)    # 2 + 2 - 1 - 1 = 2.0
        assert m.total_net_r == pytest.approx(1.80)     # 1.95 + 1.95 - 1.05 - 1.05 = 1.80
        assert m.avg_gross_r == pytest.approx(0.50)
        assert m.avg_net_r == pytest.approx(0.45)

        # Commission drag = total_gross - total_net = 2.0 - 1.80 = 0.20R
        assert m.commission_drag == pytest.approx(0.20)

        # Late entry drag (SDD §7):
        # early win rate = 2/2 = 1.0 (100%)
        # late win rate = 0/2 = 0.0 (0%)
        # drag = 1.0 - 0.0 = 1.0 (100%)
        assert m.early_trade_count == 2
        assert m.early_win_rate == pytest.approx(1.0)
        assert m.late_trade_count == 2
        assert m.late_win_rate == pytest.approx(0.0)
        assert m.late_entry_drag_win_rate == pytest.approx(1.0)

        # Equity curve: [1.95, 3.90, 2.85, 1.80]
        assert len(m.equity_curve) == 4
        assert m.equity_curve[-1] == pytest.approx(1.80)

        # Summary dict should format correctly
        s = m.summary_dict()
        assert s["Trades"] == 4
        assert s["Win Rate"] == "50.0%"
        assert s["Early Win Rate"] == "100.0%"
        assert s["Late Win Rate"] == "0.0%"
