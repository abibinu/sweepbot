"""
tests/test_backtest.py
----------------------
End-to-end integration tests for engine/backtest.py.
Verifies the complete backtest pipeline:
  - Multi-day M1 data processing
  - Level calculation and session windowing
  - Signal generation and risk sizing
  - Fill simulation
  - Metrics computation and CSV export
"""
import datetime as dt
import tempfile
from pathlib import Path
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.config import CFG, PIP
from sweepbot.engine.backtest import run_backtest

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")


def _generate_synthetic_day(date_str: str) -> list[dict]:
    """
    Generate synthetic M1 bars for one trading day:
    - Previous day (D-1): bars on 2025-06-09 to establish PDH/PDL
    - Asia: 04:30 to 12:29 IST (240 bars) with high=1.1050, low=1.0950
    - London: 12:30 to 17:29 IST (200 bars) with high=1.1080, low=1.0920
    - NY Session: 17:30 to 21:30 IST (240 bars)
        - 17:30-17:44 (bar 0): setup bar, close=1.0930
        - 17:45-17:59 (bar 1): sweeps LonL (1.0920) with low=1.0910, closes at 1.0925
        - 18:00-18:14 (bar 2): entry bar at open=1.0925
        - 18:15: price reaches 1.0965 (hitting 2R target at ~1.0958)
    """
    rows = []
    base_ts = pd.Timestamp(f"{date_str} 00:00:00", tz=TZ_IST)
    prev_date = base_ts - pd.Timedelta(days=1)

    # 1. Previous day bars (2025-06-09 12:00 to 20:00 IST) -> PDH=1.1100, PDL=1.0900
    for m in range(720, 1200, 10):
        ts = prev_date + pd.Timedelta(minutes=m)
        hi = 1.1100 if m == 800 else 1.1050
        lo = 1.0900 if m == 900 else 1.0950
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.1000, "high": hi, "low": lo, "close": 1.1000,
            "tick_volume": 20,
        })

    # 2. Asia session: 04:30 to 12:29 (240 bars, every 2 minutes)
    for m in range(270, 750, 2):
        ts = base_ts + pd.Timedelta(minutes=m)
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.1000, "high": 1.1050, "low": 1.0950, "close": 1.1000,
            "tick_volume": 30,
        })

    # 3. London session: 12:30 to 17:29 (200 bars, every 1.5 minutes)
    for m in range(750, 1050):
        if m % 3 == 0:  # ~100 bars? Let's generate every minute: 300 bars
            pass
    for m in range(750, 1050):
        ts = base_ts + pd.Timedelta(minutes=m)
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.1000, "high": 1.1080, "low": 1.0920, "close": 1.1000,
            "tick_volume": 40,
        })

    # 4. NY session: 17:30 to 21:30 (every minute)
    # 17:30 - 17:44 (bar 0): dummy bar close above LonL (1.0920)
    for m in range(1050, 1065):
        ts = base_ts + pd.Timedelta(minutes=m)
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.0930, "high": 1.0935, "low": 1.0925, "close": 1.0930,
            "tick_volume": 50,
        })

    # 17:45 - 17:59 (bar 1): sweeps LonL (1.0920) -> low drops to 1.0910, closes at 1.0925
    for m in range(1065, 1080):
        ts = base_ts + pd.Timedelta(minutes=m)
        lo = 1.0910 if m == 1070 else 1.0925
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.0930, "high": 1.0935, "low": lo, "close": 1.0925,
            "tick_volume": 50,
        })

    # 18:00 - 18:14 (bar 2): entry bar (entry at open = 1.0925)
    for m in range(1080, 1095):
        ts = base_ts + pd.Timedelta(minutes=m)
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.0925, "high": 1.0935, "low": 1.0920, "close": 1.0930,
            "tick_volume": 50,
        })

    # 18:15+: hits target (risk ~16.2 pips -> target ~1.0958)
    for m in range(1095, 1290):
        ts = base_ts + pd.Timedelta(minutes=m)
        hi = 1.0970 if m == 1100 else 1.0935
        rows.append({
            "timestamp": ts.tz_convert(TZ_UTC),
            "open": 1.0930, "high": hi, "low": 1.0920, "close": 1.0930,
            "tick_volume": 50,
        })

    return rows


class TestRunBacktest:
    def test_full_backtest_pipeline_and_csv_export(self):
        # Generate data for Tuesday 2025-06-10
        rows = _generate_synthetic_day("2025-06-10")
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "trades.csv"

            results, metrics = run_backtest(
                df,
                account_balance=10_000.0,
                cfg=CFG,
                print_log=False,
                save_csv=str(out_csv),
                title="Integration Test Backtest",
            )

            # We should have executed at least 1 trade
            assert len(results) >= 1
            assert metrics.trade_count >= 1
            assert metrics.win_count >= 1
            assert metrics.win_rate > 0

            # Verify CSV export was generated and has rows
            assert out_csv.exists()
            lines = out_csv.read_text().strip().split("\n")
            assert len(lines) >= 2  # header + at least 1 trade row
