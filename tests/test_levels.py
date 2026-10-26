"""
tests/test_levels.py
--------------------
Unit tests for levels.py (PDH/PDL, AsiaH/AsiaL, LonH/LonL).
Verifies:
  - Window extraction in IST
  - Correct high/low calculation
  - Minimum bar count requirements (Asia >= 200, London >= 150)
  - Level types (support vs resistance)
"""
import datetime as dt
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.config import CFG
from sweepbot.levels import DayLevels, compute_levels, compute_all_levels

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")


def _make_bar(ts_ist: pd.Timestamp, high: float, low: float, open_: float = 1.1000, close: float = 1.1000) -> dict:
    return {
        "timestamp": ts_ist.tz_convert(TZ_UTC),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": 100,
    }


class TestComputeLevels:
    def test_session_high_low_and_bar_counts(self):
        """Build M1 bars for a day and verify computed levels and bar counts."""
        trade_date = dt.date(2025, 6, 10)
        rows = []

        # Previous day: D-1 02:30 IST to D 02:29 IST
        # Add bars in prev day
        prev_start = pd.Timestamp("2025-06-09 02:30:00", tz=TZ_IST)
        for i in range(10):
            ts = prev_start + pd.Timedelta(hours=i * 2)
            # High 1.1200, low 1.0800
            hi = 1.1200 if i == 3 else 1.1050
            lo = 1.0800 if i == 5 else 1.0950
            rows.append(_make_bar(ts, high=hi, low=lo))

        # Asia session: 04:30 to 12:29 IST (8 hours = 480 minutes)
        # Generate 200 bars in Asia
        asia_start = pd.Timestamp("2025-06-10 04:30:00", tz=TZ_IST)
        for i in range(205):
            ts = asia_start + pd.Timedelta(minutes=i * 2)
            hi = 1.1080 if i == 10 else 1.1020
            lo = 1.0920 if i == 20 else 1.0980
            rows.append(_make_bar(ts, high=hi, low=lo))

        # London session: 12:30 to 17:29 IST (5 hours = 300 minutes)
        # Generate 160 bars in London
        lon_start = pd.Timestamp("2025-06-10 12:30:00", tz=TZ_IST)
        for i in range(160):
            ts = lon_start + pd.Timedelta(minutes=i)
            hi = 1.1150 if i == 50 else 1.1010
            lo = 1.0880 if i == 60 else 1.0970
            rows.append(_make_bar(ts, high=hi, low=lo))

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        levels = compute_levels(df, trade_date)

        assert levels.date == trade_date
        assert levels.pdh == 1.1200
        assert levels.pdl == 1.0800
        assert levels.asia_h == 1.1080
        assert levels.asia_l == 1.0920
        assert levels.lon_h == 1.1150
        assert levels.lon_l == 1.0880
        assert levels.asia_bar_count == 205
        assert levels.lon_bar_count == 160
        assert levels.is_valid is True

        all_lvls = levels.all_levels
        assert all_lvls["PDH"] == (1.1200, "resistance")
        assert all_lvls["PDL"] == (1.0800, "support")
        assert all_lvls["AsiaH"] == (1.1080, "resistance")
        assert all_lvls["AsiaL"] == (1.0920, "support")
        assert all_lvls["LonH"] == (1.1150, "resistance")
        assert all_lvls["LonL"] == (1.0880, "support")

    def test_skip_when_asia_bars_fewer_than_200(self):
        """If Asian session has < 200 bars, is_valid must be False (SDD §2.3)."""
        trade_date = dt.date(2025, 6, 10)
        rows = []
        asia_start = pd.Timestamp("2025-06-10 04:30:00", tz=TZ_IST)
        # Only 199 bars in Asia
        for i in range(199):
            ts = asia_start + pd.Timedelta(minutes=i * 2)
            rows.append(_make_bar(ts, high=1.1050, low=1.0950))

        lon_start = pd.Timestamp("2025-06-10 12:30:00", tz=TZ_IST)
        for i in range(150):
            ts = lon_start + pd.Timedelta(minutes=i)
            rows.append(_make_bar(ts, high=1.1050, low=1.0950))

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        levels = compute_levels(df, trade_date)
        assert levels.asia_bar_count == 199
        assert levels.is_valid is False

    def test_skip_when_london_bars_fewer_than_150(self):
        """If London session has < 150 bars, is_valid must be False (SDD §2.3)."""
        trade_date = dt.date(2025, 6, 10)
        rows = []
        asia_start = pd.Timestamp("2025-06-10 04:30:00", tz=TZ_IST)
        for i in range(200):
            ts = asia_start + pd.Timedelta(minutes=i * 2)
            rows.append(_make_bar(ts, high=1.1050, low=1.0950))

        lon_start = pd.Timestamp("2025-06-10 12:30:00", tz=TZ_IST)
        # Only 149 bars in London
        for i in range(149):
            ts = lon_start + pd.Timedelta(minutes=i)
            rows.append(_make_bar(ts, high=1.1050, low=1.0950))

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        levels = compute_levels(df, trade_date)
        assert levels.lon_bar_count == 149
        assert levels.is_valid is False
