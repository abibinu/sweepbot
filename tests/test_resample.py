"""
tests/test_resample.py
----------------------
Unit tests for data/resample.py (M1 to M15 resampling per SDD §2.4).
Verifies:
  - Left-labelled, left-closed M15 bars
  - Open = first M1 open, High = max, Low = min, Close = last M1 close
  - Volume = sum of M1 tick_volume
  - Resampling within session bounds
  - Tz-aware enforcement
"""
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.data.resample import resample_m1_to_m15, resample_session

TZ_UTC = ZoneInfo("UTC")


class TestResampleM1ToM15:
    def test_basic_ohlcv_aggregation(self):
        """15 M1 bars should compress into one M15 bar with correct OHLCV."""
        start = pd.Timestamp("2025-06-10 12:00:00", tz=TZ_UTC)
        rows = []
        for i in range(15):
            ts = start + pd.Timedelta(minutes=i)
            rows.append({
                "timestamp": ts,
                "open": 1.1000 + (0.0001 if i == 0 else 0),
                "high": 1.1050 if i == 5 else 1.1010,
                "low": 1.0950 if i == 10 else 1.0990,
                "close": 1.1025 if i == 14 else 1.1005,
                "tick_volume": 10,
            })
        df = pd.DataFrame(rows).set_index("timestamp")

        m15 = resample_m1_to_m15(df)

        assert len(m15) == 1
        bar = m15.iloc[0]
        # Left-labelled: index should be 12:00:00
        assert m15.index[0] == start
        # First open: 1.1001
        assert bar["open"] == pytest.approx(1.1001)
        # Max high: 1.1050
        assert bar["high"] == pytest.approx(1.1050)
        # Min low: 1.0950
        assert bar["low"] == pytest.approx(1.0950)
        # Last close: 1.1025
        assert bar["close"] == pytest.approx(1.1025)
        # Sum volume: 15 * 10 = 150
        assert bar["tick_volume"] == 150

    def test_left_labelled_left_closed_boundary(self):
        """The 12:00 bar covers [12:00, 12:14]. Bar at 12:15 belongs to next bar."""
        start = pd.Timestamp("2025-06-10 12:00:00", tz=TZ_UTC)
        rows = []
        for i in range(16):  # 0 to 15 (16 bars)
            rows.append({
                "timestamp": start + pd.Timedelta(minutes=i),
                "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1000,
                "tick_volume": 1,
            })
        df = pd.DataFrame(rows).set_index("timestamp")

        m15 = resample_m1_to_m15(df)
        assert len(m15) == 2
        assert m15.index[0] == pd.Timestamp("2025-06-10 12:00:00", tz=TZ_UTC)
        assert m15.index[1] == pd.Timestamp("2025-06-10 12:15:00", tz=TZ_UTC)
        assert m15.iloc[0]["tick_volume"] == 15
        assert m15.iloc[1]["tick_volume"] == 1

    def test_requires_tz_aware_index(self):
        df = pd.DataFrame(
            {"open": [1.1], "high": [1.11], "low": [1.09], "close": [1.10]},
            index=pd.DatetimeIndex(["2025-06-10 12:00:00"]),
        )
        with pytest.raises(ValueError, match="tz-aware"):
            resample_m1_to_m15(df)


class TestResampleSession:
    def test_session_slice(self):
        """resample_session should only include bars in [session_start, session_end)."""
        start = pd.Timestamp("2025-06-10 10:00:00", tz=TZ_UTC)
        rows = []
        for i in range(120):  # 2 hours
            rows.append({
                "timestamp": start + pd.Timedelta(minutes=i),
                "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1000,
                "tick_volume": 5,
            })
        df = pd.DataFrame(rows).set_index("timestamp")

        sess_start = pd.Timestamp("2025-06-10 10:30:00", tz=TZ_UTC)
        sess_end = pd.Timestamp("2025-06-10 11:30:00", tz=TZ_UTC)

        m15 = resample_session(df, sess_start, sess_end)
        # 10:30 to 11:30 is 60 minutes = 4 M15 bars (10:30, 10:45, 11:00, 11:15)
        assert len(m15) == 4
        assert m15.index[0] == sess_start
        assert m15.index[-1] == pd.Timestamp("2025-06-10 11:15:00", tz=TZ_UTC)

    def test_empty_session_returns_empty_df(self):
        start = pd.Timestamp("2025-06-10 10:00:00", tz=TZ_UTC)
        df = pd.DataFrame([{
            "timestamp": start,
            "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "tick_volume": 1
        }]).set_index("timestamp")

        sess_start = pd.Timestamp("2025-06-10 12:00:00", tz=TZ_UTC)
        sess_end = pd.Timestamp("2025-06-10 13:00:00", tz=TZ_UTC)
        res = resample_session(df, sess_start, sess_end)
        assert res.empty
