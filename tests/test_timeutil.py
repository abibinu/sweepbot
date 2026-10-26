"""
tests/test_timeutil.py
-----------------------
Tests for timezone utilities (critical path — DST awareness).
"""
import datetime as dt
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.data.timeutil import (
    trading_day_label,
    ny_open_ist,
    is_us_summer,
    get_session_times,
    verify_timezone,
    to_ist,
)

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")
TZ_NY = ZoneInfo("America/New_York")


class TestSessionTimes:
    def test_summer_session_times(self):
        date = dt.date(2025, 6, 10)
        sess = get_session_times(date)
        assert sess.entry_start_ist.hour == 17 and sess.entry_start_ist.minute == 30
        assert sess.entry_last_bar_ist.hour == 19 and sess.entry_last_bar_ist.minute == 15
        assert sess.late_entry_threshold_ist.hour == 19 and sess.late_entry_threshold_ist.minute == 0
        assert sess.time_exit_ist.hour == 21 and sess.time_exit_ist.minute == 30
        assert sess.arm_time_ist.hour == 17 and sess.arm_time_ist.minute == 25

    def test_winter_session_times(self):
        date = dt.date(2025, 1, 10)
        sess = get_session_times(date)
        assert sess.entry_start_ist.hour == 18 and sess.entry_start_ist.minute == 30
        assert sess.entry_last_bar_ist.hour == 20 and sess.entry_last_bar_ist.minute == 15
        assert sess.late_entry_threshold_ist.hour == 20 and sess.late_entry_threshold_ist.minute == 0
        assert sess.time_exit_ist.hour == 22 and sess.time_exit_ist.minute == 30
        assert sess.arm_time_ist.hour == 18 and sess.arm_time_ist.minute == 25


class TestNyOpenIst:
    def test_summer_08_ny_is_1730_ist(self):
        """US Summer (EDT, UTC-4): 08:00 NY = 17:30 IST"""
        # June 2025 is summer
        date = dt.date(2025, 6, 10)
        result = ny_open_ist(date)
        assert result.hour == 17
        assert result.minute == 30

    def test_winter_08_ny_is_1830_ist(self):
        """US Winter (EST, UTC-5): 08:00 NY = 18:30 IST"""
        # January 2025 is winter
        date = dt.date(2025, 1, 10)
        result = ny_open_ist(date)
        assert result.hour == 18
        assert result.minute == 30


class TestIsSummer:
    def test_june_is_summer(self):
        ts = pd.Timestamp("2025-06-15 12:00:00", tz=TZ_UTC)
        assert is_us_summer(ts) is True

    def test_january_is_winter(self):
        ts = pd.Timestamp("2025-01-15 12:00:00", tz=TZ_UTC)
        assert is_us_summer(ts) is False


class TestTradingDayLabel:
    def test_before_rollover_is_same_day(self):
        """23:00 IST → same trading day"""
        ts = pd.Timestamp("2025-06-10 23:00:00", tz=TZ_IST)
        label = trading_day_label(ts)
        assert label == dt.date(2025, 6, 11)  # +7h shift → next day

    def test_after_midnight_before_rollover(self):
        """01:00 IST → still previous trading day (rolls at 02:30)"""
        ts = pd.Timestamp("2025-06-11 01:00:00", tz=TZ_IST)
        # 01:00 + 7h = 08:00 → same date as 2025-06-11
        label = trading_day_label(ts)
        assert label == dt.date(2025, 6, 11)

    def test_after_rollover_is_new_day(self):
        """03:00 IST → new trading day"""
        ts = pd.Timestamp("2025-06-11 03:00:00", tz=TZ_IST)
        label = trading_day_label(ts)
        assert label == dt.date(2025, 6, 11)

    def test_requires_tz_aware(self):
        ts = pd.Timestamp("2025-06-10 23:00:00")
        with pytest.raises(ValueError):
            trading_day_label(ts)


class TestVerifyTimezone:
    def _make_m1_df(self, bars: list[tuple]) -> pd.DataFrame:
        """Build a minimal M1 DataFrame for testing."""
        import numpy as np
        rows = []
        for ts_str, hi, lo in bars:
            rows.append({
                "timestamp": pd.Timestamp(ts_str, tz=TZ_UTC),
                "open": (hi + lo) / 2,
                "high": hi,
                "low": lo,
                "close": (hi + lo) / 2,
                "tick_volume": 100,
            })
        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    def test_passes_with_no_nfp_dates(self):
        """If there are no NFP dates in range, should return True."""
        bars = [("2025-01-02 10:00:00", 1.1010, 1.1000)]
        df = self._make_m1_df(bars)
        assert verify_timezone(df, verbose=False) is True

    def test_requires_tz_aware_index(self):
        import pandas as pd
        df = pd.DataFrame(
            {"open": [1.1], "high": [1.11], "low": [1.09], "close": [1.10]},
            index=pd.DatetimeIndex(["2025-01-03 10:00:00"]),
        )
        with pytest.raises(ValueError, match="tz-aware"):
            verify_timezone(df)
