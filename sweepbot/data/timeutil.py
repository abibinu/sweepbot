"""
sweepbot/data/timeutil.py
--------------------------
CRITICAL: Handles IST ↔ America/New_York DST-aware time mapping.

Key contract (from SDD §2):
- All strategy times are defined against IST assuming US Summer Time (EDT, UTC-4).
  e.g. NY open = 08:00 EDT = 17:30 IST
- During US Winter (EST, UTC-5) the same NY wall-clock event falls at 18:30 IST.
- The engine *always* anchors to the NY wall-clock (08:00 NY open) and converts
  to IST dynamically, so DST shifts are transparent to callers.

verify_timezone():
- Finds the highest-range M1 bar between 16:30–00:30 IST on known US NFP Fridays
  (first Friday of each month) and asserts it is at 18:00 IST (Summer) or
  19:00 IST (Winter). This validates that the data feed timestamps are correct.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from sweepbot.config import TZ_IST, TZ_NY, TZ_UTC

# ── Constants ────────────────────────────────────────────────────────────────
NY_OPEN_HOUR = 8   # 08:00 America/New_York
NY_OPEN_MIN = 0

# IST offset from UTC: +05:30 (always, no DST)
IST_OFFSET = dt.timedelta(hours=5, minutes=30)


# ── Core helpers ─────────────────────────────────────────────────────────────

def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert any tz-aware Timestamp to IST."""
    if ts.tzinfo is None:
        raise ValueError("Timestamp must be tz-aware before converting to IST")
    return ts.tz_convert(TZ_IST)


def to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert any tz-aware Timestamp to America/New_York."""
    if ts.tzinfo is None:
        raise ValueError("Timestamp must be tz-aware before converting to NY")
    return ts.tz_convert(TZ_NY)


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert any tz-aware Timestamp to UTC."""
    if ts.tzinfo is None:
        raise ValueError("Timestamp must be tz-aware")
    return ts.tz_convert(TZ_UTC)


def localize_utc(ts: pd.Timestamp | dt.datetime) -> pd.Timestamp:
    """Attach UTC timezone to a naive timestamp."""
    return pd.Timestamp(ts).tz_localize(TZ_UTC)


# ── DST-aware NY-open reference ───────────────────────────────────────────────

def ny_open_ist(date: dt.date) -> pd.Timestamp:
    """
    Return the IST timestamp of 08:00 AM New York time on *date*.
    Handles EDT/EST transitions automatically via ZoneInfo.

    Examples:
        Summer (EDT, UTC-4): 08:00 NY = 17:30 IST
        Winter (EST, UTC-5): 08:00 NY = 18:30 IST
    """
    naive_ny = dt.datetime(date.year, date.month, date.day, NY_OPEN_HOUR, NY_OPEN_MIN)
    aware_ny = naive_ny.replace(tzinfo=TZ_NY)  # ZoneInfo handles DST fold
    return pd.Timestamp(aware_ny).tz_convert(TZ_IST)


def ist_time_on_date(date: dt.date, hour: int, minute: int, ny_ref: bool = False) -> pd.Timestamp:
    """
    Build an IST timestamp for a specific hour:minute on *date*.

    If ny_ref=False (default):  interpret h:m as IST directly.
    If ny_ref=True:             interpret h:m as NY time and convert.
    """
    if ny_ref:
        naive_ny = dt.datetime(date.year, date.month, date.day, hour, minute)
        aware_ny = naive_ny.replace(tzinfo=TZ_NY)
        return pd.Timestamp(aware_ny).tz_convert(TZ_IST)
    else:
        naive_ist = dt.datetime(date.year, date.month, date.day, hour, minute)
        return pd.Timestamp(naive_ist, tzinfo=TZ_IST)


def is_us_summer(ts: pd.Timestamp) -> bool:
    """
    Return True when New York is on EDT (UTC-4, summer).
    Return False when on EST (UTC-5, winter).
    """
    ny_ts = to_ny(ts)
    # ZoneInfo stores the UTC offset in utcoffset()
    return ny_ts.utcoffset() == dt.timedelta(hours=-4)


@dataclass(frozen=True)
class SessionTimes:
    """
    Session timestamps localized to IST for a trading day.
    Dynamically shifts +1h during US Winter (EST) per SDD §2.
    """
    entry_start_ist: pd.Timestamp          # 17:30 IST summer / 18:30 IST winter
    entry_last_bar_ist: pd.Timestamp       # 19:15 IST summer / 20:15 IST winter
    late_entry_threshold_ist: pd.Timestamp # 19:00 IST summer / 20:00 IST winter
    time_exit_ist: pd.Timestamp            # 21:30 IST summer / 22:30 IST winter
    arm_time_ist: pd.Timestamp             # 17:25 IST summer / 18:25 IST winter


def get_session_times(date: dt.date) -> SessionTimes:
    """
    Compute operational window IST timestamps for *date*.
    Anchors to NY open (08:00 New York) and shifts +1h in winter per SDD §2.
    """
    open_ist = ny_open_ist(date)
    return SessionTimes(
        entry_start_ist=open_ist,
        entry_last_bar_ist=open_ist + pd.Timedelta(hours=1, minutes=45),
        late_entry_threshold_ist=open_ist + pd.Timedelta(hours=1, minutes=30),
        time_exit_ist=open_ist + pd.Timedelta(hours=4),
        arm_time_ist=open_ist - pd.Timedelta(minutes=5),
    )


# ── Trading day label ─────────────────────────────────────────────────────────

def trading_day_label(ts: pd.Timestamp) -> dt.date:
    """
    A trading day rolls at 02:30 IST.
    Label = (IST_timestamp + 7 hours).date()
    This ensures the NY session (17:30–02:30 IST) falls on one calendar date.

    Day D covers: (D-1) 02:30 IST → D 02:30 IST
    """
    if ts.tzinfo is None:
        raise ValueError("Timestamp must be tz-aware")
    ist = to_ist(ts)
    # Shift forward 7 h so midnight+30m becomes the next day boundary
    shifted = ist + dt.timedelta(hours=7)
    return shifted.date()


# ── Verify timezone integrity ─────────────────────────────────────────────────

def _first_fridays_of_months(start: dt.date, end: dt.date) -> list[dt.date]:
    """Generate all 'first Friday of the month' dates in [start, end]."""
    result = []
    d = start.replace(day=1)
    while d <= end:
        # Find first Friday of d.month/d.year
        day = d
        while day.weekday() != 4:  # 4 = Friday
            day += dt.timedelta(days=1)
        if start <= day <= end:
            result.append(day)
        # Move to next month
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)
    return result


def verify_timezone(m1_df: pd.DataFrame, verbose: bool = False) -> bool:
    """
    Validate that M1 data timestamps are correctly timezone-aware.

    SDD §5.1 mandate: On known US NFP Fridays (first Friday of each month),
    the highest-range M1 bar in the window 16:30–00:30 IST must fall at:
      - 18:00 IST during US Summer (EDT)
      - 19:00 IST during US Winter (EST)

    Parameters
    ----------
    m1_df : DataFrame with columns [open, high, low, close] and a
            tz-aware DatetimeIndex in *any* timezone.
    verbose : print per-NFP-date results

    Returns
    -------
    True if all NFP dates pass (or no NFP dates are found), False otherwise.
    """
    if m1_df.index.tzinfo is None:
        raise ValueError("m1_df must have a tz-aware DatetimeIndex")

    df_ist = m1_df.copy()
    df_ist.index = df_ist.index.tz_convert(TZ_IST)
    df_ist["range"] = df_ist["high"] - df_ist["low"]

    min_date = df_ist.index.min().date()
    max_date = df_ist.index.max().date()
    nfp_dates = _first_fridays_of_months(min_date, max_date)

    if not nfp_dates:
        if verbose:
            print("[verify_timezone] No NFP dates in data range — skipping check.")
        return True

    all_pass = True
    for nfp_date in nfp_dates:
        # Window: 16:30 IST on nfp_date to 00:30 IST on nfp_date+1
        win_start = pd.Timestamp(
            dt.datetime(nfp_date.year, nfp_date.month, nfp_date.day, 16, 30),
            tzinfo=TZ_IST,
        )
        win_end = pd.Timestamp(
            dt.datetime(nfp_date.year, nfp_date.month, nfp_date.day + 1, 0, 30),
            tzinfo=TZ_IST,
        )
        window = df_ist[(df_ist.index >= win_start) & (df_ist.index < win_end)]
        if window.empty:
            if verbose:
                print(f"[verify_timezone] {nfp_date}: no data in window — skip")
            continue

        peak_bar = window["range"].idxmax()
        peak_hour = peak_bar.hour
        peak_minute = peak_bar.minute

        # Expected: 18:00 IST (summer) or 19:00 IST (winter)
        summer = is_us_summer(peak_bar)
        expected_hour = 18 if summer else 19
        expected_minute = 0

        passed = (peak_hour == expected_hour) and (peak_minute == expected_minute)
        if verbose:
            status = "✓" if passed else "✗"
            season = "Summer (EDT)" if summer else "Winter (EST)"
            print(
                f"[verify_timezone] {nfp_date} [{season}]: "
                f"Peak bar at {peak_hour:02d}:{peak_minute:02d} IST — "
                f"expected {expected_hour:02d}:00 IST {status}"
            )
        if not passed:
            all_pass = False

    return all_pass
