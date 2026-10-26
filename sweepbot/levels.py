"""
sweepbot/levels.py
-------------------
Computes PDH/PDL, AsiaH/AsiaL, LonH/LonL for a given trading day.

SDD §2.3:
  Level    Definition                                 Type
  -------  -----------------------------------------  ----------
  PDH/PDL  high/low of the previous trading day       resistance/support
  AsiaH/L  high/low of M1 bars 04:30–12:29 IST        resistance/support
  LonH/L   high/low of M1 bars 12:30–17:29 IST        resistance/support

All levels are computed *once* and frozen before 17:30 IST.

Skip the whole day if:
  - Asian range has < 200 M1 bars
  - London range has < 150 M1 bars
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from sweepbot.config import CFG, TZ_IST
from sweepbot.data.timeutil import trading_day_label

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DayLevels:
    """All key price levels for a single trading day."""
    date: dt.date

    # Previous day
    pdh: float
    pdl: float

    # Asia session
    asia_h: float
    asia_l: float

    # London session
    lon_h: float
    lon_l: float

    # Metadata
    asia_bar_count: int
    lon_bar_count: int

    @property
    def is_valid(self) -> bool:
        """True if the day has sufficient data to trade."""
        return (
            self.asia_bar_count >= CFG.min_asia_bars
            and self.lon_bar_count >= CFG.min_london_bars
        )

    @property
    def all_levels(self) -> dict[str, tuple[float, str]]:
        """Return all levels as {name: (price, type)} where type is 'support'/'resistance'."""
        return {
            "PDH": (self.pdh, "resistance"),
            "PDL": (self.pdl, "support"),
            "AsiaH": (self.asia_h, "resistance"),
            "AsiaL": (self.asia_l, "support"),
            "LonH": (self.lon_h, "resistance"),
            "LonL": (self.lon_l, "support"),
        }

    def __repr__(self) -> str:
        v = "VALID" if self.is_valid else "SKIP"
        return (
            f"DayLevels({self.date} [{v}] "
            f"PDH={self.pdh:.5f} PDL={self.pdl:.5f} "
            f"AsiaH={self.asia_h:.5f} AsiaL={self.asia_l:.5f} "
            f"LonH={self.lon_h:.5f} LonL={self.lon_l:.5f})"
        )


def _ist_window(
    day_m1_ist: pd.DataFrame,
    start_h: int, start_m: int,
    end_h: int, end_m: int,
    date: dt.date,
) -> pd.DataFrame:
    """Filter IST-indexed M1 bars to an intra-day IST window [start, end]."""
    win_start = pd.Timestamp(
        dt.datetime(date.year, date.month, date.day, start_h, start_m, tzinfo=TZ_IST)
    )
    win_end = pd.Timestamp(
        dt.datetime(date.year, date.month, date.day, end_h, end_m, tzinfo=TZ_IST)
    )
    return day_m1_ist[(day_m1_ist.index >= win_start) & (day_m1_ist.index <= win_end)]


def compute_levels(
    m1_df_utc: pd.DataFrame,
    trade_date: dt.date,
) -> DayLevels:
    """
    Compute DayLevels for *trade_date*.

    Parameters
    ----------
    m1_df_utc : Full M1 DataFrame with UTC tz-aware DatetimeIndex.
                Should span at least D-1 02:30 IST → D 17:29 IST.
    trade_date : The date we want to trade (IST calendar date).

    Returns
    -------
    DayLevels (check .is_valid before using)
    """
    # Convert to IST for windowing
    df_ist = m1_df_utc.copy()
    df_ist.index = df_ist.index.tz_convert(TZ_IST)

    # ── Previous trading day: D-1 02:30 IST to D 02:29 IST ─────────────────
    prev_day_start = pd.Timestamp(
        dt.datetime(trade_date.year, trade_date.month, trade_date.day,
                    CFG.day_roll_h, CFG.day_roll_m, tzinfo=TZ_IST)
    ) - pd.Timedelta(days=1)
    prev_day_end = pd.Timestamp(
        dt.datetime(trade_date.year, trade_date.month, trade_date.day,
                    CFG.day_roll_h, CFG.day_roll_m, tzinfo=TZ_IST)
    ) - pd.Timedelta(minutes=1)

    prev_bars = df_ist[(df_ist.index >= prev_day_start) & (df_ist.index <= prev_day_end)]
    if prev_bars.empty:
        log.warning("%s: No previous day bars found", trade_date)
        pdh = float("nan")
        pdl = float("nan")
    else:
        pdh = float(prev_bars["high"].max())
        pdl = float(prev_bars["low"].min())

    # ── Asia session: 04:30–12:29 IST on trade_date ──────────────────────────
    asia_bars = _ist_window(
        df_ist,
        CFG.asia_start_h, CFG.asia_start_m,
        CFG.asia_end_h, CFG.asia_end_m,
        trade_date,
    )

    # ── London session: 12:30–17:29 IST on trade_date ────────────────────────
    lon_bars = _ist_window(
        df_ist,
        CFG.london_start_h, CFG.london_start_m,
        CFG.london_end_h, CFG.london_end_m,
        trade_date,
    )

    levels = DayLevels(
        date=trade_date,
        pdh=pdh,
        pdl=pdl,
        asia_h=float(asia_bars["high"].max()) if not asia_bars.empty else float("nan"),
        asia_l=float(asia_bars["low"].min()) if not asia_bars.empty else float("nan"),
        lon_h=float(lon_bars["high"].max()) if not lon_bars.empty else float("nan"),
        lon_l=float(lon_bars["low"].min()) if not lon_bars.empty else float("nan"),
        asia_bar_count=len(asia_bars),
        lon_bar_count=len(lon_bars),
    )

    if not levels.is_valid:
        log.info(
            "%s: SKIP — Asia bars=%d (min %d), London bars=%d (min %d)",
            trade_date,
            levels.asia_bar_count, CFG.min_asia_bars,
            levels.lon_bar_count, CFG.min_london_bars,
        )

    return levels


def compute_all_levels(
    m1_df_utc: pd.DataFrame,
) -> dict[dt.date, DayLevels]:
    """
    Compute DayLevels for every trading day present in *m1_df_utc*.

    Returns a dict keyed by trade_date (dt.date).
    """
    df_ist = m1_df_utc.copy()
    df_ist.index = df_ist.index.tz_convert(TZ_IST)

    # Discover all unique trading days
    trade_dates = sorted({trading_day_label(ts) for ts in df_ist.index})

    result: dict[dt.date, DayLevels] = {}
    for date in trade_dates:
        levels = compute_levels(m1_df_utc, date)
        result[date] = levels

    valid = sum(1 for l in result.values() if l.is_valid)
    log.info("Computed levels for %d days (%d valid, %d skipped)",
             len(result), valid, len(result) - valid)
    return result
