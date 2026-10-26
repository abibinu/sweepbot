"""
sweepbot/data/resample.py
--------------------------
Resample M1 bars into M15 bars.

SDD §2.4 spec:
- M15 bars are left-labelled, left-closed.
- The 17:30 bar covers 17:30 to 17:44 IST.
- Open  = first M1 open
- High  = max of M1 highs
- Low   = min of M1 lows
- Close = last M1 close
- Volume = sum of M1 tick_volumes

The resampler is called with data already filtered to the relevant session.
"""
from __future__ import annotations

import pandas as pd

from sweepbot.config import CFG

_RESAMPLE_RULE = f"{CFG.signal_tf_minutes}min"

# Aggregation spec (standard OHLCV)
_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "tick_volume": "sum",
}


def resample_m1_to_m15(
    m1_df: pd.DataFrame,
    closed: str = "left",
    label: str = "left",
) -> pd.DataFrame:
    """
    Resample a UTC-indexed M1 DataFrame into M15 bars.

    Parameters
    ----------
    m1_df  : M1 DataFrame with tz-aware UTC DatetimeIndex and OHLCV columns.
    closed : Which side of the interval is closed ('left' = standard).
    label  : Which side is used as the bar label ('left' = SDD spec).

    Returns
    -------
    DataFrame of M15 bars (UTC DatetimeIndex), with columns:
        open, high, low, close, tick_volume
    Bars with fewer than 1 constituent M1 bar are dropped (NaN-free).
    """
    if m1_df.index.tzinfo is None:
        raise ValueError("m1_df must have a tz-aware DatetimeIndex")

    cols = {c: _AGG[c] for c in _AGG if c in m1_df.columns}
    m15 = (
        m1_df.resample(_RESAMPLE_RULE, closed=closed, label=label)
        .agg(cols)
        .dropna(subset=["open", "close"])
    )
    return m15


def resample_session(
    m1_df: pd.DataFrame,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Resample only the bars inside [session_start, session_end) into M15.

    Useful for building the signal window bars (17:30 to 21:30 IST).

    Parameters
    ----------
    session_start, session_end : tz-aware Timestamps (any tz, compared after UTC conversion).
    """
    mask = (m1_df.index >= session_start) & (m1_df.index < session_end)
    session_m1 = m1_df.loc[mask]
    if session_m1.empty:
        return pd.DataFrame(columns=list(_AGG.keys()))
    return resample_m1_to_m15(session_m1)
