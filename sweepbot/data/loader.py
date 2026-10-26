"""
sweepbot/data/loader.py
------------------------
Downloads M1 EURUSD bars from MT5 via the MetaTrader5 Python API
and caches them as Parquet files.

Cache layout:
    data/cache/EURUSD_M1_<YYYY>.parquet   (one file per year)

The DataFrame returned always has:
    - DatetimeIndex in UTC (tz-aware)
    - Columns: open, high, low, close, tick_volume, spread, real_volume
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd

from sweepbot.config import CFG, TZ_UTC

log = logging.getLogger(__name__)

# MT5 timeframe constant (imported lazily to allow running without MT5 installed)
_MT5_M1 = None


def _get_mt5():
    """Lazy MT5 import — avoids hard failure on machines without MT5."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError as exc:
        raise ImportError(
            "MetaTrader5 package not installed. Run: pip install MetaTrader5"
        ) from exc


def _ensure_mt5_connected(cfg=CFG) -> None:
    mt5 = _get_mt5()
    kwargs = {}
    if cfg.mt5_login is not None:
        kwargs["login"] = cfg.mt5_login
    if cfg.mt5_password is not None:
        kwargs["password"] = cfg.mt5_password
    if cfg.mt5_server is not None:
        kwargs["server"] = cfg.mt5_server

    if not mt5.initialize(**kwargs):
        raise ConnectionError(
            f"MT5 initialize() failed: {mt5.last_error()}"
        )


def _cache_path(year: int, data_dir: str = CFG.data_dir, symbol: str = CFG.symbol) -> Path:
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{symbol}_M1_{year}.parquet"


def _download_year(year: int, cfg=CFG) -> pd.DataFrame:
    """
    Download one year of M1 data from MT5 in monthly chunks and return as DataFrame (UTC index).
    Chunking prevents 'Invalid params' errors caused by MT5 terminal buffer limits.
    """
    mt5 = _get_mt5()
    _ensure_mt5_connected(cfg)

    log.info("Downloading %s M1 for %d from MT5...", cfg.symbol, year)
    all_frames: list[pd.DataFrame] = []

    for month in range(1, 13):
        m_start = dt.datetime(year, month, 1, tzinfo=TZ_UTC)
        if month == 12:
            m_end = dt.datetime(year + 1, 1, 1, tzinfo=TZ_UTC)
        else:
            m_end = dt.datetime(year, month + 1, 1, tzinfo=TZ_UTC)

        # Convert to POSIX seconds for robust MT5 API compatibility
        t1 = int(m_start.timestamp())
        t2 = int(m_end.timestamp())

        rates = mt5.copy_rates_range(cfg.symbol, mt5.TIMEFRAME_M1, t1, t2)
        if rates is not None and len(rates) > 0:
            df_m = pd.DataFrame(rates)
            df_m["time"] = pd.to_datetime(df_m["time"], unit="s", utc=True)
            df_m = df_m.set_index("time").rename_axis("timestamp")
            cols = [c for c in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"] if c in df_m.columns]
            df_m = df_m[cols]
            all_frames.append(df_m)

    if not all_frames:
        raise ValueError(
            f"MT5 returned no data for {cfg.symbol} M1 {year}. "
            "Please ensure historical M1 bars are downloaded in your MT5 terminal (Tools -> History Center -> EURUSD M1 -> Download), "
            "or load from CSV using --csv <filepath>."
        )

    df = pd.concat(all_frames).drop_duplicates().sort_index()
    log.info("  Got %d bars for %d", len(df), year)
    return df


def load_m1(
    start_date: dt.date | str,
    end_date: dt.date | str,
    force_download: bool = False,
    cfg=CFG,
) -> pd.DataFrame:
    """
    Load M1 EURUSD data for the given date range.

    Downloads from MT5 if cache is missing or force_download=True.
    Returns a single DataFrame with tz-aware UTC DatetimeIndex.

    Parameters
    ----------
    start_date, end_date : inclusive date range (YYYY-MM-DD or date objects)
    force_download       : ignore existing cache
    cfg                  : Config instance
    """
    if isinstance(start_date, str):
        start_date = dt.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = dt.date.fromisoformat(end_date)

    years = range(start_date.year, end_date.year + 1)
    frames: list[pd.DataFrame] = []

    for year in years:
        cache = _cache_path(year, cfg.data_dir, cfg.symbol)
        if cache.exists() and not force_download:
            log.info("Loading cache: %s", cache)
            df_year = pd.read_parquet(cache)
            # Ensure tz-aware UTC index after loading from parquet
            if df_year.index.tzinfo is None:
                df_year.index = df_year.index.tz_localize(TZ_UTC)
        else:
            df_year = _download_year(year, cfg)
            df_year.to_parquet(cache)
            log.info("Saved cache: %s", cache)
        frames.append(df_year)

    if not frames:
        raise ValueError("No data loaded — check date range")

    df = pd.concat(frames).sort_index()

    # Slice to exact date range (UTC boundaries)
    utc_start = pd.Timestamp(start_date, tz=TZ_UTC)
    utc_end = pd.Timestamp(end_date, tz=TZ_UTC) + pd.Timedelta(days=1)
    df = df[(df.index >= utc_start) & (df.index < utc_end)]

    if df.empty:
        raise ValueError(f"No M1 bars in date range {start_date} to {end_date}")

    log.info("Loaded %d M1 bars (%s to %s)", len(df), start_date, end_date)
    return df


def load_m1_from_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load M1 data from a CSV file (MT5 History Center export format).

    Expected columns (flexible naming):
        Date, Time, Open, High, Low, Close, Volume
    OR combined: Timestamp, Open, High, Low, Close, Volume

    The timestamp is assumed to be in UTC. Returns UTC-aware DataFrame.
    """
    path = Path(filepath)
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle MT5's typical "date" + "time" split columns
    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str), utc=True
        )
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        # Assume first column is the datetime
        df["timestamp"] = pd.to_datetime(df.iloc[:, 0], utc=True)

    df = df.set_index("timestamp")
    # Standardize column names
    rename = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "tick_volume", "tickvol": "tick_volume",
        "vol": "real_volume",
    }
    df = df.rename(columns=rename)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}' in {filepath}")
    df = df.sort_index()
    return df
