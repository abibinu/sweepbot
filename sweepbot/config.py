"""
sweepbot/config.py
------------------
Single source of truth for all strategy parameters.
All times expressed in IST (Asia/Kolkata). The engine
handles the DST-aware shift to NY time internally via timeutil.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

# ── Canonical timezones ─────────────────────────────────────────────────────
TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_NY = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")

# ── 1 pip in price terms ────────────────────────────────────────────────────
PIP = 0.0001  # EURUSD


@dataclass(frozen=True)
class Config:
    # ── Symbol ───────────────────────────────────────────────────────────────
    symbol: str = "EURUSD"

    # ── Session windows (IST hours, minutes) ─────────────────────────────────
    # Entry window: bars whose *start* time is in [17:30, 19:15] IST
    entry_window_start_h: int = 17
    entry_window_start_m: int = 30
    entry_window_last_bar_h: int = 19   # Last bar that may enter is 19:15
    entry_window_last_bar_m: int = 15

    # Hard time exit (IST)
    time_exit_h: int = 21
    time_exit_m: int = 30

    # Asia session for level computation (M1 bars from 04:30 to 12:29 IST)
    asia_start_h: int = 4
    asia_start_m: int = 30
    asia_end_h: int = 12
    asia_end_m: int = 29

    # London session for level computation (M1 bars from 12:30 to 17:29 IST)
    london_start_h: int = 12
    london_start_m: int = 30
    london_end_h: int = 17
    london_end_m: int = 29

    # Trading day rollover at 02:30 IST
    day_roll_h: int = 2
    day_roll_m: int = 30

    # Minimum bars required (skip day if fewer)
    min_asia_bars: int = 200
    min_london_bars: int = 150

    # ── Signal timeframe ─────────────────────────────────────────────────────
    signal_tf_minutes: int = 15  # M15 bars built from M1

    # ── Strategy parameters (in pips) ────────────────────────────────────────
    sweep_min_pips: float = 2.0
    stop_buffer_pips: float = 1.0
    min_stop_pips: float = 3.0
    max_stop_pips: float = 20.0
    target_r: float = 2.0
    max_trades_per_day: int = 2

    # ── Risk ─────────────────────────────────────────────────────────────────
    # Use 0.25% until strategy is validated, then promote to 0.5%
    risk_per_trade_pct: float = 0.25

    # ── Cost model ───────────────────────────────────────────────────────────
    spread_pips: float = 0.2          # Half-spread applied to entry/exit
    commission_usd_per_lot: float = 3.00  # Per standard lot (round-trip)

    # ── Daily safety limits ───────────────────────────────────────────────────
    daily_loss_kill_pct: float = 1.5  # Hard kill at -1.5% daily loss

    # ── Data cache directory (relative to project root) ──────────────────────
    data_dir: str = "data/cache"

    # ── Lot size constraints ──────────────────────────────────────────────────
    min_lot: float = 0.01
    lot_step: float = 0.01

    # ── MT5 connection ───────────────────────────────────────────────────────
    mt5_login: int | None = None       # Filled from env / CLI
    mt5_password: str | None = None
    mt5_server: str | None = None


# Module-level default instance (import and use directly)
CFG = Config()
