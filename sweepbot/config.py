
"""
sweepbot/config.py
------------------
Single source of truth for all strategy parameters.
All times expressed in IST (Asia/Kolkata). The engine
handles the DST-aware shift to NY time internally via timeutil.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

# ── Canonical timezones ─────────────────────────────────────────────────────
TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_NY = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")

def load_local_config() -> dict:
    """Load config.json from project root if present."""
    import json
    from pathlib import Path
    cfg_file = Path("config.json")
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ── 1 pip in price terms ────────────────────────────────────────────────────
PIP = 0.0001  # Default (EURUSD, GBPUSD)


def pip_size(symbol: str) -> float:
    """Return pip size for a given forex symbol."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


# ── Calibrated parameters per asset ──────────────────────────────────────────
SYMBOL_PROFILES: dict[str, dict] = {
    "EURUSD": {
        "sweep_min_pips": 2.0,
        "min_stop_pips": 5.0,
        "max_stop_pips": 20.0,
        "excluded_levels": ("LonL",),
    },
    "GBPUSD": {
        "sweep_min_pips": 2.5,
        "min_stop_pips": 8.0,
        "max_stop_pips": 20.0,
        "excluded_levels": ("LonL", "LonH"),
    },
    "USDJPY": {
        "sweep_min_pips": 2.0,
        "min_stop_pips": 5.0,
        "max_stop_pips": 20.0,
        "excluded_levels": ("LonL", "LonH"),
    },
    "AUDUSD": {
        "sweep_min_pips": 1.5,
        "min_stop_pips": 6.0,
        "max_stop_pips": 15.0,
        "excluded_levels": (),
    },
    "USDCAD": {
        "sweep_min_pips": 2.0,
        "min_stop_pips": 7.0,
        "max_stop_pips": 20.0,
        "excluded_levels": (),
    },
}


@dataclass(frozen=True)
class Config:
    # ── Symbol ───────────────────────────────────────────────────────────────
    symbol: str = "EURUSD"

    @property
    def pip(self) -> float:
        return pip_size(self.symbol)


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
    sweep_min_pips: Optional[float] = None
    stop_buffer_pips: float = 1.0
    min_stop_pips: Optional[float] = None
    max_stop_pips: Optional[float] = None
    target_r: float = 2.0
    max_trades_per_day: int = 2

    # ── Strategy fine-tuning (SDD Section 8 refinement) ──────────────────────
    # Require rejection candle to close in trade direction (bullish for LONG, bearish for SHORT)
    require_rejection_color: bool = True
    # Favorable candle range ratio (0.0 to disable, 0.5 = close in favorable half)
    min_rejection_ratio: float = 0.0
    # Allowed patterns: ("one_candle", "two_candle") or ("two_candle",)
    allowed_patterns: tuple[str, ...] = ("one_candle", "two_candle")
    # Excluded levels to avoid low-expectancy continuation traps (e.g. LonL, LonH)
    excluded_levels: Optional[tuple[str, ...]] = None
    # Restrict entries strictly before US cash equity open (19:00 IST summer / 20:00 IST winter)
    early_entries_only: bool = True

    # ── Dynamic Breakeven ────────────────────────────────────────────────────
    enable_breakeven: bool = True
    breakeven_trigger_r: float = 1.3
    breakeven_buffer_pips: float = 0.2

    # ── Opening Bar (Bar 0) Seeding ──────────────────────────────────────────
    enable_bar_zero_seed: bool = True

    def __post_init__(self) -> None:
        profile = SYMBOL_PROFILES.get(self.symbol.upper(), SYMBOL_PROFILES["EURUSD"])
        if self.sweep_min_pips is None:
            object.__setattr__(self, "sweep_min_pips", profile["sweep_min_pips"])
        if self.min_stop_pips is None:
            object.__setattr__(self, "min_stop_pips", profile["min_stop_pips"])
        if self.max_stop_pips is None:
            object.__setattr__(self, "max_stop_pips", profile["max_stop_pips"])
        if self.excluded_levels is None:
            object.__setattr__(self, "excluded_levels", profile["excluded_levels"])

        local_cfg = load_local_config()
        if self.telegram_bot_token is None and local_cfg.get("telegram_bot_token"):
            object.__setattr__(self, "telegram_bot_token", local_cfg["telegram_bot_token"])
        if self.telegram_chat_id is None and local_cfg.get("telegram_chat_id"):
            object.__setattr__(self, "telegram_chat_id", str(local_cfg["telegram_chat_id"]))

    @classmethod
    def for_symbol(cls, symbol: str, **overrides) -> "Config":
        """Instantiate Config with calibrated defaults for the given symbol."""
        profile = SYMBOL_PROFILES.get(symbol.upper(), SYMBOL_PROFILES["EURUSD"]).copy()
        profile["symbol"] = symbol.upper()
        profile.update(overrides)
        return cls(**profile)

    # ── Risk ─────────────────────────────────────────────────────────────────
    # Use 0.25% until strategy is validated, then promote to 0.5%
    risk_per_trade_pct: float = 0.25

    # ── Cost model ───────────────────────────────────────────────────────────
    spread_pips: float = 0.2          # Half-spread applied to entry/exit
    commission_usd_per_lot: float = 3.00  # Per standard lot (round-trip)

    # ── Daily safety limits ───────────────────────────────────────────────────
    daily_loss_kill_pct: float = 1.5  # Hard kill at -1.5% daily loss
    max_open_positions: int = 3       # Max concurrent open positions in portfolio

    # ── Data cache directory (relative to project root) ──────────────────────
    data_dir: str = "data/cache"

    # ── Lot size constraints ──────────────────────────────────────────────────
    min_lot: float = 0.01
    lot_step: float = 0.01

    # ── MT5 connection ───────────────────────────────────────────────────────
    mt5_login: int | None = None       # Filled from env / CLI
    mt5_password: str | None = None
    mt5_server: str | None = None

    # ── Telegram notifications ───────────────────────────────────────────────
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None


# Module-level default instance (import and use directly)
CFG = Config()
