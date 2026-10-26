"""
sweepbot/strategy.py
---------------------
Pure state machine implementing the NY-open sweep-and-reject strategy.

SDD §2.5 – Signal detection (evaluated on M15 bar close):

Support level L (long setup):
  1. Sweep:            bar.low <= L - sweep_min_pips  AND  prev_close > L
  2a. One-candle rej:  bar.close > L  → entry_bar = next bar
  2b. Two-candle rej:  bar.close <= L AND next_bar.close > L
                       → extreme = min(bar.low, next_bar.low), entry = bar after next

Resistance level L (short setup): exact mirror.

Rules:
  - A level is consumed by its first complete pattern of the day.
  - Entry bar must start at 19:15 IST or earlier.
  - At most max_trades_per_day signals are collected.
  - Signals are ordered by entry time; first 2 are taken.

This module is a *pure state machine* — no I/O, no MT5, no fills.
It processes M15 bars one at a time and emits Signal objects.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import pandas as pd

from sweepbot.config import CFG, PIP, TZ_IST
from sweepbot.data.timeutil import get_session_times
from sweepbot.levels import DayLevels

log = logging.getLogger(__name__)


# ── Signal dataclass ──────────────────────────────────────────────────────────

class Direction(Enum):
    LONG = auto()
    SHORT = auto()


@dataclass
class Signal:
    """A trade setup emitted by the strategy state machine."""
    direction: Direction
    level_name: str          # e.g. "PDL", "AsiaH"
    level_price: float
    pattern: str             # "one_candle" or "two_candle"
    extreme: float           # Low (long) or High (short) used for stop calc
    entry_bar_time: pd.Timestamp  # Start time of the bar we enter on (IST)
    # Populated after signal is confirmed (by fills/risk layer)
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    risk_pips: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"Signal({self.direction.name} {self.level_name}@{self.level_price:.5f} "
            f"[{self.pattern}] entry_bar={self.entry_bar_time})"
        )


# ── Level state tracker (per level, per day) ──────────────────────────────────

@dataclass
class _LevelState:
    """Tracks sweep-and-reject detection state for one level on one day."""
    name: str
    price: float
    level_type: str          # "support" or "resistance"
    consumed: bool = False   # Once a complete pattern fires, level is used up

    # Pending sweep state (waiting for rejection candle)
    swept: bool = False
    sweep_bar_idx: Optional[int] = None   # index of the bar that swept
    sweep_bar_low: Optional[float] = None
    sweep_bar_high: Optional[float] = None
    sweep_bar_close: Optional[float] = None

    # Two-candle: waiting for the second bar
    awaiting_two_candle: bool = False
    two_candle_bar_idx: Optional[int] = None
    two_candle_extreme: Optional[float] = None


# ── Main strategy state machine ───────────────────────────────────────────────

class StrategyStateMachine:
    """
    Processes M15 bars one at a time, emitting Signal objects.

    Usage (backtest):
        sm = StrategyStateMachine()
        sm.on_day_start(trade_date, levels)
        for bar in m15_bars_in_session:
            signals = sm.on_m15_close(bar_idx, bar)
        day_signals = sm.get_day_signals()

    Usage (live):
        Call on_day_start once at 17:25 IST, then on_m15_close every 15 min.
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self._levels: list[_LevelState] = []
        self._signals: list[Signal] = []
        self._bar_history: list[pd.Series] = []   # all bars seen today, in order
        self._trade_date: Optional[dt.date] = None

    # ── Day lifecycle ─────────────────────────────────────────────────────────

    def on_day_start(self, trade_date: dt.date, levels: DayLevels) -> None:
        """
        Reset state and initialise levels for a new trading day.
        Must be called before the first on_m15_close of the day.
        """
        self._trade_date = trade_date
        self._signals = []
        self._bar_history = []

        self._levels = []
        for name, (price, ltype) in levels.all_levels.items():
            if price != price:  # NaN check
                continue
            self._levels.append(_LevelState(name=name, price=price, level_type=ltype))

        log.debug(
            "%s: Initialised %d levels: %s",
            trade_date,
            len(self._levels),
            [f"{l.name}={l.price:.5f}" for l in self._levels],
        )

    def get_day_signals(self) -> list[Signal]:
        """Return collected signals (ordered by entry time, capped at max_trades_per_day)."""
        ordered = sorted(self._signals, key=lambda s: s.entry_bar_time)
        return ordered[: self.cfg.max_trades_per_day]

    # ── Core bar processing ───────────────────────────────────────────────────

    def on_m15_close(self, bar_idx: int, bar: pd.Series) -> list[Signal]:
        """
        Process one closed M15 bar. Returns any new Signal(s) detected.

        Parameters
        ----------
        bar_idx : Sequential index of the bar within the session (0-based).
        bar     : pd.Series with index [open, high, low, close] and a
                  tz-aware Timestamp name (the bar's start time in UTC).
        """
        self._bar_history.append(bar)

        # Convert bar start time to IST for window checks
        bar_ist = pd.Timestamp(bar.name).tz_convert(TZ_IST)

        new_signals: list[Signal] = []

        for lvl in self._levels:
            if lvl.consumed:
                continue

            # ── Step 1: Resolve any pending two-candle pattern ────────────────
            if lvl.awaiting_two_candle:
                sig = self._resolve_two_candle(lvl, bar_idx, bar, bar_ist)
                if sig is not None:
                    lvl.consumed = True
                    lvl.awaiting_two_candle = False
                    self._signals.append(sig)
                    new_signals.append(sig)
                    log.info("Two-candle signal: %s", sig)
                else:
                    # Second bar did NOT close beyond L → pattern cancelled
                    lvl.awaiting_two_candle = False
                    lvl.swept = False
                continue

            # ── Step 2: Check for a fresh sweep ──────────────────────────────
            if bar_idx == 0:
                prev_close = None  # No previous bar in session — use first bar's open
            else:
                prev_close = self._bar_history[-2]["close"]

            if prev_close is None:
                continue

            if not self._is_sweep(lvl, bar, prev_close):
                continue

            # ── Step 3: Sweep detected — classify rejection on the SAME bar ───
            # SDD §2.5: bar.close > L (support) → one-candle, entry = next bar
            #           bar.close <= L (support) → two-candle pending
            lvl.swept = True
            lvl.sweep_bar_idx = bar_idx
            lvl.sweep_bar_low = bar["low"]
            lvl.sweep_bar_high = bar["high"]
            lvl.sweep_bar_close = bar["close"]

            # The entry bar is always the *next* bar (bar_ist + 15 min)
            next_bar_ist = self._next_bar_ist(bar_ist)

            if lvl.level_type == "support":
                if bar["close"] > lvl.price:
                    # One-candle rejection
                    if self._entry_bar_in_window(next_bar_ist):
                        sig = Signal(
                            direction=Direction.LONG,
                            level_name=lvl.name,
                            level_price=lvl.price,
                            pattern="one_candle",
                            extreme=lvl.sweep_bar_low,
                            entry_bar_time=next_bar_ist,
                        )
                        lvl.consumed = True
                        lvl.swept = False
                        self._signals.append(sig)
                        new_signals.append(sig)
                        log.info("One-candle signal: %s", sig)
                    else:
                        log.debug("%s: One-candle LONG too late — entry bar %s",
                                  lvl.name, next_bar_ist)
                        lvl.swept = False
                else:
                    # Two-candle: close <= L, need next bar to close > L
                    lvl.awaiting_two_candle = True
                    lvl.two_candle_bar_idx = bar_idx
                    lvl.two_candle_extreme = lvl.sweep_bar_low
            else:  # resistance
                if bar["close"] < lvl.price:
                    # One-candle rejection
                    if self._entry_bar_in_window(next_bar_ist):
                        sig = Signal(
                            direction=Direction.SHORT,
                            level_name=lvl.name,
                            level_price=lvl.price,
                            pattern="one_candle",
                            extreme=lvl.sweep_bar_high,
                            entry_bar_time=next_bar_ist,
                        )
                        lvl.consumed = True
                        lvl.swept = False
                        self._signals.append(sig)
                        new_signals.append(sig)
                        log.info("One-candle signal: %s", sig)
                    else:
                        log.debug("%s: One-candle SHORT too late — entry bar %s",
                                  lvl.name, next_bar_ist)
                        lvl.swept = False
                else:
                    # Two-candle: close >= L
                    lvl.awaiting_two_candle = True
                    lvl.two_candle_bar_idx = bar_idx
                    lvl.two_candle_extreme = lvl.sweep_bar_high

        return new_signals

    # ── Sweep detection ───────────────────────────────────────────────────────

    def _is_sweep(self, lvl: _LevelState, bar: pd.Series, prev_close: float) -> bool:
        """SDD §2.5 sweep condition."""
        pip = self.cfg.sweep_min_pips * PIP
        if lvl.level_type == "support":
            # bar.low <= L - sweep_min_pips  AND  prev_close > L
            return bar["low"] <= (lvl.price - pip) and prev_close > lvl.price
        else:  # resistance
            # bar.high >= L + sweep_min_pips  AND  prev_close < L
            return bar["high"] >= (lvl.price + pip) and prev_close < lvl.price

    # ── Rejection detection ───────────────────────────────────────────────────

    def _entry_bar_in_window(self, entry_bar_ist: pd.Timestamp) -> bool:
        """Check if the entry bar start time is within the allowed window."""
        date = entry_bar_ist.date()
        sess = get_session_times(date)
        return sess.entry_start_ist <= entry_bar_ist <= sess.entry_last_bar_ist

    def _next_bar_ist(self, bar_ist: pd.Timestamp) -> pd.Timestamp:
        """Return the expected start time of the next M15 bar."""
        return bar_ist + pd.Timedelta(minutes=self.cfg.signal_tf_minutes)

    def _check_rejection(
        self,
        lvl: _LevelState,
        bar_idx: int,
        bar: pd.Series,
        bar_ist: pd.Timestamp,
    ) -> Signal | str | None:
        """
        Check bar following a sweep for one-candle or two-candle rejection.

        Returns:
          Signal         → one-candle rejection confirmed
          "two_candle_pending" → two-candle pattern needs next bar
          None           → no valid rejection
        """
        # This bar is the rejection candidate (bar after the sweep bar)
        # The entry bar would be the NEXT bar after this one
        next_bar_ist = self._next_bar_ist(bar_ist)

        if lvl.level_type == "support":
            # One-candle: rejection bar closes ABOVE L → entry is next bar
            if bar["close"] > lvl.price:
                if self._entry_bar_in_window(next_bar_ist):
                    return Signal(
                        direction=Direction.LONG,
                        level_name=lvl.name,
                        level_price=lvl.price,
                        pattern="one_candle",
                        extreme=lvl.sweep_bar_low,
                        entry_bar_time=next_bar_ist,
                    )
                else:
                    log.debug(
                        "%s: One-candle LONG signal too late — entry bar %s",
                        lvl.name, next_bar_ist,
                    )
                    return None
            # Two-candle: rejection bar closes BELOW or AT L → need next bar
            elif bar["close"] <= lvl.price:
                return "two_candle_pending"

        else:  # resistance
            # One-candle: rejection bar closes BELOW L → entry is next bar
            if bar["close"] < lvl.price:
                if self._entry_bar_in_window(next_bar_ist):
                    return Signal(
                        direction=Direction.SHORT,
                        level_name=lvl.name,
                        level_price=lvl.price,
                        pattern="one_candle",
                        extreme=lvl.sweep_bar_high,
                        entry_bar_time=next_bar_ist,
                    )
                else:
                    log.debug(
                        "%s: One-candle SHORT signal too late — entry bar %s",
                        lvl.name, next_bar_ist,
                    )
                    return None
            # Two-candle: rejection bar closes ABOVE or AT L → need next bar
            elif bar["close"] >= lvl.price:
                return "two_candle_pending"

        return None

    def _resolve_two_candle(
        self,
        lvl: _LevelState,
        bar_idx: int,
        bar: pd.Series,
        bar_ist: pd.Timestamp,
    ) -> Signal | None:
        """
        Resolve a pending two-candle pattern.

        The entry bar would be the NEXT bar after this one.
        Extreme = min/max of the sweep bar extreme + this bar's extreme.
        """
        next_bar_ist = self._next_bar_ist(bar_ist)

        if lvl.level_type == "support":
            # Update extreme: min of sweep bar low AND this bar's low
            combined_extreme = min(lvl.two_candle_extreme, bar["low"])
            # Second bar must close ABOVE L
            if bar["close"] > lvl.price:
                if self._entry_bar_in_window(next_bar_ist):
                    return Signal(
                        direction=Direction.LONG,
                        level_name=lvl.name,
                        level_price=lvl.price,
                        pattern="two_candle",
                        extreme=combined_extreme,
                        entry_bar_time=next_bar_ist,
                    )
                else:
                    log.debug(
                        "%s: Two-candle LONG too late — entry bar %s",
                        lvl.name, next_bar_ist,
                    )
        else:  # resistance
            combined_extreme = max(lvl.two_candle_extreme, bar["high"])
            # Second bar must close BELOW L
            if bar["close"] < lvl.price:
                if self._entry_bar_in_window(next_bar_ist):
                    return Signal(
                        direction=Direction.SHORT,
                        level_name=lvl.name,
                        level_price=lvl.price,
                        pattern="two_candle",
                        extreme=combined_extreme,
                        entry_bar_time=next_bar_ist,
                    )
                else:
                    log.debug(
                        "%s: Two-candle SHORT too late — entry bar %s",
                        lvl.name, next_bar_ist,
                    )
        return None

