"""
sweepbot/engine/fills.py
--------------------------
M1 fill simulator — walks M1 bars from entry minute to 21:29 IST.

SDD §2.8 fill rules:
  1. Walk M1 bars from the entry minute to 21:29 IST.
  2. If stop AND target are touched in the same M1 bar → count as STOP (worst case).
  3. At 21:30 IST → time exit at current bid.
  4. Net result in R = gross R - commission_in_R.

Returns a TradeResult with gross/net R, exit price, exit reason, and timestamp.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import pandas as pd

from sweepbot.config import CFG, PIP, TZ_IST
from sweepbot.data.timeutil import get_session_times
from sweepbot.risk import OrderSpec, commission_in_r
from sweepbot.strategy import Direction

log = logging.getLogger(__name__)


class ExitReason(Enum):
    TARGET = auto()
    STOP = auto()
    TIME = auto()    # 21:30 IST hard exit


@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    order: OrderSpec
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: ExitReason
    gross_r: float           # Signed R before commission
    commission_r: float      # Commission cost in R units
    net_r: float             # gross_r - commission_r
    # Metadata for reporting
    entry_time: pd.Timestamp
    entry_bar_ist: pd.Timestamp   # IST version of entry_bar_time
    is_early_entry: bool          # entry_bar_time < 19:00 IST

    @property
    def is_winner(self) -> bool:
        return self.net_r > 0

    def __repr__(self) -> str:
        return (
            f"TradeResult({self.order.direction.name} "
            f"{self.exit_reason.name} gross={self.gross_r:.3f}R "
            f"net={self.net_r:.3f}R @{self.exit_time})"
        )


def _time_exit_ist(trade_date: dt.date) -> pd.Timestamp:
    """Return the hard exit timestamp for *trade_date* (21:30 summer / 22:30 winter)."""
    return get_session_times(trade_date).time_exit_ist


def simulate_fill(
    order: OrderSpec,
    m1_df_utc: pd.DataFrame,
    trade_date: dt.date,
    cfg=CFG,
) -> Optional[TradeResult]:
    """
    Simulate a single trade by walking M1 bars.

    Parameters
    ----------
    order       : Fully constructed OrderSpec (entry, stop, target prices).
    m1_df_utc   : M1 DataFrame with UTC tz-aware DatetimeIndex.
    trade_date  : Calendar date of the trade (IST).
    cfg         : Config instance.

    Returns
    -------
    TradeResult, or None if no M1 bars exist from entry time onward.
    """
    direction = order.direction
    entry_price = order.entry_price
    stop_price = order.stop_price
    target_price = order.target_price

    sess = get_session_times(trade_date)
    time_exit = sess.time_exit_ist.tz_convert("UTC")
    entry_bar_time_utc = order.signal.entry_bar_time.tz_convert("UTC")
    entry_bar_ist = order.signal.entry_bar_time.tz_convert(TZ_IST)

    # Threshold for early/late entry tracking (SDD §7: 19:00 summer / 20:00 winter)
    late_threshold = sess.late_entry_threshold_ist.tz_convert("UTC")
    is_early_entry = entry_bar_time_utc < late_threshold

    # Walk M1 bars from entry bar onward (excluding 21:30 IST bar and beyond)
    mask = (m1_df_utc.index >= entry_bar_time_utc) & (m1_df_utc.index < time_exit)
    sim_bars = m1_df_utc.loc[mask]

    if sim_bars.empty:
        log.warning("No M1 bars available for fill simulation on %s", trade_date)
        return None

    entry_time = sim_bars.index[0]  # Actual first bar after entry bar time

    for ts, bar in sim_bars.iterrows():
        # For longs: check stop (low) and target (high)
        # For shorts: check stop (high) and target (low)
        if direction == Direction.LONG:
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
        else:
            stop_hit = bar["high"] >= stop_price
            target_hit = bar["low"] <= target_price

        # SDD §2.8: if both touched in same bar → count as stop
        if stop_hit and target_hit:
            exit_p = stop_price
            reason = ExitReason.STOP
        elif stop_hit:
            exit_p = stop_price
            reason = ExitReason.STOP
        elif target_hit:
            exit_p = target_price
            reason = ExitReason.TARGET
        else:
            continue  # Position still open

        gross_r = _calc_gross_r(direction, entry_price, exit_p, stop_price)
        comm_r = commission_in_r(order.commission_usd, order.risk_amount_usd)
        return TradeResult(
            order=order,
            exit_price=exit_p,
            exit_time=ts.tz_convert(TZ_IST),
            exit_reason=reason,
            gross_r=gross_r,
            commission_r=comm_r,
            net_r=gross_r - comm_r,
            entry_time=entry_time.tz_convert(TZ_IST),
            entry_bar_ist=entry_bar_ist,
            is_early_entry=is_early_entry,
        )

    # ── Time exit at 21:30 IST ────────────────────────────────────────────────
    last_bar = sim_bars.iloc[-1]
    # Use last bar close as market exit price
    if direction == Direction.LONG:
        # Sell at bid (close)
        exit_p = float(last_bar["close"])
    else:
        # Buy at ask = bid + spread
        exit_p = float(last_bar["close"]) + cfg.spread_pips * PIP

    gross_r = _calc_gross_r(direction, entry_price, exit_p, stop_price)
    comm_r = commission_in_r(order.commission_usd, order.risk_amount_usd)
    return TradeResult(
        order=order,
        exit_price=exit_p,
        exit_time=time_exit.tz_convert(TZ_IST),
        exit_reason=ExitReason.TIME,
        gross_r=gross_r,
        commission_r=comm_r,
        net_r=gross_r - comm_r,
        entry_time=entry_time.tz_convert(TZ_IST),
        entry_bar_ist=entry_bar_ist,
        is_early_entry=is_early_entry,
    )


def _calc_gross_r(
    direction: Direction,
    entry: float,
    exit_p: float,
    stop: float,
) -> float:
    """Gross R = signed P&L / risk per R."""
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if direction == Direction.LONG:
        return (exit_p - entry) / risk
    else:
        return (entry - exit_p) / risk
