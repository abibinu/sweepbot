"""
sweepbot/live/runner.py
------------------------
Live / paper trading runner loop.

SDD §9 Runner loop:
  - At 17:25 IST: load day's levels.
  - Every 15 min: call on_m15_close.
  - At 21:30 IST: hard kill — close all positions.
  - Kill switch: stop if daily loss hits -1.5%.
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from sweepbot.config import CFG, TZ_IST
from sweepbot.data.timeutil import get_session_times, trading_day_label
from sweepbot.levels import compute_levels
from sweepbot.live.broker_base import BrokerBase
from sweepbot.risk import build_order
from sweepbot.strategy import StrategyStateMachine

log = logging.getLogger(__name__)


def _now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ_IST)


def run_live(broker: BrokerBase, cfg=CFG, dry_run: bool = False) -> None:
    """
    Main live trading loop. Blocks until the session ends at 21:30 IST (22:30 winter).

    Parameters
    ----------
    broker  : Connected BrokerBase instance (MT5Broker or PaperBroker).
    cfg     : Config instance.
    dry_run : If True, detect signals but do NOT place orders (signal-only mode).
    """
    symbol = cfg.symbol
    sm = StrategyStateMachine(cfg)

    log.info("=== SweepBot Live Runner starting ===")
    log.info("Mode: %s | Symbol: %s", "DRY RUN" if dry_run else "LIVE", symbol)

    # ── Wait for arm time (5 min before NY open: 17:25 summer / 18:25 winter) ─
    now = _now_ist()
    sess = get_session_times(now.date())
    arm_time = sess.arm_time_ist
    time_exit = sess.time_exit_ist

    if now < arm_time:
        wait_sec = (arm_time - now).total_seconds()
        log.info("Waiting %.0f seconds until %s IST arm time...", wait_sec, arm_time.strftime("%H:%M"))
        time.sleep(max(0, wait_sec))

    # ── Load levels ───────────────────────────────────────────────────────────
    today = _now_ist().date()
    trade_date = trading_day_label(_now_ist())

    log.info("Loading levels for %s...", trade_date)
    m1_hist = broker.get_m1_bars(symbol, count=1000)   # ~16h of M1 bars
    if m1_hist.empty:
        log.error("Failed to fetch M1 data — aborting session.")
        return

    levels = compute_levels(m1_hist, trade_date)
    if not levels.is_valid:
        log.warning("Levels invalid for %s — no trading today.", trade_date)
        return

    log.info("Levels: %s", levels)
    start_balance = broker.get_account_balance()
    kill_threshold = start_balance * (1 - cfg.daily_loss_kill_pct / 100)
    log.info("Balance: %.2f | Kill at: %.2f", start_balance, kill_threshold)

    sm.on_day_start(trade_date, levels)

    # Track open tickets for this session
    session_tickets: list[str] = []
    trades_today = 0
    bar_idx = 0

    # ── Main 15-minute loop ───────────────────────────────────────────────────
    while True:
        now = _now_ist()

        # Hard exit at 21:30 IST
        if now >= time_exit:
            log.warning("[KILL] 21:30 IST — closing all positions.")
            for ticket in session_tickets:
                broker.close_position(ticket)
            log.info("Session ended.")
            break

        # Daily loss kill switch
        current_balance = broker.get_account_balance()
        if current_balance <= kill_threshold:
            log.error(
                "[KILL] Daily loss limit hit — balance %.2f <= %.2f. STOPPING.",
                current_balance, kill_threshold,
            )
            for ticket in session_tickets:
                broker.close_position(ticket)
            break

        # Wait for the next M15 bar close
        next_bar_close = _next_m15_close(now, cfg)
        wait_sec = (next_bar_close - now).total_seconds()
        log.info("Next M15 bar close at %s IST (%.0fs)", next_bar_close.strftime("%H:%M"), wait_sec)

        if wait_sec > 0:
            time.sleep(wait_sec)

        # Fetch recent M1 bars for level computation and bar data
        m1_recent = broker.get_m1_bars(symbol, count=200)
        if m1_recent.empty:
            log.warning("No M1 data at M15 bar close — skipping bar %d", bar_idx)
            bar_idx += 1
            continue

        # Build M15 bar from last 15 M1 bars
        from sweepbot.data.resample import resample_m1_to_m15
        m15 = resample_m1_to_m15(m1_recent)
        if m15.empty:
            bar_idx += 1
            continue

        current_m15 = m15.iloc[-1]
        log.info(
            "M15 bar %d: O=%.5f H=%.5f L=%.5f C=%.5f",
            bar_idx, current_m15["open"], current_m15["high"],
            current_m15["low"], current_m15["close"],
        )

        new_signals = sm.on_m15_close(bar_idx, current_m15)

        # ── Process signals ───────────────────────────────────────────────────
        for signal in new_signals:
            if trades_today >= cfg.max_trades_per_day:
                log.info("Max trades/day (%d) reached — ignoring signal: %s",
                         cfg.max_trades_per_day, signal)
                continue

            if dry_run:
                log.info("[DRY RUN] Signal detected: %s", signal)
                continue

            # Check for state recovery block
            if hasattr(broker, "has_recovered_positions") and broker.has_recovered_positions:
                log.warning("[SAFETY] Blocking entry — recovered position still open.")
                continue

            bid = broker.get_current_bid(symbol)
            balance = broker.get_account_balance()
            order = build_order(signal, bid, balance, cfg)

            if order is None:
                log.info("Signal %s skipped — risk_pips out of range", signal)
                continue

            ticket = broker.place_order(order)
            if ticket:
                session_tickets.append(ticket)
                trades_today += 1
                log.info("Order placed: %s | ticket=%s", order, ticket)

        bar_idx += 1

    log.info("=== Live session complete. Trades today: %d ===", trades_today)


def _next_m15_close(now: pd.Timestamp, cfg=CFG) -> pd.Timestamp:
    """
    Return the timestamp of the next M15 bar close from *now*.
    M15 bars close at :00, :15, :30, :45 of every hour.
    We add 1 second buffer to avoid hitting exactly on close.
    """
    minute = now.minute
    # Round up to next 15-minute boundary
    next_min = ((minute // cfg.signal_tf_minutes) + 1) * cfg.signal_tf_minutes
    if next_min >= 60:
        next_close = now.replace(minute=0, second=1, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_close = now.replace(minute=next_min, second=1, microsecond=0)
    return next_close
