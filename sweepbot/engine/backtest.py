"""
sweepbot/engine/backtest.py
-----------------------------
Main backtest loop. Orchestrates all modules per the SDD §6 flow:

  1. Group M1 data by trading day (02:30 IST rollover).
  2. Compute DayLevels (frozen before 17:30 IST).
  3. Build M15 bars for the session window (17:30 – 21:30 IST).
  4. Stream M15 bars through the strategy state machine.
  5. For each signal: build OrderSpec via risk module → simulate fill.
  6. Collect TradeResults → compute metrics → generate report.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import pandas as pd

from sweepbot.config import CFG, TZ_IST
from sweepbot.data.resample import resample_session
from sweepbot.data.timeutil import get_session_times, trading_day_label
from sweepbot.engine.fills import TradeResult, simulate_fill
from sweepbot.engine.metrics import MetricsResult, compute_metrics
from sweepbot.engine.reports import print_summary, print_trade_log, save_trade_csv
from sweepbot.levels import DayLevels, compute_levels
from sweepbot.risk import build_order
from sweepbot.strategy import StrategyStateMachine

log = logging.getLogger(__name__)


def _session_window(trade_date: dt.date, start_h: int, start_m: int,
                     end_h: int, end_m: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Build IST-timezone session start/end timestamps for a trade date."""
    start = pd.Timestamp(
        dt.datetime(trade_date.year, trade_date.month, trade_date.day,
                    start_h, start_m, tzinfo=TZ_IST)
    )
    end = pd.Timestamp(
        dt.datetime(trade_date.year, trade_date.month, trade_date.day,
                    end_h, end_m, tzinfo=TZ_IST)
    )
    return start, end


def run_backtest(
    m1_df_utc: pd.DataFrame,
    account_balance: float = 10_000.0,
    cfg=CFG,
    print_log: bool = True,
    save_csv: Optional[str] = None,
    title: str = "Backtest",
) -> tuple[list[TradeResult], MetricsResult]:
    """
    Run the full backtest on *m1_df_utc*.

    Parameters
    ----------
    m1_df_utc     : M1 OHLCV DataFrame, UTC tz-aware DatetimeIndex.
    account_balance: Starting account equity in USD.
    cfg           : Config instance.
    print_log     : Print per-trade log to console.
    save_csv      : If set, save trade log to this CSV path.
    title         : Title for the report.

    Returns
    -------
    (list[TradeResult], MetricsResult)
    """
    log.info("=== %s ===", title)
    log.info("Bars: %d  |  Dates: %s -> %s",
             len(m1_df_utc),
             m1_df_utc.index[0].date(),
             m1_df_utc.index[-1].date())

    # ── Group by trading day (02:30 IST rollover) ─────────────────────────────
    df_ist = m1_df_utc.copy()
    df_ist.index = df_ist.index.tz_convert(TZ_IST)
    df_ist["_trade_date"] = df_ist.index.map(lambda ts: trading_day_label(ts))

    trade_dates = sorted(df_ist["_trade_date"].unique())
    log.info("Found %d trading days", len(trade_dates))

    sm = StrategyStateMachine(cfg)
    all_results: list[TradeResult] = []
    skipped_days = 0

    for trade_date in trade_dates:
        # Skip weekends (Saturday=5, Sunday=6)
        if trade_date.weekday() >= 5:
            continue

        # ── Compute levels (uses all M1 data up to 17:30 IST) ─────────────────
        levels: DayLevels = compute_levels(m1_df_utc, trade_date)
        if not levels.is_valid:
            skipped_days += 1
            continue

        # ── Build session M15 bars (17:30–21:30 summer / 18:30–22:30 winter) ─────────
        sess = get_session_times(trade_date)
        m15_bars = resample_session(
            m1_df_utc,
            sess.entry_start_ist.tz_convert("UTC"),
            sess.time_exit_ist.tz_convert("UTC"),
        )

        if m15_bars.empty:
            log.debug("%s: No M15 bars in session", trade_date)
            skipped_days += 1
            continue

        # ── Run strategy state machine ────────────────────────────────────────
        seed_prev_close = None
        if getattr(cfg, "enable_bar_zero_seed", True):
            pre_start = sess.entry_start_ist.tz_convert("UTC") - pd.Timedelta(minutes=15)
            pre_bars = m1_df_utc.loc[
                (m1_df_utc.index >= pre_start) & (m1_df_utc.index < sess.entry_start_ist.tz_convert("UTC"))
            ]
            if not pre_bars.empty:
                seed_prev_close = float(pre_bars.iloc[-1]["close"])

        sm.on_day_start(trade_date, levels, seed_prev_close=seed_prev_close)

        for bar_idx, (ts, bar) in enumerate(m15_bars.iterrows()):
            sm.on_m15_close(bar_idx, bar)

        day_signals = sm.get_day_signals()
        if not day_signals:
            continue

        # ── Build orders and simulate fills ───────────────────────────────────
        day_results: list[TradeResult] = []
        for signal in day_signals:
            # Entry bar open = bid price at entry
            # Find the M1 bar at entry bar time
            entry_bar_utc = signal.entry_bar_time.tz_convert("UTC")
            entry_bars = m1_df_utc[m1_df_utc.index >= entry_bar_utc]
            if entry_bars.empty:
                log.warning("%s: No M1 bar for entry at %s", trade_date, entry_bar_utc)
                continue

            bid_at_entry = float(entry_bars.iloc[0]["open"])
            order = build_order(signal, bid_at_entry, account_balance, cfg)
            if order is None:
                log.info(
                    "%s: Signal %s skipped — risk_pips out of range",
                    trade_date, signal,
                )
                continue

            result = simulate_fill(order, m1_df_utc, trade_date, cfg)
            if result is not None:
                day_results.append(result)
                # Update running account balance (compound equity)
                account_balance += result.net_r * order.risk_amount_usd

        all_results.extend(day_results)

    log.info("Backtest complete: %d trades | %d days skipped",
             len(all_results), skipped_days)

    # ── Metrics + report ──────────────────────────────────────────────────────
    metrics = compute_metrics(all_results)

    from rich.console import Console
    c = Console()
    c.print()
    print_summary(metrics, title=title)

    if print_log and all_results:
        print_trade_log(all_results)

    if save_csv:
        save_trade_csv(all_results, save_csv)

    return all_results, metrics
