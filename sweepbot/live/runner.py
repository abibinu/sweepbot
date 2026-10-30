"""
sweepbot/live/runner.py
------------------------
Live / paper trading multi-pair portfolio runner loop.

SDD §9 Runner loop:
  - At 17:25 IST (18:25 winter): load day's levels for all active symbols.
  - Multi-pair portfolio risk management:
      * Max concurrent open positions across the portfolio (default 3).
      * Max 1 trade at a time per symbol, capped at max_trades_per_day.
      * Global daily loss kill switch: stops if portfolio equity drops by -1.5%.
  - Every 15 min: resample and call on_m15_close for each symbol.
  - Periodic polling: checks SL/TP and trails breakeven (+1.3R) between bars.
  - At 21:30 IST (22:30 winter): hard kill — close all positions across all symbols.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from sweepbot.config import CFG, Config, TZ_IST
from sweepbot.data.resample import resample_m1_to_m15
from sweepbot.data.timeutil import get_session_times, trading_day_label
from sweepbot.levels import DayLevels, compute_levels
from sweepbot.live.broker_base import BrokerBase
from sweepbot.risk import build_order
from sweepbot.strategy import StrategyStateMachine

log = logging.getLogger(__name__)
console = Console()


def _now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ_IST)


def _next_m15_close(now: pd.Timestamp, cfg=CFG) -> pd.Timestamp:
    """
    Return the timestamp of the next M15 bar close from *now*.
    M15 bars close at :00, :15, :30, :45 of every hour.
    We add 1 second buffer to avoid hitting exactly on close.
    """
    minute = now.minute
    next_min = ((minute // cfg.signal_tf_minutes) + 1) * cfg.signal_tf_minutes
    if next_min >= 60:
        next_close = now.replace(minute=0, second=1, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_close = now.replace(minute=next_min, second=1, microsecond=0)
    return next_close


def print_portfolio_dashboard(
    levels_map: dict[str, DayLevels],
    cfg_map: dict[str, Config],
    start_balance: float,
    kill_threshold: float,
    trade_date: dt.date,
    prev_close_map: Optional[dict[str, float]] = None,
) -> None:
    """Display an informative, formatted multi-pair status table."""
    table = Table(
        title=f"SweepBot Multi-Pair Portfolio Dashboard — {trade_date}",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Symbol", style="bold cyan", justify="center")
    table.add_column("PDH", justify="right")
    table.add_column("PDL", justify="right")
    table.add_column("Asia High", justify="right")
    table.add_column("Asia Low", justify="right")
    table.add_column("Prev Close", justify="right")
    table.add_column("Stop Range (pips)", justify="center")
    table.add_column("Status", justify="center")

    for sym, levels in levels_map.items():
        cfg_s = cfg_map[sym]
        status = "[green]ACTIVE[/green]" if levels.is_valid else "[red]INVALID[/red]"
        prev_close = prev_close_map.get(sym) if prev_close_map else getattr(levels, "prev_close", None)
        prev_close_str = f"{prev_close:.5f}" if prev_close is not None else "-"
        stop_range_str = f"{cfg_s.min_stop_pips:.1f} - {cfg_s.max_stop_pips:.1f}"
        table.add_row(
            sym,
            f"{levels.pdh:.5f}",
            f"{levels.pdl:.5f}",
            f"{levels.asia_h:.5f}",
            f"{levels.asia_l:.5f}",
            prev_close_str,
            stop_range_str,
            status,
        )

    console.print(table)
    console.print(
        f"[bold]Account Equity:[/bold] ${start_balance:,.2f}  |  "
        f"[bold red]Kill Switch Threshold (-1.5%):[/bold red] ${kill_threshold:,.2f}  |  "
        f"[bold]Max Concurrent Positions:[/bold] {CFG.max_open_positions}"
    )


def run_live(
    broker: BrokerBase,
    symbols: list[str] | str | None = None,
    cfg: Config = CFG,
    dry_run: bool = False,
    poll_interval_sec: float = 1.0,
    wait_for_arm: bool = True,
    max_bars: Optional[int] = None,
    skip_bar_wait: bool = False,
) -> None:
    """
    Main live/paper trading loop with multi-pair portfolio execution.
    Blocks until the session ends at 21:30 IST (22:30 winter) or kill switch fires.

    Parameters
    ----------
    broker            : Connected BrokerBase instance (MT5Broker or PaperBroker).
    symbols           : List of symbols or comma-separated string (e.g. "EURUSD,GBPUSD").
    cfg               : Base Config instance.
    dry_run           : If True, detect signals but do NOT place orders.
    poll_interval_sec : Sleep interval in seconds between bar polling/trailing checks.
    wait_for_arm      : If True, wait until arm time (17:25 IST summer / 18:25 winter).
    max_bars          : If specified, terminate after processing N M15 bar closes (for tests).
    """
    if symbols is None:
        symbol_list = [cfg.symbol]
    elif isinstance(symbols, str):
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        symbol_list = [s.strip().upper() for s in symbols]

    log.info("=== SweepBot Multi-Pair Portfolio Runner Starting ===")
    log.info("Mode: %s | Symbols: %s", "DRY RUN" if dry_run else "LIVE", ", ".join(symbol_list))

    # ── Wait for arm time (5 min before NY open: 17:25 summer / 18:25 winter) ─
    now = _now_ist()
    sess = get_session_times(now.date())
    arm_time = sess.arm_time_ist
    time_exit = sess.time_exit_ist

    if wait_for_arm and now < arm_time:
        wait_sec = (arm_time - now).total_seconds()
        log.info(
            "Waiting %.0f seconds until %s IST arm time...",
            wait_sec, arm_time.strftime("%H:%M"),
        )
        time.sleep(max(0, wait_sec))

    # ── Load levels & initialize state machines per symbol ────────────────────
    trade_date = trading_day_label(_now_ist())
    log.info("Loading levels for %s across %d symbols...", trade_date, len(symbol_list))

    cfg_map: dict[str, Config] = {}
    sm_map: dict[str, StrategyStateMachine] = {}
    levels_map: dict[str, DayLevels] = {}
    trades_today: dict[str, int] = {}

    for sym in symbol_list:
        cfg_s = Config.for_symbol(
            sym,
            early_entries_only=cfg.early_entries_only,
            risk_per_trade_pct=cfg.risk_per_trade_pct,
            daily_loss_kill_pct=cfg.daily_loss_kill_pct,
            max_open_positions=cfg.max_open_positions,
        )
        cfg_map[sym] = cfg_s
        trades_today[sym] = 0

        # Fetch 2880 M1 bars (~48 hours) to ensure Friday session coverage on Mondays
        m1_hist = broker.get_m1_bars(sym, count=2880)
        if m1_hist.empty:
            log.warning("No historical M1 bars returned for %s — skipping today.", sym)
            continue

        levels = compute_levels(m1_hist, trade_date)
        levels_map[sym] = levels
        if not levels.is_valid:
            log.warning("Levels invalid for %s on %s — skipping today.", sym, trade_date)
            continue

        seed_prev_close = float(m1_hist["close"].iloc[-1]) if not m1_hist.empty and "close" in m1_hist else None
        sm = StrategyStateMachine(cfg_s)
        sm.on_day_start(trade_date, levels, seed_prev_close=seed_prev_close)
        sm_map[sym] = sm

    active_symbols = list(sm_map.keys())
    if not active_symbols:
        log.error("No valid symbols to trade today — terminating session.")
        return

    start_balance = broker.get_account_balance()
    kill_threshold = start_balance * (1.0 - cfg.daily_loss_kill_pct / 100.0)
    print_portfolio_dashboard(levels_map, cfg_map, start_balance, kill_threshold, trade_date)

    session_tickets: list[str] = []
    bar_idx = 0

    # ── Main 15-minute loop with continuous trailing & risk monitoring ────────
    while True:
        if max_bars is not None and bar_idx >= max_bars:
            log.info("Reached maximum bar count (%d) — exiting session loop.", max_bars)
            break

        now = _now_ist()

        # 1. Hard time exit at 21:30 IST (22:30 winter)
        if now >= time_exit:
            log.warning(
                "[TIME EXIT] %s IST reached — closing all open positions.",
                time_exit.strftime("%H:%M"),
            )
            _close_all_open(broker, session_tickets)
            break

        # 2. Portfolio daily loss kill switch
        current_balance = broker.get_account_balance()
        if current_balance <= kill_threshold:
            log.error(
                "[KILL SWITCH] Daily loss limit triggered: balance %.2f <= %.2f (-%.2f%%). STOPPING.",
                current_balance, kill_threshold, cfg.daily_loss_kill_pct,
            )
            _close_all_open(broker, session_tickets)
            break

        # 3. Dynamic Breakeven and SL/TP checking
        if hasattr(broker, "check_sl_tp"):
            closed = broker.check_sl_tp()
            for t in closed:
                if t in session_tickets:
                    session_tickets.remove(t)

        # 4. Wait for next M15 bar close with continuous trailing poll
        if not skip_bar_wait:
            next_bar_close = _next_m15_close(now, cfg)
            while _now_ist() < next_bar_close:
                cur_now = _now_ist()
                if cur_now >= time_exit:
                    break
                if broker.get_account_balance() <= kill_threshold:
                    break
                if hasattr(broker, "check_sl_tp"):
                    closed = broker.check_sl_tp()
                    for t in closed:
                        if t in session_tickets:
                            session_tickets.remove(t)

                remaining_sec = (next_bar_close - cur_now).total_seconds()
                sleep_chunk = min(poll_interval_sec, max(0.05, remaining_sec))
                time.sleep(sleep_chunk)

            # Re-check kill conditions after waking
            now_post_wake = _now_ist()
            if now_post_wake >= time_exit:
                log.warning("[TIME EXIT] Time exit reached after bar wait — closing positions.")
                _close_all_open(broker, session_tickets)
                break
            if broker.get_account_balance() <= kill_threshold:
                log.error("[KILL SWITCH] Kill threshold breached — closing positions.")
                _close_all_open(broker, session_tickets)
                break

        log.info(
            "--- M15 Bar %d Close [%s IST] ---",
            bar_idx, _now_ist().strftime("%H:%M:%S"),
        )

        # 5. Process signals across all active symbols
        for sym in active_symbols:
            cfg_s = cfg_map[sym]
            sm = sm_map[sym]

            # Portfolio risk check: max concurrent open positions
            open_positions = broker.get_open_positions()
            if len(open_positions) >= cfg.max_open_positions:
                log.info(
                    "[PORTFOLIO RISK] Max concurrent positions (%d/%d) reached — skipping %s entries",
                    len(open_positions), cfg.max_open_positions, sym,
                )
                break

            # Per-symbol trade limits
            if trades_today[sym] >= cfg_s.max_trades_per_day:
                log.info("Max trades/day (%d) reached for %s — skipping", cfg_s.max_trades_per_day, sym)
                continue

            # Symbol concurrency: only 1 open position per symbol
            sym_positions = broker.get_open_positions(symbol=sym)
            if sym_positions:
                log.debug("Symbol %s already has an open position (%s) — skipping", sym, sym_positions[0]["ticket"])
                continue

            # State recovery safety check
            if hasattr(broker, "has_recovered_positions_for") and broker.has_recovered_positions_for(sym):
                log.warning("[SAFETY] Blocking entry on %s — recovered position still open", sym)
                continue

            # Fetch recent M1 bars to construct the closed M15 bar
            m1_recent = broker.get_m1_bars(sym, count=200)
            if m1_recent.empty:
                log.warning("No recent M1 bars for %s — skipping bar %d", sym, bar_idx)
                continue

            m15 = resample_m1_to_m15(m1_recent)
            if m15.empty:
                continue

            current_m15 = m15.iloc[-1]
            new_signals = sm.on_m15_close(bar_idx, current_m15)

            for signal in new_signals:
                if len(broker.get_open_positions()) >= cfg.max_open_positions:
                    log.info("[PORTFOLIO RISK] Max open positions reached during signal processing.")
                    break

                if trades_today[sym] >= cfg_s.max_trades_per_day:
                    break

                if dry_run:
                    log.info("[DRY RUN] Signal on %s: %s", sym, signal)
                    continue

                try:
                    bid = broker.get_current_bid(sym)
                except Exception as e:
                    log.error("Failed to get bid for %s: %e", sym, e)
                    continue

                balance = broker.get_account_balance()
                order = build_order(signal, bid, balance, cfg_s)

                if order is None:
                    log.info("Signal on %s skipped — risk_pips out of bounds", sym)
                    continue

                ticket = broker.place_order(order)
                if ticket:
                    session_tickets.append(ticket)
                    trades_today[sym] += 1
                    log.info(
                        "Order placed: %s %s %.2fL @ %.5f ticket=%s (today: %d/%d)",
                        order.direction.name, sym, order.lot_size, order.entry_price,
                        ticket, trades_today[sym], cfg_s.max_trades_per_day,
                    )

        bar_idx += 1

    log.info("=== Live Session Finished ===")
    log.info("Total orders placed this session: %d", len(session_tickets))
    for sym in active_symbols:
        log.info("  • %s trades: %d", sym, trades_today.get(sym, 0))


def _close_all_open(broker: BrokerBase, session_tickets: list[str]) -> None:
    """Close all open positions safely across all symbols."""
    if hasattr(broker, "close_all_positions"):
        broker.close_all_positions(reason="kill-switch")
    else:
        for ticket in list(session_tickets):
            broker.close_position(ticket)
        for pos in broker.get_open_positions():
            broker.close_position(pos["ticket"])
