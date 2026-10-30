"""
sweepbot/cli.py
----------------
Click-based CLI entry point.

Commands:
  sweepbot backtest  --start 2025-01-01 --end 2025-12-31
  sweepbot backtest  --csv path/to/data.csv --start ... --end ...
  sweepbot live      --mode paper|live [--dry-run]
  sweepbot download  --start 2025-01-01 --end 2026-01-01
  sweepbot verify-tz --csv path/to/data.csv
"""
from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """SweepBot — EURUSD NY-open sweep-and-reject automated trading bot."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ── backtest ──────────────────────────────────────────────────────────────────

@main.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option("--csv", "csv_path", default=None, help="Load M1 data from CSV instead of MT5")
@click.option("--balance", default=10_000.0, show_default=True, help="Starting account balance USD")
@click.option("--no-log", is_flag=True, default=False, help="Suppress per-trade log")
@click.option("--out-csv", default=None, help="Export trade log to this CSV file")
@click.option("--risk-pct", default=None, type=float,
              help="Override risk_per_trade_pct (e.g. 0.5)")
@click.option("--symbol", default="EURUSD", show_default=True, help="Forex symbol (e.g. EURUSD, GBPUSD, USDJPY)")
@click.option("--early-only/--all-window", "early_only", default=True, help="Restrict entries before 19:00 IST / 09:30 NY cash open")
@click.pass_context
def backtest(ctx, start, end, csv_path, balance, no_log, out_csv, risk_pct, symbol, early_only):
    """Run a backtest over historical M1 data."""
    import dataclasses
    from sweepbot.config import CFG, Config
    from sweepbot.data.loader import load_m1, load_m1_from_csv
    from sweepbot.engine.backtest import run_backtest

    symbols = [s.strip().upper() for s in symbol.split(",") if s.strip()]
    if not symbols:
        symbols = ["EURUSD"]

    if len(symbols) == 1:
        sym = symbols[0]
        overrides = {"early_entries_only": early_only}
        if risk_pct is not None:
            overrides["risk_per_trade_pct"] = risk_pct
            console.print(f"[yellow][!] risk_per_trade_pct overridden to {risk_pct}%[/yellow]")

        cfg = Config(symbol=sym, **overrides)
        console.print(f"[bold cyan]Running backtest ({sym}): {start} -> {end}[/bold cyan]")

        try:
            if csv_path:
                console.print(f"Loading data from CSV: {csv_path}")
                m1 = load_m1_from_csv(csv_path)
                import pandas as pd
                from sweepbot.config import TZ_UTC
                m1 = m1[(m1.index >= pd.Timestamp(start, tz=TZ_UTC)) &
                         (m1.index <= pd.Timestamp(end, tz=TZ_UTC) + pd.Timedelta(days=1))]
            else:
                console.print("Downloading / loading cached M1 data from MT5...")
                m1 = load_m1(start, end, cfg=cfg)
        except Exception as e:
            console.print(f"[red]Failed to load data: {e}[/red]")
            sys.exit(1)

        results, metrics = run_backtest(
            m1,
            account_balance=balance,
            cfg=cfg,
            print_log=not no_log,
            save_csv=out_csv,
            title=f"Backtest {start} -> {end}",
        )
    else:
        # Multi-asset portfolio backtest
        from sweepbot.engine.metrics import compute_metrics
        from sweepbot.engine.reports import print_summary, print_trade_log, save_trade_csv

        console.print(f"[bold cyan]Running multi-pair portfolio backtest ({', '.join(symbols)}): {start} -> {end}[/bold cyan]")
        all_portfolio_trades = []
        symbol_summaries = {}

        for sym in symbols:
            overrides = {"early_entries_only": early_only}
            if risk_pct is not None:
                overrides["risk_per_trade_pct"] = risk_pct
            cfg_sym = Config(symbol=sym, **overrides)
            console.print(f"Loading data for {sym}...")
            try:
                m1_sym = load_m1(start, end, cfg=cfg_sym)
            except Exception as e:
                console.print(f"[red]Failed to load data for {sym}: {e}[/red]")
                sys.exit(1)

            trades_sym, metrics_sym = run_backtest(
                m1_sym,
                account_balance=balance,
                cfg=cfg_sym,
                print_log=False,
                title=f"{sym} Backtest",
            )
            all_portfolio_trades.extend(trades_sym)
            symbol_summaries[sym] = (len(trades_sym), metrics_sym.total_net_r, metrics_sym.win_rate)

        all_portfolio_trades.sort(key=lambda t: t.entry_time)
        metrics = compute_metrics(all_portfolio_trades)

        if not no_log:
            print_trade_log(all_portfolio_trades)

        print_summary(metrics, title=f"Portfolio ({', '.join(symbols)}) {start} -> {end}")

        console.print("\n[bold]Symbol Breakdown:[/bold]")
        for sym, (cnt, net_r, wr) in symbol_summaries.items():
            console.print(f"  • {sym}: {cnt} trades | WR: {wr:.1%} | Net: {net_r:+.2f}R")

        if out_csv:
            save_trade_csv(all_portfolio_trades, out_csv)

    # SDD Section 8 acceptance criteria check
    console.print("\n[bold]Validation Protocol (SDD Section 8):[/bold]")
    checks = {
        f"Trades > 60": (metrics.trade_count > 60, f"{metrics.trade_count}"),
        f"Net avg R > 0": (metrics.avg_net_r > 0, f"{metrics.avg_net_r:+.4f}"),
        f"CI lower > -inf (any positive signal)": (metrics.ci_lower > -1.0, f"{metrics.ci_lower:+.3f}"),
    }
    all_pass = True
    for check, (passed, val) in checks.items():
        icon = "[OK]" if passed else "[FAIL]"
        color = "green" if passed else "red"
        console.print(f"  [{color}]{icon}[/{color}] {check}: {val}")
        if not passed:
            all_pass = False

    if all_pass:
        console.print("[bold green]All acceptance criteria passed.[/bold green]")
    else:
        console.print("[bold red]One or more acceptance criteria FAILED.[/bold red]")


# ── download ──────────────────────────────────────────────────────────────────

@main.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--symbol", default="EURUSD", show_default=True, help="Forex symbol to download")
@click.pass_context
def download(ctx, start, end, symbol):
    """Download M1 data from MT5 and cache locally."""
    import dataclasses
    from sweepbot.config import CFG
    from sweepbot.data.loader import load_m1
    cfg = dataclasses.replace(CFG, symbol=symbol)
    console.print(f"[cyan]Downloading {symbol} {start} -> {end}...[/cyan]")
    try:
        df = load_m1(start, end, force_download=True, cfg=cfg)
        console.print(f"[green]Downloaded {len(df):,} M1 bars for {symbol}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        sys.exit(1)


# ── verify-tz ─────────────────────────────────────────────────────────────────

@main.command("verify-tz")
@click.option("--csv", "csv_path", required=True, help="Path to M1 CSV file")
@click.pass_context
def verify_tz(ctx, csv_path):
    """Verify timezone integrity of M1 CSV data (SDD §5.1)."""
    from sweepbot.data.loader import load_m1_from_csv
    from sweepbot.data.timeutil import verify_timezone
    console.print(f"[cyan]Verifying timezone: {csv_path}[/cyan]")
    try:
        df = load_m1_from_csv(csv_path)
        ok = verify_timezone(df, verbose=True)
        if ok:
            console.print("[bold green]Timezone verification PASSED[/bold green]")
        else:
            console.print("[bold red]Timezone verification FAILED — check data source![/bold red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# ── live ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--mode", type=click.Choice(["paper", "live"]), default="paper",
              show_default=True, help="Trading mode")
@click.option("--dry-run", is_flag=True, default=False,
              help="Detect signals but do NOT place orders")
@click.option("--symbol", default="EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD", show_default=True,
              help="Forex symbol(s) to trade (comma-separated for multi-pair)")
@click.option("--early-only/--all-window", "early_only", default=True,
              help="Restrict entries before 19:00 IST / 09:30 NY cash open")
@click.option("--risk-pct", default=None, type=float,
              help="Risk per trade as % of equity (default: 0.25%)")
@click.option("--max-positions", default=None, type=int,
              help="Max concurrent open positions across portfolio (default: 3)")
@click.option("--skip-wait", is_flag=True, default=False,
              help="Skip waiting for 17:25 IST arm time and start immediately")
@click.option("--tg-token", default=None, help="Telegram bot token")
@click.option("--tg-chat-id", default=None, help="Telegram chat ID")
@click.option("--login", default=None, type=int, help="MT5 account login")
@click.option("--password", default=None, help="MT5 account password")
@click.option("--server", default=None, help="MT5 broker server")
@click.pass_context
def live(ctx, mode, dry_run, symbol, early_only, risk_pct, max_positions, skip_wait, tg_token, tg_chat_id, login, password, server):
    """Run the live / paper trading session."""
    import dataclasses
    from sweepbot.config import CFG

    symbols = [s.strip().upper() for s in symbol.split(",") if s.strip()]
    primary_symbol = symbols[0] if symbols else "EURUSD"

    overrides = {"symbol": primary_symbol, "early_entries_only": early_only}
    if risk_pct is not None:
        overrides["risk_per_trade_pct"] = risk_pct
    if max_positions is not None:
        overrides["max_open_positions"] = max_positions
    import os
    token = tg_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = tg_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if token:
        overrides["telegram_bot_token"] = token
    if chat:
        overrides["telegram_chat_id"] = chat
    if login or password or server:
        overrides.update({
            "mt5_login": login or CFG.mt5_login,
            "mt5_password": password or CFG.mt5_password,
            "mt5_server": server or CFG.mt5_server,
        })
    cfg = dataclasses.replace(CFG, **overrides)

    console.print(f"[bold cyan]Starting SweepBot [{mode.upper()}] mode for {len(symbols)} symbol(s): {', '.join(symbols)}[/bold cyan]")
    if dry_run:
        console.print("[yellow][!] DRY RUN -- no real orders will be placed[/yellow]")

    # Connect to MT5
    if mode == "paper":
        from sweepbot.live.mt5 import MT5Broker
        from sweepbot.live.paper import PaperBroker
        mt5_broker = MT5Broker(cfg=cfg)
        if not mt5_broker.connect():
            console.print("[red]Failed to connect to MT5[/red]")
            sys.exit(1)
        broker = PaperBroker(mt5_module=mt5_broker._mt5, cfg=cfg)
    else:
        from sweepbot.live.mt5 import MT5Broker
        broker = MT5Broker(cfg=cfg)
        if not broker.connect():
            console.print("[red]Failed to connect to MT5[/red]")
            sys.exit(1)

    try:
        from sweepbot.live.runner import run_live
        run_live(broker, symbols=symbols, cfg=cfg, dry_run=dry_run, wait_for_arm=not skip_wait)
    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()
        console.print("[bold]Session finished.[/bold]")




# ── setup-telegram ────────────────────────────────────────────────────────────

@main.command("setup-telegram")
@click.option("--token", default=None, help="Telegram Bot Token from BotFather")
@click.pass_context
def setup_telegram(ctx, token):
    """Interactively verify Telegram bot token and auto-detect chat ID."""
    import json
    import time
    import urllib.request
    import urllib.error
    from pathlib import Path
    from sweepbot.config import load_local_config

    cfg_data = load_local_config()
    bot_token = token or cfg_data.get("telegram_bot_token")

    if not bot_token:
        bot_token = click.prompt("Enter your Telegram Bot Token (from @BotFather)")
    bot_token = bot_token.strip()

    console.print(f"[cyan]Verifying bot token with Telegram API...[/cyan]")
    url_me = f"https://api.telegram.org/bot{bot_token}/getMe"
    try:
        req = urllib.request.Request(url_me, headers={"User-Agent": "SweepBot/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        console.print(f"[bold red]Telegram verification failed (HTTP {e.code}):[/bold red] {err_msg}")
        console.print("[yellow]Please check BotFather on Telegram -> /mybots -> select your bot -> API Token, and enter the fresh token.[/yellow]")
        return
    except Exception as e:
        console.print(f"[bold red]Network error connecting to Telegram:[/bold red] {e}")
        return

    bot_info = data.get("result", {})
    username = bot_info.get("username", "your bot")
    first_name = bot_info.get("first_name", "Bot")
    console.print(f"[bold green]Bot verified successfully:[/bold green] {first_name} (@{username})")

    console.print("\n[bold yellow]To automatically discover your Chat ID:[/bold yellow]")
    console.print(f"  1. Open Telegram on your phone or PC.")
    console.print(f"  2. Search for: [bold cyan]@{username}[/bold cyan]")
    console.print(f"  3. Click [bold]Start[/bold] or send any message (e.g. 'hello').")
    console.print("[cyan]Waiting for your message (listening for up to 60 seconds)...[/cyan]")

    url_updates = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    chat_id = None
    chat_name = ""

    for _ in range(30):
        time.sleep(2)
        try:
            req = urllib.request.Request(url_updates, headers={"User-Agent": "SweepBot/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                upd = json.loads(resp.read().decode())
                results = upd.get("result", [])
                if results:
                    last_msg = results[-1].get("message") or results[-1].get("channel_post")
                    if last_msg and "chat" in last_msg:
                        chat_id = str(last_msg["chat"]["id"])
                        chat_name = last_msg["chat"].get("first_name") or last_msg["chat"].get("title") or "User"
                        break
        except Exception:
            pass

    if not chat_id:
        console.print("[bold red]Timed out waiting for message.[/bold red]")
        console.print("[yellow]You can also find your chat ID by messaging @userinfobot on Telegram, then edit config.json.[/yellow]")
        return

    console.print(f"[bold green]Discovered Chat ID:[/bold green] {chat_id} ({chat_name})")

    # Send a confirmation test message
    url_send = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"<b>SweepBot Connected</b>\nTelegram notifications have been successfully configured for {chat_name}!",
        "parse_mode": "HTML",
    }
    try:
        req = urllib.request.Request(
            url_send,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "SweepBot/1.0"},
        )
        urllib.request.urlopen(req, timeout=5)
        console.print("[bold green]Confirmation message sent to your Telegram![/bold green]")
    except Exception as e:
        console.print(f"[yellow]Could not send test message: {e}[/yellow]")

    # Save to config.json
    cfg_file = Path("config.json")
    cfg_data["telegram_bot_token"] = bot_token
    cfg_data["telegram_chat_id"] = str(chat_id)
    cfg_file.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved to config.json successfully![/bold green]")

if __name__ == "__main__":
    main()
