"""
sweepbot/engine/reports.py
---------------------------
Format and print backtest results to console and optionally to CSV.
Uses Rich for beautiful terminal output.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from sweepbot.engine.fills import ExitReason, TradeResult
from sweepbot.engine.metrics import MetricsResult

console = Console()


def print_summary(metrics: MetricsResult, title: str = "Backtest Results") -> None:
    """Print a rich formatted summary table to the terminal."""
    table = Table(title=title, box=box.ASCII, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value", justify="right")

    for k, v in metrics.summary_dict().items():
        # Colour-code key metrics
        if "Net R" in k and "Total" not in k:
            val = metrics.avg_net_r
            style = "green" if val > 0 else "red"
        elif "Win Rate" in k and "Early" not in k and "Late W" not in k:
            val_str = v.replace("%", "")
            try:
                val = float(val_str)
                style = "green" if val >= 50 else "yellow"
            except ValueError:
                style = ""
        elif "Drawdown" in k:
            style = "red"
        elif "Drag" in k:
            style = "yellow"
        else:
            style = ""
        table.add_row(k, f"[{style}]{v}[/{style}]" if style else str(v))

    console.print(table)


def print_trade_log(results: list[TradeResult], max_rows: int = 50) -> None:
    """Print a per-trade table (limited to max_rows)."""
    table = Table(
        title=f"Trade Log (showing first {min(len(results), max_rows)})",
        box=box.ASCII,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", width=4)
    table.add_column("Entry Time (IST)", width=20)
    table.add_column("Dir", width=6)
    table.add_column("Level", width=8)
    table.add_column("Pattern", width=12)
    table.add_column("Entry", justify="right", width=9)
    table.add_column("SL", justify="right", width=9)
    table.add_column("TP", justify="right", width=9)
    table.add_column("Risk (p)", justify="right", width=9)
    table.add_column("Exit", justify="right", width=9)
    table.add_column("Reason", width=8)
    table.add_column("Net R", justify="right", width=8)

    for i, r in enumerate(results[:max_rows], 1):
        net_r = r.net_r
        r_style = "green" if net_r > 0 else "red"
        table.add_row(
            str(i),
            str(r.entry_bar_ist.strftime("%Y-%m-%d %H:%M")),
            r.order.direction.name,
            r.order.signal.level_name,
            r.order.signal.pattern,
            f"{r.order.entry_price:.5f}",
            f"{r.order.stop_price:.5f}",
            f"{r.order.target_price:.5f}",
            f"{r.order.risk_pips:.1f}",
            f"{r.exit_price:.5f}",
            r.exit_reason.name,
            f"[{r_style}]{net_r:+.3f}[/{r_style}]",
        )

    console.print(table)


def save_trade_csv(results: list[TradeResult], filepath: str | Path) -> None:
    """Export all trades to a CSV file for further analysis."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "entry_date", "entry_time_ist", "direction", "level_name", "pattern",
        "entry_price", "stop_price", "target_price", "risk_pips",
        "lots", "exit_price", "exit_reason", "exit_time_ist",
        "gross_r", "commission_r", "net_r", "is_early_entry",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "entry_date": r.entry_bar_ist.date(),
                "entry_time_ist": r.entry_bar_ist.strftime("%H:%M"),
                "direction": r.order.direction.name,
                "level_name": r.order.signal.level_name,
                "pattern": r.order.signal.pattern,
                "entry_price": f"{r.order.entry_price:.5f}",
                "stop_price": f"{r.order.stop_price:.5f}",
                "target_price": f"{r.order.target_price:.5f}",
                "risk_pips": f"{r.order.risk_pips:.2f}",
                "lots": f"{r.order.lot_size:.2f}",
                "exit_price": f"{r.exit_price:.5f}",
                "exit_reason": r.exit_reason.name,
                "exit_time_ist": r.exit_time.strftime("%Y-%m-%d %H:%M"),
                "gross_r": f"{r.gross_r:.4f}",
                "commission_r": f"{r.commission_r:.4f}",
                "net_r": f"{r.net_r:.4f}",
                "is_early_entry": r.is_early_entry,
            })

    console.print(f"[green][OK] Trade log saved to {filepath}[/green]")
