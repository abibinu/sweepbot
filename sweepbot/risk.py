"""
sweepbot/risk.py
-----------------
Position sizing and commission calculations.

SDD §2.6 — Order, Stop and Target:
  1. Entry: market at entry bar open.
     Longs:  execution_price = Ask = Bid + spread_pips
     Shorts: execution_price = Bid
  2. Stop: extreme - stop_buffer_pips (long) / extreme + stop_buffer_pips (short)
  3. risk_pips = distance from entry to stop. Skip if < 3 or > 20.
  4. Target: entry + 2 * risk (long) / entry - 2 * risk (short).

SDD §2.8 — Commissions:
  commission_usd per standard lot is deducted from net PnL.
  The commission drag is tracked separately for reporting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from sweepbot.config import CFG, PIP
from sweepbot.strategy import Direction, Signal


@dataclass
class OrderSpec:
    """Complete order specification ready for execution."""
    signal: Signal
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    risk_pips: float
    lot_size: float          # in standard lots
    account_balance: float   # snapshot at time of order
    risk_amount_usd: float   # dollars at risk (before commission)
    commission_usd: float    # commission for this trade
    symbol: str = "EURUSD"

    @property
    def net_risk_usd(self) -> float:
        return self.risk_amount_usd + self.commission_usd

    def __repr__(self) -> str:
        d = "BUY" if self.direction == Direction.LONG else "SELL"
        return (
            f"OrderSpec({d} {self.lot_size:.2f}L @ {self.entry_price:.5f} "
            f"SL={self.stop_price:.5f} TP={self.target_price:.5f} "
            f"risk={self.risk_pips:.1f}p lot={self.lot_size:.2f})"
        )


def pip_value_usd_per_lot(symbol: str, current_price: float = 1.0) -> float:
    """Return USD value of 1 pip for 1.0 standard lot."""
    sym = symbol.upper()
    if sym.endswith("USD"):  # EURUSD, GBPUSD, AUDUSD
        return 10.0
    elif "JPY" in sym:
        # USDJPY: 1 pip = 0.01 JPY -> 1000 JPY / price in USD
        return 1000.0 / current_price if current_price > 0 else 6.70
    elif sym.startswith("USD"):  # USDCAD, USDCHF
        # 10 quote currency / price
        return 10.0 / current_price if current_price > 0 else 7.50
    return 10.0


def compute_entry_price(signal: Signal, bid_price: float, cfg=CFG) -> float:
    """
    Compute execution price from a bid price.

    Longs:  Ask = Bid + spread (buy at the ask)
    Shorts: Bid (sell at bid)
    """
    pip = getattr(cfg, "pip", PIP)
    if signal.direction == Direction.LONG:
        return bid_price + cfg.spread_pips * pip
    else:
        return bid_price


def compute_stop_target(
    signal: Signal,
    entry_price: float,
    cfg=CFG,
) -> tuple[float, float, float]:
    """
    Compute stop, target, and risk_pips from signal + entry price.

    Returns (stop_price, target_price, risk_pips).
    Returns (nan, nan, nan) if risk_pips is outside [min_stop, max_stop].
    """
    pip = getattr(cfg, "pip", PIP)
    buf = cfg.stop_buffer_pips * pip

    if signal.direction == Direction.LONG:
        stop_price = signal.extreme - buf
        risk_pips = (entry_price - stop_price) / pip
        target_price = entry_price + cfg.target_r * (entry_price - stop_price)
    else:  # SHORT
        stop_price = signal.extreme + buf
        risk_pips = (stop_price - entry_price) / pip
        target_price = entry_price - cfg.target_r * (stop_price - entry_price)

    if not (cfg.min_stop_pips <= risk_pips <= cfg.max_stop_pips):
        return float("nan"), float("nan"), float("nan")

    return stop_price, target_price, risk_pips


def compute_lot_size(
    account_balance: float,
    risk_pips: float,
    cfg=CFG,
    current_price: float = 1.0,
) -> float:
    """
    Compute position size in standard lots.

    Formula:
        risk_amount = account_balance * risk_per_trade_pct / 100
        pip_value   = pip value per lot in USD
        lots        = risk_amount / (risk_pips * pip_value)

    Rounds down to the nearest lot_step and clips to [min_lot, ∞).
    """
    pip_val = pip_value_usd_per_lot(cfg.symbol, current_price)
    risk_usd = account_balance * (cfg.risk_per_trade_pct / 100.0)
    raw_lots = risk_usd / (risk_pips * pip_val)

    # Round DOWN to nearest lot_step
    lots = math.floor(raw_lots / cfg.lot_step) * cfg.lot_step
    lots = max(lots, cfg.min_lot)
    return round(lots, 2)


def compute_commission(lot_size: float, cfg=CFG) -> float:
    """Commission in USD for a given lot size."""
    return lot_size * cfg.commission_usd_per_lot


def build_order(
    signal: Signal,
    bid_price: float,
    account_balance: float,
    cfg=CFG,
) -> Optional[OrderSpec]:
    """
    Build a complete OrderSpec from a signal and current market price.

    Returns None if the trade should be skipped (risk_pips out of range).
    Also populates signal.entry_price, .stop_price, .target_price, .risk_pips.
    """
    entry = compute_entry_price(signal, bid_price, cfg)
    stop, target, risk_pips = compute_stop_target(signal, entry, cfg)

    if math.isnan(risk_pips):
        return None

    lots = compute_lot_size(account_balance, risk_pips, cfg, current_price=bid_price)
    commission = compute_commission(lots, cfg)
    risk_usd = account_balance * (cfg.risk_per_trade_pct / 100.0)

    # Mutate signal with computed prices for downstream use
    signal.entry_price = entry
    signal.stop_price = stop
    signal.target_price = target
    signal.risk_pips = risk_pips

    return OrderSpec(
        signal=signal,
        direction=signal.direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_pips=risk_pips,
        lot_size=lots,
        account_balance=account_balance,
        risk_amount_usd=risk_usd,
        commission_usd=commission,
        symbol=cfg.symbol,
    )


def commission_in_r(commission_usd: float, risk_amount_usd: float) -> float:
    """Express commission as a fraction of R (risk)."""
    if risk_amount_usd <= 0:
        return 0.0
    return commission_usd / risk_amount_usd
