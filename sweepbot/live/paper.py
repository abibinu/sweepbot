"""
sweepbot/live/paper.py
-----------------------
Paper trading broker — simulates execution using live MT5 price feed
but does NOT send real orders. Tracks positions in memory.

Uses the same M1 fill and dynamic breakeven logic as the backtest for consistency.
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from sweepbot.config import CFG, TZ_IST, pip_size
from sweepbot.live.broker_base import BrokerBase
from sweepbot.risk import OrderSpec, pip_value_usd_per_lot
from sweepbot.strategy import Direction

log = logging.getLogger(__name__)


@dataclass
class _PaperPosition:
    ticket: str
    direction: Direction
    lots: float
    open_price: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    symbol: str
    risk_pips: float = 0.0
    initial_sl: float = 0.0
    breakeven_active: bool = False
    commission_usd: float = 0.0


class PaperBroker(BrokerBase):
    """
    Paper broker backed by a live MT5 price feed for tick data.
    Positions are held in memory with multi-pair support and realized P&L tracking.
    """

    def __init__(self, mt5_module=None, cfg=CFG, initial_balance: float = 10_000.0):
        self.cfg = cfg
        self._positions: dict[str, _PaperPosition] = {}
        self._closed_trades: list[dict] = []
        self._mt5 = mt5_module  # optional real MT5 for price feed
        self._balance: float = initial_balance
        self._initial_balance: float = initial_balance

    # ── BrokerBase implementation ─────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Paper equity: realized balance + unrealized floating PnL."""
        return self._balance + self._get_unrealized_pnl()

    def _get_unrealized_pnl(self) -> float:
        """Calculate floating unrealized P&L in USD across all open positions."""
        total_unrealized = 0.0
        for pos in self._positions.values():
            try:
                bid = self.get_current_bid(pos.symbol)
                pip = pip_size(pos.symbol)
                usd_per_pip_lot = pip_value_usd_per_lot(pos.symbol, bid)
                if pos.direction == Direction.LONG:
                    pips = (bid - pos.open_price) / pip
                else:
                    ask = bid + self.cfg.spread_pips * pip
                    pips = (pos.open_price - ask) / pip
                total_unrealized += pips * usd_per_pip_lot * pos.lots
            except Exception:
                continue
        return total_unrealized

    def get_current_bid(self, symbol: str) -> float:
        if self._mt5 is not None:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick:
                return float(tick.bid)
        raise RuntimeError(f"No MT5 price feed available for {symbol}")

    def place_order(self, order: OrderSpec) -> Optional[str]:
        ticket = str(uuid.uuid4())[:8].upper()
        sym = getattr(order, "symbol", self.cfg.symbol)
        pos = _PaperPosition(
            ticket=ticket,
            direction=order.direction,
            lots=order.lot_size,
            open_price=order.entry_price,
            sl=order.stop_price,
            tp=order.target_price,
            open_time=pd.Timestamp.now(tz=TZ_IST),
            symbol=sym,
            risk_pips=order.risk_pips,
            initial_sl=order.stop_price,
            commission_usd=order.commission_usd,
        )
        self._positions[ticket] = pos
        self._balance -= order.commission_usd
        log.info(
            "[PAPER] Order placed: %s %s %.2fL @ %.5f SL=%.5f TP=%.5f ticket=%s (comm: $%.2f)",
            order.direction.name, sym,
            order.lot_size, order.entry_price,
            order.stop_price, order.target_price, ticket,
            order.commission_usd,
        )
        return ticket

    def close_position(self, ticket: str, exit_price: Optional[float] = None) -> bool:
        pos = self._positions.pop(ticket, None)
        if pos is None:
            log.warning("[PAPER] Tried to close unknown ticket: %s", ticket)
            return False

        pip = pip_size(pos.symbol)
        if exit_price is None:
            try:
                bid = self.get_current_bid(pos.symbol)
                if pos.direction == Direction.LONG:
                    exit_price = bid
                else:
                    exit_price = bid + self.cfg.spread_pips * pip
            except Exception:
                exit_price = pos.open_price

        # Calculate realized P&L
        if pos.direction == Direction.LONG:
            pips = (exit_price - pos.open_price) / pip
        else:
            pips = (pos.open_price - exit_price) / pip

        usd_per_pip_lot = pip_value_usd_per_lot(pos.symbol, exit_price)
        gross_usd = pips * usd_per_pip_lot * pos.lots
        self._balance += gross_usd

        net_r = (gross_usd - pos.commission_usd) / (pos.lots * pos.risk_pips * usd_per_pip_lot) if (pos.lots * pos.risk_pips * usd_per_pip_lot) > 0 else 0.0

        log.info(
            "[PAPER] Closed %s %s @ %.5f (open: %.5f | PnL: %+.2f USD | %+.2fR)",
            ticket, pos.symbol, exit_price, pos.open_price, gross_usd, net_r,
        )
        self._closed_trades.append({
            "ticket": ticket,
            "symbol": pos.symbol,
            "direction": pos.direction.name,
            "lots": pos.lots,
            "open_price": pos.open_price,
            "close_price": exit_price,
            "sl": pos.sl,
            "tp": pos.tp,
            "gross_pnl_usd": gross_usd,
            "commission_usd": pos.commission_usd,
            "net_pnl_usd": gross_usd - pos.commission_usd,
            "net_r": net_r,
        })
        return True

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict]:
        positions = self._positions.values()
        if symbol is not None:
            positions = [p for p in positions if p.symbol == symbol]
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "direction": p.direction.name,
                "lots": p.lots,
                "open_price": p.open_price,
                "sl": p.sl,
                "tp": p.tp,
                "open_time": p.open_time,
                "breakeven_active": p.breakeven_active,
            }
            for p in positions
        ]

    def get_m1_bars(self, symbol: str, count: int) -> pd.DataFrame:
        if self._mt5 is None:
            raise RuntimeError("No MT5 module available")
        mt5 = self._mt5
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename_axis("timestamp")
        df = df[["open", "high", "low", "close", "tick_volume"]]
        return df

    # ── Paper-specific helpers ────────────────────────────────────────────────

    def check_sl_tp(self, symbol: Optional[str] = None) -> list[str]:
        """
        Check open positions against current price.
        Applies dynamic breakeven trailing (SDD §2.7) and closes any that hit SL or TP.
        Returns list of tickets closed.
        """
        closed = []
        for ticket, pos in list(self._positions.items()):
            if symbol is not None and pos.symbol != symbol:
                continue

            try:
                bid = self.get_current_bid(pos.symbol)
            except Exception:
                continue

            pip = pip_size(pos.symbol)
            ask = bid + self.cfg.spread_pips * pip

            # 1. Dynamic Breakeven Trailing (trigger at +1.3R)
            if getattr(self.cfg, "enable_breakeven", True) and not pos.breakeven_active and pos.risk_pips > 0:
                trigger_dist = self.cfg.breakeven_trigger_r * pos.risk_pips * pip
                buf = self.cfg.breakeven_buffer_pips * pip

                if pos.direction == Direction.LONG:
                    if bid >= pos.open_price + trigger_dist:
                        pos.sl = pos.open_price + buf
                        pos.breakeven_active = True
                        log.info("[BREAKEVEN] %s %s moved to %.5f", pos.symbol, ticket, pos.sl)
                else:  # SHORT
                    if ask <= pos.open_price - trigger_dist:
                        pos.sl = pos.open_price - buf
                        pos.breakeven_active = True
                        log.info("[BREAKEVEN] %s %s moved to %.5f", pos.symbol, ticket, pos.sl)

            # 2. Check SL / TP Hit
            if pos.direction == Direction.LONG:
                if bid <= pos.sl:
                    self.close_position(ticket, exit_price=pos.sl)
                    closed.append(ticket)
                elif bid >= pos.tp:
                    self.close_position(ticket, exit_price=pos.tp)
                    closed.append(ticket)
            else:  # SHORT
                if ask >= pos.sl:
                    self.close_position(ticket, exit_price=pos.sl)
                    closed.append(ticket)
                elif ask <= pos.tp:
                    self.close_position(ticket, exit_price=pos.tp)
                    closed.append(ticket)

        return closed
