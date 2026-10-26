"""
sweepbot/live/paper.py
-----------------------
Paper trading broker — simulates execution using live MT5 price feed
but does NOT send real orders. Tracks positions in memory.

Uses the same M1 fill logic as the backtest for consistency.
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from sweepbot.config import CFG, PIP, TZ_IST
from sweepbot.live.broker_base import BrokerBase
from sweepbot.risk import OrderSpec
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


class PaperBroker(BrokerBase):
    """
    Paper broker backed by a live MT5 price feed for tick data.
    Positions are held in memory. All P&L is virtual.
    """

    def __init__(self, mt5_module=None, cfg=CFG):
        self.cfg = cfg
        self._positions: dict[str, _PaperPosition] = {}
        self._closed_trades: list[dict] = []
        self._mt5 = mt5_module  # optional real MT5 for price feed

    # ── BrokerBase implementation ─────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Paper balance: start at 10,000 USD + simulated PnL."""
        # In a real implementation, track cumulative PnL
        return 10_000.0  # TODO: compound from closed trade P&L

    def get_current_bid(self, symbol: str) -> float:
        if self._mt5 is not None:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick:
                return float(tick.bid)
        raise RuntimeError("No MT5 price feed available for paper broker")

    def place_order(self, order: OrderSpec) -> Optional[str]:
        ticket = str(uuid.uuid4())[:8].upper()
        pos = _PaperPosition(
            ticket=ticket,
            direction=order.direction,
            lots=order.lot_size,
            open_price=order.entry_price,
            sl=order.stop_price,
            tp=order.target_price,
            open_time=pd.Timestamp.now(tz=TZ_IST),
            symbol=order.signal.level_name,  # just metadata
        )
        self._positions[ticket] = pos
        log.info(
            "[PAPER] Order placed: %s %s %.2fL @ %.5f SL=%.5f TP=%.5f ticket=%s",
            order.direction.name, self.cfg.symbol,
            order.lot_size, order.entry_price,
            order.stop_price, order.target_price, ticket,
        )
        return ticket

    def close_position(self, ticket: str) -> bool:
        pos = self._positions.pop(ticket, None)
        if pos is None:
            log.warning("[PAPER] Tried to close unknown ticket: %s", ticket)
            return False
        try:
            bid = self.get_current_bid(self.cfg.symbol)
            if pos.direction == Direction.LONG:
                exit_price = bid  # sell at bid
            else:
                exit_price = bid + self.cfg.spread_pips * PIP  # buy at ask
        except Exception:
            exit_price = pos.open_price  # fallback

        log.info("[PAPER] Closed %s @ %.5f (was @ %.5f)", ticket, exit_price, pos.open_price)
        self._closed_trades.append({
            "ticket": ticket,
            "direction": pos.direction.name,
            "lots": pos.lots,
            "open_price": pos.open_price,
            "close_price": exit_price,
            "sl": pos.sl,
            "tp": pos.tp,
        })
        return True

    def get_open_positions(self, symbol: str) -> list[dict]:
        return [
            {
                "ticket": p.ticket,
                "direction": p.direction.name,
                "lots": p.lots,
                "open_price": p.open_price,
                "sl": p.sl,
                "tp": p.tp,
                "open_time": p.open_time,
            }
            for p in self._positions.values()
        ]

    def get_m1_bars(self, symbol: str, count: int) -> pd.DataFrame:
        if self._mt5 is None:
            raise RuntimeError("No MT5 module available")
        from sweepbot.config import TZ_UTC
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

    def check_sl_tp(self, symbol: str) -> list[str]:
        """
        Check all open positions against current price. Close any that have
        hit SL or TP. Returns list of tickets closed.
        Used in the live runner loop between M15 bars.
        """
        try:
            bid = self.get_current_bid(symbol)
        except Exception:
            return []

        closed = []
        for ticket, pos in list(self._positions.items()):
            if pos.direction == Direction.LONG:
                if bid <= pos.sl or bid >= pos.tp:
                    self.close_position(ticket)
                    closed.append(ticket)
            else:
                ask = bid + self.cfg.spread_pips * PIP
                if ask >= pos.sl or ask <= pos.tp:
                    self.close_position(ticket)
                    closed.append(ticket)
        return closed
