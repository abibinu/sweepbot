"""
sweepbot/live/broker_base.py
------------------------------
Abstract base class for live/paper broker adapters.

Both PaperBroker (paper.py) and MT5Broker (mt5.py) implement this interface,
allowing the live runner to be broker-agnostic.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from sweepbot.risk import OrderSpec
from sweepbot.strategy import Direction


class BrokerBase(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def get_account_balance(self) -> float:
        """Return current account equity in USD."""
        ...

    @abstractmethod
    def get_current_bid(self, symbol: str) -> float:
        """Return the current bid price for *symbol*."""
        ...

    @abstractmethod
    def place_order(self, order: OrderSpec) -> Optional[str]:
        """
        Place a market order with SL and TP.

        Returns a broker order/ticket ID string, or None if rejected.
        """
        ...

    @abstractmethod
    def close_position(self, ticket: str) -> bool:
        """Close an open position by ticket. Returns True if successful."""
        ...

    @abstractmethod
    def get_open_positions(self, symbol: str) -> list[dict]:
        """
        Return open positions for *symbol* as list of dicts with keys:
            ticket, direction, lots, open_price, sl, tp, open_time
        """
        ...

    @abstractmethod
    def get_m1_bars(self, symbol: str, count: int) -> pd.DataFrame:
        """
        Return the last *count* M1 bars as UTC-indexed DataFrame.
        Columns: open, high, low, close, tick_volume
        """
        ...

    def get_ask(self, symbol: str) -> float:
        """Default: ask = bid + spread. Adapters may override."""
        from sweepbot.config import CFG, PIP
        return self.get_current_bid(symbol) + CFG.spread_pips * PIP
