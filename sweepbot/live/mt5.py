"""
sweepbot/live/mt5.py
---------------------
Real MetaTrader5 broker adapter with State Recovery (SDD §9 CRITICAL).

State Recovery:
  Upon init or reconnect, queries MT5 for open EURUSD positions.
  If found, reconstructs SL/TP state and SKIPS new entries to prevent
  double-firing after a laptop restart or network drop.

Runner loop (called externally from cli.py):
  - At 17:25 IST: load levels, arm state machine.
  - Every 15 min: call on_m15_close, process signals.
  - At 21:30 IST: hard close all open positions.
  - Daily loss kill: if equity drops > 1.5%, stop all trading.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional

import pandas as pd

from sweepbot.config import CFG, PIP, TZ_IST, TZ_UTC
from sweepbot.live.broker_base import BrokerBase
from sweepbot.risk import OrderSpec, pip_value_usd_per_lot
from sweepbot.strategy import Direction

log = logging.getLogger(__name__)


class MT5Broker(BrokerBase):
    """
    Live MetaTrader5 broker adapter.

    Usage:
        broker = MT5Broker(cfg=CFG)
        broker.connect()
        # … use via live runner …
        broker.disconnect()
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self._mt5 = None
        self._connected = False
        # State recovery: positions reconstructed on init
        self._recovered_tickets: set[str] = set()
        self._active_orders: dict[str, OrderSpec] = {}
        self._breakeven_active: set[str] = set()

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize MT5 connection. Returns True if successful."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError as e:
            raise ImportError("MetaTrader5 package not installed") from e

        kwargs = {}
        if self.cfg.mt5_login:
            kwargs["login"] = self.cfg.mt5_login
        if self.cfg.mt5_password:
            kwargs["password"] = self.cfg.mt5_password
        if self.cfg.mt5_server:
            kwargs["server"] = self.cfg.mt5_server

        if not self._mt5.initialize(**kwargs):
            log.error("MT5 initialize() failed: %s", self._mt5.last_error())
            return False

        self._connected = True
        terminal_info = self._mt5.terminal_info()
        log.info("MT5 connected: %s", terminal_info)
        if terminal_info and not terminal_info.trade_allowed:
            log.warning(
                "\n====================================================================\n"
                "[ALGO TRADING DISABLED] Automated trading is turned OFF in MT5 terminal!\n"
                "Orders cannot be sent until you click 'Algo Trading' (Ctrl+E) in MT5.\n"
                "===================================================================="
            )

        # ── State Recovery (SDD §9 CRITICAL) ────────────────────────────────
        self._recover_state()
        return True

    def disconnect(self) -> None:
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            log.info("MT5 disconnected")

    def _require_connection(self) -> None:
        if not self._connected or self._mt5 is None:
            raise RuntimeError("MT5 not connected. Call connect() first.")

    # ── State Recovery ────────────────────────────────────────────────────────

    def _recover_state(self, symbol: Optional[str] = None) -> None:
        """
        SDD §9 State Recovery:
        Query existing open positions. If any are found,
        record their tickets so the runner skips new entry signals
        to prevent double-firing after restart.
        """
        positions = self.get_open_positions(symbol)
        if positions:
            log.warning(
                "[STATE RECOVERY] Found %d open position(s) — "
                "new entries will be BLOCKED until all positions are closed.",
                len(positions),
            )
            for pos in positions:
                ticket = str(pos["ticket"])
                self._recovered_tickets.add(ticket)
                log.warning(
                    "  Recovered: ticket=%s sym=%s dir=%s lots=%.2f SL=%.5f TP=%.5f",
                    ticket, pos.get("symbol", self.cfg.symbol), pos["direction"],
                    pos["lots"], pos["sl"], pos["tp"],
                )
        else:
            log.info("[STATE RECOVERY] No open positions found — clean start.")

    @property
    def has_recovered_positions(self) -> bool:
        """True if there are unresolved positions from state recovery."""
        open_tickets = {str(p["ticket"]) for p in self.get_open_positions()}
        return bool(self._recovered_tickets & open_tickets)

    def has_recovered_positions_for(self, symbol: Optional[str] = None) -> bool:
        """True if there are unresolved positions for a specific symbol."""
        open_tickets = {str(p["ticket"]) for p in self.get_open_positions(symbol)}
        return bool(self._recovered_tickets & open_tickets)

    # ── BrokerBase implementation ─────────────────────────────────────────────

    def get_account_balance(self) -> float:
        self._require_connection()
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError(f"MT5 account_info() failed: {self._mt5.last_error()}")
        return float(info.equity)

    def get_current_bid(self, symbol: str) -> float:
        self._require_connection()
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol}: {self._mt5.last_error()}")
        return float(tick.bid)

    def get_ask(self, symbol: str) -> float:
        self._require_connection()
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol}: {self._mt5.last_error()}")
        return float(tick.ask)

    def place_order(self, order: OrderSpec) -> Optional[str]:
        self._require_connection()
        symbol = getattr(order, "symbol", self.cfg.symbol)

        # Block if state recovery has unresolved positions for this symbol
        if self.has_recovered_positions_for(symbol):
            log.warning(
                "[SAFETY] Blocking new order on %s — recovered position still open. "
                "Order: %s", symbol, order
            )
            return None

        mt5 = self._mt5

        # Determine order type
        if order.direction == Direction.LONG:
            order_type = mt5.ORDER_TYPE_BUY
            price = self.get_ask(symbol)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = self.get_current_bid(symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": order.lot_size,
            "type": order_type,
            "price": price,
            "sl": order.stop_price,
            "tp": order.target_price,
            "deviation": 20,           # max slippage in points
            "magic": 20261026,         # magic number for this bot
            "comment": f"sweepbot {order.signal.level_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                "MT5 order_send failed: %s | request=%s",
                getattr(result, "comment", mt5.last_error()),
                request,
            )
            return None

        ticket = str(result.order)
        self._active_orders[ticket] = order
        log.info(
            "[MT5] Order filled: ticket=%s %s %s %.2fL @ %.5f SL=%.5f TP=%.5f",
            ticket, symbol, order.direction.name, order.lot_size,
            result.price, order.stop_price, order.target_price,
        )
        return ticket

    def close_position(self, ticket: str) -> bool:
        self._require_connection()
        mt5 = self._mt5

        positions = mt5.positions_get(ticket=int(ticket))
        if not positions:
            log.warning("No position found for ticket %s", ticket)
            return False

        pos = positions[0]
        pos_sym = getattr(pos, "symbol", None)
        symbol = pos_sym if isinstance(pos_sym, str) else self.cfg.symbol

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = self.get_current_bid(symbol)
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = self.get_ask(symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 20261026,
            "comment": "sweepbot time-exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                "MT5 close_position failed: ticket=%s %s",
                ticket, getattr(result, "comment", mt5.last_error()),
            )
            return False

        log.info("[MT5] Closed position ticket=%s (%s) @ %.5f", ticket, symbol, result.price)
        self._recovered_tickets.discard(ticket)
        return True

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict]:
        self._require_connection()
        mt5 = self._mt5
        if symbol is not None:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            pos_sym = getattr(p, "symbol", None)
            sym_str = pos_sym if isinstance(pos_sym, str) else (symbol or self.cfg.symbol)
            result.append({
                "ticket": str(p.ticket),
                "symbol": sym_str,
                "direction": "LONG" if p.type == mt5.ORDER_TYPE_BUY else "SHORT",
                "lots": p.volume,
                "open_price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "open_time": pd.Timestamp(p.time, unit="s", tz=TZ_UTC).tz_convert(TZ_IST),
            })
        return result

    def get_m1_bars(self, symbol: str, count: int) -> pd.DataFrame:
        self._require_connection()
        mt5 = self._mt5
        mt5.symbol_select(symbol, True)

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        retries = 3
        while (rates is None or len(rates) == 0) and retries > 0:
            time.sleep(0.3)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
            retries -= 1

        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename_axis("timestamp")
        df = df[["open", "high", "low", "close", "tick_volume"]]
        return df.sort_index()

    # ── Safety: close all positions ───────────────────────────────────────────

    def close_all_positions(self, symbol: Optional[str] = None, reason: str = "time-exit") -> None:
        """Close all open positions (filtered by symbol if provided)."""
        positions = self.get_open_positions(symbol)
        if not positions:
            log.info("[MT5] No positions to close at %s", reason)
            return
        for pos in positions:
            self.close_position(pos["ticket"])
        log.info("[MT5] All positions closed at %s", reason)

    def check_sl_tp(self, symbol: Optional[str] = None, notifier: Optional[any] = None) -> list[str]:
        """
        Check open MT5 positions for dynamic breakeven (+1.3R) and track closed positions.
        Returns list of closed tickets.
        """
        if not self._connected or self._mt5 is None:
            return []

        mt5 = self._mt5
        open_positions = {str(p["ticket"]): p for p in self.get_open_positions(symbol)}
        closed_tickets = []

        # 1. Detect closed positions
        tracked_tickets = list(self._active_orders.keys())
        for ticket in tracked_tickets:
            if symbol is not None and self._active_orders[ticket].symbol != symbol:
                continue
            if ticket not in open_positions:
                order = self._active_orders.pop(ticket)
                self._breakeven_active.discard(ticket)
                closed_tickets.append(ticket)

                deals = mt5.history_deals_get(position=int(ticket))
                profit_usd = 0.0
                exit_price = order.entry_price
                if deals:
                    exit_deal = deals[-1]
                    profit_usd = getattr(exit_deal, "profit", 0.0)
                    exit_price = getattr(exit_deal, "price", order.entry_price)

                usd_per_pip_lot = pip_value_usd_per_lot(order.symbol, exit_price)
                risk_usd = order.risk_pips * usd_per_pip_lot * order.lot_size
                net_r = (profit_usd - order.commission_usd) / risk_usd if risk_usd > 0 else 0.0

                log.info(
                    "[MT5] Position %s (%s) closed @ %.5f | Profit: %+.2f USD (%+.2fR)",
                    ticket, order.symbol, exit_price, profit_usd, net_r,
                )
                if notifier and hasattr(notifier, "notify_close"):
                    notifier.notify_close(
                        ticket=ticket,
                        symbol=order.symbol,
                        reason="Broker SL/TP Exit",
                        net_r=net_r,
                        pnl_usd=profit_usd - order.commission_usd,
                        exit_price=exit_price,
                    )

        # 2. Dynamic Breakeven Trailing (+1.3R)
        if getattr(self.cfg, "enable_breakeven", True):
            for ticket, order in list(self._active_orders.items()):
                if ticket in self._breakeven_active or ticket not in open_positions:
                    continue
                pos = open_positions[ticket]
                sym = order.symbol
                pip = pip_size(sym)
                trigger_dist = self.cfg.breakeven_trigger_r * order.risk_pips * pip
                buf = self.cfg.breakeven_buffer_pips * pip

                try:
                    bid = self.get_current_bid(sym)
                    ask = self.get_ask(sym)
                except Exception:
                    continue

                new_sl = None
                if order.direction == Direction.LONG:
                    if bid >= order.entry_price + trigger_dist:
                        new_sl = order.entry_price + buf
                else:  # SHORT
                    if ask <= order.entry_price - trigger_dist:
                        new_sl = order.entry_price - buf

                if new_sl is not None:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": int(ticket),
                        "symbol": sym,
                        "sl": new_sl,
                        "tp": pos["tp"],
                    }
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self._breakeven_active.add(ticket)
                        log.info("[BREAKEVEN] MT5 position %s (%s) SL trailed to %.5f", ticket, sym, new_sl)
                        if notifier and hasattr(notifier, "notify_breakeven"):
                            notifier.notify_breakeven(sym, ticket, new_sl)
                    else:
                        log.warning("[BREAKEVEN] Failed to update SL for %s: %s", ticket, getattr(result, "comment", mt5.last_error()))

        return closed_tickets
