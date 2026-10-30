"""
sweepbot/live/notify.py
-----------------------
Telegram notifications for live/paper trading events.

Sends instant updates for:
  - Session startup & armed symbols
  - Order entry / fill confirmation
  - Dynamic breakeven trailing trigger (+1.3R)
  - Trade close (TP, SL, Time Exit, P&L in USD and R)
  - Daily loss kill switch alert
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

from sweepbot.config import Config
from sweepbot.risk import OrderSpec

log = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends trading notifications to a Telegram chat."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def send_message(self, text: str) -> bool:
        if not self._enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "SweepBot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                res = json.loads(resp.read().decode())
                return bool(res.get("ok"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            log.warning("[TELEGRAM] HTTP Error %d: %s", e.code, err_body)
            return False
        except Exception as e:
            log.warning("[TELEGRAM] Notification failed: %s", e)
            return False

    def notify_startup(
        self,
        date_str: str,
        symbols: list[str],
        equity: float,
        kill_threshold: float,
        mode: str,
    ) -> None:
        msg = (
            f"<b>SweepBot Session Started</b>\n"
            f"<b>Date:</b> {date_str}\n"
            f"<b>Mode:</b> {mode.upper()}\n"
            f"<b>Symbols:</b> {', '.join(symbols)}\n"
            f"<b>Starting Equity:</b> ${equity:,.2f}\n"
            f"<b>Kill Switch (-1.5%):</b> ${kill_threshold:,.2f}"
        )
        self.send_message(msg)

    def notify_order(self, order: OrderSpec, ticket: str) -> None:
        direction = "BUY" if order.direction.name == "LONG" else "SELL"
        msg = (
            f"<b>SweepBot Order Placed</b>\n"
            f"<b>Symbol:</b> {order.symbol}\n"
            f"<b>Direction:</b> {direction}\n"
            f"<b>Lot Size:</b> {order.lot_size:.2f}L\n"
            f"<b>Entry Price:</b> {order.entry_price:.5f}\n"
            f"<b>Stop Loss:</b> {order.stop_price:.5f} ({order.risk_pips:.1f} pips)\n"
            f"<b>Take Profit:</b> {order.target_price:.5f}\n"
            f"<b>Risk Amount:</b> ${order.risk_amount_usd:,.2f}\n"
            f"<b>Ticket:</b> {ticket}"
        )
        self.send_message(msg)

    def notify_breakeven(self, symbol: str, ticket: str, new_sl: float) -> None:
        msg = (
            f"<b>Dynamic Breakeven Trailed</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Ticket:</b> {ticket}\n"
            f"<b>New SL:</b> {new_sl:.5f} (+1.3R reached, risk removed)"
        )
        self.send_message(msg)

    def notify_close(
        self,
        ticket: str,
        symbol: str,
        reason: str,
        net_r: float,
        pnl_usd: float,
        exit_price: float,
    ) -> None:
        outcome = "WIN" if pnl_usd > 0 else ("SCRATCH" if abs(pnl_usd) < 1.0 else "LOSS")
        msg = (
            f"<b>SweepBot Position Closed [{outcome}]</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Ticket:</b> {ticket}\n"
            f"<b>Reason:</b> {reason}\n"
            f"<b>Exit Price:</b> {exit_price:.5f}\n"
            f"<b>P&L:</b> {pnl_usd:+.2f} USD ({net_r:+.2f}R)"
        )
        self.send_message(msg)

    def notify_kill_switch(self, current_balance: float, threshold: float) -> None:
        msg = (
            f"<b>KILL SWITCH TRIGGERED</b>\n"
            f"Current Balance: ${current_balance:,.2f} <= ${threshold:,.2f}\n"
            f"All positions closed. Trading halted."
        )
        self.send_message(msg)
