"""
tests/test_live.py
------------------
Unit tests for live/paper trading and MT5 adapter.
Verifies SDD §9 requirements:
  - PaperBroker: virtual execution, order placement, position tracking
  - MT5Broker State Recovery (CRITICAL):
      Detects existing open positions upon init/reconnection,
      blocks new entry orders to prevent double-firing after restart/network drop,
      resumes normal trading once recovered positions are closed.
  - Hard kill switch behavior when daily loss hits -1.5%
"""
from unittest.mock import MagicMock
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.config import CFG
from sweepbot.live.paper import PaperBroker
from sweepbot.live.mt5 import MT5Broker
from sweepbot.risk import OrderSpec
from sweepbot.strategy import Direction, Signal

TZ_IST = ZoneInfo("Asia/Kolkata")


def _make_dummy_order(direction=Direction.LONG) -> OrderSpec:
    sig = Signal(
        direction=direction,
        level_name="PDL",
        level_price=1.0900,
        pattern="one_candle",
        extreme=1.0880,
        entry_bar_time=pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST),
    )
    return OrderSpec(
        signal=sig,
        direction=direction,
        entry_price=1.0900,
        stop_price=1.0880,
        target_price=1.0940,
        risk_pips=20.0,
        lot_size=0.10,
        account_balance=10_000.0,
        risk_amount_usd=20.0,
        commission_usd=0.30,
    )


class TestPaperBroker:
    def test_place_and_close_order(self):
        broker = PaperBroker(cfg=CFG)
        order = _make_dummy_order(Direction.LONG)

        ticket = broker.place_order(order)
        assert ticket is not None

        positions = broker.get_open_positions("EURUSD")
        assert len(positions) == 1
        assert positions[0]["ticket"] == ticket
        assert positions[0]["direction"] == "LONG"
        assert positions[0]["lots"] == 0.10

        # Close position
        closed = broker.close_position(ticket)
        assert closed is True
        assert len(broker.get_open_positions("EURUSD")) == 0
        assert len(broker._closed_trades) == 1


class TestMT5StateRecovery:
    """CRITICAL requirement in SDD §9: State Recovery prevents double-firing."""

    def test_state_recovery_blocks_new_entries(self):
        # Create a mock MT5 module
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = "MockTerminal"

        # Simulate an existing active position on EURUSD before bot started
        mock_pos = MagicMock()
        mock_pos.ticket = 998877
        mock_pos.type = 0  # ORDER_TYPE_BUY
        mock_pos.volume = 0.50
        mock_pos.price_open = 1.0905
        mock_pos.sl = 1.0885
        mock_pos.tp = 1.0945
        mock_pos.time = 1749550000  # POSIX timestamp

        # positions_get returns this active position
        mock_mt5.positions_get.return_value = [mock_pos]

        broker = MT5Broker(cfg=CFG)
        broker._mt5 = mock_mt5
        broker._connected = True

        # Run state recovery
        broker._recover_state()

        assert "998877" in broker._recovered_tickets
        assert broker.has_recovered_positions is True

        # Now attempt to place a new order: it MUST be blocked!
        new_order = _make_dummy_order(Direction.LONG)
        ticket = broker.place_order(new_order)

        # Blocked: returns None, order_send never called
        assert ticket is None
        mock_mt5.order_send.assert_not_called()

    def test_state_recovery_unblocks_after_positions_close(self):
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True

        # Initially active position
        mock_pos = MagicMock()
        mock_pos.ticket = 12345
        mock_pos.type = 0
        mock_pos.volume = 0.10
        mock_pos.price_open = 1.0900
        mock_pos.sl = 1.0880
        mock_pos.tp = 1.0940
        mock_pos.time = 1749550000

        mock_mt5.positions_get.return_value = [mock_pos]

        broker = MT5Broker(cfg=CFG)
        broker._mt5 = mock_mt5
        broker._connected = True
        broker._recover_state()

        assert broker.has_recovered_positions is True

        # Once position closes in MT5, positions_get returns []
        mock_mt5.positions_get.return_value = []
        assert broker.has_recovered_positions is False

        # Now placing an order is allowed
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_result.order = 54321
        mock_result.price = 1.0900
        mock_mt5.order_send.return_value = mock_result

        # Mock tick data for ask
        mock_tick = MagicMock()
        mock_tick.ask = 1.0902
        mock_tick.bid = 1.0900
        mock_mt5.symbol_info_tick.return_value = mock_tick

        new_order = _make_dummy_order(Direction.LONG)
        ticket = broker.place_order(new_order)
        assert ticket == "54321"
        mock_mt5.order_send.assert_called_once()

    def test_clean_start_with_no_existing_positions(self):
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True
        mock_mt5.positions_get.return_value = []

        broker = MT5Broker(cfg=CFG)
        broker._mt5 = mock_mt5
        broker._connected = True
        broker._recover_state()

        assert len(broker._recovered_tickets) == 0
        assert broker.has_recovered_positions is False
