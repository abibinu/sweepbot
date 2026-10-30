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


def _make_dummy_order(
    direction=Direction.LONG,
    symbol: str = "EURUSD",
    entry_price: float = 1.0900,
    stop_price: float = 1.0880,
    target_price: float = 1.0940,
    risk_pips: float = 20.0,
    lot_size: float = 0.10,
) -> OrderSpec:
    sig = Signal(
        direction=direction,
        level_name="PDL",
        level_price=entry_price,
        pattern="one_candle",
        extreme=stop_price,
        entry_bar_time=pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST),
    )
    return OrderSpec(
        signal=sig,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_pips=risk_pips,
        lot_size=lot_size,
        account_balance=10_000.0,
        risk_amount_usd=20.0,
        commission_usd=0.30,
        symbol=symbol,
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


class TestPaperBrokerAdvanced:
    def test_balance_compounding_on_win_and_loss(self):
        broker = PaperBroker(cfg=CFG, initial_balance=10_000.0)
        order = _make_dummy_order(
            direction=Direction.LONG,
            symbol="EURUSD",
            entry_price=1.0900,
            stop_price=1.0880,
            target_price=1.0940,
            risk_pips=20.0,
            lot_size=0.10,
        )

        ticket = broker.place_order(order)
        # Commission ($0.30) deducted on open
        assert broker._balance == pytest.approx(9999.70)

        # Close at target (1.0940): +40 pips -> 40 * $10/pip * 0.10 = +$40.00 gross
        broker.close_position(ticket, exit_price=1.0940)
        assert broker._balance == pytest.approx(9999.70 + 40.0)
        assert len(broker._closed_trades) == 1

        tr = broker._closed_trades[0]
        assert tr["gross_pnl_usd"] == pytest.approx(40.0)
        assert tr["net_pnl_usd"] == pytest.approx(39.70)
        assert tr["net_r"] == pytest.approx(39.70 / 20.0, abs=0.01)

    def test_unrealized_pnl_in_equity(self):
        mock_mt5 = MagicMock()
        mock_tick = MagicMock()
        mock_tick.bid = 1.0920  # +20 pips floating profit
        mock_mt5.symbol_info_tick.return_value = mock_tick

        broker = PaperBroker(mt5_module=mock_mt5, cfg=CFG, initial_balance=10_000.0)
        order = _make_dummy_order(
            direction=Direction.LONG,
            symbol="EURUSD",
            entry_price=1.0900,
            stop_price=1.0880,
            target_price=1.0940,
            risk_pips=20.0,
            lot_size=0.10,
        )
        broker.place_order(order)

        # Realized balance is 9999.70
        # Floating unrealized = +20 pips * $10 * 0.10 lots = +$20.00
        # Equity should equal 10019.70
        equity = broker.get_account_balance()
        assert equity == pytest.approx(10_019.70)

    def test_dynamic_breakeven_trailing(self):
        mock_mt5 = MagicMock()
        mock_tick = MagicMock()
        mock_mt5.symbol_info_tick.return_value = mock_tick

        broker = PaperBroker(mt5_module=mock_mt5, cfg=CFG, initial_balance=10_000.0)
        order = _make_dummy_order(
            direction=Direction.LONG,
            symbol="EURUSD",
            entry_price=1.0900,
            stop_price=1.0880,
            target_price=1.0940,
            risk_pips=20.0,
            lot_size=0.10,
        )
        ticket = broker.place_order(order)
        pos = broker._positions[ticket]

        # 1. Price below trigger (+1.3R = 26 pips -> 1.0926)
        mock_tick.bid = 1.0915
        broker.check_sl_tp()
        assert pos.breakeven_active is False
        assert pos.sl == 1.0880

        # 2. Price reaches +1.35R (1.0927) -> breakeven triggered
        mock_tick.bid = 1.0927
        broker.check_sl_tp()
        assert pos.breakeven_active is True
        # SL moved to open + buffer (1.0900 + 0.2 pips = 1.09002)
        assert pos.sl == pytest.approx(1.09002, abs=1e-5)

        # 3. Price drops back to 1.0900 -> SL hit at 1.09002 (near scratch, protected profit)
        mock_tick.bid = 1.0900
        closed = broker.check_sl_tp()
        assert ticket in closed
        assert ticket not in broker._positions
        assert len(broker._closed_trades) == 1
        assert broker._closed_trades[0]["net_r"] >= -0.05  # near breakeven, not -1.0R!

    def test_multi_pair_pip_sizing_jpy(self):
        broker = PaperBroker(cfg=CFG, initial_balance=10_000.0)
        order = _make_dummy_order(
            direction=Direction.LONG,
            symbol="USDJPY",
            entry_price=150.00,
            stop_price=149.80,
            target_price=150.40,
            risk_pips=20.0,
            lot_size=0.10,
        )
        ticket = broker.place_order(order)
        assert broker._positions[ticket].symbol == "USDJPY"

        # Exit at TP 150.40 (+40 pips)
        # JPY pip = 0.01 -> 40 pips = 0.40 / 0.01. USD/pip/lot = 1000 / 150.40 = ~6.6489 USD.
        broker.close_position(ticket, exit_price=150.40)
        tr = broker._closed_trades[0]
        assert tr["gross_pnl_usd"] == pytest.approx(40.0 * (1000.0 / 150.40) * 0.10, abs=0.1)
        assert tr["net_r"] > 1.8  # close to +2R target

    def test_multi_pair_position_tracking(self):
        broker = PaperBroker(cfg=CFG)
        order_eur = _make_dummy_order(symbol="EURUSD", entry_price=1.0900, stop_price=1.0880)
        order_jpy = _make_dummy_order(symbol="USDJPY", entry_price=150.00, stop_price=149.80)

        t_eur = broker.place_order(order_eur)
        t_jpy = broker.place_order(order_jpy)

        all_open = broker.get_open_positions()
        assert len(all_open) == 2

        eur_open = broker.get_open_positions("EURUSD")
        assert len(eur_open) == 1
        assert eur_open[0]["ticket"] == t_eur

        jpy_open = broker.get_open_positions("USDJPY")
        assert len(jpy_open) == 1
        assert jpy_open[0]["ticket"] == t_jpy


class TestPortfolioLiveRunner:
    def test_print_portfolio_dashboard(self):
        from sweepbot.config import Config
        from sweepbot.levels import DayLevels
        from sweepbot.live.runner import print_portfolio_dashboard

        levels_eur = DayLevels(
            date=pd.Timestamp("2026-10-30").date(),
            pdh=1.0950,
            pdl=1.0850,
            asia_h=1.0920,
            asia_l=1.0880,
            lon_h=1.0930,
            lon_l=1.0870,
            asia_bar_count=400,
            lon_bar_count=300,
        )
        levels_map = {"EURUSD": levels_eur}
        cfg_map = {"EURUSD": Config.for_symbol("EURUSD")}

        # Verify running without exceptions
        print_portfolio_dashboard(
            levels_map=levels_map,
            cfg_map=cfg_map,
            start_balance=10_000.0,
            kill_threshold=9850.0,
            trade_date=pd.Timestamp("2026-10-30").date(),
        )

    def test_portfolio_runner_clean_single_bar_execution(self, monkeypatch):
        from sweepbot.levels import DayLevels
        from sweepbot.live.runner import run_live

        mock_broker = MagicMock()
        mock_broker.get_account_balance.return_value = 10_000.0

        levels_valid = DayLevels(
            date=pd.Timestamp("2026-10-30").date(),
            pdh=1.0950,
            pdl=1.0850,
            asia_h=1.0920,
            asia_l=1.0880,
            lon_h=1.0930,
            lon_l=1.0870,
            asia_bar_count=400,
            lon_bar_count=300,
        )
        monkeypatch.setattr("sweepbot.live.runner.compute_levels", lambda m1, d: levels_valid)

        # Mock M1 bars for M15 processing
        idx_recent = pd.date_range("2026-10-30 11:30:00", periods=60, freq="1min", tz="UTC")
        df_recent = pd.DataFrame({
            "open": [1.0900] * 60,
            "high": [1.0910] * 60,
            "low": [1.0890] * 60,
            "close": [1.0905] * 60,
            "tick_volume": [50] * 60,
        }, index=idx_recent)
        df_recent.index.name = "timestamp"

        mock_broker.get_m1_bars.return_value = df_recent
        mock_broker.get_open_positions.return_value = []

        # Run with max_bars=1, dry_run=True, wait_for_arm=False
        run_live(
            broker=mock_broker,
            symbols=["EURUSD"],
            dry_run=True,
            wait_for_arm=False,
            max_bars=1,
            skip_bar_wait=True,
            poll_interval_sec=0.01,
        )

        assert mock_broker.get_account_balance.called

    def test_portfolio_runner_kill_switch(self, monkeypatch):
        from sweepbot.levels import DayLevels
        from sweepbot.live.runner import run_live

        mock_broker = MagicMock()
        # Initial balance 10,000 -> kill threshold 9850; then balance drops to 9800 -> kill switch triggers
        mock_broker.get_account_balance.side_effect = [10_000.0, 9800.0]

        levels_valid = DayLevels(
            date=pd.Timestamp("2026-10-30").date(),
            pdh=1.0950,
            pdl=1.0850,
            asia_h=1.0920,
            asia_l=1.0880,
            lon_h=1.0930,
            lon_l=1.0870,
            asia_bar_count=400,
            lon_bar_count=300,
        )
        monkeypatch.setattr("sweepbot.live.runner.compute_levels", lambda m1, d: levels_valid)

        idx = pd.date_range("2026-10-28 00:00:00", periods=2880, freq="1min", tz="UTC")
        df_hist = pd.DataFrame({
            "open": [1.0900] * 2880,
            "high": [1.0950] * 2880,
            "low": [1.0850] * 2880,
            "close": [1.0900] * 2880,
            "tick_volume": [100] * 2880,
        }, index=idx)
        mock_broker.get_m1_bars.return_value = df_hist
        mock_broker.get_open_positions.return_value = []

        run_live(
            broker=mock_broker,
            symbols=["EURUSD"],
            dry_run=True,
            wait_for_arm=False,
            max_bars=1,
            skip_bar_wait=True,
            poll_interval_sec=0.01,
        )

        # close_all_positions should be called when kill switch triggers
        mock_broker.close_all_positions.assert_called_once()

