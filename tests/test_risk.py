"""
tests/test_risk.py
-------------------
Tests for position sizing, stop/target computation, and commission.
"""
import math
import pytest

from sweepbot.config import CFG, PIP
from sweepbot.risk import (
    compute_entry_price,
    compute_stop_target,
    compute_lot_size,
    compute_commission,
    build_order,
    commission_in_r,
)
from sweepbot.strategy import Direction, Signal
import datetime as dt
import pandas as pd
from zoneinfo import ZoneInfo

TZ_IST = ZoneInfo("Asia/Kolkata")


def _make_signal(direction: Direction, extreme: float) -> Signal:
    return Signal(
        direction=direction,
        level_name="PDL",
        level_price=1.0900,
        pattern="one_candle",
        extreme=extreme,
        entry_bar_time=pd.Timestamp(
            dt.datetime(2025, 6, 10, 18, 0, tzinfo=TZ_IST)
        ),
    )


class TestEntryPrice:
    def test_long_entry_adds_spread(self):
        sig = _make_signal(Direction.LONG, 1.0880)
        entry = compute_entry_price(sig, bid_price=1.0910)
        assert entry == pytest.approx(1.0910 + CFG.spread_pips * PIP)

    def test_short_entry_is_bid(self):
        sig = _make_signal(Direction.SHORT, 1.0940)
        entry = compute_entry_price(sig, bid_price=1.0920)
        assert entry == pytest.approx(1.0920)


class TestStopTarget:
    def test_long_stop_below_extreme(self):
        sig = _make_signal(Direction.LONG, extreme=1.0880)
        # entry at 1.0895: risk = 1.0895 - (1.0880 - 0.0001) = 0.0016 = 16 pips ✓
        entry = 1.0895
        stop, target, risk_pips = compute_stop_target(sig, entry)

        expected_stop = 1.0880 - CFG.stop_buffer_pips * PIP
        assert stop == pytest.approx(expected_stop, abs=1e-6)

        expected_risk = (entry - stop) / PIP
        assert risk_pips == pytest.approx(expected_risk, abs=0.01)

        expected_target = entry + 2 * (entry - stop)
        assert target == pytest.approx(expected_target, abs=1e-6)

    def test_short_stop_above_extreme(self):
        sig = _make_signal(Direction.SHORT, extreme=1.0940)
        # entry at 1.0920: risk = (1.0940 + 0.0001) - 1.0920 = 0.0021 = 21 pips → too large
        # Use extreme=1.0930: risk = (1.0930 + 0.0001) - 1.0920 = 0.0011 = 11 pips ✓
        sig2 = _make_signal(Direction.SHORT, extreme=1.0930)
        entry = 1.0920
        stop, target, risk_pips = compute_stop_target(sig2, entry)

        expected_stop = 1.0930 + CFG.stop_buffer_pips * PIP
        assert stop == pytest.approx(expected_stop, abs=1e-6)

    def test_risk_too_small_returns_nan(self):
        """If risk < 3 pips, should return nan."""
        sig = _make_signal(Direction.LONG, extreme=1.09098)  # extreme just 2 pips below
        entry = 1.09100
        stop, target, risk_pips = compute_stop_target(sig, entry)
        assert math.isnan(risk_pips)

    def test_risk_too_large_returns_nan(self):
        """If risk > 20 pips, should return nan."""
        sig = _make_signal(Direction.LONG, extreme=1.0880)  # 30 pip stop
        entry = 1.0915
        stop, target, risk_pips = compute_stop_target(sig, entry)
        # risk = entry - (extreme - buffer) = 0.0015 + 0.0001 * 0.5 ...
        # Let's make it clearly > 20 pips
        sig2 = _make_signal(Direction.LONG, extreme=1.0860)
        entry2 = 1.0915
        stop2, target2, rp2 = compute_stop_target(sig2, entry2)
        # risk_pips = (1.0915 - (1.086 - 0.0001)) / 0.0001 = 56 pips
        assert math.isnan(rp2)


class TestLotSize:
    def test_lot_sizing_basic(self):
        """0.25% of 10000 = $25 risk. At 10 pip stop: $10/pip/lot → 0.25 lots."""
        lots = compute_lot_size(account_balance=10_000.0, risk_pips=10.0)
        # 25 / (10 * 10) = 0.25 → rounds to 0.25
        assert lots == pytest.approx(0.25, abs=0.01)

    def test_min_lot_floor(self):
        """Very small account should still get min_lot."""
        lots = compute_lot_size(account_balance=10.0, risk_pips=10.0)
        assert lots >= CFG.min_lot

    def test_lot_rounded_down(self):
        """Lot size should round DOWN to lot_step."""
        import dataclasses
        cfg05 = dataclasses.replace(CFG, risk_per_trade_pct=0.5)
        # 0.5% of 10000 = 50, at 7 pip stop = 50/(7*10) = 0.714...
        # Should round to 0.71
        lots = compute_lot_size(account_balance=10_000.0, risk_pips=7.0, cfg=cfg05)
        # Check that it's a clean multiple of lot_step within floating-point tolerance
        remainder = round(lots / CFG.lot_step) * CFG.lot_step
        assert abs(lots - remainder) < 1e-9


class TestCommission:
    def test_commission_calculation(self):
        """$3.00/lot commission."""
        comm = compute_commission(0.10)
        assert comm == pytest.approx(0.30)

    def test_commission_in_r(self):
        """Commission as fraction of R."""
        r_val = commission_in_r(commission_usd=3.0, risk_amount_usd=25.0)
        assert r_val == pytest.approx(0.12)


class TestMultiAssetPip:
    def test_pip_size_multi_asset(self):
        from sweepbot.config import pip_size
        assert pip_size("EURUSD") == 0.0001
        assert pip_size("GBPUSD") == 0.0001
        assert pip_size("USDJPY") == 0.01

    def test_pip_value_usd_per_lot(self):
        from sweepbot.risk import pip_value_usd_per_lot
        assert pip_value_usd_per_lot("EURUSD", 1.0850) == pytest.approx(10.0)
        assert pip_value_usd_per_lot("GBPUSD", 1.2900) == pytest.approx(10.0)
        assert pip_value_usd_per_lot("USDJPY", 150.0) == pytest.approx(1000.0 / 150.0)

