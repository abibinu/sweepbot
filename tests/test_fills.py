"""
tests/test_fills.py
-------------------
Unit tests for engine/fills.py (M1 fill simulator).
Verifies:
  - Long & Short target fills (+2.0R gross)
  - Long & Short stop fills (-1.0R gross)
  - Same-bar touch of stop and target counts as STOP (SDD §2.8)
  - Time exit at 21:30 IST / 22:30 IST (market close)
  - Commission deduction in R
  - Early vs Late entry tagging (19:00 IST summer / 20:00 IST winter threshold)
"""
import datetime as dt
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.config import CFG, PIP
from sweepbot.engine.fills import ExitReason, TradeResult, simulate_fill
from sweepbot.risk import OrderSpec
from sweepbot.strategy import Direction, Signal

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")


def _make_order(
    direction: Direction,
    entry_price: float,
    stop_price: float,
    target_price: float,
    entry_bar_ist: pd.Timestamp,
    risk_pips: float = 10.0,
    lot_size: float = 0.50,
    account_balance: float = 10_000.0,
    risk_usd: float = 50.0,
    comm_usd: float = 1.50,
) -> OrderSpec:
    sig = Signal(
        direction=direction,
        level_name="PDL",
        level_price=1.0900,
        pattern="one_candle",
        extreme=1.0890,
        entry_bar_time=entry_bar_ist,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_pips=risk_pips,
    )
    return OrderSpec(
        signal=sig,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_pips=risk_pips,
        lot_size=lot_size,
        account_balance=account_balance,
        risk_amount_usd=risk_usd,
        commission_usd=comm_usd,
    )


def _make_m1_df(bars: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for ts_str, o, h, l, c in bars:
        rows.append({
            "timestamp": pd.Timestamp(ts_str, tz=TZ_UTC),
            "open": o, "high": h, "low": l, "close": c,
            "tick_volume": 50,
        })
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


class TestSimulateFillLong:
    def test_target_hit(self):
        """Long order hits target price."""
        trade_date = dt.date(2025, 6, 10)
        entry_time_ist = pd.Timestamp("2025-06-10 17:45:00", tz=TZ_IST)
        # Entry at 1.0900, Stop at 1.0890 (risk 10 pips = 0.0010), Target at 1.0920 (+2R)
        order = _make_order(
            direction=Direction.LONG,
            entry_price=1.0900,
            stop_price=1.0890,
            target_price=1.0920,
            entry_bar_ist=entry_time_ist,
            risk_pips=10.0,
            risk_usd=50.0,
            comm_usd=1.50,  # 1.50 / 50 = 0.03R
        )

        entry_utc = entry_time_ist.tz_convert(TZ_UTC)
        bars = [
            (str(entry_utc), 1.0900, 1.0910, 1.0895, 1.0905),
            (str(entry_utc + pd.Timedelta(minutes=1)), 1.0905, 1.0925, 1.0902, 1.0922),
        ]
        df = _make_m1_df(bars)

        res = simulate_fill(order, df, trade_date)
        assert res is not None
        assert res.exit_reason == ExitReason.TARGET
        assert res.exit_price == 1.0920
        assert res.gross_r == pytest.approx(2.0)
        assert res.commission_r == pytest.approx(0.03)
        assert res.net_r == pytest.approx(1.97)
        assert res.is_early_entry is True  # 17:45 < 19:00

    def test_stop_hit(self):
        """Long order hits stop price."""
        trade_date = dt.date(2025, 6, 10)
        entry_time_ist = pd.Timestamp("2025-06-10 17:45:00", tz=TZ_IST)
        order = _make_order(
            direction=Direction.LONG,
            entry_price=1.0900,
            stop_price=1.0890,
            target_price=1.0920,
            entry_bar_ist=entry_time_ist,
        )
        entry_utc = entry_time_ist.tz_convert(TZ_UTC)
        bars = [
            (str(entry_utc), 1.0900, 1.0905, 1.0895, 1.0900),
            (str(entry_utc + pd.Timedelta(minutes=1)), 1.0900, 1.0902, 1.0885, 1.0888),
        ]
        df = _make_m1_df(bars)

        res = simulate_fill(order, df, trade_date)
        assert res is not None
        assert res.exit_reason == ExitReason.STOP
        assert res.exit_price == 1.0890
        assert res.gross_r == pytest.approx(-1.0)

    def test_same_bar_stop_and_target_counts_as_stop(self):
        """SDD §2.8 rule 2: If stop and target are touched in same bar, count the stop."""
        trade_date = dt.date(2025, 6, 10)
        entry_time_ist = pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST)
        order = _make_order(
            direction=Direction.LONG,
            entry_price=1.0900,
            stop_price=1.0890,
            target_price=1.0920,
            entry_bar_ist=entry_time_ist,
        )
        entry_utc = entry_time_ist.tz_convert(TZ_UTC)
        # Bar spans 1.0885 (below stop) AND 1.0925 (above target)
        bars = [
            (str(entry_utc), 1.0900, 1.0925, 1.0885, 1.0910),
        ]
        df = _make_m1_df(bars)

        res = simulate_fill(order, df, trade_date)
        assert res is not None
        assert res.exit_reason == ExitReason.STOP
        assert res.exit_price == 1.0890
        assert res.gross_r == pytest.approx(-1.0)


class TestSimulateFillShort:
    def test_short_target_hit(self):
        trade_date = dt.date(2025, 6, 10)
        entry_time_ist = pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST)
        order = _make_order(
            direction=Direction.SHORT,
            entry_price=1.0900,
            stop_price=1.0910,   # 10 pips above
            target_price=1.0880, # 20 pips below (+2R)
            entry_bar_ist=entry_time_ist,
        )
        entry_utc = entry_time_ist.tz_convert(TZ_UTC)
        bars = [
            (str(entry_utc), 1.0900, 1.0905, 1.0895, 1.0898),
            (str(entry_utc + pd.Timedelta(minutes=1)), 1.0898, 1.0900, 1.0875, 1.0878),
        ]
        df = _make_m1_df(bars)

        res = simulate_fill(order, df, trade_date)
        assert res is not None
        assert res.exit_reason == ExitReason.TARGET
        assert res.exit_price == 1.0880
        assert res.gross_r == pytest.approx(2.0)

    def test_short_stop_hit(self):
        trade_date = dt.date(2025, 6, 10)
        entry_time_ist = pd.Timestamp("2025-06-10 18:00:00", tz=TZ_IST)
        order = _make_order(
            direction=Direction.SHORT,
            entry_price=1.0900,
            stop_price=1.0910,
            target_price=1.0880,
            entry_bar_ist=entry_time_ist,
        )
        entry_utc = entry_time_ist.tz_convert(TZ_UTC)
        bars = [
            (str(entry_utc), 1.0900, 1.0915, 1.0895, 1.0912),
        ]
        df = _make_m1_df(bars)

        res = simulate_fill(order, df, trade_date)
        assert res is not None
        assert res.exit_reason == ExitReason.STOP
        assert res.exit_price == 1.0910
        assert res.gross_r == pytest.approx(-1.0)


class TestTimeExitAndLateEntry:
    def test_time_exit_at_hard_close(self):
        """If neither SL nor TP is hit by 21:30 IST, exit at market close."""
        trade_date = dt.date(2025, 6, 10)
        entry_time_ist = pd.Timestamp("2025-06-10 19:15:00", tz=TZ_IST)
        order = _make_order(
            direction=Direction.LONG,
            entry_price=1.0900,
            stop_price=1.0880,
            target_price=1.0940,
            entry_bar_ist=entry_time_ist,
        )
        # Entry at 19:15 IST >= 19:00 IST -> is_early_entry must be False
        entry_utc = entry_time_ist.tz_convert(TZ_UTC)
        # Create bars up to 21:29 IST (15:59 UTC in summer)
        time_exit_utc = pd.Timestamp("2025-06-10 21:30:00", tz=TZ_IST).tz_convert(TZ_UTC)
        last_bar_utc = time_exit_utc - pd.Timedelta(minutes=1)

        bars = [
            (str(entry_utc), 1.0900, 1.0910, 1.0890, 1.0905),
            (str(last_bar_utc), 1.0905, 1.0915, 1.0900, 1.0910),
        ]
        df = _make_m1_df(bars)

        res = simulate_fill(order, df, trade_date)
        assert res is not None
        assert res.exit_reason == ExitReason.TIME
        # Long market exit sells at bid (last bar close = 1.0910)
        assert res.exit_price == pytest.approx(1.0910)
        assert res.is_early_entry is False  # 19:15 is late entry
        # Gross R = (1.0910 - 1.0900) / 0.0020 = +0.50R
        assert res.gross_r == pytest.approx(0.50)
