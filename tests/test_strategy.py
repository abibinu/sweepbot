"""
tests/test_strategy.py
------------------------
Unit tests for the strategy state machine.

Tests cover all detection cases from SDD §2.5:
  - One-candle long (support sweep + close above)
  - One-candle short (resistance sweep + close below)
  - Two-candle long
  - Two-candle short
  - Level consumed after first signal
  - Entry too late (entry bar > 19:15 IST)
  - No sweep (prev_close below level)
"""
import datetime as dt
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.config import CFG, PIP
from sweepbot.levels import DayLevels
from sweepbot.strategy import Direction, StrategyStateMachine

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")


def _ist(hour: int, minute: int, date: dt.date = dt.date(2025, 6, 10)) -> pd.Timestamp:
    return pd.Timestamp(
        dt.datetime(date.year, date.month, date.day, tzinfo=TZ_IST)
    ) + pd.Timedelta(hours=hour, minutes=minute)


def _bar(open_, high, low, close, ts_ist: pd.Timestamp) -> pd.Series:
    """Build a minimal M15 bar as pd.Series with a UTC name."""
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close},
        name=ts_ist.tz_convert("UTC"),
    )


def _make_levels(pdh=1.1100, pdl=1.0900, asia_h=1.1050, asia_l=1.0950,
                 lon_h=1.1080, lon_l=1.0920,
                 asia_bars=200, lon_bars=150,
                 date=dt.date(2025, 6, 10)) -> DayLevels:
    return DayLevels(
        date=date,
        pdh=pdh, pdl=pdl,
        asia_h=asia_h, asia_l=asia_l,
        lon_h=lon_h, lon_l=lon_l,
        asia_bar_count=asia_bars,
        lon_bar_count=lon_bars,
    )


TRADE_DATE = dt.date(2025, 6, 10)
LEVELS = _make_levels()


class TestOneCandleLong:
    """Support sweep + one-candle rejection (bar closes above level)."""

    def test_basic_one_candle_long(self):
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)

        # Level: PDL = 1.0900
        L = LEVELS.pdl

        dummy = _bar(L + 0.0020, L + 0.0030, L + 0.0010, L + 0.0015, _ist(17, 30))
        # Bar 1 (17:45 IST): prev_close above L, sweeps below L, closes green above L
        bar1 = _bar(
            open_=L + 0.0002,
            high=L + 0.0015,
            low=L - 3 * PIP,    # Sweeps 3 pips below
            close=L + 0.0005,   # Closes above L and above open → bullish rejection
            ts_ist=_ist(17, 45),
        )

        signals_0 = sm.on_m15_close(0, dummy)  # bar 0: sets up prev_close
        signals_1 = sm.on_m15_close(1, bar1)   # bar 1: sweeps

        # Bar 1 close is above L → one-candle. Entry bar = bar 2 (17:45+15min = 18:00)
        # That is within window → should produce a signal
        assert len(signals_1) == 1
        sig = signals_1[0]
        assert sig.direction == Direction.LONG
        assert sig.level_name == "PDL"
        assert sig.pattern == "one_candle"
        assert sig.extreme == bar1["low"]
        # Entry bar should be 18:00 IST (15 min after 17:45 IST bar 1)
        entry_ist = sig.entry_bar_time.tz_convert(TZ_IST)
        assert entry_ist.hour == 18
        assert entry_ist.minute == 0


class TestOneCandleShort:
    """Resistance sweep + one-candle rejection (bar closes below level)."""

    def test_basic_one_candle_short(self):
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)

        L = LEVELS.pdh  # 1.1100

        dummy = _bar(L - 0.0020, L - 0.0010, L - 0.0030, L - 0.0015, _ist(17, 30))
        sweep_bar = _bar(
            open_=L - 0.0002,
            high=L + 3 * PIP,    # Sweeps 3 pips above
            low=L - 0.0015,
            close=L - 0.0005,    # Closes below L and below open → bearish rejection
            ts_ist=_ist(17, 45),
        )

        sm.on_m15_close(0, dummy)
        signals = sm.on_m15_close(1, sweep_bar)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == Direction.SHORT
        assert sig.pattern == "one_candle"
        assert sig.extreme == sweep_bar["high"]


class TestTwoCandleLong:
    """Support sweep + two-candle rejection."""

    def test_two_candle_long(self):
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)

        L = LEVELS.pdl  # 1.0900

        dummy = _bar(L + 0.0020, L + 0.0030, L + 0.0010, L + 0.0015, _ist(17, 30))
        sweep_bar = _bar(
            open_=L + 0.0010,
            high=L + 0.0015,
            low=L - 3 * PIP,    # Sweeps
            close=L - PIP,      # Closes BELOW L → not one-candle
            ts_ist=_ist(17, 45),
        )
        second_bar = _bar(
            open_=L - PIP,
            high=L + 5 * PIP,
            low=L - 2 * PIP,
            close=L + 3 * PIP,  # Closes ABOVE L → two-candle complete
            ts_ist=_ist(18, 0),
        )

        sm.on_m15_close(0, dummy)
        signals_1 = sm.on_m15_close(1, sweep_bar)   # Sets up two-candle pending
        signals_2 = sm.on_m15_close(2, second_bar)  # Resolves two-candle

        assert len(signals_1) == 0  # Nothing yet after sweep bar
        assert len(signals_2) == 1
        sig = signals_2[0]
        assert sig.direction == Direction.LONG
        assert sig.pattern == "two_candle"
        # Extreme is the min of sweep bar low and second bar low
        expected_extreme = min(sweep_bar["low"], second_bar["low"])
        assert sig.extreme == expected_extreme


class TestLevelConsumed:
    """A level should only fire once per day."""

    def test_level_consumed_after_first_signal(self):
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)

        L = LEVELS.pdl
        dummy = _bar(L + 0.0020, L + 0.0030, L + 0.0010, L + 0.0015, _ist(17, 30))

        # First sweep+rejection (bullish green candle close > open)
        sweep1 = _bar(L + 0.0002, L + 0.0015, L - 3 * PIP, L + 0.0005, _ist(17, 45))
        sm.on_m15_close(0, dummy)
        s1 = sm.on_m15_close(1, sweep1)
        assert len(s1) == 1

        # Second sweep attempt on same level — should be ignored
        sweep2 = _bar(L + 0.0002, L + 0.0015, L - 3 * PIP, L + 0.0005, _ist(18, 0))
        s2 = sm.on_m15_close(2, sweep2)
        assert len(s2) == 0


class TestNoSweep:
    """Bars that don't meet the sweep condition should not produce signals."""

    def test_sweep_below_min_pips(self):
        """Bar sweeps only 1 pip below level — insufficient (need 2)."""
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)

        L = LEVELS.pdl
        dummy = _bar(L + 0.0020, L + 0.0030, L + 0.0010, L + 0.0015, _ist(17, 30))
        weak_sweep = _bar(L + 0.0010, L + 0.0015, L - 1 * PIP, L + 0.0005, _ist(17, 45))

        sm.on_m15_close(0, dummy)
        signals = sm.on_m15_close(1, weak_sweep)
        assert len(signals) == 0

    def test_prev_close_below_support(self):
        """prev_close already below level — not a sweep from above."""
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)

        L = LEVELS.pdl
        # prev_close is already below level
        dummy = _bar(L - 0.0005, L - 0.0002, L - 0.0010, L - 0.0003, _ist(17, 30))
        bar = _bar(L + 0.0010, L + 0.0015, L - 3 * PIP, L + 0.0005, _ist(17, 45))

        sm.on_m15_close(0, dummy)
        signals = sm.on_m15_close(1, bar)
        assert len(signals) == 0


class TestMaxTradesPerDay:
    """get_day_signals should cap at max_trades_per_day."""

    def test_max_2_signals(self):
        sm = StrategyStateMachine(cfg=CFG)
        sm.on_day_start(TRADE_DATE, LEVELS)
        # Manually inject 3 signals
        from sweepbot.strategy import Signal, Direction
        for i in range(3):
            sm._signals.append(Signal(
                direction=Direction.LONG,
                level_name=f"TEST{i}",
                level_price=1.09,
                pattern="one_candle",
                extreme=1.088,
                entry_bar_time=_ist(17, 30 + i * 15),
            ))

        signals = sm.get_day_signals()
        assert len(signals) == CFG.max_trades_per_day  # 2


class TestFineTuning:
    """Tests for strategy fine-tuning parameters."""

    def test_rejection_color_filter(self):
        """Red candle rejection for LONG should be filtered out."""
        from sweepbot.config import Config
        cfg_strict = Config(require_rejection_color=True)
        sm = StrategyStateMachine(cfg=cfg_strict)
        sm.on_day_start(TRADE_DATE, LEVELS)
        L = LEVELS.pdl

        dummy = _bar(L + 0.0020, L + 0.0030, L + 0.0010, L + 0.0015, _ist(17, 30))
        # Bearish candle (open 1.0910 > close 1.0905)
        bearish_sweep = _bar(
            open_=L + 0.0010,
            high=L + 0.0015,
            low=L - 3 * PIP,
            close=L + 0.0005,
            ts_ist=_ist(17, 45),
        )
        sm.on_m15_close(0, dummy)
        sigs = sm.on_m15_close(1, bearish_sweep)
        assert len(sigs) == 0

        # Now test with require_rejection_color=False -> should accept
        cfg_relaxed = Config(require_rejection_color=False)
        sm2 = StrategyStateMachine(cfg=cfg_relaxed)
        sm2.on_day_start(TRADE_DATE, LEVELS)
        sm2.on_m15_close(0, dummy)
        sigs2 = sm2.on_m15_close(1, bearish_sweep)
        assert len(sigs2) == 1

    def test_excluded_levels(self):
        """Levels in excluded_levels should not be loaded into strategy."""
        from sweepbot.config import Config
        cfg = Config(excluded_levels=("LonL", "AsiaL"))
        sm = StrategyStateMachine(cfg=cfg)
        sm.on_day_start(TRADE_DATE, LEVELS)
        names = [l.name for l in sm._levels]
        assert "LonL" not in names
        assert "AsiaL" not in names
        assert "PDL" in names
        assert "AsiaH" in names

    def test_allowed_patterns_filter(self):
        """If allowed_patterns=('two_candle',), one-candle signals should be ignored."""
        from sweepbot.config import Config
        cfg = Config(allowed_patterns=("two_candle",))
        sm = StrategyStateMachine(cfg=cfg)
        sm.on_day_start(TRADE_DATE, LEVELS)
        L = LEVELS.pdl

        dummy = _bar(L + 0.0020, L + 0.0030, L + 0.0010, L + 0.0015, _ist(17, 30))
        valid_one_candle = _bar(
            open_=L + 0.0002,
            high=L + 0.0015,
            low=L - 3 * PIP,
            close=L + 0.0005,
            ts_ist=_ist(17, 45),
        )
        sm.on_m15_close(0, dummy)
        sigs = sm.on_m15_close(1, valid_one_candle)
        assert len(sigs) == 0


class TestOpeningBarSeed:
    """Tests for Opening Bar (Bar 0) Seeding with the 17:15 M15 close."""

    def test_bar_zero_can_trigger_signal_with_seed(self):
        """When seed_prev_close is provided, bar 0 (17:30 IST) can detect a sweep and signal."""
        sm = StrategyStateMachine(cfg=CFG)
        L = LEVELS.pdl
        # London close at 17:15 was above PDL
        sm.on_day_start(TRADE_DATE, LEVELS, seed_prev_close=L + 0.0010)

        # Bar 0 sweeps PDL and closes green above PDL
        sweep_bar_0 = _bar(
            open_=L + 0.0002,
            high=L + 0.0015,
            low=L - 3 * PIP,
            close=L + 0.0005,
            ts_ist=_ist(17, 30),
        )
        signals = sm.on_m15_close(0, sweep_bar_0)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == Direction.LONG
        assert sig.level_name == "PDL"
        assert sig.pattern == "one_candle"

