"""
tests/test_config.py
--------------------
Unit tests for Config symbol calibration profiles and post_init behavior.
"""
from sweepbot.config import Config, SYMBOL_PROFILES


class TestConfigSymbolProfiles:
    def test_eurusd_defaults(self):
        cfg = Config()
        assert cfg.symbol == "EURUSD"
        assert cfg.sweep_min_pips == 2.0
        assert cfg.min_stop_pips == 5.0
        assert cfg.max_stop_pips == 20.0
        assert cfg.excluded_levels == ("LonL",)

    def test_gbpusd_calibrated_defaults(self):
        cfg = Config(symbol="GBPUSD")
        assert cfg.symbol == "GBPUSD"
        assert cfg.sweep_min_pips == 2.5
        assert cfg.min_stop_pips == 8.0
        assert cfg.max_stop_pips == 20.0
        assert cfg.excluded_levels == ("LonL", "LonH")

    def test_explicit_overrides_respected(self):
        cfg = Config(symbol="GBPUSD", min_stop_pips=10.0, excluded_levels=("LonL",))
        assert cfg.symbol == "GBPUSD"
        assert cfg.sweep_min_pips == 2.5
        assert cfg.min_stop_pips == 10.0
        assert cfg.excluded_levels == ("LonL",)

    def test_for_symbol_helper(self):
        cfg = Config.for_symbol("GBPUSD", min_stop_pips=9.0)
        assert cfg.symbol == "GBPUSD"
        assert cfg.sweep_min_pips == 2.5
        assert cfg.min_stop_pips == 9.0
        assert cfg.excluded_levels == ("LonL", "LonH")
