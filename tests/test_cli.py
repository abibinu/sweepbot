"""
tests/test_cli.py
-----------------
Unit tests for cli.py commands using Click's CliRunner.
Verifies:
  - backtest command options and execution with CSV input
  - verify-tz command with valid/invalid data
  - CLI help and options
"""
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner
import pandas as pd
from zoneinfo import ZoneInfo

from sweepbot.cli import main
from tests.test_backtest import _generate_synthetic_day

TZ_UTC = ZoneInfo("UTC")


class TestCli:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SweepBot" in result.output
        assert "backtest" in result.output
        assert "live" in result.output
        assert "verify-tz" in result.output

    def test_backtest_with_csv(self):
        # Generate synthetic day data and save as CSV
        rows = _generate_synthetic_day("2025-06-10")
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_m1.csv"
            out_csv = Path(tmpdir) / "out_trades.csv"

            # Save in format expected by load_m1_from_csv
            df_to_save = df.reset_index()
            df_to_save.rename(columns={"timestamp": "Timestamp"}, inplace=True)
            df_to_save.to_csv(csv_path, index=False)

            runner = CliRunner()
            result = runner.invoke(main, [
                "backtest",
                "--csv", str(csv_path),
                "--start", "2025-06-10",
                "--end", "2025-06-10",
                "--out-csv", str(out_csv),
            ])

            assert result.exit_code == 0
            assert "Running backtest" in result.output
            assert out_csv.exists()

    def test_verify_tz_command(self):
        # Create CSV with valid tz data
        start = pd.Timestamp("2025-01-02 10:00:00", tz=TZ_UTC)
        df = pd.DataFrame([{
            "Timestamp": start,
            "Open": 1.1000, "High": 1.1010, "Low": 1.0990, "Close": 1.1000,
            "Volume": 100
        }])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "nfp.csv"
            df.to_csv(csv_path, index=False)

            runner = CliRunner()
            result = runner.invoke(main, ["verify-tz", "--csv", str(csv_path)])
            assert result.exit_code == 0
            assert "PASSED" in result.output

    def test_portfolio_backtest_with_multiple_symbols(self, monkeypatch):
        rows = _generate_synthetic_day("2025-06-10")
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        import sweepbot.data.loader
        monkeypatch.setattr(sweepbot.data.loader, "load_m1", lambda start, end, cfg: df)

        runner = CliRunner()
        result = runner.invoke(main, [
            "backtest",
            "--start", "2025-06-10",
            "--end", "2025-06-10",
            "--symbol", "EURUSD,GBPUSD",
            "--no-log",
        ])

        assert result.exit_code == 0
        assert "Running multi-pair portfolio backtest" in result.output
        assert "EURUSD" in result.output
        assert "GBPUSD" in result.output
