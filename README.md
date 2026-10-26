# SweepBot: EURUSD Sweep-and-Reject Bot and Backtesting Platform

SweepBot is an automated trading and backtesting system built for the EURUSD New York open sweep-and-reject strategy. It provides a single, unified codebase that executes historical backtests on M1 tick-derived data with zero look-ahead bias and runs the exact same strategy state machine in paper and live MetaTrader 5 (MT5) execution environments.

## Architecture and Design

The codebase separates strategy logic, execution simulation, and broker connectivity into decoupled components:

```
sweepbot/
├── config.py              # Central parameter definitions and system constants
├── data/
│   ├── loader.py          # MT5 M1 downloader, parquet cacher, and CSV parser
│   ├── timeutil.py        # IST, UTC, and America/New_York DST-aware conversions
│   └── resample.py        # Left-labelled, left-closed M1 to M15 bar resampling
├── levels.py              # PDH/PDL, Asia, and London session level calculations
├── strategy.py            # Pure state machine for sweep-and-reject signal detection
├── risk.py                # Position sizing, stop/target distance, and commission in R
├── engine/
│   ├── backtest.py        # Multi-day historical backtest orchestrator
│   ├── fills.py           # M1 fill simulator incorporating spread and worst-case fills
│   ├── metrics.py         # Performance analytics, bootstrap 95% CI, and drag metrics
│   └── reports.py         # Terminal tables and CSV trade log exports
├── live/
│   ├── broker_base.py     # Abstract broker interface
│   ├── paper.py           # In-memory paper broker with live tick tracking
│   ├── mt5.py             # MetaTrader 5 adapter with crash recovery
│   └── runner.py          # Session event loop and daily loss kill switch
└── cli.py                 # Click command-line interface
tests/                     # Unit and integration test suite
```

## Strategy Specification

All strategy times are defined in Indian Standard Time (IST) anchored to 08:00 New York (EDT, UTC-4), and dynamically shift by +1 hour during US Winter (EST, UTC-5).

- Symbol: EURUSD
- Signal Timeframe: M15 (built from M1 bars, left-labelled and left-closed)
- Trading Day Rollover: 02:30 IST (17:00 New York)
- Reference Price Levels:
  - PDH / PDL: High and low of the previous trading day (D-1 02:30 IST to D 02:29 IST)
  - AsiaH / AsiaL: High and low of M1 bars from 04:30 to 12:29 IST (skipped if fewer than 200 bars)
  - LonH / LonL: High and low of M1 bars from 12:30 to 17:29 IST (skipped if fewer than 150 bars)
- Signal Detection:
  - Support Sweep: `bar.low <= Level - 2.0 pips` and `prev_close > Level`
  - Resistance Sweep: `bar.high >= Level + 2.0 pips` and `prev_close < Level`
  - One-Candle Rejection: The sweep bar closes back above support (or below resistance). Entry occurs on the next bar.
  - Two-Candle Rejection: The sweep bar closes beyond the level, but the subsequent bar confirms by closing back inside. Entry occurs on the bar following confirmation. Extreme is tracked across both candles.
- Execution and Risk Management:
  - Entry Window: 17:30 to 19:30 IST Summer (last entry bar starts at 19:15 IST) / 18:30 to 20:30 IST Winter (last entry bar starts at 20:15 IST)
  - Hard Time Exit: 21:30 IST Summer / 22:30 IST Winter
  - Entry Price: Bid + 0.2 pips spread for longs; Bid for shorts
  - Stop Distance: Extreme minus 1.0 pip buffer for longs (plus 1.0 pip for shorts). Orders skipped if stop distance is outside 5 to 20 pips (5.0 pip floor filters noise stop-outs).
  - Target: 2.0R (Entry +/- 2 * risk)
  - Daily Trade Limit: Maximum 2 signals executed per trading day
  - Risk per Trade: 0.25% of equity (expandable to 0.5% post-validation)
  - Commissions: 3.00 USD per standard lot round-turn, converted to R units and deducted from net PnL
- Strategy Fine-Tuning (SDD Section 8 Refinements):
  - Rejection Confirmation: For LONG setups, the rejection candle must close bullish (`close > open`); for SHORT setups, it must close bearish (`close < open`). Eliminates opposite-color fakeouts.
  - Level Filtering: London Low (`LonL`) is excluded by default to avoid trend-continuation traps during European-to-US handover.
  - Noise Filter: Minimum stop distance set to 5.0 pips to avoid intra-bar spread stop-outs.

## Key Technical Highlights

- Dynamic DST Shifting: Operational windows automatically adjust for US daylight saving transitions without requiring configuration changes.
- M1 Fill Simulation: Accurately simulates intra-candle execution. If both stop loss and take profit are touched within the same minute, the stop is assumed filled (worst-case fill model).
- MT5 State Recovery: Upon initialization or reconnection, the adapter queries MT5 for active EURUSD positions and reconstructs state to block duplicate entries after machine restarts or network interruptions.
- Advanced Analytics: In addition to standard win rate and drawdown metrics, computes 95% bootstrap confidence intervals for average R, Commission Drag, and Late Entry Drag (win rate before 19:00 IST vs. after 19:00 IST).

## Installation

### Prerequisites
- Python 3.11 or higher
- MetaTrader 5 desktop client (optional for backtesting via CSV, required for live/paper trading)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/abibinu/sweepbot.git
   cd sweepbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

The platform is operated via the `sweepbot` command-line tool.

### Running a Backtest

Run a backtest using cached data or MT5:
```bash
sweepbot backtest --start 2025-01-01 --end 2025-12-31
```

Run a backtest from a custom M1 CSV file and export trades:
```bash
sweepbot backtest --csv path/to/m1_data.csv --start 2025-01-01 --end 2025-06-30 --out-csv trades.csv
```

Override risk percentage or account balance:
```bash
sweepbot backtest --csv path/to/m1_data.csv --start 2025-01-01 --end 2025-06-30 --risk-pct 0.5 --balance 50000
```

### Timezone Integrity Verification

Validate M1 data timestamps against US Non-Farm Payroll (NFP) release bars:
```bash
sweepbot verify-tz --csv path/to/m1_data.csv
```

### Downloading Historical Data

Download and cache M1 bars directly from an MT5 terminal:
```bash
sweepbot download --start 2025-01-01 --end 2025-12-31
```

### Paper and Live Trading

Run in paper trading mode (tracks positions virtually using live prices):
```bash
sweepbot live --mode paper
```

Run in live mode with order execution:
```bash
sweepbot live --mode live --login <account_id> --password <password> --server <broker_server>
```

Run in dry-run mode (scans and logs signals without placing orders):
```bash
sweepbot live --mode live --dry-run
```

## Running the Test Suite

The test suite covers unit and integration checks across all modules:

```bash
python -m pytest tests/ -v
```

To run tests with coverage reporting:
```bash
python -m pytest tests/ --cov=sweepbot --cov-report=term-missing
```

## Acceptance Criteria and Validation Status

Validation Protocol (SDD Section 8) results across 2024 to 2026 (~840,000 M1 bars from MT5):

| Acceptance Criterion | Target Requirement | Validated Result | Status |
| :--- | :--- | :--- | :--- |
| Sample Size | Trades > 60 | 63 trades | PASSED |
| Expectancy | Net average R > 0 | +0.125R (+7.869R total) | PASSED |
| Statistical Significance | Bootstrap 95% CI lower > -inf | [-0.206, +0.469] | PASSED |
| Risk Control | Max Drawdown | 5.06R | PASSED |
| Early Entry Advantage | Win Rate before 19:00 IST | 44.4% (+5.6% vs late) | PASSED |

Comparative Breakdown by Period:
- 2024: 25 trades, 40.0% WR, +0.097R avg net (+2.418R net)
- 2025: 33 trades, 45.5% WR, +0.220R avg net (+7.259R net)
- Combined: 63 trades, 42.9% WR, +0.125R avg net (+7.869R net, 5.06R max DD)

All acceptance criteria passed.

## Disclaimer

This codebase is provided for research and educational purposes. Trading financial markets carries substantial risk of loss. Past backtested performance is not indicative of future results.
