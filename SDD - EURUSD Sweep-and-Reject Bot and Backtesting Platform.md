# SDD: EURUSD Sweep-and-Reject Bot and Backtesting Platform

Version 0.2 (Revised) | Written 25 Oct 2026 | Language: Python 3.11+

## 1. Purpose and scope

Build one Python codebase that (a) backtests the NY-open sweep-and-reject strategy on M1 data with no look-ahead, and (b) runs the exact same strategy code in paper and live mode. 

**Improvements in v0.2:** 
- All operational windows are localized to Indian Standard Time (IST). 
- Execution cost models now separate raw spread from broker commissions.
- State recovery added for MT5 live environments.
- Time-of-day edge case tracking implemented for late entries.

Status of the strategy: exploratory. It must pass the out-of-sample test in Section 8 before any real money is used.

## 2. Strategy specification (exact rules)

**Time Zone Anchor:** All times below are defined in **IST based on US Summer Time (EDT, UTC-4)**. (e.g., 08:00 NY = 17:30 IST). The internal engine must dynamically shift these by +1 hour (e.g., 18:30 IST) during US Winter (EST, UTC-5). 

1 pip = 0.0001. Prices are bid prices.

### 2.1 Parameters

| Parameter | Value |
| --- | --- |
| symbol | EURUSD |
| entry window | 17:30 to 19:30 IST (Entry bar may start at 19:15 IST max) |
| time exit | 21:30 IST |
| signal timeframe | M15 (built from M1) |
| sweep_min_pips | 2.0 |
| stop_buffer_pips | 1.0 |
| min_stop / max_stop | 3 pips / 20 pips |
| target_r | 2.0 |
| max_trades_per_day | 2 |
| risk_per_trade_pct | 0.5 (use 0.25 until validated) |
| spread_pips | 0.2 (applied to entry/exit modeling) |
| commission_usd | $3.00 per standard lot (~0.3 pips equivalent deducted from net PnL) |
| trading days | Monday to Friday |

### 2.2 Trading day

A trading day rolls at 02:30 IST (17:00 NY). Label = (timestamp + 7 hours).date. Day D covers D-1 02:30 IST to D 02:30 IST. 

### 2.3 Levels (computed once, frozen before 17:30 IST)

| Level | Definition | Type |
| --- | --- | --- |
| PDH / PDL | high / low of the previous trading day | resistance / support |
| AsiaH / AsiaL | high / low of M1 bars from 04:30 to 12:29 IST | resistance / support |
| LonH / LonL | high / low of M1 bars from 12:30 to 17:29 IST | resistance / support |

Skip the whole day if the Asian range has fewer than 200 M1 bars or the London range has fewer than 150.

### 2.4 Bars

M15 bars are built from M1: left-labelled, left-closed (the 17:30 bar covers 17:30 to 17:44). Open = first M1 open, high = max, low = min, close = last M1 close. 

### 2.5 Signal detection

Evaluated when an M15 bar closes. Each level is scanned independently. Let prev_close be the close of the bar before the candidate bar.

Support level L (long setup):
1. Sweep: bar.low <= L - 2 pips AND prev_close > L.
2. One-candle rejection: bar.close > L. The extreme is bar.low and the entry bar is the next bar.
3. Two-candle rejection: bar.close <= L, and the next bar's close > L. The extreme is the lower of the two lows and the entry bar is the bar after the next.
4. The entry bar must exist inside the window (its start time is 19:15 IST or earlier).

Resistance level L (short setup) is the exact mirror.
A level is consumed by its first complete pattern of the day.

### 2.6 Order, stop and target (Incorporating Spread)

1. Entry: market order at the open of the entry bar. For Longs, execution price = Bid + spread_pips. For Shorts, execution price = Bid.
2. Stop: extreme minus 1 pip for longs, extreme plus 1 pip for shorts.
3. risk_pips = distance from entry to stop. Skip if risk_pips < 3 or > 20.
4. Target: entry plus 2 x risk for longs (minus for shorts).
5. Time exit: at 21:30 IST close any open position at market.

### 2.7 Daily limits

Collect all signals for the day, order by entry time, take at most the first 2. Two trades may be open concurrently. 

### 2.8 Backtest fill rules & Commissions

- Walk M1 bars from the entry minute to 21:29 IST.
- If stop and target are touched inside the same minute, count the stop.
- Net result in R = gross R - (commission_usd_equivalent_in_R). 

## 3. Reference results from the research phase

*Note: These numbers were based on an older cost model (0.8 flat cost) and NY time. They serve as a baseline for logic validation.*
- Trades: 141
- Win rate: ~49%
- Average R per trade: +0.255
- Max drawdown: 15.6R

## 4. Architecture

```
sweepbot/
  config.py          
  data/
    loader.py        
    timeutil.py      # CRITICAL: Handles IST to America/New_York DST shifts
    resample.py      
  levels.py          
  strategy.py        # Pure state machine
  risk.py            # Lot sizing, commission calc
  engine/
    backtest.py      
    fills.py         # M1 fill simulator incorporating spread modeling
    metrics.py       
    reports.py       
  live/
    broker_base.py   
    paper.py         
    mt5.py           # MetaTrader5 adapter with State Recovery
  cli.py             
tests/
```

## 5. Data layer & Time Normalisation

### 5.1 Time normalisation (critical)

The internal engine **must** convert the M1 CSV files to UTC, and then to `America/New_York` to establish the correct 08:00 NY open, before projecting the timestamps to `Asia/Kolkata` (IST) for logging and reporting. 

Mandatory check: `verify_timezone()` finds the largest-range M1 bar between 16:30 and 00:30 IST on known US NFP first Fridays and asserts it falls exactly at 18:00 IST (Summer) or 19:00 IST (Winter).

## 6. Backtest engine

1. Group M1 data by date. Call `strategy.on_day_start` with data before 17:30 IST.
2. Stream M1 bars. Call `on_m15_close(bar)` and collect orders.
3. Apply the fill simulator (fills.py).
4. At 21:30 IST close all open positions. 

## 7. Metrics and reports

Standard metrics: trade count, win rate, average R, max drawdown, bootstrap 95% confidence interval.
**New v0.2 Metrics:**
- **Commission drag:** Total raw R minus total net R.
- **Late Entry Drag:** Breakout win rate for entries *before* 19:00 IST vs. entries *between* 19:00 and 19:30 IST (to measure the impact of having less time before the 21:30 IST hard close).

## 8. Validation protocol and acceptance criteria

1. Freeze parameters. No tuning on the test period.
2. Run on 2026 EURUSD M1 (Jan to latest full month) as Out-Of-Sample data.
3. Pass criteria: > 60 trades; net average R > 0 at modeled spread+commission; max drawdown below 60% of prop firm limit.
4. Compare with 2025. If 2026 fails, STOP. Do not curve-fit.
5. If passed, paper trade for 4-8 weeks.

## 9. Paper and live automation (MT5 Adapter)

- **State Recovery (CRITICAL):** Upon initialization or reconnection, `mt5.py` must call `mt5.positions_get(symbol="EURUSD")`. If a position is active, it must reconstruct the state (Stop Loss and Target) and skip new entries to prevent double-firing after a laptop restart or network drop.
- **Runner loop:** At 17:25 IST, load the day's levels. Every 15 mins, call `on_m15_close`. 
- **Safety:** Hard kill switch at 21:30 IST. Stop all trading if daily loss hits -1.5%.