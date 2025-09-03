# BTC Coin-M DCA Ladder Backtest

## Overview
A deterministic backtester for coin-margined BTC perpetuals implementing a DCA ladder with explicit perp mechanics (inverse PnL in BTC), funding, fees, and liquidation.

## Data
- BTC-USD daily OHLC (source: <source>, timezone: <tz>)
- Funding: constant 10% p.a. unless series provided.

## Methodology
- Entry/scale-in via ladder rules (deterministic fills with slippage/commission).
- Leverage cap, maintenance margin, liquidation model.
- IS window: 2020-05-11 → 2021-10-24 (objective end date).
- OOS window: 2024-04-20 → 2024-10-20 (halving + 6m).

## How to Run
- Python 3.10+
- `pip install -r requirements.txt`
- Open `MS_backtesting.ipynb` and run all cells.

## Key Results (Summary)
| Window | CAGR | Sharpe | Sortino | MDD | VaR(95%) | ES(95%) |
|--------|------|--------|---------|-----|----------|---------|
| IS     |  936 |    3   |   4.4   | -49 |   -6.7   |  -10.4  |
| OOS    |  179 |   1.5  |   2.6   | -38 |   -7.1   |   -9.4  |

(Insert equity curve and drawdown plot here)

## Limitations
- Isolated regime
- Constant funding unless series supplied.
- Daily bars; no intraday path dependency.

## Next Steps
- Optional T+1 execution.
- Historical funding feed.