# run_backtest.py  (root-level script)
"""
Entry point: run all strategies and print comparison table.
"""

import pandas as pd
import logging
from backtester.data_loader import load_universe
from backtester.signals import (
    cross_sectional_momentum,
    time_series_momentum,
    zscore_mean_reversion,
    signal_to_positions
)
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import summarise

logging.basicConfig(level=logging.INFO)

# ── Load data ──────────────────────────────────────────────────
prices, returns = load_universe("sp500", "2018-01-01", "2024-12-31")

config = BacktestConfig(
    transaction_cost_bps=5,
    slippage_bps=2,
    initial_capital=1_000_000
)

results_list = []

# ── Strategy 1: Cross-sectional momentum ──────────────────────
sig1 = cross_sectional_momentum(prices, lookback=252, n_long=5, n_short=5)
pos1 = signal_to_positions(sig1)
res1 = run_backtest(returns, pos1, config)
results_list.append(summarise(res1, "XS Momentum (12M)"))

# ── Strategy 2: Time-series momentum ──────────────────────────
sig2 = time_series_momentum(prices, lookback=252)
pos2 = signal_to_positions(sig2)
res2 = run_backtest(returns, pos2, config)
results_list.append(summarise(res2, "TS Momentum (12M)"))

# ── Strategy 3: Z-score mean-reversion ────────────────────────
sig3 = zscore_mean_reversion(prices, lookback=20, entry_z=1.5)
pos3 = signal_to_positions(sig3)
res3 = run_backtest(returns, pos3, config)
results_list.append(summarise(res3, "Z-Score Mean-Rev (20D)"))

# ── SPY Buy & Hold benchmark ───────────────────────────────────
spy_return = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]
spy_pos    = pd.DataFrame(1.0, index=returns.index, columns=["SPY"])
spy_returns_df = spy_return.to_frame()
res_spy = run_backtest(spy_returns_df, spy_pos, config)
results_list.append(summarise(res_spy, "SPY Buy & Hold"))

# ── Print comparison table ────────────────────────────────────
comparison = pd.DataFrame(results_list).set_index("Strategy")
print("\n" + "="*70)
print("STRATEGY COMPARISON")
print("="*70)
print(comparison.to_string())
print("="*70)

# Save results
comparison.to_csv("results/strategy_comparison.csv")
print("\nSaved to results/strategy_comparison.csv")
