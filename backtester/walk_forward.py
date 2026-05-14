# backtester/walk_forward.py
"""
Walk-forward (rolling-window) out-of-sample validation.

This is the most important piece for quant credibility —
it shows you understand the difference between in-sample
and out-of-sample performance.
"""

import pandas as pd
import numpy as np
from typing import Callable
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import summarise


def walk_forward_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_fn: Callable,
    signal_kwargs: dict,
    train_window: int = 504,     # ~2 years in-sample
    test_window: int = 63,       # ~3 months out-of-sample
    config: BacktestConfig | None = None
) -> pd.DataFrame:
    """
    Expanding/rolling walk-forward validation.
    
    Timeline:
    |---- train_window ----|-- test_window --|
                           |---- train_window ----|-- test_window --|
                                                  ...
    
    Signal parameters are fixed (not re-fitted) in this version —
    this is intentional: it tests signal robustness, not parameter fitting.
    
    Returns:
        Concatenated out-of-sample results DataFrame.
    """
    if config is None:
        config = BacktestConfig()

    all_oos_results = []
    n = len(prices)
    start = train_window

    while start + test_window <= n:
        # OOS window
        oos_idx  = prices.index[start : start + test_window]
        oos_prices  = prices.iloc[max(0, start - 252) : start + test_window]
        oos_returns = returns.loc[oos_idx]

        # Generate signal on the full available history up to test start
        full_prices = prices.iloc[:start + test_window]
        signal = signal_fn(full_prices, **signal_kwargs)

        from backtester.signals import signal_to_positions
        positions = signal_to_positions(signal)
        oos_positions = positions.loc[oos_idx]

        res = run_backtest(oos_returns, oos_positions, config)
        all_oos_results.append(res)

        start += test_window

    if not all_oos_results:
        raise ValueError("No OOS windows generated. Check train/test sizes.")

    combined = pd.concat(all_oos_results)
    return combined


def compare_is_vs_oos(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_fn: Callable,
    signal_kwargs: dict,
    config: BacktestConfig | None = None
) -> None:
    """Run in-sample full-period vs. walk-forward OOS and print comparison."""
    from backtester.signals import signal_to_positions
    from backtester.engine import run_backtest

    # In-sample (full period — optimistic, for reference only)
    sig_full = signal_fn(prices, **signal_kwargs)
    pos_full = signal_to_positions(sig_full)
    res_full = run_backtest(returns, pos_full, config)

    # Walk-forward OOS
    res_oos = walk_forward_backtest(
        prices, returns, signal_fn, signal_kwargs, config=config
    )

    is_summary  = summarise(res_full, "In-Sample (Full Period)")
    oos_summary = summarise(res_oos, "Out-of-Sample (Walk-Forward)")

    comparison = pd.DataFrame([is_summary, oos_summary]).set_index("Strategy")
    print("\nIN-SAMPLE vs. OUT-OF-SAMPLE COMPARISON")
    print("="*60)
    print(comparison.to_string())
    print("="*60)
    print("\nNote: IS overestimates OOS — the Sharpe gap is your overfitting estimate.")
