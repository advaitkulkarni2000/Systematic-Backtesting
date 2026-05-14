# backtester/sensitivity.py
"""
Parameter sweep: test how Sharpe varies with lookback window.
Shows robustness — not a one-off lucky parameter.
"""

import pandas as pd
import numpy as np
import itertools
from backtester.signals import cross_sectional_momentum, signal_to_positions
from backtester.engine import run_backtest, BacktestConfig
from backtester.metrics import sharpe_ratio


def momentum_parameter_sweep(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookbacks: list[int] = [63, 126, 189, 252, 315],
    n_longs:   list[int] = [3, 5, 7],
    config: BacktestConfig | None = None
) -> pd.DataFrame:
    """
    Grid search over momentum lookback and portfolio size.
    Returns a pivoted heatmap of Sharpe ratios.
    """
    if config is None:
        config = BacktestConfig()

    records = []
    for lb, nl in itertools.product(lookbacks, n_longs):
        sig  = cross_sectional_momentum(prices, lookback=lb, n_long=nl, n_short=nl)
        pos  = signal_to_positions(sig)
        res  = run_backtest(returns, pos, config)
        sr   = sharpe_ratio(res["net_return"])
        records.append({"lookback": lb, "n_long": nl, "sharpe": round(sr, 3)})

    df = pd.DataFrame(records)
    heatmap = df.pivot(index="lookback", columns="n_long", values="sharpe")
    return heatmap


# Usage (in notebook):
# heatmap = momentum_parameter_sweep(prices, returns)
# sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
# plt.title("Momentum Sharpe Ratio — Parameter Sensitivity")
