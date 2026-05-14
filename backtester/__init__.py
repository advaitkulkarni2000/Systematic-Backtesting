"""
backtester — systematic equity backtesting framework
"""

from backtester.data_loader import load_universe, fetch_prices, clean_prices, compute_returns
from backtester.signals     import (
    cross_sectional_momentum,
    time_series_momentum,
    zscore_mean_reversion,
    ma_crossover,
    signal_to_positions,
    combine_signals,
)
from backtester.engine      import run_backtest, walk_forward, BacktestConfig
from backtester.metrics     import summarise, sharpe_ratio, max_drawdown

__all__ = [
    "load_universe", "fetch_prices", "clean_prices", "compute_returns",
    "cross_sectional_momentum", "time_series_momentum",
    "zscore_mean_reversion", "ma_crossover",
    "signal_to_positions", "combine_signals",
    "run_backtest", "walk_forward", "BacktestConfig",
    "summarise", "sharpe_ratio", "max_drawdown",
]
