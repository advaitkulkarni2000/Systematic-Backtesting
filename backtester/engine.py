# backtester/engine.py
"""
Vectorised backtest engine.
No loops over dates — uses pure Pandas/NumPy operations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    transaction_cost_bps: float = 5.0    # basis points per trade (one-way)
    slippage_bps: float = 2.0            # additional slippage
    initial_capital: float = 1_000_000   # $1M notional
    rebalance_freq: str = "daily"        # 'daily', 'weekly', 'monthly'


def run_backtest(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    config: BacktestConfig | None = None
) -> pd.DataFrame:
    """
    Core vectorised backtest.
    
    Args:
        returns:   daily log-returns  (T × N)
        positions: daily target weights (T × N), already shifted to avoid look-ahead
        config:    BacktestConfig
    
    Returns:
        results DataFrame with columns:
            portfolio_return, gross_return, transaction_costs,
            cumulative_pnl, drawdown
    """
    if config is None:
        config = BacktestConfig()

    # Align
    positions = positions.reindex(returns.index).fillna(0)

    # ── Transaction costs ──────────────────────────────────────
    turnover = positions.diff().abs().sum(axis=1)          # daily one-way turnover
    total_cost_bps = config.transaction_cost_bps + config.slippage_bps
    tc = turnover * (total_cost_bps / 10_000)              # as fraction of portfolio

    # ── Gross portfolio return ─────────────────────────────────
    # positions are already lagged (signal → positions uses shift(1))
    # returns are log-returns; use simple approx: sum(w_i * r_i)
    gross_return = (positions * returns).sum(axis=1)

    # ── Net return ─────────────────────────────────────────────
    net_return = gross_return - tc

    # ── Cumulative PnL (in dollar terms) ──────────────────────
    cumulative = (1 + net_return).cumprod() * config.initial_capital

    # ── Drawdown ───────────────────────────────────────────────
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max

    results = pd.DataFrame({
        "gross_return":   gross_return,
        "transaction_cost": tc,
        "net_return":     net_return,
        "cumulative_pnl": cumulative,
        "drawdown":       drawdown,
        "turnover":       turnover,
    })
    return results
