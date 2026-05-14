# backtester/metrics.py
"""
Performance metrics. All standard quant interview metrics included.
"""

import pandas as pd
import numpy as np


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (assumes 0% risk-free rate)."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sortino ratio: penalises only downside volatility."""
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    rolling_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - rolling_max) / rolling_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, cumulative_pnl: pd.Series,
                 periods_per_year: int = 252) -> float:
    """Calmar = annualised return / |max drawdown|."""
    annual_return = returns.mean() * periods_per_year
    mdd = abs(max_drawdown(cumulative_pnl))
    return annual_return / mdd if mdd != 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with positive return."""
    return (returns > 0).mean()


def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.mean() * periods_per_year


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def average_daily_turnover(turnover: pd.Series) -> float:
    return turnover.mean()


def summarise(results: pd.DataFrame, label: str = "Strategy") -> pd.Series:
    """
    One-call summary of all metrics.
    Returns a named Series — easy to compare strategies in a table.
    """
    r = results["net_return"]
    c = results["cumulative_pnl"]
    t = results["turnover"]

    return pd.Series({
        "Strategy":            label,
        "Ann. Return (%)":     round(annualised_return(r) * 100, 2),
        "Ann. Vol (%)":        round(annualised_vol(r) * 100, 2),
        "Sharpe Ratio":        round(sharpe_ratio(r), 3),
        "Sortino Ratio":       round(sortino_ratio(r), 3),
        "Calmar Ratio":        round(calmar_ratio(r, c), 3),
        "Max Drawdown (%)":    round(max_drawdown(c) * 100, 2),
        "Hit Rate (%)":        round(hit_rate(r) * 100, 2),
        "Avg Daily Turnover":  round(average_daily_turnover(t), 4),
        "Final PnL ($)":       round(c.iloc[-1], 0),
    })
