"""
backtester/metrics.py

Performance metrics used throughout the backtesting framework.
All functions operate on pandas Series of daily returns or cumulative P&L.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Core risk-adjusted return metrics ─────────────────────────────────────

def sharpe_ratio(returns: pd.Series, ann: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(ann)


def sortino_ratio(returns: pd.Series, ann: int = 252) -> float:
    """
    Sortino ratio: like Sharpe but penalises only downside volatility.
    More appropriate for strategies with asymmetric return distributions.
    """
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(ann)


def calmar_ratio(returns: pd.Series, cumulative: pd.Series,
                 ann: int = 252) -> float:
    """
    Calmar ratio = annualised return / |max drawdown|.
    Useful for comparing strategies with different drawdown profiles.
    """
    mdd = abs(max_drawdown(cumulative))
    if mdd == 0:
        return np.nan
    return (returns.mean() * ann) / mdd


# ── Drawdown ───────────────────────────────────────────────────────────────

def max_drawdown(cumulative: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction (negative value)."""
    return ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()


# ── Tail-risk metrics ──────────────────────────────────────────────────────

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the return not exceeded on the worst (1-confidence) fraction of days.
    E.g. VaR 95% = 5th percentile of daily returns.
    """
    return np.percentile(returns, (1 - confidence) * 100)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR): mean loss on days worse than VaR.
    A more conservative and coherent risk measure than VaR alone.
    """
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else np.nan


# ── Distribution statistics ────────────────────────────────────────────────

def skewness(returns: pd.Series) -> float:
    """Skewness of daily returns (negative = left tail, bad for long-only)."""
    return float(stats.skew(returns.dropna()))


def excess_kurtosis(returns: pd.Series) -> float:
    """
    Excess kurtosis (0 = Normal).
    Positive value = fat tails (more extreme events than Normal predicts).
    """
    return float(stats.kurtosis(returns.dropna()))


# ── Turnover & activity ────────────────────────────────────────────────────

def annualised_turnover(turnover: pd.Series, ann: int = 252) -> float:
    """Average daily turnover × 252."""
    return turnover.mean() * ann


# ── Full summary ───────────────────────────────────────────────────────────

def summarise(results: pd.DataFrame, label: str = "") -> pd.Series:
    """
    Compute the full suite of performance metrics for one strategy.

    Parameters
    ----------
    results : output DataFrame from engine.run_backtest()
    label   : strategy name (used as index in comparison tables)

    Returns
    -------
    pd.Series of metrics — easy to concat into a comparison DataFrame.
    """
    r = results["net_return"]
    c = results["cumulative_pnl"]
    t = results["turnover"]

    return pd.Series(
        {
            "Label":                 label,
            "Ann. Return (%)":       round(r.mean() * 252 * 100, 2),
            "Ann. Vol (%)":          round(r.std() * np.sqrt(252) * 100, 2),
            "Sharpe Ratio":          round(sharpe_ratio(r), 3),
            "Sortino Ratio":         round(sortino_ratio(r), 3),
            "Calmar Ratio":          round(calmar_ratio(r, c), 3),
            "Max Drawdown (%)":      round(max_drawdown(c) * 100, 2),
            "VaR 95% (daily %)":     round(value_at_risk(r) * 100, 3),
            "CVaR 95% (daily %)":    round(expected_shortfall(r) * 100, 3),
            "Skewness":              round(skewness(r), 3),
            "Excess Kurtosis":       round(excess_kurtosis(r), 3),
            "Hit Rate (%)":          round((r > 0).mean() * 100, 2),
            "Avg Daily Turnover":    round(t.mean(), 4),
            "Ann. Turnover (×)":     round(annualised_turnover(t), 2),
            "Final PnL ($M)":        round(c.iloc[-1] / 1e6, 3),
            "Total Trading Days":    len(r),
        }
    )
