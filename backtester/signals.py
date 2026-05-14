# backtester/signals.py
"""
Signal generation module.
All functions take a prices DataFrame and return a signal DataFrame
of the same shape with values in [-1, 0, +1] or continuous weights.
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  MOMENTUM SIGNALS
# ═══════════════════════════════════════════════════════════════

def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip_last: int = 21,
    n_long: int = 5,
    n_short: int = 5
) -> pd.DataFrame:
    """
    Classic Jegadeesh-Titman (1993) cross-sectional momentum.
    
    At each date:
      1. Compute trailing return over [t - lookback, t - skip_last]
         (skip last month to avoid short-term reversal contamination)
      2. Rank stocks; go long top-n_long, short bottom-n_short
      3. Equal-weight within long/short legs
    
    Returns: signal DataFrame with values in {-1, 0, +1}
    """
    # Step 1: compute formation-period returns
    formation_return = prices.shift(skip_last) / prices.shift(lookback) - 1

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for date in prices.index[lookback:]:
        row = formation_return.loc[date].dropna()
        if len(row) < n_long + n_short:
            continue

        ranked = row.rank(ascending=False)
        longs  = ranked[ranked <= n_long].index
        shorts = ranked[ranked > len(row) - n_short].index

        signals.loc[date, longs]  =  1.0 / n_long
        signals.loc[date, shorts] = -1.0 / n_short

    return signals


def time_series_momentum(
    prices: pd.DataFrame,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Time-series momentum (Moskowitz, Ooi, Pedersen 2012).
    If trailing 12-month return is positive → long (+1), else short (-1).
    Applied independently to each asset.
    """
    trailing_return = prices / prices.shift(lookback) - 1
    signals = np.sign(trailing_return).shift(1)  # shift 1 to avoid look-ahead
    return signals.fillna(0)


def moving_average_crossover(
    prices: pd.DataFrame,
    fast: int = 50,
    slow: int = 200
) -> pd.DataFrame:
    """
    Golden/death cross momentum signal.
    +1 when fast MA > slow MA, -1 otherwise.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signals = np.sign(fast_ma - slow_ma).shift(1).fillna(0)
    return signals


# ═══════════════════════════════════════════════════════════════
#  MEAN-REVERSION SIGNALS
# ═══════════════════════════════════════════════════════════════

def zscore_mean_reversion(
    prices: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5
) -> pd.DataFrame:
    """
    Short-term mean-reversion using rolling z-score.
    
    z = (price - rolling_mean) / rolling_std
    
    Signal:
      z < -entry_z  → Long  (+1)   [oversold]
      z >  entry_z  → Short (-1)   [overbought]
      |z| < exit_z  → Flat  ( 0)   [exit zone]
    """
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std  = prices.rolling(lookback).std()
    z = (prices - rolling_mean) / rolling_std

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signals[z < -entry_z] =  1.0   # oversold → buy
    signals[z >  entry_z] = -1.0   # overbought → sell
    signals = signals.shift(1).fillna(0)  # avoid look-ahead
    return signals


def rsi_mean_reversion(
    prices: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70
) -> pd.DataFrame:
    """
    RSI-based mean-reversion.
    RSI < oversold  → Long
    RSI > overbought → Short
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signals[rsi < oversold]   =  1.0
    signals[rsi > overbought] = -1.0
    return signals.shift(1).fillna(0)


# ═══════════════════════════════════════════════════════════════
#  SIGNAL UTILITIES
# ═══════════════════════════════════════════════════════════════

def combine_signals(
    signal_list: list[pd.DataFrame],
    weights: list[float] | None = None
) -> pd.DataFrame:
    """
    Linearly combine multiple signal DataFrames.
    If weights=None, equal-weights all signals.
    """
    if weights is None:
        weights = [1.0 / len(signal_list)] * len(signal_list)
    combined = sum(w * s for w, s in zip(weights, signal_list))
    return combined


def signal_to_positions(
    signals: pd.DataFrame,
    max_position: float = 1.0
) -> pd.DataFrame:
    """
    Normalise signal DataFrame so row absolute values sum to max_position.
    Handles zero-signal rows gracefully.
    """
    abs_sum = signals.abs().sum(axis=1).replace(0, np.nan)
    positions = signals.div(abs_sum, axis=0) * max_position
    return positions.fillna(0)
