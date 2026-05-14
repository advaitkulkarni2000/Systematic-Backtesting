"""
backtester/signals.py

Signal generation functions.

All functions:
  - Accept a prices DataFrame (DatetimeIndex × tickers)
  - Return a signal DataFrame of the same shape
  - Apply .shift(1) internally so there is NO look-ahead bias
    (signal generated at close of day T is applied at open of day T+1)
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  1 — CROSS-SECTIONAL MOMENTUM  (Jegadeesh & Titman 1993)
# ═══════════════════════════════════════════════════════════════════════════

def cross_sectional_momentum(
    prices:    pd.DataFrame,
    lookback:  int   = 252,
    skip_last: int   = 21,
    n_long:    int   = 10,
    n_short:   int   = 10,
) -> pd.DataFrame:
    """
    Rank all stocks by their formation-period return, go equal-weight long
    in the top n_long and equal-weight short in the bottom n_short.

    skip_last: skip the most recent `skip_last` days before measuring
    the formation return to avoid the well-documented 1-month reversal
    contaminating the 12-month momentum signal.

    Parameters
    ----------
    prices    : adjusted close price DataFrame
    lookback  : formation window in trading days (default 252 ≈ 12 months)
    skip_last : days to skip at the end of the formation period (default 21 ≈ 1 month)
    n_long    : number of stocks in the long leg
    n_short   : number of stocks in the short leg

    Returns
    -------
    Signal DataFrame with values in {+1/n_long, 0, -1/n_short}.
    Shifted by 1 day (no look-ahead).
    """
    formation_ret = prices.shift(skip_last) / prices.shift(lookback) - 1

    ranked  = formation_ret.rank(axis=1, ascending=False, method="first")
    n_valid = formation_ret.notna().sum(axis=1)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    long_mask  = ranked.le(n_long)
    short_mask = ranked.ge(n_valid.values.reshape(-1, 1) - n_short + 1)

    signal[long_mask]  =  1.0 / n_long
    signal[short_mask] = -1.0 / n_short

    # Zero out rows with insufficient stocks
    signal[n_valid < (n_long + n_short)] = 0.0

    return signal.shift(1).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  2 — TIME-SERIES MOMENTUM  (Moskowitz, Ooi & Pedersen 2012)
# ═══════════════════════════════════════════════════════════════════════════

def time_series_momentum(
    prices:   pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Each asset independently: long if trailing 12-month return > 0, short otherwise.
    Positions are inverse-volatility scaled (down-weight high-vol assets).

    Returns
    -------
    Continuous signal DataFrame (not ±1 — normalise with signal_to_positions).
    Shifted by 1 day.
    """
    trailing_ret = prices / prices.shift(lookback) - 1
    direction    = np.sign(trailing_ret)

    # 63-day rolling vol for inverse-vol scaling
    log_ret      = np.log(prices / prices.shift(1))
    rolling_vol  = log_ret.rolling(63).std() * np.sqrt(252)
    inv_vol      = 1.0 / rolling_vol.replace(0, np.nan)

    signal = direction * inv_vol
    return signal.shift(1).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  3 — Z-SCORE MEAN-REVERSION
# ═══════════════════════════════════════════════════════════════════════════

def zscore_mean_reversion(
    prices:   pd.DataFrame,
    lookback: int   = 20,
    entry_z:  float = 1.5,
    exit_z:   float = 0.5,
) -> pd.DataFrame:
    """
    Short-term reversal based on rolling z-score.

    z = (price − rolling_mean) / rolling_std

    z < −entry_z  →  Long  (+1)   oversold
    z >  entry_z  →  Short (−1)   overbought
    |z| <  exit_z →  Flat  (0)    no signal

    Parameters
    ----------
    lookback : rolling window for mean/std (default 20 ≈ 1 month)
    entry_z  : z-score threshold to enter a position
    exit_z   : z-score threshold to flatten (not used in current binary
                implementation but kept for extension)

    Returns
    -------
    Signal DataFrame with values in {+1, 0, −1}.
    Shifted by 1 day.
    """
    roll_mean = prices.rolling(lookback).mean()
    roll_std  = prices.rolling(lookback).std().replace(0, np.nan)
    z         = (prices - roll_mean) / roll_std

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[z < -entry_z] =  1.0
    signal[z >  entry_z] = -1.0

    return signal.shift(1).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  4 — MA CROSSOVER  (bonus strategy)
# ═══════════════════════════════════════════════════════════════════════════

def ma_crossover(
    prices: pd.DataFrame,
    fast:   int = 50,
    slow:   int = 200,
) -> pd.DataFrame:
    """
    Golden / death cross: long when fast MA > slow MA, short otherwise.

    Returns
    -------
    Signal DataFrame with values in {+1, −1}.
    Shifted by 1 day.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal  = np.sign(fast_ma - slow_ma)
    return signal.shift(1).fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def signal_to_positions(
    signals:   pd.DataFrame,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """
    Normalise a signal DataFrame so each row's gross exposure equals max_gross.
    Rows with zero signal remain zero.

    Parameters
    ----------
    signals   : raw signal DataFrame (any numeric values)
    max_gross : target gross exposure per day (default 1.0 = 100 % of capital)

    Returns
    -------
    Position-weight DataFrame summing to max_gross in absolute value per row.
    """
    abs_sum   = signals.abs().sum(axis=1).replace(0, np.nan)
    positions = signals.div(abs_sum, axis=0) * max_gross
    return positions.fillna(0.0)


def combine_signals(
    signal_list: list,
    weights:     list = None,
) -> pd.DataFrame:
    """
    Linearly combine multiple signal DataFrames.
    If weights=None, equal-weights all inputs.
    """
    if weights is None:
        weights = [1.0 / len(signal_list)] * len(signal_list)
    return sum(w * s for w, s in zip(weights, signal_list))
