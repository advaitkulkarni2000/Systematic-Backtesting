"""
backtester/engine.py

Vectorised backtest engine.
No loops over dates — all operations are pure Pandas / NumPy,
making it fast even for 100-stock universes over 10-year histories.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """
    Parameters controlling the simulation.

    transaction_cost_bps : one-way cost per trade (execution + spread), basis points
    slippage_bps         : additional market-impact slippage, basis points
    initial_capital      : starting notional in USD (default $1 M)

    Total cost per unit of turnover = (transaction_cost_bps + slippage_bps) / 10_000
    """
    transaction_cost_bps: float = 5.0
    slippage_bps:         float = 2.0
    initial_capital:      float = 1_000_000.0


# ═══════════════════════════════════════════════════════════════════════════
#  CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    returns:   pd.DataFrame,
    positions: pd.DataFrame,
    config:    BacktestConfig = None,
) -> pd.DataFrame:
    """
    Fully vectorised daily backtest.

    Design decisions
    ----------------
    * Positions must already be lagged by 1 trading day before being passed in
      (all signal functions in signals.py call .shift(1) internally).
      This enforces the rule: signal generated at close of day T is
      first tradeable at open of day T+1.

    * Returns are log-returns.  Portfolio log-return ≈ sum(w_i * r_i),
      which is exact for small position sizes and accurate enough here.

    * Transaction costs are applied as:
          TC_t = turnover_t × (tc_bps + slippage_bps) / 10_000
      where turnover_t = Σ |Δw_{i,t}|  (one-way, dollar-weighted).

    Parameters
    ----------
    returns   : daily log-returns  (T × N)
    positions : daily target weights (T × N), already lagged
    config    : BacktestConfig; uses defaults if None

    Returns
    -------
    pd.DataFrame with columns:
        gross_return     – portfolio return before costs
        transaction_cost – daily cost drag
        net_return       – gross_return − transaction_cost
        cumulative_pnl   – dollar P&L (starting from initial_capital)
        drawdown         – peak-to-trough drawdown fraction
        turnover         – one-way daily turnover
        gross_exposure   – Σ |w_i|  per day
        net_exposure     – Σ  w_i   per day (long − short)
    """
    if config is None:
        config = BacktestConfig()

    # ── Align ─────────────────────────────────────────────────
    common_idx = returns.index.intersection(positions.index)
    ret = returns.reindex(common_idx).fillna(0.0)
    pos = positions.reindex(common_idx).fillna(0.0)

    # ── Transaction costs ─────────────────────────────────────
    tc_rate  = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()   # entry cost on day 1
    tc = turnover * tc_rate

    # ── Gross portfolio return ────────────────────────────────
    gross_ret = (pos * ret).sum(axis=1)

    # ── Net return ────────────────────────────────────────────
    net_ret = gross_ret - tc

    # ── Cumulative P&L ────────────────────────────────────────
    cumulative = (1 + net_ret).cumprod() * config.initial_capital

    # ── Drawdown ──────────────────────────────────────────────
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max

    return pd.DataFrame(
        {
            "gross_return":     gross_ret,
            "transaction_cost": tc,
            "net_return":       net_ret,
            "cumulative_pnl":   cumulative,
            "drawdown":         drawdown,
            "turnover":         turnover,
            "gross_exposure":   pos.abs().sum(axis=1),
            "net_exposure":     pos.sum(axis=1),
        },
        index=common_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward(
    prices:         pd.DataFrame,
    returns:        pd.DataFrame,
    signal_fn,
    signal_kwargs:  dict,
    config:         BacktestConfig = None,
    train_window:   int = 504,   # ~2 years
    test_window:    int = 63,    # ~3 months
) -> pd.DataFrame:
    """
    Rolling walk-forward out-of-sample validation.

    At each step:
      1. Generate the signal using ALL history up to the end of the OOS window
         (parameters are FIXED — not re-optimised — to test signal robustness).
      2. Evaluate performance on the next `test_window` days only.
      3. Concatenate all OOS windows into a single results DataFrame.

    The IS / OOS Sharpe gap is the key diagnostic:
      gap < 0.3  → signal is robust
      gap > 0.5  → possible overfitting or regime fragility

    Parameters
    ----------
    train_window : minimum history required before first OOS window
    test_window  : length of each OOS evaluation window

    Returns
    -------
    Concatenated OOS results DataFrame (same schema as run_backtest output).
    """
    from backtester.signals import signal_to_positions

    if config is None:
        config = BacktestConfig()

    oos_pieces = []
    dates      = prices.index
    n          = len(dates)
    n_folds    = (n - train_window) // test_window

    print(f"Walk-forward: {n_folds} folds  |  "
          f"train={train_window}d  test={test_window}d")

    for fold in range(n_folds):
        oos_start = train_window + fold * test_window
        oos_end   = min(oos_start + test_window, n)

        oos_idx        = dates[oos_start:oos_end]
        history_prices = prices.iloc[:oos_end]   # expanding window

        sig = signal_fn(history_prices, **signal_kwargs)
        pos = signal_to_positions(sig)

        oos_ret = returns.loc[oos_idx]
        oos_pos = pos.reindex(oos_idx).fillna(0.0)

        res = run_backtest(oos_ret, oos_pos, config)
        oos_pieces.append(res)

        if (fold + 1) % 5 == 0:
            print(f"  fold {fold + 1:>3}/{n_folds} done")

    combined = pd.concat(oos_pieces)
    print("Walk-forward complete.")
    return combined
