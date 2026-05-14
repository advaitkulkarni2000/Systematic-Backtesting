"""
run_backtest.py — command-line entry point

Runs all three strategies on the full 100-stock universe,
prints a comparison table and saves results to results/.

Usage:
    python run_backtest.py

Optional flags:
    --no-walkforward   skip walk-forward validation (faster)
    --no-sensitivity   skip parameter grid search (faster)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtester.data_loader import load_universe
from backtester.signals import (
    cross_sectional_momentum,
    time_series_momentum,
    zscore_mean_reversion,
    signal_to_positions,
)
from backtester.engine  import run_backtest, walk_forward, BacktestConfig
from backtester.metrics import summarise, sharpe_ratio

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

Path("results").mkdir(exist_ok=True)


def main(run_wf: bool = True, run_sens: bool = True) -> None:

    # ── 1. Data ───────────────────────────────────────────────────────────
    log.info("Loading universe …")
    prices, returns, dropped = load_universe()
    log.info(f"Universe: {prices.shape[1]} tickers × {prices.shape[0]} days")
    if dropped:
        log.warning(f"Dropped {len(dropped)} tickers: {dropped}")

    cfg = BacktestConfig(
        transaction_cost_bps=5,
        slippage_bps=2,
        initial_capital=1_000_000,
    )

    results_list = []

    # ── 2. Cross-Sectional Momentum ───────────────────────────────────────
    log.info("Running XS Momentum …")
    sig_xs = cross_sectional_momentum(prices, lookback=252, skip_last=21,
                                       n_long=10, n_short=10)
    pos_xs = signal_to_positions(sig_xs)
    res_xs = run_backtest(returns, pos_xs, cfg)
    results_list.append(summarise(res_xs, "XS Momentum (12M)"))

    # ── 3. Time-Series Momentum ───────────────────────────────────────────
    log.info("Running TS Momentum …")
    sig_ts = time_series_momentum(prices, lookback=252)
    pos_ts = signal_to_positions(sig_ts)
    res_ts = run_backtest(returns, pos_ts, cfg)
    results_list.append(summarise(res_ts, "TS Momentum (12M)"))

    # ── 4. Z-Score Mean-Reversion ─────────────────────────────────────────
    log.info("Running Z-Score Mean-Reversion …")
    sig_mr = zscore_mean_reversion(prices, lookback=20, entry_z=1.5)
    pos_mr = signal_to_positions(sig_mr)
    res_mr = run_backtest(returns, pos_mr, cfg)
    results_list.append(summarise(res_mr, "Z-Score Mean-Rev (20D)"))

    # ── 5. SPY Buy-and-Hold benchmark ─────────────────────────────────────
    if "SPY" in returns.columns:
        log.info("Running SPY benchmark …")
        spy_ret = returns[["SPY"]]
        spy_pos = pd.DataFrame(1.0, index=returns.index, columns=["SPY"])
        cfg_bh  = BacktestConfig(transaction_cost_bps=0, slippage_bps=0,
                                  initial_capital=1_000_000)
        res_spy = run_backtest(spy_ret, spy_pos, cfg_bh)
        results_list.append(summarise(res_spy, "SPY Buy & Hold"))

    # ── 6. Comparison table ───────────────────────────────────────────────
    comp = pd.DataFrame(results_list).set_index("Label")
    display_cols = [
        "Ann. Return (%)", "Ann. Vol (%)", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Max Drawdown (%)",
        "Hit Rate (%)", "Avg Daily Turnover", "Final PnL ($M)",
    ]
    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON")
    print("=" * 90)
    print(comp[display_cols].to_string())
    print("=" * 90)
    comp.to_csv("results/strategy_comparison.csv")
    log.info("Saved results/strategy_comparison.csv")

    # ── 7. Walk-forward OOS validation ───────────────────────────────────
    if run_wf:
        log.info("Running walk-forward validation (XS Momentum) …")
        res_oos = walk_forward(
            prices, returns,
            cross_sectional_momentum,
            {"lookback": 252, "skip_last": 21, "n_long": 10, "n_short": 10},
            cfg, train_window=504, test_window=63,
        )
        m_is  = summarise(res_xs,  "XS Momentum — In-Sample")
        m_oos = summarise(res_oos, "XS Momentum — OOS (Walk-Forward)")
        wf    = pd.DataFrame([m_is, m_oos]).set_index("Label")
        print("\nIN-SAMPLE vs. OUT-OF-SAMPLE")
        print(wf[["Ann. Return (%)", "Ann. Vol (%)", "Sharpe Ratio",
                   "Max Drawdown (%)", "Hit Rate (%)"]].to_string())
        print(f"Sharpe degradation: {m_is['Sharpe Ratio'] - m_oos['Sharpe Ratio']:.3f}")
        wf.to_csv("results/walkforward_is_vs_oos.csv")
        log.info("Saved results/walkforward_is_vs_oos.csv")

    # ── 8. Parameter sensitivity ─────────────────────────────────────────
    if run_sens:
        import itertools
        log.info("Running parameter sensitivity grid search …")
        records = []
        for lb, ns in itertools.product([63, 126, 189, 252, 315], [5, 7, 10, 15]):
            sig = cross_sectional_momentum(prices, lookback=lb, n_long=ns, n_short=ns)
            pos = signal_to_positions(sig)
            res = run_backtest(returns, pos, cfg)
            records.append({
                "lookback": lb, "n_long/short": ns,
                "sharpe":   round(sharpe_ratio(res["net_return"]), 3),
            })
        sens = pd.DataFrame(records)
        pivot = sens.pivot(index="lookback", columns="n_long/short", values="sharpe")
        print("\nSHARPE RATIO — PARAMETER SENSITIVITY")
        print(pivot.to_string())
        pivot.to_csv("results/parameter_sensitivity.csv")
        log.info("Saved results/parameter_sensitivity.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-walkforward",  action="store_true")
    parser.add_argument("--no-sensitivity",  action="store_true")
    args = parser.parse_args()
    main(run_wf=not args.no_walkforward, run_sens=not args.no_sensitivity)
