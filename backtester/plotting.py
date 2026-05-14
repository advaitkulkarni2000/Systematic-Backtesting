# backtester/plotting.py
"""
Clean matplotlib/seaborn plots for the README and notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


COLORS = {
    "momentum":      "#00e5a0",
    "mean_reversion":"#4d9aff",
    "benchmark":     "#aaaaaa",
    "drawdown":      "#ff6b6b",
}


def plot_strategy_comparison(
    results_dict: dict[str, pd.DataFrame],
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Plot cumulative PnL + drawdown for multiple strategies side-by-side.
    
    Args:
        results_dict: {"Strategy Name": results_df, ...}
    """
    colors = list(COLORS.values())
    fig = plt.figure(figsize=figsize, facecolor="#0a0c10")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax_pnl = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_pnl)

    for i, (name, res) in enumerate(results_dict.items()):
        c = colors[i % len(colors)]
        # Normalise to 100 for comparability
        norm_pnl = res["cumulative_pnl"] / res["cumulative_pnl"].iloc[0] * 100
        ax_pnl.plot(norm_pnl.index, norm_pnl.values, label=name, color=c, lw=1.5)
        ax_dd.fill_between(res.index, res["drawdown"] * 100, 0,
                           alpha=0.4, color=c)

    for ax in [ax_pnl, ax_dd]:
        ax.set_facecolor("#111318")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_color("#1f2430")
        ax.yaxis.label.set_color("white")

    ax_pnl.set_ylabel("Portfolio Value (rebased to 100)", color="white")
    ax_pnl.legend(facecolor="#111318", labelcolor="white", framealpha=0.8)
    ax_pnl.axhline(100, color="#333", linestyle="--", lw=0.7)
    ax_pnl.set_title("Systematic Strategy Backtest — S&P 500 Universe",
                     color="white", pad=12, fontsize=13)

    ax_dd.set_ylabel("Drawdown (%)", color="white")
    ax_dd.set_xlabel("Date", color="white")
    ax_dd.axhline(0, color="#333", lw=0.5)

    plt.tight_layout()
    return fig


def plot_signal_heatmap(
    signals: pd.DataFrame,
    title: str = "Signal Heatmap",
    figsize: tuple = (16, 6)
) -> plt.Figure:
    """Heatmap of signal values over time — useful for spotting clustering/regime effects."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0a0c10")
    ax.set_facecolor("#111318")

    sample = signals.iloc[-252:].T  # last year, transposed
    sns.heatmap(
        sample, ax=ax, cmap="RdYlGn", center=0,
        linewidths=0, xticklabels=False,
        cbar_kws={"shrink": 0.5}
    )
    ax.set_title(title, color="white", pad=8)
    ax.tick_params(colors="white")
    return fig
