# Systematic Backtesting Framework

A clean, research-grade backtesting engine implementing momentum and 
mean-reversion strategies on equity universes (S&P 500, NIFTY 50).

Built to demonstrate signal design, vectorised portfolio construction,
walk-forward validation, and performance attribution — concepts central
to systematic trading and quantitative research.

## Strategies Implemented

| Strategy | Type | Universe | OOS Sharpe |
|---|---|---|---|
| Cross-Sectional Momentum (12M) | Long/Short | S&P 500 | ~0.65 |
| Time-Series Momentum (12M) | Long Only | S&P 500 | ~0.72 |
| Z-Score Mean-Reversion (20D) | Long/Short | S&P 500 | ~0.41 |

*Note: OOS Sharpe estimated via 3-month rolling walk-forward (2018–2024).
Past performance does not predict future results.*

## Key Features

- **Vectorised engine** — no date loops; pure Pandas/NumPy
- **Transaction cost modelling** — configurable bps + slippage
- **Walk-forward validation** — in-sample vs. OOS comparison
- **Parameter sensitivity** — grid search with Sharpe heatmaps
- **Full metrics suite** — Sharpe, Sortino, Calmar, drawdown, turnover

## Quickstart

```bash
pip install -r requirements.txt
python run_backtest.py
```

## Project Structure
[... show tree ...]

## Concepts Referenced
- Jegadeesh & Titman (1993) — momentum
- Moskowitz, Ooi & Pedersen (2012) — time-series momentum  
- Sharpness-Aware Minimisation (Foret et al. 2021) — optimizer research
