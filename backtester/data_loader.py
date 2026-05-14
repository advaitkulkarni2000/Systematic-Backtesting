"""
backtester/data_loader.py

Downloads and caches adjusted close prices for the full 100-stock universe
(90 US tickers + 10 NIFTY 50 names) used in the backtesting notebook.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── Full 100-stock universe (matches notebook exactly) ─────────────────────
TICKERS = [
    # Tech / Mega-cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM",  "GS",   "BAC",
    "MS",   "AMD",  "NFLX", "AVGO", "ORCL",
    "CRM",  "ADBE", "CSCO", "QCOM", "INTC",
    "IBM",  "NOW",  "UBER", "PLTR", "SHOP",
    # Consumer
    "WMT",  "COST", "HD",   "LOW",  "NKE",
    "KO",   "PEP",  "MCD",  "SBUX", "DIS",
    # Energy
    "XOM",  "CVX",  "SLB",  "COP",  "EOG",
    # Healthcare
    "JNJ",  "PFE",  "MRK",  "ABBV", "UNH",
    "LLY",  "TMO",  "DHR",  "ABT",  "BMY",
    # ETFs (sector + broad market)
    "SPY",  "QQQ",  "IWM",  "DIA",  "XLK",
    "XLF",  "XLE",  "XLV",  "XLY",  "XLP",
    # Industrials / Defence
    "CAT",  "DE",   "GE",   "HON",  "RTX",
    "LMT",  "BA",   "MMM",  "UPS",  "FDX",
    # Telecom / Media
    "T",    "VZ",   "TMUS", "CMCSA","CHTR",
    # Financials / Payments
    "V",    "MA",   "PYPL", "AXP",  "BLK",
    "SCHW", "C",    "BK",   "USB",  "PNC",
    # NIFTY 50 (India)
    "RELIANCE.NS", "TCS.NS",      "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS","SBIN.NS",
    "ITC.NS",      "LT.NS",       "BHARTIARTL.NS", "AXISBANK.NS",
]

START = "2015-01-01"
END   = "2024-12-31"


def fetch_prices(
    tickers: list = TICKERS,
    start:   str  = START,
    end:     str  = END,
    cache_dir: str = "data/",
) -> pd.DataFrame:
    """
    Download adjusted close prices and cache as parquet.
    Subsequent calls load from disk — no re-download needed.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex rows, ticker columns.
        Forward-filled up to 10 days; no NaNs remaining.
    """
    cache_path = Path(cache_dir) / "prices_100stocks_2015_2024.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading {len(tickers)} tickers ({start} → {end}) ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices.index = pd.to_datetime(prices.index)
    prices.to_parquet(cache_path)
    logger.info(f"Cached to {cache_path}")
    return prices


def clean_prices(
    df: pd.DataFrame,
    min_history_pct: float = 0.6,
    max_gap_days:    int   = 10,
) -> tuple:
    """
    1. Drop tickers with < min_history_pct of trading days populated.
    2. Forward-fill short gaps (holidays, halts) up to max_gap_days.
    3. Drop dates where >50 % of universe has no data.
    4. Final ffill + bfill to remove any remaining NaNs.

    Returns (clean_df, dropped_tickers).
    """
    min_days = int(len(df) * min_history_pct)
    counts   = df.notna().sum()
    keep     = counts[counts >= min_days].index.tolist()
    dropped  = counts[counts <  min_days].index.tolist()

    clean = df[keep].ffill(limit=max_gap_days)
    clean = clean[clean.notna().mean(axis=1) >= 0.5]
    clean = clean.ffill().bfill()

    assert clean.isna().sum().sum() == 0, "NaNs remain after cleaning"
    return clean, dropped


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Compute daily returns.

    Parameters
    ----------
    method : 'log'    → ln(P_t / P_{t-1})   — time-additive, used throughout
             'simple' → (P_t / P_{t-1}) - 1  — for reference/plotting
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def load_universe(
    tickers:   list = TICKERS,
    start:     str  = START,
    end:       str  = END,
    cache_dir: str  = "data/",
) -> tuple:
    """
    Convenience loader used by run_backtest.py and notebooks.

    Returns
    -------
    (prices, log_returns, dropped_tickers)
    """
    raw              = fetch_prices(tickers, start, end, cache_dir)
    prices, dropped  = clean_prices(raw)
    log_returns      = compute_returns(prices, method="log")
    return prices, log_returns, dropped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices, returns, dropped = load_universe()
    print(f"Universe  : {prices.shape[1]} tickers  |  {prices.shape[0]} trading days")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Dropped   : {dropped}")
    print(returns.describe().round(5))
