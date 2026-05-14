# backtester/data_loader.py
"""
Fetch and cache OHLCV data using yfinance.
Supports S&P 500 (via SPY components or a curated list)
and NIFTY 50 symbols.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── A curated 20-stock universe (extend as needed) ─────────────
SP500_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "GS", "BAC", "MS",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
    "WMT", "PG", "KO", "SPY", "QQQ"
]

NIFTY50_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS"
]


def fetch_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    cache_dir: str = "data/",
    field: str = "Adj Close"
) -> pd.DataFrame:
    """
    Download adjusted close prices and cache locally as parquet.
    Returns a DataFrame with dates as index, tickers as columns.
    """
    cache_path = Path(cache_dir) / f"prices_{'_'.join(tickers[:3])}_{start[:4]}_{end[:4]}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading {len(tickers)} tickers from yfinance...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices.to_parquet(cache_path)
    logger.info(f"Saved to {cache_path}")
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute daily returns from price series.
    method: 'log' (log returns) or 'simple' (pct change)
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def load_universe(
    universe: str = "sp500",
    start: str = "2015-01-01",
    end: str = "2024-12-31"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function. Returns (prices, log_returns)."""
    tickers = SP500_UNIVERSE if universe == "sp500" else NIFTY50_UNIVERSE
    prices = fetch_prices(tickers, start=start, end=end)
    returns = compute_returns(prices, method="log")
    return prices, returns


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices, returns = load_universe("sp500", "2018-01-01", "2024-12-31")
    print(prices.tail())
    print(f"\nShape: {prices.shape}")
    print(f"Missing values: {prices.isna().sum().sum()}")
