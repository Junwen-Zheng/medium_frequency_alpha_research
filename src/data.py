from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    tickers: int
    start: str
    end: str
    missing_close_pct: float
    duplicate_rows: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def download_ohlcv(tickers: Iterable[str], start: str, end: str, cache_path: str | Path) -> pd.DataFrame:
    """Download public OHLCV data using yfinance and cache it locally.

    Returns a tidy DataFrame indexed by date/ticker with columns:
    open, high, low, close, adj_close, volume.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for real-data mode. Install requirements.txt.") from exc

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)
    if raw.empty:
        raise RuntimeError("No data downloaded. Check tickers, dates, or internet access.")

    frames = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            df = raw[ticker].copy()
        else:
            df = raw.copy()
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume",
        })
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        df = df[keep]
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df.index)
        frames.append(df.reset_index(drop=True))

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["adj_close"]).sort_values(["date", "ticker"])
    out = out.set_index(["date", "ticker"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def make_synthetic_ohlcv(tickers: Iterable[str], start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Offline synthetic dataset for tests and smoke-test runs only.

    The synthetic generator creates correlated returns and volume regimes. It is not a substitute
    for genuine data and should not be used as evidence for signal quality.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    tickers = list(tickers)
    market = rng.normal(0.00025, 0.012, len(dates))
    rows = []
    for i, ticker in enumerate(tickers):
        beta = rng.uniform(0.7, 1.3)
        idio = rng.normal(0, 0.018, len(dates))
        # weak momentum/reversal pattern to give diagnostics something to find
        ret = beta * market + idio
        price = 100 * np.exp(np.cumsum(ret))
        vol_regime = 1.0 + 0.5 * (np.abs(market) > np.quantile(np.abs(market), 0.8))
        volume = rng.lognormal(mean=14.0 + i * 0.03, sigma=0.35, size=len(dates)) * vol_regime
        high = price * (1 + np.abs(rng.normal(0.004, 0.004, len(dates))))
        low = price * (1 - np.abs(rng.normal(0.004, 0.004, len(dates))))
        open_ = price * (1 + rng.normal(0, 0.003, len(dates)))
        for d, o, h, l, c, v in zip(dates, open_, high, low, price, volume):
            rows.append((d, ticker, o, h, l, c, c, v))
    df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    return df.set_index(["date", "ticker"]).sort_index()


def quality_report(df: pd.DataFrame) -> DataQualityReport:
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    return DataQualityReport(
        rows=len(df),
        tickers=tickers.nunique(),
        start=str(dates.min().date()),
        end=str(dates.max().date()),
        missing_close_pct=float(df["adj_close"].isna().mean()),
        duplicate_rows=int(df.index.duplicated().sum()),
    )
