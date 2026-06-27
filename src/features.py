from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_cross_section(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score for one date."""
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_z_by_ticker(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score computed inside each ticker history."""
    grouped = s.groupby(level="ticker")
    mean = grouped.rolling(window).mean().droplevel(0)
    std = grouped.rolling(window).std().droplevel(0)
    return _safe_divide(s - mean, std)


def clip_cross_sectional_zscores(
    df: pd.DataFrame,
    cols: list[str],
    limit: float = 5.0,
) -> pd.DataFrame:
    """Clip extreme feature z-scores after cross-sectional scaling."""
    out = df.copy()
    for col in cols:
        out[col] = out[col].clip(-limit, limit)
    return out


def forward_return_by_ticker(adj_close: pd.Series, horizon_days: int) -> pd.Series:
    """Forward return from t to t+h, shifted only within each ticker.

    This helper exists because a global Series.shift after a group operation can
    accidentally move values across ticker boundaries. Every return label must
    be computed inside the stock's own time series.
    """
    future = adj_close.groupby(level="ticker").shift(-horizon_days)
    return (future / adj_close - 1.0).rename(f"fwd_{horizon_days}d")


def relative_forward_return(ohlcv: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Cross-sectionally demeaned forward return target."""
    adj = ohlcv["adj_close"].sort_index()
    fwd = forward_return_by_ticker(adj, horizon_days)
    market_mean = fwd.groupby(level="date").transform("mean")
    return (fwd - market_mean).rename(f"target_rel_fwd_{horizon_days}d")


def build_features(ohlcv: pd.DataFrame, horizon_days: int = 10) -> pd.DataFrame:
    """Build point-in-time features and cross-sectional forward-return targets.

    Design rule: features for date t may use information available through t-1.
    Targets are future returns from t to t+h, computed independently for each
    ticker and then demeaned by the same-date cross-section.
    """
    df = ohlcv.copy().sort_index()
    g = df.groupby(level="ticker", group_keys=False)

    close = df["adj_close"]
    volume = df["volume"].replace(0, np.nan)

    returns_1d = g["adj_close"].pct_change()
    raw_ret_5d = g["adj_close"].pct_change(5)
    raw_ret_20d = g["adj_close"].pct_change(20)
    raw_ret_60d = g["adj_close"].pct_change(60)

    vol_20d_raw = returns_1d.groupby(level="ticker").rolling(20).std().droplevel(0)
    vol_60d_raw = returns_1d.groupby(level="ticker").rolling(60).std().droplevel(0)

    features = pd.DataFrame(index=df.index)

    # Raw price/volume hypothesis family.
    features["ret_1d_lag1"] = returns_1d.groupby(level="ticker").shift(1)
    features["reversal_5d"] = (raw_ret_5d * -1).groupby(level="ticker").shift(1)
    features["momentum_20d"] = raw_ret_20d.groupby(level="ticker").shift(1)
    features["momentum_60d"] = raw_ret_60d.groupby(level="ticker").shift(1)
    features["volatility_20d"] = vol_20d_raw.groupby(level="ticker").shift(1)
    features["volume_z_20d"] = _rolling_z_by_ticker(volume, 20).groupby(level="ticker").shift(1)
    features["dollar_volume_z_20d"] = _rolling_z_by_ticker(close * volume, 20).groupby(level="ticker").shift(1)
    features["high_low_range"] = ((df["high"] - df["low"]) / df["adj_close"]).groupby(level="ticker").shift(1)

    # Follow-up hypothesis family: risk-normalized price action.
    features["reversal_5d_vol_adj"] = _safe_divide(raw_ret_5d * -1, vol_20d_raw).groupby(level="ticker").shift(1)
    features["momentum_20d_vol_adj"] = _safe_divide(raw_ret_20d, vol_20d_raw).groupby(level="ticker").shift(1)
    features["momentum_60d_vol_adj"] = _safe_divide(raw_ret_60d, vol_60d_raw).groupby(level="ticker").shift(1)
    features["liquidity_adjusted_momentum"] = (
        features["momentum_20d_vol_adj"] * features["dollar_volume_z_20d"].fillna(0)
    )

    target = relative_forward_return(df, horizon_days)
    features[target.name] = target

    feature_cols = feature_columns(features)
    for col in feature_cols:
        features[col] = features.groupby(level="date")[col].transform(_zscore_cross_section)

    features = clip_cross_sectional_zscores(features, feature_cols)
    return features.dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]


def feature_family_columns() -> dict[str, list[str]]:
    return {
        "raw_price_volume": [
            "reversal_5d",
            "momentum_20d",
            "momentum_60d",
            "volume_z_20d",
            "dollar_volume_z_20d",
        ],
        "volatility_adjusted_price_action": [
            "reversal_5d_vol_adj",
            "momentum_20d_vol_adj",
            "momentum_60d_vol_adj",
            "liquidity_adjusted_momentum",
        ],
    }


def composite_family_score(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Transparent equal-weight score for a hypothesis family."""
    available = [c for c in cols if c in frame.columns]
    if not available:
        return pd.Series(dtype=float, name="family_score")
    return frame[available].mean(axis=1).rename("family_score")
