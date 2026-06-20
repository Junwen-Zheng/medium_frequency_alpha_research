from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_cross_section(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def clip_cross_sectional_zscores(df: pd.DataFrame, cols: list[str], limit: float = 5.0) -> pd.DataFrame:
    """Clip feature z-scores to reduce the impact of extreme outliers.

    This is intentionally lightweight so the demo can run quickly while still
    documenting an explicit outlier policy.
    """
    out = df.copy()
    for col in cols:
        out[col] = out[col].clip(-limit, limit)
    return out


def build_features(ohlcv: pd.DataFrame, horizon_days: int = 10) -> pd.DataFrame:
    """Build point-in-time features and forward relative-return targets.

    All features are lagged by one day so today's label cannot leak into the input.
    """
    df = ohlcv.copy().sort_index()
    g = df.groupby(level="ticker", group_keys=False)

    close = df["adj_close"]
    volume = df["volume"].replace(0, np.nan)
    returns_1d = g["adj_close"].pct_change()

    features = pd.DataFrame(index=df.index)
    features["ret_1d_lag1"] = returns_1d.groupby(level="ticker").shift(1)
    features["reversal_5d"] = g["adj_close"].pct_change(5).groupby(level="ticker").shift(1) * -1
    features["momentum_20d"] = g["adj_close"].pct_change(20).groupby(level="ticker").shift(1)
    features["momentum_60d"] = g["adj_close"].pct_change(60).groupby(level="ticker").shift(1)
    features["volatility_20d"] = returns_1d.groupby(level="ticker").rolling(20).std().droplevel(0).groupby(level="ticker").shift(1)
    features["volume_z_20d"] = volume.groupby(level="ticker").transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()).groupby(level="ticker").shift(1)
    features["dollar_volume_z_20d"] = (close * volume).groupby(level="ticker").transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()).groupby(level="ticker").shift(1)
    features["high_low_range"] = ((df["high"] - df["low"]) / df["adj_close"]).groupby(level="ticker").shift(1)

    # Forward return target: stock forward return minus cross-sectional average on that date.
    fwd = g["adj_close"].shift(-horizon_days) / df["adj_close"] - 1.0
    rel_fwd = fwd - fwd.groupby(level="date").transform("mean")
    features[f"target_rel_fwd_{horizon_days}d"] = rel_fwd

    feature_cols = [c for c in features.columns if c.startswith(("ret", "reversal", "momentum", "volatility", "volume", "dollar", "high"))]
    for col in feature_cols:
        features[col] = features.groupby(level="date")[col].transform(_zscore_cross_section)
    features = clip_cross_sectional_zscores(features, feature_cols)

    return features.dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]
