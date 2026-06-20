from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_cross_section(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while avoiding infinite values from zero/near-zero volatility."""
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


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
    The feature set now includes two hypothesis families:

    1. Raw price/volume behaviour: reversal, momentum, volatility, liquidity.
    2. Volatility-adjusted price action: reversal/momentum normalized by recent
       realized volatility to test whether risk-normalized moves are more stable.

    The second family is intentionally modest. It is not presented as a discovered
    alpha; it is a concrete implementation of a follow-up hypothesis that can be
    evaluated against the raw baseline.
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

    # Baseline raw price / volume family.
    features["ret_1d_lag1"] = returns_1d.groupby(level="ticker").shift(1)
    features["reversal_5d"] = raw_ret_5d.groupby(level="ticker").shift(1) * -1
    features["momentum_20d"] = raw_ret_20d.groupby(level="ticker").shift(1)
    features["momentum_60d"] = raw_ret_60d.groupby(level="ticker").shift(1)
    features["volatility_20d"] = vol_20d_raw.groupby(level="ticker").shift(1)
    features["volume_z_20d"] = volume.groupby(level="ticker").transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()).groupby(level="ticker").shift(1)
    features["dollar_volume_z_20d"] = (close * volume).groupby(level="ticker").transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()).groupby(level="ticker").shift(1)
    features["high_low_range"] = ((df["high"] - df["low"]) / df["adj_close"]).groupby(level="ticker").shift(1)

    # Follow-up hypothesis family: volatility-adjusted momentum / reversal.
    # These features directly implement the second hypothesis documented in the
    # research log rather than leaving it as a TODO.
    rev_5d = raw_ret_5d * -1
    mom_20d = raw_ret_20d
    mom_60d = raw_ret_60d
    features["reversal_5d_vol_adj"] = _safe_divide(rev_5d, vol_20d_raw).groupby(level="ticker").shift(1)
    features["momentum_20d_vol_adj"] = _safe_divide(mom_20d, vol_20d_raw).groupby(level="ticker").shift(1)
    features["momentum_60d_vol_adj"] = _safe_divide(mom_60d, vol_60d_raw).groupby(level="ticker").shift(1)
    features["liquidity_adjusted_momentum"] = (
        features["momentum_20d_vol_adj"] * features["dollar_volume_z_20d"].fillna(0)
    )

    # Forward return target: stock forward return minus cross-sectional average on that date.
    fwd = g["adj_close"].shift(-horizon_days) / df["adj_close"] - 1.0
    rel_fwd = fwd - fwd.groupby(level="date").transform("mean")
    features[f"target_rel_fwd_{horizon_days}d"] = rel_fwd

    feature_cols = feature_columns(features)
    for col in feature_cols:
        features[col] = features.groupby(level="date")[col].transform(_zscore_cross_section)
    features = clip_cross_sectional_zscores(features, feature_cols)

    return features.dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]


def feature_family_columns() -> dict[str, list[str]]:
    """Named feature families used by the hypothesis-comparison diagnostics."""
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
    """Create a simple interpretable score from a feature family.

    The point is not to optimize alpha. This gives a transparent baseline for
    asking whether a hypothesis family contains directional information before
    fitting more flexible models.
    """
    available = [c for c in cols if c in frame.columns]
    if not available:
        return pd.Series(dtype=float, name="family_score")
    return frame[available].mean(axis=1).rename("family_score")
