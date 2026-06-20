import pandas as pd

from src.data import make_synthetic_ohlcv
from src.features import build_features, feature_columns


def test_feature_pipeline_has_no_missing_targets_after_dropna():
    df = make_synthetic_ohlcv(["AAA", "BBB", "CCC", "DDD", "EEE"], "2020-01-01", "2021-01-01")
    frame = build_features(df, horizon_days=5)
    assert len(frame) > 0
    assert "target_rel_fwd_5d" in frame.columns
    assert frame["target_rel_fwd_5d"].isna().sum() == 0
    assert len(feature_columns(frame)) >= 4


def test_features_are_lagged_and_do_not_use_same_day_return():
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    df = make_synthetic_ohlcv(tickers, "2020-01-01", "2020-06-30")
    frame = build_features(df, horizon_days=5)

    # ret_1d_lag1 should not equal same-day return. This is a lightweight leakage guard.
    same_day_return = df["adj_close"].groupby(level="ticker").pct_change().reindex(frame.index)
    overlap = pd.concat([frame["ret_1d_lag1"], same_day_return.rename("same_day")], axis=1).dropna()
    assert not overlap["ret_1d_lag1"].equals(overlap["same_day"])


def test_target_is_cross_sectionally_centered():
    df = make_synthetic_ohlcv(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], "2020-01-01", "2021-01-01")
    frame = build_features(df, horizon_days=10)
    daily_mean = frame["target_rel_fwd_10d"].groupby(level="date").mean()
    assert daily_mean.abs().max() < 1e-10


def test_volatility_adjusted_feature_family_exists():
    df = make_synthetic_ohlcv(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], "2020-01-01", "2021-06-30")
    frame = build_features(df, horizon_days=10)
    assert "reversal_5d_vol_adj" in frame.columns
    assert "momentum_20d_vol_adj" in frame.columns
    assert "liquidity_adjusted_momentum" in frame.columns
    assert frame[["reversal_5d_vol_adj", "momentum_20d_vol_adj", "liquidity_adjusted_momentum"]].isna().sum().sum() == 0
