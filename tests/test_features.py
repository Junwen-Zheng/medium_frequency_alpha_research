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
