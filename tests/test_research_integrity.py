import numpy as np
import pandas as pd

from src.backtest import forward_returns_by_ticker
from src.features import forward_return_by_ticker, relative_forward_return
from src.data import make_synthetic_ohlcv
from src.features import build_features


def _toy_ohlcv() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    rows = []
    prices = {
        "AAA": [100.0, 110.0, 121.0],
        "BBB": [200.0, 100.0, 50.0],
    }
    for date_idx, date in enumerate(dates):
        for ticker in ["AAA", "BBB"]:
            px = prices[ticker][date_idx]
            rows.append((date, ticker, px, px, px, px, px, 1000.0))
    df = pd.DataFrame(
        rows,
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
    )
    return df.set_index(["date", "ticker"]).sort_index()


def test_forward_returns_are_shifted_within_each_ticker():
    ohlcv = _toy_ohlcv()
    fwd = forward_returns_by_ticker(ohlcv, horizon_days=1)

    assert np.isclose(fwd.loc[(pd.Timestamp("2020-01-01"), "AAA")], 0.10)
    assert np.isclose(fwd.loc[(pd.Timestamp("2020-01-01"), "BBB")], -0.50)
    assert np.isclose(fwd.loc[(pd.Timestamp("2020-01-02"), "AAA")], 0.10)
    assert np.isclose(fwd.loc[(pd.Timestamp("2020-01-02"), "BBB")], -0.50)
    assert pd.isna(fwd.loc[(pd.Timestamp("2020-01-03"), "AAA")])
    assert pd.isna(fwd.loc[(pd.Timestamp("2020-01-03"), "BBB")])


def test_feature_forward_return_helper_matches_backtest_forward_return_helper():
    ohlcv = _toy_ohlcv()
    feature_fwd = forward_return_by_ticker(ohlcv["adj_close"], horizon_days=1)
    backtest_fwd = forward_returns_by_ticker(ohlcv, horizon_days=1)
    pd.testing.assert_series_equal(feature_fwd, backtest_fwd, check_names=False)


def test_relative_forward_return_is_cross_sectionally_centered_by_date():
    ohlcv = _toy_ohlcv()
    rel = relative_forward_return(ohlcv, horizon_days=1).dropna()
    daily_mean = rel.groupby(level="date").mean()
    assert daily_mean.abs().max() < 1e-12


def test_build_features_keeps_target_cross_section_centered():
    df = make_synthetic_ohlcv(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"], "2020-01-01", "2021-06-30")
    frame = build_features(df, horizon_days=10)
    daily_mean = frame["target_rel_fwd_10d"].groupby(level="date").mean()
    assert daily_mean.abs().max() < 1e-10
