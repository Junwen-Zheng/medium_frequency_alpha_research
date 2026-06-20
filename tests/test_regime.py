from src.data import make_synthetic_ohlcv
from src.regime import market_regime_labels


def test_market_regime_labels_are_date_level_and_nonempty():
    df = make_synthetic_ohlcv(["AAA", "BBB", "CCC", "DDD", "EEE"], "2019-01-01", "2021-01-01")
    regimes = market_regime_labels(df, lookback=20)
    assert regimes.index.name == "date"
    assert len(regimes) > 0
    assert regimes.nunique() >= 1
