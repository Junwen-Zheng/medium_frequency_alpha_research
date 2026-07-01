import numpy as np
import pandas as pd

from src.backtest import (
    _forward_1d_returns,
    backtest_cost_sensitivity,
    backtest_long_short,
)


def _toy_scores_and_prices():
    dates = pd.bdate_range("2024-01-01", periods=12)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    score_rows = []
    price_rows = []

    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            score_rows.append((date, ticker, float(j)))

            # Make a deterministic but non-identical price path per ticker.
            base = 100.0 + 10.0 * j
            drift = 0.001 * (j - 2)
            price = base * ((1.0 + drift) ** i)
            price_rows.append((date, ticker, price))

    score_index = pd.MultiIndex.from_tuples(
        [(d, t) for d, t, _ in score_rows],
        names=["date", "ticker"],
    )
    scores = pd.Series([x for _, _, x in score_rows], index=score_index, name="score")

    price_index = pd.MultiIndex.from_tuples(
        [(d, t) for d, t, _ in price_rows],
        names=["date", "ticker"],
    )
    ohlcv = pd.DataFrame(
        {"adj_close": [x for _, _, x in price_rows]},
        index=price_index,
    )

    return scores, ohlcv


def test_forward_1d_returns_are_shifted_within_ticker():
    dates = pd.bdate_range("2024-01-01", periods=3)
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    ohlcv = pd.DataFrame(
        {
            "adj_close": [
                100.0,
                200.0,
                110.0,
                180.0,
                121.0,
                162.0,
            ]
        },
        index=index,
    )

    fwd = _forward_1d_returns(ohlcv)

    assert np.isclose(fwd.loc[(dates[0], "AAA")], 0.10)
    assert np.isclose(fwd.loc[(dates[0], "BBB")], -0.10)
    assert np.isclose(fwd.loc[(dates[1], "AAA")], 0.10)
    assert np.isclose(fwd.loc[(dates[1], "BBB")], -0.10)
    assert pd.isna(fwd.loc[(dates[2], "AAA")])
    assert pd.isna(fwd.loc[(dates[2], "BBB")])


def test_transaction_cost_sensitivity_penalizes_net_returns():
    scores, ohlcv = _toy_scores_and_prices()

    daily, summary = backtest_cost_sensitivity(
        scores,
        ohlcv,
        long_q=0.8,
        short_q=0.2,
        cost_bps_grid=[0, 5, 10, 25],
        rebalance_frequency=None,
    )

    assert summary["cost_bps"].tolist() == [0.0, 5.0, 10.0, 25.0]
    assert set(daily.index.get_level_values("cost_bps")) == {0.0, 5.0, 10.0, 25.0}
    assert (summary["average_daily_turnover"] > 0).all()

    zero_cost = daily.xs(0.0, level="cost_bps")
    high_cost = daily.xs(25.0, level="cost_bps")

    assert (high_cost["net_return"] <= zero_cost["net_return"] + 1e-12).all()
    assert (
        summary.loc[summary["cost_bps"] == 25.0, "total_cost_drag"].iloc[0]
        > summary.loc[summary["cost_bps"] == 0.0, "total_cost_drag"].iloc[0]
    )


def test_backtest_long_short_single_cost_api_still_works():
    scores, ohlcv = _toy_scores_and_prices()

    returns, metrics = backtest_long_short(
        scores,
        ohlcv,
        long_q=0.8,
        short_q=0.2,
        cost_bps=5,
    )

    assert isinstance(returns, pd.Series)
    assert returns.name == "net_return"
    assert "average_daily_turnover" in metrics
    assert "annualized_return" in metrics
