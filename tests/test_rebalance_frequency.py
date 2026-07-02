import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_rebalance_frequency_sensitivity,
    make_market_neutral_weights,
)


def _rotating_scores_and_prices():
    dates = pd.bdate_range("2024-01-01", periods=28)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]

    score_rows = []
    price_rows = []

    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            score_rows.append((date, ticker, float((j + i) % len(tickers))))

            base = 100.0 + 5.0 * j
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


def test_rebalance_frequency_sensitivity_supported_frequencies_work():
    scores, ohlcv = _rotating_scores_and_prices()

    summary = backtest_rebalance_frequency_sensitivity(
        scores,
        ohlcv,
        long_q=0.8,
        short_q=0.2,
        cost_bps_grid=[0, 10],
        rebalance_frequencies=["daily", "weekly", "biweekly", "monthly"],
    )

    assert set(summary["rebalance_frequency"]) == {
        "daily",
        "weekly",
        "biweekly",
        "monthly",
    }
    assert set(summary["cost_bps"]) == {0.0, 10.0}
    assert "net_sharpe" in summary.columns
    assert "gross_sharpe" in summary.columns
    assert "num_rebalance_dates" in summary.columns


def test_invalid_rebalance_frequency_raises_clear_error():
    scores, ohlcv = _rotating_scores_and_prices()

    with pytest.raises(ValueError, match="Unsupported rebalance_frequency"):
        backtest_rebalance_frequency_sensitivity(
            scores,
            ohlcv,
            long_q=0.8,
            short_q=0.2,
            cost_bps_grid=[0],
            rebalance_frequencies=["hourly"],
        )


def test_slower_rebalance_has_fewer_rebalance_dates_and_turnover():
    scores, ohlcv = _rotating_scores_and_prices()

    summary = backtest_rebalance_frequency_sensitivity(
        scores,
        ohlcv,
        long_q=0.8,
        short_q=0.2,
        cost_bps_grid=[0],
        rebalance_frequencies=["daily", "weekly", "monthly"],
    )

    rows = summary.set_index("rebalance_frequency")

    assert rows.loc["daily", "num_rebalance_dates"] >= rows.loc["weekly", "num_rebalance_dates"]
    assert rows.loc["weekly", "num_rebalance_dates"] >= rows.loc["monthly", "num_rebalance_dates"]

    assert rows.loc["daily", "average_daily_turnover"] >= rows.loc["weekly", "average_daily_turnover"]
    assert rows.loc["weekly", "average_daily_turnover"] >= rows.loc["monthly", "average_daily_turnover"]


def test_weekly_weights_hold_between_rebalance_dates():
    scores, _ = _rotating_scores_and_prices()

    daily_weights = make_market_neutral_weights(
        scores,
        long_q=0.8,
        short_q=0.2,
        rebalance_frequency=None,
    )
    weekly_weights = make_market_neutral_weights(
        scores,
        long_q=0.8,
        short_q=0.2,
        rebalance_frequency="W-FRI",
    )

    assert not daily_weights.empty
    assert not weekly_weights.empty
    assert weekly_weights.unstack("ticker").diff().abs().sum(axis=1).mean() <= daily_weights.unstack("ticker").diff().abs().sum(axis=1).mean()
