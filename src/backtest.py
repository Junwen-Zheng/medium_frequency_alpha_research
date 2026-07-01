from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns_by_ticker(
    ohlcv: pd.DataFrame,
    horizon_days: int = 1,
    price_col: str = "adj_close",
) -> pd.Series:
    """Compute forward close-to-close returns within each ticker.

    This public helper is kept for research-integrity tests and to make the
    no-cross-ticker-leakage behavior explicit.
    """
    prices = ohlcv[price_col].sort_index()
    fwd = prices.groupby(level="ticker").shift(-horizon_days).div(prices).sub(1.0)
    return fwd.rename(f"fwd_{horizon_days}d_return")


def _forward_1d_returns(ohlcv: pd.DataFrame) -> pd.Series:
    """Compute next-day close-to-close returns within each ticker."""
    return forward_returns_by_ticker(
        ohlcv,
        horizon_days=1,
        price_col="adj_close",
    ).rename("fwd_1d_return")


def _daily_market_neutral_weights(
    scores: pd.Series,
    long_q: float,
    short_q: float,
) -> pd.Series:
    weights = []

    for _, s in scores.dropna().sort_index().groupby(level="date"):
        if len(s) < 5:
            continue

        lo = s.quantile(short_q)
        hi = s.quantile(long_q)

        w = pd.Series(0.0, index=s.index, dtype=float)

        longs = s[s >= hi].index
        shorts = s[s <= lo].index

        if len(longs) > 0:
            w.loc[longs] = 1.0 / len(longs)

        if len(shorts) > 0:
            w.loc[shorts] = -1.0 / len(shorts)

        weights.append(w)

    if not weights:
        return pd.Series(dtype=float, name="weight")

    return pd.concat(weights).sort_index().rename("weight")


def _period_end_rebalance_dates(
    dates: pd.DatetimeIndex,
    rebalance_frequency: str | None,
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)), name="date")

    if not rebalance_frequency:
        return dates

    if rebalance_frequency.upper() in {"D", "DAILY"}:
        return dates

    period_end_dates = (
        pd.Series(dates, index=dates)
        .groupby(pd.Grouper(freq=rebalance_frequency))
        .max()
        .dropna()
    )

    return pd.DatetimeIndex(period_end_dates.to_list(), name="date")


def make_market_neutral_weights(
    scores: pd.Series,
    long_q: float = 0.8,
    short_q: float = 0.2,
    rebalance_frequency: str | None = None,
) -> pd.Series:
    """Create market-neutral long/short weights from cross-sectional scores.

    If `rebalance_frequency` is provided, weights are formed only on period-end
    rebalance dates and then carried forward until the next rebalance. This
    makes the portfolio diagnostic closer to a tradable schedule than daily
    full re-ranking.
    """
    scores = scores.dropna().sort_index()

    if scores.empty:
        return pd.Series(dtype=float, name="weight")

    daily_weights = _daily_market_neutral_weights(scores, long_q, short_q)

    if daily_weights.empty:
        return daily_weights

    wide = daily_weights.unstack("ticker").fillna(0.0).sort_index()
    all_dates = pd.DatetimeIndex(
        sorted(pd.unique(scores.index.get_level_values("date"))),
        name="date",
    )

    if not rebalance_frequency or rebalance_frequency.upper() in {"D", "DAILY"}:
        return daily_weights

    rebalance_dates = _period_end_rebalance_dates(all_dates, rebalance_frequency)
    rebalance_dates = rebalance_dates.intersection(wide.index)

    if rebalance_dates.empty:
        return pd.Series(dtype=float, name="weight")

    rebalanced = wide.loc[rebalance_dates]
    expanded = rebalanced.reindex(all_dates).ffill().fillna(0.0)
    expanded.index.name = "date"

    return expanded.stack().sort_index().rename("weight")


def _portfolio_gross_returns_and_turnover(
    scores: pd.Series,
    ohlcv: pd.DataFrame,
    long_q: float,
    short_q: float,
    rebalance_frequency: str | None,
) -> tuple[pd.Series, pd.Series]:
    weights = make_market_neutral_weights(
        scores,
        long_q=long_q,
        short_q=short_q,
        rebalance_frequency=rebalance_frequency,
    )

    if weights.empty:
        empty = pd.Series(dtype=float)
        return empty.rename("gross_return"), empty.rename("turnover")

    wide_weights = weights.unstack("ticker").fillna(0.0).sort_index()

    fwd_1d = _forward_1d_returns(ohlcv)
    wide_fwd = (
        fwd_1d.unstack("ticker")
        .reindex(index=wide_weights.index, columns=wide_weights.columns)
    )

    gross = (
        wide_weights.mul(wide_fwd)
        .sum(axis=1, min_count=1)
        .fillna(0.0)
        .rename("gross_return")
    )

    turnover = wide_weights.diff().abs().sum(axis=1)

    if not turnover.empty:
        turnover.iloc[0] = wide_weights.iloc[0].abs().sum()

    turnover = turnover.rename("turnover")

    return gross, turnover


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    gross_returns: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict:
    returns = returns.dropna()

    if returns.empty:
        return {}

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    ann_ret = equity.iloc[-1] ** (252 / max(len(returns), 1)) - 1.0
    ann_vol = returns.std(ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0

    metrics = {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(ann_ret / (ann_vol + 1e-12)),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "average_daily_turnover": float(turnover.reindex(returns.index).mean()),
    }

    if gross_returns is not None:
        aligned_gross = gross_returns.reindex(returns.index)
        metrics["average_daily_gross_return"] = float(aligned_gross.mean())
        metrics["average_daily_net_return"] = float(returns.mean())

    if costs is not None:
        aligned_costs = costs.reindex(returns.index).fillna(0.0)
        metrics["average_daily_cost"] = float(aligned_costs.mean())
        metrics["total_cost_drag"] = float(aligned_costs.sum())

    return metrics


def backtest_cost_sensitivity(
    scores: pd.Series,
    ohlcv: pd.DataFrame,
    long_q: float,
    short_q: float,
    cost_bps_grid: list[float] | tuple[float, ...],
    rebalance_frequency: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the same portfolio under multiple transaction-cost assumptions."""
    gross, turnover = _portfolio_gross_returns_and_turnover(
        scores,
        ohlcv,
        long_q=long_q,
        short_q=short_q,
        rebalance_frequency=rebalance_frequency,
    )

    if gross.empty:
        return pd.DataFrame(), pd.DataFrame()

    daily_frames = []
    summary_rows = []

    for cost_bps in cost_bps_grid:
        cost_bps = float(cost_bps)
        costs = turnover * (cost_bps / 10000.0)
        net = (gross - costs).rename("net_return")

        daily = pd.DataFrame(
            {
                "gross_return": gross,
                "turnover": turnover,
                "trading_cost": costs,
                "net_return": net,
            }
        )
        daily.index.name = "date"
        daily_frames.append(daily)

        metrics = performance_metrics(
            net,
            turnover,
            gross_returns=gross,
            costs=costs,
        )
        metrics["cost_bps"] = cost_bps
        summary_rows.append(metrics)

    daily_by_cost = pd.concat(
        daily_frames,
        keys=[float(x) for x in cost_bps_grid],
        names=["cost_bps", "date"],
    )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("cost_bps")
        .reset_index(drop=True)
    )

    return daily_by_cost, summary


def backtest_long_short(
    scores: pd.Series,
    ohlcv: pd.DataFrame,
    long_q: float,
    short_q: float,
    cost_bps: float,
) -> tuple[pd.Series, dict]:
    """Backward-compatible single-cost wrapper."""
    daily, summary = backtest_cost_sensitivity(
        scores,
        ohlcv,
        long_q=long_q,
        short_q=short_q,
        cost_bps_grid=[cost_bps],
        rebalance_frequency=None,
    )

    if daily.empty or summary.empty:
        return pd.Series(dtype=float, name="net_return"), {}

    actual_cost_key = daily.index.get_level_values("cost_bps").unique()[0]
    returns = daily.xs(actual_cost_key, level="cost_bps")["net_return"]
    metrics = summary.iloc[0].drop(labels=["cost_bps"]).to_dict()

    return returns.rename("net_return"), metrics
