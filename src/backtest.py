from __future__ import annotations

import numpy as np
import pandas as pd


def make_market_neutral_weights(scores: pd.Series, long_q: float = 0.8, short_q: float = 0.2) -> pd.Series:
    weights = []
    for date, s in scores.dropna().groupby(level="date"):
        if len(s) < 5:
            continue
        lo = s.quantile(short_q)
        hi = s.quantile(long_q)
        w = pd.Series(0.0, index=s.index)
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


def backtest_long_short(scores: pd.Series, ohlcv: pd.DataFrame, long_q: float, short_q: float, cost_bps: float) -> tuple[pd.Series, dict]:
    adj = ohlcv["adj_close"].sort_index()
    fwd_1d = adj.groupby(level="ticker").pct_change().shift(-1)
    weights = make_market_neutral_weights(scores, long_q, short_q)
    gross = (weights * fwd_1d.reindex(weights.index)).groupby(level="date").sum().rename("gross_return")

    wide_w = weights.unstack("ticker").fillna(0).sort_index()
    turnover = wide_w.diff().abs().sum(axis=1).fillna(wide_w.abs().sum(axis=1))
    costs = turnover * (cost_bps / 10000.0)
    net = (gross.reindex(costs.index).fillna(0) - costs).rename("net_return")

    metrics = performance_metrics(net, turnover)
    return net, metrics


def performance_metrics(returns: pd.Series, turnover: pd.Series) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    ann_ret = equity.iloc[-1] ** (252 / max(len(returns), 1)) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(252)
    return {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(ann_ret / (ann_vol + 1e-12)),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "average_daily_turnover": float(turnover.reindex(returns.index).mean()),
    }
