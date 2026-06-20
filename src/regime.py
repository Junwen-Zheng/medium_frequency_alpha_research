from __future__ import annotations

import numpy as np
import pandas as pd


def market_regime_labels(ohlcv: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Create simple ex-ante market regime labels from equal-weight returns.

    Regime labels are deliberately simple and only use information available up
    to the prior day. They are used to check whether signal quality clusters in
    certain market states.
    """
    adj = ohlcv["adj_close"].sort_index()
    ret = adj.groupby(level="ticker").pct_change()
    market_ret = ret.groupby(level="date").mean().sort_index()
    trend = market_ret.rolling(lookback).sum().shift(1)
    realized_vol = market_ret.rolling(lookback).std().shift(1)
    vol_cutoff = realized_vol.rolling(252, min_periods=lookback).median()

    labels = []
    for date in market_ret.index:
        tr = trend.loc[date]
        vol = realized_vol.loc[date]
        cutoff = vol_cutoff.loc[date]
        if not np.isfinite(tr) or not np.isfinite(vol) or not np.isfinite(cutoff):
            labels.append("unclassified")
            continue
        direction = "uptrend" if tr >= 0 else "downtrend"
        vol_state = "high_vol" if vol > cutoff else "normal_vol"
        labels.append(f"{direction}_{vol_state}")
    return pd.Series(labels, index=market_ret.index, name="market_regime")
