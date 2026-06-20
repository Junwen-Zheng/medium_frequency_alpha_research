from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(scores: pd.Series, target: pd.Series) -> pd.Series:
    """Compute daily Spearman rank correlation without scipy overhead.

    The calculation ranks score and target cross-sectionally each day and then
    computes Pearson correlation between the two rank vectors. This is equivalent
    to Spearman correlation and is stable for repeated diagnostic calls.
    """
    df = pd.concat([scores.rename("score"), target.rename("target")], axis=1).dropna()
    vals = {}
    for date, grp in df.groupby(level="date"):
        if grp["score"].nunique() < 3 or grp["target"].nunique() < 3:
            continue
        score_rank = grp["score"].rank(method="average")
        target_rank = grp["target"].rank(method="average")
        corr = score_rank.corr(target_rank)
        if np.isfinite(corr):
            vals[date] = corr
    return pd.Series(vals).sort_index().rename("rank_ic")


def summarize_ic(rank_ic: pd.Series) -> dict:
    rank_ic = rank_ic.dropna()
    if rank_ic.empty:
        return {"mean_rank_ic": np.nan, "ir": np.nan, "positive_days": np.nan, "n_days": 0}
    return {
        "mean_rank_ic": float(rank_ic.mean()),
        "ir": float(rank_ic.mean() / (rank_ic.std(ddof=1) + 1e-12) * np.sqrt(252)),
        "positive_days": float((rank_ic > 0).mean()),
        "n_days": int(rank_ic.count()),
    }


def signal_decay(feature_frame: pd.DataFrame, score: pd.Series, ohlcv: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Summarize rank-IC at multiple forward horizons.

    The score is usually only available on the final evaluation window. Reindexing
    each horizon target to the score index keeps the diagnostic fast and avoids
    accidentally evaluating periods where no score was produced.
    """
    rows = []
    adj = ohlcv["adj_close"].sort_index()
    score = score.dropna().sort_index()
    for h in horizons:
        fwd = adj.groupby(level="ticker").shift(-h) / adj - 1
        rel = fwd - fwd.groupby(level="date").transform("mean")
        rel = rel.reindex(score.index)
        ric = daily_rank_ic(score, rel)
        summary = summarize_ic(ric)
        summary["horizon"] = h
        rows.append(summary)
    return pd.DataFrame(rows).set_index("horizon")
