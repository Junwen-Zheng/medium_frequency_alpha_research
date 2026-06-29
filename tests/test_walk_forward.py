import numpy as np
import pandas as pd

from src.models import ModelResult
from src.walk_forward import make_walk_forward_folds, run_walk_forward_validation


def _toy_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    tickers = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    values = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "feature": np.sin(values / 17.0),
            "target_rel_fwd_10d": np.sin(values / 17.0) + 0.01 * np.cos(values / 5.0),
        },
        index=index,
    )
    return frame


def test_walk_forward_folds_are_chronological_with_embargo():
    frame = _toy_frame()
    folds = make_walk_forward_folds(
        frame,
        validation_start="2020-07-01",
        test_start="2021-01-01",
        horizon_days=10,
        fold_months=3,
        min_train_rows=50,
        min_eval_rows=20,
    )

    assert folds

    for fold in folds:
        assert fold.train_end < fold.validation_start
        assert fold.validation_end < fold.test_start
        assert fold.train_end < fold.validation_start - pd.offsets.BDay(9)
        assert len(fold.train) >= 50
        assert len(fold.validation) >= 20
        assert len(fold.test) >= 20


def test_walk_forward_outputs_selected_model_metrics():
    frame = _toy_frame()

    def fit_stack(train, eval_frame, features, target, seed):
        score = eval_frame["feature"].rename("mock_score")
        return [ModelResult("mock", score, None)]

    def summarize_predictions(predictions, target):
        joined = pd.concat([predictions.rename("score"), target.rename("target")], axis=1).dropna()
        daily = joined.groupby(level="date").apply(lambda x: x["score"].rank().corr(x["target"].rank()))
        return {
            "mean_rank_ic": float(daily.mean()),
            "ir": float(daily.mean() / (daily.std(ddof=1) + 1e-12)),
            "positive_days": float((daily > 0).mean()),
            "n_days": int(daily.count()),
        }

    metrics, diagnostics = run_walk_forward_validation(
        frame=frame,
        features=["feature"],
        target="target_rel_fwd_10d",
        fit_model_stack=fit_stack,
        summarize_predictions=summarize_predictions,
        validation_start="2020-07-01",
        test_start="2021-01-01",
        horizon_days=10,
        seed=42,
        fold_months=3,
        min_train_rows=50,
        min_eval_rows=20,
    )

    assert not metrics.empty
    assert not diagnostics.empty
    assert set(metrics["selected_model"]) == {"mock"}
    assert np.isfinite(metrics["test_mean_rank_ic"]).all()
