from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    fold_id: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_folds(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    horizon_days: int,
    fold_months: int = 6,
    min_train_rows: int = 500,
    min_eval_rows: int = 30,
) -> list[WalkForwardFold]:
    """Create chronological walk-forward folds with an embargo before validation.

    Each fold uses:
    - train: dates before validation_start minus horizon embargo
    - validation: immediately before the test window
    - test: the forward evaluation window

    This keeps model selection out-of-sample and prevents direct overlap between
    forward-return labels and the next evaluation window.
    """
    if "date" not in frame.index.names:
        raise ValueError("Expected frame index to include a 'date' level.")

    dates = pd.Series(frame.index.get_level_values("date")).drop_duplicates().sort_values()
    if dates.empty:
        return []

    validation_anchor = pd.Timestamp(validation_start)
    fold_test_start = pd.Timestamp(test_start)
    max_date = pd.Timestamp(dates.max())

    folds: list[WalkForwardFold] = []
    fold_id = 1

    while fold_test_start <= max_date:
        fold_test_end = min(fold_test_start + pd.DateOffset(months=fold_months), max_date + pd.Timedelta(days=1))
        fold_validation_start = max(validation_anchor, fold_test_start - pd.DateOffset(months=fold_months))
        fold_validation_end = fold_test_start - pd.Timedelta(days=1)
        train_end = fold_validation_start - pd.offsets.BDay(horizon_days)

        frame_dates = frame.index.get_level_values("date")
        train = frame[frame_dates < train_end]
        validation = frame[(frame_dates >= fold_validation_start) & (frame_dates < fold_test_start)]
        test = frame[(frame_dates >= fold_test_start) & (frame_dates < fold_test_end)]

        if len(train) >= min_train_rows and len(validation) >= min_eval_rows and len(test) >= min_eval_rows:
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train=train,
                    validation=validation,
                    test=test,
                    train_end=pd.Timestamp(train.index.get_level_values("date").max()),
                    validation_start=pd.Timestamp(validation.index.get_level_values("date").min()),
                    validation_end=pd.Timestamp(validation.index.get_level_values("date").max()),
                    test_start=pd.Timestamp(test.index.get_level_values("date").min()),
                    test_end=pd.Timestamp(test.index.get_level_values("date").max()),
                )
            )
            fold_id += 1

        fold_test_start = fold_test_end

    return folds


def _finite_summary(summary: dict) -> dict:
    clean = {}
    for key, value in summary.items():
        if isinstance(value, (int, np.integer)):
            clean[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            clean[key] = float(value) if np.isfinite(value) else np.nan
        else:
            clean[key] = value
    return clean


def run_walk_forward_validation(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    fit_model_stack: Callable,
    summarize_predictions: Callable,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    horizon_days: int,
    seed: int,
    fold_months: int = 6,
    min_train_rows: int = 500,
    min_eval_rows: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated chronological model selection and test evaluation."""
    folds = make_walk_forward_folds(
        frame=frame,
        validation_start=validation_start,
        test_start=test_start,
        horizon_days=horizon_days,
        fold_months=fold_months,
        min_train_rows=min_train_rows,
        min_eval_rows=min_eval_rows,
    )

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for fold in folds:
        validation_models = fit_model_stack(fold.train, fold.validation, features, target, seed)
        validation_summaries = {
            m.name: _finite_summary(summarize_predictions(m.predictions, fold.validation[target]))
            for m in validation_models
        }

        selected_model_name = max(
            validation_summaries,
            key=lambda k: validation_summaries[k].get("mean_rank_ic", -999),
        )

        train_full = pd.concat([fold.train, fold.validation]).sort_index()
        test_models = fit_model_stack(train_full, fold.test, features, target, seed)

        for model_result in test_models:
            test_summary = _finite_summary(summarize_predictions(model_result.predictions, fold.test[target]))
            val_summary = validation_summaries[model_result.name]

            diagnostic_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": model_result.name,
                    "selected_on_validation": model_result.name == selected_model_name,
                    "train_start": str(fold.train.index.get_level_values("date").min().date()),
                    "train_end": str(fold.train_end.date()),
                    "validation_start": str(fold.validation_start.date()),
                    "validation_end": str(fold.validation_end.date()),
                    "test_start": str(fold.test_start.date()),
                    "test_end": str(fold.test_end.date()),
                    "train_rows": len(fold.train),
                    "validation_rows": len(fold.validation),
                    "test_rows": len(fold.test),
                    "validation_mean_rank_ic": val_summary.get("mean_rank_ic"),
                    "validation_ir": val_summary.get("ir"),
                    "validation_positive_days": val_summary.get("positive_days"),
                    "validation_n_days": val_summary.get("n_days"),
                    "test_mean_rank_ic": test_summary.get("mean_rank_ic"),
                    "test_ir": test_summary.get("ir"),
                    "test_positive_days": test_summary.get("positive_days"),
                    "test_n_days": test_summary.get("n_days"),
                }
            )

            if model_result.name == selected_model_name:
                metric_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "selected_model": selected_model_name,
                        "train_start": str(fold.train.index.get_level_values("date").min().date()),
                        "train_end": str(fold.train_end.date()),
                        "validation_start": str(fold.validation_start.date()),
                        "validation_end": str(fold.validation_end.date()),
                        "test_start": str(fold.test_start.date()),
                        "test_end": str(fold.test_end.date()),
                        "train_rows": len(fold.train),
                        "validation_rows": len(fold.validation),
                        "test_rows": len(fold.test),
                        "validation_mean_rank_ic": val_summary.get("mean_rank_ic"),
                        "test_mean_rank_ic": test_summary.get("mean_rank_ic"),
                        "test_ir": test_summary.get("ir"),
                        "test_positive_days": test_summary.get("positive_days"),
                        "test_n_days": test_summary.get("n_days"),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)
