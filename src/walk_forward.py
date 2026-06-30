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


def summarize_walk_forward_results(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate walk-forward results into robustness diagnostics.

    The goal is not to declare alpha. The goal is to summarize whether
    out-of-sample rank IC is stable across chronological folds and whether
    validation-based model selection reliably chooses the best test model.
    """
    if metrics is None or metrics.empty:
        return pd.DataFrame()

    out: dict[str, object] = {}

    test_ic = pd.to_numeric(metrics["test_mean_rank_ic"], errors="coerce")
    test_ir = pd.to_numeric(metrics["test_ir"], errors="coerce")
    val_ic = pd.to_numeric(metrics["validation_mean_rank_ic"], errors="coerce")

    out["n_folds"] = int(len(metrics))
    out["mean_selected_test_rank_ic"] = float(test_ic.mean())
    out["median_selected_test_rank_ic"] = float(test_ic.median())
    out["worst_selected_test_rank_ic"] = float(test_ic.min())
    out["best_selected_test_rank_ic"] = float(test_ic.max())
    out["mean_selected_test_ir"] = float(test_ir.mean())
    out["positive_test_folds"] = int((test_ic > 0).sum())
    out["positive_test_fold_rate"] = float((test_ic > 0).mean())

    selected_models = metrics["selected_model"].astype(str)
    out["selected_model_counts"] = "; ".join(
        f"{model}:{count}" for model, count in selected_models.value_counts().sort_index().items()
    )
    out["selected_model_switches"] = int((selected_models != selected_models.shift()).sum() - 1)

    valid_corr = pd.concat(
        [val_ic.rename("validation"), test_ic.rename("test")],
        axis=1,
    ).dropna()
    if len(valid_corr) >= 2 and valid_corr["validation"].std(ddof=0) > 0 and valid_corr["test"].std(ddof=0) > 0:
        out["validation_test_rank_ic_correlation"] = float(valid_corr["validation"].corr(valid_corr["test"]))
    else:
        out["validation_test_rank_ic_correlation"] = np.nan

    if diagnostics is not None and not diagnostics.empty:
        best_or_tied: list[bool] = []
        for _, fold_rows in diagnostics.groupby("fold_id"):
            selected = fold_rows[fold_rows["selected_on_validation"] == True]
            if selected.empty:
                continue
            selected_test_ic = float(selected.iloc[0]["test_mean_rank_ic"])
            best_test_ic = float(pd.to_numeric(fold_rows["test_mean_rank_ic"], errors="coerce").max())
            best_or_tied.append(selected_test_ic >= best_test_ic - 1e-12)

        if best_or_tied:
            out["selected_model_best_or_tied_folds"] = int(sum(best_or_tied))
            out["selected_model_best_or_tied_rate"] = float(np.mean(best_or_tied))
        else:
            out["selected_model_best_or_tied_folds"] = 0
            out["selected_model_best_or_tied_rate"] = np.nan

    return pd.DataFrame([out])


def run_feature_family_walk_forward(
    frame: pd.DataFrame,
    target: str,
    family_columns: dict[str, list[str]],
    summarize_predictions: Callable,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    horizon_days: int,
    fold_months: int = 6,
    min_train_rows: int = 500,
    min_eval_rows: int = 30,
) -> pd.DataFrame:
    """Evaluate transparent feature-family scores across walk-forward folds.

    This is an ablation diagnostic, not a model search. Each family uses a simple
    equal-weight composite score so the research question stays interpretable:
    which hypothesis family, if any, is stable across chronological folds?
    """
    folds = make_walk_forward_folds(
        frame=frame,
        validation_start=validation_start,
        test_start=test_start,
        horizon_days=horizon_days,
        fold_months=fold_months,
        min_train_rows=min_train_rows,
        min_eval_rows=min_eval_rows,
    )

    rows: list[dict] = []

    for fold in folds:
        for family, cols in family_columns.items():
            available = [c for c in cols if c in frame.columns]
            if not available:
                continue

            validation_score = fold.validation[available].mean(axis=1).rename(f"{family}_score")
            test_score = fold.test[available].mean(axis=1).rename(f"{family}_score")

            validation_summary = _finite_summary(
                summarize_predictions(validation_score, fold.validation[target])
            )
            test_summary = _finite_summary(
                summarize_predictions(test_score, fold.test[target])
            )

            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "family": family,
                    "n_features": len(available),
                    "features": "; ".join(available),
                    "train_start": str(fold.train.index.get_level_values("date").min().date()),
                    "train_end": str(fold.train_end.date()),
                    "validation_start": str(fold.validation_start.date()),
                    "validation_end": str(fold.validation_end.date()),
                    "test_start": str(fold.test_start.date()),
                    "test_end": str(fold.test_end.date()),
                    "train_rows": len(fold.train),
                    "validation_rows": len(fold.validation),
                    "test_rows": len(fold.test),
                    "validation_mean_rank_ic": validation_summary.get("mean_rank_ic"),
                    "validation_ir": validation_summary.get("ir"),
                    "validation_positive_days": validation_summary.get("positive_days"),
                    "validation_n_days": validation_summary.get("n_days"),
                    "test_mean_rank_ic": test_summary.get("mean_rank_ic"),
                    "test_ir": test_summary.get("ir"),
                    "test_positive_days": test_summary.get("positive_days"),
                    "test_n_days": test_summary.get("n_days"),
                }
            )

    return pd.DataFrame(rows)


def summarize_feature_family_walk_forward_results(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature-family walk-forward ablation metrics."""
    if metrics is None or metrics.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for family, grp in metrics.groupby("family"):
        test_ic = pd.to_numeric(grp["test_mean_rank_ic"], errors="coerce")
        test_ir = pd.to_numeric(grp["test_ir"], errors="coerce")
        validation_ic = pd.to_numeric(grp["validation_mean_rank_ic"], errors="coerce")

        valid_corr = pd.concat(
            [validation_ic.rename("validation"), test_ic.rename("test")],
            axis=1,
        ).dropna()

        if len(valid_corr) >= 2 and valid_corr["validation"].std(ddof=0) > 0 and valid_corr["test"].std(ddof=0) > 0:
            validation_test_corr = float(valid_corr["validation"].corr(valid_corr["test"]))
        else:
            validation_test_corr = np.nan

        rows.append(
            {
                "family": family,
                "n_folds": int(len(grp)),
                "n_features": int(grp["n_features"].max()),
                "mean_test_rank_ic": float(test_ic.mean()),
                "median_test_rank_ic": float(test_ic.median()),
                "worst_test_rank_ic": float(test_ic.min()),
                "best_test_rank_ic": float(test_ic.max()),
                "mean_test_ir": float(test_ir.mean()),
                "positive_test_folds": int((test_ic > 0).sum()),
                "positive_test_fold_rate": float((test_ic > 0).mean()),
                "mean_validation_rank_ic": float(validation_ic.mean()),
                "validation_test_rank_ic_correlation": validation_test_corr,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["positive_test_fold_rate", "mean_test_rank_ic"],
        ascending=[False, False],
    )
