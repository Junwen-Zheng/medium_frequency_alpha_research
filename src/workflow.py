from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import yaml

from .data import download_ohlcv, make_synthetic_ohlcv, quality_report
from .regime import market_regime_labels
from .features import build_features, feature_columns, feature_family_columns, composite_family_score
from .models import fit_ridge, fit_random_forest, fit_pytorch_mlp
from .evaluation import daily_rank_ic, summarize_ic, signal_decay, compare_feature_families, regime_sliced_rank_ic
from .walk_forward import run_walk_forward_validation, summarize_walk_forward_results, run_feature_family_walk_forward, summarize_feature_family_walk_forward_results
from .backtest import backtest_cost_sensitivity, backtest_rebalance_frequency_sensitivity
from .reporting import write_report


@dataclass
class WorkflowResult:
    model_summaries: dict
    backtest_metrics: dict
    report_path: str
    data_mode: str


class ResearchWorkflow:
    """Reproducible research pipeline runner.

    This class deliberately avoids the word "agentic". It is a fixed research
    pipeline: load data, apply point-in-time feature construction, run several
    model baselines, evaluate out-of-sample rank IC, simulate a market-neutral
    portfolio, and write diagnostics. Human research judgment still drives the
    hypotheses, interpretation, and next experiments.
    """

    def __init__(self, config_path: str | Path, smoke_test: bool = False):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.smoke_test = smoke_test
        self.outputs = Path("outputs")
        self.reports = Path("reports")
        self.outputs.mkdir(exist_ok=True)
        self.reports.mkdir(exist_ok=True)

    def _split_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        val_start = pd.Timestamp(self.config["research"]["validation_start"])
        test_start = pd.Timestamp(self.config["research"]["test_start"])
        dates = frame.index.get_level_values("date")
        train = frame[dates < val_start]
        validation = frame[(dates >= val_start) & (dates < test_start)]
        test = frame[dates >= test_start]
        return train, validation, test

    def _fit_model_stack(self, train: pd.DataFrame, eval_frame: pd.DataFrame, features: list[str], target: str, seed: int):
        models = [
            fit_ridge(train, eval_frame, features, target),
            fit_random_forest(
                train,
                eval_frame,
                features,
                target,
                seed=seed,
                n_estimators=self.config["model"].get("random_forest_estimators", 40),
                max_depth=self.config["model"].get("random_forest_max_depth", 5),
            ),
        ]
        if self.config["model"].get("include_pytorch", False):
            models.append(
                fit_pytorch_mlp(
                    train,
                    eval_frame,
                    features,
                    target,
                    epochs=self.config["model"].get("pytorch_epochs", 80),
                    hidden_dim=self.config["model"].get("hidden_dim", 32),
                    lr=self.config["model"].get("learning_rate", 1e-3),
                    weight_decay=self.config["model"].get("weight_decay", 1e-4),
                    seed=seed,
                    max_train_rows=self.config["model"].get("max_pytorch_train_rows", 5000),
                )
            )
        return models

    def _summarize_predictions(self, predictions: pd.Series, target: pd.Series) -> dict:
        ric = daily_rank_ic(predictions, target)
        return summarize_ic(ric)

    def run(self) -> WorkflowResult:
        cfg = self.config
        tickers = cfg["universe"]["tickers"]
        start, end = cfg["universe"]["start"], cfg["universe"]["end"]
        seed = cfg["research"].get("random_seed", 42)
        horizon = cfg["research"].get("horizon_days", 10)

        if self.smoke_test:
            data_mode = "synthetic_smoke_test"
            ohlcv = make_synthetic_ohlcv(tickers, start, end, seed=seed)
        else:
            data_mode = "real_public_ohlcv"
            try:
                ohlcv = download_ohlcv(tickers, start, end, cache_path="data/ohlcv.parquet")
            except Exception as exc:
                raise RuntimeError(
                    "Real-data workflow failed. This research path does not silently fall back "
                    "to synthetic data. Fix the data issue, or run "
                    "`python -m src.cli run --smoke-test` only to verify the offline pipeline."
                ) from exc

        dq = quality_report(ohlcv).to_dict()
        frame = build_features(ohlcv, horizon_days=horizon)
        features = feature_columns(frame)
        target = f"target_rel_fwd_{horizon}d"
        train, validation, test = self._split_frame(frame)

        # Stage 1: compare models on validation before touching test diagnostics.
        validation_models = self._fit_model_stack(train, validation, features, target, seed=seed)
        validation_summaries = {
            m.name: self._summarize_predictions(m.predictions, validation[target])
            for m in validation_models
        }
        selected_model_name = max(
            validation_summaries,
            key=lambda k: validation_summaries[k].get("mean_rank_ic", -999),
        )

        # Stage 2: train on train+validation and evaluate once on the test slice.
        train_full = pd.concat([train, validation]).sort_index()
        test_models = self._fit_model_stack(train_full, test, features, target, seed=seed)

        model_summaries: dict[str, dict] = {}
        best_score = None
        for model_result in test_models:
            summary = self._summarize_predictions(model_result.predictions, test[target])
            summary["validation_mean_rank_ic"] = validation_summaries[model_result.name].get("mean_rank_ic")
            summary["selected_on_validation"] = model_result.name == selected_model_name
            model_summaries[model_result.name] = summary
            model_result.predictions.to_frame().to_csv(self.outputs / f"{model_result.name}_test_scores.csv")
            if model_result.feature_importance is not None:
                model_result.feature_importance.to_csv(self.outputs / f"{model_result.name}_feature_importance.csv")
            if model_result.name == selected_model_name:
                best_score = model_result.predictions

        if best_score is None:
            raise RuntimeError("Selected model was not found in test model stack.")

        # Transparent hypothesis-family comparison. This is deliberately separate
        # from the model stack so the repo shows a research question being tested,
        # not only a black-box prediction pipeline.
        family_scores = {
            name: composite_family_score(test, cols)
            for name, cols in feature_family_columns().items()
        }
        family_comparison = compare_feature_families(test, test[target], family_scores)
        family_comparison.to_csv(self.outputs / "hypothesis_family_comparison.csv")

        regimes = market_regime_labels(ohlcv)
        regime_ic = regime_sliced_rank_ic(best_score, test[target], regimes)
        regime_ic.to_csv(self.outputs / "regime_sliced_rank_ic.csv")

        if cfg["research"].get("run_walk_forward", True):
            walk_forward_metrics, walk_forward_diagnostics = run_walk_forward_validation(
                frame=frame,
                features=features,
                target=target,
                fit_model_stack=self._fit_model_stack,
                summarize_predictions=self._summarize_predictions,
                validation_start=cfg["research"]["validation_start"],
                test_start=cfg["research"]["test_start"],
                horizon_days=horizon,
                seed=seed,
                fold_months=cfg["research"].get("walk_forward_fold_months", 6),
                min_train_rows=cfg["research"].get("walk_forward_min_train_rows", 500),
                min_eval_rows=cfg["research"].get("walk_forward_min_eval_rows", 30),
            )
            walk_forward_summary = summarize_walk_forward_results(
                walk_forward_metrics,
                walk_forward_diagnostics,
            )
            feature_family_walk_forward = run_feature_family_walk_forward(
                frame=frame,
                target=target,
                family_columns=feature_family_columns(),
                summarize_predictions=self._summarize_predictions,
                validation_start=cfg["research"]["validation_start"],
                test_start=cfg["research"]["test_start"],
                horizon_days=horizon,
                fold_months=cfg["research"].get("walk_forward_fold_months", 6),
                min_train_rows=cfg["research"].get("walk_forward_min_train_rows", 500),
                min_eval_rows=cfg["research"].get("walk_forward_min_eval_rows", 30),
            )
            feature_family_walk_forward_summary = summarize_feature_family_walk_forward_results(
                feature_family_walk_forward
            )

            walk_forward_metrics.to_csv(self.outputs / "walk_forward_metrics.csv", index=False)
            walk_forward_diagnostics.to_csv(self.outputs / "walk_forward_fold_diagnostics.csv", index=False)
            walk_forward_summary.to_csv(self.outputs / "walk_forward_summary.csv", index=False)
            feature_family_walk_forward.to_csv(self.outputs / "feature_family_walk_forward.csv", index=False)
            feature_family_walk_forward_summary.to_csv(self.outputs / "feature_family_walk_forward_summary.csv", index=False)
        else:
            walk_forward_metrics = pd.DataFrame()
            walk_forward_diagnostics = pd.DataFrame()
            walk_forward_summary = pd.DataFrame()
            feature_family_walk_forward = pd.DataFrame()
            feature_family_walk_forward_summary = pd.DataFrame()

        # Signal-decay diagnostics are part of the full research workflow, but
        # they can be toggled off for a faster exploratory run.
        if cfg["research"].get("run_signal_decay", False):
            decay = signal_decay(frame, best_score, ohlcv, horizons=[1, 5, 10, 20])
            decay.to_csv(self.outputs / "signal_decay.csv")
        else:
            decay = pd.DataFrame()

        if cfg["research"].get("run_backtest", False):
            cost_grid = cfg["research"].get("transaction_cost_bps_grid")
            if cost_grid is None:
                cost_grid = [cfg["research"].get("transaction_cost_bps", 5)]

            daily_costs, cost_summary = backtest_cost_sensitivity(
                best_score,
                ohlcv,
                long_q=cfg["research"].get("long_quantile", 0.8),
                short_q=cfg["research"].get("short_quantile", 0.2),
                cost_bps_grid=cost_grid,
                rebalance_frequency=cfg["research"].get("rebalance_frequency"),
            )

            if not daily_costs.empty:
                daily_costs.to_csv(self.outputs / "cost_sensitive_long_short_returns.csv")

            if not cost_summary.empty:
                cost_summary.to_csv(
                    self.outputs / "transaction_cost_sensitivity.csv",
                    index=False,
                )

            rebalance_grid = cfg["research"].get(
                "rebalance_frequency_grid",
                ["daily", "weekly", "biweekly", "monthly"],
            )
            rebalance_summary = backtest_rebalance_frequency_sensitivity(
                best_score,
                ohlcv,
                long_q=cfg["research"].get("long_quantile", 0.8),
                short_q=cfg["research"].get("short_quantile", 0.2),
                cost_bps_grid=cost_grid,
                rebalance_frequencies=rebalance_grid,
            )

            if not rebalance_summary.empty:
                rebalance_summary.to_csv(
                    self.outputs / "rebalance_frequency_sensitivity.csv",
                    index=False,
                )

            base_cost_bps = float(cfg["research"].get("transaction_cost_bps", 5))

            if not cost_summary.empty:
                base_rows = cost_summary[
                    cost_summary["cost_bps"].astype(float).sub(base_cost_bps).abs()
                    < 1e-12
                ]
                selected_cost_row = (
                    base_rows.iloc[0]
                    if not base_rows.empty
                    else cost_summary.iloc[0]
                )
                bt_metrics = json.loads(selected_cost_row.to_json())
            else:
                bt_metrics = {}

            bt_metrics["selected_model"] = selected_model_name
            bt_metrics["rebalance_frequency"] = cfg["research"].get(
                "rebalance_frequency"
            )
            bt_metrics["cost_bps_grid"] = [float(x) for x in cost_grid]
            bt_metrics[
                "note"
            ] = "Cost-sensitive portfolio diagnostics are written to outputs/transaction_cost_sensitivity.csv. Rebalance-frequency diagnostics are written to outputs/rebalance_frequency_sensitivity.csv."
        else:
            cost_summary = pd.DataFrame()
            rebalance_summary = pd.DataFrame()
            bt_metrics = {
                "selected_model": selected_model_name,
                "note": "Backtest disabled for this configuration. Set research.run_backtest: true to enable portfolio diagnostics.",
            }
        summary_payload = {
            "data_mode": data_mode,
            "data_quality": dq,
            "validation": validation_summaries,
            "test": model_summaries,
            "hypothesis_family_comparison": family_comparison.to_dict(),
            "regime_sliced_rank_ic": regime_ic.to_dict(),
            "walk_forward": {
                "summary": walk_forward_summary.to_dict(orient="records"),
                "metrics": walk_forward_metrics.to_dict(orient="records"),
                "diagnostics": walk_forward_diagnostics.to_dict(orient="records"),
            },
            "feature_family_walk_forward": {
                "summary": feature_family_walk_forward_summary.to_dict(orient="records"),
                "metrics": feature_family_walk_forward.to_dict(orient="records"),
            },
            "backtest": bt_metrics,
            "transaction_cost_sensitivity": cost_summary.to_dict(orient="records"),
            "rebalance_frequency_sensitivity": rebalance_summary.to_dict(orient="records"),
        }
        (self.outputs / "workflow_summary.json").write_text(json.dumps(summary_payload, indent=2))

        report_path = self.reports / "generated_research_report.md"
        write_report(
            report_path,
            cfg["report"]["title"],
            dq,
            validation_summaries,
            model_summaries,
            bt_metrics,
            decay,
            family_comparison,
            regime_ic,
            walk_forward_metrics,
            walk_forward_diagnostics,
            walk_forward_summary,
            feature_family_walk_forward,
            feature_family_walk_forward_summary,
            cost_sensitivity=cost_summary,
            rebalance_frequency_sensitivity=rebalance_summary,
        )
        return WorkflowResult(model_summaries, bt_metrics, str(report_path), data_mode)
