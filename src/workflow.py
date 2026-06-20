from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import yaml

from .data import download_ohlcv, make_synthetic_ohlcv, quality_report
from .features import build_features, feature_columns
from .models import fit_ridge, fit_random_forest, fit_pytorch_mlp
from .evaluation import daily_rank_ic, summarize_ic, signal_decay
from .backtest import backtest_long_short
from .reporting import write_report


@dataclass
class WorkflowResult:
    model_summaries: dict
    backtest_metrics: dict
    report_path: str


class ResearchWorkflow:
    """Reproducible research pipeline runner.

    This class deliberately avoids the word "agentic". It is a fixed research
    pipeline: load data, apply point-in-time feature construction, run several
    model baselines, evaluate out-of-sample rank IC, simulate a market-neutral
    portfolio, and write diagnostics. Human research judgment still drives the
    hypotheses, interpretation, and next experiments.
    """

    def __init__(self, config_path: str | Path, synthetic: bool = False):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.synthetic = synthetic
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

        if self.synthetic:
            ohlcv = make_synthetic_ohlcv(tickers, start, end, seed=seed)
        else:
            ohlcv = download_ohlcv(tickers, start, end, cache_path="data/ohlcv.parquet")

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

        # Run decay diagnostics before portfolio diagnostics. This keeps the
        # pure signal evaluation separate from portfolio construction.
        decay = signal_decay(frame, best_score, ohlcv, horizons=[1, 5, 10, 20])
        decay.to_csv(self.outputs / "signal_decay.csv")

        returns, bt_metrics = backtest_long_short(
            best_score,
            ohlcv,
            long_q=cfg["research"].get("long_quantile", 0.8),
            short_q=cfg["research"].get("short_quantile", 0.2),
            cost_bps=cfg["research"].get("transaction_cost_bps", 5),
        )
        bt_metrics["selected_model"] = selected_model_name
        returns.to_csv(self.outputs / "long_short_returns.csv")

        summary_payload = {
            "data_quality": dq,
            "validation": validation_summaries,
            "test": model_summaries,
            "backtest": bt_metrics,
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
        )
        return WorkflowResult(model_summaries, bt_metrics, str(report_path))
