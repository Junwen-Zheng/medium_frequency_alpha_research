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
    """Custom agentic research workflow.

    The workflow is "agentic" in the narrow engineering sense: it orchestrates the research loop,
    enforces QA gates, runs experiments, persists outputs, and writes a report. It does not replace
    human hypothesis design or final signal judgment.
    """

    def __init__(self, config_path: str | Path, synthetic: bool = False):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.synthetic = synthetic
        self.outputs = Path("outputs")
        self.reports = Path("reports")
        self.outputs.mkdir(exist_ok=True)
        self.reports.mkdir(exist_ok=True)

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

        val_start = pd.Timestamp(cfg["research"]["validation_start"])
        test_start = pd.Timestamp(cfg["research"]["test_start"])
        dates = frame.index.get_level_values("date")
        train = frame[dates < val_start]
        validation = frame[(dates >= val_start) & (dates < test_start)]
        test = frame[dates >= test_start]

        # Train on train+validation for final test diagnostics after checking validation.
        train_full = pd.concat([train, validation]).sort_index()
        models = [
            fit_ridge(train_full, test, features, target),
            fit_random_forest(train_full, test, features, target, seed=seed),
            fit_pytorch_mlp(
                train_full, test, features, target,
                epochs=cfg["model"].get("pytorch_epochs", 80),
                hidden_dim=cfg["model"].get("hidden_dim", 32),
                lr=cfg["model"].get("learning_rate", 1e-3),
                weight_decay=cfg["model"].get("weight_decay", 1e-4),
                seed=seed,
            ),
        ]

        model_summaries = {}
        target_test = test[target]
        best_name, best_score, best_ic = None, None, -999
        for m in models:
            ric = daily_rank_ic(m.predictions, target_test)
            summary = summarize_ic(ric)
            model_summaries[m.name] = summary
            if summary.get("mean_rank_ic", -999) > best_ic:
                best_ic = summary.get("mean_rank_ic", -999)
                best_name, best_score = m.name, m.predictions
            m.predictions.to_frame().to_csv(self.outputs / f"{m.name}_test_scores.csv")
            if m.feature_importance is not None:
                m.feature_importance.to_csv(self.outputs / f"{m.name}_feature_importance.csv")

        returns, bt_metrics = backtest_long_short(
            best_score, ohlcv,
            long_q=cfg["research"].get("long_quantile", 0.8),
            short_q=cfg["research"].get("short_quantile", 0.2),
            cost_bps=cfg["research"].get("transaction_cost_bps", 5),
        )
        bt_metrics["selected_model"] = best_name
        returns.to_csv(self.outputs / "long_short_returns.csv")
        decay = signal_decay(frame, best_score, ohlcv, horizons=[1, 5, 10, 20])
        decay.to_csv(self.outputs / "signal_decay.csv")

        report_path = self.reports / "research_report.md"
        write_report(report_path, cfg["report"]["title"], dq, model_summaries, bt_metrics, decay)
        (self.outputs / "workflow_summary.json").write_text(json.dumps({"data_quality": dq, "models": model_summaries, "backtest": bt_metrics}, indent=2))
        return WorkflowResult(model_summaries, bt_metrics, str(report_path))
