from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def _markdown_table_or_note(df: pd.DataFrame, note: str) -> str:
    return df.to_markdown(index=False) if df is not None and not df.empty else note


def write_report(
    path: str | Path,
    title: str,
    data_quality: dict,
    validation_summaries: dict,
    test_summaries: dict,
    backtest_metrics: dict,
    decay_table: pd.DataFrame,
    family_comparison: pd.DataFrame | None = None,
    regime_ic: pd.DataFrame | None = None,
    walk_forward_metrics: pd.DataFrame | None = None,
    walk_forward_diagnostics: pd.DataFrame | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}: Generated Diagnostics",
        "",
        "This generated report is a diagnostic output from the reproducible pipeline. The main case-study interpretation lives in `reports/research_report.md`.",
        "",
        "## Data quality",
        "",
        "```json",
        json.dumps(data_quality, indent=2),
        "```",
        "",
        "## Validation model comparison",
        "",
        "Models are selected on the validation slice before final test diagnostics.",
        "",
        "```json",
        json.dumps(validation_summaries, indent=2),
        "```",
        "",
        "## Test diagnostics",
        "",
        "```json",
        json.dumps(test_summaries, indent=2),
        "```",
        "",
        "## Hypothesis-family comparison",
        "",
        "This table compares transparent composite scores for the original raw price/volume family versus the implemented volatility-adjusted follow-up hypothesis. It is meant to test whether a feature idea deserves further research before relying on a more flexible model.",
        "",
        _markdown_table_or_note(family_comparison, "Hypothesis-family comparison was not generated."),
        "",
        "## Regime-sliced rank IC",
        "",
        "This table checks whether the selected model's cross-sectional ranking quality is concentrated in specific market regimes. Concentrated performance should be treated as weaker evidence than broad stability.",
        "",
        _markdown_table_or_note(regime_ic, "Regime-sliced rank IC was not generated."),
        "",
        "## Walk-forward validation",
        "",
        "This section repeats model selection across chronological folds. It tests whether the signal survives across time instead of relying on one validation/test split.",
        "",
        _markdown_table_or_note(walk_forward_metrics, "Walk-forward validation was disabled or produced no valid folds."),
        "",
        "Per-model fold diagnostics are written to `outputs/walk_forward_fold_diagnostics.csv`. Mixed fold results should be treated as weaker evidence than consistently positive out-of-sample performance.",
        "",
        "## Signal decay",
        "",
        decay_table.to_markdown(index=False) if not decay_table.empty else "Signal decay was disabled for this configuration. Set `research.run_signal_decay: true` in the config to run it.",
        "",
        "## Market-neutral backtest",
        "",
        "```json",
        json.dumps(backtest_metrics, indent=2),
        "```",
        "",
        "Set `research.run_backtest: true` in the config to generate full portfolio diagnostics.",
        "",
        "## Honest interpretation guardrails",
        "",
        "- Treat these outputs as diagnostics, not proof of tradable alpha.",
        "- Public OHLCV-only data is intentionally limited; stronger research needs additional event, fundamental, and alternative data.",
        "- Good-looking results must be stress-tested for leakage, turnover, cost assumptions, universe choice, and regime dependence.",
        "- The research process matters more than any single score in this toy-sized universe.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
