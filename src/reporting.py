from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_report(
    path: str | Path,
    title: str,
    data_quality: dict,
    validation_summaries: dict,
    test_summaries: dict,
    backtest_metrics: dict,
    decay_table: pd.DataFrame,
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
        "## Signal decay",
        "",
        decay_table.to_markdown(),
        "",
        "## Market-neutral backtest",
        "",
        "```json",
        json.dumps(backtest_metrics, indent=2),
        "```",
        "",
        "## Honest interpretation guardrails",
        "",
        "- Treat these outputs as diagnostics, not proof of tradable alpha.",
        "- Public OHLCV-only data is intentionally limited; stronger research needs additional event, fundamental, and alternative data.",
        "- Good-looking results must be stress-tested for leakage, turnover, cost assumptions, universe choice, and regime dependence.",
        "- The research process matters more than any single score in this toy-sized universe.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
