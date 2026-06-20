from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_report(path: str | Path, title: str, data_quality: dict, model_summaries: dict, backtest_metrics: dict, decay_table: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Data quality",
        "",
        "```json",
        json.dumps(data_quality, indent=2),
        "```",
        "",
        "## Model validation summary",
        "",
        "```json",
        json.dumps(model_summaries, indent=2),
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
        "## Honest notes",
        "",
        "- Treat these outputs as diagnostics, not proof of tradable alpha.",
        "- Public OHLCV-only data is intentionally limited; stronger research needs additional event, fundamental, and alternative data.",
        "- Good-looking results must be stress-tested for leakage, turnover, cost assumptions, universe choice, and regime dependence.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
