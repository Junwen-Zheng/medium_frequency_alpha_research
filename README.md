# Medium-Frequency Equity Signal Research

This repository is a reproducible research case study for medium-frequency equity signal evaluation. It is intentionally framed as a **research process**, not as a trading system and not as an “agentic” demo.

The goal is to test whether simple public price/volume signals contain stable cross-sectional information over a 5–20 trading-day horizon, then document the evidence clearly enough that another researcher can understand what was tried, what failed, and what I would test next.

## Research question

Can a small set of point-in-time OHLCV-derived features produce useful cross-sectional ranking signals for medium-horizon U.S. equities after controlling for leakage, turnover, transaction costs, and out-of-sample validation?

This is a deliberately limited question. The project does **not** claim to find production-ready alpha. It shows how I structure a research investigation: data cleaning, hypothesis design, baselines, model comparison, ablation, backtesting diagnostics, and honest limitations.

## What this demonstrates

- Point-in-time feature construction with explicit lagging rules.
- Medium-frequency cross-sectional targets over a configurable 5–20 day horizon.
- Baseline and ML models: ridge regression, random forest, and a small PyTorch MLP ranker.
- IC / rank-IC validation, signal decay checks, and market-neutral long/short backtest diagnostics.
- Documentation of failed experiments and research decisions, not only final code.
- Reproducible experiment pipeline that generates model outputs, backtest metrics, and Markdown reports.

## What changed from the first version

This repo was reworked after feedback that the previous version looked too much like a clean tool and not enough like a researcher’s work. The changes are designed to show the research trail:

- Removed overstated “agentic workflow” language; this is now described as a reproducible research pipeline.
- Added a research report with hypotheses, dataset assumptions, evaluation design, results, limitations, and next experiments.
- Added a research log showing the sequence of ideas, failed attempts, and conclusions.
- Added an experiment matrix documenting what was tested and why each experiment did or did not survive.
- Kept the code modest and interpretable rather than pretending to be a production quant platform.

## Repository structure

```text
config/default.yaml                   Experiment configuration
src/data.py                           Data download / synthetic fallback / data quality checks
src/features.py                       Point-in-time feature construction and target definition
src/models.py                         Ridge, random forest, and PyTorch MLP ranker
src/evaluation.py                     Rank-IC, IC summary, signal decay diagnostics
src/backtest.py                       Market-neutral long/short backtest with cost assumptions
src/workflow.py                       Reproducible research pipeline runner
src/reporting.py                      Markdown report generation
src/cli.py                            Command-line entrypoint
reports/research_report.md            Static research write-up / case-study report
docs/experiments/experiment_matrix.csv  Research experiment matrix
docs/research_log/                    Chronological research notes
scripts/commit_sequence.md            Suggested commit sequence for GitHub history
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Real public dataset via yfinance
python -m src.cli run --config config/default.yaml

# No-internet demo using synthetic data
python -m src.cli run --config config/default.yaml --synthetic
```

Outputs are written to `outputs/` and the generated Markdown report is written to `reports/generated_research_report.md`.

## Dataset note

The real-data mode downloads public OHLCV data using `yfinance`. This is not a survivorship-free institutional dataset and should not be treated as production research. I use it here because the purpose is to demonstrate research discipline: point-in-time feature construction, train/validation/test splits, outlier policy, signal validation, transaction-cost assumptions, and reproducibility.

## Current conclusion

The project currently supports a cautious conclusion: simple price/volume features can sometimes produce interesting cross-sectional diagnostics, but the evidence is not strong enough to claim durable tradable alpha. The most useful signal candidates are simple and interpretable; the more flexible model stack needs more data, stronger features, and broader universe coverage before it would be credible.

## Suggested resume wording

> Built a reproducible medium-frequency equity signal research case study using public OHLCV data, point-in-time feature construction, ridge/random forest/PyTorch ranking models, rank-IC and decay analysis, failed-experiment documentation, and market-neutral backtesting with turnover and transaction-cost controls.

## Important limitations

- Public OHLCV-only data is intentionally limited.
- The universe is small and not survivorship-free.
- Results are research diagnostics, not investment advice.
- A stronger project would require broader universe coverage, point-in-time fundamentals/events/news, more robust transaction-cost modeling, and deeper walk-forward research.
