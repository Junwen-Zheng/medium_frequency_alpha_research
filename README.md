# Medium-Frequency Equity Signal Research

A GitHub-ready research project for a software engineer transitioning into systematic investing research. It focuses on one genuine workflow: medium-frequency equity signals using public price/volume data, strict point-in-time feature construction, statistical validation, PyTorch ranking models, and market-neutral backtesting.

This project is intentionally written as a reproducible research package rather than a one-off notebook. It emphasizes data cleaning, leakage controls, honest evaluation, and experiment reporting.

## What this demonstrates

- Point-in-time equity data construction from public OHLCV data.
- Cross-sectional feature engineering for 5-20 trading-day horizons.
- Custom statistical/ML models: linear baseline, tree/ranking baseline, and PyTorch MLP ranker.
- IC / rank-IC, signal decay, walk-forward validation, and regime checks.
- Market-neutral long/short portfolio simulation with turnover and transaction-cost assumptions.
- Custom agentic workflow automation for data QA, experiment orchestration, result summarization, and report generation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A: real public dataset via yfinance
python -m src.cli run --config config/default.yaml

# Option B: no-internet demo using synthetic data
python -m src.cli run --config config/default.yaml --synthetic
```

Outputs are written to `outputs/` and a Markdown research report is generated in `reports/`.

## Data note

The real-data mode downloads public OHLCV data using `yfinance`. It is not intended for production trading. The goal is to demonstrate a rigorous research workflow: point-in-time joins, lagging, missing-data handling, leakage controls, signal validation, and reproducibility.

## Suggested resume wording

> Built a reproducible medium-frequency equity signal research package using public OHLCV data, point-in-time feature construction, PyTorch ranking models, IC/decay analysis, walk-forward validation, and market-neutral backtesting with transaction-cost and turnover controls.

## Project structure

```text
config/default.yaml       Experiment config
src/data.py               Data download / synthetic fallback / data QA
src/features.py           Point-in-time feature construction and targets
src/models.py             Linear baseline and PyTorch ranker
src/evaluation.py         IC, rank-IC, decay, walk-forward metrics
src/backtest.py           Market-neutral long-short portfolio backtest
src/workflow.py           Custom agentic research workflow runner
src/reporting.py          Markdown report generation
src/cli.py                Command-line entrypoint
tests/                    Lightweight unit tests
```

## Important limitations

- This is research infrastructure and evaluation code, not a live trading system.
- Public OHLCV data is a limited dataset; stronger research would add fundamental, event, news, and alternative data.
- The signal examples are deliberately simple. The value here is the workflow: data quality, leakage control, validation, and reproducibility.
- Backtest results should be treated as research diagnostics, not investment advice.
