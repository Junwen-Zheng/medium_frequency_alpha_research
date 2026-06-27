# Medium-Frequency Equity Signal Research

This repository is a research notebook-style codebase for testing whether simple public OHLCV-derived features contain stable cross-sectional information over a 5-20 trading-day horizon.

The project is deliberately limited. It does **not** claim to find production-ready alpha. The purpose is to document a research process: define a hypothesis, construct point-in-time features, avoid leakage, measure rank IC / decay, run a simple market-neutral diagnostic, record negative results, and decide what should be tested next.

## Research question

Can a small set of lagged price, volume, and volatility-adjusted features produce useful cross-sectional ranking signals for medium-horizon U.S. equities after leakage checks, out-of-sample validation, turnover, and transaction-cost assumptions?

## Current scope

- Public OHLCV data via `yfinance` for a small liquid U.S. equity universe.
- Point-in-time features with explicit one-day feature lagging.
- Forward-return labels computed within each ticker, then cross-sectionally demeaned by date.
- Baseline models: ridge regression, random forest, and an optional small PyTorch MLP.
- Diagnostics: daily rank IC, IC summary, signal decay, simple market-neutral long/short backtest, and regime-sliced stability checks.
- Synthetic data exists only for offline tests and CI-style smoke checks. It is not treated as research evidence.

## Repository structure

```text
config/default.yaml        Experiment configuration
src/data.py                Public OHLCV download and synthetic smoke-test generator
src/features.py            Point-in-time feature construction and target definition
src/models.py              Ridge, random forest, and optional PyTorch MLP baselines
src/evaluation.py          Rank IC, signal decay, and stability diagnostics
src/regime.py              Ex-ante market-regime labels for diagnostic slicing
src/backtest.py            Market-neutral long/short diagnostic backtest
tests/                     Leakage, alignment, and pipeline integrity tests
docs/research_log/         Research notes written after each meaningful iteration
docs/study_notes/          Notes used to understand and audit the implementation
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Primary research path: real public data
python -m src.cli run --config config/default.yaml

# Offline smoke test only; do not use this as research evidence
python -m src.cli run --config config/default.yaml --smoke-test

pytest -q
```

Outputs are written to `outputs/` and the generated Markdown report is written to `reports/generated_research_report.md`.

## Important limitations

- The universe is small and not survivorship-free.
- Public OHLCV data is not institutional point-in-time data.
- Transaction costs are simplified.
- Results should be interpreted as research diagnostics, not investment advice.
- A stronger version would require broader universe coverage, richer data, more realistic costs, stricter walk-forward validation, and deeper failed-experiment documentation.

## Current research direction

The next serious pass focuses on correctness before sophistication:

1. Verify all forward-return labels are shifted within ticker.
2. Keep synthetic data out of the main research conclusion.
3. Add tests for leakage, alignment, and future-data usage.
4. Run real-data experiments first, then document what fails.
5. Make each research log explain the decision process, not just the final output.
