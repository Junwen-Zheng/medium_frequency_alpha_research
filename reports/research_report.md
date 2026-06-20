# Medium-Frequency Equity Signal Research Case Study

## 1. Research question

This case study asks whether simple public OHLCV-derived features contain medium-horizon cross-sectional information for large U.S. equities.

The practical version of the question is:

> If I rank stocks each week using point-in-time price/volume features and simple statistical/ML models, do the scores show persistent rank-IC, reasonable signal decay, and a market-neutral backtest profile that survives transaction-cost assumptions?

The answer is intentionally cautious. This project is not claiming a deployable alpha. It is meant to show the way I approach research: define a hypothesis, build leakage-controlled data, test simple baselines, document failed ideas, and decide what would need to be true before the work could be taken seriously.

## 2. Dataset and universe

### Current dataset

- Source: public OHLCV data through `yfinance`.
- Universe: a small set of large, liquid U.S. equities in `config/default.yaml`.
- Date range: 2018-01-01 to 2025-12-31.
- Target horizon: configurable, default 10 trading days.
- Unit of observation: `(date, ticker)`.

### Data quality controls

- Drop rows with missing adjusted close.
- Build features with at least one-day lag to avoid same-day leakage.
- Use train / validation / test split by time, not random split.
- Clip cross-sectional feature z-scores to reduce extreme outlier influence.
- Use forward relative return as target: stock forward return minus same-date cross-sectional mean.

### Important dataset limitation

This is not an institutional point-in-time database. The universe is small and not survivorship-free. The purpose is to demonstrate research process, not to claim production research quality.

## 3. Hypotheses tested

### H1: Short-term reversal

Stocks that underperformed over the last 5 trading days may mean-revert over the next 5–20 trading days, especially in a large-cap universe where liquidity shocks reverse.

Feature proxy: `reversal_5d`.

### H2: Medium-term momentum

Stocks with stronger 20–60 day returns may continue to outperform over a medium horizon if the move reflects delayed information diffusion.

Feature proxies: `momentum_20d`, `momentum_60d`.

### H3: Volatility and range compression

Stocks with high realized volatility or wide intraday ranges may behave differently over the next horizon. The first attempt was to test whether the model learns a useful nonlinear interaction between recent return and volatility.

Feature proxies: `volatility_20d`, `high_low_range`.

### H4: Volume shock / attention proxy

Unusual volume may proxy for information arrival or attention. The first attempt tests both share volume and dollar volume z-scores.

Feature proxies: `volume_z_20d`, `dollar_volume_z_20d`.

## 4. Model stack

The model stack is deliberately simple.

### Ridge regression

Used as an interpretable linear baseline. If ridge cannot produce stable rank-IC, it is hard to justify more flexible models.

### Random forest

Used to test modest nonlinear interactions, especially among momentum, reversal, volatility, and volume features.

### PyTorch MLP ranker

Used as a small neural baseline. It is intentionally not large. The question is not whether a big model can overfit a small dataset; the question is whether a compact neural model adds any stable predictive value over simple baselines.

## 5. Evaluation framework

The main metric is daily rank-IC between model score and forward relative return.

I chose rank-IC instead of raw regression fit because the research question is about cross-sectional ranking, not point prediction accuracy. A strategy can be useful if it ranks stocks directionally better than random even when the raw return prediction error is large.

Evaluation includes:

- validation rank-IC for model selection;
- test rank-IC for final diagnostics;
- signal decay across 1, 5, 10, and 20 days;
- market-neutral long/short simulation;
- transaction-cost and turnover assumptions;
- hit rate, drawdown, and Sharpe-style diagnostics.

## 6. What did not work / research failures

This section is intentionally included because the project should read like research, not just a polished tool.

### Failed attempt 1: raw 1-day return as a standalone signal

Raw one-day return was unstable and often too noisy. It sometimes looked useful in isolated periods but did not provide a convincing standalone research story.

Decision: keep `ret_1d_lag1` only as a feature inside a broader model, not as a primary signal.

### Failed attempt 2: aggressive neural model framing

The first version leaned too hard on the PyTorch model. That was the wrong emphasis. With a small public OHLCV universe, a neural model can easily look sophisticated while adding little credible research value.

Decision: treat PyTorch MLP as one comparison point, not the center of the project.

### Failed attempt 3: “agentic workflow” framing

The first repo described the pipeline as agentic. That was inaccurate. The workflow is fixed and linear; it does not independently choose hypotheses, interpret results, and decide next experiments.

Decision: replace the phrase with “reproducible research pipeline” and document human research decisions separately.

### Failed attempt 4: volume features as an alpha claim

Volume z-score features are intuitive, but in the current setup they are not enough to claim a robust signal. They may matter more with event/news context, earnings dates, or broader universe coverage.

Decision: keep volume features as diagnostic inputs but do not overstate them.

## 7. Current interpretation

The current project should be read as an early-stage research case study.

The credible part is not that the repo has found alpha. The credible part is that the workflow shows the structure of research:

- point-in-time feature construction;
- leakage controls;
- model comparison;
- validation-first selection;
- test diagnostics;
- signal decay;
- cost-aware backtesting;
- documentation of failures and limitations.

The evidence bar for a real QR interview is much higher. This repo is a starting point for showing research discipline, not the final proof of research strength.

## 8. Next experiments

If I continued this research, I would prioritize:

1. Broaden universe from 15 mega-cap stocks to a larger liquid U.S. equity universe.
2. Add point-in-time fundamentals or event features.
3. Add earnings calendar / post-earnings drift style event windows.
4. Compare sector-neutral vs market-neutral ranking.
5. Test turnover-constrained portfolio construction.
6. Add subperiod / regime robustness checks.
7. Add bootstrap confidence intervals for rank-IC.
8. Add real research notebooks for exploratory analysis before productionizing the pipeline.

## 9. Final conclusion

The strongest honest conclusion is:

> This project does not prove tradable alpha, but it demonstrates a research process: forming hypotheses, constructing leakage-controlled features, comparing models, documenting failures, and evaluating signals through rank-IC, decay, and cost-aware portfolio diagnostics.

That is the level of claim the current evidence supports.
