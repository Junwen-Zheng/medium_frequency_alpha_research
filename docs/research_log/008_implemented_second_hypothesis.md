# 008 - Implemented Second Hypothesis in Code

## Purpose

The previous note documented a second hypothesis family: volatility-adjusted momentum and reversal. This update implements that hypothesis directly in the codebase rather than leaving it as a research note only.

The goal is to make the project more credible as a financial research case study by showing the loop from idea to implementation to evaluation output.

## Implemented feature family

Added the following point-in-time features:

- `reversal_5d_vol_adj`
- `momentum_20d_vol_adj`
- `momentum_60d_vol_adj`
- `liquidity_adjusted_momentum`

These features test whether normalizing recent price action by realized volatility produces a more stable signal than raw returns alone.

## Added diagnostics

The workflow now writes two additional diagnostic outputs:

- `outputs/hypothesis_family_comparison.csv`
- `outputs/regime_sliced_rank_ic.csv`

The first compares raw price/volume signals against the volatility-adjusted hypothesis family. The second checks whether the selected model's rank IC is concentrated in particular market regimes.

## Why this matters

A common weakness in toy quant projects is that they show a clean pipeline without showing whether a research idea survived comparison against a baseline. This update makes the project more research-like by adding a specific second hypothesis and a concrete diagnostic comparison.

## Interpretation

This still does not prove tradable alpha. The implementation is deliberately simple and should be treated as a research iteration. Its value is that it creates a more honest trail:

1. Form a hypothesis.
2. Implement the feature family.
3. Compare it against a baseline.
4. Slice results by regime.
5. Decide whether further research is justified.
