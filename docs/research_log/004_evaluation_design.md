# 004 — Evaluation design

## Date
2026-06-19

## Why rank-IC
The strategy is a ranking problem. I care whether scores rank future relative returns, not whether the model predicts exact returns.

## Splitting policy
Use time-based train / validation / test splits. Do not randomly split rows because that would leak time structure and cross-sectional regimes.

## Model selection
Select the model on validation mean rank-IC, then evaluate once on the test slice.

## Backtest role
The backtest is diagnostic only. It checks whether rank-IC plausibly translates into a market-neutral portfolio after turnover and cost assumptions. It is not production portfolio construction.
