# Research log: forward-return alignment and real-data discipline

## Problem noticed

A review pointed out that the market-neutral backtest could compute next-day returns incorrectly if a global shift moved values across ticker boundaries. That would invalidate the P&L diagnostic because weights for one stock could be multiplied against another stock's return.

## Fix

I added explicit ticker-level forward-return helpers:

- `src.features.forward_return_by_ticker`
- `src.backtest.forward_returns_by_ticker`

Both use group-level shifting by ticker. The test suite now includes a tiny hand-built dataset where AAA rises 10% and BBB falls 50%, making it easy to verify that the helper does not mix tickers.

## Synthetic data policy

I also changed the README language so synthetic data is described only as an offline smoke-test path. It should not be presented as evidence for signal quality.

## What I learned

The most important issue was not the specific bug. It was the habit of proving alignment assumptions with small deterministic examples before trusting a full backtest. Going forward, every research metric should have a minimal test that checks the data alignment behind it.

## Next work

- Run the pipeline on real public data.
- Save the generated report only after checking whether the outputs came from real data or synthetic mode.
- Add a test for train/validation/test split boundaries.
- Add a note on how missing prices and universe membership affect the interpretation.
