# Stage 2: real-data workflow boundary

James's review made the data boundary issue clear: synthetic data can be useful for tests, but it should not be allowed to look like research evidence.

Changes in this pass:

- Real public OHLCV data remains the default workflow path.
- Synthetic runs are exposed only through an explicit `--smoke-test` flag.
- Synthetic data is kept for unit tests and offline pipeline checks only.
- Real-data failure now fails loudly instead of silently switching to synthetic data.
- Workflow summaries now record `data_mode` so generated outputs can be traced back to either `real_public_ohlcv` or `synthetic_smoke_test`.

Study notes for ownership:

- A backtest is only meaningful if the data source is clear.
- Synthetic data can test code paths, but it cannot validate alpha.
- A serious research workflow should fail on missing real data rather than quietly producing convenient fake results.
- Before interpreting any report, check whether the run used `real_public_ohlcv` or `synthetic_smoke_test`.
