# Stage 1 study notes: owning the alpha repo

This stage is not about making the repository look more polished. It is about being able to defend the research logic under review.

## 1. Forward-return alignment

The label for `(date=t, ticker=AAPL)` must use AAPL's own future price, not the next row in the full multi-indexed dataframe. A global `shift(-1)` after a grouped calculation can move values across ticker boundaries if the dataframe is sorted by date/ticker.

Correct mental model:

```text
AAPL at t  -> AAPL at t+h
MSFT at t  -> MSFT at t+h
NVDA at t  -> NVDA at t+h
```

Incorrect mental model:

```text
row i -> row i+1
```

What changed:

- Added `forward_return_by_ticker()` in `src/features.py`.
- Added `forward_returns_by_ticker()` in `src/backtest.py`.
- Added tests that use a tiny hand-built dataset where the correct answer can be checked manually.

Study resources:

- Pandas `SeriesGroupBy.shift`: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.shift.html
- Pandas `Series.shift`: https://pandas.pydata.org/docs/reference/api/pandas.Series.shift.html

## 2. Synthetic data is for tests, not evidence

Synthetic data is useful for CI, unit tests, and examples that need to run offline. It is not evidence that a signal works. If the main result comes from synthetic data, a reviewer will assume the project avoided the hard parts: missing data, corporate actions, bad history, universe definition, survivorship bias, and unstable regimes.

What changed:

- README now says the primary research path is real public data.
- Synthetic mode is explicitly described as an offline smoke test only.
- The repo should not include generated synthetic outputs as persuasive research evidence.

Study resources:

- Read about survivorship bias, look-ahead bias, and backtest overfitting.
- Practical goal: be able to explain why yfinance is imperfect but still useful for a public learning project.

## 3. Reviewer-proof tests

A serious research repo needs tests that defend the assumptions most likely to break the conclusion.

Current stage-1 tests check:

- Forward returns are shifted within ticker.
- Feature and backtest forward-return helpers agree.
- Relative forward returns are cross-sectionally centered by date.
- The feature pipeline keeps targets centered after dropping missing rows.

Next tests to add:

- Train/validation/test dates do not overlap.
- Feature columns do not use the future target column.
- Signal decay uses predictions only on the intended evaluation window.
- Real-data mode fails loudly when data cannot be downloaded, instead of silently switching to synthetic results.

## 4. How to explain this to James

A good explanation is not "I fixed a bug." A better explanation is:

> I realized the previous backtest label construction could silently shift returns across ticker boundaries. I replaced it with explicit ticker-level forward-return helpers and added a toy-dataset test where the expected returns can be verified manually. I also changed the README so synthetic data is clearly a smoke-test path rather than research evidence.

## 5. Commit plan

Make these as real commits, one at a time:

```bash
git add README.md
git commit -m "Remove resume-oriented wording from research README"

git add src/features.py src/backtest.py tests/test_research_integrity.py
git commit -m "Fix forward-return alignment and add leakage guards"

git add docs/study_notes/stage1_research_ownership.md docs/research_log/2026-06-forward-return-alignment.md
git commit -m "Document forward-return alignment review"

git rm -f scripts/commit_sequence.md docs/github_review_checklist.md docs/research_note_template.md 2>/dev/null || true
git commit -m "Remove recruiter-facing repository artifacts"
```

Do not fake old history. From this point onward, the value is in real iteration and being able to explain every change.
