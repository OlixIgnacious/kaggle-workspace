# Linear regression: what to learn next (S4E5 flood prediction)

Where `s4e5.ipynb` is so far: fitting the model, OLS coefficients, a VIF check, P-P and Q-Q plots of the
residuals, and a held-out R² of **0.845**.

## 1. Fix two things in the current notebook

- **Drop `id` from `X`.** It's just a row number, and its p-value of 0.98 confirms it does nothing.
  It's also behind the "condition number 7.79e+06, multicollinearity?" warning: `id` goes up to
  about 1.1 million while the other features are around 0–16, so the warning comes from that
  difference in scale, not from collinearity. The VIF table, which excludes `id`, shows every
  feature at about 1.00, so there's no multicollinearity. Remove `id` and watch the warning disappear.
- **Look at the coefficients.** All 20 are about 0.0056. That suggests the target depends only on
  the *sum* of the features. Test: a model with the sum as its only input gets **R² = 0.8449**,
  the same as the full 20-feature model.

## 2. Finish the regression assumption checks

Normality is already checked. The other standard ones:

- **Linearity and equal variance:** plot residuals against fitted values. Then run
  `statsmodels.stats.diagnostic.het_breuschpagan` to test for heteroscedasticity (residual variance
  that changes with the prediction).
- **Influential points:** Cook's distance (the commented-out `influence_plot`). With 900k rows the
  plot is slow, so use a sample of about 5,000 rows.
- **Residual skew of 0.7:** usually means a non-linearity the model hasn't captured — see step 3.

## 3. Find the non-linearity

Mean target grouped by the feature sum rises steadily but not in a straight line:

| Feature sum | Mean FloodProbability |
|---|---|
| 75 | 0.388 |
| 87 | 0.436 |
| 99 | 0.508 |
| 111 | 0.576 |
| 123 | 0.636 |

Plot `FloodProbability` against `sum` to see the curve. Every row below is still plain
`LinearRegression`; only the features change (same 80/20 split, `random_state=42`):

| Features | R² |
|---|---|
| 20 raw features (current model) | 0.8449 |
| Sum only | 0.8449 |
| Raw + row std/max/min | 0.8450 |
| Sorted feature values per row | 0.8450 |
| Sorted values + one-hot of `sum` | **0.8658** |

Key idea: the model has to be linear in its *coefficients*, but the features can be any
transformation of the data. One-hot encoding `sum` lets the model fit a separate level for each sum
value. Also try `sum²` and `sum³`, or sklearn's `PolynomialFeatures` / `SplineTransformer` on `sum`.

## 4. Regularization

Try `Ridge` and `Lasso` with a few values of `alpha`. On the raw features they'll barely change
anything, because there's no multicollinearity and there are 900k rows. Then try them on the one-hot
model, where there are many more features and some sum values have very few rows, and compare.

## 5. Evaluate properly

- Use `cross_val_score` with `KFold(5)` instead of a single 80/20 split; report mean ± std of R².
- The target is a probability: check for predictions below 0 or above 1 and clip with `np.clip`.
- Make a submission: predict on `df_test`, write `id,FloodProbability` in the format of
  `sample_submission.csv`, and upload. The competition scores on R², so the leaderboard score is
  directly comparable to the validation number.

## 6. Compare with other models

Fit a `HistGradientBoostingRegressor` on the raw features. It finds non-linearities without manual
feature engineering, so it shows how much of the gap step 3 closed by hand.
