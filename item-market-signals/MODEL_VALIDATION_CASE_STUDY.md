# Structural Value Model Validation Case Study

Run this case study offline from `item-market-signals`:

```bash
python scripts/run_value_model_case_study.py
```

The command reads the committed `outputs/feature_matrix_master.csv` and prints
a Markdown report. It writes no files and makes no network request. The report
records the matrix path, its SHA-256 digest, its row count and source
`generated_at` range, and the fixed random state. The saved matrix does not
record its exact gpovalues or tier snapshot filenames, so the report marks
those as unavailable rather than inferring them from the latest files. To
inspect a different saved matrix, pass `--feature-matrix path/to/matrix.csv`;
the report identifies that input separately.

## What Is Validated

The runner calls the same `value_regression.py` functions as Model Insights.
Training uses gpovalues rows with medium or high confidence and a positive
numeric published `value`. The target is `log(value)`. The current model code
uses one deterministic random approximately 60/20/20 fit/selection/final-test
split with random state 42 (at least nine eligible rows are required; the
default training minimum remains 30). The baseline is
`LinearRegression`. A 300-tree `RandomForestRegressor` with at least three
samples per leaf is considered when the linear selection log-scale R2 is below
0.25 or its predictions fail the numerical stability guard. The forest is
selected when the linear model is unstable or, in the weak-fit case, its
selection R2 improves by at least 0.10. The guard inspects observed development
rows, including noneligible rows, but excludes the final test partition. The
generated report shows both candidates' selection metrics and final test metrics
separately. Selection scores guide the choice; they are not unbiased holdout
estimates.

The **final test** table reports log-scale R2 and MAE, plus MAE and median
absolute error after converting predictions back to value units. Those metrics
are calculated once, after model selection and before the selected model is
refit on all eligible rows. The linear baseline's final test score is shown only
for context; it cannot change the chosen model. The **in-sample structural residual**
examples use that refit. They are diagnostics of structural fit, not a second
held-out evaluation.
The runner selects the largest positive residual, largest negative residual,
and residual nearest zero by a stated deterministic rule with item-label
tie-breaking. The residual is `log(observed gpovalues value) -
log(model-derived prediction)`.

Feature importance describes the selected model. Linear coefficients act on
the log target; forest importances describe relative split importance. When a
forest is selected, the runner uses the existing TreeExplainer and prints one
SHAP example. Its base value plus all contributions is checked against the
raw `log(value)` prediction. SHAP contributions are log-space values, not
currency deltas.

## Interpretation Limits

This is one item-level random split, with no time-based validation or
repeated-split stability estimate. Small selection/test partitions are noisy,
and duplicate identities or shared market context could limit generalization.
The structural feature set is limited, and results change
when the feature matrix changes. An observed gpovalues value is a published
estimate; a model prediction is model-derived; a residual is a diagnostic.
None of these establishes a future price, a completed trade, or realized
profit. The command's live calculations provide the examples and metrics so
the narrative does not freeze a performance claim from an older snapshot.
