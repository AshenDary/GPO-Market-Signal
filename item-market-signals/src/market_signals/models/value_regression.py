"""Structural value regression for item prices.

This model is deliberately structural: it learns how tier/demand/category
features explain gpovalues' own solved prices. It is not a replacement for
observed market values. Estimates from this module must stay labeled as
model-derived wherever they are displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import OUTPUT_DIR, SNAPSHOT_DIR
from market_signals.features.build_feature_matrix import build_feature_matrix

MIN_TRAINING_SAMPLES = 30
CONFIDENT_TRAINING_LABELS = {"medium", "high"}
ANOMALY_STD_THRESHOLD = 1.5
LINEAR_INADEQUATE_R2 = 0.25
RANDOM_FOREST_MIN_R2_GAIN = 0.10
RANDOM_FOREST_N_ESTIMATORS = 300
RANDOM_FOREST_MIN_SAMPLES_LEAF = 3
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "tier_ordinal",
    "sub_tier_ordinal",
    "demand_ordinal",
    "demand_ratio",
]
CATEGORICAL_FEATURES = ["category", "obtainability"]
BOOLEAN_FEATURES = ["is_unstable", "is_unobtainable"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


class InsufficientTrainingDataError(ValueError):
    """Raised when the confident gpovalues sample is too small to train."""


@dataclass(frozen=True)
class ValueRegressionMetrics:
    r2_log: float
    mae_log: float
    mae_value: float
    median_abs_error_value: float
    test_sample_count: int


@dataclass(frozen=True)
class ValueRegressionResult:
    pipeline: Pipeline
    model_type: str
    metrics: ValueRegressionMetrics
    baseline_metrics: ValueRegressionMetrics
    training_sample_count: int
    residual_std_log: float
    feature_names: list[str]
    selection_note: str


@dataclass(frozen=True)
class ValueShapExplanation:
    base_value_log: float
    prediction_log: float
    predicted_value: float
    contributions: pd.DataFrame


def _latest(pattern: str, snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    matches = sorted(snapshot_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching '{pattern}' in {snapshot_dir}.")
    return matches[-1]


def load_latest_feature_matrix(path: Path = OUTPUT_DIR / "feature_matrix_master.csv") -> pd.DataFrame:
    """Load the saved merged matrix, or build it from the latest snapshots."""
    try:
        return build_feature_matrix()
    except FileNotFoundError:
        if path.exists():
            return pd.read_csv(path)
        raise


def load_latest_tier_reference(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Load the latest structural tier-reference snapshot."""
    return pd.read_csv(_latest("tier_reference_*.csv", snapshot_dir))


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame with stable columns and missing values."""
    features = pd.DataFrame(index=df.index)
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(df.get(column), errors="coerce")
    for column in CATEGORICAL_FEATURES:
        values = df.get(column, pd.Series(index=df.index, dtype=object))
        features[column] = values.fillna("unknown").astype(str).replace({"": "unknown"})
    for column in BOOLEAN_FEATURES:
        values = df.get(column, pd.Series(index=df.index, dtype=object))
        mapped = values.map(_coerce_bool)
        features[column] = mapped.map(lambda value: bool(value) if value is not None else False).astype(float)
    return features[MODEL_FEATURES]


def train_value_regression(
    feature_matrix: pd.DataFrame | None = None,
    *,
    min_training_samples: int = MIN_TRAINING_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> ValueRegressionResult:
    """Train a linear structural value model with a guarded RF fallback.

    The baseline is plain linear regression because the point is interpretably
    measuring structural fit. Random forest is only selected when the held-out
    linear R^2 is plainly weak and the forest improves it by a meaningful
    margin, which keeps complexity from hiding a mediocre structural signal.
    """
    df = load_latest_feature_matrix() if feature_matrix is None else feature_matrix.copy()
    training_df = select_training_rows(df)
    training_count = len(training_df)
    if training_count < min_training_samples:
        raise InsufficientTrainingDataError(
            f"Need at least {min_training_samples} medium/high-confidence items to train "
            f"the structural value model; found {training_count}."
        )

    x = prepare_feature_frame(training_df)
    y = np.log(pd.to_numeric(training_df["value"], errors="coerce").astype(float))
    test_size = value_regression_test_size(training_count)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    linear_pipeline = _build_pipeline(LinearRegression())
    linear_pipeline.fit(x_train, y_train)
    linear_metrics = _evaluate_pipeline(linear_pipeline, x_test, y_test)

    selected_pipeline = linear_pipeline
    selected_metrics = linear_metrics
    selected_type = "LinearRegression"
    selection_note = "Linear baseline selected for interpretability; held-out fit was adequate."

    # Keep linear regression as the baseline, but do not ship it when sparse
    # structural labels make its extrapolations numerically unstable.
    linear_unstable, instability_note = _has_unstable_predictions(linear_pipeline, df)
    if linear_metrics.r2_log < LINEAR_INADEQUATE_R2 or linear_unstable:
        forest_pipeline = _build_pipeline(
            RandomForestRegressor(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                random_state=random_state,
            ),
            scale_numeric=False,
        )
        forest_pipeline.fit(x_train, y_train)
        forest_metrics = _evaluate_pipeline(forest_pipeline, x_test, y_test)
        if linear_unstable or forest_metrics.r2_log >= linear_metrics.r2_log + RANDOM_FOREST_MIN_R2_GAIN:
            selected_pipeline = forest_pipeline
            selected_metrics = forest_metrics
            selected_type = "RandomForestRegressor"
            selection_note = (
                f"Random forest selected because the linear baseline was inadequate: {instability_note}"
                if linear_unstable
                else "Random forest selected because linear held-out R2 was plainly weak and RF improved it."
            )

    selected_pipeline.fit(x, y)
    all_predictions = selected_pipeline.predict(x)
    residual_std = float(np.std(y - all_predictions, ddof=1)) if len(y) > 1 else 0.0
    feature_names = _feature_names(selected_pipeline)

    return ValueRegressionResult(
        pipeline=selected_pipeline,
        model_type=selected_type,
        metrics=selected_metrics,
        baseline_metrics=linear_metrics,
        training_sample_count=training_count,
        residual_std_log=residual_std,
        feature_names=feature_names,
        selection_note=selection_note,
    )


def predict_value(model: ValueRegressionResult, item_features: Mapping[str, object] | pd.Series) -> float:
    """Predict one model-derived value from structural features."""
    row = pd.DataFrame([dict(item_features)])
    prediction_log = float(_predict(model.pipeline, prepare_feature_frame(row))[0])
    return float(np.exp(prediction_log))


def build_value_explainer(model: ValueRegressionResult) -> Any:
    """Build a SHAP tree explainer for the already-trained random forest."""
    estimator = model.pipeline.named_steps["model"]
    if not isinstance(estimator, RandomForestRegressor):
        raise TypeError("SHAP TreeExplainer is only wired for the selected RandomForestRegressor model.")

    import shap

    return shap.TreeExplainer(estimator)


def explain_value_prediction(
    model: ValueRegressionResult,
    item_features: Mapping[str, object] | pd.Series,
    explainer: Any | None = None,
) -> ValueShapExplanation:
    """Explain one model prediction in log-value space with per-feature SHAP values."""
    row = pd.DataFrame([dict(item_features)])
    encoded = encode_feature_frame(model, row)
    value_explainer = explainer or build_value_explainer(model)
    raw_shap_values = value_explainer.shap_values(encoded)

    if isinstance(raw_shap_values, list):
        raw_shap_values = raw_shap_values[0]
    shap_values = np.asarray(raw_shap_values, dtype=float)
    if shap_values.ndim == 1:
        row_values = shap_values
    else:
        row_values = shap_values[0]

    expected_value = np.asarray(value_explainer.expected_value, dtype=float).reshape(-1)[0]
    base_value_log = float(expected_value)
    prediction_log = float(model.pipeline.named_steps["model"].predict(encoded.to_numpy())[0])
    contributions = (
        pd.DataFrame(
            {
                "feature": [_clean_feature_name(name) for name in model.feature_names],
                "shap_value_log": row_values,
                "abs_shap_value_log": np.abs(row_values),
            }
        )
        .sort_values("abs_shap_value_log", ascending=False)
        .reset_index(drop=True)
    )
    contributions["direction"] = np.where(
        contributions["shap_value_log"] >= 0,
        "pushes prediction up",
        "pushes prediction down",
    )

    return ValueShapExplanation(
        base_value_log=base_value_log,
        prediction_log=prediction_log,
        predicted_value=float(np.exp(prediction_log)),
        contributions=contributions,
    )


def add_predictions(model: ValueRegressionResult, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with model-derived predicted values."""
    scored = df.copy()
    predicted_log = _predict(model.pipeline, prepare_feature_frame(scored))
    scored["predicted_log_value"] = predicted_log
    scored["predicted_value"] = np.exp(predicted_log)
    scored["value_source"] = "model-derived"
    return scored


def score_actual_values(model: ValueRegressionResult, df: pd.DataFrame) -> pd.DataFrame:
    """Score observed gpovalues rows and flag notable structural residuals."""
    scored = add_predictions(model, df)
    scored["actual_value"] = pd.to_numeric(scored.get("value"), errors="coerce")
    has_actual = scored["actual_value"].notna() & (scored["actual_value"] > 0)
    scored = scored[has_actual].copy()
    scored["actual_log_value"] = np.log(scored["actual_value"])
    scored["log_residual"] = scored["actual_log_value"] - scored["predicted_log_value"]
    scored["abs_log_residual"] = scored["log_residual"].abs()
    residual_std = float(scored["log_residual"].std(ddof=1)) if len(scored) > 1 else 0.0
    scored["residual_std_log"] = residual_std
    scored["notable_structural_residual"] = (
        scored["abs_log_residual"] > (ANOMALY_STD_THRESHOLD * residual_std)
        if residual_std > 0
        else False
    )
    scored["structural_note"] = np.where(
        scored["notable_structural_residual"],
        "Priced differently than similar items by tier/category; not well-explained by structural features alone.",
        "Within the ordinary structural residual range.",
    )
    return scored


def build_model_estimates(
    model: ValueRegressionResult,
    feature_matrix: pd.DataFrame,
    tier_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate low-confidence and tier-only items with permanent labels."""
    low_confidence = feature_matrix[
        feature_matrix["confidence"].fillna("").astype(str).str.lower().eq("low")
    ].copy()
    low_confidence["estimate_reason"] = "thin gpovalues data"
    low_confidence["actual_value"] = pd.to_numeric(low_confidence.get("value"), errors="coerce")

    tier_only = _tier_only_rows(feature_matrix, tier_reference)
    if not tier_only.empty:
        tier_only["estimate_reason"] = "no gpovalues match"
        tier_only["actual_value"] = np.nan

    candidates = pd.concat([low_confidence, tier_only], ignore_index=True, sort=False)
    if candidates.empty:
        return candidates

    estimates = add_predictions(model, candidates)
    estimates["estimate_label"] = "model-derived estimate"
    return estimates


def estimate_item_by_name(
    model: ValueRegressionResult,
    item_name: str,
    feature_matrix: pd.DataFrame,
    tier_reference: pd.DataFrame | None = None,
) -> tuple[pd.Series | None, str | None]:
    """Estimate a non-gpovalues item when it has one structural match."""
    name = item_name.lower().strip()
    if not name:
        return None, "Search for an item before estimating it."

    candidates = _tier_only_rows(feature_matrix, tier_reference)
    if candidates.empty:
        return None, "No structural tier-reference items are available for model estimates."

    exact = candidates[candidates["join_key"] == name]
    if exact.empty and "alias" in candidates.columns:
        exact = candidates[candidates["alias"].fillna("").astype(str).str.lower().str.strip() == name]
    if len(exact) == 1:
        estimated = add_predictions(model, exact.head(1)).iloc[0]
        return estimated, None
    if len(exact) > 1:
        return None, f"Multiple structural matches found for '{item_name}'. Be more specific."

    contains = candidates[candidates["name"].fillna("").astype(str).str.lower().str.contains(name, regex=False)]
    if len(contains) == 1:
        estimated = add_predictions(model, contains.head(1)).iloc[0]
        return estimated, None
    if len(contains) > 1:
        return None, f"Multiple structural matches found for '{item_name}'. Be more specific."
    return None, f"No gpovalues or structural tier-reference match found for '{item_name}'."


def feature_importance(model: ValueRegressionResult) -> pd.DataFrame:
    """Return coefficient/importances with larger absolute effects first."""
    estimator = model.pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float)
        importance = np.abs(values)
        metric_name = "log coefficient"
    else:
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance = values
        metric_name = "importance"

    return (
        pd.DataFrame(
            {
                "feature": [_clean_feature_name(name) for name in model.feature_names],
                metric_name: values,
                "absolute_effect": importance,
            }
        )
        .sort_values("absolute_effect", ascending=False)
        .reset_index(drop=True)
    )


def encode_feature_frame(model: ValueRegressionResult, df: pd.DataFrame) -> pd.DataFrame:
    """Expose the fitted encoder output for tests and diagnostics."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        encoded = model.pipeline.named_steps["preprocessor"].transform(prepare_feature_frame(df))
    return pd.DataFrame(encoded, columns=model.feature_names, index=df.index)


def select_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return positive-value rows with gpovalues medium/high confidence."""
    confidence = df["confidence"].fillna("").astype(str).str.lower()
    values = pd.to_numeric(df.get("value"), errors="coerce")
    return df[confidence.isin(CONFIDENT_TRAINING_LABELS) & values.notna() & (values > 0)].copy()


def value_regression_test_size(training_count: int) -> float:
    """Return the configured held-out fraction for a training-row count."""
    if training_count < 1:
        raise ValueError("training_count must be greater than zero.")
    return 0.25 if training_count >= 40 else max(0.2, min(0.35, 8 / training_count))


def _build_pipeline(estimator: object, *, scale_numeric: bool = True) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("boolean", SimpleImputer(strategy="constant", fill_value=0.0), BOOLEAN_FEATURES),
        ],
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _evaluate_pipeline(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> ValueRegressionMetrics:
    predictions = _predict(pipeline, x_test)
    actual_values = np.exp(y_test)
    predicted_values = np.exp(predictions)
    value_errors = np.abs(actual_values - predicted_values)
    return ValueRegressionMetrics(
        r2_log=float(r2_score(y_test, predictions)),
        mae_log=float(mean_absolute_error(y_test, predictions)),
        mae_value=float(mean_absolute_error(actual_values, predicted_values)),
        median_abs_error_value=float(np.median(value_errors)),
        test_sample_count=len(y_test),
    )


def _has_unstable_predictions(pipeline: Pipeline, df: pd.DataFrame) -> tuple[bool, str]:
    values = pd.to_numeric(df.get("value"), errors="coerce")
    actual_df = df[values.notna() & (values > 0)].copy()
    if actual_df.empty:
        return False, "no observed values available for stability check"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        predictions = pipeline.predict(prepare_feature_frame(actual_df))
    runtime_warnings = [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    if runtime_warnings:
        return True, "linear predictions emitted numerical runtime warnings on observed rows"
    if not np.isfinite(predictions).all():
        return True, "linear predictions included non-finite values"

    residuals = np.log(pd.to_numeric(actual_df["value"], errors="coerce").astype(float)) - predictions
    max_abs_residual = float(np.max(np.abs(residuals)))
    if max_abs_residual > 5.0:
        return True, f"linear extrapolated beyond a 5.0 log-residual stability guard (max {max_abs_residual:.2f})"
    return False, "linear predictions stayed within the stability guard"


def _predict(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        return pipeline.predict(features)


def _feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def _tier_only_rows(feature_matrix: pd.DataFrame, tier_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    tier_ref = load_latest_tier_reference() if tier_reference is None else tier_reference.copy()
    tier_ref = tier_ref.rename(columns={"item_name": "name", "tier": "tier_tier"})
    tier_ref["join_key"] = tier_ref["name"].astype(str).str.lower().str.strip()

    matched_keys = set(
        feature_matrix.get("tier_item_name", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
    )
    if not matched_keys:
        matched_keys = set(feature_matrix.get("join_key", pd.Series(dtype=object)).dropna().astype(str))

    return tier_ref[~tier_ref["join_key"].isin(matched_keys)].copy()


def _coerce_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _clean_feature_name(name: str) -> str:
    for prefix in ("numeric__", "categorical__", "boolean__"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name
