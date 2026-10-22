"""Reproducible, offline validation case study for structural value regression."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from market_signals.models.value_regression import (
    CONFIDENT_TRAINING_LABELS,
    LINEAR_INADEQUATE_R2,
    MIN_TRAINING_SAMPLES,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_FOREST_MIN_R2_GAIN,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_STATE,
    ValueRegressionMetrics,
    ValueRegressionResult,
    build_value_explainer,
    explain_value_prediction,
    feature_importance,
    score_actual_values,
    select_training_rows,
    train_value_regression,
    value_regression_test_size,
)

DEFAULT_FEATURE_MATRIX_PATH = OUTPUT_DIR / "feature_matrix_master.csv"


@dataclass(frozen=True)
class ValueModelCaseStudy:
    """Structured tables and metadata used by the Markdown case study."""

    provenance: dict[str, object]
    model_summary: dict[str, object]
    selection_metrics: pd.DataFrame
    final_test_metrics: pd.DataFrame
    residual_examples: pd.DataFrame
    feature_importance: pd.DataFrame
    shap_summary: dict[str, object] | None
    shap_contributions: pd.DataFrame


def run_value_model_case_study(
    feature_matrix_path: Path = DEFAULT_FEATURE_MATRIX_PATH,
    *,
    min_training_samples: int = MIN_TRAINING_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> ValueModelCaseStudy:
    """Read one committed feature matrix and build the case study without writes or network calls."""
    feature_matrix = pd.read_csv(feature_matrix_path)
    return build_value_model_case_study(
        feature_matrix,
        input_path=feature_matrix_path,
        min_training_samples=min_training_samples,
        random_state=random_state,
    )


def build_value_model_case_study(
    feature_matrix: pd.DataFrame,
    *,
    input_path: Path | None = None,
    min_training_samples: int = MIN_TRAINING_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> ValueModelCaseStudy:
    """Build deterministic selection, final-test, residual, and SHAP artifacts."""
    model = train_value_regression(
        feature_matrix,
        min_training_samples=min_training_samples,
        random_state=random_state,
    )
    training_rows = select_training_rows(feature_matrix)
    scored = score_actual_values(model, training_rows)
    selected_rows = _select_residual_examples(scored)
    residual_examples = _residual_example_table(selected_rows)
    importance = feature_importance(model).copy()

    shap_summary: dict[str, object] | None = None
    shap_contributions = pd.DataFrame(
        columns=["feature", "shap_value_log", "abs_shap_value_log", "direction"]
    )
    if model.model_type == "RandomForestRegressor" and not selected_rows.empty:
        example = selected_rows[selected_rows["example_type"] == "nearest_zero_log_residual"]
        if example.empty:
            example = selected_rows.head(1)
        source_index = example.index[0]
        source_row = training_rows.loc[source_index]
        explanation = explain_value_prediction(model, source_row, build_value_explainer(model))
        contribution_sum = float(explanation.contributions["shap_value_log"].sum())
        reconstructed = explanation.base_value_log + contribution_sum
        shap_summary = {
            "example_selection": "nearest absolute in-sample structural log residual",
            "item_name": _item_label(source_row),
            "observed_gpovalues_value": _optional_float(source_row.get("value")),
            "model_derived_predicted_value": explanation.predicted_value,
            "base_value_log": explanation.base_value_log,
            "contribution_sum_log": contribution_sum,
            "reconstructed_prediction_log": reconstructed,
            "model_prediction_log": explanation.prediction_log,
            "additivity_error_log": reconstructed - explanation.prediction_log,
        }
        shap_contributions = explanation.contributions.copy()

    return ValueModelCaseStudy(
        provenance=_provenance(feature_matrix, training_rows, input_path, random_state),
        model_summary=_model_summary(model, len(training_rows)),
        selection_metrics=_metrics_table(
            model.selection_metrics,
            model.baseline_selection_metrics,
            model.model_type,
            "selection partition",
        ),
        final_test_metrics=_final_test_metrics(model),
        residual_examples=residual_examples,
        feature_importance=importance,
        shap_summary=shap_summary,
        shap_contributions=shap_contributions,
    )


def render_value_model_case_study(case_study: ValueModelCaseStudy) -> str:
    """Render the structured case study as portable Markdown."""
    provenance = pd.DataFrame(
        [{"field": key, "value": value} for key, value in case_study.provenance.items()]
    )
    model_summary = pd.DataFrame(
        [{"field": key, "value": value} for key, value in case_study.model_summary.items()]
    )
    lines = [
        "# Structural Value Model Validation Case Study",
        "",
        "This report evaluates how the current structural model explains gpovalues' published values. "
        "It does not forecast future prices, correct gpovalues, or represent realized trades or profit.",
        "",
        "## Input Provenance",
        "",
        _markdown_table(provenance),
        "",
        "## Implemented Method",
        "",
        "Only positive-value rows with gpovalues confidence labeled medium or high are eligible. "
        "The target is `log(value)`. A deterministic random approximately 60/20/20 split separates initial "
        "fitting, model selection, and final testing. The linear stability guard sees observed rows "
        "outside the final test partition. Candidate models are compared only on the selection "
        "partition. After final testing, the chosen pipeline is refit on every eligible row.",
        "",
        _markdown_table(model_summary),
        "",
        "## Selection Metrics",
        "",
        "These scores guide model choice and are not unbiased holdout estimates. The stability "
        "guard also checks observed development rows, including rows outside the eligible set.",
        "",
        _markdown_table(case_study.selection_metrics),
        "",
        "## Final Test Metrics",
        "",
        "The final test partition is untouched by preprocessing fits, model choice, and the "
        "stability guard. These metrics are evaluated once after selection and before the "
        "all-eligible-row refit. The linear baseline is shown for context, not used to revisit "
        "the choice. Log-space and value-unit errors are reported separately.",
        "",
        _markdown_table(case_study.final_test_metrics),
        "",
        "## In-Sample Structural Residual Diagnostics",
        "",
        "These deterministic examples are selected after the all-eligible-row refit. A log residual is "
        "`log(observed gpovalues value) - log(model-derived prediction)`. It is a structural diagnostic, "
        "not held-out performance and not evidence that either value is a better trade price.",
        "",
        _markdown_table(case_study.residual_examples),
        "",
        "## Selected-Model Feature Importance",
        "",
        "Importance describes the selected model's structural fit. Linear coefficients are effects on the "
        "log target; RandomForest importances are relative split importances.",
        "",
        _markdown_table(case_study.feature_importance.head(15)),
        "",
        "## SHAP Log-Space Example",
        "",
    ]
    if case_study.shap_summary is None:
        lines.append(
            "The selected model is not RandomForestRegressor, so the current TreeExplainer path does not apply."
        )
    else:
        shap_summary = pd.DataFrame(
            [{"field": key, "value": value} for key, value in case_study.shap_summary.items()]
        )
        lines.extend(
            [
                "SHAP contributions are additive contributions to `log(value)`, not currency deltas. "
                "The base value plus every contribution is checked against the model's raw log prediction.",
                "",
                _markdown_table(shap_summary),
                "",
                "Largest absolute log-space contributions:",
                "",
                _markdown_table(case_study.shap_contributions.head(15)),
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One random item-level split is not a stability study or time-based validation; "
            "selection scores are not unbiased holdout estimates.",
            "- Small selection/test partitions make metrics noisy; duplicate item identities or "
            "shared market context can still limit generalization.",
            "- The model uses a limited structural feature set and may miss item-specific market context.",
            "- Inputs and results depend on the committed feature-matrix snapshot identified above.",
            "- The matrix records source generation times but not its exact gpovalues or tier snapshot filenames.",
            "- In-sample residuals are diagnostics after refitting, not held-out evidence.",
            "- Published-value fit does not predict future trades or measure realized transaction results.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _provenance(
    feature_matrix: pd.DataFrame,
    training_rows: pd.DataFrame,
    input_path: Path | None,
    random_state: int,
) -> dict[str, object]:
    generated_at = feature_matrix.get("generated_at", pd.Series(dtype=object)).dropna().astype(str)
    return {
        "feature_matrix_path": str(input_path) if input_path is not None else "in-memory dataframe",
        "feature_matrix_sha256": _file_sha256(input_path),
        "feature_matrix_rows": len(feature_matrix),
        "eligible_training_rows": len(training_rows),
        "source_generated_at_min": generated_at.min() if not generated_at.empty else "not available",
        "source_generated_at_max": generated_at.max() if not generated_at.empty else "not available",
        "gpovalues_snapshot_file": "not recorded in feature matrix",
        "tier_snapshot_file": "not recorded in feature matrix",
        "random_state": random_state,
    }


def _model_summary(model: ValueRegressionResult, training_count: int) -> dict[str, object]:
    return {
        "training_filter": "gpovalues confidence in medium/high and positive numeric value",
        "target": "log(value)",
        "selected_model": model.model_type,
        "eligible_rows": training_count,
        "initial_fit_rows": model.fit_sample_count,
        "selection_rows": model.selection_metrics.sample_count,
        "configured_test_fraction": value_regression_test_size(training_count),
        "final_test_rows": model.metrics.sample_count,
        "selection_note": model.selection_note,
        "linear_unstable_before_test": model.linear_unstable_before_test,
        "linear_stability_note": model.linear_stability_note,
        "linear_inadequate_r2_threshold": LINEAR_INADEQUATE_R2,
        "required_random_forest_r2_gain": RANDOM_FOREST_MIN_R2_GAIN,
        "random_forest_trees": RANDOM_FOREST_N_ESTIMATORS,
        "random_forest_min_samples_leaf": RANDOM_FOREST_MIN_SAMPLES_LEAF,
        "eligible_confidence_labels": ", ".join(sorted(CONFIDENT_TRAINING_LABELS)),
    }


def _final_test_metrics(model: ValueRegressionResult) -> pd.DataFrame:
    return _metrics_table(
        model.metrics, model.baseline_metrics, model.model_type, "final test partition"
    )


def _metrics_table(
    selected: ValueRegressionMetrics,
    baseline: ValueRegressionMetrics,
    selected_type: str,
    scope: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _metrics_row(scope, "selected model", selected_type, selected),
            _metrics_row(scope, "linear baseline", "LinearRegression", baseline),
        ]
    )


def _metrics_row(
    scope: str, role: str, model_type: str, metrics: ValueRegressionMetrics
) -> dict[str, object]:
    return {
        "evaluation_scope": scope,
        "model_role": role,
        "model_type": model_type,
        "r2_log": metrics.r2_log,
        "mae_log": metrics.mae_log,
        "mae_value_units": metrics.mae_value,
        "median_abs_error_value_units": metrics.median_abs_error_value,
        "rows": metrics.sample_count,
    }


def _select_residual_examples(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    candidates = scored.copy()
    candidates["_item_label"] = candidates.apply(_item_label, axis=1)
    selections: list[pd.Series] = []
    specs = [
        (
            "largest_positive_log_residual",
            "maximum in-sample structural log residual; item label breaks ties",
            candidates[candidates["log_residual"] > 0].sort_values(
                ["log_residual", "_item_label"], ascending=[False, True]
            ),
        ),
        (
            "largest_negative_log_residual",
            "minimum in-sample structural log residual; item label breaks ties",
            candidates[candidates["log_residual"] < 0].sort_values(
                ["log_residual", "_item_label"], ascending=[True, True]
            ),
        ),
        (
            "nearest_zero_log_residual",
            "minimum absolute in-sample structural log residual; item label breaks ties",
            candidates.sort_values(["abs_log_residual", "_item_label"], ascending=[True, True]),
        ),
    ]
    for example_type, selection_rule, ordered in specs:
        if ordered.empty:
            continue
        selected = ordered.iloc[0].copy()
        selected["example_type"] = example_type
        selected["selection_rule"] = selection_rule
        selections.append(selected)
    if not selections:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame(selections)


def _residual_example_table(selected_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "example_type",
        "selection_rule",
        "item_name",
        "observed_gpovalues_value",
        "model_derived_predicted_value",
        "in_sample_structural_log_residual",
        "absolute_log_residual",
    ]
    if selected_rows.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "example_type": selected_rows["example_type"],
            "selection_rule": selected_rows["selection_rule"],
            "item_name": selected_rows["_item_label"],
            "observed_gpovalues_value": selected_rows["actual_value"],
            "model_derived_predicted_value": selected_rows["predicted_value"],
            "in_sample_structural_log_residual": selected_rows["log_residual"],
            "absolute_log_residual": selected_rows["abs_log_residual"],
        }
    ).reset_index(drop=True)


def _item_label(row: pd.Series) -> str:
    for column in ("name", "slug", "join_key"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return f"row-{row.name}"


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return "not available"
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    headers = [str(column) for column in frame.columns]
    rows = [[_format_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)
