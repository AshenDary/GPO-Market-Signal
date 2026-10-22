from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_signals.models.value_regression import (
    InsufficientTrainingDataError,
    build_value_explainer,
    encode_feature_frame,
    explain_value_prediction,
    predict_value,
    prepare_feature_frame,
    select_training_rows,
    split_value_regression_rows,
    train_value_regression,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "value_regression_items.csv"


def _fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


def test_min_training_samples_guard_triggers_below_threshold() -> None:
    df = _fixture().head(2)

    with pytest.raises(InsufficientTrainingDataError, match="Need at least 9"):
        train_value_regression(df, min_training_samples=3)


def test_encoding_produces_sane_output() -> None:
    df = _fixture()
    model = train_value_regression(df, min_training_samples=5)

    prepared = prepare_feature_frame(df)
    encoded = encode_feature_frame(model, prepared)

    assert not encoded.empty
    assert not encoded.isna().any().any()
    assert any("tier_ordinal" in column for column in encoded.columns)
    assert any("is_unstable" in column for column in encoded.columns)


def test_higher_tier_predicts_higher_value_all_else_equal() -> None:
    df = _fixture()
    model = train_value_regression(df, min_training_samples=5)
    low_tier = {
        "tier_ordinal": 2,
        "sub_tier_ordinal": 2,
        "demand_ordinal": 2,
        "demand_ratio": 1.0,
        "category": "Weapon",
        "obtainability": "DROP",
        "is_unstable": False,
        "is_unobtainable": False,
    }
    high_tier = {**low_tier, "tier_ordinal": 8}

    assert predict_value(model, high_tier) > predict_value(model, low_tier)


def test_shap_values_reconstruct_raw_log_prediction() -> None:
    df = _fixture()
    model = train_value_regression(df, min_training_samples=5)
    explainer = build_value_explainer(model)
    row = df.iloc[0]

    explanation = explain_value_prediction(model, row, explainer)
    raw_prediction = float(model.pipeline.predict(prepare_feature_frame(pd.DataFrame([row])))[0])
    reconstructed = explanation.base_value_log + float(explanation.contributions["shap_value_log"].sum())

    assert reconstructed == pytest.approx(raw_prediction, rel=1e-6, abs=1e-6)


def test_noneligible_observed_row_still_triggers_linear_stability_guard() -> None:
    model = train_value_regression(_fixture(), min_training_samples=5)

    assert model.linear_unstable_before_test
    assert model.model_type == "RandomForestRegressor"


def test_final_test_rows_cannot_change_selection_or_stability_guard() -> None:
    df = _fixture().head(9).copy()
    baseline = train_value_regression(df, min_training_samples=5)
    _, _, test_positions = split_value_regression_rows(len(select_training_rows(df)))
    changed = df.copy()
    changed["value"] = changed["value"].astype(float)
    changed.loc[test_positions, "value"] = 1e100
    changed.loc[test_positions, "demand_ratio"] = 1e10
    after = train_value_regression(changed, min_training_samples=5)

    assert baseline.model_type == "LinearRegression"
    assert not baseline.linear_unstable_before_test
    assert baseline.model_type == after.model_type
    assert baseline.selection_note == after.selection_note
    assert baseline.linear_unstable_before_test == after.linear_unstable_before_test
    assert baseline.linear_stability_note == after.linear_stability_note
    assert baseline.selection_metrics == after.selection_metrics
    assert baseline.baseline_selection_metrics == after.baseline_selection_metrics
    assert baseline.metrics != after.metrics


def test_small_three_way_split_and_reproducibility() -> None:
    df = _fixture().head(9)
    first = train_value_regression(df, min_training_samples=5)
    second = train_value_regression(df, min_training_samples=5)
    fit, selection, test = split_value_regression_rows(9)

    assert len(fit) == 5
    assert len(selection) == len(test) == 2
    assert sorted(np.concatenate([fit, selection, test]).tolist()) == list(range(9))
    assert first.fit_sample_count == len(fit)
    assert first.selection_metrics.sample_count == len(selection)
    assert first.metrics.sample_count == len(test)
    assert first.model_type == second.model_type
    assert first.metrics == second.metrics
    assert first.selection_metrics == second.selection_metrics
    with pytest.raises(InsufficientTrainingDataError, match="at least 9"):
        train_value_regression(df.head(8), min_training_samples=5)
