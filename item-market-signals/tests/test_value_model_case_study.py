from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import requests

from market_signals.analysis.value_model_case_study import (
    build_value_model_case_study,
    render_value_model_case_study,
    run_value_model_case_study,
)
from market_signals.models.value_regression import (
    score_actual_values,
    select_training_rows,
    train_value_regression,
)


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "value_regression_items.csv"


def test_case_study_uses_model_results_and_deterministic_residual_examples() -> None:
    feature_matrix = pd.read_csv(FIXTURE)
    report = build_value_model_case_study(feature_matrix, min_training_samples=5)
    model = train_value_regression(feature_matrix, min_training_samples=5)
    scored = score_actual_values(model, select_training_rows(feature_matrix))
    examples = report.residual_examples.set_index("example_type")

    assert report.model_summary["selected_model"] == model.model_type
    assert report.model_summary["final_test_rows"] == model.metrics.sample_count
    assert report.model_summary["selection_rows"] == model.selection_metrics.sample_count
    selected_metrics = report.final_test_metrics.iloc[0]
    assert selected_metrics["r2_log"] == pytest.approx(model.metrics.r2_log)
    assert selected_metrics["mae_value_units"] == pytest.approx(model.metrics.mae_value)
    assert set(report.final_test_metrics["evaluation_scope"]) == {"final test partition"}
    assert set(report.selection_metrics["evaluation_scope"]) == {"selection partition"}
    assert report.selection_metrics.iloc[0]["r2_log"] == pytest.approx(model.selection_metrics.r2_log)
    assert examples.loc[
        "largest_positive_log_residual", "in_sample_structural_log_residual"
    ] == pytest.approx(scored["log_residual"].max())
    assert examples.loc[
        "largest_negative_log_residual", "in_sample_structural_log_residual"
    ] == pytest.approx(scored["log_residual"].min())
    assert examples.loc["nearest_zero_log_residual", "absolute_log_residual"] == pytest.approx(
        scored["abs_log_residual"].min()
    )
    assert not report.feature_importance.empty
    assert render_value_model_case_study(report) == render_value_model_case_study(
        build_value_model_case_study(feature_matrix, min_training_samples=5)
    )


def test_case_study_shap_reconstructs_log_prediction_when_forest_is_selected() -> None:
    report = build_value_model_case_study(pd.read_csv(FIXTURE), min_training_samples=5)

    assert report.model_summary["selected_model"] == "RandomForestRegressor"
    assert report.shap_summary is not None
    assert report.shap_summary["reconstructed_prediction_log"] == pytest.approx(
        report.shap_summary["model_prediction_log"], rel=1e-6, abs=1e-6
    )
    assert abs(float(report.shap_summary["additivity_error_log"])) < 1e-6
    assert "shap_value_log" in report.shap_contributions.columns


def test_case_study_reads_fixture_without_network_or_writes(tmp_path: Path, monkeypatch) -> None:
    feature_matrix = tmp_path / "fixture.csv"
    feature_matrix.write_bytes(FIXTURE.read_bytes())
    original_bytes = feature_matrix.read_bytes()

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("case study must not call the network")

    def fail_csv_write(*args: object, **kwargs: object) -> None:
        raise AssertionError("case study must not write CSV")

    monkeypatch.setattr(requests, "get", fail_network)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv_write)
    report = run_value_model_case_study(feature_matrix, min_training_samples=5)

    assert report.provenance["feature_matrix_sha256"] == sha256(original_bytes).hexdigest()
    assert report.provenance["gpovalues_snapshot_file"] == "not recorded in feature matrix"
    assert feature_matrix.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [feature_matrix]


def test_case_study_command_prints_expected_sections_offline() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_value_model_case_study.py"),
            "--feature-matrix",
            str(FIXTURE),
            "--min-training-samples",
            "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for heading in (
        "## Input Provenance",
        "## Selection Metrics",
        "## Final Test Metrics",
        "## In-Sample Structural Residual Diagnostics",
        "## Selected-Model Feature Importance",
        "## SHAP Log-Space Example",
        "## Limitations",
    ):
        assert heading in completed.stdout
    assert "log-space" in completed.stdout
    assert "not unbiased holdout estimates" in completed.stdout
