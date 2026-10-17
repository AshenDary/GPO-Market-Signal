from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from market_signals.analysis.market_change_alerts import MarketChangeAlertRules
from market_signals.analysis.snapshot_quality import (
    DEFAULT_METRIC_EXTREME_LIMITS,
    SnapshotQualityResult,
    run_snapshot_quality_preflight,
)


def _row(name: str, **overrides: object) -> dict[str, object]:
    base = {
        "slug": name.lower().replace(" ", "-"),
        "name": name,
        "join_key": name.lower().strip(),
        "value": 100.0,
        "demand_ratio": 1.0,
        "trade_count": 250,
    }
    return {**base, **overrides}


def _write_snapshot(snapshot_dir: Path, snapshot_date: str, rows: list[dict[str, object]]) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"gpovalues_{snapshot_date}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _snapshot(result: SnapshotQualityResult, snapshot_date: str) -> pd.Series:
    matches = result.snapshots[result.snapshots["snapshot_date"] == snapshot_date]
    assert len(matches) == 1
    return matches.iloc[0]


def test_clean_history_reports_no_gaps_or_concerns(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    rows = [_row("Candy Cane"), _row("Flower Sword")]
    _write_snapshot(snapshot_dir, "2026-09-01", rows)
    _write_snapshot(snapshot_dir, "2026-09-02", rows)

    result = run_snapshot_quality_preflight(snapshot_dir)
    second = _snapshot(result, "2026-09-02")

    assert result.summary["snapshot_count"] == 2
    assert result.summary["missing_calendar_dates"] == ()
    assert result.summary["quality_concern_snapshot_count"] == 0
    assert result.summary["coverage_change_snapshot_count"] == 0
    assert pd.isna(_snapshot(result, "2026-09-01")["added_identity_count"])
    assert second["added_identity_count"] == 0
    assert second["removed_identity_count"] == 0
    assert not bool(second["has_quality_concerns"])


def test_missing_calendar_date_is_reported(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane")])
    _write_snapshot(snapshot_dir, "2026-09-03", [_row("Candy Cane")])

    result = run_snapshot_quality_preflight(snapshot_dir)

    assert result.summary["missing_calendar_dates"] == ("2026-09-02",)
    assert result.summary["missing_calendar_date_count"] == 1


def test_coverage_changes_compare_with_preceding_available_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane"), _row("Flower Sword")])
    _write_snapshot(snapshot_dir, "2026-09-03", [_row("Candy Cane"), _row("New Item")])

    result = run_snapshot_quality_preflight(snapshot_dir)
    second = _snapshot(result, "2026-09-03")

    assert second["added_identity_count"] == 1
    assert second["removed_identity_count"] == 1
    assert bool(second["has_coverage_change"])
    assert not bool(second["has_quality_concerns"])


def test_duplicate_missing_and_fallback_identities_are_counted(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        "2026-09-01",
        [
            _row("Candy Cane"),
            _row("Candy Cane"),
            _row("Flower Sword", slug=""),
            _row("Name Only", slug="", join_key=""),
            _row("", slug="", join_key=""),
        ],
    )

    result = run_snapshot_quality_preflight(snapshot_dir)
    row = _snapshot(result, "2026-09-01")

    assert row["unique_identity_count"] == 3
    assert row["missing_identity_count"] == 1
    assert row["duplicate_identity_count"] == 1
    assert row["duplicate_identity_row_count"] == 2
    assert bool(row["has_quality_concerns"])


def test_metric_quality_categories_are_separate_and_reuse_alert_limits(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    rows = []
    values: list[object] = [None, "not-a-number", -1, 0, 100_000_001]
    demands: list[object] = [None, "not-a-number", -1, 0, 51]
    activity: list[object] = [None, "not-a-number", -1, 0, 1_000_001]
    for index in range(5):
        rows.append(
            _row(
                f"Item {index}",
                value=values[index],
                demand_ratio=demands[index],
                trade_count=activity[index],
            )
        )
    _write_snapshot(snapshot_dir, "2026-09-01", rows)

    result = run_snapshot_quality_preflight(snapshot_dir)
    row = _snapshot(result, "2026-09-01")

    for metric in ("value", "demand_ratio", "trade_count"):
        for category in ("missing", "non_numeric", "negative", "zero", "extreme"):
            assert row[f"{metric}_{category}_count"] == 1

    rules = MarketChangeAlertRules()
    assert DEFAULT_METRIC_EXTREME_LIMITS == {
        "value": rules.max_reasonable_value,
        "demand_ratio": rules.max_reasonable_demand_ratio,
        "trade_count": rules.max_reasonable_trade_count,
    }


def test_empty_directory_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No gpovalues snapshots"):
        run_snapshot_quality_preflight(tmp_path / "empty")


def test_preflight_never_calls_network_or_writes_files(tmp_path: Path, monkeypatch) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_path = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane")])
    output_path = tmp_path / "outputs" / "feature_matrix_master.csv"
    decision_path = tmp_path / "data" / "decisions" / "decision_log.csv"
    output_path.parent.mkdir(parents=True)
    decision_path.parent.mkdir(parents=True)
    output_path.write_text("generated output sentinel\n")
    decision_path.write_text("private decision sentinel\n")
    before = {
        snapshot_path: snapshot_path.read_bytes(),
        output_path: output_path.read_bytes(),
        decision_path: decision_path.read_bytes(),
    }

    def fail_network(*args, **kwargs):
        raise AssertionError("snapshot quality preflight must not call the network")

    def fail_write(*args, **kwargs):
        raise AssertionError("snapshot quality preflight must not write CSV files")

    monkeypatch.setattr(requests, "get", fail_network)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)

    result = run_snapshot_quality_preflight(snapshot_dir)

    assert result.summary["snapshot_count"] == 1
    assert {path: path.read_bytes() for path in before} == before
