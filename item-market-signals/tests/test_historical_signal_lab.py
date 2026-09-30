from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_signals.analysis.historical_signal_lab import (
    MAX_OUTCOME_DELAY_DAYS,
    load_gpovalues_snapshots,
    run_historical_signal_lab,
)


def _write_snapshot(snapshot_dir: Path, snapshot_date: str, rows: list[dict[str, object]]) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"gpovalues_{snapshot_date}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(name: str, **overrides: object) -> dict[str, object]:
    base = {
        "slug": name.lower().replace(" ", "-"),
        "name": name,
        "shortcut": "",
        "value": 100.0,
        "ci_low": 90.0,
        "ci_high": 110.0,
        "confidence": "medium",
        "rarity": "Legendary",
        "tier": "A",
        "demand": "Stable Demand",
        "demand_ratio": 1.0,
        "trade_count": 250,
        "derivation": "test",
        "share_url": "https://example.test/item",
        "image_url": "https://example.test/item.png",
        "generated_at": f"{overrides.get('generated_at_date', '2026-09-01')}T01:00:00+00:00",
        "n_trades_used": 12345,
        "join_key": name.lower().strip(),
    }
    base.pop("generated_at_date", None)
    return {**base, **overrides}


def _observation(result, evaluation_date: str, item_name: str) -> pd.Series:
    matches = result.observations[
        (result.observations["evaluation_date"] == evaluation_date)
        & (result.observations["item_name"] == item_name)
    ]
    assert len(matches) == 1
    return matches.iloc[0]


def test_prior_signals_use_exact_past_snapshot_without_future_leakage(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100, demand_ratio=1.0)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=110, demand_ratio=1.5)])
    _write_snapshot(snapshot_dir, "2026-09-10", [_row("Candy Cane", value=999, demand_ratio=9.9)])
    _write_snapshot(snapshot_dir, "2026-09-22", [_row("Candy Cane", value=121, demand_ratio=2.0)])

    result = run_historical_signal_lab(snapshot_dir)
    row = _observation(result, "2026-09-08", "Candy Cane")

    assert row["prior_date"] == "2026-09-01"
    assert row["prior_value"] == pytest.approx(100)
    assert row["prior_7d_value_change_pct"] == pytest.approx(10)
    assert row["prior_7d_demand_ratio_change"] == pytest.approx(0.5)
    assert row["later_published_value"] == pytest.approx(121)
    assert row["outcome_change_pct"] == pytest.approx(10)


def test_duplicate_later_item_rows_are_not_used_as_outcomes(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=110)])
    _write_snapshot(
        snapshot_dir,
        "2026-09-22",
        [
            _row("Candy Cane", value=120),
            _row("Candy Cane", value=130),
        ],
    )

    result = run_historical_signal_lab(snapshot_dir)
    row = _observation(result, "2026-09-08", "Candy Cane")

    assert row["outcome_status"] == "duplicate_item"
    assert pd.isna(row["actual_outcome_date"])
    assert pd.isna(row["outcome_change_pct"])


def test_duplicate_evaluation_item_rows_are_explicit_and_not_eligible(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(
        snapshot_dir,
        "2026-09-08",
        [
            _row("Candy Cane", value=110),
            _row("Candy Cane", value=120),
        ],
    )
    _write_snapshot(snapshot_dir, "2026-09-22", [_row("Candy Cane", value=130)])

    result = run_historical_signal_lab(snapshot_dir)
    duplicate_rows = result.observations[
        (result.observations["evaluation_date"] == "2026-09-08")
        & (result.observations["item_name"] == "Candy Cane")
    ]

    assert len(duplicate_rows) == 2
    assert set(duplicate_rows["evaluation_status"]) == {"duplicate_evaluation_identity"}
    assert set(duplicate_rows["outcome_status"]) == {"not_evaluated"}


def test_missing_prior_dates_are_explicit_but_confidence_still_summarizes(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=100, confidence="low")])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=110, confidence="high")])
    _write_snapshot(snapshot_dir, "2026-09-22", [_row("Candy Cane", value=120, confidence="high")])

    result = run_historical_signal_lab(snapshot_dir)
    row = _observation(result, "2026-09-08", "Candy Cane")
    confidence_summary = result.summary[
        (result.summary["signal"] == "confidence_category")
        & (result.summary["group"] == "high")
        & (result.summary["cohort"] == "signal group")
    ]

    assert row["prior_7d_value_change_group"] == "missing"
    assert row["signal_note"] == "missing_prior_snapshot"
    assert len(confidence_summary) == 1
    assert int(confidence_summary.iloc[0]["observations"]) >= 1


def test_delayed_outcomes_use_first_snapshot_within_max_delay(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-24", [_row("Candy Cane", value=150)])

    result = run_historical_signal_lab(snapshot_dir)
    row = _observation(result, "2026-09-08", "Candy Cane")

    assert MAX_OUTCOME_DELAY_DAYS == 3
    assert row["target_date"] == "2026-09-22"
    assert row["actual_outcome_date"] == "2026-09-24"
    assert row["elapsed_days"] == 16
    assert row["outcome_change_pct"] == pytest.approx(50)


def test_outcomes_after_max_delay_are_reported_missing(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane", value=150)])

    result = run_historical_signal_lab(snapshot_dir)
    row = _observation(result, "2026-09-08", "Candy Cane")

    assert row["outcome_status"] == "no_snapshot_in_window"
    assert pd.isna(row["actual_outcome_date"])


def test_zero_evaluation_values_are_explicit_and_excluded_from_baseline(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100), _row("Kraken Blade", value=50)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=0), _row("Kraken Blade", value=60)])
    _write_snapshot(snapshot_dir, "2026-09-22", [_row("Candy Cane", value=10), _row("Kraken Blade", value=66)])

    result = run_historical_signal_lab(snapshot_dir)
    zero_row = _observation(result, "2026-09-08", "Candy Cane")
    eligible_on_date = result.observations[
        (result.observations["evaluation_date"] == "2026-09-08")
        & (result.observations["evaluation_status"] == "eligible")
    ]

    assert zero_row["evaluation_status"] == "zero_or_negative_evaluation_value"
    assert len(eligible_on_date) == 1
    assert eligible_on_date.iloc[0]["item_name"] == "Kraken Blade"


def test_baseline_aligns_to_signal_group_evaluation_dates(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100), _row("Kraken Blade", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=110), _row("Kraken Blade", value=90)])
    _write_snapshot(snapshot_dir, "2026-09-09", [_row("Extra Item", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-22", [_row("Candy Cane", value=120), _row("Kraken Blade", value=80)])
    _write_snapshot(snapshot_dir, "2026-09-23", [_row("Extra Item", value=150)])

    result = run_historical_signal_lab(snapshot_dir)
    baseline = result.summary[
        (result.summary["signal"] == "prior_7d_value_change")
        & (result.summary["group"] == "up")
        & (result.summary["cohort"] == "same-date baseline")
    ]

    assert len(baseline) == 1
    assert int(baseline.iloc[0]["observations"]) == 2
    assert int(baseline.iloc[0]["unique_items"]) == 2


def test_repeated_item_dates_are_observations_not_unique_items(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-08", [_row("Candy Cane", value=110)])
    _write_snapshot(snapshot_dir, "2026-09-15", [_row("Candy Cane", value=120)])
    _write_snapshot(snapshot_dir, "2026-09-22", [_row("Candy Cane", value=130)])
    _write_snapshot(snapshot_dir, "2026-09-29", [_row("Candy Cane", value=140)])

    result = run_historical_signal_lab(snapshot_dir)
    repeated_summary = result.summary[
        (result.summary["signal"] == "prior_7d_value_change")
        & (result.summary["group"] == "up")
        & (result.summary["cohort"] == "signal group")
    ]

    assert int(repeated_summary.iloc[0]["observations"]) == 4
    assert int(repeated_summary.iloc[0]["unique_items"]) == 1


def test_snapshot_loader_requires_snapshot_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gpovalues_snapshots(tmp_path / "empty")
