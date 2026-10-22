from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from market_signals.analysis.market_change_alerts import (
    MarketChangeAlertResult,
    MarketChangeAlertRules,
    run_market_change_alerts,
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


def _evaluation(result, item_name: str, alert_type: str) -> pd.Series:
    matches = result.evaluations[
        (result.evaluations["item_name"] == item_name)
        & (result.evaluations["alert_type"] == alert_type)
    ]
    assert len(matches) == 1
    return matches.iloc[0]


def _market_value_context(result: MarketChangeAlertResult) -> dict[str, object]:
    context = result.metadata["market_value_context"]
    assert isinstance(context, dict)
    return context


def test_rising_value_change_fires_with_exact_reason(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-03", [_row("Candy Cane", value=1200)])

    result = run_market_change_alerts(snapshot_dir)
    row = _evaluation(result, "Candy Cane", "value_movement")

    assert bool(row["fired"]) is True
    assert row["status"] == "alert"
    assert row["current_value"] == pytest.approx(1200)
    assert row["comparison_value"] == pytest.approx(100)
    assert row["percent_change"] == pytest.approx(1100)
    assert "Published value increased" in row["reason"]


def test_falling_demand_change_fires_as_absolute_ratio_delta(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", demand_ratio=4.0)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", demand_ratio=4.0)])
    _write_snapshot(snapshot_dir, "2026-09-03", [_row("Candy Cane", demand_ratio=2.5)])

    result = run_market_change_alerts(snapshot_dir)
    row = _evaluation(result, "Candy Cane", "demand_change")

    assert bool(row["fired"]) is True
    assert row["direction"] == "down"
    assert row["absolute_change"] == pytest.approx(-1.5)
    assert pd.isna(row["percent_change"])
    assert "Demand ratio decreased -1.50 points" in row["reason"]


def test_missing_history_is_visible_and_does_not_fire(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=125)])

    result = run_market_change_alerts(snapshot_dir)
    row = _evaluation(result, "Candy Cane", "value_movement")

    assert bool(row["fired"]) is False
    assert row["status"] == "insufficient_history"
    assert row["item_snapshot_dates"] == 2
    assert "Need at least 3 unambiguous snapshot dates" in row["reason"]


def test_latest_item_missing_from_comparison_snapshot_is_explicit(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=100)])
    _write_snapshot(
        snapshot_dir,
        "2026-09-03",
        [_row("Candy Cane", value=100), _row("New Item", value=500)],
    )

    result = run_market_change_alerts(snapshot_dir)
    row = _evaluation(result, "New Item", "value_movement")

    assert row["status"] == "missing_comparison_item"
    assert bool(row["fired"]) is False
    assert "not in the comparison snapshot" in row["reason"]


def test_duplicate_latest_identities_do_not_fire(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=100)])
    _write_snapshot(
        snapshot_dir,
        "2026-09-03",
        [_row("Candy Cane", value=125), _row("Candy Cane", value=130)],
    )

    result = run_market_change_alerts(snapshot_dir)
    rows = result.evaluations[
        (result.evaluations["item_name"] == "Candy Cane")
        & (result.evaluations["alert_type"] == "value_movement")
    ]

    assert len(rows) == 2
    assert set(rows["status"]) == {"duplicate_current_identity"}
    assert not rows["fired"].any()


def test_invalid_and_extreme_values_are_guarded(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    rules = MarketChangeAlertRules(max_reasonable_value=1_000)
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=100)])
    _write_snapshot(
        snapshot_dir,
        "2026-09-03",
        [
            _row("Candy Cane", value=-1),
            _row("Prestige Item", value=5_000),
        ],
    )
    _write_snapshot(
        snapshot_dir,
        "2026-09-04",
        [
            _row("Candy Cane", value=-1),
            _row("Prestige Item", value=5_000),
        ],
    )

    result = run_market_change_alerts(snapshot_dir, rules=rules)
    invalid_row = _evaluation(result, "Candy Cane", "value_movement")
    extreme_row = _evaluation(result, "Prestige Item", "value_movement")

    assert invalid_row["status"] == "negative_current"
    assert bool(invalid_row["fired"]) is False
    assert extreme_row["status"] == "extreme_current"
    assert bool(extreme_row["fired"]) is False


def test_decreasing_trade_count_can_fire_as_activity_change(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", trade_count=500)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", trade_count=500)])
    _write_snapshot(snapshot_dir, "2026-09-03", [_row("Candy Cane", trade_count=200)])

    result = run_market_change_alerts(snapshot_dir)
    row = _evaluation(result, "Candy Cane", "activity_change")

    assert bool(row["fired"]) is True
    assert row["direction"] == "down"
    assert row["absolute_change"] == pytest.approx(-300)
    assert row["percent_change"] == pytest.approx(-60)
    assert "activity signal decreased" in row["reason"]
    assert "not new trades today" in row["reason"]


def test_stale_latest_snapshot_is_reported_in_metadata(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-03", [_row("Candy Cane", value=100)])

    result = run_market_change_alerts(snapshot_dir, as_of=date(2026, 9, 10))

    assert result.metadata["days_since_latest_snapshot"] == 7
    assert result.metadata["is_stale"] is True


def test_market_value_context_reports_mostly_rising_values(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        "2026-09-01",
        [
            _row("Candy Cane", value=100),
            _row("Flower Sword", value=200),
            _row("Prestige Bag", value=400),
            _row("Flat Hat", value=100),
        ],
    )
    _write_snapshot(
        snapshot_dir,
        "2026-09-02",
        [
            _row("Candy Cane", value=110),
            _row("Flower Sword", value=260),
            _row("Prestige Bag", value=420),
            _row("Flat Hat", value=100),
        ],
    )

    context = _market_value_context(run_market_change_alerts(snapshot_dir))

    assert context["status"] == "ready"
    assert context["comparison_date"] == "2026-09-01"
    assert context["current_date"] == "2026-09-02"
    assert context["compared_items"] == 4
    assert context["rose_count"] == 3
    assert context["fell_count"] == 0
    assert context["flat_count"] == 1
    assert context["rose_percentage"] == pytest.approx(75)
    assert context["fell_percentage"] == pytest.approx(0)
    assert context["flat_percentage"] == pytest.approx(25)
    assert context["direction"] == "mostly_rising"
    assert context["median_percent_change"] == pytest.approx(7.5)


def test_market_value_context_reports_mostly_falling_values(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        "2026-09-01",
        [
            _row("Candy Cane", value=100),
            _row("Flower Sword", value=200),
            _row("Prestige Bag", value=400),
            _row("Flat Hat", value=100),
        ],
    )
    _write_snapshot(
        snapshot_dir,
        "2026-09-02",
        [
            _row("Candy Cane", value=90),
            _row("Flower Sword", value=150),
            _row("Prestige Bag", value=380),
            _row("Flat Hat", value=100),
        ],
    )

    context = _market_value_context(run_market_change_alerts(snapshot_dir))

    assert context["status"] == "ready"
    assert context["compared_items"] == 4
    assert context["rose_count"] == 0
    assert context["fell_count"] == 3
    assert context["flat_count"] == 1
    assert context["direction"] == "mostly_falling"
    assert context["median_percent_change"] == pytest.approx(-7.5)


def test_market_value_context_reports_mixed_values(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        "2026-09-01",
        [
            _row("Candy Cane", value=100),
            _row("Flower Sword", value=100),
            _row("Flat Hat", value=100),
        ],
    )
    _write_snapshot(
        snapshot_dir,
        "2026-09-02",
        [
            _row("Candy Cane", value=110),
            _row("Flower Sword", value=90),
            _row("Flat Hat", value=100),
        ],
    )

    context = _market_value_context(run_market_change_alerts(snapshot_dir))

    assert context["status"] == "ready"
    assert context["compared_items"] == 3
    assert context["rose_count"] == 1
    assert context["fell_count"] == 1
    assert context["flat_count"] == 1
    assert context["direction"] == "mixed"
    assert context["median_percent_change"] == pytest.approx(0)


def test_market_value_context_reports_insufficient_comparable_data(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-02", [_row("Flower Sword", value=120)])

    context = _market_value_context(run_market_change_alerts(snapshot_dir))

    assert context["status"] == "insufficient_comparable_data"
    assert context["comparison_date"] == "2026-09-01"
    assert context["current_date"] == "2026-09-02"
    assert context["compared_items"] == 0
    assert context["median_percent_change"] is None
    assert context["rose_count"] == 0
    assert context["fell_count"] == 0
    assert context["flat_count"] == 0
    assert context["direction"] == "insufficient_comparable_data"
    assert "No item had one unambiguous row" in str(context["reason"])


def test_market_direction_uses_inclusive_dominance_boundary_and_valid_comparisons(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    names = ["A", "B", "C", "D", "E", "Invalid"]
    _write_snapshot(snapshot_dir, "2026-09-01", [_row(name, value=10_000) for name in names])
    _write_snapshot(
        snapshot_dir,
        "2026-09-02",
        [
            _row("A", value=11_000),
            _row("B", value=11_000),
            _row("C", value=11_000),
            _row("D", value=9_000),
            _row("E", value=9_000),
            _row("Invalid", value=0),
        ],
    )
    rules = MarketChangeAlertRules(
        market_direction_dominance_pct=60,
        market_direction_min_comparable_items=5,
    )

    context = _market_value_context(run_market_change_alerts(snapshot_dir, rules=rules))

    assert context["compared_items"] == 5
    assert context["rose_percentage"] == pytest.approx(60)
    assert context["fell_percentage"] == pytest.approx(40)
    assert context["direction"] == "mostly_rising"


def test_market_direction_tie_is_mixed_and_small_cohort_is_insufficient(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    previous = [_row(name, value=10_000) for name in ("A", "B", "C", "D")]
    current = [
        _row("A", value=11_000),
        _row("B", value=11_000),
        _row("C", value=9_000),
        _row("D", value=9_000),
    ]
    _write_snapshot(snapshot_dir, "2026-09-01", previous)
    _write_snapshot(snapshot_dir, "2026-09-02", current)

    mixed_result = run_market_change_alerts(
        snapshot_dir,
        rules=MarketChangeAlertRules(
            minimum_item_snapshot_dates=2,
            value_change_pct_threshold=5,
            value_change_min_abs=500,
            market_direction_min_comparable_items=4,
        ),
    )
    insufficient_result = run_market_change_alerts(
        snapshot_dir,
        rules=MarketChangeAlertRules(
            minimum_item_snapshot_dates=2,
            value_change_pct_threshold=5,
            value_change_min_abs=500,
            market_direction_min_comparable_items=5,
        ),
    )
    mixed = _market_value_context(mixed_result)
    insufficient = _market_value_context(insufficient_result)

    assert mixed["rose_percentage"] == pytest.approx(50)
    assert mixed["fell_percentage"] == pytest.approx(50)
    assert mixed["direction"] == "mixed"
    assert insufficient["status"] == "insufficient_comparable_data"
    assert insufficient["direction"] == "insufficient_comparable_data"
    assert "Need at least 5 comparable items" in str(insufficient["reason"])
    assert _evaluation(mixed_result, "A", "value_movement")["broad_market_alignment"] == "mixed_or_insufficient"
    assert (
        _evaluation(insufficient_result, "A", "value_movement")["broad_market_alignment"]
        == "mixed_or_insufficient"
    )


def test_value_alert_alignment_preserves_all_alerts_and_other_alert_types(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    previous = [
        _row(name, value=10_000, demand_ratio=1, trade_count=250)
        for name in ("A", "B", "C", "D", "E")
    ]
    current = [
        _row("A", value=14_000, demand_ratio=3, trade_count=500),
        _row("B", value=14_000),
        _row("C", value=14_000),
        _row("D", value=6_000),
        _row("E", value=6_000),
    ]
    _write_snapshot(snapshot_dir, "2026-09-01", previous)
    _write_snapshot(snapshot_dir, "2026-09-02", previous)
    _write_snapshot(snapshot_dir, "2026-09-03", current)

    result = run_market_change_alerts(snapshot_dir)
    value_alerts = result.alerts[result.alerts["alert_type"] == "value_movement"]

    assert len(value_alerts) == 5
    assert (value_alerts["broad_market_alignment"] == "with_broad_direction").sum() == 3
    assert (value_alerts["broad_market_alignment"] == "against_broad_direction").sum() == 2
    demand = _evaluation(result, "A", "demand_change")
    activity = _evaluation(result, "A", "activity_change")
    assert bool(demand["fired"]) is True
    assert bool(activity["fired"]) is True
    assert demand["broad_market_alignment"] == "not_applicable"
    assert activity["broad_market_alignment"] == "not_applicable"


@pytest.mark.parametrize("dominance", [50, 100.1])
def test_invalid_market_direction_dominance_is_rejected(tmp_path: Path, dominance: float) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("A")])

    with pytest.raises(ValueError, match="market_direction_dominance_pct"):
        run_market_change_alerts(
            snapshot_dir,
            rules=MarketChangeAlertRules(market_direction_dominance_pct=dominance),
        )
