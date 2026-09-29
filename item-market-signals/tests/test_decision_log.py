from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from market_signals.decisions.cli import app
from market_signals.decisions.log import (
    AmbiguousItemMatchError,
    DECISION_COLUMNS,
    DecisionNotFoundError,
    DuplicateDecisionIDError,
    MissingItemPriceError,
    PurchasePriceForNoBuyError,
    ResaleAlreadyRecordedError,
    ResaleWithoutPurchaseError,
    append_decision,
    build_decision_entry,
    get_decision,
    load_decision_log,
    log_decision,
    marked_outcome_for_decision,
    record_resale_outcome,
    recompute_saved_verdict,
)


FIXED_TIME = datetime(2026, 9, 26, 12, 30, tzinfo=timezone.utc)
RUNNER = CliRunner()


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
        "demand_ratio": 1.2,
        "trade_count": 250,
        "derivation": "test",
        "share_url": "https://example.test/item",
        "image_url": "https://example.test/item.png",
        "generated_at": "2026-09-26T01:00:00+00:00",
        "n_trades_used": 12345,
        "join_key": name.lower().strip(),
    }
    return {**base, **overrides}


def test_logs_buy_entry_with_snapshot_provenance(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Prestige Candy Cane", shortcut="PCC")])

    entry = log_decision(
        "Prestige Candy Cane",
        80.0,
        True,
        purchase_price=78.0,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=FIXED_TIME,
    )
    saved = load_decision_log(log_path).iloc[0]

    assert entry["verdict"] == "GOOD DEAL -- asking price is below the typical trade range"
    assert saved["snapshot_date"] == "2026-09-26"
    assert saved["snapshot_file"] == "gpovalues_2026-09-26.csv"
    assert saved["snapshot_generated_at"] == "2026-09-26T01:00:00+00:00"
    assert saved["item_slug"] == "prestige-candy-cane"
    assert bool(saved["bought"]) is True
    assert saved["purchase_price"] == pytest.approx(78.0)


def test_logs_no_buy_entry_without_purchase_price(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane")])

    log_decision(
        "Candy Cane",
        125.0,
        False,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=FIXED_TIME,
    )
    saved = load_decision_log(log_path).iloc[0]

    assert saved["verdict"] == "OVERPRICED -- asking price is above the typical trade range"
    assert bool(saved["bought"]) is False
    assert pd.isna(saved["purchase_price"])
    assert pd.isna(saved["resale_price"])


def test_rejects_purchase_price_for_no_buy_in_core_function(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane")])

    with pytest.raises(PurchasePriceForNoBuyError):
        build_decision_entry(
            "Candy Cane",
            125.0,
            False,
            purchase_price=100.0,
            snapshot_dir=snapshot_dir,
            decision_timestamp=FIXED_TIME,
        )


def test_cli_rejects_purchase_price_for_no_buy(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    snapshot_path = _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane")])

    result = RUNNER.invoke(
        app,
        [
            "log",
            "Candy Cane",
            "--asking-price",
            "125",
            "--decision",
            "no-buy",
            "--purchase-price",
            "100",
            "--snapshot-file",
            str(snapshot_path),
            "--log-path",
            str(log_path),
        ],
    )

    assert result.exit_code == 1
    assert "purchase_price is only valid for a buy decision" in result.output
    assert not log_path.exists()


def test_ambiguous_item_matches_are_rejected(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        "2026-09-26",
        [_row("Candy Cane"), _row("Prestige Candy Cane", shortcut="PCC")],
    )

    with pytest.raises(AmbiguousItemMatchError, match="Multiple matches"):
        build_decision_entry("cane", 100.0, False, snapshot_dir=snapshot_dir)


def test_duplicate_exact_item_matches_are_rejected(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(
        snapshot_dir,
        "2026-09-26",
        [
            _row("Kraken Blade", slug="kraken-blade-a", shortcut="KB-A"),
            _row("Kraken Blade", slug="kraken-blade-b", shortcut="KB-B"),
        ],
    )

    with pytest.raises(AmbiguousItemMatchError, match="Multiple exact matches"):
        build_decision_entry("Kraken Blade", 100.0, False, snapshot_dir=snapshot_dir)


def test_missing_price_fields_are_rejected(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Mystery Item", value=None)])

    with pytest.raises(MissingItemPriceError, match="value"):
        build_decision_entry("Mystery Item", 100.0, False, snapshot_dir=snapshot_dir)


def test_saved_verdict_reproduces_without_newer_snapshots(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    old_snapshot = _write_snapshot(snapshot_dir, "2026-09-10", [_row("Candy Cane", value=100, ci_low=90, ci_high=110)])
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane", value=1000, ci_low=900, ci_high=1100)])

    entry = build_decision_entry(
        "Candy Cane",
        95.0,
        False,
        snapshot_path=old_snapshot,
        snapshot_dir=snapshot_dir,
        decision_timestamp=FIXED_TIME,
    )

    assert entry["snapshot_file"] == "gpovalues_2026-09-10.csv"
    assert entry["verdict"] == "FAIR -- asking price is at or below the solved fair value"
    assert recompute_saved_verdict(entry) == entry["verdict"]


def test_records_actual_resale_without_marked_value_fields(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane")])
    decision_time = datetime.combine(date.today(), datetime.min.time(), timezone.utc)
    entry = log_decision(
        "Candy Cane",
        80.0,
        True,
        purchase_price=80.0,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=decision_time,
    )

    valid_resale_date = date.today().isoformat()
    updated = record_resale_outcome(
        str(entry["decision_id"]),
        120.0,
        resale_date=valid_resale_date,
        log_path=log_path,
    )

    assert updated["resale_date"] == valid_resale_date
    assert updated["resale_price"] == pytest.approx(120.0)
    assert pd.isna(updated["marked_value"])
    assert pd.isna(updated["marked_value_source"])
    with pytest.raises(ResaleAlreadyRecordedError):
        record_resale_outcome(str(entry["decision_id"]), 130.0, log_path=log_path)


def test_rejects_invalid_resale_dates(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane")])
    decision_time = datetime.combine(date.today(), datetime.min.time(), timezone.utc)
    entry = log_decision(
        "Candy Cane",
        80.0,
        True,
        purchase_price=80.0,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=decision_time,
    )

    invalid_cases = [
        ("not-a-date", "valid ISO date"),
        ((decision_time.date() - timedelta(days=1)).isoformat(), "before the original decision date"),
        ((date.today() + timedelta(days=1)).isoformat(), "in the future"),
    ]
    for resale_date, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            record_resale_outcome(str(entry["decision_id"]), 120.0, resale_date=resale_date, log_path=log_path)


def test_rejects_resale_for_no_buy_decision(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    _write_snapshot(snapshot_dir, "2026-09-26", [_row("Candy Cane")])
    entry = log_decision(
        "Candy Cane",
        125.0,
        False,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=FIXED_TIME,
    )

    with pytest.raises(ResaleWithoutPurchaseError):
        record_resale_outcome(str(entry["decision_id"]), 120.0, log_path=log_path)


def test_duplicate_decision_id_cannot_be_appended(tmp_path: Path) -> None:
    log_path = tmp_path / "decision_log.csv"
    entry = {column: "" for column in DECISION_COLUMNS}
    entry.update(
        {
            "decision_id": "duplicate-id",
            "decision_timestamp": FIXED_TIME.isoformat(),
            "asking_price": 95.0,
            "published_value": 100.0,
            "ci_low": 90.0,
            "ci_high": 110.0,
            "verdict": "FAIR -- asking price is at or below the solved fair value",
            "bought": True,
        }
    )

    append_decision(entry, log_path)
    with pytest.raises(DuplicateDecisionIDError, match="already exists"):
        append_decision(entry, log_path)


def test_lookup_distinguishes_missing_and_duplicated_decision_ids(tmp_path: Path) -> None:
    log_path = tmp_path / "decision_log.csv"
    rows = []
    for _ in range(2):
        row = {column: "" for column in DECISION_COLUMNS}
        row.update(
            {
                "decision_id": "duplicated",
                "decision_timestamp": FIXED_TIME.isoformat(),
                "asking_price": 95.0,
                "published_value": 100.0,
                "ci_low": 90.0,
                "ci_high": 110.0,
                "verdict": "FAIR -- asking price is at or below the solved fair value",
                "bought": True,
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=DECISION_COLUMNS).to_csv(log_path, index=False)

    with pytest.raises(DuplicateDecisionIDError, match="appears 2 times"):
        get_decision("duplicated", log_path)
    with pytest.raises(DecisionNotFoundError, match="No decision found"):
        get_decision("missing", log_path)


def test_marked_outcome_is_pending_before_target_date(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    entry = log_decision(
        "Candy Cane",
        95.0,
        False,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )

    outcome = marked_outcome_for_decision(
        str(entry["decision_id"]),
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        as_of=date(2026, 9, 14),
    )

    assert outcome.status == "pending"
    assert outcome.target_date == date(2026, 9, 15)
    assert outcome.later_published_value is None


def test_marked_outcome_uses_exact_14_day_snapshot_and_does_not_write_log(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    original_snapshot = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-15", [_row("Candy Cane", value=125)])
    entry = log_decision(
        "Candy Cane",
        95.0,
        True,
        purchase_price=90.0,
        snapshot_path=original_snapshot,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )
    record_resale_outcome(
        str(entry["decision_id"]),
        130.0,
        resale_date="2026-09-10",
        log_path=log_path,
    )
    before = log_path.read_text()

    outcome = marked_outcome_for_decision(
        str(entry["decision_id"]),
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        as_of=date(2026, 9, 29),
    )

    assert log_path.read_text() == before
    assert outcome.status == "ready"
    assert outcome.original_published_value == pytest.approx(100)
    assert outcome.later_published_value == pytest.approx(125)
    assert outcome.percentage_change == pytest.approx(25)
    assert outcome.actual_snapshot_date == date(2026, 9, 15)
    assert outcome.days_elapsed == 14
    assert outcome.actual_resale_price == pytest.approx(130)


def test_cli_marked_outcome_labels_gpovalues_estimate_and_actual_resale_separately(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    original_snapshot = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-15", [_row("Candy Cane", value=80)])
    entry = log_decision(
        "Candy Cane",
        95.0,
        True,
        purchase_price=90.0,
        snapshot_path=original_snapshot,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )
    record_resale_outcome(
        str(entry["decision_id"]),
        110.0,
        resale_date="2026-09-10",
        log_path=log_path,
    )

    result = RUNNER.invoke(
        app,
        [
            "marked-outcome",
            str(entry["decision_id"]),
            "--log-path",
            str(log_path),
            "--snapshot-dir",
            str(snapshot_dir),
        ],
    )

    assert result.exit_code == 0
    assert "14-day change in gpovalues' published estimate" in result.output
    assert "not profit and not a completed trade" in result.output
    assert "Estimate change" in result.output
    assert "-20.0%" in result.output
    assert "Actual resale" in result.output
    assert "110 on 2026-09-10" in result.output


def test_marked_outcome_uses_first_delayed_snapshot_with_same_identity(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    original_snapshot = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-15", [_row("Different Item", value=999)])
    _write_snapshot(snapshot_dir, "2026-09-17", [_row("Candy Cane", value=110)])
    _write_snapshot(snapshot_dir, "2026-09-18", [_row("Candy Cane", value=130)])
    entry = log_decision(
        "Candy Cane",
        95.0,
        False,
        snapshot_path=original_snapshot,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )

    outcome = marked_outcome_for_decision(
        str(entry["decision_id"]),
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        as_of=date(2026, 9, 29),
    )

    assert outcome.status == "ready"
    assert outcome.later_published_value == pytest.approx(110)
    assert outcome.actual_snapshot_date == date(2026, 9, 17)
    assert outcome.days_elapsed == 16


def test_marked_outcome_reports_no_later_snapshot_when_item_is_missing(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    original_snapshot = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-15", [_row("Different Item", value=999)])
    entry = log_decision(
        "Candy Cane",
        95.0,
        False,
        snapshot_path=original_snapshot,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )

    outcome = marked_outcome_for_decision(
        str(entry["decision_id"]),
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        as_of=date(2026, 9, 29),
    )

    assert outcome.status == "no later snapshot"
    assert outcome.later_published_value is None


def test_marked_outcome_rejects_duplicate_later_item_rows_as_unsuitable(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    original_snapshot = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(
        snapshot_dir,
        "2026-09-15",
        [
            _row("Candy Cane", slug="candy-cane", value=110),
            _row("Candy Cane", slug="candy-cane", value=120),
        ],
    )
    entry = log_decision(
        "Candy Cane",
        95.0,
        False,
        snapshot_path=original_snapshot,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )

    outcome = marked_outcome_for_decision(
        str(entry["decision_id"]),
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        as_of=date(2026, 9, 29),
    )

    assert outcome.status == "no later snapshot"
    assert outcome.reason == "no later snapshot contains one matching item row"


@pytest.mark.parametrize("original_value", [0, None])
def test_marked_outcome_reports_zero_or_missing_original_values(tmp_path: Path, original_value: object) -> None:
    snapshot_dir = tmp_path / "snapshots"
    log_path = tmp_path / "decision_log.csv"
    original_snapshot = _write_snapshot(snapshot_dir, "2026-09-01", [_row("Candy Cane", value=100)])
    _write_snapshot(snapshot_dir, "2026-09-15", [_row("Candy Cane", value=125)])
    entry = log_decision(
        "Candy Cane",
        95.0,
        False,
        snapshot_path=original_snapshot,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        decision_timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    )
    log = load_decision_log(log_path)
    log.loc[log["decision_id"] == entry["decision_id"], "published_value"] = original_value
    log.to_csv(log_path, index=False)

    outcome = marked_outcome_for_decision(
        str(entry["decision_id"]),
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        as_of=date(2026, 9, 29),
    )

    assert outcome.status == "no original value"
    assert outcome.percentage_change is None
