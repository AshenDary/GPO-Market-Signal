"""Daily market-change alert candidates from dated gpovalues snapshots.

Alerts are review leads, not trading recommendations. The rules deliberately
use simple threshold checks against the immediately previous available snapshot
so a user can see exactly why an item did or did not fire.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from config.settings import SNAPSHOT_DIR
from market_signals.analysis.historical_signal_lab import (
    SnapshotFrame,
    load_gpovalues_snapshots,
)


@dataclass(frozen=True)
class MarketChangeAlertRules:
    """Central rule settings for current-vs-previous snapshot alerts."""

    minimum_item_snapshot_dates: int = 3
    stale_snapshot_days: int = 2
    value_change_pct_threshold: float = 25.0
    value_change_min_abs: float = 1_000.0
    demand_ratio_abs_change_threshold: float = 1.0
    activity_change_pct_threshold: float = 50.0
    activity_change_min_abs: float = 100.0
    max_reasonable_value: float = 100_000_000.0
    max_reasonable_demand_ratio: float = 50.0
    max_reasonable_trade_count: float = 1_000_000.0


DEFAULT_ALERT_RULES = MarketChangeAlertRules()


@dataclass(frozen=True)
class MarketChangeAlertResult:
    """Alert outputs for dashboard rendering and offline tests."""

    evaluations: pd.DataFrame
    alerts: pd.DataFrame
    item_history: pd.DataFrame
    rules: MarketChangeAlertRules
    metadata: dict[str, object]


ALERT_TYPES: tuple[str, ...] = ("value_movement", "demand_change", "activity_change")


def run_market_change_alerts(
    snapshot_dir: Path = SNAPSHOT_DIR,
    *,
    rules: MarketChangeAlertRules = DEFAULT_ALERT_RULES,
    as_of: date | None = None,
) -> MarketChangeAlertResult:
    """Evaluate latest gpovalues rows against the previous dated snapshot."""
    _validate_rules(rules)
    snapshots = _prepare_snapshots(load_gpovalues_snapshots(snapshot_dir))
    snapshot_dates = [snapshot.snapshot_date for snapshot in snapshots]
    latest_snapshot = snapshots[-1]
    comparison_snapshot = snapshots[-2] if len(snapshots) >= 2 else None
    effective_as_of = as_of or date.today()

    history = _build_item_history(snapshots)
    latest = latest_snapshot.frame.copy()
    records: list[dict[str, object]] = []

    for _, latest_row in latest.iterrows():
        identity = _string_or_blank(latest_row.get("item_identity"))
        latest_status = _latest_item_status(latest_row)
        history_count = _unambiguous_history_count(history, identity)
        comparison_status, comparison_row = _comparison_match(comparison_snapshot, identity)

        for alert_type in ALERT_TYPES:
            records.append(
                _evaluate_alert_type(
                    alert_type,
                    latest_row,
                    comparison_row,
                    latest_snapshot,
                    comparison_snapshot,
                    latest_status=latest_status,
                    comparison_status=comparison_status,
                    history_count=history_count,
                    rules=rules,
                )
            )

    evaluations = pd.DataFrame(records, columns=_evaluation_columns())
    alerts = evaluations[evaluations["fired"]].copy().reset_index(drop=True)
    metadata = _metadata(snapshots, evaluations, rules, effective_as_of)

    return MarketChangeAlertResult(
        evaluations=evaluations.reset_index(drop=True),
        alerts=alerts,
        item_history=history,
        rules=rules,
        metadata=metadata,
    )


def _prepare_snapshots(snapshots: list[SnapshotFrame]) -> list[SnapshotFrame]:
    prepared: list[SnapshotFrame] = []
    for snapshot in snapshots:
        frame = snapshot.frame.copy()
        if "trade_count" in frame.columns:
            frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce")
        else:
            frame["trade_count"] = pd.NA
        prepared.append(SnapshotFrame(snapshot.snapshot_date, snapshot.path, frame))
    return prepared


def _build_item_history(snapshots: list[SnapshotFrame]) -> pd.DataFrame:
    frames = []
    for snapshot in snapshots:
        frame = snapshot.frame.copy()
        frame["snapshot_date"] = snapshot.snapshot_date.isoformat()
        frames.append(frame)
    if not frames:
        return _empty_history()
    history = pd.concat(frames, ignore_index=True)
    for column in ("value", "demand_ratio", "trade_count"):
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
        else:
            history[column] = pd.NA
    return history


def _latest_item_status(row: pd.Series) -> str:
    if _is_blank(row.get("item_identity")):
        return "missing_item_identity"
    if int(row.get("identity_count", 0)) != 1:
        return "duplicate_current_identity"
    return "ready"


def _comparison_match(
    comparison_snapshot: SnapshotFrame | None,
    identity: str,
) -> tuple[str, pd.Series | None]:
    if comparison_snapshot is None:
        return "insufficient_snapshot_dates", None
    if _is_blank(identity):
        return "missing_item_identity", None
    matches = comparison_snapshot.frame[comparison_snapshot.frame["item_identity"] == identity]
    if matches.empty:
        return "missing_comparison_item", None
    if len(matches) > 1:
        return "duplicate_comparison_identity", None
    return "ready", matches.iloc[0]


def _unambiguous_history_count(history: pd.DataFrame, identity: str) -> int:
    if _is_blank(identity) or history.empty:
        return 0
    rows = history[history["item_identity"] == identity]
    if rows.empty:
        return 0
    per_date = rows.groupby("snapshot_date")["item_identity"].size()
    return int((per_date == 1).sum())


def _evaluate_alert_type(
    alert_type: str,
    latest_row: pd.Series,
    comparison_row: pd.Series | None,
    latest_snapshot: SnapshotFrame,
    comparison_snapshot: SnapshotFrame | None,
    *,
    latest_status: str,
    comparison_status: str,
    history_count: int,
    rules: MarketChangeAlertRules,
) -> dict[str, object]:
    base = _base_record(
        alert_type,
        latest_row,
        comparison_row,
        latest_snapshot,
        comparison_snapshot,
        history_count,
        rules,
    )
    if latest_status != "ready":
        return {**base, "status": latest_status, "reason": _guard_reason(latest_status, base)}
    if comparison_status != "ready":
        return {**base, "status": comparison_status, "reason": _guard_reason(comparison_status, base)}

    assert comparison_row is not None
    status, reason, current, comparison, abs_change, pct_change, fired = _metric_evaluation(
        alert_type,
        latest_row,
        comparison_row,
        rules,
    )
    metric_values = {
        "current_value": current,
        "comparison_value": comparison,
        "absolute_change": abs_change,
        "percent_change": pct_change,
        "direction": _direction(abs_change),
    }
    if status not in {"alert", "no_alert"}:
        return {
            **base,
            **metric_values,
            "status": status,
            "reason": reason,
        }
    if history_count < rules.minimum_item_snapshot_dates:
        return {
            **base,
            **metric_values,
            "status": "insufficient_history",
            "reason": (
                "Need at least "
                f"{rules.minimum_item_snapshot_dates} unambiguous snapshot dates for this item; "
                f"found {history_count}."
            ),
        }
    return {
        **base,
        **metric_values,
        "status": status,
        "fired": fired,
        "reason": reason,
    }


def _base_record(
    alert_type: str,
    latest_row: pd.Series,
    comparison_row: pd.Series | None,
    latest_snapshot: SnapshotFrame,
    comparison_snapshot: SnapshotFrame | None,
    history_count: int,
    rules: MarketChangeAlertRules,
) -> dict[str, object]:
    return {
        "alert_type": alert_type,
        "fired": False,
        "status": "not_evaluated",
        "item_identity": _string_or_blank(latest_row.get("item_identity")),
        "identity_source": _string_or_blank(latest_row.get("identity_source")),
        "item_name": _string_or_blank(latest_row.get("name")),
        "item_slug": _string_or_blank(latest_row.get("slug")),
        "item_shortcut": _string_or_blank(latest_row.get("shortcut")),
        "current_date": latest_snapshot.snapshot_date.isoformat(),
        "comparison_date": comparison_snapshot.snapshot_date.isoformat() if comparison_snapshot else None,
        "days_between_snapshots": (
            (latest_snapshot.snapshot_date - comparison_snapshot.snapshot_date).days
            if comparison_snapshot
            else None
        ),
        "item_snapshot_dates": history_count,
        "required_item_snapshot_dates": rules.minimum_item_snapshot_dates,
        "current_value": None,
        "comparison_value": None,
        "absolute_change": None,
        "percent_change": None,
        "direction": "missing",
        "reason": "",
    }


def _metric_evaluation(
    alert_type: str,
    latest_row: pd.Series,
    comparison_row: pd.Series,
    rules: MarketChangeAlertRules,
) -> tuple[str, str, float | None, float | None, float | None, float | None, bool]:
    if alert_type == "value_movement":
        return _evaluate_value_movement(latest_row, comparison_row, rules)
    if alert_type == "demand_change":
        return _evaluate_demand_change(latest_row, comparison_row, rules)
    if alert_type == "activity_change":
        return _evaluate_activity_change(latest_row, comparison_row, rules)
    raise ValueError(f"Unknown alert type: {alert_type}")


def _evaluate_value_movement(
    latest_row: pd.Series,
    comparison_row: pd.Series,
    rules: MarketChangeAlertRules,
) -> tuple[str, str, float | None, float | None, float | None, float | None, bool]:
    current_status, current = _valid_positive_number(
        latest_row.get("value"), max_value=rules.max_reasonable_value
    )
    comparison_status, comparison = _valid_positive_number(
        comparison_row.get("value"), max_value=rules.max_reasonable_value
    )
    invalid = _invalid_metric_reason("published value", current_status, comparison_status)
    if invalid is not None:
        status, reason = invalid
        return status, reason, current, comparison, None, None, False

    assert current is not None and comparison is not None
    abs_change = current - comparison
    pct_change = (abs_change / comparison) * 100
    fired = abs(pct_change) >= rules.value_change_pct_threshold and abs(abs_change) >= rules.value_change_min_abs
    if fired:
        reason = (
            f"Published value {_direction_word(abs_change)} {pct_change:+.2f}% "
            f"({abs_change:+,.0f}) versus the comparison snapshot; rule requires "
            f"at least {rules.value_change_pct_threshold:.1f}% and "
            f"{rules.value_change_min_abs:,.0f} value units."
        )
        return "alert", reason, current, comparison, abs_change, pct_change, True

    reason = (
        f"Published value changed {pct_change:+.2f}% ({abs_change:+,.0f}), below the "
        f"{rules.value_change_pct_threshold:.1f}% and {rules.value_change_min_abs:,.0f} unit rule."
    )
    return "no_alert", reason, current, comparison, abs_change, pct_change, False


def _evaluate_demand_change(
    latest_row: pd.Series,
    comparison_row: pd.Series,
    rules: MarketChangeAlertRules,
) -> tuple[str, str, float | None, float | None, float | None, float | None, bool]:
    current_status, current = _valid_nonnegative_number(
        latest_row.get("demand_ratio"), max_value=rules.max_reasonable_demand_ratio
    )
    comparison_status, comparison = _valid_nonnegative_number(
        comparison_row.get("demand_ratio"), max_value=rules.max_reasonable_demand_ratio
    )
    invalid = _invalid_metric_reason("demand ratio", current_status, comparison_status)
    if invalid is not None:
        status, reason = invalid
        return status, reason, current, comparison, None, None, False

    assert current is not None and comparison is not None
    abs_change = current - comparison
    fired = abs(abs_change) >= rules.demand_ratio_abs_change_threshold
    if fired:
        reason = (
            f"Demand ratio {_direction_word(abs_change)} {abs_change:+.2f} points versus the "
            "comparison snapshot; rule requires at least "
            f"{rules.demand_ratio_abs_change_threshold:.2f} point."
        )
        return "alert", reason, current, comparison, abs_change, None, True

    reason = (
        f"Demand ratio changed {abs_change:+.2f} points, below the "
        f"{rules.demand_ratio_abs_change_threshold:.2f} point rule."
    )
    return "no_alert", reason, current, comparison, abs_change, None, False


def _evaluate_activity_change(
    latest_row: pd.Series,
    comparison_row: pd.Series,
    rules: MarketChangeAlertRules,
) -> tuple[str, str, float | None, float | None, float | None, float | None, bool]:
    current_status, current = _valid_nonnegative_number(
        latest_row.get("trade_count"), max_value=rules.max_reasonable_trade_count
    )
    comparison_status, comparison = _valid_nonnegative_number(
        comparison_row.get("trade_count"), max_value=rules.max_reasonable_trade_count
    )
    invalid = _invalid_metric_reason("trade_count activity signal", current_status, comparison_status)
    if invalid is not None:
        status, reason = invalid
        return status, reason, current, comparison, None, None, False

    assert current is not None and comparison is not None
    abs_change = current - comparison
    pct_change = None if comparison == 0 else (abs_change / comparison) * 100
    pct_passes = comparison == 0 or (pct_change is not None and abs(pct_change) >= rules.activity_change_pct_threshold)
    fired = abs(abs_change) >= rules.activity_change_min_abs and pct_passes
    if fired:
        pct_text = "from zero" if pct_change is None else f"{pct_change:+.2f}%"
        reason = (
            f"trade_count activity signal {_direction_word(abs_change)} {pct_text} "
            f"({abs_change:+,.0f}) versus the comparison snapshot; rule requires "
            f"at least {rules.activity_change_pct_threshold:.1f}% and "
            f"{rules.activity_change_min_abs:,.0f} count change. This is an activity "
            "indicator, not new trades today."
        )
        return "alert", reason, current, comparison, abs_change, pct_change, True

    pct_text = "from zero" if pct_change is None else f"{pct_change:+.2f}%"
    reason = (
        f"trade_count activity signal changed {pct_text} ({abs_change:+,.0f}), below the "
        f"{rules.activity_change_pct_threshold:.1f}% and {rules.activity_change_min_abs:,.0f} count rule. "
        "This is an activity indicator, not new trades today."
    )
    return "no_alert", reason, current, comparison, abs_change, pct_change, False


def _invalid_metric_reason(
    label: str,
    current_status: str,
    comparison_status: str,
) -> tuple[str, str] | None:
    if current_status != "ready":
        status = f"{current_status}_current"
        return status, f"Current {label} is {current_status.replace('_', ' ')}."
    if comparison_status != "ready":
        status = f"{comparison_status}_comparison"
        return status, f"Comparison {label} is {comparison_status.replace('_', ' ')}."
    return None


def _valid_positive_number(value: object, *, max_value: float) -> tuple[str, float | None]:
    status, parsed = _valid_nonnegative_number(value, max_value=max_value)
    if status != "ready":
        return status, parsed
    if parsed is None or parsed <= 0:
        return "zero_or_negative", parsed
    return "ready", parsed


def _valid_nonnegative_number(value: object, *, max_value: float) -> tuple[str, float | None]:
    parsed = _optional_float(value)
    if parsed is None:
        return "missing", None
    if parsed < 0:
        return "negative", parsed
    if parsed > max_value:
        return "extreme", parsed
    return "ready", parsed


def _metadata(
    snapshots: list[SnapshotFrame],
    evaluations: pd.DataFrame,
    rules: MarketChangeAlertRules,
    as_of: date,
) -> dict[str, object]:
    latest_date = snapshots[-1].snapshot_date
    days_since_latest = (as_of - latest_date).days
    fired = evaluations[evaluations["fired"]]
    return {
        "snapshot_count": len(snapshots),
        "first_snapshot_date": snapshots[0].snapshot_date.isoformat(),
        "latest_snapshot_date": latest_date.isoformat(),
        "comparison_snapshot_date": snapshots[-2].snapshot_date.isoformat() if len(snapshots) >= 2 else None,
        "as_of_date": as_of.isoformat(),
        "days_since_latest_snapshot": days_since_latest,
        "is_stale": days_since_latest > rules.stale_snapshot_days,
        "latest_item_rows": int(len(snapshots[-1].frame)),
        "latest_unique_items": int(snapshots[-1].frame["item_identity"].nunique()),
        "evaluation_count": int(len(evaluations)),
        "alert_count": int(len(fired)),
        "value_alert_count": int((fired["alert_type"] == "value_movement").sum()),
        "demand_alert_count": int((fired["alert_type"] == "demand_change").sum()),
        "activity_alert_count": int((fired["alert_type"] == "activity_change").sum()),
        "guard_counts": evaluations["status"].value_counts().to_dict(),
        "market_value_context": _market_value_context(snapshots, rules),
        "rules": asdict(rules),
    }


def _market_value_context(
    snapshots: list[SnapshotFrame],
    rules: MarketChangeAlertRules,
) -> dict[str, object]:
    latest_snapshot = snapshots[-1]
    comparison_snapshot = snapshots[-2] if len(snapshots) >= 2 else None
    base: dict[str, object] = {
        "current_date": latest_snapshot.snapshot_date.isoformat(),
        "comparison_date": comparison_snapshot.snapshot_date.isoformat() if comparison_snapshot else None,
        "compared_items": 0,
        "median_percent_change": None,
        "rose_count": 0,
        "fell_count": 0,
        "flat_count": 0,
        "status": "insufficient_comparable_data",
        "reason": "",
    }
    if comparison_snapshot is None:
        return {
            **base,
            "reason": "Need at least two dated gpovalues snapshots before summarizing market-wide value movement.",
        }

    changes: list[float] = []
    rose_count = 0
    fell_count = 0
    flat_count = 0
    for _, latest_row in latest_snapshot.frame.iterrows():
        identity = _string_or_blank(latest_row.get("item_identity"))
        if _latest_item_status(latest_row) != "ready":
            continue
        comparison_status, comparison_row = _comparison_match(comparison_snapshot, identity)
        if comparison_status != "ready" or comparison_row is None:
            continue

        current_status, current = _valid_positive_number(
            latest_row.get("value"), max_value=rules.max_reasonable_value
        )
        comparison_value_status, comparison = _valid_positive_number(
            comparison_row.get("value"), max_value=rules.max_reasonable_value
        )
        if current_status != "ready" or comparison_value_status != "ready":
            continue
        assert current is not None and comparison is not None

        abs_change = current - comparison
        changes.append((abs_change / comparison) * 100)
        if abs_change > 0:
            rose_count += 1
        elif abs_change < 0:
            fell_count += 1
        else:
            flat_count += 1

    if not changes:
        return {
            **base,
            "reason": (
                "No item had one unambiguous row in both snapshots with valid positive "
                "published values."
            ),
        }

    return {
        **base,
        "compared_items": len(changes),
        "median_percent_change": round(float(pd.Series(changes).median()), 4),
        "rose_count": rose_count,
        "fell_count": fell_count,
        "flat_count": flat_count,
        "status": "ready",
        "reason": (
            "Compared items with one unambiguous row in both snapshots and valid positive "
            "published values."
        ),
    }


def _guard_reason(status: str, base: dict[str, object]) -> str:
    if status == "insufficient_snapshot_dates":
        return "Need at least two dated gpovalues snapshots before comparing current and prior values."
    if status == "missing_item_identity":
        return "Latest row has no usable identity; expected slug, join_key, or exact item name."
    if status == "duplicate_current_identity":
        return "Latest snapshot has duplicate rows for this item identity, so no alert is fired."
    if status == "missing_comparison_item":
        return (
            "Item is in the latest snapshot but not in the comparison snapshot "
            f"{base.get('comparison_date')}; no prior value can be compared."
        )
    if status == "duplicate_comparison_identity":
        return (
            "Comparison snapshot has duplicate rows for this item identity, so no alert is fired."
        )
    return status.replace("_", " ")


def _validate_rules(rules: MarketChangeAlertRules) -> None:
    if rules.minimum_item_snapshot_dates < 2:
        raise ValueError("minimum_item_snapshot_dates must be at least 2.")
    if rules.stale_snapshot_days < 0:
        raise ValueError("stale_snapshot_days cannot be negative.")
    for name, value in asdict(rules).items():
        if name in {"minimum_item_snapshot_dates", "stale_snapshot_days"}:
            continue
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")


def _direction(value: object) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "missing"
    if parsed > 0:
        return "up"
    if parsed < 0:
        return "down"
    return "flat"


def _direction_word(value: float) -> str:
    if value > 0:
        return "increased"
    if value < 0:
        return "decreased"
    return "stayed flat"


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _string_or_blank(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _evaluation_columns() -> list[str]:
    return [
        "alert_type",
        "fired",
        "status",
        "item_identity",
        "identity_source",
        "item_name",
        "item_slug",
        "item_shortcut",
        "current_date",
        "comparison_date",
        "days_between_snapshots",
        "item_snapshot_dates",
        "required_item_snapshot_dates",
        "current_value",
        "comparison_value",
        "absolute_change",
        "percent_change",
        "direction",
        "reason",
    ]


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "snapshot_date",
            "item_identity",
            "identity_source",
            "name",
            "slug",
            "shortcut",
            "value",
            "demand_ratio",
            "trade_count",
            "identity_count",
        ]
    )
