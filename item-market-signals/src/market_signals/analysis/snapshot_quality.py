"""Read-only quality preflight for dated gpovalues snapshots.

The report makes input gaps visible before downstream alert or historical
analysis. It never repairs rows, drops snapshots, or decides whether ordinary
source-value movement is suspicious.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path

import pandas as pd

from config.settings import SNAPSHOT_DIR
from market_signals.analysis.historical_signal_lab import (
    gpovalues_snapshot_date,
    snapshot_identity,
)

METRICS = ("value", "demand_ratio", "trade_count")
QUALITY_CATEGORIES = ("missing", "non_numeric", "negative", "zero", "extreme")

# These are the same defensive upper bounds used by Market Change Alerts.
# They identify values that need inspection; they are not market thresholds.
DEFAULT_METRIC_EXTREME_LIMITS: dict[str, float] = {
    "value": 100_000_000.0,
    "demand_ratio": 50.0,
    "trade_count": 1_000_000.0,
}


@dataclass(frozen=True)
class SnapshotQualityResult:
    """Overall metadata and one quality row per dated snapshot."""

    summary: dict[str, object]
    snapshots: pd.DataFrame


def run_snapshot_quality_preflight(
    snapshot_dir: Path = SNAPSHOT_DIR,
    *,
    extreme_limits: Mapping[str, float] | None = None,
) -> SnapshotQualityResult:
    """Inspect saved snapshots without calling the network or writing files."""
    limits = _validated_limits(extreme_limits)
    files = sorted(snapshot_dir.glob("gpovalues_*.csv"))
    if not files:
        raise FileNotFoundError(f"No gpovalues snapshots found in {snapshot_dir}.")

    records: list[dict[str, object]] = []
    previous_identities: set[str] | None = None
    for path in files:
        snapshot_date = gpovalues_snapshot_date(path)
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            records.append(_unreadable_snapshot_record(snapshot_date, path, exc))
            continue

        record, current_identities = _snapshot_quality_record(
            frame,
            snapshot_date,
            path,
            previous_identities=previous_identities,
            extreme_limits=limits,
        )
        records.append(record)
        previous_identities = current_identities

    snapshots = pd.DataFrame(records, columns=_result_columns())
    for column in ("added_identity_count", "removed_identity_count"):
        snapshots[column] = snapshots[column].astype("Int64")

    dates = [date.fromisoformat(value) for value in snapshots["snapshot_date"]]
    missing_dates = _missing_calendar_dates(dates)
    row_counts = pd.to_numeric(snapshots["row_count"], errors="coerce").dropna()
    metric_totals = {
        metric: {
            category: int(snapshots[f"{metric}_{category}_count"].sum())
            for category in QUALITY_CATEGORIES
        }
        for metric in METRICS
    }
    summary: dict[str, object] = {
        "snapshot_count": len(snapshots),
        "first_snapshot_date": min(dates).isoformat(),
        "last_snapshot_date": max(dates).isoformat(),
        "expected_calendar_date_count": (max(dates) - min(dates)).days + 1,
        "missing_calendar_dates": tuple(value.isoformat() for value in missing_dates),
        "missing_calendar_date_count": len(missing_dates),
        "minimum_row_count": int(row_counts.min()) if not row_counts.empty else None,
        "maximum_row_count": int(row_counts.max()) if not row_counts.empty else None,
        "quality_concern_snapshot_count": int(snapshots["has_quality_concerns"].sum()),
        "coverage_change_snapshot_count": int(snapshots["has_coverage_change"].sum()),
        "empty_snapshot_count": int(snapshots["is_empty"].sum()),
        "unreadable_snapshot_count": int(snapshots["read_error"].ne("").sum()),
        "missing_identity_row_count": int(snapshots["missing_identity_count"].sum()),
        "duplicate_identity_count": int(snapshots["duplicate_identity_count"].sum()),
        "duplicate_identity_row_count": int(snapshots["duplicate_identity_row_count"].sum()),
        "metric_totals": metric_totals,
        "metric_extreme_limits": dict(limits),
    }
    return SnapshotQualityResult(summary=summary, snapshots=snapshots)


def _snapshot_quality_record(
    frame: pd.DataFrame,
    snapshot_date: date,
    path: Path,
    *,
    previous_identities: set[str] | None,
    extreme_limits: Mapping[str, float],
) -> tuple[dict[str, object], set[str]]:
    identities = frame.apply(snapshot_identity, axis=1) if not frame.empty else pd.Series(dtype=str)
    identities = identities.fillna("").astype(str).str.strip()
    missing_identity_count = int(identities.eq("").sum())
    present_identities = identities[identities.ne("")]
    identity_counts = present_identities.value_counts()
    duplicate_counts = identity_counts[identity_counts > 1]
    current_identities = set(str(value) for value in identity_counts.index)

    if previous_identities is None:
        added_identity_count: int | None = None
        removed_identity_count: int | None = None
    else:
        added_identity_count = len(current_identities - previous_identities)
        removed_identity_count = len(previous_identities - current_identities)

    record: dict[str, object] = {
        "snapshot_date": snapshot_date.isoformat(),
        "snapshot_file": path.name,
        "row_count": len(frame),
        "unique_identity_count": len(current_identities),
        "missing_identity_count": missing_identity_count,
        "duplicate_identity_count": int(len(duplicate_counts)),
        "duplicate_identity_row_count": int(duplicate_counts.sum()),
        "added_identity_count": added_identity_count,
        "removed_identity_count": removed_identity_count,
        "is_empty": frame.empty,
        "read_error": "",
    }

    metric_issue_count = 0
    for metric in METRICS:
        counts = _metric_quality_counts(frame, metric, extreme_limits[metric])
        for category, count in counts.items():
            record[f"{metric}_{category}_count"] = count
        metric_issue_count += (
            counts["missing"]
            + counts["non_numeric"]
            + counts["negative"]
            + counts["extreme"]
        )
        if metric == "value":
            metric_issue_count += counts["zero"]

    identity_issue_count = missing_identity_count + int(duplicate_counts.sum())
    quality_issue_count = identity_issue_count + metric_issue_count + int(frame.empty)
    record["quality_issue_count"] = quality_issue_count
    record["has_quality_concerns"] = quality_issue_count > 0
    record["has_coverage_change"] = bool(
        previous_identities is not None
        and ((added_identity_count or 0) > 0 or (removed_identity_count or 0) > 0)
    )
    return record, current_identities


def _unreadable_snapshot_record(snapshot_date: date, path: Path, error: Exception) -> dict[str, object]:
    record: dict[str, object] = {
        "snapshot_date": snapshot_date.isoformat(),
        "snapshot_file": path.name,
        "row_count": pd.NA,
        "unique_identity_count": pd.NA,
        "missing_identity_count": pd.NA,
        "duplicate_identity_count": pd.NA,
        "duplicate_identity_row_count": pd.NA,
        "added_identity_count": pd.NA,
        "removed_identity_count": pd.NA,
        "is_empty": False,
        "read_error": "CSV parse error" if isinstance(error, pd.errors.ParserError) else "invalid text encoding",
        "quality_issue_count": 1,
        "has_quality_concerns": True,
        "has_coverage_change": False,
    }
    for metric in METRICS:
        for category in QUALITY_CATEGORIES:
            record[f"{metric}_{category}_count"] = pd.NA
    return record


def _metric_quality_counts(frame: pd.DataFrame, column: str, extreme_limit: float) -> dict[str, int]:
    if column not in frame.columns:
        raw = pd.Series(pd.NA, index=frame.index, dtype=object)
    else:
        raw = frame[column]

    missing = raw.isna() | raw.astype(str).str.strip().eq("")
    parsed = pd.to_numeric(raw.where(~missing), errors="coerce")
    non_numeric = ~missing & parsed.isna()
    numeric = ~missing & parsed.notna()
    return {
        "missing": int(missing.sum()),
        "non_numeric": int(non_numeric.sum()),
        "negative": int((numeric & (parsed < 0)).sum()),
        "zero": int((numeric & (parsed == 0)).sum()),
        "extreme": int((numeric & (parsed > extreme_limit)).sum()),
    }


def _validated_limits(extreme_limits: Mapping[str, float] | None) -> dict[str, float]:
    limits = dict(DEFAULT_METRIC_EXTREME_LIMITS if extreme_limits is None else extreme_limits)
    missing = [metric for metric in METRICS if metric not in limits]
    if missing:
        raise ValueError(f"Missing extreme limit(s): {', '.join(missing)}.")
    for metric in METRICS:
        value = float(limits[metric])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"Extreme limit for {metric} must be a positive finite number.")
        limits[metric] = value
    return limits


def _missing_calendar_dates(snapshot_dates: list[date]) -> list[date]:
    first = min(snapshot_dates)
    last = max(snapshot_dates)
    available = set(snapshot_dates)
    return [
        first + timedelta(days=offset)
        for offset in range((last - first).days + 1)
        if first + timedelta(days=offset) not in available
    ]


def _result_columns() -> list[str]:
    columns = [
        "snapshot_date",
        "snapshot_file",
        "row_count",
        "unique_identity_count",
        "missing_identity_count",
        "duplicate_identity_count",
        "duplicate_identity_row_count",
        "added_identity_count",
        "removed_identity_count",
        "is_empty",
        "read_error",
    ]
    for metric in METRICS:
        columns.extend(f"{metric}_{category}_count" for category in QUALITY_CATEGORIES)
    columns.extend(["quality_issue_count", "has_quality_concerns", "has_coverage_change"])
    return columns
