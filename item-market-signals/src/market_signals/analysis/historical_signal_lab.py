"""Historical signal evaluation from dated gpovalues snapshots.

This module intentionally uses raw gpovalues snapshots only. It does not read
the private decision log and does not reconstruct past state from the latest
merged feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config.settings import SNAPSHOT_DIR

PRIOR_SIGNAL_DAYS = 7
OUTCOME_DAYS = 14
MAX_OUTCOME_DELAY_DAYS = 3


@dataclass(frozen=True)
class SnapshotFrame:
    """One dated gpovalues snapshot with normalized identity columns."""

    snapshot_date: date
    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class HistoricalSignalResult:
    """Analysis outputs for dashboard and tests."""

    observations: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, object]


def run_historical_signal_lab(
    snapshot_dir: Path = SNAPSHOT_DIR,
    *,
    prior_signal_days: int = PRIOR_SIGNAL_DAYS,
    outcome_days: int = OUTCOME_DAYS,
    max_outcome_delay_days: int = MAX_OUTCOME_DELAY_DAYS,
) -> HistoricalSignalResult:
    """Evaluate simple historical signals against later published estimates."""
    if prior_signal_days <= 0:
        raise ValueError("prior_signal_days must be greater than zero.")
    if outcome_days <= 0:
        raise ValueError("outcome_days must be greater than zero.")
    if max_outcome_delay_days < 0:
        raise ValueError("max_outcome_delay_days cannot be negative.")

    snapshots = load_gpovalues_snapshots(snapshot_dir)
    observations = build_signal_observations(
        snapshots,
        prior_signal_days=prior_signal_days,
        outcome_days=outcome_days,
        max_outcome_delay_days=max_outcome_delay_days,
    )
    summary = summarize_signal_groups(observations)
    snapshot_dates = [snap.snapshot_date for snap in snapshots]
    metadata = {
        "snapshot_count": len(snapshots),
        "first_snapshot_date": min(snapshot_dates).isoformat(),
        "last_snapshot_date": max(snapshot_dates).isoformat(),
        "prior_signal_days": prior_signal_days,
        "outcome_days": outcome_days,
        "max_outcome_delay_days": max_outcome_delay_days,
        "observation_count": int(len(observations)),
        "eligible_observation_count": int((observations["evaluation_status"] == "eligible").sum()),
        "ready_outcome_count": int((observations["outcome_status"] == "ready").sum()),
        "missing_outcome_count": int((observations["outcome_status"] != "ready").sum()),
    }
    return HistoricalSignalResult(observations=observations, summary=summary, metadata=metadata)


def load_gpovalues_snapshots(snapshot_dir: Path = SNAPSHOT_DIR) -> list[SnapshotFrame]:
    """Load dated gpovalues snapshots sorted by filename date."""
    files = sorted(snapshot_dir.glob("gpovalues_*.csv"))
    if not files:
        raise FileNotFoundError(f"No gpovalues snapshots found in {snapshot_dir}.")

    snapshots: list[SnapshotFrame] = []
    for path in files:
        snapshot_date = _snapshot_date_from_path(path)
        frame = _prepare_snapshot(pd.read_csv(path), snapshot_date)
        snapshots.append(SnapshotFrame(snapshot_date=snapshot_date, path=path, frame=frame))
    return snapshots


def build_signal_observations(
    snapshots: list[SnapshotFrame],
    *,
    prior_signal_days: int = PRIOR_SIGNAL_DAYS,
    outcome_days: int = OUTCOME_DAYS,
    max_outcome_delay_days: int = MAX_OUTCOME_DELAY_DAYS,
) -> pd.DataFrame:
    """Build one item-date observation per row in each evaluation snapshot."""
    snapshots_by_date = {snapshot.snapshot_date: snapshot for snapshot in snapshots}
    records: list[dict[str, object]] = []

    for evaluation_snapshot in snapshots:
        prior_date = evaluation_snapshot.snapshot_date - timedelta(days=prior_signal_days)
        prior_snapshot = snapshots_by_date.get(prior_date)

        for _, row in evaluation_snapshot.frame.iterrows():
            record = _base_record(row, evaluation_snapshot.snapshot_date, prior_date, outcome_days)
            evaluation_status = _evaluation_status(row)
            record["evaluation_status"] = evaluation_status

            if evaluation_status == "eligible":
                _attach_prior_signals(record, row, prior_snapshot)
                _attach_outcome(
                    record,
                    row,
                    snapshots,
                    outcome_days=outcome_days,
                    max_outcome_delay_days=max_outcome_delay_days,
                )
            else:
                record["outcome_status"] = "not_evaluated"
                record["outcome_note"] = evaluation_status
                _mark_missing_signals(record, evaluation_status)

            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return _empty_observations()
    df["outcome_direction"] = df["outcome_change_pct"].apply(_outcome_direction)
    return df


def summarize_signal_groups(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize each signal group and its same-date baseline."""
    if observations.empty:
        return _empty_summary()

    eligible = observations[observations["evaluation_status"] == "eligible"].copy()
    summary_rows: list[dict[str, object]] = []
    signal_specs = [
        ("prior_7d_value_change", "prior_7d_value_change_group"),
        ("prior_7d_demand_ratio_change", "prior_7d_demand_ratio_change_group"),
        ("confidence_category", "confidence_category"),
    ]

    for signal_name, group_column in signal_specs:
        if group_column not in eligible.columns:
            continue
        for group_value in sorted(str(value) for value in eligible[group_column].dropna().unique()):
            group_rows = eligible[eligible[group_column].astype(str) == group_value]
            if group_rows.empty:
                continue
            summary_rows.append(_summarize_rows(signal_name, group_value, "signal group", group_rows))

            same_dates = set(group_rows["evaluation_date"])
            baseline_rows = eligible[eligible["evaluation_date"].isin(same_dates)]
            summary_rows.append(_summarize_rows(signal_name, group_value, "same-date baseline", baseline_rows))

    if not summary_rows:
        return _empty_summary()
    return pd.DataFrame(summary_rows).sort_values(["signal", "group", "cohort"]).reset_index(drop=True)


def _prepare_snapshot(df: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    prepared = df.copy()
    prepared["snapshot_date"] = snapshot_date.isoformat()
    prepared["item_identity"] = prepared.apply(_identity_key, axis=1)
    prepared["identity_source"] = prepared.apply(_identity_source, axis=1)
    for column in ("value", "demand_ratio"):
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        else:
            prepared[column] = pd.NA
    if "confidence" not in prepared.columns:
        prepared["confidence"] = ""

    counts = prepared["item_identity"].value_counts(dropna=False)
    prepared["identity_count"] = prepared["item_identity"].map(counts).fillna(0).astype(int)
    return prepared


def _base_record(row: pd.Series, evaluation_date: date, prior_date: date, outcome_days: int) -> dict[str, object]:
    target_date = evaluation_date + timedelta(days=outcome_days)
    return {
        "evaluation_date": evaluation_date.isoformat(),
        "item_identity": _string_or_blank(row.get("item_identity")),
        "identity_source": _string_or_blank(row.get("identity_source")),
        "item_name": _string_or_blank(row.get("name")),
        "item_slug": _string_or_blank(row.get("slug")),
        "item_shortcut": _string_or_blank(row.get("shortcut")),
        "evaluation_value": _optional_float(row.get("value")),
        "evaluation_demand_ratio": _optional_float(row.get("demand_ratio")),
        "confidence_category": _confidence_group(row.get("confidence")),
        "prior_date": prior_date.isoformat(),
        "prior_value": None,
        "prior_demand_ratio": None,
        "prior_7d_value_change_pct": None,
        "prior_7d_value_change_group": None,
        "prior_7d_demand_ratio_change": None,
        "prior_7d_demand_ratio_change_group": None,
        "target_date": target_date.isoformat(),
        "actual_outcome_date": None,
        "elapsed_days": None,
        "later_published_value": None,
        "outcome_change_pct": None,
        "outcome_status": None,
        "outcome_note": "",
        "signal_note": "",
    }


def _evaluation_status(row: pd.Series) -> str:
    if _is_blank(row.get("item_identity")):
        return "missing_item_identity"
    if int(row.get("identity_count", 0)) != 1:
        return "duplicate_evaluation_identity"
    value = _optional_float(row.get("value"))
    if value is None:
        return "missing_evaluation_value"
    if value <= 0:
        return "zero_or_negative_evaluation_value"
    return "eligible"


def _attach_prior_signals(
    record: dict[str, object],
    evaluation_row: pd.Series,
    prior_snapshot: SnapshotFrame | None,
) -> None:
    if prior_snapshot is None:
        _mark_missing_signals(record, "missing_prior_snapshot")
        return

    prior_status, prior_row = _match_identity(prior_snapshot.frame, str(evaluation_row["item_identity"]))
    if prior_status != "ready" or prior_row is None:
        _mark_missing_signals(record, f"prior_{prior_status}")
        return

    prior_value = _optional_float(prior_row.get("value"))
    evaluation_value = _optional_float(evaluation_row.get("value"))
    record["prior_value"] = prior_value
    if prior_value is None:
        record["prior_7d_value_change_group"] = "missing"
    elif prior_value <= 0:
        record["prior_7d_value_change_group"] = "invalid_zero"
    elif evaluation_value is not None:
        value_change = ((evaluation_value - prior_value) / prior_value) * 100
        record["prior_7d_value_change_pct"] = value_change
        record["prior_7d_value_change_group"] = _signed_group(value_change)

    prior_demand = _optional_float(prior_row.get("demand_ratio"))
    evaluation_demand = _optional_float(evaluation_row.get("demand_ratio"))
    record["prior_demand_ratio"] = prior_demand
    if prior_demand is None or evaluation_demand is None:
        record["prior_7d_demand_ratio_change_group"] = "missing"
    else:
        demand_change = evaluation_demand - prior_demand
        record["prior_7d_demand_ratio_change"] = demand_change
        record["prior_7d_demand_ratio_change_group"] = _signed_group(demand_change)


def _attach_outcome(
    record: dict[str, object],
    evaluation_row: pd.Series,
    snapshots: list[SnapshotFrame],
    *,
    outcome_days: int,
    max_outcome_delay_days: int,
) -> None:
    evaluation_date = date.fromisoformat(str(record["evaluation_date"]))
    target_date = evaluation_date + timedelta(days=outcome_days)
    latest_allowed_date = target_date + timedelta(days=max_outcome_delay_days)
    identity = str(evaluation_row["item_identity"])
    original_value = _optional_float(evaluation_row.get("value"))

    candidates = [
        snapshot
        for snapshot in snapshots
        if target_date <= snapshot.snapshot_date <= latest_allowed_date
    ]
    if not candidates:
        record["outcome_status"] = "no_snapshot_in_window"
        record["outcome_note"] = f"no snapshot from {target_date.isoformat()} through {latest_allowed_date.isoformat()}"
        return

    first_failure = ""
    for snapshot in candidates:
        match_status, outcome_row = _match_identity(snapshot.frame, identity)
        if match_status != "ready" or outcome_row is None:
            first_failure = first_failure or match_status
            continue
        later_value = _optional_float(outcome_row.get("value"))
        if later_value is None:
            first_failure = first_failure or "missing_later_value"
            continue
        if later_value <= 0:
            first_failure = first_failure or "zero_or_negative_later_value"
            continue
        if original_value is None or original_value <= 0:
            first_failure = first_failure or "invalid_original_value"
            continue

        record["actual_outcome_date"] = snapshot.snapshot_date.isoformat()
        record["elapsed_days"] = (snapshot.snapshot_date - evaluation_date).days
        record["later_published_value"] = later_value
        record["outcome_change_pct"] = ((later_value - original_value) / original_value) * 100
        record["outcome_status"] = "ready"
        return

    record["outcome_status"] = first_failure or "no_later_snapshot"
    record["outcome_note"] = "no unambiguous valid item outcome inside the allowed window"


def _mark_missing_signals(record: dict[str, object], note: str) -> None:
    record["prior_7d_value_change_group"] = "missing"
    record["prior_7d_demand_ratio_change_group"] = "missing"
    record["signal_note"] = note


def _match_identity(frame: pd.DataFrame, identity: str) -> tuple[str, pd.Series | None]:
    if _is_blank(identity):
        return "missing_item_identity", None
    matches = frame[frame["item_identity"] == identity]
    if matches.empty:
        return "missing_item", None
    if len(matches) > 1:
        return "duplicate_item", None
    return "ready", matches.iloc[0]


def _summarize_rows(signal: str, group: str, cohort: str, rows: pd.DataFrame) -> dict[str, object]:
    ready = rows[rows["outcome_status"] == "ready"]
    positive_pct = None
    if len(ready) > 0:
        positive_pct = (ready["outcome_change_pct"] > 0).mean() * 100

    return {
        "signal": signal,
        "group": group,
        "cohort": cohort,
        "observations": int(len(rows)),
        "unique_items": int(rows["item_identity"].nunique()),
        "missing_outcome_count": int((rows["outcome_status"] != "ready").sum()),
        "outcome_observations": int(len(ready)),
        "mean_later_published_value_change_pct": _rounded_or_none(ready["outcome_change_pct"].mean()),
        "median_later_published_value_change_pct": _rounded_or_none(ready["outcome_change_pct"].median()),
        "positive_change_pct": _rounded_or_none(positive_pct),
    }


def _identity_key(row: pd.Series) -> str:
    slug = _string_or_blank(row.get("slug")).strip()
    if slug:
        return slug
    join_key = _string_or_blank(row.get("join_key")).strip()
    if join_key:
        return join_key
    name = _string_or_blank(row.get("name")).strip().lower()
    return name


def _identity_source(row: pd.Series) -> str:
    if _string_or_blank(row.get("slug")).strip():
        return "slug"
    if _string_or_blank(row.get("join_key")).strip():
        return "join_key"
    if _string_or_blank(row.get("name")).strip():
        return "name"
    return ""


def _confidence_group(value: object) -> str:
    normalized = _string_or_blank(value).strip().lower()
    return normalized or "missing"


def _signed_group(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _outcome_direction(value: object) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return "missing"
    return _signed_group(parsed)


def _snapshot_date_from_path(snapshot_path: Path) -> date:
    name = snapshot_path.stem
    prefix = "gpovalues_"
    if not name.startswith(prefix):
        raise ValueError(f"Snapshot file must be named like gpovalues_YYYY-MM-DD.csv: {snapshot_path}")
    return date.fromisoformat(name.removeprefix(prefix))


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _rounded_or_none(value: object, digits: int = 2) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _string_or_blank(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _empty_observations() -> pd.DataFrame:
    columns = [
        "evaluation_date",
        "item_identity",
        "identity_source",
        "item_name",
        "item_slug",
        "item_shortcut",
        "evaluation_status",
        "evaluation_value",
        "evaluation_demand_ratio",
        "confidence_category",
        "prior_date",
        "prior_value",
        "prior_demand_ratio",
        "prior_7d_value_change_pct",
        "prior_7d_value_change_group",
        "prior_7d_demand_ratio_change",
        "prior_7d_demand_ratio_change_group",
        "target_date",
        "actual_outcome_date",
        "elapsed_days",
        "later_published_value",
        "outcome_change_pct",
        "outcome_status",
        "outcome_direction",
        "outcome_note",
        "signal_note",
    ]
    return pd.DataFrame(columns=columns)


def _empty_summary() -> pd.DataFrame:
    columns = [
        "signal",
        "group",
        "cohort",
        "observations",
        "unique_items",
        "missing_outcome_count",
        "outcome_observations",
        "mean_later_published_value_change_pct",
        "median_later_published_value_change_pct",
        "positive_change_pct",
    ]
    return pd.DataFrame(columns=columns)
