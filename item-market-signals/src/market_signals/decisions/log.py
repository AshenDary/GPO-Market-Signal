"""Private decision log built from snapshot-time market signals.

The log records what the tool knew when a buy/no-buy decision was made. It
stores gpovalues' published market value and range separately from personal
asking, purchase, and resale fields so later analysis does not mix observed
market estimates with actual completed transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from config.settings import DECISION_LOG_PATH, SNAPSHOT_DIR
from market_signals.evaluator.evaluate import find_item_matches, verdict

DECISION_COLUMNS = [
    "decision_id",
    "decision_timestamp",
    "snapshot_date",
    "snapshot_file",
    "snapshot_generated_at",
    "item_slug",
    "item_name",
    "item_shortcut",
    "join_key",
    "asking_price",
    "published_value",
    "ci_low",
    "ci_high",
    "confidence",
    "demand",
    "demand_ratio",
    "trade_count",
    "n_trades_used",
    "verdict",
    "bought",
    "purchase_price",
    "resale_date",
    "resale_price",
    "resale_notes",
    "marked_value_date",
    "marked_value",
    "marked_value_source",
    "notes",
]


class DecisionLogError(ValueError):
    """Base class for decision-log validation errors."""


class ItemNotFoundError(DecisionLogError):
    """Raised when a requested item is absent from the selected snapshot."""


class AmbiguousItemMatchError(DecisionLogError):
    """Raised when a lookup matches multiple snapshot rows."""


class MissingItemPriceError(DecisionLogError):
    """Raised when a snapshot row lacks the fields needed for a verdict."""


class DecisionNotFoundError(DecisionLogError):
    """Raised when an update targets a decision id that is not logged."""


class DuplicateDecisionIDError(DecisionLogError):
    """Raised when a decision id appears more than once."""


class ResaleAlreadyRecordedError(DecisionLogError):
    """Raised when a resale update would overwrite an existing actual sale."""


class ResaleWithoutPurchaseError(DecisionLogError):
    """Raised when a resale update targets a no-buy decision."""


class PurchasePriceForNoBuyError(DecisionLogError):
    """Raised when a no-buy decision includes an actual purchase price."""


@dataclass(frozen=True)
class MarkedOutcome:
    """Read-only 14-day gpovalues estimate change for one saved decision."""

    status: str
    decision_id: str
    item_name: str
    target_date: date
    original_published_value: float | None
    later_published_value: float | None = None
    percentage_change: float | None = None
    actual_snapshot_date: date | None = None
    days_elapsed: int | None = None
    snapshot_file: str = ""
    identity_field: str = ""
    actual_resale_date: str = ""
    actual_resale_price: float | None = None
    reason: str = ""


def latest_gpovalues_snapshot(snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    """Return the latest dated gpovalues snapshot by filename date."""
    matches = sorted(snapshot_dir.glob("gpovalues_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No gpovalues snapshots found in {snapshot_dir}.")
    return matches[-1]


def snapshot_date_from_path(snapshot_path: Path) -> str:
    """Extract the ISO date from a gpovalues snapshot filename."""
    name = snapshot_path.stem
    prefix = "gpovalues_"
    if not name.startswith(prefix):
        raise ValueError(f"Snapshot file must be named like gpovalues_YYYY-MM-DD.csv: {snapshot_path}")
    return name.removeprefix(prefix)


def load_snapshot(snapshot_path: Path | None = None, snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Load the selected gpovalues snapshot, defaulting to the latest one."""
    selected_path = snapshot_path or latest_gpovalues_snapshot(snapshot_dir)
    snapshot = pd.read_csv(selected_path)
    snapshot["snapshot_date"] = snapshot_date_from_path(selected_path)
    snapshot["snapshot_file"] = selected_path.name
    return snapshot


def resolve_snapshot_item(snapshot: pd.DataFrame, item_name: str) -> pd.Series:
    """Resolve one item in a snapshot using the evaluator's matching rules."""
    key = item_name.lower().strip()
    exact = snapshot[snapshot["join_key"] == key]
    if len(exact) > 1:
        names = ", ".join(str(name) for name in exact["name"].dropna().head(8))
        raise AmbiguousItemMatchError(
            f"Multiple exact matches for '{item_name}': {names}. The snapshot has duplicate item keys."
        )
    if len(exact) == 1:
        return exact.iloc[0]

    matches = find_item_matches(snapshot, item_name)
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        names = ", ".join(str(name) for name in matches["name"].dropna().head(8))
        raise AmbiguousItemMatchError(
            f"Multiple matches for '{item_name}': {names}. Use an exact item name."
        )
    raise ItemNotFoundError(f"No match found for '{item_name}' in the selected snapshot.")


def recompute_saved_verdict(entry: pd.Series | dict[str, object]) -> str:
    """Recompute a verdict only from fields saved in the decision log."""
    return verdict(
        _required_float(entry, "asking_price"),
        _required_float(entry, "published_value"),
        _required_float(entry, "ci_low"),
        _required_float(entry, "ci_high"),
    )


def build_decision_entry(
    item_name: str,
    asking_price: float,
    bought: bool,
    *,
    purchase_price: float | None = None,
    notes: str = "",
    snapshot_path: Path | None = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
    decision_timestamp: datetime | None = None,
) -> dict[str, object]:
    """Build one log row from the snapshot available at decision time."""
    if asking_price <= 0:
        raise ValueError("asking_price must be greater than zero.")
    if not bought and purchase_price is not None:
        raise PurchasePriceForNoBuyError("purchase_price is only valid for a buy decision.")
    if purchase_price is not None and purchase_price <= 0:
        raise ValueError("purchase_price must be greater than zero when provided.")

    selected_snapshot_path = snapshot_path or latest_gpovalues_snapshot(snapshot_dir)
    snapshot = load_snapshot(selected_snapshot_path, snapshot_dir)
    row = resolve_snapshot_item(snapshot, item_name)

    value = _required_float(row, "value")
    ci_low = _required_float(row, "ci_low")
    ci_high = _required_float(row, "ci_high")
    decision_time = decision_timestamp or datetime.now(timezone.utc)
    saved_verdict = verdict(asking_price, value, ci_low, ci_high)

    return {
        "decision_id": uuid4().hex,
        "decision_timestamp": decision_time.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": str(row["snapshot_date"]),
        "snapshot_file": str(row["snapshot_file"]),
        "snapshot_generated_at": _string_or_blank(row.get("generated_at")),
        "item_slug": _string_or_blank(row.get("slug")),
        "item_name": _string_or_blank(row.get("name")),
        "item_shortcut": _string_or_blank(row.get("shortcut")),
        "join_key": _string_or_blank(row.get("join_key")),
        "asking_price": float(asking_price),
        "published_value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": _string_or_blank(row.get("confidence")),
        "demand": _string_or_blank(row.get("demand")),
        "demand_ratio": _optional_float(row.get("demand_ratio")),
        "trade_count": _optional_float(row.get("trade_count")),
        "n_trades_used": _optional_float(row.get("n_trades_used")),
        "verdict": saved_verdict,
        "bought": bool(bought),
        "purchase_price": "" if purchase_price is None else float(purchase_price),
        "resale_date": "",
        "resale_price": "",
        "resale_notes": "",
        "marked_value_date": "",
        "marked_value": "",
        "marked_value_source": "",
        "notes": notes,
    }


def append_decision(entry: dict[str, object], log_path: Path = DECISION_LOG_PATH) -> dict[str, object]:
    """Append one decision entry to the private CSV log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_decision_log(log_path)
    decision_id = str(entry.get("decision_id", "")).strip()
    if not decision_id:
        raise ValueError("decision_id must be present before appending a decision.")
    if not existing.empty and (existing["decision_id"].astype(str) == decision_id).any():
        raise DuplicateDecisionIDError(f"Decision id '{decision_id}' already exists in {log_path}.")

    new_row = pd.DataFrame([entry], columns=DECISION_COLUMNS)
    updated = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(log_path, index=False)
    return entry


def log_decision(
    item_name: str,
    asking_price: float,
    bought: bool,
    *,
    purchase_price: float | None = None,
    notes: str = "",
    snapshot_path: Path | None = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
    log_path: Path = DECISION_LOG_PATH,
    decision_timestamp: datetime | None = None,
) -> dict[str, object]:
    """Build and append one private decision-log row."""
    entry = build_decision_entry(
        item_name,
        asking_price,
        bought,
        purchase_price=purchase_price,
        notes=notes,
        snapshot_path=snapshot_path,
        snapshot_dir=snapshot_dir,
        decision_timestamp=decision_timestamp,
    )
    return append_decision(entry, log_path)


def load_decision_log(log_path: Path = DECISION_LOG_PATH) -> pd.DataFrame:
    """Load the private decision log, returning an empty schema if absent."""
    if not log_path.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)
    df = pd.read_csv(log_path, dtype={"decision_id": str})
    for column in DECISION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[DECISION_COLUMNS]


def get_decision(decision_id: str, log_path: Path = DECISION_LOG_PATH) -> pd.Series:
    """Return one logged decision by id."""
    log = load_decision_log(log_path)
    matches = log[log["decision_id"] == decision_id]
    if len(matches) > 1:
        raise DuplicateDecisionIDError(
            f"Decision id '{decision_id}' appears {len(matches)} times in {log_path}."
        )
    if len(matches) == 0:
        raise DecisionNotFoundError(f"No decision found with id '{decision_id}'.")
    return matches.iloc[0]


def marked_outcome_for_decision(
    decision_id: str,
    *,
    log_path: Path = DECISION_LOG_PATH,
    snapshot_dir: Path = SNAPSHOT_DIR,
    as_of: date | None = None,
) -> MarkedOutcome:
    """Calculate a read-only 14-day gpovalues published estimate outcome."""
    row = get_decision(decision_id, log_path)
    decision_day = _decision_date(row["decision_timestamp"])
    target_date = decision_day + timedelta(days=14)
    original_value = _optional_float(row.get("published_value"))
    actual_resale_price = _optional_float(row.get("resale_price"))
    actual_resale_date = "" if _is_blank(row.get("resale_date")) else str(row.get("resale_date"))

    base = {
        "decision_id": str(row["decision_id"]),
        "item_name": _string_or_blank(row.get("item_name")),
        "target_date": target_date,
        "original_published_value": original_value,
        "actual_resale_date": actual_resale_date,
        "actual_resale_price": actual_resale_price,
    }

    today = as_of or date.today()
    if target_date > today:
        return MarkedOutcome(status="pending", reason="target date has not arrived", **base)
    if original_value is None:
        return MarkedOutcome(status="no original value", reason="saved decision has no original published value", **base)
    if original_value == 0:
        return MarkedOutcome(status="no original value", reason="saved original published value is zero", **base)

    for snapshot_path in gpovalues_snapshots_on_or_after(target_date, snapshot_dir):
        snapshot_date = date.fromisoformat(snapshot_date_from_path(snapshot_path))
        snapshot = load_snapshot(snapshot_path, snapshot_dir)
        match = _resolve_saved_identity(snapshot, row)
        if match is None:
            continue

        matched_row, identity_field = match
        later_value = _optional_float(matched_row.get("value"))
        if later_value is None:
            continue
        pct_change = ((later_value - original_value) / original_value) * 100
        return MarkedOutcome(
            status="ready",
            later_published_value=later_value,
            percentage_change=pct_change,
            actual_snapshot_date=snapshot_date,
            days_elapsed=(snapshot_date - decision_day).days,
            snapshot_file=snapshot_path.name,
            identity_field=identity_field,
            **base,
        )

    return MarkedOutcome(status="no later snapshot", reason="no later snapshot contains one matching item row", **base)


def gpovalues_snapshots_on_or_after(target_date: date, snapshot_dir: Path = SNAPSHOT_DIR) -> list[Path]:
    """Return dated gpovalues snapshots at or after target_date, sorted oldest first."""
    snapshots: list[tuple[date, Path]] = []
    for path in snapshot_dir.glob("gpovalues_*.csv"):
        try:
            snapshot_date = date.fromisoformat(snapshot_date_from_path(path))
        except ValueError:
            continue
        if snapshot_date >= target_date:
            snapshots.append((snapshot_date, path))
    return [path for _, path in sorted(snapshots)]


def record_resale_outcome(
    decision_id: str,
    resale_price: float,
    *,
    resale_date: date | str | None = None,
    resale_notes: str = "",
    log_path: Path = DECISION_LOG_PATH,
    force: bool = False,
) -> pd.Series:
    """Record an actual resale outcome without touching marked-value fields."""
    if resale_price <= 0:
        raise ValueError("resale_price must be greater than zero.")

    log = load_decision_log(log_path)
    matches = log.index[log["decision_id"] == decision_id].tolist()
    if len(matches) > 1:
        raise DuplicateDecisionIDError(
            f"Decision id '{decision_id}' appears {len(matches)} times in {log_path}."
        )
    if len(matches) == 0:
        raise DecisionNotFoundError(f"No decision found with id '{decision_id}'.")

    idx = matches[0]
    if not _as_bool(log.at[idx, "bought"]):
        raise ResaleWithoutPurchaseError(
            f"Decision '{decision_id}' was logged as no-buy, so it cannot receive an actual resale outcome."
        )

    existing_price = log.at[idx, "resale_price"]
    if not force and not _is_blank(existing_price):
        raise ResaleAlreadyRecordedError(
            f"Decision '{decision_id}' already has a resale price. Use force=True to overwrite."
        )

    for column in ("resale_date", "resale_notes"):
        log[column] = log[column].astype("object")
    actual_date = _parse_resale_date(resale_date)
    decision_date = _decision_date(log.at[idx, "decision_timestamp"])
    today = date.today()
    if actual_date < decision_date:
        raise ValueError("resale_date cannot be before the original decision date.")
    if actual_date > today:
        raise ValueError("resale_date cannot be in the future for an actual completed sale.")

    log.at[idx, "resale_date"] = actual_date.isoformat()
    log.at[idx, "resale_price"] = float(resale_price)
    log.at[idx, "resale_notes"] = resale_notes
    log.to_csv(log_path, index=False)
    return log.iloc[idx]


def _resolve_saved_identity(snapshot: pd.DataFrame, decision: pd.Series) -> tuple[pd.Series, str] | None:
    """Resolve the saved item in one snapshot by exact identity fields only."""
    candidate_indices: set[int] = set()
    matched_fields: list[str] = []

    identity_specs = [
        ("slug", "item_slug", False),
        ("join_key", "join_key", False),
        ("name", "item_name", True),
        ("shortcut", "item_shortcut", True),
    ]
    for snapshot_column, decision_column, case_insensitive in identity_specs:
        if snapshot_column not in snapshot.columns:
            continue
        saved_value = decision.get(decision_column)
        if _is_blank(saved_value):
            continue

        snapshot_values = snapshot[snapshot_column]
        if case_insensitive:
            matches = snapshot_values.astype(str).str.lower().str.strip() == str(saved_value).lower().strip()
        else:
            matches = snapshot_values.astype(str).str.strip() == str(saved_value).strip()
        indices = set(snapshot.index[matches].tolist())
        if indices:
            candidate_indices.update(indices)
            matched_fields.append(snapshot_column)

    if len(candidate_indices) != 1:
        return None
    idx = next(iter(candidate_indices))
    return snapshot.loc[idx], "+".join(matched_fields)


def _required_float(row: pd.Series | dict[str, object], field: str) -> float:
    value = row[field] if isinstance(row, dict) else row.get(field)
    parsed = _optional_float(value)
    if parsed is None:
        raise MissingItemPriceError(f"Selected item is missing required price field '{field}'.")
    return parsed


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "buy", "bought"}


def _parse_resale_date(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("resale_date must be a valid ISO date in YYYY-MM-DD format.") from exc


def _decision_date(decision_timestamp: object) -> date:
    timestamp = str(decision_timestamp).strip()
    if not timestamp:
        raise ValueError("decision_timestamp is required to validate resale_date.")
    normalized = timestamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError as exc:
        raise ValueError("decision_timestamp must be a valid ISO timestamp.") from exc
