"""CLI for the private decision log."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from config.settings import DECISION_LOG_PATH, SNAPSHOT_DIR
from market_signals.decisions.log import (
    AmbiguousItemMatchError,
    DecisionLogError,
    MarkedOutcome,
    append_decision,
    build_decision_entry,
    get_decision,
    load_decision_log,
    marked_outcome_for_decision,
    record_resale_outcome,
    recompute_saved_verdict,
)

app = typer.Typer(help="Record and inspect private GPO trading decisions.")


@app.command("log")
def log_command(
    item_name: str = typer.Argument(..., help="Item name as it appears in gpovalues."),
    asking_price: float = typer.Option(..., "--asking-price", min=0.0, help="Seller's asking price."),
    decision: str = typer.Option(..., "--decision", help="Use 'buy' or 'no-buy'."),
    purchase_price: Optional[float] = typer.Option(None, "--purchase-price", min=0.0, help="Actual paid price, if bought."),
    notes: str = typer.Option("", "--notes", help="Optional private note."),
    snapshot_file: Optional[Path] = typer.Option(None, "--snapshot-file", help="Optional gpovalues_YYYY-MM-DD.csv file."),
    log_path: Path = typer.Option(DECISION_LOG_PATH, "--log-path", help="Private CSV log path."),
) -> None:
    """Append one buy/no-buy decision using snapshot-time values."""
    try:
        bought = _parse_decision(decision)
        entry = build_decision_entry(
            item_name,
            asking_price,
            bought,
            purchase_price=purchase_price,
            notes=notes,
            snapshot_path=snapshot_file,
        )
        append_decision(entry, log_path)
    except AmbiguousItemMatchError as exc:
        typer.echo(f"Ambiguous item match: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (DecisionLogError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not log decision: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Logged decision {entry['decision_id']}")
    typer.echo(f"  Item       : {entry['item_name']}")
    typer.echo(f"  Snapshot   : {entry['snapshot_file']}")
    typer.echo(f"  Verdict    : {entry['verdict']}")
    typer.echo(f"  Decision   : {'buy' if entry['bought'] else 'no-buy'}")


@app.command("list")
def list_command(
    log_path: Path = typer.Option(DECISION_LOG_PATH, "--log-path", help="Private CSV log path."),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum rows to show."),
) -> None:
    """List recent private decisions."""
    log = load_decision_log(log_path)
    if log.empty:
        typer.echo(f"No decisions logged yet at {log_path}.")
        return

    display = log.tail(limit)[
        [
            "decision_id",
            "decision_timestamp",
            "item_name",
            "asking_price",
            "verdict",
            "bought",
            "purchase_price",
            "resale_price",
        ]
    ].copy()
    typer.echo(display.to_string(index=False))


@app.command("show")
def show_command(
    decision_id: str = typer.Argument(..., help="Decision id to inspect."),
    log_path: Path = typer.Option(DECISION_LOG_PATH, "--log-path", help="Private CSV log path."),
) -> None:
    """Show one saved decision and verify its saved verdict inputs."""
    try:
        row = get_decision(decision_id, log_path)
        reproduced = recompute_saved_verdict(row)
    except (DecisionLogError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not show decision: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(_format_decision(row, reproduced))


@app.command("marked-outcome")
def marked_outcome_command(
    decision_id: str = typer.Argument(..., help="Decision id to inspect."),
    log_path: Path = typer.Option(DECISION_LOG_PATH, "--log-path", help="Private CSV log path."),
    snapshot_dir: Path = typer.Option(SNAPSHOT_DIR, "--snapshot-dir", help="Directory containing gpovalues_YYYY-MM-DD.csv files."),
) -> None:
    """Show the 14-day gpovalues published-estimate change for one decision."""
    try:
        outcome = marked_outcome_for_decision(
            decision_id,
            log_path=log_path,
            snapshot_dir=snapshot_dir,
        )
    except (DecisionLogError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not inspect marked outcome: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(_format_marked_outcome(outcome))


@app.command("record-resale")
def record_resale_command(
    decision_id: str = typer.Argument(..., help="Decision id to update."),
    resale_price: float = typer.Option(..., "--resale-price", min=0.0, help="Actual resale price."),
    resale_date: Optional[str] = typer.Option(None, "--resale-date", help="Actual resale date, YYYY-MM-DD."),
    resale_notes: str = typer.Option("", "--notes", help="Optional resale note."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing resale outcome."),
    log_path: Path = typer.Option(DECISION_LOG_PATH, "--log-path", help="Private CSV log path."),
) -> None:
    """Record an actual resale outcome for a logged decision."""
    try:
        row = record_resale_outcome(
            decision_id,
            resale_price,
            resale_date=resale_date or date.today().isoformat(),
            resale_notes=resale_notes,
            log_path=log_path,
            force=force,
        )
    except (DecisionLogError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not record resale: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Recorded actual resale for {row['decision_id']}: {float(row['resale_price']):,.0f}")


def _parse_decision(decision: str) -> bool:
    normalized = decision.lower().strip()
    if normalized in {"buy", "bought", "yes", "y"}:
        return True
    if normalized in {"no-buy", "no_buy", "pass", "skip", "no", "n"}:
        return False
    raise ValueError("--decision must be 'buy' or 'no-buy'.")


def _format_decision(row: pd.Series, reproduced_verdict: str) -> str:
    purchase_price = "blank" if _is_blank(row.get("purchase_price")) else f"{float(row['purchase_price']):,.0f}"
    resale_price = "blank" if _is_blank(row.get("resale_price")) else f"{float(row['resale_price']):,.0f}"
    return "\n".join(
        [
            f"Decision {row['decision_id']}",
            f"  Timestamp          : {row['decision_timestamp']}",
            f"  Snapshot           : {row['snapshot_file']} ({row['snapshot_generated_at']})",
            f"  Item               : {row['item_name']} ({row['item_slug']})",
            f"  Asking price       : {float(row['asking_price']):,.0f}",
            f"  Published value    : {float(row['published_value']):,.0f}",
            f"  Published range    : {float(row['ci_low']):,.0f} - {float(row['ci_high']):,.0f}",
            f"  Confidence         : {row['confidence']}",
            f"  Demand/activity    : {row['demand']} / demand_ratio={row['demand_ratio']} / trade_count={row['trade_count']}",
            f"  Saved verdict      : {row['verdict']}",
            f"  Recomputed verdict : {reproduced_verdict}",
            f"  Decision           : {'buy' if _as_bool(row['bought']) else 'no-buy'}",
            f"  Purchase price     : {purchase_price}",
            f"  Actual resale      : {resale_price}",
        ]
    )


def _format_marked_outcome(outcome: MarkedOutcome) -> str:
    lines = [
        f"Decision {outcome.decision_id}",
        f"  Item                         : {outcome.item_name}",
        "  Measure                      : 14-day change in gpovalues' published estimate",
        "  Interpretation               : not profit and not a completed trade",
        f"  Target date                  : {outcome.target_date.isoformat()}",
        f"  Status                       : {outcome.status}",
        f"  Original published estimate  : {_format_value(outcome.original_published_value)}",
    ]
    if outcome.status == "ready":
        lines.extend(
            [
                f"  Later published estimate     : {_format_value(outcome.later_published_value)}",
                f"  Estimate change              : {outcome.percentage_change:+.1f}%",
                f"  Actual snapshot date         : {outcome.actual_snapshot_date.isoformat()}",
                f"  Days elapsed                 : {outcome.days_elapsed}",
                f"  Snapshot source              : {outcome.snapshot_file}",
                f"  Identity match               : {outcome.identity_field}",
            ]
        )
    elif outcome.reason:
        lines.append(f"  Reason                       : {outcome.reason}")

    if outcome.actual_resale_price is not None:
        resale_date = outcome.actual_resale_date or "date not recorded"
        lines.append(f"  Actual resale                : {_format_value(outcome.actual_resale_price)} on {resale_date}")
    else:
        lines.append("  Actual resale                : blank")
    return "\n".join(lines)


def _format_value(value: float | None) -> str:
    if value is None:
        return "blank"
    return f"{value:,.0f}"


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "buy", "bought"}


if __name__ == "__main__":
    app()
