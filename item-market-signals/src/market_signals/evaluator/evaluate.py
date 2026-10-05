"""
The actual point of the project: given an item name (and optionally what a
seller is asking for it), tell you what it's really worth and whether the
asking price is a good deal.

Usage:
    python -m market_signals.evaluator.evaluate "Prestige Candy Cane"
    python -m market_signals.evaluator.evaluate "Candy Cane" --asking-price 350000

Note: typer collapses a single-command app, so there's no subcommand name --
just the item name and options directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from config.settings import OUTPUT_DIR
from market_signals.models.trend_model import compute_trend

app = typer.Typer(help="Look up an item's fair value and judge an asking price.")


def _load_feature_matrix(path: Path = OUTPUT_DIR / "feature_matrix_master.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No feature matrix at {path}. Run "
            "market_signals.features.build_feature_matrix first."
        )
    return pd.read_csv(path)


def find_item_matches(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return exact match candidates first, then substring candidates.

    Ambiguous substring matches are returned to the caller rather than
    resolved here so every interface can surface them honestly.
    """
    key = name.lower().strip()
    exact = df[df["join_key"] == key]
    if not exact.empty:
        return exact.head(1)

    # fall back to substring match, e.g. "candy cane" catching "Prestige
    # Candy Cane" too -- surface all matches rather than silently picking one
    return df[df["name"].str.lower().str.contains(key, na=False, regex=False)]


def resolve_item(df: pd.DataFrame, name: str) -> pd.Series | None:
    matches = find_item_matches(df, name)
    if len(matches) == 1:
        return matches.iloc[0]
    return None


def _find_item(df: pd.DataFrame, name: str) -> pd.Series | None:
    matches = find_item_matches(df, name)
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        typer.echo(f"Multiple matches for '{name}':")
        for _, row in matches.iterrows():
            typer.echo(f"  - {row['name']}")
        typer.echo("Be more specific.")
        return None
    return None


def verdict(asking_price: float, value: float, ci_low: float, ci_high: float) -> str:
    if asking_price <= ci_low:
        return "GOOD DEAL -- asking price is below the typical trade range"
    if asking_price <= value:
        return "FAIR -- asking price is at or below the solved fair value"
    if asking_price <= ci_high:
        return "SLIGHTLY HIGH -- still inside the typical range, room to negotiate"
    return "OVERPRICED -- asking price is above the typical trade range"


def trade_range_verdict(
    give_ci_low: float,
    give_ci_high: float,
    get_ci_low: float,
    get_ci_high: float,
) -> str:
    """Compare summed trade ranges without forcing a false precise answer."""
    if give_ci_low > get_ci_high:
        return "FAVORS OTHER PERSON -- you're giving up more than you'd get"
    if get_ci_low > give_ci_high:
        return "FAVORS YOU -- you'd get more value than you're giving"
    return "ROUGHLY FAIR -- summed ranges overlap, so it is hard to call precisely"


def _verdict(asking_price: float, value: float, ci_low: float, ci_high: float) -> str:
    return verdict(asking_price, value, ci_low, ci_high)


@app.command()
def check(
    item_name: str = typer.Argument(..., help="Item name, e.g. 'Prestige Candy Cane'"),
    asking_price: float = typer.Option(None, "--asking-price", help="What the seller is asking"),
):
    df = _load_feature_matrix()
    row = _find_item(df, item_name)
    if row is None:
        typer.echo(f"No match found for '{item_name}'.")
        raise typer.Exit(code=1)

    typer.echo(f"\n{row['name']} ({row.get('shortcut', '')})")
    typer.echo(f"  Fair value      : {row['value']:,.0f}")
    typer.echo(f"  Typical range   : {row['ci_low']:,.0f} - {row['ci_high']:,.0f}")
    typer.echo(f"  Confidence      : {row['confidence']}")
    typer.echo(f"  Activity signal : trade_count={int(row['trade_count'])}")
    typer.echo(f"  Demand          : {row.get('demand', 'unknown')}")

    if row["confidence"] == "low":
        typer.echo(
            "  Note: low-confidence item (thin trade_count activity signal). "
            "Treat this value as directional, not exact."
        )

    trend = compute_trend(row["join_key"])
    if trend:
        typer.echo(
            f"  Trend           : {trend['direction']} {trend['pct_change']:+.1f}% "
            f"over {trend['n_snapshots']} snapshots "
            f"({trend['first_date']} -> {trend['last_date']})"
        )
    else:
        typer.echo("  Trend           : not enough snapshot history yet")

    if asking_price is not None:
        verdict = _verdict(asking_price, row["value"], row["ci_low"], row["ci_high"])
        typer.echo(f"\n  Asking price {asking_price:,.0f} -> {verdict}")

    typer.echo("")


if __name__ == "__main__":
    app()
