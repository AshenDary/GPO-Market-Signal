# Data dictionary and walkthrough

This reference connects the repository's saved tables to the code paths that
produce and consume them. gpovalues fields are published market estimates and
signals. They are not records of this project's completed trades.

## Data flow

1. `scripts/run_ingest_gpovalues.py` fetches the gpovalues API, saves the raw
   response under ignored `data/raw/`, and writes a dated
   `data/snapshots/gpovalues_YYYY-MM-DD.csv` snapshot.
2. `scripts/run_ingest_tier.py` parses the curated
   `data/raw/gpo_market_dataset.json` into a dated structural
   `tier_reference_YYYY-MM-DD.csv` snapshot.
3. `scripts/run_feature_build.py` merges the latest snapshot of each type into
   `outputs/feature_matrix_master.csv`. It tries normalized exact item name
   first and then gpovalues shortcut versus tier alias. Unmatched gpovalues rows
   remain in the output with `has_tier_enrichment=false`.
4. Item Lookup, Trade Simulator, Value List, and the evaluator use the current
   merged matrix. Trend reads all gpovalues snapshots.
5. Historical Signal Lab reads only dated gpovalues snapshots. Market Change
   Alerts compares the latest two and uses older snapshots for its minimum
   item-history guard. Neither analysis reads the private decision log.
6. The private decision log saves the gpovalues row available at decision time
   beside personal buy/no-buy fields. Actual resale fields are entered manually;
   read-only marked outcomes come from later gpovalues snapshots.

## gpovalues snapshots

These columns come from `pull_gpovalues_snapshot.py` and are preserved in the
merged feature matrix.

| Field | Meaning in this project |
|---|---|
| `slug` | gpovalues item identifier; preferred identity for historical analyses when present |
| `name`, `shortcut` | Published display name and abbreviation |
| `value` | gpovalues published solved-value estimate, not a recorded resale price |
| `ci_low`, `ci_high` | gpovalues published lower and upper range used by price verdicts |
| `confidence` | Confidence category published by gpovalues; the project preserves this label |
| `rarity`, `tier` | gpovalues source classifications |
| `demand`, `demand_ratio` | gpovalues demand label and numeric demand signal |
| `trade_count` | gpovalues activity signal; it can decrease and is not daily volume or a count of new trades |
| `derivation` | Source-provided value derivation detail |
| `share_url`, `image_url` | Source links used for display or reference |
| `generated_at`, `n_trades_used` | API-level snapshot metadata repeated on each row |
| `join_key` | Lowercased, trimmed `name` used for matching fallback |

The Historical Signal Lab and Market Change Alerts identify an item by `slug`
when available, then `join_key`, then exact item name. Missing or duplicated
identities are reported as statuses rather than silently resolved.

## Tier reference and merged matrix

The tier reference is structural context, not another market-price source.

| Field | Meaning in this project |
|---|---|
| `item_name`, `alias` | Curated display name and optional alias used for enrichment matching |
| `category`, `tier`, `sub_tier` | Curated structural classifications |
| `tier_ordinal`, `sub_tier_ordinal` | Numeric encodings used by model features |
| `obtainability`, `is_unobtainable`, `is_unstable` | Curated availability and stability context |
| `demand_status`, `demand_ordinal` | Curated demand context and its numeric encoding |
| `rarity_score`, `popularity_score` | Curated structural scores when a detailed record exists |
| `has_detailed_record` | Whether detailed curated scores were available |
| `fallback_value_score` | Hand-weighted structural score; not a price or gpovalues estimate |

After merging, `tier_item_name` records the matched tier-reference name and
`has_tier_enrichment` records whether a match succeeded. Because both sources
have a `tier` column, `tier` is the gpovalues field and `tier_tier` is the
curated tier-reference field in the generated matrix.

## Historical analysis interpretation

Historical Signal Lab creates item-date observations. Its prior value signal
requires the same item exactly seven calendar days earlier. Its demand-ratio
signal requires numeric values on both dates. The 14-day outcome is the first
unambiguous same-item snapshot from the target date through three days later.

- `pending`: the target date is after the report's `as_of` date.
- `awaiting_window`: the target has arrived, but the allowed three-day search
  window is not complete and no valid outcome has appeared yet.
- `ready`: a valid later published estimate was found.
- Other explicit statuses cover missing snapshots or items, duplicate
  identities, and invalid or zero values after a completed search window.

The default `as_of` date is the latest saved gpovalues snapshot, not today's
date. A stale snapshot directory therefore gives a reproducible historical
report, but not current market evidence. Signal-group results are exploratory
comparisons with same-date baselines; they are not forecasts or independent
completed trades.

Market Change Alerts compares the latest saved snapshot with the immediately
previous one. It surfaces missing history, missing comparison rows, ambiguous
identities, invalid or extreme values, and stale latest data. Its market-wide
context summarizes comparable published-value changes so an item move can be
read against a broad rise, fall, or mixed window. It does not explain the cause
of that broad movement or turn an alert into a recommendation.

## Private decision log

`data/decisions/decision_log.csv` is ignored by Git. It combines four kinds of
fields that should stay distinct:

| Field group | Examples | Interpretation |
|---|---|---|
| Snapshot-time gpovalues data | `published_value`, `ci_low`, `ci_high`, `confidence`, `demand_ratio`, `trade_count` | What gpovalues published when the decision was logged |
| Personal decision data | `asking_price`, `verdict`, `bought`, `purchase_price`, `notes` | The user's decision and transaction inputs |
| Actual resale data | `resale_date`, `resale_price`, `resale_notes` | A manually recorded completed resale |
| Marked-value fields | `marked_value_date`, `marked_value`, `marked_value_source` | Reserved saved columns; the current `marked-outcome` command is read-only and does not update them |

A marked outcome reports a later change in gpovalues' published estimate. It
is not realized profit and does not prove that a purchase or resale occurred.

## Read-only walkthrough

After completing the setup in the README, these commands use committed data and
do not fetch live data or write a decision:

```bash
python -m market_signals.evaluator.evaluate "Prestige Candy Cane"
python -m market_signals.evaluator.evaluate "Candy Cane" --asking-price 300000
python -m market_signals.models.trend_model
python scripts/run_decision_log.py list
streamlit run dashboard/app.py
```

In the dashboard, use Item Lookup for the current merged row, Trend for saved
published-value history, Historical Signal Lab for exploratory 7-day signals
and 14-day published-estimate outcomes, and Market Change Alerts for the latest
snapshot comparison. A live refresh is a separate, network-writing operation;
run the three pipeline commands in the README only when updated market data is
intended.
