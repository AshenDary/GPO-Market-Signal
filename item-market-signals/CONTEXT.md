# CONTEXT.md

Read this before doing anything else in this repo. It is the condensed
architecture + state doc so you do not need the full chat history that
produced this project. For command recipes, see `SKILLS.md`. For the
phase-by-phase plan, see `ROADMAP.md`. For setup/usage, see `README.md`.

## What this is

Item Market Signals is a market intelligence dashboard and evaluator for the
Grand Piece Online trading economy. It combines community-solved values from
gpovalues.com with a curated tier/rarity reference, stores dated snapshots, and
turns the result into:

- current fair-value lookup with confidence bands
- asking-price verdicts: good deal, fair, slightly high, or overpriced
- private saved-decision inspection with read-only 14-day gpovalues marked
  outcomes
- snapshot-based trend context
- latest-vs-previous snapshot Market Change Alerts for unusual value,
  demand-ratio, and `trade_count` activity changes
- exploratory historical signal evaluation from raw gpovalues snapshots
- trade-side comparison through a simulator
- structural value-model diagnostics, low-confidence estimates, and SHAP
  explanations for the value regression model

This is a personal tool and a data science portfolio project. Both matter:
code should work correctly for real buying decisions and be defensible in an
interview. Honest uncertainty handling matters as much as the happy path.

The project now includes a private, local decision log for real buy/no-buy
checks, asking prices, purchase prices, and actual resale outcomes when they
exist. This log is personal workflow evidence, not the market-value source of
truth. gpovalues snapshots remain the primary source for published market
values, confidence bands, demand signals, and marked 14-day published-estimate
outcomes.

## Data sources

1. **gpovalues.com API** (primary, real prices) - public JSON API at
   `https://gpovalues.com/api/v1/items.json`, no auth. It provides solved
   `value`, `ci_low`, `ci_high`, `confidence`, `demand`, `demand_ratio`,
   `trade_count`, images, and share URLs. See
   https://gpovalues.com/legal/methodology for their value methodology.
2. **Tier/rarity reference JSON** (secondary, structural) - curated community
   tier-list data in `data/raw/gpo_market_dataset.json`. This provides
   category, tier, sub-tier, obtainability, rarity/popularity signals, aliases,
   and flags such as unstable/unobtainable. It is structural context, not price
   data.

## Architecture

```text
item-market-signals/
  dashboard/
    app.py                      Streamlit router using st.Page top navigation
    pages/
      guide.py                  Start Here / dashboard directory
      overview.py               coverage, confidence, tier scatter, most-traded ranking
      lookup.py                 one-item lookup, asking-price verdict, low-confidence estimate panel
      simulator.py              two-sided trade simulator with dialog-based item add flow
      model_insights.py         value regression diagnostics, anomalies, SHAP explanations
      trend.py                  per-item snapshot trend chart
      market_change_alerts.py   latest-vs-previous alert candidates and guard rows
      historical_signal_lab.py  simple historical signal cohorts and same-date baselines
      value_list.py             searchable full gpovalues catalog
    components/
      data.py                   cached Streamlit loaders; model uses st.cache_resource
      layout.py                 refresh button, metric cards, footer links
      styling.py                shared monochrome visual system
      views.py                  page renderers and chart/table UI
    assets/strawhat_favicon.png favicon
  data/
    raw/gpo_market_dataset.json curated tier input
    snapshots/
      gpovalues_{date}.csv      dated public API pulls
      tier_reference_{date}.csv dated parses of the curated tier JSON
  outputs/
    feature_matrix_master.csv   generated merged matrix for dashboard/CLI fallback
  scripts/
    run_ingest_gpovalues.py     thin wrapper around primary ingestion
    run_ingest_tier.py          thin wrapper around tier parser
    run_feature_build.py        thin wrapper around merge output
  src/
    config/settings.py          paths, API URL/user-agent, ordinal encodings
    market_signals/
      analysis/
        market_change_alerts.py   latest-vs-previous snapshot alert candidates
        historical_signal_lab.py raw-snapshot historical signal evaluation
      ingest/
        pull_gpovalues_snapshot.py fetch live API, flatten payload, write dated snapshot
        parse_tier_dataset.py      flatten tier JSON, derive structural columns, write dated snapshot
      features/
        build_feature_matrix.py    merge latest gpovalues + tier reference by name then alias
      models/
        trend_model.py             first-to-last snapshot deltas, guarded by MIN_SNAPSHOTS
        value_regression.py        structural value model, RF fallback, estimates, SHAP explanations
      evaluator/
        evaluate.py                Typer CLI for lookup/verdict/trend
      decisions/
        log.py                     private decision rows, resale updates, read-only marked outcomes
        cli.py                     Typer commands for the private decision log
      utils/                       currently empty placeholder package
  tests/
    fixtures/                  trimmed real API/model fixtures
    test_*.py                  offline tests; no live network calls
```

Data flow:

1. `pull_gpovalues_snapshot.py` writes dated real-price snapshots.
2. `parse_tier_dataset.py` writes dated structural tier-reference snapshots.
3. `build_feature_matrix.py` merges the latest of each using exact normalized
   item name, then gpovalues shortcut vs tier alias. Unmatched gpovalues rows
   are kept and flagged.
4. `trend_model.py` reads all gpovalues snapshots for historical movement.
5. `market_change_alerts.py` uses raw dated gpovalues snapshots to compare the
   latest snapshot with the immediately previous available snapshot. It emits
   review-lead alerts and explicit guard rows for duplicate identities, missing
   comparison items, insufficient item history, stale snapshots, and invalid or
   extreme values. It also reports market-wide published-value context for
   items matched unambiguously across the comparison window with valid positive
   values: comparable item count, median percent change, and rose/fell/flat
   counts. This context is descriptive only and does not infer causes.
6. `historical_signal_lab.py` uses only raw dated gpovalues snapshots to
   evaluate prior 7-day value movement, prior 7-day demand-ratio movement, and
   confidence categories against later published estimates.
7. `value_regression.py` trains on medium/high-confidence rows, predicts
   `log(value)`, converts final predictions back with `exp`, and uses SHAP
   TreeExplainer on the selected RandomForestRegressor to explain model
   predictions in log-space.
8. The CLI and Streamlit dashboard reuse these package modules instead of
   duplicating parsing, matching, trend, or verdict logic.

The repository contains accumulated dated snapshots under `data/snapshots/`.
Check the snapshot filenames for the current coverage window instead of
treating this document as the source of truth for a fixed latest date.

`.github/workflows/daily_ingest.yml` defines the scheduled data-refresh job at
the Git root. It runs at `0 1 * * *` (1:00 AM UTC / 9:00 AM Asia/Manila) and
can also be started manually with `workflow_dispatch`. The job runs from the
`item-market-signals/` working directory on Python 3.9, installs dependencies
and the editable package, then runs:

1. `python scripts/run_ingest_gpovalues.py`
2. `python scripts/run_ingest_tier.py`
3. `python scripts/run_feature_build.py`

The final step stages `data/snapshots/*.csv` and `outputs/*.csv`, commits them
as `github-actions[bot]` with `Update daily market snapshots` only when there
are staged changes, and pushes that commit. Hosted dashboard deploy/reload
behavior is separate from this data commit workflow and should be verified in
the hosting environment.

## Current phase

The core product is built and usable on `main`: ingestion, feature merging,
CLI evaluator, dashboard, trend views, Market Change Alerts, Value List,
footer/navigation polish, Trade Simulator, structural value regression,
model-derived estimates, anomaly diagnostics, Historical Signal Lab, and SHAP
explainability are implemented.

Current work is validation, packaging, and model-quality refinement:

- monitor scheduled ingest commits and verify hosted deploy/reload behavior
  separately
- improve structural feature quality, especially prestige/item-family signals
- refine model-insight language so regression output never masquerades as
  observed value
- upgrade trend modeling once there is enough long-run history for a real
  slope/forecast instead of first-to-last deltas
- prepare portfolio screenshots/write-up around methodology and limitations

## Design principles

- **Every uncertainty-bearing calculation has an explicit guard and says so out
  loud.** Trend requires `MIN_SNAPSHOTS`; low-confidence values are caveated;
  model estimates are labeled as model-derived.
- **Observed market values beat model-derived estimates.** The regression model
  is structural context and low-confidence/tier-only support, not a replacement
  for gpovalues.
- **Log-scale model math must stay honest.** The value model predicts
  `log(value)`. Display final predictions in value units if useful, but show
  SHAP contributions as relative log-space pushes, not per-feature currency
  deltas.
- **Network fetch and parsing logic stay separated.** Tests call pure flattening
  functions against fixtures, never the live gpovalues API.
- **Unmatched/ambiguous items are flagged, never silently guessed.** Avoid fuzzy
  matching unless the output clearly marks it and the behavior is discussed.
- **Marked outcomes are published-estimate changes, not profit.** The
  decision-log marked outcome path uses the first snapshot on or after
  decision date plus 14 days that contains one exact saved item identity match.
  It does not overwrite `marked_value*` or resale fields, and it shows
  `pending` or `no later snapshot` rather than substituting another item or
  hiding the gap between target and actual snapshot dates.
- **Historical Signal Lab is exploratory signal evaluation, not a trading
  backtest.** It reads raw `gpovalues_*.csv` snapshots only, never the private
  decision log and never the latest merged feature matrix for past features.
  Prior value change requires the same item exactly seven calendar days
  earlier. Demand-ratio change is an absolute ratio delta and is shown only
  when both ratios are numeric. Outcomes target 14 calendar days later and may
  use the first exact same-item snapshot up to three days after target. The
  default `as_of` date is the latest available gpovalues snapshot, making file-
  backed reports reproducible. Future targets are `pending`; arrived targets
  with an incomplete three-day outcome window are `awaiting_window` unless a
  valid outcome is already found; completed windows without a valid outcome are
  confirmed missing. Repeated daily rows are correlated item-date observations,
  not independent trades.
- **Market Change Alerts are review leads, not recommendations.** They compare
  latest-vs-previous snapshots using simple documented rules: value movement
  requires both a 25% and 1,000-unit change, demand-ratio movement requires a
  1.00 point absolute change, and `trade_count` activity requires both a 50%
  and 100-count change. Each item needs at least three unambiguous snapshot
  dates before a valid metric can fire. `trade_count` can decrease and must be
  described as an activity indicator, not daily volume or new trades.
- **Snapshots are dated and never overwritten.** This is what makes historical
  trend analysis possible.
- **Dashboard code is presentation-only.** Pages/components import package
  modules for data logic. Do not duplicate parser, matching, model, or verdict
  behavior in Streamlit for convenience.
- **`src` layout, single package root.** `config` and `market_signals` live
  under `src/`. `pip install -e .` makes them importable. Only `conftest.py`
  and thin script wrappers should compensate for a missing editable install.

## Known gotchas

- **There are two requirements files.** Keep repo-root `requirements.txt` and
  `item-market-signals/requirements.txt` in sync when adding dependencies.
- **Git root vs project folder is easy to mix up.** The repository root is
  `GPO_Market/`; the active Python project is `GPO_Market/item-market-signals/`.
  Most commands should be run after `cd item-market-signals`.
- **Typer collapses a single-command app.** Invoke the CLI as
  `python -m market_signals.evaluator.evaluate "Item Name"` with no `check`
  subcommand.
- **gpovalues names and tier-list names do not always match.** The merge tries
  exact name, then shortcut/alias. Avoid silent fuzzy matching on item names.
- **Some environments block live network.** Fixture-based tests are the source
  of truth for offline verification; live ingestion needs network access.
- **Streamlit caches can look stale.** Use the page `Refresh data` button after
  running ingestion/build scripts. For hosted Streamlit Cloud, pushes may still
  require a manual app reboot before dependency/runtime changes take effect.
- **Local Python and hosted Python may differ.** This machine currently runs
  Python 3.9; Streamlit Cloud may use a much newer runtime such as Python 3.14.
  Dependency issues can be runtime-specific, so verify both when changing
  packages like Streamlit, sklearn, Plotly, or SHAP.
- **Plotly `titlefont` is deprecated.** Use nested title dictionaries such as
  `xaxis={"title": {"text": "...", "font": {...}}}` instead of old
  `titlefont` keys.
- **Value-regression SHAP explains encoded model features.** Categorical values
  appear as one-hot feature labels like `category_X` or `obtainability_Y`.
  Interpret them as model drivers in log-space, not literal money deltas.
