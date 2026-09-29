# item-market-signals

Item Market Signals is a market intelligence dashboard and evaluator for the
Grand Piece Online trading economy. It pulls community-solved values from
gpovalues.com, enriches them with curated tier/rarity context, preserves dated
snapshots, and turns the data into practical item lookup, trade comparison,
trend, and model-diagnostic views.

The project is both a daily-use tool and a portfolio project. The guiding rule
is simple: observed market values are treated as observed values, model-derived
estimates are labeled as model-derived, and uncertainty is shown plainly.

## Features

- gpovalues.com API ingestion into dated `gpovalues_*.csv` snapshots
- curated tier/rarity JSON parsing into dated `tier_reference_*.csv` snapshots
- merged feature matrix using exact item-name match, then shortcut/alias match
- Typer CLI for quick fair-value and asking-price checks
- private local decision log for real buy/no-buy checks, actual resale
  outcomes, and read-only 14-day gpovalues marked outcomes
- Streamlit dashboard with:
  - **Start Here** - dashboard directory and signal explanations
  - **Overview** - coverage, confidence counts, tier/value scatter, most-traded items
  - **Item lookup** - one-item value, confidence band, asking-price verdict
  - **Trade Simulator** - compare items on both sides of a proposed trade
  - **Model Insights** - regression diagnostics, anomaly table, SHAP breakdowns
  - **Trend** - item value movement across dated snapshots
  - **Value List** - searchable full value catalog
- Structural value regression trained on medium/high-confidence rows for
  low-confidence and tier-only context
- SHAP TreeExplainer support for the selected RandomForestRegressor, shown as
  log-space relative feature contributions rather than fake currency deltas
- Offline tests using fixtures instead of live network calls

## Setup

Most commands should be run from this project folder:

```bash
cd item-market-signals
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The editable install makes both `config` and `market_signals` importable from
the `src/` layout.

Run tests:

```bash
pytest
```

Tests are offline. Live gpovalues ingestion needs network access, but parser,
feature, trend, simulator, and model logic are tested against fixtures.

## Run the pipeline

Pull current gpovalues data:

```bash
python scripts/run_ingest_gpovalues.py
```

Parse the curated tier reference:

```bash
python scripts/run_ingest_tier.py
```

Merge the latest snapshots into the dashboard/CLI feature matrix:

```bash
python scripts/run_feature_build.py
```

Each ingestion run writes a dated snapshot instead of overwriting the previous
one. Trend views become more useful as snapshot history accumulates.

## Use the CLI

```bash
python -m market_signals.evaluator.evaluate "Prestige Candy Cane"
python -m market_signals.evaluator.evaluate "Candy Cane" --asking-price 300000
```

Typer treats this as a single-command app, so there is no `check` subcommand in
the invocation.

## Private decision log

The decision log is local and private. It records real buy/no-buy decisions,
asking prices, snapshot-time gpovalues signals, and later actual resale
outcomes if you add them. The real CSV lives at
`data/decisions/decision_log.csv` and is ignored by Git; do not commit personal
trading decisions.

Log a buy or no-buy decision:

```bash
python scripts/run_decision_log.py log "Prestige Candy Cane" --asking-price 2900000 --decision buy --purchase-price 2850000
python scripts/run_decision_log.py log "Candy Cane" --asking-price 350000 --decision no-buy
```

Inspect or update the private log:

```bash
python scripts/run_decision_log.py list
python scripts/run_decision_log.py show <decision_id>
python scripts/run_decision_log.py marked-outcome <decision_id>
python scripts/run_decision_log.py record-resale <decision_id> --resale-price 3100000 --resale-date 2026-10-10
```

Resale fields are for actual completed sales only. Future marked gpovalues
values are separate fields and should not be described as realized profit.

The `marked-outcome` command is read-only. It calculates `decision_date + 14
days`, then uses the first available `gpovalues_YYYY-MM-DD.csv` snapshot on or
after that target date that contains exactly one saved item identity match. The
output reports the original gpovalues published estimate, the later published
estimate, percent change, target date, actual snapshot date, and days elapsed.
If the target date has not arrived it shows `pending`; if no later snapshot has
one unambiguous matching item, it shows `no later snapshot`. This is only a
change in gpovalues' published estimate, not profit and not evidence that a
trade was completed. Actual resale, when recorded, is displayed separately.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard uses cached data/model loaders. If you run ingestion scripts while
the app is open, use the page `Refresh data` button. On hosted Streamlit Cloud,
dependency/runtime changes may require a manual app reboot after pushing.

Local launcher files are also included:

- `run_dashboard.command` for macOS
- `run_dashboard.bat` for Windows

## File structure

```text
item-market-signals/
  README.md
  CONTEXT.md
  ROADMAP.md
  SKILLS.md
  pyproject.toml
  requirements.txt
  dashboard/
    app.py
    pages/
      guide.py
      overview.py
      lookup.py
      simulator.py
      model_insights.py
      trend.py
      value_list.py
    components/
      data.py
      layout.py
      styling.py
      views.py
    assets/strawhat_favicon.png
  data/
    decisions/decision_log.csv  # local/private, ignored by Git
    raw/gpo_market_dataset.json
    snapshots/
      gpovalues_*.csv
      tier_reference_*.csv
  outputs/
    feature_matrix_master.csv
  scripts/
    run_ingest_gpovalues.py
    run_ingest_tier.py
    run_feature_build.py
    run_decision_log.py
  src/
    config/settings.py
    market_signals/
      decisions/
      ingest/
      features/
      models/
      evaluator/
      utils/
  tests/
    fixtures/
    test_*.py
```

## Data and model notes

gpovalues.com is the primary source for solved market values and confidence
bands. The tier JSON is structural context only; it does not replace observed
market prices.

The value regression model predicts `log(value)`. The dashboard converts the
final model prediction back to normal value units for readability, but SHAP
feature contributions stay in log-space and are presented as relative drivers,
not exact per-feature money amounts.

## Current status

The ingestion pipeline, feature builder, CLI, dashboard, trade simulator,
snapshot trend views, value regression, and SHAP explainability are working.
The main deferred modeling work is richer trend forecasting once enough
long-term snapshot history exists and refinement of structural features such as
prestige/item-family signals.
