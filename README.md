# Item Market Signals

Item Market Signals is a market intelligence dashboard and evaluator for the
Grand Piece Online trading economy. It combines gpovalues.com market data with
curated tier/rarity context, stores dated snapshots, and exposes both a
Streamlit dashboard and a CLI evaluator.

The active Python project lives in [`item-market-signals/`](item-market-signals/).

## What It Includes

- gpovalues.com API ingestion into dated snapshots
- tier/rarity reference parsing from a curated JSON file
- feature merging by exact item name, then shortcut/alias fallback
- CLI value lookup with asking-price verdicts
- private local decision log for real buy/no-buy checks, actual resale outcomes,
  and separate read-only gpovalues marked outcomes
- Streamlit dashboard with Start Here, Overview, Item lookup, Trade Simulator,
  Model Insights, Trend, Market Change Alerts, Historical Signal Lab, and
  Value List pages
- snapshot-based trend context
- daily market-change alert candidates for unusually large latest-vs-previous
  snapshot moves in published value, demand ratio, or `trade_count` activity
- structural value regression for low-confidence/tier-only estimates
- SHAP explanations for the RandomForestRegressor value model, displayed as
  log-space relative contributions
- offline tests built around saved fixtures

## Quick Start

```bash
cd item-market-signals
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Refresh the market data (network required; this writes dated snapshots and
generated outputs):

```bash
python scripts/run_ingest_gpovalues.py
python scripts/run_ingest_tier.py
python scripts/run_feature_build.py
```

Check one item from the CLI:

```bash
python -m market_signals.evaluator.evaluate "Prestige Candy Cane"
python -m market_signals.evaluator.evaluate "Candy Cane" --asking-price 300000
```

## Repository Layout

```text
GPO_Market/
  README.md
  requirements.txt                 # kept in sync with project requirements
  item-market-signals/
    README.md
    DATA_DICTIONARY.md
    CONTEXT.md
    ROADMAP.md
    SKILLS.md
    dashboard/
    data/
    outputs/
    scripts/
    src/
    tests/
```

See the project [`README`](item-market-signals/README.md) for full usage,
[`DATA_DICTIONARY.md`](item-market-signals/DATA_DICTIONARY.md) for field
meanings and a read-only walkthrough,
[`CONTEXT.md`](item-market-signals/CONTEXT.md) for architecture and gotchas,
and [`ROADMAP.md`](item-market-signals/ROADMAP.md) for what is done versus
intentionally deferred.
