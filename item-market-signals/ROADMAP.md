# Roadmap

gpovalues.com remains the source of truth for community-solved published
market values from observed trades. The project adds a private, local decision
log for buy/no-buy workflow evidence and actual resale outcomes, but keeps
those personal fields separate from published-value changes and model
diagnostics.

## Completed

| Phase | Goal | Status |
|---|---|---|
| 0. Setup | `src` package layout, installable project, offline tests | Done |
| 1. First real pull | gpovalues ingestion, tier parser, merged feature matrix | Done |
| 2. Snapshot accumulation | Dated gpovalues and tier snapshots for trend context | Done and ongoing |
| 3. Dashboard foundation | Streamlit app with routed pages and shared styling | Done |
| 4. Daily-use evaluator | CLI and Item Lookup with confidence bands and price verdicts | Done |
| 5. Dashboard polish | Start Here, footer links, Value List, top nav, refresh controls | Done |
| 6. Trade Simulator | Two-sided trade comparison with dialog-based add-item flow | Done |
| 7. Structural value regression | RandomForest fallback, low-confidence/tier-only estimates, anomaly diagnostics | Done |
| 8. SHAP explainability | TreeExplainer for the existing RandomForestRegressor, log-space contribution chart, additivity test | Done |
| 9. Scheduled data refresh | GitHub Actions workflow for daily/manual ingest, feature build, and data-output commits | Done |
| 10. Private decision log | Local buy/no-buy logging, actual resale updates, and read-only 14-day gpovalues marked outcomes | Done |
| 11. Historical review tools | Historical Signal Lab and Market Change Alerts for snapshot-based review leads | Done |

## Current focus

- Validate model-insight wording against real anomalous items so the UI explains
  model behavior without implying the model corrects gpovalues.
- Monitor scheduled ingest commits and keep hosted dashboard refresh/deploy
  behavior verified separately from the data workflow.
- Prepare the portfolio narrative: what gpovalues provides, what this project
  adds, where the model is useful, and where it should not be trusted.
- Keep decision-log language clear: read-only marked outcomes are changes in
  gpovalues' published estimates, while recorded resale fields are actual
  realized outcomes.
- Continue tightening dashboard ergonomics and responsive presentation as real
  use reveals friction.

## Next modeling work

1. **Trend quality pass**
   The current trend model is a first-snapshot to latest-snapshot delta. Upgrade
   it to a slope/regression-style trend only after enough long-run snapshot
   history exists to defend the result.

2. **Structural feature refinement**
   Improve features that the regression model currently treats too coarsely,
   especially prestige/item-family signals, unit-like rows such as bounty, and
   richer rarity/content context.

3. **Model validation write-up**
   Document the train/test split, log-scale target, RandomForest selection rule,
   residual/anomaly meaning, and SHAP additivity guarantee in portfolio language.

## Intentionally deferred

- **Real trend forecasting.** There are multiple snapshots now, but robust
  forecasting still needs more history and careful validation. Do not present
  first-to-last deltas as forecasts.
- **Hosted app reload/deploy guarantees.** The GitHub Actions workflow commits
  refreshed snapshots and outputs, but hosted dashboard reload behavior still
  depends on the deployment environment and should be verified there.
- **`is_prestige` / item-family feature engineering.** The model would likely
  benefit from explicit prestige/family flags, but those need careful parsing
  and tests so they do not become brittle string hacks.
- **Contextual glossary/popover UX.** Terms like confidence, demand ratio,
  structural residual, and SHAP log contribution could use inline explanations
  in the dashboard, but the core pages work without it.
- **Wiki/content feature ingestion.** PvP relevance, drop mechanics, update
  timing, and content source could enrich the model later. They are not needed
  for the current evaluator to work.
- **Discord bot wrapper.** Useful for daily ergonomics eventually, but the CLI
  and Streamlit dashboard cover the current use cases.
- **Decision-log analytics.** The private log now records buy/no-buy decisions,
  asking prices, purchase prices, and actual resale outcomes. Future work can
  summarize whether using the tool changed behavior, but those analytics must
  keep realized resale/profit separate from gpovalues published-estimate
  movement.
- **Historical signal expansion.** Historical Signal Lab and Market Change
  Alerts are implemented as review and evaluation tools. Future additions can
  broaden signals or baselines only when the output still labels them as
  published-value changes, not completed trades or realized returns.

## Definition of done for future additions

- User-facing model output is labeled as observed, model-derived, or diagnostic.
- Tests remain offline; live network calls stay outside tests.
- Dashboard additions reuse package modules rather than duplicating business
  logic in Streamlit.
- Any dependency added to `item-market-signals/requirements.txt` is also added
  to the repo-root `requirements.txt`.
