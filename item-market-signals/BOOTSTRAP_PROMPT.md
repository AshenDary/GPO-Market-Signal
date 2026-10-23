# Bootstrap prompt

Copy everything below the line into Codex (or any coding agent) when
setting up this repo fresh, or whenever you've dropped new/loose files in
and want them sorted into place. Keep this file up to date if the
architecture changes -- it's meant to replace re-explaining the project
from scratch each session.

---

From the Git root `GPO_Market/`, work in the Python project directory
`item-market-signals/`. Read `item-market-signals/CONTEXT.md` and
`item-market-signals/SKILLS.md` before doing anything else. They contain the
full architecture, data flow, module responsibilities, file placement rules,
and coding conventions for this project. Don't ask me to re-explain the
project -- everything you need is in those two files. If they're missing, stop
and tell me instead of guessing at the structure.

**Task 1 -- Arrange loose files.**
Check the Git root, the `item-market-signals/` project directory, and any
`_inbox/`-style staging folder for files that aren't yet in their correct
location per CONTEXT.md's architecture and SKILLS.md's file placement table.
For each loose file:
- Read its docstring/imports to identify what it does.
- Match it against SKILLS.md's file placement rules (ingestion ->
  `src/market_signals/ingest/`, merging -> `features/`, historical snapshot
  evaluation -> `analysis/`, predictive/structural models -> `models/`, current
  lookup CLI -> `evaluator/`, private decision logic and CLI -> `decisions/`,
  tests -> `tests/`, fixtures -> `tests/fixtures/`, thin runners -> `scripts/`,
  raw data -> `data/raw/`).
- Move it to the correct path, renaming only if the existing file at that
  path has a clearly different (non-conflicting) purpose -- if there's a
  real naming collision, stop and ask rather than overwriting.
- If you can't confidently tell where a file belongs, leave it in place and
  list it as "unresolved" in your final report -- don't guess.

**Task 2 -- Fill structural gaps.**
Create any missing directories or empty `__init__.py` files needed to match
the package structure in CONTEXT.md. Don't overwrite any file that already
matches the described structure and already has content.

**Task 3 -- Environment setup and offline verification.**
Run these from `item-market-signals/` in order. Stop and report immediately if
any step fails, rather than continuing past a failure:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python -m market_signals.evaluator.evaluate "Prestige Candy Cane"
# Optional dashboard check:
streamlit run dashboard/app.py
# Or double-click run_dashboard.command / run_dashboard.bat
```

Do not run live ingestion as routine setup. It requires network access and
writes dated snapshot/output files. Only run the data refresh pipeline when the
task explicitly asks for refreshed market data:

```
python scripts/run_ingest_gpovalues.py
python scripts/run_ingest_tier.py
python scripts/run_feature_build.py
```

If a requested live refresh cannot reach `gpovalues.com`, say so explicitly and
leave the existing committed snapshots in place.

**Task 4 -- Report back.**
Summarize: which files you moved and to where, anything left unresolved,
whether `pytest` passed (paste the failure if not), the final evaluator
command's output, and whether you intentionally ran any live data refresh.
