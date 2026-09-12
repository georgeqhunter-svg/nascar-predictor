# NASCAR Cup Race Prediction

Phase-1 backtest build for a NASCAR Cup Series prediction system, per the design doc at
`../nascar_prediction_design.md`.

**Scope:** Next Gen car era only (2022 → present). Cup Series. Head-to-head matchup and
race-winner markets. No code runs against real money until phase 1 clears the gate.

## Layout

```
src/scrape/        source-specific scrapers (racing-reference, loop data, weather, odds)
src/features/      feature builders (ratings, track similarity, pit, team dynamics)
src/models/        baseline / gbm / dl / ensemble
src/eval/          backtest, market benchmarking
src/cli/           entry points (backfill, train, predict-race)
data/raw/          cached raw HTML/JSON responses (git-ignored)
data/processed/    parquet tables (git-ignored)
configs/           model + feature YAML
```

## Quick start

```
pip install -r requirements.txt
python -m src.cli.backfill --seasons 2022 2023 2024 2025 2026
```

## Data source

Racing Reference (racing-reference.info) is the primary source. Scraper is polite:
1 req/sec, retries with backoff, raw HTML cached to disk so re-parsing costs nothing.
