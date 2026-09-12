"""Weekly race-prediction workflow.

Steps every time you run it:
  1. Refresh season schedule from cf.nascar.com so newly-scheduled races land
     in races.parquet.
  2. For every incomplete race in the current season, pull the latest
     weekend-feed (entry list, qualifying/practice results if available),
     loopstats, and lap-times. Update parquets in place.
  3. Auto-detect the next upcoming race (or use --race-id to override).
  4. Retrain the model on all completed races.
  5. Fit per-track-type temperature.
  6. Predict the target race and write predictions/{race_id_short}.csv plus
     a printed summary.

Usage:
  python -m src.cli.predict_next                       # auto-detect next race
  python -m src.cli.predict_next --race-id 2026-5625   # explicit target
  python -m src.cli.predict_next --seasons 2026 2025   # which seasons to refresh
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build_features import build_features
from ..models import distribution as dist
from ..models.calibrate import RaceScoresGT
from ..models.gbm_ranker import GBMEnsemble
from ..scrape.nascar_com import fetch_race_detail, fetch_season_index, track_type_for
from ..scrape.nascar_laptimes import fetch_laptimes
from ..scrape.nascar_loop import fetch_loopstats

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
PREDICTIONS = REPO_ROOT / "predictions"

TIGHT_REG = {"num_leaves": 31, "min_data_in_leaf": 25, "lambda_l2": 2.0}
ALPHA = 0.85
N_SAMPLES = 20_000
N_ENSEMBLE = 15   # more seeds = less run-to-run variance
HAZARD_BY_TYPE = {"superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
                  "road": 0.04, "unique": 0.07}


# --------------------------------------------------------------------------- #
# Data refresh
# --------------------------------------------------------------------------- #
def _refresh_race(stub, races_df, entries_rows, session_rows, loop_rows, laptime_rows,
                  existing_races):
    """Fetch weekend-feed + loop + laptimes for one race and stage rows to append.

    Returns the updated race_row dict (for merging into races.parquet).
    """
    year = stub.season
    tt = track_type_for(stub.track_name)
    race_id_short = f"{year}-{stub.race_id}"
    try:
        detail = fetch_race_detail(stub)
    except Exception as e:
        log.warning("weekend-feed failed for %s: %s", stub.race_name, e)
        return None

    race_row = {
        "race_id_short": race_id_short,
        "season": year,
        "date": stub.race_date,
        "race_id_nascar": stub.race_id,
        "race_name": stub.race_name,
        "track_id": stub.track_id,
        "track_name": stub.track_name,
        "track_type": tt,
        "restrictor_plate": stub.restrictor_plate,
        "scheduled_laps": stub.scheduled_laps,
        "actual_laps": stub.actual_laps,
        "stage_1_laps": stub.stage_1_laps,
        "stage_2_laps": stub.stage_2_laps,
        "stage_3_laps": stub.stage_3_laps,
        "scheduled_distance": detail.scheduled_distance,
        "actual_distance": detail.actual_distance,
        "cars_in_field": detail.number_of_cars_in_field,
        "pole_driver_id": detail.pole_winner_driver_id,
        "pole_speed": detail.pole_winner_speed,
        "lead_changes": detail.number_of_lead_changes,
        "cautions": detail.number_of_cautions,
        "caution_laps": detail.number_of_caution_laps,
        "average_speed": detail.average_speed,
        "total_race_time": detail.total_race_time,
        "margin_of_victory": detail.margin_of_victory,
        "winner_driver_id": detail.winner_driver_id,
        "playoff_round": detail.playoff_round,
    }

    entries = detail.entries.copy()
    if not entries.empty:
        entries.insert(0, "race_id_short", race_id_short)
        entries.insert(0, "season", year)
        entries.insert(0, "date", pd.Timestamp(stub.race_date))
        entries.insert(3, "track_type", tt)
        entries_rows.append(entries)

    sessions = detail.sessions.copy()
    if not sessions.empty:
        sessions.insert(0, "race_id_short", race_id_short)
        sessions.insert(0, "season", year)
        sessions.insert(0, "date", pd.Timestamp(stub.race_date))
        sessions.insert(3, "track_type", tt)
        session_rows.append(sessions)

    # Loop stats (only exist after race is run).
    try:
        loop_df = fetch_loopstats(year, stub.race_id)
        if not loop_df.empty:
            loop_df["race_id_short"] = race_id_short
            loop_df["season"] = year
            loop_df["date"] = pd.Timestamp(stub.race_date)
            loop_rows.append(loop_df)
    except Exception as e:
        log.warning("loopstats failed for %s: %s", stub.race_name, e)

    # Lap-times (only after race).
    try:
        lt_df = fetch_laptimes(year, stub.race_id)
        if not lt_df.empty:
            lt_df["race_id_short"] = race_id_short
            lt_df["season"] = year
            laptime_rows.append(lt_df)
    except Exception as e:
        log.warning("laptimes failed for %s: %s", stub.race_name, e)

    return race_row


def refresh_data(seasons: list[int], target_year: int) -> None:
    """Refresh races, entries, sessions, loopstats, laptimes for `seasons`.

    Uses a merge strategy: existing rows are preserved, new/updated rows for
    races in `seasons` are overwritten.
    """
    log.info("refreshing seasons=%s", seasons)
    races_existing = _read_or_empty("races.parquet")
    entries_existing = _read_or_empty("entries.parquet")
    sessions_existing = _read_or_empty("sessions.parquet")
    loopstats_existing = _read_or_empty("loopstats.parquet")
    laptimes_existing = _read_or_empty("laptimes.parquet")

    race_rows: list[dict] = []
    entries_rows: list[pd.DataFrame] = []
    session_rows: list[pd.DataFrame] = []
    loop_rows: list[pd.DataFrame] = []
    laptime_rows: list[pd.DataFrame] = []

    for year in seasons:
        try:
            stubs = fetch_season_index(year)
        except Exception as e:
            log.exception("season index failed for %s: %s", year, e)
            continue
        log.info("  %s: %s races on schedule", year, len(stubs))
        for stub in stubs:
            existing_race = races_existing[
                races_existing["race_id_short"] == f"{year}-{stub.race_id}"
            ] if not races_existing.empty else pd.DataFrame()
            # Skip fully-completed races unless it's the target year (small quick refresh).
            if not existing_race.empty:
                existing_entries = entries_existing[
                    entries_existing["race_id_short"] == f"{year}-{stub.race_id}"
                ] if not entries_existing.empty else pd.DataFrame()
                is_complete = len(existing_entries) >= 20  # ~full field
                if is_complete and year != target_year:
                    continue
                if is_complete and year == target_year and stub.race_date < date.today():
                    continue
            row = _refresh_race(stub, races_existing, entries_rows, session_rows,
                                loop_rows, laptime_rows, races_existing)
            if row is not None:
                race_rows.append(row)
            time.sleep(0.3)

    # Merge and dedupe by race_id_short.
    new_races = pd.DataFrame(race_rows)
    races = _merge_dedupe(races_existing, new_races, key="race_id_short")
    entries = _merge_dedupe(
        entries_existing,
        pd.concat(entries_rows, ignore_index=True) if entries_rows else pd.DataFrame(),
        key=["race_id_short", "driver"],
    )
    sessions = _merge_dedupe(
        sessions_existing,
        pd.concat(session_rows, ignore_index=True) if session_rows else pd.DataFrame(),
        key=["race_id_short", "run_id", "driver_name"],
    )
    loopstats = _merge_dedupe(
        loopstats_existing,
        pd.concat(loop_rows, ignore_index=True) if loop_rows else pd.DataFrame(),
        key=["race_id_short", "driver_id"],
    )
    laptimes = _merge_dedupe(
        laptimes_existing,
        pd.concat(laptime_rows, ignore_index=True) if laptime_rows else pd.DataFrame(),
        key=["race_id_short", "driver_id", "lap"],
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    races.to_parquet(PROCESSED / "races.parquet", index=False)
    entries.to_parquet(PROCESSED / "entries.parquet", index=False)
    sessions.to_parquet(PROCESSED / "sessions.parquet", index=False)
    loopstats.to_parquet(PROCESSED / "loopstats.parquet", index=False)
    laptimes.to_parquet(PROCESSED / "laptimes.parquet", index=False)
    log.info("refresh done: %s races, %s entries, %s sessions, %s loop, %s laptime",
             len(races), len(entries), len(sessions), len(loopstats), len(laptimes))


def _read_or_empty(name: str) -> pd.DataFrame:
    p = PROCESSED / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _merge_dedupe(old: pd.DataFrame, new: pd.DataFrame, key) -> pd.DataFrame:
    if new.empty:
        return old
    if old.empty:
        return new
    combined = pd.concat([old, new], ignore_index=True)
    return combined.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def find_next_race(races: pd.DataFrame, entries: pd.DataFrame) -> str:
    """Return race_id_short of the next race we should predict.

    A race is 'next' if it has entries in the parquet but no finishing
    positions yet (entries with finish_pos=0 or is_dnf=False and no
    laps_completed) AND its date is not in the past.
    """
    if races.empty:
        raise RuntimeError("races.parquet is empty; run refresh first")
    races = races.copy()
    races["date"] = pd.to_datetime(races["date"])

    # Group entries by race_id_short: sum of finish_pos > 0 = completed race.
    if entries.empty:
        candidates = races
    else:
        e = entries.copy()
        e["completed"] = pd.to_numeric(e["finish_pos"], errors="coerce").fillna(0) > 0
        summary = e.groupby("race_id_short")["completed"].max().reset_index()
        completed_ids = set(summary[summary["completed"]]["race_id_short"])
        candidates = races[~races["race_id_short"].isin(completed_ids)]

    upcoming = candidates[candidates["date"] >= pd.Timestamp(date.today())]
    if upcoming.empty:
        raise RuntimeError("no upcoming races found; either refresh or check schedule")
    upcoming = upcoming.sort_values("date")
    return upcoming.iloc[0]["race_id_short"]


def _log_clip(p, floor=1e-9):
    return np.log(np.clip(p, floor, 1.0 - floor))


def _fit_per_type_T(model, train_features, hazard_by_type,
                    T_grid=(0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)):
    """Fit per-track-type matchup-optimal T on all training races (skip 30-race burn-in)."""
    race_order = (
        train_features.groupby("race_id_short")["date"].first().sort_values().index.tolist()
    )
    val_ids = race_order[30:]
    by_type: dict[str, list[RaceScoresGT]] = {}
    for rid in val_ids:
        sub = train_features[train_features["race_id_short"] == rid]
        raw = model.predict_scores(sub)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                / (sub["pl_effective"].std() + 1e-9))
        b = ALPHA * gbm_z + (1 - ALPHA) * pl_z
        b = b / max(b.std(), 1e-6)
        tt = sub["track_type"].iloc[0]
        by_type.setdefault(tt, []).append(RaceScoresGT(
            scores=b, finishes=sub["finish_pos"].to_numpy(),
            is_dnf=sub["is_dnf"].to_numpy(),
            hazards=np.full(len(sub), hazard_by_type.get(tt, 0.08)),
        ))

    from ..models.calibrate import _matchup_ll_for_race
    result: dict[str, float] = {}
    for tt, rs in by_type.items():
        if len(rs) < 6:
            result[tt] = 2.0
            continue
        rng = np.random.default_rng(0)
        best_T, best_ll = None, float("inf")
        for T in T_grid:
            losses = []
            for r in rs:
                positions = dist.sample_finishing_orders(
                    r.scores / T, r.hazards, n_samples=1500, rng=np.random.default_rng(0),
                )
                M = dist.matchup_probs(positions)
                losses.append(_matchup_ll_for_race(M, r.finishes, r.is_dnf))
            mean_ll = float(np.mean(losses))
            if mean_ll < best_ll:
                best_ll = mean_ll
                best_T = float(T)
        result[tt] = best_T
    return result


def american(p):
    if p <= 0 or p >= 1:
        return "n/a"
    dec = 1 / p
    return f"+{int(round((dec - 1) * 100))}" if dec >= 2.0 else f"-{int(round(100 / (dec - 1)))}"


def predict_race(race_id_short: str, n_ensemble: int = N_ENSEMBLE,
                 drop_driver: bool = False) -> tuple:
    races = pd.read_parquet(PROCESSED / "races.parquet")
    entries = pd.read_parquet(PROCESSED / "entries.parquet")
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    loopstats = pd.read_parquet(PROCESSED / "loopstats.parquet")
    laptimes = pd.read_parquet(PROCESSED / "laptimes.parquet")

    race_row = races[races["race_id_short"] == race_id_short]
    if race_row.empty:
        raise RuntimeError(f"race {race_id_short} not in races.parquet")
    tt = race_row.iloc[0]["track_type"]

    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    # Only train on races that have real outcomes (finish_pos > 0).
    train = features[(features["race_id_short"] != race_id_short) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == race_id_short].reset_index(drop=True)
    if target.empty:
        raise RuntimeError(f"no entries for {race_id_short}; refresh weekend-feed")

    log.info("training %s-model GBM ensemble on %s races (%s rows) ...",
             n_ensemble, train["race_id_short"].nunique(), len(train))
    drop = ["driver"] if drop_driver else None
    model = GBMEnsemble(drop_features=drop)
    model.fit(train, n_estimators=n_ensemble, **TIGHT_REG)

    log.info("fitting per-track-type T ...")
    T_by_type = _fit_per_type_T(model, train, HAZARD_BY_TYPE)
    T = T_by_type.get(tt, 1.0)
    log.info("per-type T: %s -> using T = %s for %s", T_by_type, T, tt)

    raw = model.predict_scores(target)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
            / (target["pl_effective"].std() + 1e-9))
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)

    haz = np.full(len(target), HAZARD_BY_TYPE.get(tt, 0.08))
    rng = np.random.default_rng(42)
    positions = dist.sample_finishing_orders(blended / T, haz, n_samples=N_SAMPLES, rng=rng)
    winp = dist.win_probs(positions)
    t5 = dist.topn_probs(positions, 5)
    t10 = dist.topn_probs(positions, 10)

    out = pd.DataFrame({
        "driver": target["driver"].values,
        "team": target["team"].values,
        "make": target["manufacturer"].values,
        "win_prob": winp,
        "win_odds": [american(p) for p in winp],
        "top5_prob": t5,
        "top10_prob": t10,
    }).sort_values("win_prob", ascending=False).reset_index(drop=True)
    out["win_prob"] = out["win_prob"].round(4)
    out["top5_prob"] = out["top5_prob"].round(3)
    out["top10_prob"] = out["top10_prob"].round(3)
    return out, T


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--race-id", help="race_id_short to predict (e.g. 2026-5625)")
    p.add_argument("--seasons", nargs="+", type=int, default=[date.today().year])
    p.add_argument("--no-refresh", action="store_true", help="skip data refresh")
    p.add_argument("--n-ensemble", type=int, default=N_ENSEMBLE,
                   help=f"number of GBM models in ensemble (default {N_ENSEMBLE})")
    p.add_argument("--drop-driver", action="store_true",
                   help="drop the 'driver' categorical feature (test model without it)")
    args = p.parse_args(argv)

    if not args.no_refresh:
        target_year = max(args.seasons)
        refresh_data(args.seasons, target_year)

    races = pd.read_parquet(PROCESSED / "races.parquet")
    entries = _read_or_empty("entries.parquet")
    race_id = args.race_id or find_next_race(races, entries)
    race_row = races[races["race_id_short"] == race_id].iloc[0]
    print(f"\nTarget race: {race_row['race_name']} at {race_row['track_name']} "
          f"({race_row['track_type']}), date {race_row['date']}, "
          f"playoff round {int(race_row.get('playoff_round', 0) or 0)}")

    out, T = predict_race(race_id, n_ensemble=args.n_ensemble, drop_driver=args.drop_driver)

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    csv_path = PREDICTIONS / f"{race_id}.csv"
    out.to_csv(csv_path, index=False)

    print(f"\n=== PREDICTIONS (T = {T}) ===\n")
    print(out.to_string(index=False))
    print(f"\nWritten to: {csv_path}")
    print(f"Sum of win_probs: {out['win_prob'].sum():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
