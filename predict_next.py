"""Comprehensive prediction for the next race.

Uses the shared honest-calibration pipeline in
src.models.predict_pipeline — same setup the walk-forward-pure backtest uses:
  - leave-one-race-out CV to fit T
  - per-driver hazards on both val and target
  - single calibrated temperature T handles matchup + top-N

Outputs:
  1. Winner probabilities   (P(finish == 1))
  2. Top 5 probabilities    (P(finish <= 5))
  3. Top 10 probabilities   (P(finish <= 10))
  4. Winning manufacturer probabilities
  5. Top-of-manufacturer probability per driver
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, DNF_DISPERSION,
    DAMAGE_PENALTY, DAMAGE_PENALTY_BY_TYPE,
    CV_N_ESTIMATORS, VAL_CAP_PER_TYPE,
    per_driver_hazards, per_driver_damage_hazards,
)
from src.models.predict_pipeline import calibrate_and_sample


# Bristol Motor Speedway — Bass Pro Shops Night Race (playoff, first round cutoff).
TARGET_NAME = "Bass Pro Shops"
TARGET_DATE = "2026-09-19"


def main():
    from src.features.build_features import build_features

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...")
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    # Find target race.
    target_ts = pd.Timestamp(TARGET_DATE)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(TARGET_NAME, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    if r.empty:
        print(f"Race not found in parquet — check TARGET_NAME/TARGET_DATE")
        return
    rid = r.iloc[0]["race_id_short"]
    target_date_ts = r.iloc[0]["date"]
    from src.features.tracks import resolve_track_type
    tt = resolve_track_type(r.iloc[0].get("track_name", ""),
                            fallback=r.iloc[0]["track_type"])
    print(f"Target: {r.iloc[0]['race_name']}  ({tt}, date={target_date_ts.date()})")

    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == rid].reset_index(drop=True)
    if target.empty:
        print("No entry list for target race yet — need to scrape entry list first.")
        return
    print(f"Field size: {len(target)} drivers")

    print("Training + calibrating (leave-one-race-out CV — this is the slow part)...")
    sr = calibrate_and_sample(
        train, target, tt,
        tight_reg=TIGHT_REG, alpha=ALPHA, n_samples=N_SAMPLES,
        hazard_lookup=HAZARD,
        per_driver_hazards_fn=per_driver_hazards,
        dnf_dispersion_by_type=DNF_DISPERSION,
        per_driver_damage_hazards_fn=per_driver_damage_hazards,
        damage_penalty=DAMAGE_PENALTY,
        damage_penalty_by_type=DAMAGE_PENALTY_BY_TYPE,
        cv_n_estimators=CV_N_ESTIMATORS,
        val_cap_per_type=VAL_CAP_PER_TYPE,
    )

    # positions is (N_SAMPLES, n_drivers) — 1-indexed (1=winner).
    win_prob = (sr.positions == 1).sum(axis=0) / N_SAMPLES
    top5_prob = (sr.positions <= 5).sum(axis=0) / N_SAMPLES
    top10_prob = (sr.positions <= 10).sum(axis=0) / N_SAMPLES

    drivers = target["driver"].tolist()
    makes = (target["manufacturer"].tolist() if "manufacturer" in target.columns
             else target["make"].tolist() if "make" in target.columns
             else ["Unknown"] * len(drivers))
    n_drivers = len(drivers)

    # 4. Winning manufacturer.
    make_win_counts = defaultdict(int)
    for s in range(N_SAMPLES):
        winner_idx = int(np.argmin(sr.positions[s]))
        make_win_counts[makes[winner_idx]] += 1
    manuf_win = {m: c / N_SAMPLES for m, c in make_win_counts.items()}

    # 5. Top-of-manufacturer.
    by_make_indices = defaultdict(list)
    for i, m in enumerate(makes):
        by_make_indices[m].append(i)
    top_of_make = np.zeros(n_drivers)
    for m, idxs in by_make_indices.items():
        if len(idxs) == 1:
            top_of_make[idxs[0]] = 1.0
            continue
        sub_pos = sr.positions[:, idxs]
        best_within = np.argmin(sub_pos, axis=1)
        for k, i in enumerate(idxs):
            top_of_make[i] = float(np.mean(best_within == k))

    df = pd.DataFrame({
        "driver": drivers,
        "make": makes,
        "win_prob": win_prob,
        "top5_prob": top5_prob,
        "top10_prob": top10_prob,
        "top_of_make": top_of_make,
    })

    def american_from_prob(p):
        if p <= 0: return "—"
        if p >= 1: return "-inf"
        dec = 1 / p
        if dec >= 2:
            return f"+{int((dec - 1) * 100)}"
        return f"-{int(100 / (dec - 1))}"

    df["win_odds"] = df["win_prob"].apply(american_from_prob)

    print("\n" + "=" * 90)
    print("WINNER PROBABILITIES (Top 20)")
    print("=" * 90)
    print(df.sort_values("win_prob", ascending=False).head(20)[
        ["driver", "make", "win_prob", "win_odds"]
    ].to_string(index=False, formatters={"win_prob": "{:.3%}".format}))

    print("\n" + "=" * 90)
    print("TOP 5 PROBABILITIES (Top 20 drivers)")
    print("=" * 90)
    print(df.sort_values("top5_prob", ascending=False).head(20)[
        ["driver", "make", "top5_prob"]
    ].to_string(index=False, formatters={"top5_prob": "{:.1%}".format}))

    print("\n" + "=" * 90)
    print("TOP 10 PROBABILITIES (Top 20 drivers)")
    print("=" * 90)
    print(df.sort_values("top10_prob", ascending=False).head(20)[
        ["driver", "make", "top10_prob"]
    ].to_string(index=False, formatters={"top10_prob": "{:.1%}".format}))

    print("\n" + "=" * 90)
    print("WINNING MANUFACTURER PROBABILITIES")
    print("=" * 90)
    print(pd.DataFrame([
        {"make": m, "prob": p} for m, p in sorted(manuf_win.items(), key=lambda x: -x[1])
    ]).to_string(index=False, formatters={"prob": "{:.1%}".format}))

    print("\n" + "=" * 90)
    print("TOP-OF-MANUFACTURER PROBABILITIES (per make, sorted)")
    print("=" * 90)
    for make in sorted(df["make"].unique()):
        sub = df[df["make"] == make].sort_values("top_of_make", ascending=False)
        print(f"\n--- {make} ---")
        print(sub[["driver", "top_of_make"]].to_string(
            index=False, formatters={"top_of_make": "{:.1%}".format}))


if __name__ == "__main__":
    main()
