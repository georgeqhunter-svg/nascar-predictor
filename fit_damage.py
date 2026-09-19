"""Empirically fit damage-hazard rates per track type from ≤2025 data.

Damage day = a driver who did NOT DNF but finished substantially worse than
their pre-race PL rating suggests. Currently the model treats outcomes as
binary (clean race sample OR DNF-to-back). Real races produce a third bucket:
contact / speeding penalty / uncontrolled tire / mechanical limp — driver
continues but finishes 10-25 positions worse than pace suggests.

Two outputs per track type:
  damage_rate      — probability a driver has a damage day at that track type
  penalty_positions — median position-loss when damaged (for the sampler
                      to translate to a score-space penalty)

Uses ≤2025 entries only so 2026 backtest results don't inform the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DAMAGE_THRESHOLD_POSITIONS = 12  # finished 12+ positions below expectation


def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    # Historical only.
    hist_races = races[races["season"] <= 2025].copy()
    # Use the CURRENT track classification (tracks.py is authoritative). Any
    # tracks reclassified since parquet was written (Pocono/Indy: unique ->
    # intermediate) get their new type here so damage rates are fit per the
    # new bucketing.
    from src.features.tracks import resolve_track_type
    hist_races["current_tt"] = hist_races.apply(
        lambda r: resolve_track_type(r.get("track_name", ""),
                                     fallback=r["track_type"]),
        axis=1,
    )
    hist = entries[entries["race_id_short"].isin(hist_races["race_id_short"])]
    # Drop entries' stale track_type if present, then attach the current one.
    if "track_type" in hist.columns:
        hist = hist.drop(columns=["track_type"])
    hist = hist.merge(hist_races[["race_id_short", "current_tt"]],
                      on="race_id_short", how="left")
    hist = hist.rename(columns={"current_tt": "track_type"})
    hist = hist[hist["finish_pos"] > 0].copy()

    print(f"Fitting damage rates on {len(hist)} historical driver-races "
          f"across {hist_races['season'].nunique()} seasons.\n")

    # Proxy for "expected finish": qualifying position. Assumes drivers finish
    # roughly where they qualify absent damage. Not perfect (strategy, pit
    # cycles, tire cycles all shuffle) but the noise averages out over
    # thousands of driver-races.
    hist = hist[(hist["qual_pos"] > 0) & pd.notna(hist["qual_pos"])].copy()
    hist["qual_pos"] = hist["qual_pos"].astype(int)
    hist["damage_gap"] = hist["finish_pos"] - hist["qual_pos"]
    hist["is_damaged"] = (
        (~hist["is_dnf"].astype(bool))
        & (hist["damage_gap"] > DAMAGE_THRESHOLD_POSITIONS)
    ).astype(int)

    print("=" * 70)
    print("DAMAGE_HAZARD by track type (finished P > qual_pos + "
          f"{DAMAGE_THRESHOLD_POSITIONS}, NOT DNF)")
    print("=" * 70)
    print(f"{'track_type':<15} {'n':>7} {'damage_rate':>12} {'median_gap':>12} "
          f"{'p75_gap':>10} {'p90_gap':>10}")
    for tt, sub in hist.groupby("track_type"):
        if len(sub) < 100: continue
        rate = float(sub["is_damaged"].mean())
        damaged = sub[sub["is_damaged"] == 1]["damage_gap"]
        median = float(damaged.median()) if len(damaged) else 0.0
        p75 = float(damaged.quantile(0.75)) if len(damaged) else 0.0
        p90 = float(damaged.quantile(0.90)) if len(damaged) else 0.0
        print(f"{tt:<15} {len(sub):>7} {rate:>11.4f} "
              f"{median:>12.1f} {p75:>10.1f} {p90:>10.1f}")

    print()
    print("Suggested DAMAGE_HAZARD dict (paste into backtest_oddslogic_v5.py):")
    print()
    print("DAMAGE_HAZARD = {")
    for tt, sub in hist.groupby("track_type"):
        if len(sub) < 100: continue
        rate = float(sub["is_damaged"].mean())
        print(f'    "{tt}": {rate:.3f},')
    print("}")
    print()
    print("DAMAGE_PENALTY_MEDIAN = {  # median positions lost when damaged")
    for tt, sub in hist.groupby("track_type"):
        if len(sub) < 100: continue
        damaged = sub[sub["is_damaged"] == 1]["damage_gap"]
        median = float(damaged.median()) if len(damaged) else 0.0
        print(f'    "{tt}": {median:.1f},')
    print("}")


if __name__ == "__main__":
    main()
