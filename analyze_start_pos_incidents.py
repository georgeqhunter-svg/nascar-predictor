"""Does starting position correlate with crash/DNF rates, and does that
correlation depend on track type?

For each track type, bin drivers by starting position and compute:
  - DNF rate (all causes)
  - Crash-DNF rate (Accident / DVP / Damage / Wreck)
  - Mechanical-DNF rate

If the crash-DNF rate falls sharply as you move to the front on road courses
or superspeedways, that's actionable — we'd add a start_pos × track_type
interaction feature and route it into per-driver hazard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.forward import ACCIDENT_STATUSES, MECHANICAL_STATUSES


BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40)]


def bucket_label(sp):
    for lo, hi in BUCKETS:
        if lo <= sp <= hi:
            return f"{lo:>2}-{hi:<2}"
    return "  ??  "


def main():
    entries = pd.read_parquet("data/processed/entries.parquet")
    races = pd.read_parquet("data/processed/races.parquet")
    print(f"entries cols: {list(entries.columns)}")
    print(f"races cols: {list(races.columns)}")

    # Derive track_type from track_name via TRACKS if not present.
    if "track_type" not in races.columns:
        from src.features.tracks import TRACKS
        slug_to_type = {slug.replace('_', ' ').lower(): t.track_type
                        for slug, t in TRACKS.items()}
        races["track_type"] = races["track_name"].astype(str).str.lower().map(slug_to_type)
    if "track_type" not in entries.columns:
        entries = entries.merge(
            races[["race_id_short", "track_type"]],
            on="race_id_short", how="left",
        )
    print(f"track_type dist: {entries['track_type'].value_counts().to_dict()}")
    entries = entries[
        (entries["finish_pos"] > 0) & (entries["start_pos"] > 0)
    ].copy()
    entries["start_bucket"] = entries["start_pos"].apply(bucket_label)
    entries["is_crash"] = entries["status"].isin(ACCIDENT_STATUSES).astype(int)
    entries["is_mech"] = entries["status"].isin(MECHANICAL_STATUSES).astype(int)
    entries["is_dnf"] = entries["is_dnf"].astype(int)

    for tt in ["road", "superspeedway", "short", "intermediate", "unique"]:
        sub = entries[entries["track_type"] == tt]
        if len(sub) < 100:
            continue
        print(f"\n{'=' * 72}")
        print(f"TRACK TYPE: {tt}  (n={len(sub)} driver-races)")
        print(f"{'=' * 72}")
        print(f"{'Start bucket':<14} {'n':>6} {'DNF %':>8} {'Crash %':>8} "
              f"{'Mech %':>8} {'Avg finish':>12}")
        print("-" * 72)
        for bucket_lo, bucket_hi in BUCKETS:
            b_label = f"{bucket_lo:>2}-{bucket_hi:<2}"
            b = sub[sub["start_bucket"] == b_label]
            if len(b) == 0: continue
            print(f"{b_label:<14} {len(b):>6} "
                  f"{100 * b['is_dnf'].mean():>7.1f}% "
                  f"{100 * b['is_crash'].mean():>7.1f}% "
                  f"{100 * b['is_mech'].mean():>7.1f}% "
                  f"{b['finish_pos'].mean():>12.1f}")

        # Correlation.
        corr_dnf = np.corrcoef(sub["start_pos"], sub["is_dnf"])[0, 1]
        corr_crash = np.corrcoef(sub["start_pos"], sub["is_crash"])[0, 1]
        print(f"\nPearson corr(start_pos, dnf):   {corr_dnf:+.3f}")
        print(f"Pearson corr(start_pos, crash): {corr_crash:+.3f}")


if __name__ == "__main__":
    main()
