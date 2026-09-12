"""Head-to-head beat-rate features.

For each (race_id_short, driver) row, compute (walk-forward, strictly from past
races):

  h2h_beat_rate_10  — over the driver's last 10 races, over every other driver
                      also present in each of those races, fraction of pairs
                      where our driver finished ahead. Both must have valid
                      finishing positions.

  h2h_beat_rate_type_10 — same, restricted to races whose track_type matches
                      the target race's track_type.

  h2h_shared_opponents_10 — count of unique opponents this driver has
                      completed >= 3 shared prior races with (used later as
                      confidence weight).

The intuition: avg_finish_N conflates driver quality with field strength.
Beat-rate is a field-adjusted ranking metric — finishing 10th in a race full of
strong drivers means more than finishing 10th in a weak field. It also lets a
close-call classifier see "does A generally beat B" rather than "does A finish
one position better on average."
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def compute_h2h(entries: pd.DataFrame, races: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Walk chronologically; before recording each race, compute rolling
    beat-rate features for every driver in that race using ONLY strictly prior
    races.

    Args:
        entries: must have race_id_short, driver, finish_pos, date, and
                 (via merge) track_type.
        races:   with race_id_short, date, track_type.
        window:  how many prior races to look back over.

    Returns: one row per (race_id_short, driver) with h2h_* columns.
    """
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "track_type" not in df.columns:
        df = df.merge(races[["race_id_short", "track_type"]], on="race_id_short", how="left")

    # Order races chronologically once.
    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )

    # For each driver, a deque of their last `window` completed races:
    #   list of tuples (race_id, track_type, {opponent_driver: finish_pos}, own_finish)
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        if sub.empty:
            continue
        tt = sub["track_type"].iloc[0]
        drivers = sub["driver"].tolist()
        finishes = sub["finish_pos"].tolist()

        # Emit features for each driver BEFORE updating with this race.
        for drv, fp in zip(drivers, finishes):
            hist = history[drv]
            beats_all = beats_type = pairs_all = pairs_type = 0
            opp_counts: dict[str, int] = defaultdict(int)
            for (h_rid, h_tt, h_opps, h_own) in hist:
                if h_own <= 0:  # driver didn't finish that past race
                    continue
                for opp, opp_fp in h_opps.items():
                    if opp_fp <= 0 or opp == drv:
                        continue
                    pairs_all += 1
                    won = int(h_own < opp_fp)
                    beats_all += won
                    opp_counts[opp] += 1
                    if h_tt == tt:
                        pairs_type += 1
                        beats_type += won
            h2h_all = beats_all / pairs_all if pairs_all else np.nan
            h2h_type = beats_type / pairs_type if pairs_type else np.nan
            shared = sum(1 for c in opp_counts.values() if c >= 3)
            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "h2h_beat_rate_10": h2h_all,
                "h2h_beat_rate_type_10": h2h_type,
                "h2h_shared_opponents_10": shared,
            })

        # Now update history with this race.
        opps = {d: int(f) for d, f in zip(drivers, finishes)}
        for drv, fp in zip(drivers, finishes):
            history[drv].append((rid, tt, opps, int(fp)))

    return pd.DataFrame(rows)
