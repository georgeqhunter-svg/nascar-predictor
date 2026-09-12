"""Driver-specific outcome variance and DNF rate at track type.

Currently our Gumbel-max sampling assumes every driver in a race shares a
single track-type-level hazard. That flattens real driver differences:
Larson's finishing distribution is tight and centered at 5th; Preece's is
bimodal with a fat right tail from mechanical DNFs and crashes.

These rolling walk-forward features capture that per-driver shape:

  dnf_rate_at_type_10        rolling DNF rate at same track type,
                             blended with track baseline downstream.
  finish_std_at_type_10      std dev of finishing positions at type over
                             last 10 races. Bigger = wider outcome distribution.
  finish_iqr_at_type_10      inter-quartile range at type; robust variant.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def compute_driver_variance(
    entries: pd.DataFrame, races: pd.DataFrame, window: int = 10
) -> pd.DataFrame:
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "track_type" not in df.columns:
        df = df.merge(races[["race_id_short", "track_type"]], on="race_id_short", how="left")

    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )

    # history[(driver, track_type)] = deque of (finish_pos, is_dnf)
    history: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))

    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        if sub.empty:
            continue
        tt = sub["track_type"].iloc[0]
        for drv, fp, dnf in zip(sub["driver"], sub["finish_pos"], sub["is_dnf"]):
            hist = history[(drv, tt)]
            finishes = [f for f, _ in hist if f > 0]
            dnfs = [int(bool(d)) for _, d in hist]
            n = len(hist)
            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "dnf_rate_at_type_10": (
                    float(np.mean(dnfs)) if dnfs else np.nan
                ),
                "finish_std_at_type_10": (
                    float(np.std(finishes)) if len(finishes) >= 3 else np.nan
                ),
                "finish_iqr_at_type_10": (
                    float(np.subtract(*np.percentile(finishes, [75, 25])))
                    if len(finishes) >= 4 else np.nan
                ),
                "n_races_at_type_for_variance": n,
            })

        # Update after emitting.
        for drv, fp, dnf in zip(sub["driver"], sub["finish_pos"], sub["is_dnf"]):
            if int(fp) > 0:
                history[(drv, tt)].append((int(fp), bool(dnf)))

    return pd.DataFrame(rows)
