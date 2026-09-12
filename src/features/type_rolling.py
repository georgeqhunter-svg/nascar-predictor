"""Track-type-restricted rolling driver form.

The existing `rolling.py` computes avg_finish across ALL prior races. That's
dominated by the majority track type (ovals), so drivers whose skill is
mostly on road courses (SVG, Zilisch, Allmendinger) get penalized in overall
rolling stats and only see partial credit through the sparse `pl_driver_track`
term.

This module builds walk-forward rolling stats keyed by (driver, track_type)
and evaluated across the last N races of that specific track_type,
IRRESPECTIVE OF SEASON. So SVG's 12-race road-course history across 2024-2026
becomes a strong signal on his next road course rather than getting diluted
by his oval starts.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def compute_type_rolling(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (race_id_short, driver) with columns:
      avg_finish_at_type_last_5
      avg_finish_at_type_last_10
      races_at_type_last_10
    """
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "track_type" not in df.columns:
        df = df.merge(races[["race_id_short", "track_type"]], on="race_id_short", how="left")

    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )

    # For each (driver, track_type), a deque of the last 10 finishing positions.
    history: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))

    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        if sub.empty:
            continue
        tt = sub["track_type"].iloc[0]

        # Emit features BEFORE updating.
        for drv, fp in zip(sub["driver"], sub["finish_pos"]):
            hist = history[(drv, tt)]
            valid = [f for f in hist if f > 0]
            last5 = valid[-5:]
            avg5 = float(np.mean(last5)) if last5 else np.nan
            avg10 = float(np.mean(valid)) if valid else np.nan
            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "avg_finish_at_type_last_5": avg5,
                "avg_finish_at_type_last_10": avg10,
                "races_at_type_last_10": len(valid),
            })

        # Update history with this race's outcomes.
        for drv, fp in zip(sub["driver"], sub["finish_pos"]):
            if int(fp) > 0:
                history[(drv, tt)].append(int(fp))

    return pd.DataFrame(rows)
