"""Track-specific rolling driver form.

Type-level rolling groups all intermediates together; that averages Kansas
(smooth, low tire wear) with Darlington (worn, abrasive, high tire wear) and
tells you nothing distinctive about either. Even within intermediate, tracks
behave differently.

This module builds walk-forward rolling stats keyed by (driver, track_id) —
matching only races AT THE SAME PHYSICAL TRACK across seasons. A driver with
5 recent Darlington starts averaging 8th place carries strong Darlington-
specific signal that neither type-level rolling nor all-time avg_finish_at_
track captures well.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def compute_track_rolling(
    entries: pd.DataFrame, races: pd.DataFrame
) -> pd.DataFrame:
    """Per (race_id_short, driver) return:
      avg_finish_at_track_last_5
      avg_finish_at_track_last_10
      best_finish_at_track_last_10
      races_at_track_last_10
    keyed on track_id (falls back to track_name if track_id missing).
    """
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Merge track identity.
    key_col = "track_id" if "track_id" in races.columns else "track_name"
    if key_col not in df.columns:
        df = df.merge(
            races[["race_id_short", key_col]], on="race_id_short", how="left"
        )

    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )

    history: dict[tuple[str, object], deque] = defaultdict(lambda: deque(maxlen=10))

    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        if sub.empty:
            continue
        tid = sub[key_col].iloc[0]

        # Emit features BEFORE updating.
        for drv, fp in zip(sub["driver"], sub["finish_pos"]):
            hist = history[(drv, tid)]
            valid = [f for f in hist if f > 0]
            last5 = valid[-5:]
            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "avg_finish_at_track_last_5": (
                    float(np.mean(last5)) if last5 else np.nan
                ),
                "avg_finish_at_track_last_10": (
                    float(np.mean(valid)) if valid else np.nan
                ),
                "best_finish_at_track_last_10": (
                    int(np.min(valid)) if valid else np.nan
                ),
                "races_at_track_last_10": len(valid),
            })

        # Update history AFTER emitting.
        for drv, fp in zip(sub["driver"], sub["finish_pos"]):
            if int(fp) > 0:
                history[(drv, tid)].append(int(fp))

    return pd.DataFrame(rows)
