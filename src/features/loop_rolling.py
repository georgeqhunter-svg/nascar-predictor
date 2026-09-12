"""Walk-forward rolling loop-stat features per driver.

Loop data (avg running position, quality passes, NASCAR Driver Rating, etc.)
comes from the race itself — post-hoc. So for a prediction of race R, we can
only use loop rows from races strictly BEFORE R.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


ROLL_COLS = ["avg_ps", "quality_passes", "rating", "fast_laps", "passing_diff",
             "top15_laps", "lead_laps"]


def compute_loop_rolling(loopstats: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (race_id_short, driver_id) with rolling means of
    each loop stat over the driver's last-5, last-10, last-20 races.

    Rolling values reflect data known BEFORE the race, not the race's own
    loopstats.
    """
    if loopstats.empty:
        return pd.DataFrame(columns=[
            "race_id_short", "driver_id",
            *[f"loop_{c}_5" for c in ROLL_COLS],
            *[f"loop_{c}_10" for c in ROLL_COLS],
        ])

    r = races[["race_id_short", "date"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    df = (
        loopstats.merge(r, on="race_id_short", how="left", suffixes=("", "_r"))
        .assign(date=lambda x: pd.to_datetime(x["date_r"].where(x["date_r"].notna(), x["date"])))
        .drop(columns=[c for c in ["date_r"] if c in loopstats.columns.tolist() + ["date_r"]], errors="ignore")
        .sort_values(["date", "race_id_short", "driver_id"])
    )
    # Ensure numeric.
    for c in ROLL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    hist: dict[int, dict[str, deque]] = defaultdict(
        lambda: {c: deque(maxlen=20) for c in ROLL_COLS}
    )

    out_rows = []
    for (race_id, _date), g in df.groupby(["race_id_short", "date"], sort=False):
        # EMIT rolling snapshots BEFORE ingesting this race's loop rows.
        for _, row in g.iterrows():
            drv = int(row["driver_id"])
            h = hist[drv]
            snap = {"race_id_short": race_id, "driver_id": drv}
            for c in ROLL_COLS:
                arr = list(h[c])
                snap[f"loop_{c}_5"] = float(np.mean(arr[-5:])) if arr else np.nan
                snap[f"loop_{c}_10"] = float(np.mean(arr[-10:])) if arr else np.nan
            out_rows.append(snap)

        # INGEST this race's loop rows into history for downstream races.
        for _, row in g.iterrows():
            drv = int(row["driver_id"])
            h = hist[drv]
            for c in ROLL_COLS:
                v = row[c]
                if pd.notna(v):
                    h[c].append(float(v))

    return pd.DataFrame(out_rows)
