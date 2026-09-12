"""Race-pace and manufacturer features.

Three separable signals, all walk-forward:

  1. Laps-led at track type — front-running tendency. A driver leading 40 laps
     tells you their car had race-winning speed, even if they finished 12th
     due to strategy/attrition. Captured per (driver, track_type) over last 10
     races of that type.

  2. Qual-to-finish delta at track type — race-pace vs qualifying-pace
     difference. Positive delta = qualified poorly, finished well = car
     improves through the race. Negative = fades. Real driver × car
     characteristic. Rolling last 10.

  3. Manufacturer × track_type avg finish — OEM package fit. Fords vs
     Toyotas vs Chevys perform differently at different track types. Rolling
     last 10 races of each (manufacturer, track_type) tuple.

All rolling stats are strictly prior — they never include the current race.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def compute_race_pace(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (race_id_short, driver) with:
       laps_led_avg_at_type_10, laps_led_pct_at_type_10,
       qual_to_finish_delta_at_type_10,
       manuf_avg_finish_at_type_10
    """
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "track_type" not in df.columns:
        df = df.merge(races[["race_id_short", "track_type"]], on="race_id_short", how="left")

    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )

    # Race-level laps (for laps_led percentage). Prefer actual_laps, fall back
    # to scheduled_laps.
    race_laps_lookup: dict = {}
    for _, rr in races.iterrows():
        n_laps = rr.get("actual_laps") or rr.get("scheduled_laps")
        if pd.notna(n_laps):
            race_laps_lookup[rr["race_id_short"]] = int(n_laps)

    driver_hist: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
    manuf_hist: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))

    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        if sub.empty:
            continue
        tt = sub["track_type"].iloc[0]

        for _, row in sub.iterrows():
            drv = row["driver"]
            make = row.get("make") or "Unknown"

            # Driver × track_type history
            d_hist = driver_hist[(drv, tt)]
            laps_led = [ll for ll, _fp, _qp, _tot in d_hist if ll is not None]
            laps_pct = [
                (ll / tot) for ll, _fp, _qp, tot in d_hist if ll is not None and tot
            ]
            deltas = [
                (qp - fp) for _ll, fp, qp, _tot in d_hist
                if fp is not None and qp is not None and fp > 0 and qp > 0
            ]

            # Manufacturer × track_type history
            m_hist = manuf_hist[(make, tt)]
            m_finishes = [f for f in m_hist if f > 0]

            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "laps_led_avg_at_type_10": (
                    float(np.mean(laps_led)) if laps_led else np.nan
                ),
                "laps_led_pct_at_type_10": (
                    float(np.mean(laps_pct)) if laps_pct else np.nan
                ),
                "qual_to_finish_delta_at_type_10": (
                    float(np.mean(deltas)) if deltas else np.nan
                ),
                "manuf_avg_finish_at_type_10": (
                    float(np.mean(m_finishes)) if m_finishes else np.nan
                ),
            })

        # Update history AFTER emitting.
        total_laps = race_laps_lookup.get(rid)
        for _, row in sub.iterrows():
            drv = row["driver"]
            make = row.get("make") or "Unknown"
            fp = int(row["finish_pos"]) if pd.notna(row["finish_pos"]) else 0
            qp = int(row["qual_pos"]) if pd.notna(row.get("qual_pos")) else 0
            ll = int(row["laps_led"]) if pd.notna(row.get("laps_led")) else 0
            if fp > 0:  # race actually run
                driver_hist[(drv, tt)].append((ll, fp, qp, total_laps))
                manuf_hist[(make, tt)].append(fp)

    return pd.DataFrame(rows)
