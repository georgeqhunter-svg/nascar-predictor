"""Manufacturer × track-type interaction features.

The existing pipeline has `manuf_avg_finish_at_type_10` (avg finish of the
manufacturer at this track type, last 10 races of type). We add:

  manuf_win_rate_at_type_10        - win rate of manuf at this track type
  manuf_top5_rate_at_type_10       - top-5 rate
  manuf_top10_rate_at_type_10      - top-10 rate
  manuf_avg_finish_at_type_5       - shorter, more recent window
  manuf_finish_std_at_type_10      - consistency signal
  drv_manuf_type_avg_finish_10     - INTERACTION: this driver's avg finish at
                                     this track type WHILE in the current
                                     manufacturer, last 10 relevant races

All features are strictly pre-race (indexed at each race by prior data only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_manuf_type(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (race_id_short, driver) with the features above."""
    e = entries.copy()
    e["date"] = pd.to_datetime(e["date"])
    # Attach track_type from races.
    if "track_type" not in e.columns:
        e = e.merge(
            races[["race_id_short", "track_type"]], on="race_id_short", how="left"
        )
    if "make" not in e.columns:
        e["make"] = "Unknown"
    e["make"] = e["make"].fillna("Unknown")
    e = e.sort_values("date").reset_index(drop=True)

    # Pre-index groups for speed.
    make_type_rows = []  # for manufacturer aggregates per (make, track_type)
    finished = e[e["finish_pos"] > 0]

    # We compute per-race features by iterating chronologically and using
    # only rows strictly before the race date.
    e_index = e.reset_index().rename(columns={"index": "_i"})
    rows = []
    # Cache: for each (make, track_type) prior series indexed by date.
    finished_by_make_type = {
        k: g.sort_values("date").reset_index(drop=True)
        for k, g in finished.groupby(["make", "track_type"])
    }
    finished_by_drv_make_type = {
        k: g.sort_values("date").reset_index(drop=True)
        for k, g in finished.groupby(["driver", "make", "track_type"])
    }

    for _, r in e.iterrows():
        rid = r["race_id_short"]
        drv = r["driver"]
        make = r["make"]
        tt = r["track_type"]
        d = r["date"]

        # Manufacturer aggregates over prior races of this track type.
        mt = finished_by_make_type.get((make, tt))
        if mt is not None:
            prior = mt[mt["date"] < d]
            last10 = prior.tail(10)
            last5 = prior.tail(5)
            n10 = len(last10)
            if n10 > 0:
                fin10 = last10["finish_pos"].to_numpy()
                m_avg_10 = float(fin10.mean())
                m_win_10 = float((fin10 == 1).mean())
                m_top5_10 = float((fin10 <= 5).mean())
                m_top10_10 = float((fin10 <= 10).mean())
                m_std_10 = float(fin10.std()) if n10 > 1 else np.nan
            else:
                m_avg_10 = m_win_10 = m_top5_10 = m_top10_10 = m_std_10 = np.nan
            m_avg_5 = float(last5["finish_pos"].mean()) if len(last5) > 0 else np.nan
        else:
            m_avg_10 = m_win_10 = m_top5_10 = m_top10_10 = m_std_10 = m_avg_5 = np.nan

        # Driver × manuf × track-type interaction.
        dmt = finished_by_drv_make_type.get((drv, make, tt))
        if dmt is not None:
            prior = dmt[dmt["date"] < d]
            last10 = prior.tail(10)
            drv_mt_avg_10 = float(last10["finish_pos"].mean()) if len(last10) > 0 else np.nan
            drv_mt_races_10 = int(len(last10))
        else:
            drv_mt_avg_10 = np.nan
            drv_mt_races_10 = 0

        rows.append({
            "race_id_short": rid,
            "driver": drv,
            "manuf_avg_finish_at_type_5_v2": m_avg_5,
            "manuf_avg_finish_at_type_10_v2": m_avg_10,
            "manuf_win_rate_at_type_10": m_win_10,
            "manuf_top5_rate_at_type_10": m_top5_10,
            "manuf_top10_rate_at_type_10": m_top10_10,
            "manuf_finish_std_at_type_10": m_std_10,
            "drv_manuf_type_avg_finish_10": drv_mt_avg_10,
            "drv_manuf_type_races_10": drv_mt_races_10,
        })
    return pd.DataFrame(rows)
