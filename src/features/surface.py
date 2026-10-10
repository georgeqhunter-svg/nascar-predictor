"""Track surface classes + driver history on the same surface class.

Repave years: dailydownforce.com "Every NASCAR Race Track's Last Repave"
(Jan 2024) + user (2026-10-09): North Wilkesboro repaved 2024, no full
repaves since 2024, every track not in the list is asphalt (age unknown).
Concrete: Dover, Nashville, Bristol; Martinsville has concrete turns.

Classes (ovals only, excl. superspeedways; by surface age AT RACE DATE):
  fresh < 5 yrs | mid 5-14 | old 15+ | concrete. Unknown age -> no class.

diag_surface_class.py (2022-2025 only, pre-registered): a driver's mean
finish over his last 10 races on the same class predicts finish beyond
recent form + track-type history + this-track history: t=+3.8 (old +3.1,
concrete +2.3, mid +1.6).

Column: surface_class_hist_10 (finish pct, lower = better; NaN when the race
has no class or the driver has < 3 prior races in it).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .tracks import resolve_track_type

REPAVE_YEAR = {
    "Atlanta Motor Speedway": 2022, "Sonoma Raceway": 2024, "North Wilkesboro Speedway": 2024,
    "Texas Motor Speedway": 2017, "World Wide Technology Raceway": 2017,
    "Watkins Glen International": 2016, "Kansas Speedway": 2012,
    "Michigan International Speedway": 2012, "Pocono Raceway": 2012, "Phoenix Raceway": 2011,
    "Daytona International Speedway": 2011, "Darlington Raceway": 2008,
    "Las Vegas Motor Speedway": 2007, "Charlotte Motor Speedway": 2006,
    "Talladega Superspeedway": 2006, "New Hampshire Motor Speedway": 2005,
    "Richmond Raceway": 2004, "Martinsville Speedway": 2004,
    "Indianapolis Motor Speedway": 2004, "Homestead-Miami Speedway": 2003, "Iowa Speedway": 2006,
}
CONCRETE = {"Dover Motor Speedway", "Nashville Superspeedway", "Bristol Motor Speedway",
            "Martinsville Speedway"}
# Surfaces in use BEFORE the listed repave were long-worn.
OLD_BEFORE_REPAVE = {"North Wilkesboro Speedway": 2024, "Sonoma Raceway": 2024}

WINDOW, MIN_N = 10, 3


def surface_class(track: str, year: int) -> str | None:
    if resolve_track_type(track) in ("road", "superspeedway"):
        return None
    if track in CONCRETE:
        return "concrete"
    if track in OLD_BEFORE_REPAVE and year < OLD_BEFORE_REPAVE[track]:
        return "old"
    y = REPAVE_YEAR.get(track)
    if y is None:
        return None
    age = year - y
    return "fresh" if age < 5 else ("mid" if age < 15 else "old")


def compute_surface_history(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    e = entries[["race_id_short", "driver", "finish_pos", "date"]].merge(
        races[["race_id_short", "track_name"]], on="race_id_short", how="left")
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values(["date", "race_id_short"], kind="stable")
    fp = pd.to_numeric(e["finish_pos"], errors="coerce")
    n = fp.where(fp > 0).groupby(e["race_id_short"]).transform("count")
    e["fp"] = (fp - 1) / (n - 1)
    hist: dict = defaultdict(lambda: deque(maxlen=WINDOW))
    rows = []
    for (rid, tn, d), g in e.groupby(["race_id_short", "track_name", "date"], sort=False):
        cls = surface_class(str(tn), d.year)
        for drv in g["driver"]:                                        # emit
            h = hist[(drv, cls)] if cls else ()
            rows.append({"race_id_short": rid, "driver": drv,
                         "surface_class_hist_10": float(np.mean(h)) if cls and len(h) >= MIN_N else np.nan})
        if cls:                                                        # ingest
            for drv, v in zip(g["driver"], g["fp"]):
                if pd.notna(v):
                    hist[(drv, cls)].append(float(v))
    return pd.DataFrame(rows)
