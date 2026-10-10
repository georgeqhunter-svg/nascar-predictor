"""Clean-air, tire-age-corrected race pace (walk-forward, last 5 races).

Per race, from lap-times:
  * green lap  = lap time <= 1.10 x race fast-lap reference (10th pct)
  * green run  = consecutive green laps; skip the first 2 of each (restarts)
  * clean air  = leading, or > 1.0 s behind the car ahead on that lap
                 (gap from cumulative elapsed time)
  * rel        = lap time / median clean-air lap time on that lap - 1
  * tire age   = laps into the green run; one pooled slope per race removed
  * race pace  = driver's median adjusted rel (x100, lower = faster), >= 15 laps
Feature: clean_air_pace_5 = mean race pace over the driver's last 5 races.

diag_clean_air_pace.py (2022-2025, pre-registered): beyond form, running
position, loop rating AND green_pace_pct_5: +0.0126 (t=+2.00) — small; corr
with green_pace_pct_5 0.85.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

CLEAN_GAP, GREEN_MAX, SKIP, MIN_LAPS, WINDOW = 1.0, 1.10, 2, 15, 5


def per_race_clean_air(laptimes: pd.DataFrame) -> pd.DataFrame:
    L = laptimes[["race_id_short", "driver_id", "lap", "lap_time", "running_pos"]].copy()
    L["t"] = pd.to_numeric(L["lap_time"], errors="coerce")
    L = L[(L["lap"] >= 1) & (L["t"] > 0)].sort_values(["race_id_short", "driver_id", "lap"])
    L["elapsed"] = L.groupby(["race_id_short", "driver_id"])["t"].cumsum()
    L["pos"] = pd.to_numeric(L["running_pos"], errors="coerce")
    rows = []
    for rid, g in L.groupby("race_id_short", sort=False):
        ref = g["t"].quantile(0.10)
        g = g.assign(green=g["t"] <= GREEN_MAX * ref).sort_values(["driver_id", "lap"])
        brk = (~g["green"]) | (g.groupby("driver_id")["lap"].diff() != 1)
        run_id = brk.groupby(g["driver_id"]).cumsum()
        g["age"] = g.groupby([g["driver_id"], run_id]).cumcount()
        ahead = g[["lap", "pos", "elapsed"]].rename(columns={"elapsed": "el_ahead"})
        ahead["pos"] = ahead["pos"] + 1
        g = g.merge(ahead, on=["lap", "pos"], how="left")
        g["gap"] = g["elapsed"] - g["el_ahead"]
        c = g[g["green"] & (g["age"] >= SKIP) & ((g["pos"] == 1) | (g["gap"] > CLEAN_GAP))].copy()
        if len(c) < 200:
            continue
        c["rel"] = c["t"] / c.groupby("lap")["t"].transform("median") - 1
        a = c["age"].clip(upper=60)
        b = np.polyfit(a, c["rel"], 1)[0]
        c["adj"] = c["rel"] - b * (a - a.mean())
        p = c.groupby("driver_id")["adj"].agg(["median", "size"])
        for did, (med, n) in p[p["size"] >= MIN_LAPS].iterrows():
            rows.append({"race_id_short": rid, "driver_id": int(did), "ca_pace": float(med) * 100})
    return pd.DataFrame(rows)


def compute_clean_air_pace(laptimes: pd.DataFrame | None, entries: pd.DataFrame,
                           races: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id_short", "driver", "clean_air_pace_5"]
    if laptimes is None or laptimes.empty:
        return pd.DataFrame(columns=cols)
    per = per_race_clean_air(laptimes)
    pace = {(r, d): v for r, d, v in zip(per["race_id_short"], per["driver_id"], per["ca_pace"])}
    e = entries[["race_id_short", "driver", "driver_id"]].merge(
        races[["race_id_short", "date"]], on="race_id_short", how="left")
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values(["date", "race_id_short"], kind="stable")
    hist: dict = defaultdict(lambda: deque(maxlen=WINDOW))
    rows = []
    for rid, g in e.groupby("race_id_short", sort=False):
        for drv in g["driver"]:                                             # emit (all entries)
            h = hist[drv]
            rows.append({"race_id_short": rid, "driver": drv,
                         "clean_air_pace_5": float(np.mean(h)) if len(h) >= 2 else np.nan})
        for drv, did in zip(g["driver"], g["driver_id"]):                   # ingest
            if pd.notna(did) and (rid, int(did)) in pace:
                hist[drv].append(pace[(rid, int(did))])
    return pd.DataFrame(rows, columns=cols)
