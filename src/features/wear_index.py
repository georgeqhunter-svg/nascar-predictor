"""Track surface wear index, measured from race lap-times (walk-forward).

tracks.py only knows length/banking/layout, so Kansas, Las Vegas, Homestead
and Texas (all 1.5 mi, 20 deg) are identical to the model. What actually
differs is the surface: old, abrasive asphalt (Homestead, Darlington) eats
tires and makes long-run pace fall off; fresh/smooth surfaces don't. Repave
dates are hard to source reliably, so instead we MEASURE wear: the field's
green-flag lap-time falloff over a run, per race, re-using tire_deg's stint
logic. A repave shows up automatically as lower falloff in the next races.

Per race:
  race_falloff_pct = field median over drivers of mean per-stint slope,
                     as % of the race's median green lap time, per lap.
  race_retention   = field median of (late-stint median lap / best lap).
Per (race, track) feature, from strictly PRIOR races at the same track,
last WEAR_WINDOW races, shrunk toward the walk-forward track-type mean:
  track_wear_falloff, track_wear_retention
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .tire_deg import per_race_tire_stats
from .tracks import resolve_track_type

WEAR_WINDOW = 3
WEAR_SHRINK_K = 1.0     # prior races-worth of type mean


GREEN_ABS_MAX = 1.10    # green lap = within 10% of the race's fast-lap reference
RUN_MIN = 25            # only long green runs
EARLY = (2, 6)          # run laps used as "fresh tire" pace (skip lap 1: restart)
LATE = (18, 24)         # run laps used as "worn tire" pace


def per_race_wear(laptimes: pd.DataFrame) -> pd.DataFrame:
    """Field-median pace loss over long green runs, per race.

    Green laps are judged against the race's ABSOLUTE fast-lap reference
    (10th pct of all lap times), not the per-lap field median — under caution
    the whole field is slow together, so a relative filter lets caution laps
    through (the bug in tire_deg.py's filter). A run = consecutive green laps
    for one driver; any caution or pit lap ends it.
      falloff_pct = 100 * (median LATE laps / min EARLY laps - 1)
    """
    rows = []
    for rid, lt in laptimes.groupby("race_id_short"):
        lt = lt.copy()
        lt["t"] = pd.to_numeric(lt["lap_time"], errors="coerce")
        lt = lt[lt["t"] > 0].sort_values(["driver_id", "lap"])
        if lt.empty:
            continue
        ref = float(lt["t"].quantile(0.10))
        lt["green"] = lt["t"] <= GREEN_ABS_MAX * ref
        vals = []
        for _, s in lt.groupby("driver_id"):
            g = s["green"].to_numpy(); t = s["t"].to_numpy(); lap = s["lap"].to_numpy()
            i = 0
            while i < len(g):
                if not g[i]:
                    i += 1; continue
                j = i
                while j + 1 < len(g) and g[j + 1] and lap[j + 1] == lap[j] + 1:
                    j += 1
                run = t[i:j + 1]
                if len(run) >= RUN_MIN:
                    early = run[EARLY[0] - 1:EARLY[1]].min()
                    late = np.median(run[LATE[0] - 1:LATE[1]])
                    vals.append(100.0 * (late / early - 1.0))
                i = j + 1
        if len(vals) >= 15:
            rows.append({"race_id_short": rid, "race_falloff_pct": float(np.median(vals)),
                         "race_retention": 1.0 + float(np.median(vals)) / 100.0,
                         "n_runs": len(vals)})
    return pd.DataFrame(rows)


def compute_wear_index(laptimes: pd.DataFrame | None, races: pd.DataFrame,
                       entries: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id_short", "driver", "track_wear_falloff", "track_wear_retention"]
    if laptimes is None or laptimes.empty:
        return pd.DataFrame(columns=cols)
    per = per_race_wear(laptimes).set_index("race_id_short")
    r = races[["race_id_short", "date", "track_name"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values(["date", "race_id_short"], kind="stable")
    trk_f, trk_r = defaultdict(list), defaultdict(list)
    typ_f, typ_r = defaultdict(list), defaultdict(list)
    drivers = entries.groupby("race_id_short")["driver"].apply(list).to_dict()
    rows = []
    for rr in r.itertuples(index=False):
        tn, tt = str(rr.track_name), resolve_track_type(str(rr.track_name))
        def shrunk(trk, typ):
            h = trk[tn][-WEAR_WINDOW:]
            prior = float(np.mean(typ[tt])) if typ[tt] else np.nan
            if not h:
                return prior
            if not np.isfinite(prior):
                return float(np.mean(h))
            return (float(np.sum(h)) + WEAR_SHRINK_K * prior) / (len(h) + WEAR_SHRINK_K)
        f, ret = shrunk(trk_f, typ_f), shrunk(trk_r, typ_r)
        for dname in drivers.get(rr.race_id_short, []):              # emit
            rows.append({"race_id_short": rr.race_id_short, "driver": dname,
                         "track_wear_falloff": f, "track_wear_retention": ret})
        if rr.race_id_short in per.index:                             # ingest
            pf = per.at[rr.race_id_short, "race_falloff_pct"]
            pr = per.at[rr.race_id_short, "race_retention"]
            trk_f[tn].append(pf); typ_f[tt].append(pf)
            trk_r[tn].append(pr); typ_r[tt].append(pr)
    return pd.DataFrame(rows, columns=cols)
