"""Three new walk-forward feature groups (added 2026-09-24).

1. Start-position expectation by track type
     exp_finish_from_start   — historical mean finish for drivers starting in
                               this start-position bucket at this track type,
                               shrunk toward the type's overall mean finish.
     start_stickiness_at_type — Pearson corr(start_pos, finish_pos) across all
                               prior races of this track type. High at short
                               tracks, near zero at superspeedways.

2. Green-flag pace (from laptimes)
     green_pace_pct_5 / _10  — rolling mean of a driver's per-race median
                               green-flag lap-time percentile (0 = fastest in
                               field on that lap, 1 = slowest). Caution laps
                               and pit laps are excluded, so this isolates raw
                               speed from strategy, cautions, and wrecks.

3. Crew chief
     cc_races_together       — consecutive races this driver has run with the
                               current crew chief (capped at 20). Low values
                               flag a recent crew-chief change.
     cc_avg_finish_10        — the crew chief's own rolling avg finish over
                               their last 10 races, across whatever drivers
                               they called for.

All features are strictly pre-race: for each race we emit from history, THEN
ingest that race's outcome.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .tracks import resolve_track_type


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def _race_order_with_types(races: pd.DataFrame) -> list[tuple[str, str]]:
    r = races[["race_id_short", "date", "track_type"]].copy()
    if "track_name" in races.columns:
        r["track_name"] = races["track_name"]
    else:
        r["track_name"] = ""
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values("date")
    out = []
    for row in r.itertuples(index=False):
        tt = resolve_track_type(row.track_name, fallback=row.track_type)
        out.append((row.race_id_short, tt))
    return out


# ----------------------------------------------------------------------------
# 1. Start-position expectation by track type
# ----------------------------------------------------------------------------
START_BUCKETS = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 20),
                 (21, 25), (26, 30), (31, 99)]
SHRINK_N = 30  # pseudo-observations toward the type mean for thin buckets


def _bucket(sp: float) -> int | None:
    if not np.isfinite(sp) or sp <= 0:
        return None
    for i, (lo, hi) in enumerate(START_BUCKETS):
        if lo <= sp <= hi:
            return i
    return None


def compute_start_expectation(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    order = _race_order_with_types(races)
    e = entries[["race_id_short", "driver", "start_pos", "finish_pos"]].copy()
    e["start_pos"] = pd.to_numeric(e["start_pos"], errors="coerce")
    e["finish_pos"] = pd.to_numeric(e["finish_pos"], errors="coerce")
    by_race = {rid: g for rid, g in e.groupby("race_id_short")}

    # Per type: per-bucket sums/counts, plus overall sums for the shrinkage mean
    # and running moments for the Pearson correlation.
    bsum = defaultdict(lambda: np.zeros(len(START_BUCKETS)))
    bcnt = defaultdict(lambda: np.zeros(len(START_BUCKETS)))
    mom = defaultdict(lambda: np.zeros(6))  # n, sx, sy, sxx, syy, sxy

    rows = []
    for rid, tt in order:
        g = by_race.get(rid)
        if g is None:
            continue
        m = mom[tt]
        n = m[0]
        type_mean = (m[2] / n) if n > 0 else np.nan
        if n > 30:
            cov = m[5] / n - (m[1] / n) * (m[2] / n)
            vx = m[3] / n - (m[1] / n) ** 2
            vy = m[4] / n - (m[2] / n) ** 2
            stick = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
        else:
            stick = np.nan

        for row in g.itertuples(index=False):
            b = _bucket(row.start_pos)
            if b is None or not np.isfinite(type_mean):
                exp = np.nan
            else:
                s, c = bsum[tt][b], bcnt[tt][b]
                exp = (s + SHRINK_N * type_mean) / (c + SHRINK_N)
            rows.append({
                "race_id_short": rid, "driver": row.driver,
                "exp_finish_from_start": exp,
                "start_stickiness_at_type": stick,
            })

        # Ingest this race (only finished races with a valid start).
        for row in g.itertuples(index=False):
            if not (np.isfinite(row.finish_pos) and row.finish_pos > 0):
                continue
            b = _bucket(row.start_pos)
            if b is None:
                continue
            bsum[tt][b] += row.finish_pos
            bcnt[tt][b] += 1
            x, y = float(row.start_pos), float(row.finish_pos)
            mom[tt] += np.array([1, x, y, x * x, y * y, x * y])

    return pd.DataFrame(rows).drop_duplicates(["race_id_short", "driver"])


# ----------------------------------------------------------------------------
# 2. Green-flag pace from laptimes
# ----------------------------------------------------------------------------
CAUTION_MULT = 1.8   # lap median / baseline above this = caution lap
PIT_MULT = 1.4       # driver lap / lap median above this = pit / incident lap
MIN_GREEN_LAPS = 20


def _race_green_pace(lt: pd.DataFrame) -> pd.Series:
    """Per driver_id: median green-flag lap-time percentile in this race."""
    lt = lt[["lap", "driver_id", "lap_time"]].copy()
    lt["lap_time"] = pd.to_numeric(lt["lap_time"], errors="coerce")
    lt = lt.dropna()
    if lt.empty:
        return pd.Series(dtype=float)
    times = lt["lap_time"].to_numpy()
    baseline = float(np.median(times[times <= np.median(times)]))
    if not np.isfinite(baseline) or baseline <= 0:
        return pd.Series(dtype=float)

    lap_med = lt.groupby("lap")["lap_time"].transform("median")
    green_lap = lap_med <= baseline * CAUTION_MULT
    not_pit = lt["lap_time"] <= lap_med * PIT_MULT
    lt = lt[green_lap & not_pit & (lt["lap"] > 1)].copy()
    if lt.empty:
        return pd.Series(dtype=float)

    lt["pct"] = lt.groupby("lap")["lap_time"].rank(pct=True, method="average")
    counts = lt.groupby("driver_id")["pct"].size()
    med = lt.groupby("driver_id")["pct"].median()
    return med[counts >= MIN_GREEN_LAPS]


def compute_green_pace(laptimes: pd.DataFrame, entries: pd.DataFrame,
                       races: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id_short", "driver", "green_pace_pct_5", "green_pace_pct_10"]
    if laptimes is None or laptimes.empty:
        return pd.DataFrame(columns=cols)
    order = _race_order_with_types(races)
    lt_by_race = {rid: g for rid, g in laptimes.groupby("race_id_short")}

    ent = entries[["race_id_short", "driver", "driver_id"]].copy()
    ent["driver_id"] = pd.to_numeric(ent["driver_id"], errors="coerce")
    ent_by_race = {rid: g for rid, g in ent.groupby("race_id_short")}

    hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for rid, _tt in order:
        eg = ent_by_race.get(rid)
        if eg is None:
            continue
        for row in eg.itertuples(index=False):
            if not np.isfinite(row.driver_id):
                p5 = p10 = np.nan
            else:
                h = list(hist[int(row.driver_id)])
                p5 = float(np.mean(h[-5:])) if h else np.nan
                p10 = float(np.mean(h)) if h else np.nan
            rows.append({"race_id_short": rid, "driver": row.driver,
                         "green_pace_pct_5": p5, "green_pace_pct_10": p10})

        lt = lt_by_race.get(rid)
        if lt is not None:
            for drv_id, pct in _race_green_pace(lt).items():
                if pd.notna(drv_id):
                    hist[int(drv_id)].append(float(pct))

    return pd.DataFrame(rows).drop_duplicates(["race_id_short", "driver"])


# ----------------------------------------------------------------------------
# 3. Crew chief
# ----------------------------------------------------------------------------
def compute_crew_chief(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id_short", "driver", "cc_races_together", "cc_avg_finish_10"]
    cc_col = "crew_chief_id" if "crew_chief_id" in entries.columns else (
        "crew_chief" if "crew_chief" in entries.columns else None)
    if cc_col is None:
        return pd.DataFrame(columns=cols)

    order = _race_order_with_types(races)
    e = entries[["race_id_short", "driver", cc_col, "finish_pos"]].copy()
    e["finish_pos"] = pd.to_numeric(e["finish_pos"], errors="coerce")
    by_race = {rid: g for rid, g in e.groupby("race_id_short")}

    last_cc: dict[str, object] = {}
    streak: dict[str, int] = defaultdict(int)
    cc_hist: dict[object, deque] = defaultdict(lambda: deque(maxlen=10))

    rows = []
    for rid, _tt in order:
        g = by_race.get(rid)
        if g is None:
            continue
        for row in g.itertuples(index=False):
            cc = getattr(row, cc_col)
            valid_cc = pd.notna(cc) and str(cc) not in ("", "0", "nan")
            if valid_cc and last_cc.get(row.driver) == cc:
                together = min(streak[row.driver], 20)
            else:
                together = 0  # new pairing (or no history) as of this race
            h = list(cc_hist[cc]) if valid_cc else []
            rows.append({
                "race_id_short": rid, "driver": row.driver,
                "cc_races_together": together,
                "cc_avg_finish_10": float(np.mean(h)) if h else np.nan,
            })

        # Ingest this race.
        for row in g.itertuples(index=False):
            if not (np.isfinite(row.finish_pos) and row.finish_pos > 0):
                continue
            cc = getattr(row, cc_col)
            if pd.isna(cc) or str(cc) in ("", "0", "nan"):
                continue
            if last_cc.get(row.driver) == cc:
                streak[row.driver] += 1
            else:
                streak[row.driver] = 1
                last_cc[row.driver] = cc
            cc_hist[cc].append(float(row.finish_pos))

    return pd.DataFrame(rows).drop_duplicates(["race_id_short", "driver"])


# ----------------------------------------------------------------------------
# 4. Similar-track form (added 2026-09-25)
# ----------------------------------------------------------------------------
SIM_LEN_TOL = 0.25     # miles
SIM_BANK_TOL = 6.0     # degrees
SIM_WINDOW = 10
SIM_MIN_RACES = 3


def _similar_track_map() -> dict[str, set[str]]:
    """track display key -> set of physically similar track keys (incl. itself).

    Similar = same track_type AND length within SIM_LEN_TOL AND banking within
    SIM_BANK_TOL. Road/street courses (no banking) match any road course.
    Defined from geometry only — no race outcomes involved.
    """
    from .tracks import TRACKS
    info = {slug.replace("_", " ").lower(): t for slug, t in TRACKS.items()}
    sim: dict[str, set[str]] = {}
    for k, t in info.items():
        s = set()
        for k2, t2 in info.items():
            if t2.track_type != t.track_type:
                continue
            if t.banking_deg is None or t2.banking_deg is None:
                if t.banking_deg is None and t2.banking_deg is None:
                    s.add(k2)
                continue
            if (abs(t.length_mi - t2.length_mi) <= SIM_LEN_TOL
                    and abs(t.banking_deg - t2.banking_deg) <= SIM_BANK_TOL):
                s.add(k2)
        s.add(k)
        sim[k] = s
    return sim


def compute_similar_track_form(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """similar_track_avg_finish_10 — driver's mean finish over their last 10
    races at physically similar tracks (see _similar_track_map), strictly prior.
    NaN if fewer than SIM_MIN_RACES such races."""
    sim = _similar_track_map()
    r = races[["race_id_short", "date"]].copy()
    r["tkey"] = races["track_name"].astype(str).str.lower() if "track_name" in races else ""
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values("date")

    e = entries[["race_id_short", "driver", "finish_pos"]].copy()
    e["finish_pos"] = pd.to_numeric(e["finish_pos"], errors="coerce")
    by_race = {rid: g for rid, g in e.groupby("race_id_short")}

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=120))
    rows = []
    for row in r.itertuples(index=False):
        g = by_race.get(row.race_id_short)
        if g is None:
            continue
        similar = sim.get(row.tkey, {row.tkey})
        for d in g.itertuples(index=False):
            matches = [f for tk, f in hist[d.driver] if tk in similar][-SIM_WINDOW:]
            rows.append({
                "race_id_short": row.race_id_short, "driver": d.driver,
                "similar_track_avg_finish_10": (
                    float(np.mean(matches)) if len(matches) >= SIM_MIN_RACES else np.nan),
            })
        for d in g.itertuples(index=False):
            if np.isfinite(d.finish_pos) and d.finish_pos > 0:
                hist[d.driver].append((row.tkey, float(d.finish_pos)))
    return pd.DataFrame(rows).drop_duplicates(["race_id_short", "driver"])


def compute_new_signals(entries: pd.DataFrame, races: pd.DataFrame,
                        laptimes: pd.DataFrame | None) -> pd.DataFrame:
    """Merge all groups into one frame keyed by (race_id_short, driver)."""
    a = compute_start_expectation(entries, races)
    b = compute_green_pace(laptimes, entries, races)
    c = compute_crew_chief(entries, races)
    d = compute_similar_track_form(entries, races)
    out = a.merge(b, on=["race_id_short", "driver"], how="outer")
    out = out.merge(c, on=["race_id_short", "driver"], how="outer")
    out = out.merge(d, on=["race_id_short", "driver"], how="outer")
    return out.drop_duplicates(["race_id_short", "driver"])
