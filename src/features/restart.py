"""Restart-performance features from lap-times.

We detect caution windows by looking at the field-median lap time per lap:
green flag laps run near a baseline (median across all completed laps by all
drivers), while caution laps run 1.8-3x slower.

For each restart (first green lap after a caution window ends), we compute
places gained / lost per driver over the next K green laps. Aggregating gives
a per-race per-driver 'restart_delta' signal that we then roll walk-forward.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


CAUTION_MULT = 1.8       # lap time / baseline > this = caution lap
RESTART_WINDOW = 3       # laps after caution to measure places gained


def detect_caution_windows(lap_medians: pd.Series, baseline: float) -> list[tuple[int, int]]:
    """Return list of (caution_start_lap, caution_end_lap) tuples.

    Any lap where median lap time > CAUTION_MULT * baseline is a caution lap.
    Consecutive caution laps form a window.
    """
    threshold = baseline * CAUTION_MULT
    is_caution = lap_medians > threshold
    windows: list[tuple[int, int]] = []
    start = None
    for lap, val in is_caution.items():
        if val:
            if start is None:
                start = int(lap)
            end = int(lap)
        else:
            if start is not None:
                windows.append((start, end))
                start = None
    if start is not None:
        windows.append((start, end))
    return windows


def per_race_restart_stats(laptimes_one_race: pd.DataFrame) -> pd.DataFrame:
    """For each (driver) in this race, compute:
        n_restarts     -- number of restarts they participated in
        avg_gain       -- mean places gained over `RESTART_WINDOW` laps post-restart
                          (positive = gained places).
    """
    lt = laptimes_one_race.copy()
    lt["lap_time"] = pd.to_numeric(lt["lap_time"], errors="coerce")
    lt = lt.dropna(subset=["lap_time", "running_pos"])
    if lt.empty:
        return pd.DataFrame(columns=["driver_id", "n_restarts", "avg_gain"])

    # Baseline lap time: median of all lap times, filter out obvious caution laps
    # by first taking the lower half.
    all_times = lt["lap_time"].to_numpy()
    baseline = float(np.median(all_times[all_times <= np.median(all_times)]))
    if baseline <= 0 or not np.isfinite(baseline):
        baseline = float(np.median(all_times))

    lap_medians = lt.groupby("lap")["lap_time"].median()
    windows = detect_caution_windows(lap_medians, baseline)
    if not windows:
        return (
            lt[["driver_id"]].drop_duplicates()
            .assign(n_restarts=0, avg_gain=0.0)
        )

    # Wide pivot: rows = lap, cols = driver_id, vals = running_pos.
    pos = lt.pivot_table(index="lap", columns="driver_id", values="running_pos", aggfunc="last")
    pos = pos.sort_index()

    driver_gains: dict[int, list[float]] = defaultdict(list)
    for _c_start, c_end in windows:
        # Position at end of caution.
        if c_end not in pos.index:
            continue
        pos_before = pos.loc[c_end]
        target_lap = c_end + RESTART_WINDOW
        if target_lap not in pos.index:
            # Race ended within the restart window; use the last available lap.
            if pos.index.max() > c_end:
                target_lap = int(pos.index.max())
            else:
                continue
        pos_after = pos.loc[target_lap]
        gain = pos_before - pos_after  # positive means moved up (lower number)
        for drv_id, g in gain.items():
            if pd.isna(g) or pd.isna(pos_before.get(drv_id)):
                continue
            driver_gains[int(drv_id)].append(float(g))

    rows = []
    for drv_id, gains in driver_gains.items():
        rows.append({
            "driver_id": drv_id,
            "n_restarts": len(gains),
            "avg_gain": float(np.mean(gains)) if gains else 0.0,
        })
    return pd.DataFrame(rows)


def compute_rolling_restart(
    laptimes: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Per (race_id_short, driver_id), return rolling means of restart gain
    over the driver's LAST 5 and 10 races (excluding this one).

    Also emits the raw per-race avg_gain and n_restarts as the "this race"
    values (post-hoc — only useful for feature-importance or diagnostics).
    """
    if laptimes.empty:
        return pd.DataFrame(columns=[
            "race_id_short", "driver_id",
            "restart_gain_5", "restart_gain_10",
            "restart_count_10",
        ])

    r = races[["race_id_short", "date"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    race_order = r.sort_values("date")["race_id_short"].tolist()

    hist_gain: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
    hist_count: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))

    out_rows = []
    for race_id in race_order:
        lt_race = laptimes[laptimes["race_id_short"] == race_id]
        # Emit snapshot BEFORE ingesting this race.
        drivers_this_race = lt_race["driver_id"].dropna().unique()
        for drv in drivers_this_race:
            drv = int(drv)
            g = hist_gain[drv]
            c = hist_count[drv]
            g10 = list(g)[-10:]
            out_rows.append({
                "race_id_short": race_id,
                "driver_id": drv,
                "restart_gain_5": float(np.mean(list(g)[-5:])) if g else np.nan,
                "restart_gain_10": float(np.mean(g10)) if g else np.nan,
                "restart_gain_std_10": float(np.std(g10)) if len(g10) >= 3 else np.nan,
                "restart_count_10": float(np.mean(list(c)[-10:])) if c else np.nan,
            })

        # Now compute this race's stats and INGEST for future races.
        stats = per_race_restart_stats(lt_race)
        for _, row in stats.iterrows():
            drv = int(row["driver_id"])
            hist_gain[drv].append(float(row["avg_gain"]))
            hist_count[drv].append(float(row["n_restarts"]))

    return pd.DataFrame(out_rows)
