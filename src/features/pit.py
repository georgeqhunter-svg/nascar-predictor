"""Pit-crew performance features inferred from lap-times.

Pit stops show up as anomalously slow laps (typical green-flag pit stop is
15-25s slower than a normal green-flag lap). We detect these and, for each
one, compare the driver's position a few laps BEFORE the stop to their
position a few laps AFTER. Places-gained ~= a proxy for pit-crew quality
(pit road speed + tire change + fuel time).

We also compute the median stop-time-delta (how much slower the stop lap was
vs the field median for that same lap) as a raw efficiency measure.

Aggregated per driver, walk-forward over the last 5/10 races.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


PIT_RATIO_THRESHOLD = 1.4    # lap_time / field_median > this = probable pit
POST_PIT_WINDOW = 3          # laps after pit to measure post-cycle position


def per_race_pit_stats(laptimes_one_race: pd.DataFrame) -> pd.DataFrame:
    """Detect pit stops for each driver in a race and compute:
        n_stops             number of pit stops detected
        avg_places_gained   mean positions gained across those stops
                            (positive = moved up; negative = lost track position)
        avg_pit_time_delta  mean (driver's pit-lap time - field median lap time)
                            across those stops. Lower = faster stop.
    """
    lt = laptimes_one_race.copy()
    lt["lap_time"] = pd.to_numeric(lt["lap_time"], errors="coerce")
    lt = lt.dropna(subset=["lap_time", "running_pos"])
    if lt.empty:
        return pd.DataFrame(columns=[
            "driver_id", "n_stops", "avg_places_gained", "avg_pit_time_delta",
        ])

    lap_median = lt.groupby("lap")["lap_time"].median()
    if lap_median.empty:
        return pd.DataFrame(columns=[
            "driver_id", "n_stops", "avg_places_gained", "avg_pit_time_delta",
        ])

    pos = lt.pivot_table(index="lap", columns="driver_id", values="running_pos", aggfunc="last")
    lap_time_wide = lt.pivot_table(index="lap", columns="driver_id", values="lap_time", aggfunc="last")
    pos = pos.sort_index()
    lap_time_wide = lap_time_wide.sort_index()

    stats: dict[int, dict] = defaultdict(lambda: {
        "n_stops": 0, "gained": 0.0, "time_delta": 0.0,
    })
    for drv in lap_time_wide.columns:
        drv_col = lap_time_wide[drv]
        for lap, drv_time in drv_col.dropna().items():
            median_time = lap_median.get(lap, np.nan)
            if not np.isfinite(median_time) or median_time <= 0:
                continue
            ratio = drv_time / median_time
            if ratio < PIT_RATIO_THRESHOLD:
                continue
            prev_lap = lap - 1
            after_lap = lap + POST_PIT_WINDOW
            if prev_lap not in pos.index or after_lap not in pos.index:
                continue
            pos_before = pos.loc[prev_lap, drv]
            pos_after = pos.loc[after_lap, drv]
            if pd.isna(pos_before) or pd.isna(pos_after):
                continue
            gained = float(pos_before) - float(pos_after)  # +ve = moved up
            stats[int(drv)]["n_stops"] += 1
            stats[int(drv)]["gained"] += gained
            stats[int(drv)]["time_delta"] += float(drv_time - median_time)

    rows = []
    for drv, s in stats.items():
        n = s["n_stops"]
        rows.append({
            "driver_id": drv,
            "n_stops": n,
            "avg_places_gained": s["gained"] / n if n else 0.0,
            "avg_pit_time_delta": s["time_delta"] / n if n else 0.0,
        })
    return pd.DataFrame(rows)


def compute_rolling_pit(laptimes: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward per (race_id_short, driver_id) rolling means of pit metrics.

    Rolling values reflect data known BEFORE the race, not the race's own stops.
    """
    if laptimes.empty:
        return pd.DataFrame(columns=[
            "race_id_short", "driver_id",
            "pit_gain_5", "pit_gain_10", "pit_time_delta_10",
        ])

    r = races[["race_id_short", "date"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    race_order = r.sort_values("date")["race_id_short"].tolist()

    hist_gain: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
    hist_time_delta: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))

    out_rows = []
    for race_id in race_order:
        lt_race = laptimes[laptimes["race_id_short"] == race_id]

        drivers_this_race = lt_race["driver_id"].dropna().unique()
        for drv in drivers_this_race:
            drv = int(drv)
            g = hist_gain[drv]
            t = hist_time_delta[drv]
            g10 = list(g)[-10:]
            out_rows.append({
                "race_id_short": race_id,
                "driver_id": drv,
                "pit_gain_5": float(np.mean(list(g)[-5:])) if g else np.nan,
                "pit_gain_10": float(np.mean(g10)) if g else np.nan,
                "pit_gain_std_10": float(np.std(g10)) if len(g10) >= 3 else np.nan,
                "pit_time_delta_10": float(np.mean(list(t)[-10:])) if t else np.nan,
            })

        stats = per_race_pit_stats(lt_race)
        for _, row in stats.iterrows():
            drv = int(row["driver_id"])
            if row["n_stops"] > 0:
                hist_gain[drv].append(float(row["avg_places_gained"]))
                hist_time_delta[drv].append(float(row["avg_pit_time_delta"]))

    return pd.DataFrame(out_rows)


def compute_rolling_pit_team(
    laptimes: pd.DataFrame,
    entries: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Team-level walk-forward rolling pit metrics.

    Motivation: crews shuffle across teammates (mid-season swaps for
    performance, playoff-window optimization). A pure driver-attributed
    rolling mean carries stale-crew signal. Team-level aggregation captures
    the org's current pit capability across all their entries, which for
    multi-car teams may reflect current crew quality better than a specific
    driver's 5-race window.

    We emit BOTH the mean and the STD across the team's drivers per race —
    high std = crews differ within the org (typical of Hendrick/JGR),
    low std = pretty even (typical of single-car teams).

    Features:
        team_pit_gain_5, team_pit_gain_10
        team_pit_gain_std_5, team_pit_gain_std_10
        team_pit_time_delta_10
        team_pit_n_stops_10  (sample-size guard)
    """
    empty_cols = [
        "race_id_short", "driver_id",
        "team_pit_gain_5", "team_pit_gain_10",
        "team_pit_gain_std_5", "team_pit_gain_std_10",
        "team_pit_time_delta_10", "team_pit_n_stops_10",
    ]
    if laptimes.empty or entries.empty:
        return pd.DataFrame(columns=empty_cols)

    r = races[["race_id_short", "date"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    race_order = r.sort_values("date")["race_id_short"].tolist()

    # (race_id, driver_id) -> team.
    ent = entries[["race_id_short", "driver_id", "team"]].copy()
    ent["driver_id"] = pd.to_numeric(ent["driver_id"], errors="coerce")
    ent = ent.dropna(subset=["driver_id", "team"])
    ent["driver_id"] = ent["driver_id"].astype(int)
    driver_team_map: dict[tuple[str, int], str] = {
        (row.race_id_short, row.driver_id): row.team
        for row in ent.itertuples(index=False)
    }

    # Per-team history of (gain, time_delta) samples, one entry per driver-race.
    hist_gain: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    hist_time_delta: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
    hist_n_stops: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))

    out_rows = []
    for race_id in race_order:
        lt_race = laptimes[laptimes["race_id_short"] == race_id]
        drivers_this_race = lt_race["driver_id"].dropna().unique()

        # Emit BEFORE ingesting this race's stops.
        for drv in drivers_this_race:
            drv = int(drv)
            team = driver_team_map.get((race_id, drv))
            if team is None or team == "":
                # No team info -> emit NaNs, no team rollup available.
                out_rows.append({
                    "race_id_short": race_id, "driver_id": drv,
                    "team_pit_gain_5": np.nan, "team_pit_gain_10": np.nan,
                    "team_pit_gain_std_5": np.nan, "team_pit_gain_std_10": np.nan,
                    "team_pit_time_delta_10": np.nan, "team_pit_n_stops_10": 0,
                })
                continue
            g = list(hist_gain[team])
            t = list(hist_time_delta[team])
            n = list(hist_n_stops[team])
            g5, g10 = g[-5:], g[-10:]
            t10 = t[-10:]
            n10 = n[-10:]
            out_rows.append({
                "race_id_short": race_id, "driver_id": drv,
                "team_pit_gain_5": float(np.mean(g5)) if g5 else np.nan,
                "team_pit_gain_10": float(np.mean(g10)) if g10 else np.nan,
                "team_pit_gain_std_5": float(np.std(g5)) if len(g5) >= 3 else np.nan,
                "team_pit_gain_std_10": float(np.std(g10)) if len(g10) >= 3 else np.nan,
                "team_pit_time_delta_10": float(np.mean(t10)) if t10 else np.nan,
                "team_pit_n_stops_10": int(sum(n10)) if n10 else 0,
            })

        # NOW ingest this race for downstream races.
        stats = per_race_pit_stats(lt_race)
        for _, row in stats.iterrows():
            drv = int(row["driver_id"])
            if row["n_stops"] <= 0:
                continue
            team = driver_team_map.get((race_id, drv))
            if team is None or team == "":
                continue
            hist_gain[team].append(float(row["avg_places_gained"]))
            hist_time_delta[team].append(float(row["avg_pit_time_delta"]))
            hist_n_stops[team].append(int(row["n_stops"]))

    return pd.DataFrame(out_rows)
