"""Forward-looking features — signals about current conditions rather than
aggregated past outcomes.

Three groups:

  1. Practice-weekend depth: how far off the fastest driver was this driver
     in practice, and how many practice laps did they run. Both z-scored
     within-race. delta-to-leader is a physical gap in mph, not the sanitized
     best_lap z-score we already use — it captures the actual outlier structure
     of a specific weekend rather than the shape of the field's distribution.

  2. Team-weekend signals: the same qual/practice z-score, averaged over
     the driver's TEAMMATES this weekend. A strong-teammate weekend usually
     means the car's got a good setup that transfers.

  3. Mechanical-DNF risk: rolling rate of DNFs caused by mechanical failures
     (engine, drivetrain, brakes, etc.) in the driver's last N races,
     distinct from accident DNFs which are more random.

Everything is walk-forward: for a given (race, driver), only sessions data
from that weekend and race history strictly before that date are used.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


# Mechanical failure statuses (as distinct from "Accident" / "DVP").
MECHANICAL_STATUSES: set[str] = {
    "Engine", "Drivetrain", "Fuel Pump", "Rear Gear", "Suspension",
    "Handling", "Brakes", "Chassis", "Throttle", "Power Steering",
    "Rear End", "Steering", "Water Pump", "Exhaust", "Fire", "Driveshaft",
    "Electrical", "Transmission", "Overheating", "Clutch", "Ignition",
}

# Crash / incident statuses. "DVP" = damaged-vehicle policy (car retired after
# being unable to return to speed following contact). "Vibration" is ambiguous
# but usually mechanical, so we exclude it.
ACCIDENT_STATUSES: set[str] = {
    "Accident", "DVP", "Crash", "Wreck", "Rear Damage", "Front Damage",
    "Damage", "Retired", "Wall",
}


# --------------------------------------------------------------------------- #
# Practice-weekend depth
# --------------------------------------------------------------------------- #
def practice_depth_by_driver(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per (race, driver): summed practice laps, and gap-to-fastest in mph.
    Practice gap uses the max best_lap_speed across all practice runs for
    that driver; the reference is the fastest driver's speed at that race.
    """
    p = sessions[sessions["run_type_label"] == "practice"].copy()
    if p.empty:
        return pd.DataFrame(columns=["race_id_short", "driver_name",
                                     "practice_gap_mph", "practice_laps"])
    p = p.dropna(subset=["best_lap_speed"])
    best_by = (
        p.groupby(["race_id_short", "driver_name"], as_index=False)
        .agg(best=("best_lap_speed", "max"),
             laps=("laps_completed", "sum"))
    )
    # Race-wide fastest.
    race_max = best_by.groupby("race_id_short")["best"].transform("max")
    best_by["practice_gap_mph"] = race_max - best_by["best"]
    best_by["practice_laps"] = best_by["laps"].fillna(0)
    return best_by[["race_id_short", "driver_name",
                    "practice_gap_mph", "practice_laps"]]


def practice_gap_zscore(depth: pd.DataFrame, race_id: str,
                        drivers_in_order: list[str]) -> np.ndarray:
    """Within-race z-score of practice_gap_mph. Missing -> 0."""
    n = len(drivers_in_order)
    if depth.empty:
        return np.zeros(n)
    sub = depth[depth["race_id_short"] == race_id]
    if sub.empty:
        return np.zeros(n)
    lookup = dict(zip(sub["driver_name"].values, sub["practice_gap_mph"].values))
    vals = np.array([lookup.get(d, np.nan) for d in drivers_in_order], dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() < 3:
        return np.zeros(n)
    valid = vals[mask]
    mu, sigma = valid.mean(), valid.std()
    if sigma < 1e-6:
        return np.zeros(n)
    # NOTE: bigger gap is worse, so we NEGATE the z so that positive = fast.
    return np.where(mask, -(vals - mu) / sigma, 0.0)


def practice_laps_zscore(depth: pd.DataFrame, race_id: str,
                         drivers_in_order: list[str]) -> np.ndarray:
    """Within-race z-score of practice laps completed."""
    n = len(drivers_in_order)
    if depth.empty:
        return np.zeros(n)
    sub = depth[depth["race_id_short"] == race_id]
    if sub.empty:
        return np.zeros(n)
    lookup = dict(zip(sub["driver_name"].values, sub["practice_laps"].values))
    vals = np.array([lookup.get(d, np.nan) for d in drivers_in_order], dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() < 3:
        return np.zeros(n)
    valid = vals[mask]
    mu, sigma = valid.mean(), valid.std()
    if sigma < 1e-6:
        return np.zeros(n)
    return np.where(mask, (vals - mu) / sigma, 0.0)


# --------------------------------------------------------------------------- #
# Team-weekend signals
# --------------------------------------------------------------------------- #
def team_teammate_qual_z(
    qual_by_driver: pd.DataFrame,
    entries: pd.DataFrame,
    race_id: str,
    drivers_in_order: list[str],
    teams_in_order: list[str],
) -> np.ndarray:
    """For each driver, the mean qual z-score of their teammates in this race
    (excluding themselves). Missing (no teammate or no qual) -> 0."""
    from .qualifying import qual_zscore_for_race
    n = len(drivers_in_order)
    z = qual_zscore_for_race(qual_by_driver, race_id, drivers_in_order)
    out = np.zeros(n)
    by_team: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(teams_in_order):
        by_team[t].append(i)
    for i, t in enumerate(teams_in_order):
        peers = [j for j in by_team[t] if j != i]
        if peers:
            out[i] = float(np.mean([z[j] for j in peers]))
    return out


def team_teammate_practice_z(
    practice_by_driver: pd.DataFrame,
    race_id: str,
    drivers_in_order: list[str],
    teams_in_order: list[str],
) -> np.ndarray:
    from .qualifying import practice_zscore_for_race
    n = len(drivers_in_order)
    z = practice_zscore_for_race(practice_by_driver, race_id, drivers_in_order)
    out = np.zeros(n)
    by_team: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(teams_in_order):
        by_team[t].append(i)
    for i, t in enumerate(teams_in_order):
        peers = [j for j in by_team[t] if j != i]
        if peers:
            out[i] = float(np.mean([z[j] for j in peers]))
    return out


# --------------------------------------------------------------------------- #
# Mechanical-DNF rolling risk
# --------------------------------------------------------------------------- #
def compute_crash_risk(
    entries: pd.DataFrame,
    races: pd.DataFrame | None = None,
    window: int = 10,
) -> pd.DataFrame:
    """Walk-forward: for each (race, driver), fraction of last `window` races
    ending in a crash/accident DNF (as distinct from mechanical), and the
    same rate filtered to races at this track type (last 10 relevant races).
    """
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if races is not None and "track_type" not in df.columns:
        df = df.merge(races[["race_id_short", "track_type"]],
                      on="race_id_short", how="left")

    if "status" not in df.columns:
        out = df[["race_id_short", "driver"]].drop_duplicates()
        out["crash_dnf_rate_10"] = 0.0
        out["crash_dnf_rate_at_type_10"] = np.nan
        return out

    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )
    history_all: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    history_type: dict[tuple[str, str], deque] = defaultdict(
        lambda: deque(maxlen=window)
    )
    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        tt_col = sub["track_type"] if "track_type" in sub.columns else [None] * len(sub)
        for drv, status, fp, tt in zip(sub["driver"], sub["status"],
                                       sub["finish_pos"], tt_col):
            h_all = history_all[drv]
            rate_all = (
                sum(1 for s in h_all if s in ACCIDENT_STATUSES) / len(h_all)
                if h_all else np.nan
            )
            h_type = history_type[(drv, tt)] if tt is not None else None
            if h_type and len(h_type) >= 2:
                rate_type = (
                    sum(1 for s in h_type if s in ACCIDENT_STATUSES) / len(h_type)
                )
            else:
                rate_type = np.nan
            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "crash_dnf_rate_10": rate_all,
                "crash_dnf_rate_at_type_10": rate_type,
            })
        # Update history AFTER emitting for this race.
        for drv, status, fp, tt in zip(sub["driver"], sub["status"],
                                       sub["finish_pos"], tt_col):
            if int(fp) > 0:
                history_all[drv].append(status)
                if tt is not None:
                    history_type[(drv, tt)].append(status)
    return pd.DataFrame(rows)


def compute_mechanical_risk(entries: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Walk-forward: for each (race, driver), fraction of the driver's last
    `window` completed races that ended in a mechanical DNF."""
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "status" not in df.columns:
        # No status data — return zeros so downstream is a no-op.
        out = df[["race_id_short", "driver"]].drop_duplicates()
        out["mech_dnf_rate_10"] = 0.0
        return out

    race_order = (
        df.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().index.tolist()
    )
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    rows = []
    for rid in race_order:
        sub = df[df["race_id_short"] == rid]
        for drv, status, fp in zip(sub["driver"], sub["status"], sub["finish_pos"]):
            hist = history[drv]
            if hist:
                rate = sum(1 for s in hist if s in MECHANICAL_STATUSES) / len(hist)
            else:
                rate = np.nan
            rows.append({
                "race_id_short": rid,
                "driver": drv,
                "mech_dnf_rate_10": rate,
            })
        # Update history AFTER emitting for this race.
        for drv, status, fp in zip(sub["driver"], sub["status"], sub["finish_pos"]):
            if int(fp) > 0:  # skip pre-race entries with no result yet
                history[drv].append(status)
    return pd.DataFrame(rows)
