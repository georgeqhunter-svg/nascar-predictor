"""Per-race qualifying-speed features from sessions.parquet.

The signal we care about is *relative* qualifying speed within a race — how
much faster than the field average was this driver? Absolute mph varies by
track type (superspeedways ~180 mph, road courses ~80 mph) so we always
z-score within a race.

Practice speed is captured too as a secondary same-week signal. Some race
weekends have no practice at all (recent NASCAR reduced practice); the helper
returns NaN in that case and downstream code treats it as neutral (z=0).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Aggregators
# --------------------------------------------------------------------------- #
def qualifying_best_by_driver(sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per (race_id_short, driver_name) with each driver's best
    qualifying lap speed across all qualifying rounds (Q1/Q2/Final).
    """
    q = sessions[sessions["run_type_label"] == "qualifying"].copy()
    if q.empty:
        return pd.DataFrame(columns=["race_id_short", "driver_name", "qual_speed"])
    q = q.dropna(subset=["best_lap_speed"])
    out = (
        q.groupby(["race_id_short", "driver_name"], as_index=False)["best_lap_speed"]
        .max()
        .rename(columns={"best_lap_speed": "qual_speed"})
    )
    return out


def practice_best_by_driver(sessions: pd.DataFrame) -> pd.DataFrame:
    """Best single-lap practice speed across all practice sessions for the
    race weekend, per (race, driver). NaN if no practice was held.
    """
    p = sessions[sessions["run_type_label"] == "practice"].copy()
    if p.empty:
        return pd.DataFrame(columns=["race_id_short", "driver_name", "practice_speed"])
    p = p.dropna(subset=["best_lap_speed"])
    out = (
        p.groupby(["race_id_short", "driver_name"], as_index=False)["best_lap_speed"]
        .max()
        .rename(columns={"best_lap_speed": "practice_speed"})
    )
    return out


def practice_zscore_for_race(
    practice_by_driver: pd.DataFrame,
    race_id_short: str,
    drivers_in_order: list[str],
) -> np.ndarray:
    """Z-score practice speed within a race. Missing drivers -> 0.0 (neutral)."""
    n = len(drivers_in_order)
    if practice_by_driver.empty:
        return np.zeros(n, dtype=np.float64)
    subset = practice_by_driver[practice_by_driver["race_id_short"] == race_id_short]
    if subset.empty:
        return np.zeros(n, dtype=np.float64)
    lookup = dict(zip(subset["driver_name"].values, subset["practice_speed"].values))
    speeds = np.array([lookup.get(d, np.nan) for d in drivers_in_order], dtype=np.float64)
    mask = ~np.isnan(speeds)
    if mask.sum() < 3:
        return np.zeros(n, dtype=np.float64)
    valid = speeds[mask]
    mu = valid.mean()
    sigma = valid.std()
    if sigma < 1e-6:
        return np.zeros(n, dtype=np.float64)
    return np.where(mask, (speeds - mu) / sigma, 0.0)


# --------------------------------------------------------------------------- #
# Per-race aligned lookup
# --------------------------------------------------------------------------- #
def qual_zscore_for_race(
    qual_by_driver: pd.DataFrame,
    race_id_short: str,
    drivers_in_order: list[str],
) -> np.ndarray:
    """Return an array of z-scored qualifying speeds aligned to `drivers_in_order`.

    Missing drivers (didn't qualify or no session data) get 0.0 (neutral).
    If no qualifying data exists for the race at all, returns all zeros.
    """
    n = len(drivers_in_order)
    subset = qual_by_driver[qual_by_driver["race_id_short"] == race_id_short]
    if subset.empty:
        return np.zeros(n, dtype=np.float64)

    lookup = dict(zip(subset["driver_name"].values, subset["qual_speed"].values))
    speeds = np.array([lookup.get(d, np.nan) for d in drivers_in_order], dtype=np.float64)
    mask = ~np.isnan(speeds)
    if mask.sum() < 3:
        return np.zeros(n, dtype=np.float64)

    valid = speeds[mask]
    mu = valid.mean()
    sigma = valid.std()
    if sigma < 1e-6:
        return np.zeros(n, dtype=np.float64)

    z = np.where(mask, (speeds - mu) / sigma, 0.0)
    return z
