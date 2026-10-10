"""Which drivers get a walk-forward snapshot row for a race.

BUG (found 2026-10-09): the loop / restart / pit / team-pit / tire rolling
builders emitted rows only for drivers present in THAT race's own loopstats /
laptimes. Historical races have that data; an UPCOMING race does not — so in
live predictions all 26 of those GBM features were NaN for every driver,
while the model was trained (and backtested) with them filled in.

The rows are pre-race snapshots of history, so they never need the race's own
data. With EMIT_ALL_ENTRIES on, a snapshot is emitted for every driver on the
race's entry list (union with whoever appears in the race's data).

build_features() calls set_entries(entries) before running the builders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMIT_ALL_ENTRIES = True

_RACE_DRIVERS: dict[str, set[int]] = {}


def set_entries(entries: pd.DataFrame | None) -> None:
    global _RACE_DRIVERS
    _RACE_DRIVERS = {}
    if entries is None or entries.empty or "driver_id" not in entries.columns:
        return
    e = entries[["race_id_short", "driver_id"]].copy()
    e["driver_id"] = pd.to_numeric(e["driver_id"], errors="coerce")
    e = e.dropna()
    for rid, g in e.groupby("race_id_short"):
        _RACE_DRIVERS[rid] = set(int(x) for x in g["driver_id"])


def race_drivers(race_id: str, lt_race: pd.DataFrame | None) -> np.ndarray:
    """Drivers to emit a pre-race snapshot for."""
    have = set()
    if lt_race is not None and not lt_race.empty and "driver_id" in lt_race.columns:
        have = set(int(x) for x in pd.to_numeric(lt_race["driver_id"], errors="coerce").dropna())
    if EMIT_ALL_ENTRIES:
        have |= _RACE_DRIVERS.get(race_id, set())
    return np.array(sorted(have), dtype=int)
