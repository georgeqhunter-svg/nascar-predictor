"""Tests for qualifying feature helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.qualifying import (
    qual_zscore_for_race,
    qualifying_best_by_driver,
)


def _make_sessions() -> pd.DataFrame:
    rows = [
        # Qualifying (2 rounds) for race R1
        {"race_id_short": "R1", "run_type_label": "qualifying",
         "driver_name": "A", "best_lap_speed": 180.0},
        {"race_id_short": "R1", "run_type_label": "qualifying",
         "driver_name": "A", "best_lap_speed": 181.5},  # better round
        {"race_id_short": "R1", "run_type_label": "qualifying",
         "driver_name": "B", "best_lap_speed": 179.0},
        {"race_id_short": "R1", "run_type_label": "qualifying",
         "driver_name": "C", "best_lap_speed": 180.5},
        # Practice for R1 — not read by qualifying_best_by_driver
        {"race_id_short": "R1", "run_type_label": "practice",
         "driver_name": "A", "best_lap_speed": 182.0},
        # Race R2 qualifying with tight spread
        {"race_id_short": "R2", "run_type_label": "qualifying",
         "driver_name": "A", "best_lap_speed": 100.0},
        {"race_id_short": "R2", "run_type_label": "qualifying",
         "driver_name": "B", "best_lap_speed": 100.1},
        {"race_id_short": "R2", "run_type_label": "qualifying",
         "driver_name": "C", "best_lap_speed": 99.9},
    ]
    return pd.DataFrame(rows)


def test_qualifying_best_takes_max_across_rounds():
    q = qualifying_best_by_driver(_make_sessions())
    r1_a = q[(q["race_id_short"] == "R1") & (q["driver_name"] == "A")].iloc[0]
    assert r1_a["qual_speed"] == 181.5  # not 180


def test_zscore_alignment_and_missing_driver():
    q = qualifying_best_by_driver(_make_sessions())
    # D wasn't in qualifying — should get 0.0
    z = qual_zscore_for_race(q, "R1", ["A", "B", "C", "D"])
    assert z.shape == (4,)
    assert z[3] == 0.0
    # A had the fastest lap; its z should be positive and largest.
    assert z[0] > z[1] and z[0] > z[2]
    # Mean of observed z-scores should be ~0.
    assert abs(z[:3].mean()) < 1e-8


def test_missing_race_returns_zeros():
    q = qualifying_best_by_driver(_make_sessions())
    z = qual_zscore_for_race(q, "R_UNKNOWN", ["A", "B", "C"])
    assert (z == 0).all()
