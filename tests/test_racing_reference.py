"""Smoke tests hitting real Racing Reference pages. Slow; assumes network + cache.

Run with: pytest -m network
"""
from __future__ import annotations

from datetime import date

import pytest

from src.scrape.racing_reference import fetch_race_detail, fetch_season_index

pytestmark = pytest.mark.network


def test_season_index_2024_has_36_races():
    stubs = fetch_season_index(2024)
    assert len(stubs) >= 35  # Cup season is 36 points races
    assert all(s.season == 2024 for s in stubs)
    assert stubs[0].race_number == 1
    assert stubs[0].date.year == 2024


def test_2024_daytona_500_details():
    stubs = fetch_season_index(2024)
    daytona = next(s for s in stubs if "Daytona_500" in s.slug)
    detail = fetch_race_detail(daytona)
    assert detail.entries.shape[0] >= 38  # 40-car field, some may be DNQs
    winner_row = detail.entries[detail.entries["finish_pos"] == 1].iloc[0]
    assert "Byron" in winner_row["driver"]
    assert detail.race_id_short == "2024-01"
