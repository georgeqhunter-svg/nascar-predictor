"""Offline unit tests for the NASCAR.com scraper's parsing logic."""
from __future__ import annotations

import pandas as pd

from src.scrape.nascar_com import _clean_results, _clean_sessions, track_type_for


def test_clean_results_filters_dnqs_and_flags_dnf():
    fake = [
        # DNQ
        {"finishing_position": 0, "starting_position": 0, "car_number": "78",
         "driver_id": 3973, "driver_fullname": "BJ McLeod", "team_id": 3080,
         "team_name": "Live Fast", "owner_id": 7093, "owner_fullname": "LF",
         "crew_chief_id": 1, "crew_chief_fullname": "X",
         "car_make": "Chevrolet", "car_model": "Camaro", "sponsor": "",
         "qualifying_position": 41, "qualifying_speed": 0,
         "laps_completed": 0, "laps_led": 0, "times_led": 0,
         "points_earned": 0, "playoff_points_earned": 0,
         "finishing_status": "", "disqualified": False,
         "diff_laps": 0, "diff_time": 0},
        # Winner
        {"finishing_position": 1, "starting_position": 18, "car_number": "24",
         "driver_id": 4184, "driver_fullname": "William Byron", "team_id": 1581,
         "team_name": "Hendrick", "owner_id": 1464, "owner_fullname": "Rick Hendrick",
         "crew_chief_id": 6053, "crew_chief_fullname": "Rudy Fugle",
         "car_make": "Chevrolet", "car_model": "Camaro", "sponsor": "Axalta",
         "qualifying_position": 18, "qualifying_speed": 0,
         "laps_completed": 200, "laps_led": 4, "times_led": 1,
         "points_earned": 51, "playoff_points_earned": 5,
         "finishing_status": "Running", "disqualified": False,
         "diff_laps": 0, "diff_time": 0},
        # Crashed out
        {"finishing_position": 30, "starting_position": 32, "car_number": "12",
         "driver_id": 4989, "driver_fullname": "Ryan Blaney", "team_id": 1706,
         "team_name": "Penske", "owner_id": 1465, "owner_fullname": "Roger Penske",
         "crew_chief_id": 9999, "crew_chief_fullname": "Jonathan Hassler",
         "car_make": "Ford", "car_model": "Mustang", "sponsor": "Menards",
         "qualifying_position": 12, "qualifying_speed": 0,
         "laps_completed": 192, "laps_led": 12, "times_led": 3,
         "points_earned": 17, "playoff_points_earned": 1,
         "finishing_status": "Accident", "disqualified": False,
         "diff_laps": 8, "diff_time": 0},
    ]
    df = _clean_results(fake)
    assert df.shape[0] == 2  # DNQ filtered out
    winner = df.iloc[0]
    assert winner["driver"] == "William Byron"
    assert winner["finish_pos"] == 1
    assert winner["is_dnf"] == False  # noqa: E712
    crashed = df.iloc[1]
    assert crashed["is_dnf"] == True  # noqa: E712
    assert crashed["status"] == "Accident"


def test_clean_sessions_flattens_practice_and_qualifying():
    fake = [
        {
            "weekend_run_id": 5470, "race_id": 5385, "run_type": 1,
            "run_name": "Daytona 500 Practice 1",
            "run_date_utc": "2024-02-14T18:00:00",
            "results": [
                {"driver_id": 4184, "driver_name": "William Byron",
                 "car_number": "24", "manufacturer": "Chevrolet",
                 "finishing_position": 1, "best_lap_time": 48.9,
                 "best_lap_speed": 184.05, "best_lap_number": 12,
                 "laps_completed": 18, "delta_leader": 0, "disqualified": False},
                {"driver_id": 3859, "driver_name": "Joey Logano",
                 "car_number": "22", "manufacturer": "Ford",
                 "finishing_position": 2, "best_lap_time": 49.1,
                 "best_lap_speed": 183.30, "best_lap_number": 8,
                 "laps_completed": 15, "delta_leader": -0.2, "disqualified": False},
            ],
        },
        {
            "weekend_run_id": 5471, "race_id": 5385, "run_type": 2,
            "run_name": "Busch Pole Qualifying - Final",
            "run_date_utc": "2024-02-15T01:20:00",
            "results": [
                {"driver_id": 3859, "driver_name": "Joey Logano",
                 "car_number": "22", "manufacturer": "Ford",
                 "finishing_position": 1, "best_lap_time": 49.465,
                 "best_lap_speed": 181.947, "best_lap_number": 1,
                 "laps_completed": 1, "delta_leader": 0, "disqualified": False},
            ],
        },
    ]
    df = _clean_sessions(fake)
    assert df.shape == (3, 16)
    practice = df[df["run_type_label"] == "practice"]
    qual = df[df["run_type_label"] == "qualifying"]
    assert len(practice) == 2
    assert len(qual) == 1
    assert qual.iloc[0]["driver_name"] == "Joey Logano"
    assert qual.iloc[0]["best_lap_speed"] == 181.947


def test_clean_sessions_empty_ok():
    df = _clean_sessions([])
    assert df.shape[0] == 0
    assert "best_lap_time" in df.columns


def test_track_type_lookup():
    assert track_type_for("Daytona International Speedway") == "superspeedway"
    assert track_type_for("Talladega Superspeedway") == "superspeedway"
    assert track_type_for("Martinsville Speedway") == "short"
    assert track_type_for("Watkins Glen International") == "road"
    assert track_type_for("Kansas Speedway") == "intermediate"
    # Unknown track — defaults but doesn't raise.
    assert track_type_for("Somewhere Weird") == "intermediate"
