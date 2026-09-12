"""Backfill Cup Series race data from NASCAR.com into parquet.

Usage:
    python -m src.cli.backfill --seasons 2022 2023 2024 2025 2026

Outputs (under data/processed/):
    races.parquet    one row per points race with metadata
    entries.parquet  one row per race x driver with finishing info

Uses NASCAR.com's public cacher endpoints (see src.scrape.nascar_com).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from ..scrape.nascar_com import (
    fetch_race_detail,
    fetch_season_index,
    track_type_for,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

# NASCAR.com endpoints handle rapid queries fine, but be polite.
_REQUEST_INTERVAL = 0.5


def backfill(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    race_rows = []
    entry_rows = []
    session_rows = []
    for year in seasons:
        log.info("Fetching season index %s", year)
        try:
            stubs = fetch_season_index(year)
        except Exception as e:
            log.exception("Failed season index %s: %s", year, e)
            continue
        log.info("  %s points races", len(stubs))

        for stub in tqdm(stubs, desc=str(year), unit="race"):
            try:
                detail = fetch_race_detail(stub)
            except Exception as e:
                log.exception("Failed race %s %s: %s", year, stub.race_name, e)
                continue

            tt = track_type_for(stub.track_name)
            race_id_short = f"{year}-{stub.race_id}"
            race_row = {
                "race_id_short": race_id_short,
                "season": year,
                "date": stub.race_date,
                "race_id_nascar": stub.race_id,
                "race_name": stub.race_name,
                "track_id": stub.track_id,
                "track_name": stub.track_name,
                "track_type": tt,
                "restrictor_plate": stub.restrictor_plate,
                "scheduled_laps": stub.scheduled_laps,
                "actual_laps": stub.actual_laps,
                "stage_1_laps": stub.stage_1_laps,
                "stage_2_laps": stub.stage_2_laps,
                "stage_3_laps": stub.stage_3_laps,
                "scheduled_distance": detail.scheduled_distance,
                "actual_distance": detail.actual_distance,
                "cars_in_field": detail.number_of_cars_in_field,
                "pole_driver_id": detail.pole_winner_driver_id,
                "pole_speed": detail.pole_winner_speed,
                "lead_changes": detail.number_of_lead_changes,
                "cautions": detail.number_of_cautions,
                "caution_laps": detail.number_of_caution_laps,
                "average_speed": detail.average_speed,
                "total_race_time": detail.total_race_time,
                "margin_of_victory": detail.margin_of_victory,
                "winner_driver_id": detail.winner_driver_id,
                "playoff_round": detail.playoff_round,
            }
            race_rows.append(race_row)

            entries = detail.entries.copy()
            if not entries.empty:
                entries.insert(0, "race_id_short", race_id_short)
                entries.insert(0, "season", year)
                entries.insert(0, "date", pd.Timestamp(stub.race_date))
                entries.insert(3, "track_type", tt)
                entry_rows.append(entries)

            sessions = detail.sessions.copy()
            if not sessions.empty:
                sessions.insert(0, "race_id_short", race_id_short)
                sessions.insert(0, "season", year)
                sessions.insert(0, "date", pd.Timestamp(stub.race_date))
                sessions.insert(3, "track_type", tt)
                session_rows.append(sessions)

            time.sleep(_REQUEST_INTERVAL)

    races = pd.DataFrame(race_rows)
    entries = pd.concat(entry_rows, ignore_index=True) if entry_rows else pd.DataFrame()
    sessions = pd.concat(session_rows, ignore_index=True) if session_rows else pd.DataFrame()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    races.to_parquet(PROCESSED / "races.parquet", index=False)
    entries.to_parquet(PROCESSED / "entries.parquet", index=False)
    sessions.to_parquet(PROCESSED / "sessions.parquet", index=False)
    log.info("Wrote %s races, %s entries, %s session rows", len(races), len(entries), len(sessions))
    return races, entries, sessions


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    backfill(args.seasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
