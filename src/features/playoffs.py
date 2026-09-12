"""Playoff-status features.

Two features:

* ``is_playoff_driver`` — during playoff races, 1 iff this driver is one of
  the ~16 championship contenders. During the regular season, always 0.
* ``is_elimination_race`` — 1 iff this race is the last race of a playoff
  round (Round of 16, 12, 8, or the Championship).

Playoff drivers are derived post-hoc from `playoff_points_earned` in the
data, but since playoff eligibility is fixed at the end of race 26 of a
season, we can attach the same set to every subsequent race in that season
without any leak.
"""
from __future__ import annotations

import pandas as pd


def derive_playoff_drivers(entries: pd.DataFrame, races: pd.DataFrame) -> set[tuple[int, str]]:
    """Return set of (season, driver_name) tuples that are playoff drivers.

    We identify them as drivers who earned any playoff points in a playoff
    race (playoff_round > 0). NASCAR only awards playoff_points_earned to
    the 16 playoff drivers during playoff races.
    """
    if "playoff_round" not in races.columns or "playoff_points" not in entries.columns:
        return set()
    playoff_race_ids = races.loc[races["playoff_round"] > 0, "race_id_short"]
    sub = entries[entries["race_id_short"].isin(playoff_race_ids)].copy()
    sub["playoff_points"] = pd.to_numeric(sub["playoff_points"], errors="coerce").fillna(0)
    # Any driver with cumulative playoff_points > 0 across playoff races is in.
    tallies = sub.groupby(["season", "driver"])["playoff_points"].sum()
    return set(tallies[tallies > 0].index)


def derive_elimination_races(races: pd.DataFrame) -> set[str]:
    """Return the set of race_id_short values that are elimination races
    (last race of each playoff round OR the championship itself).
    """
    if "playoff_round" not in races.columns:
        return set()
    playoff = races[races["playoff_round"] > 0].copy()
    if playoff.empty:
        return set()
    # For each (season, playoff_round), the elimination race is the one with
    # the largest race_number (or last date if race_number missing).
    if "race_number" in playoff.columns:
        idx = playoff.groupby(["season", "playoff_round"])["race_number"].idxmax()
    else:
        idx = playoff.groupby(["season", "playoff_round"])["date"].idxmax()
    return set(playoff.loc[idx, "race_id_short"])
