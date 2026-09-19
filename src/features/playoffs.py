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

    Uses REGULAR SEASON data only. Previously this function summed
    playoff_points across playoff races themselves — which meant training
    labels for playoff race 1 were informed by outcomes of playoff races
    2-10 (future info). Kimi's audit caught this: ~1,440 training entries
    carried leaky labels while the 2026 backtest season had
    playoff_points=0 across the board and never used the feature at all.

    Playoff eligibility is determined by regular-season performance
    (points through race 26). Playoff points earned in-season for stage
    wins and race wins are the cleanest single signal. Every driver in
    the top 16 of regular-season points, plus every regular-season race
    winner, makes the playoffs.
    """
    if "playoff_round" not in races.columns or "playoff_points" not in entries.columns:
        return set()
    # Regular-season races only — NO playoff races in the sum.
    reg_race_ids = races.loc[races["playoff_round"] == 0, "race_id_short"]
    sub = entries[entries["race_id_short"].isin(reg_race_ids)].copy()
    sub["playoff_points"] = pd.to_numeric(sub["playoff_points"], errors="coerce").fillna(0)
    # Sum regular-season playoff points per (season, driver).
    tallies = sub.groupby(["season", "driver"])["playoff_points"].sum()
    # Any driver with regular-season playoff points > 0 is very likely a
    # playoff driver (stage/race winners qualify). This is a slightly
    # conservative approximation — a driver who makes the field on
    # points-only with zero wins and zero stage points gets excluded —
    # but it's HONEST at every point in the season.
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
