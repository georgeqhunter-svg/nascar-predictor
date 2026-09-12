"""Walk-forward rolling driver-form features.

For each race the model will make a prediction about, we need features that
summarize each driver's recent performance using ONLY past data. This module
does the bookkeeping.

Public entry point: `compute_rolling(entries)` -> DataFrame indexed by
(race_id_short, driver) with columns:

    avg_finish_5, avg_finish_10, avg_finish_20,
    dnf_rate_5, dnf_rate_10,
    top10_rate_5, top10_rate_10,
    season_wins_ytd, season_top5_ytd, season_top10_ytd,
    races_at_type_ytd, avg_finish_at_type_ytd,
    career_races, days_since_last_race
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling(entries: pd.DataFrame) -> pd.DataFrame:
    """Build per-race, per-driver rolling stats from all past races."""
    df = entries.sort_values(["date", "race_id_short", "finish_pos"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # We'll accumulate driver history in dicts as we walk the race timeline.
    from collections import defaultdict, deque

    hist_finish: dict[str, deque] = defaultdict(deque)  # last-20 finish positions
    hist_dnf: dict[str, deque] = defaultdict(deque)     # last-20 DNF flags (0/1)
    season_stats: dict[tuple, dict] = defaultdict(lambda: {"wins": 0, "top5": 0, "top10": 0})
    type_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "sum_finish": 0})
    # NEW: per-driver-per-track history (avg finish and count at THIS specific track).
    track_stats: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "sum_finish": 0, "best": 999}
    )
    # NEW: per-TEAM-per-track-type rolling avg finish (all their drivers).
    team_type_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "sum_finish": 0})
    career_races: dict[str, int] = defaultdict(int)
    last_race_date: dict[str, pd.Timestamp] = {}

    # `track_id` is optional (present in real data via merge from races table).
    has_track_id = "track_id" in df.columns
    if not has_track_id:
        df = df.assign(track_id=0)
    has_team = "team" in df.columns
    if not has_team:
        df = df.assign(team="")

    out_rows = []
    for (race_id, race_date, season, track_type, track_id), g in df.groupby(
        ["race_id_short", "date", "season", "track_type", "track_id"], sort=False
    ):
        team_lookup = dict(zip(g["driver"].values, g["team"].values))
        for driver in g["driver"].values:
            fh = hist_finish[driver]
            dh = hist_dnf[driver]

            def _avg_last(seq: deque, n: int) -> float:
                if not seq:
                    return np.nan
                arr = list(seq)[-n:]
                return float(np.mean(arr))

            def _rate_last(seq: deque, n: int) -> float:
                if not seq:
                    return np.nan
                arr = list(seq)[-n:]
                return float(np.mean(arr))

            ss = season_stats[(driver, season)]
            ts = type_stats[(driver, track_type)]
            trk = track_stats[(driver, track_id)]
            team = team_lookup.get(driver, "")
            tt_team = team_type_stats[(team, track_type)]

            # Momentum: last-3 avg minus previous-3 avg (negative = improving).
            fh_list = list(hist_finish[driver])
            momentum_3 = (
                float(np.mean(fh_list[-3:]) - np.mean(fh_list[-6:-3]))
                if len(fh_list) >= 6 else np.nan
            )

            days_since = (
                (race_date - last_race_date[driver]).days
                if driver in last_race_date
                else np.nan
            )

            out_rows.append({
                "race_id_short": race_id,
                "driver": driver,
                "avg_finish_5": _avg_last(fh, 5),
                "avg_finish_10": _avg_last(fh, 10),
                "avg_finish_20": _avg_last(fh, 20),
                "dnf_rate_5": _rate_last(dh, 5),
                "dnf_rate_10": _rate_last(dh, 10),
                "top10_rate_5": _rate_last(
                    deque(int(f <= 10) for f in list(fh)[-5:]), 5
                ) if fh else np.nan,
                "top10_rate_10": _rate_last(
                    deque(int(f <= 10) for f in list(fh)[-10:]), 10
                ) if fh else np.nan,
                "season_wins_ytd": ss["wins"],
                "season_top5_ytd": ss["top5"],
                "season_top10_ytd": ss["top10"],
                "races_at_type_ytd": ts["n"],
                "avg_finish_at_type_ytd": (ts["sum_finish"] / ts["n"]) if ts["n"] else np.nan,
                "races_at_track": trk["n"],
                "avg_finish_at_track": (trk["sum_finish"] / trk["n"]) if trk["n"] else np.nan,
                "best_finish_at_track": trk["best"] if trk["n"] else np.nan,
                "team_avg_finish_at_type": (tt_team["sum_finish"] / tt_team["n"]) if tt_team["n"] else np.nan,
                "team_races_at_type": tt_team["n"],
                "momentum_3": momentum_3,
                "career_races": career_races[driver],
                "days_since_last_race": days_since,
            })

        # NOW update state with this race's outcomes (so downstream races see this data).
        for _, row in g.iterrows():
            driver = row["driver"]
            finish = int(row["finish_pos"])
            is_dnf = bool(row["is_dnf"])

            hist_finish[driver].append(finish)
            hist_dnf[driver].append(int(is_dnf))
            if len(hist_finish[driver]) > 20:
                hist_finish[driver].popleft()
            if len(hist_dnf[driver]) > 20:
                hist_dnf[driver].popleft()

            ss = season_stats[(driver, season)]
            if finish == 1:
                ss["wins"] += 1
            if finish <= 5:
                ss["top5"] += 1
            if finish <= 10:
                ss["top10"] += 1

            ts = type_stats[(driver, track_type)]
            ts["n"] += 1
            ts["sum_finish"] += finish

            trk = track_stats[(driver, track_id)]
            trk["n"] += 1
            trk["sum_finish"] += finish
            if finish < trk["best"]:
                trk["best"] = finish

            team = team_lookup.get(driver, "")
            tt_team = team_type_stats[(team, track_type)]
            tt_team["n"] += 1
            tt_team["sum_finish"] += finish

            career_races[driver] += 1
            last_race_date[driver] = race_date

    return pd.DataFrame(out_rows)
