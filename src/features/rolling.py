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
    type_hist: dict[tuple, deque] = defaultdict(deque)    # last-8 finishes at this track type
    type_count: dict[tuple, int] = defaultdict(int)       # career races at this track type
    # Per-driver-per-track history: career count/best + last-5 finishes window.
    track_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "best": 999})
    track_hist: dict[tuple, deque] = defaultdict(deque)   # last-5 finishes at this track
    # Per-TEAM-per-track-type: career count + last-30 team finishes window.
    team_type_hist: dict[tuple, deque] = defaultdict(deque)
    team_type_count: dict[tuple, int] = defaultdict(int)
    # Teammate battle history: one entry per driver per valid teammate race.
    tm_wr_hist: dict[str, deque] = defaultdict(deque)
    tm_adj_hist: dict[str, deque] = defaultdict(deque)
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
            th = type_hist[(driver, track_type)]
            trk = track_stats[(driver, track_id)]
            tkh = track_hist[(driver, track_id)]
            team = team_lookup.get(driver, "")
            tth = team_type_hist[(team, track_type)]
            tmh = tm_wr_hist[driver]
            tmah = tm_adj_hist[driver]

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
                "races_at_type_ytd": type_count[(driver, track_type)],
                "avg_finish_at_type_ytd": _avg_last(th, 8),
                "races_at_track": trk["n"],
                "avg_finish_at_track": _avg_last(tkh, 5),
                "best_finish_at_track": trk["best"] if trk["n"] else np.nan,
                "team_avg_finish_at_type": _avg_last(tth, 30),
                "team_races_at_type": team_type_count[(team, track_type)],
                "tm_wpct_20": _avg_last(tmh, 20) if tmh else np.nan,
                "tm_adj_wpct_20": (0.5 + _avg_last(tmah, 20)) if tmah else np.nan,
                "tm_races_20": min(len(tmh), 20),
                "momentum_3": momentum_3,
                "career_races": career_races[driver],
                "days_since_last_race": days_since,
            })

        # Teammate battles: update BEFORE hist_finish so expectations use only
        # pre-race rolling form. Excludes pairs where either driver DNF'd or was
        # unclassified; multi-car teams contribute that race's fraction beaten.
        for team_name, tg in g.groupby("team", sort=False):
            if team_name == "" or len(tg) < 2:
                continue
            for _, arow in tg.iterrows():
                a = arow["driver"]
                if bool(arow["is_dnf"]) or int(arow["finish_pos"]) <= 0:
                    continue
                fa = _avg_last(hist_finish[a], 10)
                obs_list, exp_list = [], []
                for _, brow in tg.iterrows():
                    b = brow["driver"]
                    if b == a or bool(brow["is_dnf"]) or int(brow["finish_pos"]) <= 0:
                        continue
                    fb = _avg_last(hist_finish[b], 10)
                    obs_list.append(float(int(arow["finish_pos"]) < int(brow["finish_pos"])))
                    if pd.notna(fa) and pd.notna(fb):
                        exp_list.append(float(1.0 / (1.0 + np.exp(-(fb - fa) / 6.0))))
                    else:
                        exp_list.append(0.5)
                if obs_list:
                    tm_wr_hist[a].append(float(np.mean(obs_list)))
                    tm_adj_hist[a].append(float(np.mean(obs_list) - np.mean(exp_list)))
                    if len(tm_wr_hist[a]) > 20:
                        tm_wr_hist[a].popleft()
                    if len(tm_adj_hist[a]) > 20:
                        tm_adj_hist[a].popleft()

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

            thu = type_hist[(driver, track_type)]
            thu.append(finish)
            if len(thu) > 8:
                thu.popleft()
            type_count[(driver, track_type)] += 1

            trk = track_stats[(driver, track_id)]
            trk["n"] += 1
            if finish < trk["best"]:
                trk["best"] = finish
            tku = track_hist[(driver, track_id)]
            tku.append(finish)
            if len(tku) > 5:
                tku.popleft()

            team = team_lookup.get(driver, "")
            ttu = team_type_hist[(team, track_type)]
            ttu.append(finish)
            if len(ttu) > 30:
                ttu.popleft()
            team_type_count[(team, track_type)] += 1

            career_races[driver] += 1
            last_race_date[driver] = race_date

    return pd.DataFrame(out_rows)
