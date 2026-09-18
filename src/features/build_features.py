"""Feature-matrix orchestrator: PL ratings pre-race + rolling stats + qualifying + track.

Output: one row per (race_id_short, driver), suitable for a LightGBM ranker
grouped by race_id_short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.plackett_luce import RaceRanking, Ratings, update_from_race
from .loop_rolling import ROLL_COLS as LOOP_ROLL_COLS, compute_loop_rolling
from .playoffs import derive_elimination_races, derive_playoff_drivers
from .qualifying import (
    practice_best_by_driver,
    practice_zscore_for_race,
    qual_zscore_for_race,
    qualifying_best_by_driver,
)
from .forward import (
    compute_crash_risk,
    compute_mechanical_risk,
    practice_depth_by_driver,
    practice_gap_zscore,
    practice_laps_zscore,
    team_teammate_practice_z,
    team_teammate_qual_z,
)
from .driver_variance import compute_driver_variance
from .h2h import compute_h2h
from .lapraptor_prior import compute_lr_prior
from .practice_pace import compute_practice_features
from .race_pace import compute_race_pace
from .track_rolling import compute_track_rolling
from .type_rolling import compute_type_rolling
from .pit import compute_rolling_pit, compute_rolling_pit_team
from .restart import compute_rolling_restart
from .tire_deg import compute_rolling_tire
from .rolling import compute_rolling
from .tracks import TRACKS
from .weather import compute_weather_features, load_or_fetch_weather
from .manuf_type import compute_manuf_type


def build_features(
    races: pd.DataFrame,
    entries: pd.DataFrame,
    sessions: pd.DataFrame | None = None,
    loopstats: pd.DataFrame | None = None,
    laptimes: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One-pass feature builder. Walk races in chronological order.

    For each race, snapshot PL ratings BEFORE the race, join in rolling stats
    (also computed strictly from past data), qualifying z-score (from THIS
    weekend's session data — that's OK because qualifying happens pre-race),
    and static track features.

    Returns one row per (race_id_short, driver) with columns:
      identifiers, target (finish_pos), PL features, rolling features,
      qualifying features, track features.
    """
    races = races.sort_values("date").reset_index(drop=True).copy()
    entries = entries.copy()
    entries["date"] = pd.to_datetime(entries["date"])

    # Merge track_id into entries for track-specific driver history.
    if "track_id" in races.columns and "track_id" not in entries.columns:
        entries = entries.merge(
            races[["race_id_short", "track_id"]], on="race_id_short", how="left"
        )
    # Race-level distance (miles). Falls back to NaN when unavailable.
    race_distance = (
        races.set_index("race_id_short")["scheduled_distance"]
        if "scheduled_distance" in races.columns
        else None
    )
    rolling = compute_rolling(entries).set_index(["race_id_short", "driver"])
    h2h = compute_h2h(entries, races).set_index(["race_id_short", "driver"])
    type_roll = compute_type_rolling(entries, races).set_index(["race_id_short", "driver"])
    manuf_type_df = compute_manuf_type(entries, races).set_index(["race_id_short", "driver"])
    # Weather: load or fetch cache, then compute per-driver rolling features.
    if weather is None:
        try:
            weather = load_or_fetch_weather(races, refresh=False)
        except Exception as ex:
            print(f"[weather] disabled: {ex}")
            weather = pd.DataFrame(columns=[
                "race_id_short", "temp_max_f", "wind_max_mph",
                "humidity_mean_pct", "precip_sum_in",
            ])
    if not weather.empty:
        weather_feats = compute_weather_features(entries, races, weather)
        weather_feats = weather_feats.set_index(["race_id_short", "driver"])
    else:
        weather_feats = None
    variance = compute_driver_variance(entries, races).set_index(["race_id_short", "driver"])
    race_pace = compute_race_pace(entries, races).set_index(["race_id_short", "driver"])
    track_roll = compute_track_rolling(entries, races).set_index(["race_id_short", "driver"])
    from pathlib import Path
    lr_prior = compute_lr_prior(
        entries, races, Path("data/raw"),
    ).set_index(["race_id_short", "driver"])
    practice_lr = compute_practice_features(entries, races)
    practice_lr = (practice_lr.set_index(["race_id_short", "driver_id"])
                   if not practice_lr.empty else None)
    mech_risk = (
        compute_mechanical_risk(entries).set_index(["race_id_short", "driver"])
    )
    crash_risk = (
        compute_crash_risk(entries, races).set_index(["race_id_short", "driver"])
    )
    practice_depth = (
        practice_depth_by_driver(sessions)
        if sessions is not None and not sessions.empty
        else pd.DataFrame(columns=["race_id_short", "driver_name",
                                   "practice_gap_mph", "practice_laps"])
    )
    qual = (
        qualifying_best_by_driver(sessions)
        if sessions is not None and not sessions.empty
        else pd.DataFrame(columns=["race_id_short", "driver_name", "qual_speed"])
    )
    practice = (
        practice_best_by_driver(sessions)
        if sessions is not None and not sessions.empty
        else pd.DataFrame(columns=["race_id_short", "driver_name", "practice_speed"])
    )
    # Optional loop-stat rolling features keyed by (race_id_short, driver_id).
    if loopstats is not None and not loopstats.empty:
        loop_rolling = compute_loop_rolling(loopstats, races).set_index(
            ["race_id_short", "driver_id"]
        )
    else:
        loop_rolling = None
    # Optional restart features (from lap-times).
    if laptimes is not None and not laptimes.empty:
        restart_rolling = compute_rolling_restart(laptimes, races).set_index(
            ["race_id_short", "driver_id"]
        )
        pit_rolling = compute_rolling_pit(laptimes, races).set_index(
            ["race_id_short", "driver_id"]
        )
        pit_team_rolling = compute_rolling_pit_team(laptimes, entries, races).set_index(
            ["race_id_short", "driver_id"]
        )
        tire_rolling = compute_rolling_tire(laptimes, races).set_index(
            ["race_id_short", "driver_id"]
        )
    else:
        restart_rolling = None
        pit_rolling = None
        pit_team_rolling = None
        tire_rolling = None

    playoff_drivers = derive_playoff_drivers(entries, races)
    elimination_races = derive_elimination_races(races)

    ratings = Ratings()
    rows: list[dict] = []
    for _, race_row in races.iterrows():
        race_id = race_row["race_id_short"]
        tt = race_row["track_type"]
        track_slug = race_row.get("track_name", "")
        track = None
        # Try to match by name (case insensitive) to our TRACKS table.
        for slug, tobj in TRACKS.items():
            if slug.replace("_", " ").lower() == str(track_slug).lower():
                track = tobj
                break

        e = entries[entries["race_id_short"] == race_id].sort_values("finish_pos")
        if e.empty:
            continue
        drivers = e["driver"].tolist()
        teams = e["team"].tolist()
        driver_ids = (
            e["driver_id"].astype("Int64").tolist()
            if "driver_id" in e.columns else [None] * len(e)
        )
        makes = e["make"].tolist() if "make" in e.columns else [""] * len(e)
        # Playoff pressure features from races.parquet.
        playoff_round = int(race_row.get("playoff_round", 0) or 0)
        race_num = int(race_row.get("race_number", 0) or 0)
        # NASCAR regular season is 26 races; playoffs start at race 27.
        races_to_cutoff = max(26 - race_num, 0) if race_num > 0 else -1
        this_season = int(race_row.get("season", 0) or 0)
        elimination = int(race_id in elimination_races)
        finishes = e["finish_pos"].astype(int).tolist()
        is_dnf_arr = e["is_dnf"].tolist()
        if "start_pos" in e.columns:
            start_pos = e["start_pos"].astype("float").fillna(-1).tolist()
        else:
            start_pos = [-1.0] * len(e)

        # Snapshot PL strengths BEFORE the race.
        d_strengths = [ratings.driver.get(d, 0.0) for d in drivers]
        t_strengths = [ratings.team.get(t, 0.0) for t in teams]
        dt_strengths = [
            ratings.driver_track.get(d, {}).get(tt, 0.0) for d in drivers
        ]
        eff = [d + t + dt for d, t, dt in zip(d_strengths, t_strengths, dt_strengths)]

        # Qualifying z-score for this race (uses THIS weekend's data — that's fine,
        # qualifying occurs pre-race).
        qz = qual_zscore_for_race(qual, race_id, drivers)
        pz = practice_zscore_for_race(practice, race_id, drivers)
        pgz = practice_gap_zscore(practice_depth, race_id, drivers)
        plz = practice_laps_zscore(practice_depth, race_id, drivers)
        team_qz = team_teammate_qual_z(qual, entries, race_id, drivers, teams)
        team_pz = team_teammate_practice_z(practice, race_id, drivers, teams)

        for i, drv in enumerate(drivers):
            r = rolling.loc[(race_id, drv)] if (race_id, drv) in rolling.index else None
            h = h2h.loc[(race_id, drv)] if (race_id, drv) in h2h.index else None
            mt = (manuf_type_df.loc[(race_id, drv)]
                  if (race_id, drv) in manuf_type_df.index else None)
            wx = None
            if weather_feats is not None and (race_id, drv) in weather_feats.index:
                wx = weather_feats.loc[(race_id, drv)]
            m = mech_risk.loc[(race_id, drv)] if (race_id, drv) in mech_risk.index else None
            cr = crash_risk.loc[(race_id, drv)] if (race_id, drv) in crash_risk.index else None
            tr = type_roll.loc[(race_id, drv)] if (race_id, drv) in type_roll.index else None
            v = variance.loc[(race_id, drv)] if (race_id, drv) in variance.index else None
            rp = race_pace.loc[(race_id, drv)] if (race_id, drv) in race_pace.index else None
            tk = track_roll.loc[(race_id, drv)] if (race_id, drv) in track_roll.index else None
            lp = lr_prior.loc[(race_id, drv)] if (race_id, drv) in lr_prior.index else None
            pr = None
            _drv_id = driver_ids[i]
            if practice_lr is not None and _drv_id is not None and pd.notna(_drv_id):
                key = (race_id, int(_drv_id))
                if key in practice_lr.index:
                    pr = practice_lr.loc[key]
            drv_id = driver_ids[i]
            loop_row = None
            if loop_rolling is not None and drv_id is not None and pd.notna(drv_id):
                key = (race_id, int(drv_id))
                if key in loop_rolling.index:
                    loop_row = loop_rolling.loc[key]
            loop_features = {}
            for c in LOOP_ROLL_COLS:
                loop_features[f"loop_{c}_5"] = loop_row[f"loop_{c}_5"] if loop_row is not None else np.nan
                loop_features[f"loop_{c}_10"] = loop_row[f"loop_{c}_10"] if loop_row is not None else np.nan

            restart_row = None
            if restart_rolling is not None and drv_id is not None and pd.notna(drv_id):
                key = (race_id, int(drv_id))
                if key in restart_rolling.index:
                    restart_row = restart_rolling.loc[key]
            restart_features = {
                "restart_gain_5": restart_row["restart_gain_5"] if restart_row is not None else np.nan,
                "restart_gain_10": restart_row["restart_gain_10"] if restart_row is not None else np.nan,
                "restart_gain_std_10": restart_row["restart_gain_std_10"] if restart_row is not None else np.nan,
                "restart_count_10": restart_row["restart_count_10"] if restart_row is not None else np.nan,
            }
            pit_row = None
            if pit_rolling is not None and drv_id is not None and pd.notna(drv_id):
                key = (race_id, int(drv_id))
                if key in pit_rolling.index:
                    pit_row = pit_rolling.loc[key]
            pit_features = {
                "pit_gain_5": pit_row["pit_gain_5"] if pit_row is not None else np.nan,
                "pit_gain_10": pit_row["pit_gain_10"] if pit_row is not None else np.nan,
                "pit_gain_std_10": pit_row["pit_gain_std_10"] if pit_row is not None else np.nan,
                "pit_time_delta_10": pit_row["pit_time_delta_10"] if pit_row is not None else np.nan,
            }
            pit_team_row = None
            if pit_team_rolling is not None and drv_id is not None and pd.notna(drv_id):
                key = (race_id, int(drv_id))
                if key in pit_team_rolling.index:
                    pit_team_row = pit_team_rolling.loc[key]
            pit_features.update({
                "team_pit_gain_5": pit_team_row["team_pit_gain_5"] if pit_team_row is not None else np.nan,
                "team_pit_gain_10": pit_team_row["team_pit_gain_10"] if pit_team_row is not None else np.nan,
                "team_pit_gain_std_5": pit_team_row["team_pit_gain_std_5"] if pit_team_row is not None else np.nan,
                "team_pit_gain_std_10": pit_team_row["team_pit_gain_std_10"] if pit_team_row is not None else np.nan,
                "team_pit_time_delta_10": pit_team_row["team_pit_time_delta_10"] if pit_team_row is not None else np.nan,
                "team_pit_n_stops_10": pit_team_row["team_pit_n_stops_10"] if pit_team_row is not None else 0,
            })
            tire_row = None
            if tire_rolling is not None and drv_id is not None and pd.notna(drv_id):
                key = (race_id, int(drv_id))
                if key in tire_rolling.index:
                    tire_row = tire_rolling.loc[key]
            tire_features = {
                "tire_decay_type_10": tire_row["tire_decay_type_10"] if tire_row is not None else np.nan,
                "tire_retention_type_10": tire_row["tire_retention_type_10"] if tire_row is not None else np.nan,
                "tire_races_type": tire_row["tire_races_type"] if tire_row is not None else 0,
            }
            rows.append({
                "race_id_short": race_id,
                "date": race_row["date"],
                "season": race_row["season"],
                "driver": drv,
                "team": teams[i],
                "manufacturer": makes[i] or "Unknown",
                "track_type": tt,
                "playoff_round": playoff_round,
                "races_to_cutoff": races_to_cutoff,
                "in_playoffs": int(playoff_round > 0),
                "is_playoff_driver": int(
                    playoff_round > 0
                    and (this_season, drv) in playoff_drivers
                ),
                "is_elimination_race": elimination,
                "track_length_mi": (track.length_mi if track else np.nan),
                "track_banking_deg": (
                    track.banking_deg if (track and track.banking_deg is not None) else np.nan
                ),
                "track_surface": (track.surface if track else "unknown"),
                "race_distance_mi": (
                    float(race_distance.get(race_id, np.nan))
                    if race_distance is not None else np.nan
                ),
                "restrictor_plate": bool(race_row.get("restrictor_plate", False)),
                "start_pos": start_pos[i],
                "pl_driver": d_strengths[i],
                "pl_team": t_strengths[i],
                "pl_driver_track": dt_strengths[i],
                "pl_effective": eff[i],
                "qual_z": qz[i],
                "practice_z": pz[i],
                "avg_finish_5": r["avg_finish_5"] if r is not None else np.nan,
                "avg_finish_10": r["avg_finish_10"] if r is not None else np.nan,
                "avg_finish_20": r["avg_finish_20"] if r is not None else np.nan,
                "dnf_rate_10": r["dnf_rate_10"] if r is not None else np.nan,
                "top10_rate_10": r["top10_rate_10"] if r is not None else np.nan,
                "season_wins_ytd": r["season_wins_ytd"] if r is not None else 0,
                "season_top5_ytd": r["season_top5_ytd"] if r is not None else 0,
                "season_top10_ytd": r["season_top10_ytd"] if r is not None else 0,
                "races_at_type_ytd": r["races_at_type_ytd"] if r is not None else 0,
                "avg_finish_at_type_ytd": r["avg_finish_at_type_ytd"] if r is not None else np.nan,
                "races_at_track": r["races_at_track"] if r is not None else 0,
                "avg_finish_at_track": r["avg_finish_at_track"] if r is not None else np.nan,
                "best_finish_at_track": r["best_finish_at_track"] if r is not None else np.nan,
                "team_avg_finish_at_type": r["team_avg_finish_at_type"] if r is not None else np.nan,
                "team_races_at_type": r["team_races_at_type"] if r is not None else 0,
                "tm_wpct_20": r["tm_wpct_20"] if r is not None else np.nan,
                "tm_adj_wpct_20": r["tm_adj_wpct_20"] if r is not None else np.nan,
                "tm_races_20": r["tm_races_20"] if r is not None else 0,
                "momentum_3": r["momentum_3"] if r is not None else np.nan,
                "career_races": r["career_races"] if r is not None else 0,
                "days_since_last_race": r["days_since_last_race"] if r is not None else np.nan,
                "h2h_beat_rate_10": h["h2h_beat_rate_10"] if h is not None else np.nan,
                "h2h_beat_rate_type_10": h["h2h_beat_rate_type_10"] if h is not None else np.nan,
                "h2h_shared_opponents_10": h["h2h_shared_opponents_10"] if h is not None else 0,
                "practice_gap_z": pgz[i],
                "practice_laps_z": plz[i],
                "team_teammate_qual_z": team_qz[i],
                "team_teammate_practice_z": team_pz[i],
                "mech_dnf_rate_10": m["mech_dnf_rate_10"] if m is not None else np.nan,
                "crash_dnf_rate_10": cr["crash_dnf_rate_10"] if cr is not None else np.nan,
                "crash_dnf_rate_at_type_10": cr["crash_dnf_rate_at_type_10"] if cr is not None else np.nan,
                "avg_finish_at_type_last_5": tr["avg_finish_at_type_last_5"] if tr is not None else np.nan,
                "avg_finish_at_type_last_10": tr["avg_finish_at_type_last_10"] if tr is not None else np.nan,
                "races_at_type_last_10": tr["races_at_type_last_10"] if tr is not None else 0,
                "dnf_rate_at_type_10": v["dnf_rate_at_type_10"] if v is not None else np.nan,
                "finish_std_at_type_10": v["finish_std_at_type_10"] if v is not None else np.nan,
                "finish_iqr_at_type_10": v["finish_iqr_at_type_10"] if v is not None else np.nan,
                "laps_led_avg_at_type_10": rp["laps_led_avg_at_type_10"] if rp is not None else np.nan,
                "laps_led_pct_at_type_10": rp["laps_led_pct_at_type_10"] if rp is not None else np.nan,
                "qual_to_finish_delta_at_type_10": rp["qual_to_finish_delta_at_type_10"] if rp is not None else np.nan,
                "manuf_avg_finish_at_type_10": rp["manuf_avg_finish_at_type_10"] if rp is not None else np.nan,
                "avg_finish_at_track_last_5": tk["avg_finish_at_track_last_5"] if tk is not None else np.nan,
                "avg_finish_at_track_last_10": tk["avg_finish_at_track_last_10"] if tk is not None else np.nan,
                "best_finish_at_track_last_10": tk["best_finish_at_track_last_10"] if tk is not None else np.nan,
                "races_at_track_last_10": tk["races_at_track_last_10"] if tk is not None else 0,
                "lr_prior_pfaez": lp["lr_prior_pfaez"] if lp is not None else np.nan,
                "lr_prior_wpfarp": lp["lr_prior_wpfarp"] if lp is not None else np.nan,
                "lr_prior_warp": lp["lr_prior_warp"] if lp is not None else np.nan,
                "lr_prior_grlr": lp["lr_prior_grlr"] if lp is not None else np.nan,
                "lr_prior_ss": lp["lr_prior_ss"] if lp is not None else np.nan,
                "lr_prior_cpoms": lp["lr_prior_cpoms"] if lp is not None else np.nan,
                "lr_prior_seasons_used": lp["lr_prior_seasons_used"] if lp is not None else 0,
                "practice_best_speed_z": pr["practice_best_speed_z"] if pr is not None else 0.0,
                "practice_5lap_avg_z": pr["practice_5lap_avg_z"] if pr is not None else 0.0,
                "practice_10lap_avg_z": pr["practice_10lap_avg_z"] if pr is not None else 0.0,
                "practice_consistency_z": pr["practice_consistency_z"] if pr is not None else 0.0,
                "practice_laps_run_z": pr["practice_laps_run_z"] if pr is not None else 0.0,
                "has_practice_data": int(pr["has_practice_data"]) if pr is not None else 0,
                "finish_pos": finishes[i],
                "is_dnf": is_dnf_arr[i],
                # Manufacturer × track-type
                "manuf_avg_finish_at_type_5_v2": mt["manuf_avg_finish_at_type_5_v2"] if mt is not None else np.nan,
                "manuf_avg_finish_at_type_10_v2": mt["manuf_avg_finish_at_type_10_v2"] if mt is not None else np.nan,
                "manuf_win_rate_at_type_10": mt["manuf_win_rate_at_type_10"] if mt is not None else np.nan,
                "manuf_top5_rate_at_type_10": mt["manuf_top5_rate_at_type_10"] if mt is not None else np.nan,
                "manuf_top10_rate_at_type_10": mt["manuf_top10_rate_at_type_10"] if mt is not None else np.nan,
                "manuf_finish_std_at_type_10": mt["manuf_finish_std_at_type_10"] if mt is not None else np.nan,
                "drv_manuf_type_avg_finish_10": mt["drv_manuf_type_avg_finish_10"] if mt is not None else np.nan,
                "drv_manuf_type_races_10": mt["drv_manuf_type_races_10"] if mt is not None else 0,
                # Weather (this race's conditions)
                "race_temp_max_f": wx["race_temp_max_f"] if wx is not None else np.nan,
                "race_wind_max_mph": wx["race_wind_max_mph"] if wx is not None else np.nan,
                "race_humidity_pct": wx["race_humidity_pct"] if wx is not None else np.nan,
                "race_precip_in": wx["race_precip_in"] if wx is not None else np.nan,
                # Weather (driver's rolling performance in similar conditions)
                "wx_hot_avg_finish": wx["wx_hot_avg_finish"] if wx is not None else np.nan,
                "wx_hot_races": wx["wx_hot_races"] if wx is not None else 0,
                "wx_cool_avg_finish": wx["wx_cool_avg_finish"] if wx is not None else np.nan,
                "wx_cool_races": wx["wx_cool_races"] if wx is not None else 0,
                "wx_windy_avg_finish": wx["wx_windy_avg_finish"] if wx is not None else np.nan,
                "wx_windy_races": wx["wx_windy_races"] if wx is not None else 0,
                "wx_wet_avg_finish": wx["wx_wet_avg_finish"] if wx is not None else np.nan,
                "wx_wet_races": wx["wx_wet_races"] if wx is not None else 0,
                **loop_features,
                **restart_features,
                **pit_features,
                **tire_features,
            })

        # After row emission, update PL ratings using this race's outcome so
        # future rows see the new snapshot. Skip if the race hasn't been run
        # yet (finish_pos == 0 for everyone).
        #
        # Defense in depth: filter to finish_pos > 0 entries before building
        # the RaceRanking. If a driver has finish_pos <= 0 in an otherwise-run
        # race (scratched, partial data), sort_values("finish_pos") puts them
        # BEFORE the real winner (0 < 1), and PL would treat them as the race
        # winner, poisoning ratings for every subsequent race. See
        # diag_finish_pos_zero.py for the audit that confirmed the current
        # entries.parquet is clean; this guard prevents future regressions.
        if any(f > 0 for f in finishes):
            keep = [i for i, f in enumerate(finishes) if f > 0]
            pl_drivers = [drivers[i] for i in keep]
            pl_teams = [teams[i] for i in keep]
            pl_is_dnf = [is_dnf_arr[i] for i in keep]
            first_dnf = next(
                (idx for idx, v in enumerate(pl_is_dnf) if v), len(pl_drivers)
            )
            race = RaceRanking(
                drivers=pl_drivers, teams=pl_teams, track_type=tt, dnf_at=first_dnf,
            )
            update_from_race(ratings, race)

    return pd.DataFrame(rows)
