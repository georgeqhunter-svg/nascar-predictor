from backtest_oddslogic_v5 import RACES
from src.features.build_features import build_features
import pandas as pd

races = pd.read_parquet("data/processed/races.parquet")
entries = pd.read_parquet("data/processed/entries.parquet")
sessions = pd.read_parquet("data/processed/sessions.parquet")
loopstats = pd.read_parquet("data/processed/loopstats.parquet")
laptimes = pd.read_parquet("data/processed/laptimes.parquet")
races["date"] = pd.to_datetime(races["date"])
entries["date"] = pd.to_datetime(entries["date"])

date, name, matchups = next(x for x in RACES if ("Toyota" in x[1]) or ("Save Mart" in x[1]))
target_ts = pd.Timestamp(date)
r = races[((races["date"] == target_ts) | (races["race_name"].str.contains(name, case=False, na=False))) & (races["season"] == 2026)]
rid = r.iloc[0]["race_id_short"]

feat = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
sub = feat[feat["race_id_short"] == rid].copy()
cols = [
    "driver", "team", "manufacturer", "start_pos", "finish_pos", "is_dnf",
    "qual_z", "practice_z", "has_practice_data",
    "pl_driver", "pl_team", "pl_driver_track", "pl_effective",
    "avg_finish_5", "avg_finish_10", "career_races",
    "races_at_type_last_10", "avg_finish_at_type_last_10",
    "races_at_track_last_10", "avg_finish_at_track_last_10",
    "tm_wpct_20", "tm_adj_wpct_20", "tm_races_20",
    "loop_avg_ps_10", "loop_rating_10", "lr_prior_warp", "h2h_beat_rate_10",
]
cols = [c for c in cols if c in sub.columns]
sub = sub.sort_values(["finish_pos", "driver"])
print(sub[cols].to_string(index=False))
sub[cols].to_csv("toyota_features.csv", index=False)
print("\nwrote toyota_features.csv")
