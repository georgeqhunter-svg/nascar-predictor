"""Diagnose why the model is fading Kyle Larson at Illinois.
Prints his feature row alongside the field mean and a top-3 favorite for
comparison. If qual_z / practice_z are near zero for Larson, the sessions
data didn't populate for him.
"""
import pandas as pd

from src.features.build_features import build_features

races = pd.read_parquet("data/processed/races.parquet")
entries = pd.read_parquet("data/processed/entries.parquet")
sessions = pd.read_parquet("data/processed/sessions.parquet")
loopstats = pd.read_parquet("data/processed/loopstats.parquet")
laptimes = pd.read_parquet("data/processed/laptimes.parquet")
races["date"] = pd.to_datetime(races["date"])
entries["date"] = pd.to_datetime(entries["date"])

target_ts = pd.Timestamp("2026-09-13")
r = races[
    ((races["date"] == target_ts)
     | (races["race_name"].str.contains("Enjoy Illinois", case=False)))
    & (races["season"] == 2026)
]
rid = r.iloc[0]["race_id_short"]

# Check raw entries + sessions FIRST — no features yet.
print("=" * 70)
print("RAW ENTRIES for Illinois")
print("=" * 70)
ent = entries[entries["race_id_short"] == rid]
print(f"Entries: {len(ent)} drivers")
print(ent[["driver", "start_pos", "qual_pos", "qual_speed", "make"]]
      .sort_values("start_pos").to_string(index=False))

print()
print("=" * 70)
print("RAW SESSIONS for Illinois")
print("=" * 70)
sess = sessions[sessions["race_id_short"] == rid]
print(f"Sessions rows: {len(sess)}")
if len(sess) > 0:
    print(sess["run_type_label"].value_counts().to_dict())
    qual_rows = sess[sess["run_type_label"] == "qualifying"]
    if len(qual_rows) > 0:
        print("\nQualifying session sample:")
        print(qual_rows[["driver_name", "session_position", "best_lap_speed"]]
              .sort_values("session_position").head(10).to_string(index=False))

# Now build features and inspect Larson's row.
print()
print("=" * 70)
print("BUILDING FEATURES...")
print("=" * 70)
features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
features["date"] = pd.to_datetime(features["date"])
target = features[features["race_id_short"] == rid]

# Interesting features to inspect
cols = [
    "driver", "start_pos", "qual_z", "practice_z", "practice_gap_z", "practice_laps_z",
    "team_teammate_qual_z", "team_teammate_practice_z",
    "avg_finish_5", "avg_finish_10", "avg_finish_at_track_last_5",
    "avg_finish_at_type_ytd", "season_wins_ytd", "season_top5_ytd",
    "pl_effective", "manuf_avg_finish_at_type_10", "manuf_avg_finish_at_type_10_v2",
    "crash_dnf_rate_at_type_10", "dnf_rate_at_type_10",
    "has_practice_data",
]
have = [c for c in cols if c in target.columns]

def show(name, sub):
    print(f"\n--- {name} ---")
    for c in have:
        v = sub[c].values
        if len(v) == 0: continue
        try:
            print(f"  {c:<40} {float(v[0]):>+8.3f}")
        except (TypeError, ValueError):
            print(f"  {c:<40} {v[0]}")

show("KYLE LARSON", target[target["driver"] == "Kyle Larson"])
show("CARSON HOCEVAR (works!)", target[target["driver"] == "Carson Hocevar"])
show("CHRISTOPHER BELL (top favorite)", target[target["driver"] == "Christopher Bell"])

# Field means for context
print("\n--- FIELD MEAN ---")
for c in have:
    if c == "driver": continue
    try:
        m = float(target[c].mean())
        print(f"  {c:<40} {m:>+8.3f}")
    except (TypeError, ValueError):
        pass
