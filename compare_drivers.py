# compare_drivers.py — dump feature values for two drivers side by side to
# understand why the model prefers one over the other.
#
# Usage: python compare_drivers.py "Chase Briscoe" "Denny Hamlin"
import sys
import numpy as np
import pandas as pd

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_drivers.py \"<Driver A>\" \"<Driver B>\" [--race-id 2026-5625]")
        return 1

    driver_a = sys.argv[1]
    driver_b = sys.argv[2]
    race_id = "2026-5625"
    if "--race-id" in sys.argv:
        race_id = sys.argv[sys.argv.index("--race-id") + 1]

    from src.features.build_features import build_features

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")

    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    target = features[features["race_id_short"] == race_id]
    if target.empty:
        print(f"no features for {race_id}")
        return 1

    a = target[target["driver"] == driver_a]
    b = target[target["driver"] == driver_b]
    if a.empty:
        print(f"{driver_a} not in {race_id} field")
        return 1
    if b.empty:
        print(f"{driver_b} not in {race_id} field")
        return 1

    # Features worth comparing (ordered by rough model-impact)
    key_features = [
        "pl_driver", "pl_team", "pl_driver_track", "pl_effective",
        "avg_finish_5", "avg_finish_10", "avg_finish_20",
        "avg_finish_at_type_ytd", "avg_finish_at_track", "best_finish_at_track",
        "team_avg_finish_at_type",
        "season_wins_ytd", "season_top5_ytd", "season_top10_ytd",
        "career_races", "races_at_track",
        "momentum_3", "dnf_rate_10", "top10_rate_10",
        "loop_avg_ps_5", "loop_avg_ps_10",
        "loop_rating_5", "loop_rating_10",
        "loop_quality_passes_5", "loop_quality_passes_10",
        "loop_fast_laps_10", "loop_passing_diff_10",
        "loop_top15_laps_10", "loop_lead_laps_10",
        "restart_gain_5", "restart_gain_10", "restart_count_10",
        "pit_gain_5", "pit_gain_10", "pit_time_delta_10",
        "playoff_round", "in_playoffs", "is_playoff_driver",
    ]
    key_features = [f for f in key_features if f in target.columns]

    def is_lower_better(feat):
        # Finish-position-based (lower=better) and DNF rate (lower=better)
        # and pit_time_delta (lower=faster stop).
        return any(k in feat for k in ["avg_finish", "loop_avg_ps", "dnf_rate", "pit_time_delta"])

    rows = []
    for f in key_features:
        va = a[f].iloc[0]
        vb = b[f].iloc[0]
        field = target[f].dropna().values
        median = float(np.median(field)) if len(field) else np.nan
        favors = ""
        if pd.notna(va) and pd.notna(vb) and va != vb:
            if is_lower_better(f):
                favors = driver_a if va < vb else driver_b
            else:
                favors = driver_a if va > vb else driver_b
        rows.append({
            "feature": f,
            driver_a: va,
            driver_b: vb,
            "field_median": median,
            "favors": favors,
        })

    df = pd.DataFrame(rows)
    print(f"\nComparison of {driver_a} vs {driver_b} for race {race_id}")
    print(f"(field size: {len(target)})\n")
    print(df.round(3).to_string(index=False))

    # Summary count of who each metric favors
    a_wins = (df["favors"] == driver_a).sum()
    b_wins = (df["favors"] == driver_b).sum()
    ties = (df["favors"] == "").sum()
    print(f"\nSummary: {driver_a} favored on {a_wins} features, "
          f"{driver_b} favored on {b_wins} features, {ties} ties/N-A")

if __name__ == "__main__":
    sys.exit(main() or 0)
