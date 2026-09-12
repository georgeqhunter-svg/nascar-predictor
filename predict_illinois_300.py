# predict_illinois_300.py — one-off prediction for 2026-09-13 Enjoy Illinois 300
# at WWT Raceway (Gateway). Entry list published; practice + qualifying are
# 2026-09-12 (not yet). Model will predict without qual/practice signal.

import numpy as np
import pandas as pd

# 36 drivers currently on the entry list (from NASCAR.com weekend-feed).
ENTRIES = [
    ("Tyler Reddick",       4065, "23XI Racing",              "Toyota"),
    ("Ross Chastain",       4001, "Trackhouse Racing",        "Chevrolet"),
    ("Denny Hamlin",        1361, "Joe Gibbs Racing",         "Toyota"),
    ("Ryan Blaney",         4023, "Team Penske",              "Ford"),
    ("Chris Buescher",      3989, "RFK Racing",               "Ford"),
    ("Ty Gibbs",            4368, "Joe Gibbs Racing",         "Toyota"),
    ("Chase Briscoe",       4228, "Joe Gibbs Racing",         "Toyota"),
    ("Austin Cindric",      4180, "Team Penske",              "Ford"),
    ("Christopher Bell",    4153, "Joe Gibbs Racing",         "Toyota"),
    ("Josh Berry",          4123, "Wood Brothers Racing",     "Ford"),
    ("Joey Logano",         3859, "Team Penske",              "Ford"),
    ("William Byron",       4184, "Hendrick Motorsports",     "Chevrolet"),
    ("Austin Dillon",       3873, "Richard Childress Racing", "Chevrolet"),
    ("Todd Gilliland",      4231, "Front Row Motorsports",    "Ford"),
    ("Daniel Suarez",       4113, "Spire Motorsports",        "Chevrolet"),
    ("Zane Smith",          4272, "Front Row Motorsports",    "Ford"),
    ("Cole Custer",         4104, "Haas Factory Team",        "Chevrolet"),
    ("Erik Jones",          4059, "Legacy Motor Club",        "Toyota"),
    ("Ricky Stenhouse Jr",  3888, "HYAK Motorsports",         "Chevrolet"),
    ("Kyle Larson",         4030, "Hendrick Motorsports",     "Chevrolet"),
    ("John H. Nemechek",    4092, "Legacy Motor Club",        "Toyota"),
    ("Brad Keselowski",     1816, "RFK Racing",               "Ford"),
    ("AJ Allmendinger",     3774, "Kaulig Racing",            "Chevrolet"),
    ("Austin Hill",         4133, "Richard Childress Racing", "Chevrolet"),
    ("Alex Bowman",         4045, "Hendrick Motorsports",     "Chevrolet"),
    ("Chase Elliott",       4062, "Hendrick Motorsports",     "Chevrolet"),
    ("Bubba Wallace",       4025, "23XI Racing",              "Toyota"),
    ("Carson Hocevar",      4326, "Spire Motorsports",        "Chevrolet"),
    ("Ty Dillon",           4013, "Kaulig Racing",            "Chevrolet"),
    ("Cody Ware",           4125, "Rick Ware Racing",         "Chevrolet"),
    ("Shane Van Gisbergen", 4469, "Trackhouse Racing",        "Chevrolet"),
    ("Noah Gragson",        4224, "Front Row Motorsports",    "Ford"),
    ("Ryan Preece",         4070, "RFK Racing",               "Ford"),
    ("Michael McDowell",    3832, "Spire Motorsports",        "Chevrolet"),
    ("Riley Herbst",        4269, "23XI Racing",              "Toyota"),
    ("Connor Zilisch",      4481, "Trackhouse Racing",        "Chevrolet"),
]

RACE_ID_SHORT = "2026-5625"

TIGHT_REG = {"num_leaves": 31, "min_data_in_leaf": 25, "lambda_l2": 2.0}
ALPHA = 0.85  # GBM weight in the PL ensemble
T = 4.0       # temperature (from earlier calibration on intermediates)
N_SAMPLES = 20_000

HAZARD_BY_TYPE = {"superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
                  "road": 0.04, "unique": 0.07}


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.gbm_ranker import GBMEnsemble

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")

    if RACE_ID_SHORT not in races["race_id_short"].values:
        raise RuntimeError(f"{RACE_ID_SHORT} not in races.parquet; re-run backfill.")

    race_row = races[races["race_id_short"] == RACE_ID_SHORT].iloc[0]
    tt = race_row["track_type"]
    print(f"Target race: {race_row['race_name']} at {race_row['track_name']} ({tt}), "
          f"date {race_row['date']}, playoff round {int(race_row.get('playoff_round', 0) or 0)}")

    # Synthesize entry rows so features can be built for this race.
    synth = []
    for i, (drv, drv_id, team, make) in enumerate(ENTRIES):
        synth.append({
            "date": pd.Timestamp("2026-09-13"),
            "season": 2026,
            "race_id_short": RACE_ID_SHORT,
            "track_type": tt,
            "finish_pos": i + 1,   # placeholder; not used for prediction
            "start_pos": pd.NA,
            "car_number": "",
            "driver_id": drv_id,
            "driver": drv,
            "team_id": pd.NA,
            "team": team,
            "owner_id": pd.NA,
            "owner": "",
            "crew_chief_id": pd.NA,
            "crew_chief": "",
            "make": make,
            "car_model": "",
            "sponsor": "",
            "qual_pos": pd.NA,
            "qual_speed": pd.NA,
            "laps_completed": pd.NA,
            "laps_led": pd.NA,
            "times_led": pd.NA,
            "points": pd.NA,
            "playoff_points": pd.NA,
            "status": "",
            "disqualified": False,
            "diff_laps": pd.NA,
            "diff_time": pd.NA,
            "is_dnf": False,
        })
    synth_df = pd.DataFrame(synth)
    all_entries = pd.concat([entries, synth_df], ignore_index=True)

    print(f"Building features across {races['race_id_short'].nunique()} races ({len(all_entries):,} entry rows)")
    features = build_features(races, all_entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    train = features[features["race_id_short"] != RACE_ID_SHORT]
    target = features[features["race_id_short"] == RACE_ID_SHORT].reset_index(drop=True)
    print(f"Training on {train['race_id_short'].nunique()} completed races ({len(train):,} rows)")

    print("Training 5-model GBM ensemble ...")
    model = GBMEnsemble()
    model.fit(train, n_estimators=5, **TIGHT_REG)

    gbm_raw = model.predict_scores(target)
    gbm_z = (gbm_raw - gbm_raw.mean()) / (gbm_raw.std() + 1e-9)
    pl_z = (target["pl_effective"].to_numpy() - target["pl_effective"].mean()) / (target["pl_effective"].std() + 1e-9)
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)

    hazards = np.full(len(target), HAZARD_BY_TYPE.get(tt, 0.08))
    rng = np.random.default_rng(42)
    positions = dist.sample_finishing_orders(blended / T, hazards, n_samples=N_SAMPLES, rng=rng)
    win_probs = dist.win_probs(positions)
    top5 = dist.topn_probs(positions, 5)
    top10 = dist.topn_probs(positions, 10)

    def to_american(p: float) -> str:
        if p <= 0 or p >= 1:
            return "n/a"
        dec = 1 / p
        if dec >= 2.0:
            return f"+{int(round((dec - 1) * 100))}"
        return f"-{int(round(100 / (dec - 1)))}"

    out = pd.DataFrame({
        "driver": target["driver"].values,
        "team": target["team"].values,
        "make": target["manufacturer"].values,
        "win_prob": win_probs,
        "win_odds": [to_american(p) for p in win_probs],
        "top5_prob": top5,
        "top10_prob": top10,
    }).sort_values("win_prob", ascending=False).reset_index(drop=True)
    out["win_prob"] = out["win_prob"].round(4)
    out["top5_prob"] = out["top5_prob"].round(3)
    out["top10_prob"] = out["top10_prob"].round(3)

    print("\n=== ENJOY ILLINOIS 300 — 2026-09-13 predictions ===")
    print("(No qualifying or practice signal yet; will sharpen after 9/12.)\n")
    print(out.to_string(index=False))

    # sanity
    print(f"\nSum of win_probs: {out['win_prob'].sum():.4f} (should be ~1)")


if __name__ == "__main__":
    main()
