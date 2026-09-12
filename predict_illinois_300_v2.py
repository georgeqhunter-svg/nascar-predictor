# predict_illinois_300_v2.py — same entry list as before, but:
#   (a) sweep T values so you can see how the distribution changes
#   (b) fit a per-track-type T on all completed races and apply the
#       "intermediate" T to the Illinois 300 prediction.

import numpy as np
import pandas as pd

# Copy of the entry list from predict_illinois_300.py.
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
ALPHA = 0.85
N_SAMPLES = 20_000

HAZARD_BY_TYPE = {"superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
                  "road": 0.04, "unique": 0.07}


def american(p: float) -> str:
    if p <= 0 or p >= 1:
        return "n/a"
    dec = 1 / p
    return f"+{int(round((dec - 1) * 100))}" if dec >= 2.0 else f"-{int(round(100 / (dec - 1)))}"


def _synth_entries():
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-09-13"), "season": 2026,
        "race_id_short": RACE_ID_SHORT, "track_type": "intermediate",
        "finish_pos": i + 1, "start_pos": pd.NA, "car_number": "",
        "driver_id": did, "driver": name, "team_id": pd.NA, "team": team,
        "owner_id": pd.NA, "owner": "", "crew_chief_id": pd.NA, "crew_chief": "",
        "make": make, "car_model": "", "sponsor": "",
        "qual_pos": pd.NA, "qual_speed": pd.NA,
        "laps_completed": pd.NA, "laps_led": pd.NA, "times_led": pd.NA,
        "points": pd.NA, "playoff_points": pd.NA,
        "status": "", "disqualified": False,
        "diff_laps": pd.NA, "diff_time": pd.NA, "is_dnf": False,
    } for i, (name, did, team, make) in enumerate(ENTRIES)])


def _predict(blended, T, hazards, rng):
    from src.models import distribution as dist
    positions = dist.sample_finishing_orders(blended / T, hazards, n_samples=N_SAMPLES, rng=rng)
    return dist.win_probs(positions), dist.topn_probs(positions, 5), dist.topn_probs(positions, 10)


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")

    all_entries = pd.concat([entries, _synth_entries()], ignore_index=True)
    features = build_features(races, all_entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    train = features[features["race_id_short"] != RACE_ID_SHORT]
    target = features[features["race_id_short"] == RACE_ID_SHORT].reset_index(drop=True)

    print(f"Training on {train['race_id_short'].nunique()} races ...")
    model = GBMEnsemble()
    model.fit(train, n_estimators=5, **TIGHT_REG)

    # -------- Fit per-track-type T on the LAST 40 training races -------- #
    print("Fitting per-track-type T on the last 40 completed races ...")
    recent = (
        train.groupby("race_id_short", sort=False)["date"].first()
        .sort_values().tail(40).index.tolist()
    )
    val_races, tt_list = [], []
    for rid in recent:
        sub = train[train["race_id_short"] == rid]
        raw = model.predict_scores(sub)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = (sub["pl_effective"].to_numpy() - sub["pl_effective"].mean()) / (sub["pl_effective"].std() + 1e-9)
        blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
        blended = blended / max(blended.std(), 1e-6)
        tt = sub["track_type"].iloc[0]
        val_races.append(RaceScoresGT(
            scores=blended,
            finishes=sub["finish_pos"].to_numpy(),
            is_dnf=sub["is_dnf"].to_numpy(),
            hazards=np.full(len(sub), HAZARD_BY_TYPE.get(tt, 0.08)),
        ))
        tt_list.append(tt)

    T_by_type = find_best_temperature_by_type(val_races, tt_list, default_T=3.0, n_samples=1500)
    print("\nPer-track-type T (matchup-LL optimal on last 40 races):")
    for k, v in T_by_type.items():
        print(f"  {k:15s} T = {v:.2f}")
    T_target = T_by_type.get("intermediate", 3.0)
    print(f"\nUsing T = {T_target} for intermediate track prediction.")

    # -------- Compute the target race's blended strengths -------- #
    gbm_raw = model.predict_scores(target)
    gbm_z = (gbm_raw - gbm_raw.mean()) / (gbm_raw.std() + 1e-9)
    pl_z = (target["pl_effective"].to_numpy() - target["pl_effective"].mean()) / (target["pl_effective"].std() + 1e-9)
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)
    hazards = np.full(len(target), HAZARD_BY_TYPE["intermediate"])
    rng = np.random.default_rng(42)

    # -------- T sweep -------- #
    T_sweep = sorted(set([1.5, 2.0, 2.5, 3.0, T_target, 4.0]))
    print(f"\n=== T sweep — top-10 win probabilities ===\n")
    top10_names = None
    all_tables = {}
    for T in T_sweep:
        wp, t5, t10 = _predict(blended, T, hazards, rng=np.random.default_rng(42))
        df = pd.DataFrame({
            "driver": target["driver"].values, "win_prob": wp,
            "win_odds": [american(p) for p in wp],
        }).sort_values("win_prob", ascending=False).reset_index(drop=True)
        all_tables[T] = df
        if top10_names is None:
            top10_names = df["driver"].head(10).tolist()

    header = "driver".ljust(22) + "".join(f"  T={T:>5}  " for T in T_sweep)
    print(header)
    for name in top10_names:
        row = name.ljust(22)
        for T in T_sweep:
            d = all_tables[T]
            r = d[d["driver"] == name].iloc[0]
            row += f"  {r['win_prob']*100:5.2f}%  "
        print(row)

    # -------- Full-field predictions at the calibrated T -------- #
    wp, t5, t10 = _predict(blended, T_target, hazards, rng=np.random.default_rng(43))
    out = pd.DataFrame({
        "driver": target["driver"].values,
        "team": target["team"].values,
        "make": target["manufacturer"].values,
        "win_prob": wp, "win_odds": [american(p) for p in wp],
        "top5": t5, "top10": t10,
    }).sort_values("win_prob", ascending=False).reset_index(drop=True)
    out["win_prob"] = out["win_prob"].round(4)
    out["top5"] = out["top5"].round(3)
    out["top10"] = out["top10"].round(3)
    print(f"\n=== FULL PREDICTION AT T = {T_target} (intermediate-calibrated) ===\n")
    print(out.to_string(index=False))
    print(f"\nSum of win_probs: {out['win_prob'].sum():.4f}")


if __name__ == "__main__":
    main()
