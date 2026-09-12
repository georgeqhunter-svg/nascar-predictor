# predict_illinois_300_v3.py — same entry list, but:
#   1. Fit per-track-type T on ALL completed races (not just last 40), so the
#      T for "intermediate" isn't overfit to a tiny sample.
#   2. Compute BOTH a matchup-LL-optimal T (sharp) and a winner-LL-optimal T
#      (spread) and report predictions at each.
#   3. Dump Kyle Larson's feature snapshot so we can see why the model rates
#      him so low.

import numpy as np
import pandas as pd

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
CAL_WINDOW = 133   # use all walk-forward test races (36-race burn-in + rest)
HAZARD_BY_TYPE = {"superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
                  "road": 0.04, "unique": 0.07}


def american(p):
    if p <= 0 or p >= 1:
        return "n/a"
    dec = 1 / p
    return f"+{int(round((dec - 1) * 100))}" if dec >= 2.0 else f"-{int(round(100 / (dec - 1)))}"


def synth_entries():
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


def blend(sub, model, alpha=ALPHA):
    raw = model.predict_scores(sub)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = (sub["pl_effective"].to_numpy() - sub["pl_effective"].mean()) / (sub["pl_effective"].std() + 1e-9)
    b = alpha * gbm_z + (1 - alpha) * pl_z
    return b / max(b.std(), 1e-6)


def _log_clip(p, floor=1e-9):
    return np.log(np.clip(p, floor, 1.0 - floor))


def fit_T_grid_two_objectives(val_races, T_grid=None, n_samples=1500):
    """Return (T_matchup, T_winner) grids per-type. val_races: list of (RaceScoresGT, track_type)."""
    from src.models import distribution as dist
    from src.models.calibrate import _matchup_ll_for_race
    if T_grid is None:
        T_grid = np.array([0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0])
    by_type = {}
    for r, tt in val_races:
        by_type.setdefault(tt, []).append(r)

    result = {}
    for tt, rs in by_type.items():
        if len(rs) < 6:
            result[tt] = {"matchup_T": 3.0, "winner_T": 3.0, "n": len(rs)}
            continue
        m_ll = []
        w_ll = []
        for T in T_grid:
            m_losses = []
            w_losses = []
            for r in rs:
                strengths = r.scores / T
                rng = np.random.default_rng(0)
                positions = dist.sample_finishing_orders(strengths, r.hazards, n_samples=n_samples, rng=rng)
                M = dist.matchup_probs(positions)
                winp = dist.win_probs(positions)
                m_losses.append(_matchup_ll_for_race(M, r.finishes, r.is_dnf))
                w_losses.append(-float(_log_clip(np.array([winp[int(np.argmin(r.finishes))]]))[0]))
            m_ll.append(np.mean(m_losses))
            w_ll.append(np.mean(w_losses))
        result[tt] = {
            "matchup_T": float(T_grid[int(np.argmin(m_ll))]),
            "winner_T": float(T_grid[int(np.argmin(w_ll))]),
            "n": len(rs),
        }
    return result


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT
    from src.models.gbm_ranker import GBMEnsemble

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")

    all_entries = pd.concat([entries, synth_entries()], ignore_index=True)
    features = build_features(races, all_entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    train = features[features["race_id_short"] != RACE_ID_SHORT]
    target = features[features["race_id_short"] == RACE_ID_SHORT].reset_index(drop=True)

    print(f"Training on {train['race_id_short'].nunique()} races ...")
    model = GBMEnsemble()
    model.fit(train, n_estimators=5, **TIGHT_REG)

    # Fit T on ALL completed races (after a small burn-in of 30).
    print(f"Fitting per-type T on all completed races (excluding first 30) ...")
    race_ids_ordered = (train.groupby("race_id_short")["date"].first().sort_values().index.tolist())
    val_race_ids = race_ids_ordered[30:]
    val_races = []
    for rid in val_race_ids:
        sub = train[train["race_id_short"] == rid]
        b = blend(sub, model)
        tt = sub["track_type"].iloc[0]
        val_races.append((RaceScoresGT(
            scores=b, finishes=sub["finish_pos"].to_numpy(),
            is_dnf=sub["is_dnf"].to_numpy(),
            hazards=np.full(len(sub), HAZARD_BY_TYPE.get(tt, 0.08)),
        ), tt))

    T_grids = fit_T_grid_two_objectives(val_races)
    print("\nPer-track-type T (fitted on all completed races):")
    for k, v in T_grids.items():
        print(f"  {k:15s}  n={v['n']:3d}  matchup_T = {v['matchup_T']:.2f}  winner_T = {v['winner_T']:.2f}")

    T_m = T_grids.get("intermediate", {}).get("matchup_T", 3.0)
    T_w = T_grids.get("intermediate", {}).get("winner_T", 3.0)
    print(f"\nFor Illinois 300 (intermediate): matchup-optimal T = {T_m}, winner-optimal T = {T_w}")

    # Target predictions at both T values.
    b_target = blend(target, model)
    haz = np.full(len(target), HAZARD_BY_TYPE["intermediate"])

    def predict_at(T):
        rng = np.random.default_rng(42)
        positions = dist.sample_finishing_orders(b_target / T, haz, n_samples=N_SAMPLES, rng=rng)
        return dist.win_probs(positions), dist.topn_probs(positions, 5), dist.topn_probs(positions, 10)

    for label, T in [("MATCHUP-OPTIMAL", T_m), ("WINNER-OPTIMAL", T_w)]:
        wp, t5, t10 = predict_at(T)
        out = pd.DataFrame({
            "driver": target["driver"].values,
            "team": target["team"].values,
            "make": target["manufacturer"].values,
            "win%": wp * 100,
            "odds": [american(p) for p in wp],
            "top5": t5,
            "top10": t10,
        }).sort_values("win%", ascending=False).reset_index(drop=True)
        out["win%"] = out["win%"].round(2)
        out["top5"] = out["top5"].round(3)
        out["top10"] = out["top10"].round(3)
        print(f"\n=== {label} (T = {T}) ===")
        print(out.head(15).to_string(index=False))
        print("... (bottom 5)")
        print(out.tail(5).to_string(index=False))

    # ---- Kyle Larson diagnostic ---- #
    print("\n" + "=" * 70)
    print("KYLE LARSON DIAGNOSTIC")
    print("=" * 70)
    diag_cols = [
        "driver", "pl_driver", "pl_team", "pl_driver_track", "pl_effective",
        "avg_finish_5", "avg_finish_10", "avg_finish_20",
        "avg_finish_at_type_ytd", "avg_finish_at_track",
        "season_wins_ytd", "season_top5_ytd", "season_top10_ytd",
        "career_races",
        "loop_avg_ps_10", "loop_rating_10", "loop_quality_passes_10",
        "restart_gain_10",
        "team_avg_finish_at_type", "momentum_3",
    ]
    diag = target[diag_cols].copy()
    larson = diag[diag["driver"] == "Kyle Larson"]
    if len(larson) == 0:
        print("Larson not in field?")
        return

    # For each numeric feature, rank Larson in the field.
    numeric = [c for c in diag_cols if c != "driver"]
    ranked = pd.DataFrame({
        "feature": numeric,
        "larson": [larson[c].iloc[0] for c in numeric],
        "field_median": [diag[c].median() for c in numeric],
        "field_min": [diag[c].min() for c in numeric],
        "field_max": [diag[c].max() for c in numeric],
    })
    # Rank Larson from best (lower is better for finish-related features, higher for others).
    def is_lower_better(feat):
        return any(k in feat for k in ["avg_finish", "loop_avg_ps"])
    def rank_larson(feat, val):
        vals = diag[feat].dropna().values
        if len(vals) == 0 or pd.isna(val):
            return None
        ascending = is_lower_better(feat)
        sorted_vals = np.sort(vals) if ascending else -np.sort(-vals)
        r = int(np.searchsorted(sorted_vals, val, side="left")) + 1
        return f"{r}/{len(vals)}"
    ranked["larson_rank"] = [rank_larson(f, v) for f, v in zip(ranked["feature"], ranked["larson"])]
    print("\nLarson's feature values vs field (n=36):")
    print(ranked.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
