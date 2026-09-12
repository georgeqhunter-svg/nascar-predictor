# backtest_oddslogic_v3.py — four-race comparison vs OddsLogic closing lines.
#   1. 2026-08-09 Iowa Corn 350 at Iowa (short)
#   2. 2026-08-23 Dollar Tree 301 at New Hampshire (short)
#   3. 2026-08-29 Coke Zero Sugar 400 at Daytona (superspeedway)
#   4. 2026-09-06 Cook Out Southern 500 at Darlington (intermediate)

import numpy as np
import pandas as pd

IOWA = [
    ("Denny Hamlin",      "Ryan Blaney",        -105, -115),
    ("Denny Hamlin",      "Christopher Bell",   -115, -105),
    ("Denny Hamlin",      "Kyle Larson",        -155,  135),
    ("Denny Hamlin",      "William Byron",      -180,  155),
    ("Ryan Blaney",       "Christopher Bell",   -120,  100),
    ("Ryan Blaney",       "Kyle Larson",        -160,  140),
    ("Ryan Blaney",       "William Byron",      -180,  155),
    ("Christopher Bell",  "Kyle Larson",        -150,  130),
    ("Christopher Bell",  "William Byron",      -170,  150),
    ("Kyle Larson",       "William Byron",      -135,  115),
    ("Joey Logano",       "Chase Briscoe",      -145,  125),
    ("Joey Logano",       "Chase Elliott",      -150,  130),
    ("Joey Logano",       "Ty Gibbs",           -145,  125),
    ("Joey Logano",       "Tyler Reddick",      -170,  150),
    ("Chase Briscoe",     "Chase Elliott",      -115, -105),
    ("Chase Briscoe",     "Ty Gibbs",           -110, -110),
    ("Chase Briscoe",     "Tyler Reddick",      -145,  125),
    ("Chase Elliott",     "Ty Gibbs",           -105, -115),
    ("Chase Elliott",     "Tyler Reddick",      -140,  120),
    ("Ty Gibbs",          "Tyler Reddick",      -145,  125),
    ("Bubba Wallace",     "Carson Hocevar",     -180,  155),
    ("Bubba Wallace",     "Brad Keselowski",    -145,  125),
    ("Bubba Wallace",     "Chris Buescher",     -210,  180),
    ("Bubba Wallace",     "Ross Chastain",      -235,  200),
    ("Carson Hocevar",    "Brad Keselowski",     120, -140),
    ("Carson Hocevar",    "Chris Buescher",     -140,  120),
    ("Carson Hocevar",    "Ross Chastain",      -165,  145),
    ("Brad Keselowski",   "Chris Buescher",     -180,  155),
    ("Brad Keselowski",   "Ross Chastain",      -200,  175),
    ("Chris Buescher",    "Ross Chastain",      -135,  115),
]

LOUDON = [
    ("Ryan Blaney",       "Christopher Bell",    -170,  150),
    ("Ryan Blaney",       "Joey Logano",         -200,  175),
    ("Ryan Blaney",       "Denny Hamlin",        -185,  160),
    ("Ryan Blaney",       "Chase Briscoe",       -270,  230),
    ("Christopher Bell",  "Joey Logano",         -110, -110),
    ("Christopher Bell",  "Denny Hamlin",        -125,  105),
    ("Christopher Bell",  "Chase Briscoe",       -185,  160),
    ("Joey Logano",       "Denny Hamlin",        -115, -105),
    ("Joey Logano",       "Chase Briscoe",       -165,  145),
    ("Denny Hamlin",      "Chase Briscoe",       -170,  150),
    ("Kyle Larson",       "William Byron",        145, -165),
    ("Kyle Larson",       "Ty Gibbs",             140, -160),
    ("Kyle Larson",       "Tyler Reddick",       -180,  155),
    ("Kyle Larson",       "Chase Elliott",       -185,  160),
    ("William Byron",     "Ty Gibbs",            -135,  115),
    ("William Byron",     "Tyler Reddick",       -185,  160),
    ("William Byron",     "Chase Elliott",       -245,  210),
    ("Ty Gibbs",          "Tyler Reddick",       -170,  150),
    ("Ty Gibbs",          "Chase Elliott",       -235,  200),
    ("Tyler Reddick",     "Chase Elliott",       -130,  110),
    ("Josh Berry",        "Austin Cindric",      -260,  220),
    ("Josh Berry",        "Bubba Wallace",       -250,  215),
    ("Josh Berry",        "Ross Chastain",       -265,  225),
    ("Josh Berry",        "Chris Buescher",      -235,  200),
    ("Austin Cindric",    "Bubba Wallace",       -105, -115),
    ("Austin Cindric",    "Ross Chastain",       -130,  110),
    ("Austin Cindric",    "Chris Buescher",       100, -120),
    ("Bubba Wallace",     "Ross Chastain",       -125,  105),
    ("Bubba Wallace",     "Chris Buescher",      -110, -110),
    ("Ross Chastain",     "Chris Buescher",       125, -145),
    ("Brad Keselowski",   "Shane Van Gisbergen",  125, -155),
    ("Brad Keselowski",   "Carson Hocevar",      -110, -120),
    ("Brad Keselowski",   "Alex Bowman",         -120, -110),
    ("Shane Van Gisbergen","Carson Hocevar",     -155,  125),
    ("Shane Van Gisbergen","Alex Bowman",        -145,  115),
]

DAYTONA = [
    ("Ryan Blaney",       "Joey Logano",        -110, -110),
    ("Ryan Blaney",       "William Byron",      -120,  100),
    ("Ryan Blaney",       "Tyler Reddick",      -125,  105),
    ("Ryan Blaney",       "Christopher Bell",   -130,  110),
    ("Joey Logano",       "William Byron",      -125,  105),
    ("Joey Logano",       "Tyler Reddick",      -125,  105),
    ("Joey Logano",       "Christopher Bell",   -130,  110),
    ("William Byron",     "Tyler Reddick",      -115, -105),
    ("William Byron",     "Christopher Bell",   -120,  100),
    ("Tyler Reddick",     "Christopher Bell",   -110, -110),
    ("Chase Elliott",     "Kyle Larson",        -120,  100),
    ("Chase Elliott",     "Carson Hocevar",     -115, -105),
    ("Chase Elliott",     "Austin Cindric",     -105, -115),
    ("Chase Elliott",     "Bubba Wallace",      -115, -105),
    ("Kyle Larson",       "Carson Hocevar",     -105, -115),
    ("Kyle Larson",       "Austin Cindric",      105, -125),
    ("Kyle Larson",       "Bubba Wallace",      -105, -115),
    ("Carson Hocevar",    "Austin Cindric",      100, -120),
    ("Carson Hocevar",    "Bubba Wallace",      -110, -110),
    ("Austin Cindric",    "Bubba Wallace",      -120,  100),
    ("Chase Briscoe",     "Denny Hamlin",       -110, -110),
    ("Chase Briscoe",     "Chris Buescher",     -110, -110),
    ("Chase Briscoe",     "Brad Keselowski",    -105, -115),
    ("Chase Briscoe",     "Ty Gibbs",           -120,  100),
    ("Denny Hamlin",      "Chris Buescher",     -110, -110),
    ("Denny Hamlin",      "Brad Keselowski",    -105, -115),
    ("Denny Hamlin",      "Ty Gibbs",           -120,  100),
    ("Chris Buescher",    "Brad Keselowski",    -105, -115),
    ("Chris Buescher",    "Ty Gibbs",           -120,  100),
    ("Brad Keselowski",   "Ty Gibbs",           -125,  105),
]

DARLINGTON = [
    ("Denny Hamlin",      "Tyler Reddick",      -110, -110),
    ("Denny Hamlin",      "Chase Briscoe",      -135,  115),
    ("Denny Hamlin",      "Kyle Larson",        -150,  130),
    ("Denny Hamlin",      "Ryan Blaney",        -160,  140),
    ("Tyler Reddick",     "Chase Briscoe",      -135,  115),
    ("Tyler Reddick",     "Kyle Larson",        -150,  130),
    ("Tyler Reddick",     "Ryan Blaney",        -160,  140),
    ("Chase Briscoe",     "Kyle Larson",        -125,  105),
    ("Chase Briscoe",     "Ryan Blaney",        -135,  115),
    ("Kyle Larson",       "Ryan Blaney",        -120,  100),
    ("William Byron",     "Christopher Bell",   -120,  100),
    ("William Byron",     "Joey Logano",        -165,  145),
    ("William Byron",     "Ty Gibbs",           -120,  100),
    ("William Byron",     "Chase Elliott",      -160,  140),
    ("Christopher Bell",  "Joey Logano",        -155,  135),
    ("Christopher Bell",  "Ty Gibbs",           -115, -105),
    ("Christopher Bell",  "Chase Elliott",      -150,  130),
    ("Joey Logano",       "Ty Gibbs",            130, -150),
    ("Joey Logano",       "Chase Elliott",      -110, -110),
    ("Ty Gibbs",          "Chase Elliott",      -145,  125),
    ("Bubba Wallace",     "Chris Buescher",     -120,  100),
    ("Bubba Wallace",     "Brad Keselowski",     100, -120),
    ("Bubba Wallace",     "Ross Chastain",      -210,  180),
    ("Bubba Wallace",     "Erik Jones",         -150,  130),
    ("Chris Buescher",    "Brad Keselowski",     110, -130),
    ("Chris Buescher",    "Ross Chastain",      -200,  175),
    ("Chris Buescher",    "Erik Jones",         -140,  120),
    ("Brad Keselowski",   "Ross Chastain",      -210,  180),
    ("Brad Keselowski",   "Erik Jones",         -155,  135),
    ("Ross Chastain",     "Erik Jones",          130, -150),
]

RACES = [
    ("2026-08-09", "Iowa Corn",       IOWA),
    ("2026-08-23", "Dollar Tree 301", LOUDON),
    ("2026-08-29", "Coke Zero",       DAYTONA),
    ("2026-09-06", "Southern 500",    DARLINGTON),
]

TIGHT_REG = {"num_leaves": 31, "min_data_in_leaf": 25, "lambda_l2": 2.0}
ALPHA = 0.85
N_SAMPLES = 30_000
HAZARD = {"superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
          "road": 0.04, "unique": 0.07}


def american_to_prob(odds):
    return 100.0 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def _log_clip(p, floor=1e-9):
    return float(np.log(np.clip(p, floor, 1.0 - floor)))


def run_race(target_date, name_match, matchups, races, entries, sessions, loopstats, laptimes):
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble

    target_ts = pd.Timestamp(target_date)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(name_match, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    if r.empty:
        print(f"could not find race {name_match} on {target_date}")
        return None
    target_rid = r.iloc[0]["race_id_short"]
    target_date_ts = r.iloc[0]["date"]
    tt = r.iloc[0]["track_type"]

    print(f"\n{'=' * 70}")
    print(f"{r.iloc[0]['race_name']} at {r.iloc[0]['track_name']} "
          f"({tt}), race_id={target_rid}, date={target_date_ts.date()}")
    print(f"{'=' * 70}")

    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == target_rid].reset_index(drop=True)
    if target.empty:
        return None

    print(f"Training on {train['race_id_short'].nunique()} races.")
    model = GBMEnsemble()
    model.fit(train, n_estimators=15, **TIGHT_REG)

    race_order = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
    val_ids = race_order[30:]
    val_races, tt_list = [], []
    for rid in val_ids:
        sub = train[train["race_id_short"] == rid]
        raw = model.predict_scores(sub)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                / (sub["pl_effective"].std() + 1e-9))
        b = ALPHA * gbm_z + (1 - ALPHA) * pl_z
        b = b / max(b.std(), 1e-6)
        val_races.append(RaceScoresGT(
            scores=b, finishes=sub["finish_pos"].to_numpy(),
            is_dnf=sub["is_dnf"].to_numpy(),
            hazards=np.full(len(sub), HAZARD.get(sub["track_type"].iloc[0], 0.08)),
        ))
        tt_list.append(sub["track_type"].iloc[0])
    T_by_type = find_best_temperature_by_type(val_races, tt_list, default_T=1.0, n_samples=1500)
    T = T_by_type.get(tt, 1.0)
    print(f"T for {tt} = {T}")

    raw = model.predict_scores(target)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
            / (target["pl_effective"].std() + 1e-9))
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)
    haz = np.full(len(target), HAZARD.get(tt, 0.08))
    rng = np.random.default_rng(42)
    positions = dist.sample_finishing_orders(blended / T, haz, n_samples=N_SAMPLES, rng=rng)
    matchup_mtx = dist.matchup_probs(positions)

    driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
    actual = entries[entries["race_id_short"] == target_rid][
        ["driver", "finish_pos"]
    ].set_index("driver")

    market_ll, model_ll = [], []
    market_correct = model_correct = 0
    n = 0
    for a, b, oa, ob in matchups:
        if a not in driver_to_idx or b not in driver_to_idx: continue
        if a not in actual.index or b not in actual.index: continue
        ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma, mb = ma_raw / vig, mb_raw / vig
        i, j = driver_to_idx[a], driver_to_idx[b]
        p_model_a = float(matchup_mtx[i, j]); p_model_b = 1 - p_model_a
        fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
        a_won = fa < fb
        p_market_won = ma if a_won else mb
        p_model_won = p_model_a if a_won else p_model_b
        market_correct += int((ma > mb and a_won) or (mb > ma and not a_won))
        model_correct += int((p_model_a > p_model_b and a_won) or (p_model_b > p_model_a and not a_won))
        market_ll.append(-_log_clip(p_market_won))
        model_ll.append(-_log_clip(p_model_won))
        n += 1

    mkt_mean = float(np.mean(market_ll)); mdl_mean = float(np.mean(model_ll))
    print(f"n={n}  market_ll={mkt_mean:.4f} ({market_correct}/{n})  "
          f"model_ll={mdl_mean:.4f} ({model_correct}/{n})  Δ={mdl_mean-mkt_mean:+.4f}")
    return {"race": name_match, "tt": tt, "n": n, "market_ll": mkt_mean, "model_ll": mdl_mean,
            "market_correct": market_correct, "model_correct": model_correct}


def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    summary = []
    for date, name, matchups in RACES:
        result = run_race(date, name, matchups, races, entries, sessions, loopstats, laptimes)
        if result:
            summary.append(result)

    print("\n" + "=" * 70)
    print("SUMMARY (four races vs OddsLogic closing lines)")
    print("=" * 70)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    total_mkt = sum(s["market_ll"] * s["n"] for s in summary) / sum(s["n"] for s in summary)
    total_mdl = sum(s["model_ll"] * s["n"] for s in summary) / sum(s["n"] for s in summary)
    total_n = sum(s["n"] for s in summary)
    total_mkt_c = sum(s["market_correct"] for s in summary)
    total_mdl_c = sum(s["model_correct"] for s in summary)
    print(f"\nCombined ({total_n} matchups, weighted mean):")
    print(f"  Market: log-loss {total_mkt:.4f}, correct {total_mkt_c}/{total_n} ({total_mkt_c/total_n*100:.1f}%)")
    print(f"  Model:  log-loss {total_mdl:.4f}, correct {total_mdl_c}/{total_n} ({total_mdl_c/total_n*100:.1f}%)")
    print(f"  Δ: {total_mdl - total_mkt:+.4f}")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
