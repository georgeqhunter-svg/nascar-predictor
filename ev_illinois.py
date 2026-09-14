"""EV analysis for BetUS Enjoy Illinois 300 matchups.

Runs the model to get matchup probabilities, then computes edge over BetUS
prices for each of the ~40 posted matchups. Reports bets ranked by expected
value in units per dollar staked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob,
)


TARGET_NAME = "Enjoy Illinois"
TARGET_DATE = "2026-09-13"


# FanDuel matchups (posted post-qualifying): (driver_a, driver_b, odds_a, odds_b)
MATCHUPS = [
    ("Daniel Suárez",  "Michael McDowell",  -134,  106),
    ("Joey Logano",    "William Byron",     -280,  210),
    ("Josh Berry",     "Bubba Wallace",     -108, -118),
    ("Ross Chastain",  "Brad Keselowski",   -122, -104),
    ("Ryan Blaney",    "Kyle Larson",       -142,  112),
    ("Carson Hocevar", "Ryan Preece",       -140,  110),
    # New batch (2:05 PM postings)
    ("Ty Gibbs",       "Tyler Reddick",     -130,  100),
    ("Ross Chastain",  "Chris Buescher",    -120, -110),
    ("Austin Cindric", "Josh Berry",        -115, -115),
]


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble
    from src.models.matchup_calibrate import fit_matchup_temperature

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...")
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    target_ts = pd.Timestamp(TARGET_DATE)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(TARGET_NAME, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    rid = r.iloc[0]["race_id_short"]
    target_date_ts = r.iloc[0]["date"]
    tt = r.iloc[0]["track_type"]

    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == rid].reset_index(drop=True)

    print("Training model...")
    model = GBMEnsemble()
    model.fit(train, n_estimators=15, **TIGHT_REG)

    race_ids = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
    val_ids = race_ids[30:]
    val_races, tt_list = [], []
    for vid in val_ids:
        sub = train[train["race_id_short"] == vid]
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

    raw = model.predict_scores(target)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
            / (target["pl_effective"].std() + 1e-9))
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)
    from backtest_oddslogic_v5 import per_driver_hazards
    haz = per_driver_hazards(target, tt)

    # Per-track-type matchup temperature.
    cal_probs_by_type, cal_out_by_type = {}, {}
    for vr, v_tt in zip(val_races, tt_list):
        haz_v = np.full(len(vr.scores), 0.08)
        rng_v = np.random.default_rng(0)
        pos_v = dist.sample_finishing_orders(
            vr.scores / max(T, 0.1), haz_v, n_samples=5000, rng=rng_v
        )
        m_v = dist.matchup_probs(pos_v)
        f = vr.finishes
        n_v = len(f)
        cal_probs_by_type.setdefault(v_tt, [])
        cal_out_by_type.setdefault(v_tt, [])
        for ii in range(n_v):
            for jj in range(ii + 1, n_v):
                if f[ii] <= 0 or f[jj] <= 0: continue
                if f[ii] == f[jj]: continue
                cal_probs_by_type[v_tt].append(float(m_v[ii, jj]))
                cal_out_by_type[v_tt].append(int(f[ii] < f[jj]))
    all_p = [p for lst in cal_probs_by_type.values() for p in lst]
    all_o = [o for lst in cal_out_by_type.values() for o in lst]
    T_match_global = fit_matchup_temperature(np.asarray(all_p), np.asarray(all_o)) if all_p else 1.0
    if cal_probs_by_type.get(tt) and len(cal_probs_by_type[tt]) >= 200:
        T_match = fit_matchup_temperature(
            np.asarray(cal_probs_by_type[tt]),
            np.asarray(cal_out_by_type[tt]),
        )
    else:
        T_match = T_match_global

    T_effective = T * T_match
    print(f"T={T}, T_m={T_match:.2f}, T_effective={T_effective:.3f}\n")

    # Sample once at unified temperature.
    rng = np.random.default_rng(42)
    positions = dist.sample_finishing_orders(
        blended / T_effective, haz, n_samples=N_SAMPLES, rng=rng
    )
    matchup_mtx = dist.matchup_probs(positions)

    driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}

    rows = []
    unresolved = []
    for a, b, oa, ob in MATCHUPS:
        if a not in driver_to_idx or b not in driver_to_idx:
            unresolved.append((a, b))
            continue
        i, j = driver_to_idx[a], driver_to_idx[b]
        p_a = float(matchup_mtx[i, j])
        p_b = 1 - p_a

        ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma, mb = ma_raw / vig, mb_raw / vig  # vig-free market implied

        # EV per $1 stake on each side
        b_dec_a = american_to_decimal(oa) - 1  # profit-per-$1 if wins
        b_dec_b = american_to_decimal(ob) - 1
        ev_a = p_a * b_dec_a - (1 - p_a)
        ev_b = p_b * b_dec_b - (1 - p_b)

        pick, pick_p, pick_odds, pick_ev, pick_market = ("A", p_a, oa, ev_a, ma) if ev_a > ev_b else ("B", p_b, ob, ev_b, mb)
        pick_driver = a if pick == "A" else b
        other_driver = b if pick == "A" else a
        edge = pick_p - pick_market

        rows.append({
            "pick": pick_driver,
            "against": other_driver,
            "odds": pick_odds,
            "model_p": pick_p,
            "market_p_vigfree": pick_market,
            "edge": edge,
            "ev_per_dollar": pick_ev,
        })

    df = pd.DataFrame(rows).sort_values("ev_per_dollar", ascending=False)

    print(f"{'=' * 90}")
    print("BEST-EV MATCHUPS (BetUS Enjoy Illinois 300)")
    print(f"{'=' * 90}")
    print(df.to_string(
        index=False,
        formatters={
            "model_p": "{:.1%}".format,
            "market_p_vigfree": "{:.1%}".format,
            "edge": "{:+.1%}".format,
            "ev_per_dollar": "{:+.4f}".format,
        },
    ))

    print(f"\n{'=' * 90}")
    print("BETS TO TAKE (edge >= 30% — backtest-validated threshold)")
    print(f"{'=' * 90}")
    plus = df[df["edge"] >= 0.30]
    if len(plus) > 0:
        print(plus.to_string(
            index=False,
            formatters={
                "model_p": "{:.1%}".format,
                "market_p_vigfree": "{:.1%}".format,
                "edge": "{:+.1%}".format,
                "ev_per_dollar": "{:+.4f}".format,
            },
        ))
        print(f"\n{len(plus)} bets with >=30% edge, mean EV = ${plus['ev_per_dollar'].mean():.4f}/$")
    else:
        print("No matchups clear the 30% edge threshold — skip the market.")

    print(f"\n{'=' * 90}")
    print("Marginal (10-30% edge — do NOT bet these; shown for context)")
    print(f"{'=' * 90}")
    mid = df[(df["edge"] >= 0.10) & (df["edge"] < 0.30)]
    if len(mid) > 0:
        print(mid.to_string(
            index=False,
            formatters={
                "model_p": "{:.1%}".format,
                "market_p_vigfree": "{:.1%}".format,
                "edge": "{:+.1%}".format,
                "ev_per_dollar": "{:+.4f}".format,
            },
        ))

    if unresolved:
        print(f"\n{len(unresolved)} matchups unresolved (driver not in entry list):")
        for a, b in unresolved:
            print(f"  {a} vs {b}")


if __name__ == "__main__":
    main()
