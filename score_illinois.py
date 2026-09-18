"""score_illinois.py - LOCKBOX: Enjoy Illinois 300 (2026-09-13), graded.

Scores the committed model on the 9 posted matchups, graded against actual
finishes. Reports per-matchup and aggregate log-loss / Brier vs the vig-free
market, and settles the model's EV picks at flat $1 stakes (plus the
>=30%-edge "official bet" subset separately).

Pipeline mirrors the committed backtest_oddslogic_v5.py exactly (same TIGHT_REG,
ALPHA, HAZARD, per-type T, matchup Tm, N_SAMPLES=30000, seed 42).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob, per_driver_hazards,
)

TARGET_NAME = "Illinois"
TARGET_DATE = "2026-09-13"

# FanDuel matchups posted post-qualifying: (driver_a, driver_b, odds_a, odds_b)
MATCHUPS = [
    ("Daniel Suarez",  "Michael McDowell",  -134,  106),
    ("Joey Logano",    "William Byron",     -280,  210),
    ("Josh Berry",     "Bubba Wallace",     -108, -118),
    ("Ross Chastain",  "Brad Keselowski",   -122, -104),
    ("Ryan Blaney",    "Kyle Larson",       -142,  112),
    ("Carson Hocevar", "Ryan Preece",       -140,  110),
    ("Ty Gibbs",       "Tyler Reddick",     -130,  100),
    ("Ross Chastain",  "Chris Buescher",    -120, -110),
    ("Austin Cindric", "Josh Berry",        -115, -115),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower()


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def _log_clip(p: float, floor: float = 1e-9) -> float:
    return float(np.log(min(max(p, floor), 1.0 - floor)))


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

    print("Building features (slow part)...")
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
    print(f"Target: {r.iloc[0].get('race_name', TARGET_NAME)} ({rid}, {tt})")

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
    haz = per_driver_hazards(target, tt)

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
                if f[ii] <= 0 or f[jj] <= 0:
                    continue
                if f[ii] == f[jj]:
                    continue
                cal_probs_by_type[v_tt].append(float(m_v[ii, jj]))
                cal_out_by_type[v_tt].append(int(f[ii] < f[jj]))
    all_p = [p for lst in cal_probs_by_type.values() for p in lst]
    all_o = [o for lst in cal_out_by_type.values() for o in lst]
    T_match_global = fit_matchup_temperature(np.asarray(all_p), np.asarray(all_o)) if all_p else 1.0
    if cal_probs_by_type.get(tt) and len(cal_probs_by_type[tt]) >= 200:
        T_match = fit_matchup_temperature(
            np.asarray(cal_probs_by_type[tt]), np.asarray(cal_out_by_type[tt]))
    else:
        T_match = T_match_global

    T_effective = T * T_match
    rng = np.random.default_rng(42)
    positions = dist.sample_finishing_orders(
        blended / T_effective, haz, n_samples=N_SAMPLES, rng=rng
    )
    matchup_mtx = dist.matchup_probs(positions)

    name_to_idx = {_norm(d): i for i, d in enumerate(target["driver"].values)}
    actual = entries[entries["race_id_short"] == rid][["driver", "finish_pos"]].copy()
    actual["nkey"] = actual["driver"].map(_norm)
    finish = dict(zip(actual["nkey"], actual["finish_pos"]))

    rows, unresolved = [], []
    for a, b, oa, ob in MATCHUPS:
        ka, kb = _norm(a), _norm(b)
        if ka not in name_to_idx or kb not in name_to_idx:
            unresolved.append((a, b))
            continue
        i, j = name_to_idx[ka], name_to_idx[kb]
        p_a = float(matchup_mtx[i, j])
        ma_raw, mb_raw = american_to_prob(oa), american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma = ma_raw / vig
        fa, fb = finish.get(ka), finish.get(kb)
        if fa is None or fb is None:
            unresolved.append((a, b))
            continue
        a_won = fa < fb
        p_model_won = p_a if a_won else 1 - p_a
        p_market_won = ma if a_won else 1 - ma
        # EV pick: side with higher expected value per $1
        ev_a = p_a * (american_to_decimal(oa) - 1) - (1 - p_a)
        ev_b = (1 - p_a) * (american_to_decimal(ob) - 1) - p_a
        if ev_a >= ev_b:
            pick, pick_odds, pick_won, pick_ev, edge = a, oa, a_won, ev_a, p_a - ma
        else:
            pick, pick_odds, pick_won, pick_ev, edge = b, ob, not a_won, ev_b, (1 - p_a) - (1 - ma)
        profit = (american_to_decimal(pick_odds) - 1) if pick_won else -1.0
        rows.append({
            "a": a, "b": b, "model_p_a": p_a, "mkt_p_a": ma,
            "winner": a if a_won else b,
            "model_ll": -_log_clip(p_model_won), "mkt_ll": -_log_clip(p_market_won),
            "model_brier": (p_a - float(a_won)) ** 2, "mkt_brier": (ma - float(a_won)) ** 2,
            "model_right": int((p_a > 0.5) == a_won), "mkt_right": int((ma > 0.5) == a_won),
            "pick": pick, "pick_ev": pick_ev, "edge": edge,
            "pick_won": int(pick_won), "profit": profit,
        })

    df = pd.DataFrame(rows)
    print()
    print("=" * 100)
    print(f"LOCKBOX: {TARGET_NAME} ({rid}, {tt})  T={T}  Tm={T_match:.2f}  n={len(df)} matchups")
    print("=" * 100)
    show = df[["a", "b", "model_p_a", "mkt_p_a", "winner", "model_ll", "mkt_ll"]].copy()
    print(show.to_string(index=False, formatters={
        "model_p_a": "{:.3f}".format, "mkt_p_a": "{:.3f}".format,
        "model_ll": "{:.3f}".format, "mkt_ll": "{:.3f}".format}))
    print()
    print(f"log-loss : model {df["model_ll"].mean():.4f}  market {df["mkt_ll"].mean():.4f}  "
          f"Delta {df["model_ll"].mean() - df["mkt_ll"].mean():+.4f}")
    print(f"Brier    : model {df["model_brier"].mean():.4f}  market {df["mkt_brier"].mean():.4f}")
    print(f"correct  : model {df["model_right"].sum()}/{len(df)}  market {df["mkt_right"].sum()}/{len(df)}")
    print()
    print("EV picks (higher-EV side of every matchup, flat $1):")
    print(f"  {df["pick_won"].sum()}/{len(df)} won, {df["profit"].sum():+.2f} units")
    bets = df[df["edge"] >= 0.30]
    if len(bets):
        print("Official bets (edge >= 30%):")
        print(bets[["pick", "pick_ev", "edge", "pick_won", "profit"]].to_string(
            index=False, formatters={"pick_ev": "{:+.3f}".format, "edge": "{:+.1%}".format}))
        print(f"  {bets["pick_won"].sum()}/{len(bets)} won, {bets["profit"].sum():+.2f} units")
    else:
        print("No matchup cleared the 30% edge threshold.")
    if unresolved:
        print("Unresolved (name/finish not matched):")
        for a, b in unresolved:
            print(f"  {a} vs {b}")


if __name__ == "__main__":
    main()
