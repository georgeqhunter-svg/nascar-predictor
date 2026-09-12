"""Per-race, per-driver diagnostic for the OddsLogic backtest.

For each race, prints:
  1. Driver-level table: model_win_prob, market_win_prob, actual_finish, error.
     Sorted by |model - market| so systematic mispositioning surfaces on top.
  2. "Confident-wrong" matchups: where model >= 65% and got it wrong.
  3. Disagreement summary: matchups where model and market picked different sides.

Uses the same setup as backtest_oddslogic_v5.py so it's an apples-to-apples read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob,
)


def market_win_prob(driver: str, matchups: list, drivers_in_race: list) -> float:
    """Aggregate market implied win prob for a driver from all its matchups.

    We estimate P(win) as the geometric mean of pairwise beat probabilities
    raised to (N-1)/count — a proxy since we don't have outrights."""
    beats = []
    for a, b, oa, ob in matchups:
        ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma, mb = ma_raw / vig, mb_raw / vig
        if a == driver:
            beats.append(ma)
        elif b == driver:
            beats.append(mb)
    if not beats:
        return float("nan")
    # Rough proxy: mean of pairwise beat probs.
    return float(np.mean(beats))


def run_one(target_date, name_match, matchups, races, entries, sessions, loopstats, laptimes):
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
        print(f"[SKIP] {name_match}: race not in parquet")
        return
    target_rid = r.iloc[0]["race_id_short"]
    target_date_ts = r.iloc[0]["date"]
    tt = r.iloc[0]["track_type"]

    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == target_rid].reset_index(drop=True)
    if target.empty:
        print(f"[SKIP] {name_match}: target race empty")
        return

    model = GBMEnsemble()
    model.fit(train, n_estimators=15, **TIGHT_REG)

    # Calibrate.
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
    win_prob = (positions == 0).sum(axis=1) / N_SAMPLES

    driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
    actual = entries[entries["race_id_short"] == target_rid][
        ["driver", "finish_pos"]
    ].set_index("driver")
    drivers_in_race = list(driver_to_idx.keys())

    # ---- Driver-level table ----
    rows = []
    seen = set()
    for a, b, oa, ob in matchups:
        for d in (a, b):
            if d in seen or d not in driver_to_idx or d not in actual.index:
                continue
            seen.add(d)
            i = driver_to_idx[d]
            model_beat = float(np.mean([
                matchup_mtx[i, driver_to_idx[opp]]
                for opp in drivers_in_race
                if opp != d and opp in driver_to_idx
            ]))
            market_beat = market_win_prob(d, matchups, drivers_in_race)
            fp = int(actual.loc[d, "finish_pos"])
            rows.append({
                "driver": d,
                "model_beat": model_beat,
                "market_beat": market_beat,
                "diff": model_beat - market_beat,
                "actual_finish": fp,
                "model_win_pct": win_prob[i] * 100,
            })
    df = pd.DataFrame(rows)
    df["abs_diff"] = df["diff"].abs()
    df = df.sort_values("abs_diff", ascending=False)

    print(f"\n{'='*80}\n{name_match} ({tt}, T={T})\n{'='*80}")
    print(df[["driver", "model_beat", "market_beat", "diff", "actual_finish", "model_win_pct"]]
          .head(12).to_string(index=False, float_format="%.3f"))

    # ---- Confident-wrong matchups ----
    conf_wrong = []
    for a, b, oa, ob in matchups:
        if a not in driver_to_idx or b not in driver_to_idx: continue
        if a not in actual.index or b not in actual.index: continue
        i, j = driver_to_idx[a], driver_to_idx[b]
        p = float(matchup_mtx[i, j])
        fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
        model_pick = a if p > 0.5 else b
        actual_winner = a if fa < fb else b
        if model_pick != actual_winner and max(p, 1-p) >= 0.65:
            ma = american_to_prob(oa); mb = american_to_prob(ob); v = ma + mb
            conf_wrong.append({
                "a": a, "b": b, "model_p_a": p,
                "market_p_a": ma / v,
                "picked": model_pick, "actual": actual_winner,
                "conf": max(p, 1-p),
            })
    if conf_wrong:
        cw = pd.DataFrame(conf_wrong).sort_values("conf", ascending=False)
        print(f"\nConfident-wrong (>=65%), n={len(cw)}:")
        print(cw.to_string(index=False, float_format="%.3f"))

    # ---- Disagreement summary ----
    disagreements = 0; disagree_won = 0
    agreements = 0; agree_won = 0
    for a, b, oa, ob in matchups:
        if a not in driver_to_idx or b not in driver_to_idx: continue
        if a not in actual.index or b not in actual.index: continue
        i, j = driver_to_idx[a], driver_to_idx[b]
        p = float(matchup_mtx[i, j])
        ma = american_to_prob(oa); mb = american_to_prob(ob); v = ma + mb
        ma_norm = ma / v
        model_a = p > 0.5; market_a = ma_norm > 0.5
        fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
        a_won = fa < fb
        if model_a == market_a:
            agreements += 1
            if (model_a and a_won) or (not model_a and not a_won):
                agree_won += 1
        else:
            disagreements += 1
            if (model_a and a_won) or (not model_a and not a_won):
                disagree_won += 1
    print(f"\nAgreement: {agreements} matchups, model correct on {agree_won}/{agreements} ({agree_won/max(agreements,1)*100:.0f}%)")
    print(f"Disagreement: {disagreements} matchups, model correct on {disagree_won}/{max(disagreements,1)} ({disagree_won/max(disagreements,1)*100:.0f}%)")


def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    for date, name, matchups in RACES:
        run_one(date, name, matchups, races, entries, sessions, loopstats, laptimes)


if __name__ == "__main__":
    main()
