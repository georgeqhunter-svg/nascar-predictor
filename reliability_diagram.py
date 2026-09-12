"""Reliability diagram: measure model calibration on backtest matchups.

For each matchup in our backtest, bin by predicted probability and compare
to the actual outcome rate. A perfectly calibrated model produces predictions
in the 60-70% bin that win 65% of the time on average.

Reports:
  - Per-bin: (predicted_avg, actual_rate, n_bets)
  - ECE (Expected Calibration Error): sum over bins of |pred - actual| * (n/N)
  - Overall miscalibration direction (overconfident / underconfident)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob,
)


N_BINS = 10  # bin predictions into deciles


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

    all_preds = []  # (model_prob, outcome_binary)

    for date, name, matchups in RACES:
        r = races[
            ((races["date"] == pd.Timestamp(date))
             | (races["race_name"].str.contains(name, case=False, na=False)))
            & (races["season"] == 2026)
        ]
        if r.empty:
            continue
        rid = r.iloc[0]["race_id_short"]
        target_date_ts = r.iloc[0]["date"]
        tt = r.iloc[0]["track_type"]

        train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
        target = features[features["race_id_short"] == rid].reset_index(drop=True)
        if target.empty:
            continue

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
        haz = np.full(len(target), HAZARD.get(tt, 0.08))

        # Per-track-type matchup temperature.
        cal_p, cal_o = {}, {}
        for vr, v_tt in zip(val_races, tt_list):
            haz_v = np.full(len(vr.scores), 0.08)
            rng_v = np.random.default_rng(0)
            pos_v = dist.sample_finishing_orders(
                vr.scores / max(T, 0.1), haz_v, n_samples=5000, rng=rng_v
            )
            m_v = dist.matchup_probs(pos_v)
            f = vr.finishes
            n_v = len(f)
            cal_p.setdefault(v_tt, [])
            cal_o.setdefault(v_tt, [])
            for ii in range(n_v):
                for jj in range(ii + 1, n_v):
                    if f[ii] <= 0 or f[jj] <= 0: continue
                    if f[ii] == f[jj]: continue
                    cal_p[v_tt].append(float(m_v[ii, jj]))
                    cal_o[v_tt].append(int(f[ii] < f[jj]))
        all_p = [p for lst in cal_p.values() for p in lst]
        all_o = [o for lst in cal_o.values() for o in lst]
        T_match_g = fit_matchup_temperature(np.asarray(all_p), np.asarray(all_o)) if all_p else 1.0
        if cal_p.get(tt) and len(cal_p[tt]) >= 200:
            T_match = fit_matchup_temperature(
                np.asarray(cal_p[tt]), np.asarray(cal_o[tt]),
            )
        else:
            T_match = T_match_g

        T_effective = T * T_match
        rng = np.random.default_rng(42)
        positions = dist.sample_finishing_orders(
            blended / T_effective, haz, n_samples=N_SAMPLES, rng=rng
        )
        matchup_mtx = dist.matchup_probs(positions)

        driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
        actual = entries[entries["race_id_short"] == rid][["driver", "finish_pos"]].set_index("driver")

        for a, b, oa, ob in matchups:
            if a not in driver_to_idx or b not in driver_to_idx: continue
            if a not in actual.index or b not in actual.index: continue
            i, j = driver_to_idx[a], driver_to_idx[b]
            p_a = float(matchup_mtx[i, j])
            fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
            a_won = fa < fb
            all_preds.append((p_a, int(a_won)))

        print(f"  processed {name}: cumulative {len(all_preds)} preds")

    df = pd.DataFrame(all_preds, columns=["p", "outcome"])
    n = len(df)
    print(f"\nTotal predictions: {n}")
    print()

    # Bin predictions
    bins = np.linspace(0, 1, N_BINS + 1)
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)

    print("=" * 80)
    print("RELIABILITY DIAGRAM")
    print("=" * 80)
    print(f"{'Bin range':<18} {'n':>6} {'Predicted avg':>14} {'Actual rate':>13} {'Gap':>10}")
    print("-" * 80)
    ece = 0.0
    for interval, sub in df.groupby("bin", observed=True):
        if len(sub) == 0: continue
        pred_avg = sub["p"].mean()
        actual = sub["outcome"].mean()
        gap = actual - pred_avg
        ece += (len(sub) / n) * abs(gap)
        print(f"{str(interval):<18} {len(sub):>6} {pred_avg:>13.3f}  {actual:>12.3f}  {gap:>+9.3f}")

    print(f"\nECE (Expected Calibration Error): {ece:.4f}")
    print()

    # Sign of miscalibration by confidence side
    high_conf = df[df["p"] > 0.7]
    low_conf = df[df["p"] < 0.3]
    if len(high_conf) > 0:
        pred_hi = high_conf["p"].mean(); act_hi = high_conf["outcome"].mean()
        print(f"High-confidence (p>0.70): predicted avg {pred_hi:.3f}, actual {act_hi:.3f}, "
              f"{'OVERCONFIDENT' if pred_hi > act_hi else 'UNDERCONFIDENT'} by {abs(pred_hi-act_hi):.3f}")
    if len(low_conf) > 0:
        pred_lo = low_conf["p"].mean(); act_lo = low_conf["outcome"].mean()
        print(f"Low-confidence  (p<0.30): predicted avg {pred_lo:.3f}, actual {act_lo:.3f}, "
              f"{'UNDERCONFIDENT' if pred_lo > act_lo else 'OVERCONFIDENT'} by {abs(pred_lo-act_lo):.3f}")


if __name__ == "__main__":
    main()
