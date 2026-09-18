"""Top-N / winner / manufacturer reliability diagram on the backtest.

For each race in RACES, trains the walk-forward model, samples the field
at T_top_n (same setup as predict_next.py), and records for every
driver:
  - predicted winner prob vs actual (did they finish P1?)
  - predicted top-5 prob   vs actual (did they finish P<=5?)
  - predicted top-10 prob  vs actual (did they finish P<=10?)
  - predicted top-of-make prob vs actual

Also aggregates manufacturer-level (one per race):
  - predicted manuf-win prob vs actual (did their make win?)

Bins predictions, computes ECE, and reports per-bin over/underconfidence.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, per_driver_hazards,
)


N_BINS = 10


def bin_report(p_arr, y_arr, label):
    df = pd.DataFrame({"p": p_arr, "y": y_arr})
    bins = np.linspace(0, 1, N_BINS + 1)
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    n = len(df)
    print(f"\n{label}  (n={n})")
    print(f"{'Bin':<18} {'n':>5} {'pred':>8} {'actual':>8} {'gap':>8}")
    ece = 0.0
    for interval, sub in df.groupby("bin", observed=True):
        if len(sub) == 0: continue
        pa, ya = sub["p"].mean(), sub["y"].mean()
        ece += (len(sub) / n) * abs(ya - pa)
        print(f"{str(interval):<18} {len(sub):>5} {pa:>8.3f} {ya:>8.3f} {ya-pa:>+8.3f}")
    print(f"ECE: {ece:.4f}")
    return ece


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble
    from src.models.matchup_calibrate import fit_matchup_temperature

    races_df = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races_df["date"] = pd.to_datetime(races_df["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...")
    features = build_features(races_df, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    # Per-driver rows: p_win, p_top5, p_top10, p_top_of_make + outcomes
    win_p, win_y = [], []
    t5_p, t5_y = [], []
    t10_p, t10_y = [], []
    tom_p, tom_y = [], []
    # Per-race rows: manuf_win_prob per make, plus outcome
    mw_p, mw_y = [], []
    # Top-10 with race labels for LOO isotonic fit
    top10_rows: list[dict] = []

    for date, name, _ in RACES:
        r = races_df[
            ((races_df["date"] == pd.Timestamp(date))
             | (races_df["race_name"].str.contains(name, case=False, na=False)))
            & (races_df["season"] == 2026)
        ]
        if r.empty: continue
        rid = r.iloc[0]["race_id_short"]
        target_date_ts = r.iloc[0]["date"]
        tt = r.iloc[0]["track_type"]

        train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
        target = features[features["race_id_short"] == rid].reset_index(drop=True)
        if target.empty: continue

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

        # Matchup temperature per track type.
        cal_p, cal_o = {}, {}
        for vr, v_tt in zip(val_races, tt_list):
            haz_v = np.full(len(vr.scores), 0.08)
            rng_v = np.random.default_rng(0)
            pos_v = dist.sample_finishing_orders(
                vr.scores / max(T, 0.1), haz_v, n_samples=5000, rng=rng_v
            )
            m_v = dist.matchup_probs(pos_v)
            f = vr.finishes
            cal_p.setdefault(v_tt, []); cal_o.setdefault(v_tt, [])
            for ii in range(len(f)):
                for jj in range(ii + 1, len(f)):
                    if f[ii] <= 0 or f[jj] <= 0: continue
                    if f[ii] == f[jj]: continue
                    cal_p[v_tt].append(float(m_v[ii, jj]))
                    cal_o[v_tt].append(int(f[ii] < f[jj]))
        all_p = [p for lst in cal_p.values() for p in lst]
        all_o = [o for lst in cal_o.values() for o in lst]
        T_match_g = fit_matchup_temperature(np.asarray(all_p), np.asarray(all_o)) if all_p else 1.0
        if cal_p.get(tt) and len(cal_p[tt]) >= 200:
            T_match = fit_matchup_temperature(np.asarray(cal_p[tt]), np.asarray(cal_o[tt]))
        else:
            T_match = T_match_g

        T_effective = T * T_match
        T_top_n = max(T_effective * 1.5, 1.0) if T_effective < 1.0 else T_effective

        rng = np.random.default_rng(43)
        positions = dist.sample_finishing_orders(
            blended / T_top_n, haz, n_samples=N_SAMPLES, rng=rng
        )

        drivers = target["driver"].tolist()
        makes = (target["manufacturer"].tolist() if "manufacturer" in target.columns
                 else target["make"].tolist() if "make" in target.columns
                 else ["Unknown"] * len(drivers))

        # Model probabilities.
        p_win = (positions == 1).sum(axis=0) / N_SAMPLES
        p_top5 = (positions <= 5).sum(axis=0) / N_SAMPLES
        p_top10 = (positions <= 10).sum(axis=0) / N_SAMPLES

        # Top of make.
        by_make_indices = defaultdict(list)
        for i, m in enumerate(makes):
            by_make_indices[m].append(i)
        p_tom = np.zeros(len(drivers))
        for m, idxs in by_make_indices.items():
            if len(idxs) == 1:
                p_tom[idxs[0]] = 1.0
                continue
            sub_pos = positions[:, idxs]
            best_within = np.argmin(sub_pos, axis=1)
            for k, i in enumerate(idxs):
                p_tom[i] = float(np.mean(best_within == k))

        # Manufacturer win prob.
        make_win_counts = defaultdict(int)
        for s in range(N_SAMPLES):
            winner_idx = int(np.argmin(positions[s]))
            make_win_counts[makes[winner_idx]] += 1
        manuf_probs = {m: c / N_SAMPLES for m, c in make_win_counts.items()}

        # Actual outcomes.
        actual = entries[entries["race_id_short"] == rid][
            ["driver", "finish_pos", "make"]
        ].set_index("driver")

        # Winner: only one driver per race, but each driver contributes.
        for i, drv in enumerate(drivers):
            if drv not in actual.index: continue
            fp = int(actual.loc[drv, "finish_pos"])
            if fp <= 0: continue
            win_p.append(float(p_win[i]));   win_y.append(int(fp == 1))
            t5_p.append(float(p_top5[i]));   t5_y.append(int(fp <= 5))
            t10_p.append(float(p_top10[i])); t10_y.append(int(fp <= 10))
            top10_rows.append({
                "race": name, "driver": drv,
                "p_top10": float(p_top10[i]),
                "outcome_top10": int(fp <= 10),
            })
            # Top of make: did this driver beat the others of their make?
            drv_make = makes[i]
            teammates = [
                d for d, mk in zip(drivers, makes) if mk == drv_make and d != drv
            ]
            if teammates:
                other_finishes = [
                    int(actual.loc[t, "finish_pos"]) for t in teammates
                    if t in actual.index and actual.loc[t, "finish_pos"] > 0
                ]
                if other_finishes:
                    tom_actual = int(fp < min(other_finishes))
                    tom_p.append(float(p_tom[i]))
                    tom_y.append(tom_actual)

        # Manuf-win: which make actually won?
        winners = actual[actual["finish_pos"] == 1]
        if len(winners):
            winning_make = winners.iloc[0]["make"]
            for m, p in manuf_probs.items():
                mw_p.append(p)
                mw_y.append(int(m == winning_make))

        print(f"  processed {name}: cumulative {len(win_p)} driver-races")

    print(f"\n\n{'=' * 60}")
    print("RELIABILITY DIAGRAMS")
    print(f"{'=' * 60}")
    bin_report(np.array(win_p), np.array(win_y), "=== WINNER ===")
    bin_report(np.array(t5_p), np.array(t5_y),  "=== TOP 5 ===")
    bin_report(np.array(t10_p), np.array(t10_y), "=== TOP 10 ===")
    bin_report(np.array(tom_p), np.array(tom_y), "=== TOP OF MAKE ===")
    bin_report(np.array(mw_p), np.array(mw_y),  "=== MANUFACTURER WIN ===")

    # Cache raw top-10 (pred, outcome, race) for the isotonic fitter.
    pd.DataFrame(top10_rows).to_parquet(
        "data/processed/backtest_top10.parquet", index=False
    )
    print(f"\nSaved {len(top10_rows)} rows to data/processed/backtest_top10.parquet")


if __name__ == "__main__":
    main()
