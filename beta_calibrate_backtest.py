"""Leave-one-race-out beta calibration on the backtest.

For each race:
  1. Run the model to get matchup probabilities for every posted matchup.
  2. Fit a beta calibrator on the (p, outcome) pairs from the OTHER 21 races.
  3. Apply that calibrator to this race's probabilities.
  4. Score EV against the OddsLogic closing lines.

Reports before/after reliability, ECE, hit rate, and Δ vs closing lines
(units per bet at flat stake).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob,
)


N_BINS = 10
EPS = 1e-4


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def fit_beta_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Beta calibration: logistic regression on [log(p), log(1-p)].

    This class of calibrator handles the classic 'stretched to tails'
    miscalibration without pinning any single-point mapping.
    """
    p = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(p), np.log(1 - p)])
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, y)
    return lr


def apply_beta(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(p), np.log(1 - p)])
    return cal.predict_proba(X)[:, 1]


def bin_report(p, y, label):
    df = pd.DataFrame({"p": p, "y": y})
    bins = np.linspace(0, 1, N_BINS + 1)
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    n = len(df)
    print(f"\n{label}")
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
    features = build_features(
        races_df, entries, sessions, loopstats=loopstats, laptimes=laptimes
    )
    features["date"] = pd.to_datetime(features["date"])

    rows = []  # per matchup: race_idx, p_a (uncal), outcome, odds_a, odds_b

    for race_idx, (date, name, matchups) in enumerate(RACES):
        r = races_df[
            ((races_df["date"] == pd.Timestamp(date))
             | (races_df["race_name"].str.contains(name, case=False, na=False)))
            & (races_df["season"] == 2026)
        ]
        if r.empty:
            continue
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
        T_by_type = find_best_temperature_by_type(
            val_races, tt_list, default_T=1.0, n_samples=1500
        )
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
            T_match = fit_matchup_temperature(
                np.asarray(cal_p[tt]), np.asarray(cal_o[tt])
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
            rows.append({
                "race_idx": race_idx,
                "race": name,
                "a": a, "b": b,
                "p_a_raw": p_a,
                "outcome_a": int(fa < fb),
                "odds_a": oa, "odds_b": ob,
            })
        print(f"  {name}: cumulative {len(rows)} rows")

    df = pd.DataFrame(rows)
    print(f"\nTotal predictions: {len(df)}\n")
    df.to_parquet("data/processed/backtest_matchups.parquet", index=False)
    print("Saved cache -> data/processed/backtest_matchups.parquet")

    # ---- Leave-one-race-out beta calibration ----
    p_cal = np.zeros(len(df))
    for race_idx in df["race_idx"].unique():
        mask_train = df["race_idx"] != race_idx
        mask_test = df["race_idx"] == race_idx
        cal = fit_beta_calibrator(
            df.loc[mask_train, "p_a_raw"].to_numpy(),
            df.loc[mask_train, "outcome_a"].to_numpy(),
        )
        p_cal[mask_test] = apply_beta(cal, df.loc[mask_test, "p_a_raw"].to_numpy())
    df["p_a_cal"] = p_cal

    # ---- Reliability diagrams before / after ----
    # Symmetrize: include both A>B and B>A rows so tails are populated evenly.
    def symmetrize(col_p, col_y):
        p1 = df[col_p].to_numpy()
        y1 = df[col_y].to_numpy()
        p2 = 1 - p1
        y2 = 1 - y1
        return np.concatenate([p1, p2]), np.concatenate([y1, y2])

    p_pre, y_pre = symmetrize("p_a_raw", "outcome_a")
    p_post, y_post = symmetrize("p_a_cal", "outcome_a")
    ece_pre = bin_report(p_pre, y_pre, "=== BEFORE beta calibration ===")
    ece_post = bin_report(p_post, y_post, "=== AFTER  beta calibration ===")
    print(f"\nECE: {ece_pre:.4f} -> {ece_post:.4f}")

    # ---- Betting ROI pre / post ----
    def score_bets(p_col):
        wins = losses = stake = pl = 0.0
        picks = 0
        for _, row in df.iterrows():
            p_a = row[p_col]; p_b = 1 - p_a
            oa, ob = row["odds_a"], row["odds_b"]
            ma = american_to_prob(oa); mb = american_to_prob(ob)
            vig = ma + mb
            ma, mb = ma / vig, mb / vig
            edge_a = p_a - ma
            edge_b = p_b - mb
            if edge_a <= 0 and edge_b <= 0:
                continue
            if edge_a >= edge_b:
                dec = american_to_decimal(oa) - 1
                won = row["outcome_a"] == 1
            else:
                dec = american_to_decimal(ob) - 1
                won = row["outcome_a"] == 0
            picks += 1
            stake += 1.0
            if won:
                pl += dec
                wins += 1
            else:
                pl -= 1.0
                losses += 1
        return picks, wins, losses, pl, (pl / stake if stake else 0.0)

    picks_pre, w_pre, l_pre, pl_pre, roi_pre = score_bets("p_a_raw")
    picks_post, w_post, l_post, pl_post, roi_post = score_bets("p_a_cal")
    print(f"\n{'':<20} {'picks':>7} {'W-L':>10} {'hit%':>7} {'P/L':>8} {'Δ/bet':>8}")
    print(f"{'Raw':<20} {picks_pre:>7} {w_pre:>4.0f}-{l_pre:<5.0f} "
          f"{100*w_pre/max(w_pre+l_pre,1):>6.1f}% {pl_pre:>+8.2f} {roi_pre:>+8.4f}")
    print(f"{'Beta calibrated':<20} {picks_post:>7} {w_post:>4.0f}-{l_post:<5.0f} "
          f"{100*w_post/max(w_post+l_post,1):>6.1f}% {pl_post:>+8.2f} {roi_post:>+8.4f}")


if __name__ == "__main__":
    main()
