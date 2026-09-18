"""Leave-one-race-out beta calibration on the backtest.

Reads data/processed/backtest_matchups.parquet — written by
backtest_oddslogic_v5.py using the *honest* calibration pipeline (leave-one-
race-out T, no T_match, empirical hazards). No model retraining here — this
script is just calibration analysis on top of the cached matchup probs, so
it runs in seconds and is guaranteed consistent with the backtest's numbers.

Previously this file had its own copy of the calibration/sampling pipeline,
which had silently drifted from the backtest (still using T_match, no LOO
CV, silent-drop bug on Suárez/Nemechek). Kimi's structural refactor point.

Outputs:
  - reliability diagram (before + after beta calibration)
  - Δ/bet ROI (raw vs calibrated)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest_oddslogic_v5 import american_to_prob

N_BINS = 10
EPS = 1e-4
CACHE = Path("data/processed/backtest_matchups.parquet")


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def fit_beta_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
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
    if not CACHE.exists():
        raise SystemExit(
            f"No cache at {CACHE}. Run `python backtest_oddslogic_v5.py` first "
            "so the honest pipeline can populate it."
        )
    df = pd.read_parquet(CACHE)
    print(f"Loaded {len(df)} matchup rows from {df['race_idx'].nunique()} races.\n")

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

    # Symmetrize so both A>B and B>A rows populate the reliability bins.
    def symmetrize(col_p, col_y):
        p1 = df[col_p].to_numpy(); y1 = df[col_y].to_numpy()
        return np.concatenate([p1, 1 - p1]), np.concatenate([y1, 1 - y1])

    p_pre, y_pre = symmetrize("p_a_raw", "outcome_a")
    p_post, y_post = symmetrize("p_a_cal", "outcome_a")
    ece_pre = bin_report(p_pre, y_pre, "=== BEFORE beta calibration ===")
    ece_post = bin_report(p_post, y_post, "=== AFTER  beta calibration ===")
    print(f"\nECE: {ece_pre:.4f} -> {ece_post:.4f}")

    # ---- Betting ROI pre / post ----
    def score_bets(p_col):
        wins = losses = picks = 0
        pl = 0.0
        for _, row in df.iterrows():
            p_a = row[p_col]; p_b = 1 - p_a
            oa, ob = row["odds_a"], row["odds_b"]
            ma = american_to_prob(oa); mb = american_to_prob(ob)
            vig = ma + mb; ma, mb = ma / vig, mb / vig
            edge_a, edge_b = p_a - ma, p_b - mb
            if edge_a <= 0 and edge_b <= 0: continue
            if edge_a >= edge_b:
                dec = american_to_decimal(oa) - 1
                won = row["outcome_a"] == 1
            else:
                dec = american_to_decimal(ob) - 1
                won = row["outcome_a"] == 0
            picks += 1
            if won: pl += dec; wins += 1
            else:   pl -= 1;   losses += 1
        return picks, wins, losses, pl, (pl / picks if picks else 0.0)

    picks_pre, w_pre, l_pre, pl_pre, roi_pre = score_bets("p_a_raw")
    picks_post, w_post, l_post, pl_post, roi_post = score_bets("p_a_cal")
    print(f"\n{'':<20} {'picks':>7} {'W-L':>10} {'hit%':>7} {'P/L':>8} {'Δ/bet':>8}")
    print(f"{'Raw':<20} {picks_pre:>7} {w_pre:>4d}-{l_pre:<5d} "
          f"{100*w_pre/max(w_pre+l_pre,1):>6.1f}% {pl_pre:>+8.2f} {roi_pre:>+8.4f}")
    print(f"{'Beta calibrated':<20} {picks_post:>7} {w_post:>4d}-{l_post:<5d} "
          f"{100*w_post/max(w_post+l_post,1):>6.1f}% {pl_post:>+8.2f} {roi_post:>+8.4f}")


if __name__ == "__main__":
    main()
