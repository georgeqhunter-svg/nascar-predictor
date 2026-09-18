"""Fit an isotonic calibrator for top-10 probabilities.

Reads data/processed/backtest_top10.parquet (from reliability_top_n.py).
1. Leave-one-race-out fits + evaluation to verify improvement.
2. Fits final isotonic on ALL data, saves pickle for predict_next.py.

Only top-10 needs this — the reliability diagram showed winner, top-5,
and top-of-make are already well-calibrated.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

N_BINS = 10
EPS = 1e-4


class BetaCalibrator:
    """Beta calibration: logistic regression on [log(p), log(1-p)].
    Smooth 2-parameter fit — preserves driver ordering, unlike isotonic
    which collapses into flat plateaus on limited data.
    """
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, max_iter=1000)

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        X = np.column_stack([np.log(p), np.log(1 - p)])
        self.lr.fit(X, y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        X = np.column_stack([np.log(p), np.log(1 - p)])
        return self.lr.predict_proba(X)[:, 1]


def fit(p, y):
    return BetaCalibrator().fit(p, y)


def bin_report(p, y, label):
    df = pd.DataFrame({"p": p, "y": y})
    bins = np.linspace(0, 1, N_BINS + 1)
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    print(f"\n{label}  (n={len(df)})")
    print(f"{'Bin':<18} {'n':>5} {'pred':>8} {'actual':>8} {'gap':>8}")
    ece = 0.0
    for interval, sub in df.groupby("bin", observed=True):
        if len(sub) == 0: continue
        pa, ya = sub["p"].mean(), sub["y"].mean()
        ece += (len(sub) / len(df)) * abs(ya - pa)
        print(f"{str(interval):<18} {len(sub):>5} {pa:>8.3f} {ya:>8.3f} {ya-pa:>+8.3f}")
    print(f"ECE: {ece:.4f}")
    return ece


def main():
    df = pd.read_parquet("data/processed/backtest_top10.parquet")
    print(f"Loaded {len(df)} rows from {df['race'].nunique()} races")

    # -------- Leave-one-race-out evaluation --------
    p_cal = np.zeros(len(df))
    for race in df["race"].unique():
        m_train = df["race"] != race
        m_test = df["race"] == race
        cal = fit(
            df.loc[m_train, "p_top10"].to_numpy(),
            df.loc[m_train, "outcome_top10"].to_numpy(),
        )
        p_cal[m_test] = cal.transform(df.loc[m_test, "p_top10"].to_numpy())

    ece_raw = bin_report(
        df["p_top10"].to_numpy(), df["outcome_top10"].to_numpy(),
        "=== RAW TOP-10 (before calibration) ===",
    )
    ece_cal = bin_report(
        p_cal, df["outcome_top10"].to_numpy(),
        "=== LOO CALIBRATED TOP-10 ===",
    )
    print(f"\nECE: {ece_raw:.4f} -> {ece_cal:.4f}   "
          f"({'+' if ece_cal < ece_raw else '-'}"
          f"{100 * abs(ece_cal - ece_raw) / max(ece_raw, EPS):.1f}%)")

    # -------- Final fit on all data --------
    final_cal = fit(
        df["p_top10"].to_numpy(),
        df["outcome_top10"].to_numpy(),
    )
    out = Path("data/processed/top10_calibrator.pkl")
    with open(out, "wb") as f:
        pickle.dump(final_cal, f)
    print(f"\nSaved final calibrator -> {out}")

    # Sanity: show the mapping at a few key points.
    print("\n=== Mapping preview ===")
    for x in [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]:
        y = float(final_cal.transform([x])[0])
        print(f"  raw {x:.2f} -> calibrated {y:.3f}")


if __name__ == "__main__":
    main()
