"""Compare calibrators and edge thresholds on the cached backtest matchups.

Uses data/processed/backtest_matchups.parquet from beta_calibrate_backtest.py
(no re-training needed — instant).

Calibrators evaluated (all leave-one-race-out):
  raw:   uncalibrated model probabilities.
  temp:  logit temperature — p' = sigmoid(logit(p) / tau), 1 free parameter.
  beta:  logistic regression on [log p, log(1-p)] — 2 free parameters.

For each, reports:
  - ECE
  - ROI at edge thresholds: any, 3%, 5%, 7%, 10%
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from backtest_oddslogic_v5 import american_to_prob

EPS = 1e-4


def american_to_decimal(odds):
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def fit_beta(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(p), np.log(1 - p)])
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, y)
    return lr


def apply_beta(cal, p):
    p = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(p), np.log(1 - p)])
    return cal.predict_proba(X)[:, 1]


def fit_temp(p, y):
    """Fit tau where p' = sigmoid(logit(p) / tau) minimizes log loss."""
    p = np.clip(p, EPS, 1 - EPS)
    z = np.log(p / (1 - p))
    def loss(tau):
        z2 = z / tau
        p2 = 1 / (1 + np.exp(-z2))
        p2 = np.clip(p2, EPS, 1 - EPS)
        return -np.mean(y * np.log(p2) + (1 - y) * np.log(1 - p2))
    res = minimize_scalar(loss, bounds=(0.1, 10.0), method="bounded")
    return res.x


def apply_temp(tau, p):
    p = np.clip(p, EPS, 1 - EPS)
    z = np.log(p / (1 - p)) / tau
    return 1 / (1 + np.exp(-z))


def loo(df, fit_fn, apply_fn):
    p_out = np.zeros(len(df))
    for rid in df["race_idx"].unique():
        m_tr = df["race_idx"] != rid
        m_te = df["race_idx"] == rid
        cal = fit_fn(df.loc[m_tr, "p_a_raw"].to_numpy(),
                     df.loc[m_tr, "outcome_a"].to_numpy())
        p_out[m_te] = apply_fn(cal, df.loc[m_te, "p_a_raw"].to_numpy())
    return p_out


def ece(p, y, n_bins=10):
    df = pd.DataFrame({"p": p, "y": y})
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    total = 0.0
    for _, sub in df.groupby("bin", observed=True):
        if len(sub) == 0: continue
        total += (len(sub) / len(df)) * abs(sub["y"].mean() - sub["p"].mean())
    return total


def bets_at_threshold(df, p_col, thresh):
    """Kelly-style: bet the side with the biggest positive edge above threshold."""
    picks = wins = losses = 0
    pl = 0.0
    for _, row in df.iterrows():
        p_a = row[p_col]; p_b = 1 - p_a
        oa, ob = row["odds_a"], row["odds_b"]
        ma = american_to_prob(oa); mb = american_to_prob(ob)
        vig = ma + mb
        ma, mb = ma / vig, mb / vig
        edge_a = p_a - ma; edge_b = p_b - mb
        best_edge = max(edge_a, edge_b)
        if best_edge < thresh:
            continue
        if edge_a >= edge_b:
            dec = american_to_decimal(oa) - 1
            won = row["outcome_a"] == 1
        else:
            dec = american_to_decimal(ob) - 1
            won = row["outcome_a"] == 0
        picks += 1
        if won: pl += dec; wins += 1
        else:   pl -= 1;   losses += 1
    roi = (pl / picks) if picks else 0.0
    return picks, wins, losses, pl, roi


def symmetrize(p, y):
    return np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y])


def main():
    df = pd.read_parquet("data/processed/backtest_matchups.parquet")
    print(f"Loaded {len(df)} matchups across {df['race_idx'].nunique()} races\n")

    df["p_a_temp"] = loo(df, fit_temp, apply_temp)
    df["p_a_beta"] = loo(df, fit_beta, apply_beta)

    print("=" * 70)
    print("CALIBRATION QUALITY (ECE, symmetrized)")
    print("=" * 70)
    for label, col in [("raw", "p_a_raw"), ("temp", "p_a_temp"), ("beta", "p_a_beta")]:
        p_s, y_s = symmetrize(df[col].to_numpy(), df["outcome_a"].to_numpy())
        print(f"  {label:<6} ECE = {ece(p_s, y_s):.4f}")

    # Also inspect the temperature we're fitting.
    tau_all = fit_temp(df["p_a_raw"].to_numpy(), df["outcome_a"].to_numpy())
    print(f"\n  Full-data logit temperature tau = {tau_all:.3f}")
    print(f"  (>1 means stretching; we're pulling in by that factor)")

    print()
    print("=" * 70)
    print("BETTING ROI BY EDGE THRESHOLD (flat $1 stake, LOO calibration)")
    print("=" * 70)
    thresholds = [0.0, 0.03, 0.05, 0.07, 0.10]
    print(f"{'thresh':>7} {'method':>7} {'picks':>6} {'W-L':>10} {'hit%':>7} "
          f"{'P/L':>9} {'roi/bet':>9}")
    for th in thresholds:
        for label, col in [("raw", "p_a_raw"), ("temp", "p_a_temp"), ("beta", "p_a_beta")]:
            picks, w, l, pl, roi = bets_at_threshold(df, col, th)
            print(f"{th:>7.2f} {label:>7} {picks:>6} {w:>4d}-{l:<5d} "
                  f"{100*w/max(w+l,1):>6.1f}% {pl:>+9.2f} {roi:>+9.4f}")
        print()


if __name__ == "__main__":
    main()
