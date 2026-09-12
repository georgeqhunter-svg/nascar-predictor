"""Sweep mild temperature values + wider edge thresholds.

Uses the cached parquet from beta_calibrate_backtest.py.

Also reports: for each (tau, thresh) pair, how often the calibrated pick
DIFFERS from the raw pick — high divergence means calibration is flipping
directions and destroying informational signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import american_to_prob

EPS = 1e-4


def american_to_decimal(odds):
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def apply_temp(p, tau):
    p = np.clip(p, EPS, 1 - EPS)
    z = np.log(p / (1 - p)) / tau
    return 1 / (1 + np.exp(-z))


def ece(p, y, n_bins=10):
    df = pd.DataFrame({"p": p, "y": y})
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True)
    total = 0.0
    for _, sub in df.groupby("bin", observed=True):
        if len(sub) == 0: continue
        total += (len(sub) / len(df)) * abs(sub["y"].mean() - sub["p"].mean())
    return total


def symmetrize(p, y):
    return np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y])


def bets(df, p_col, thresh):
    picks = wins = losses = 0
    pl = 0.0
    for _, row in df.iterrows():
        p_a = row[p_col]; p_b = 1 - p_a
        oa, ob = row["odds_a"], row["odds_b"]
        ma = american_to_prob(oa); mb = american_to_prob(ob)
        vig = ma + mb; ma, mb = ma / vig, mb / vig
        e_a = p_a - ma; e_b = p_b - mb
        best = max(e_a, e_b)
        if best < thresh: continue
        if e_a >= e_b:
            dec = american_to_decimal(oa) - 1
            won = row["outcome_a"] == 1
        else:
            dec = american_to_decimal(ob) - 1
            won = row["outcome_a"] == 0
        picks += 1
        if won: pl += dec; wins += 1
        else:   pl -= 1;   losses += 1
    return picks, wins, losses, pl, (pl / picks if picks else 0.0)


def pick_flip_rate(df, tau):
    """Fraction of matchups where calibration flips which side has positive edge."""
    p_raw = df["p_a_raw"].to_numpy()
    p_cal = apply_temp(p_raw, tau)
    m_a = np.array([american_to_prob(o) for o in df["odds_a"]])
    m_b = np.array([american_to_prob(o) for o in df["odds_b"]])
    v = m_a + m_b; m_a /= v; m_b /= v
    raw_pick_a = (p_raw - m_a) >= (1 - p_raw - m_b)
    cal_pick_a = (p_cal - m_a) >= (1 - p_cal - m_b)
    return (raw_pick_a != cal_pick_a).mean()


def main():
    df = pd.read_parquet("data/processed/backtest_matchups.parquet")
    n = len(df)
    y = df["outcome_a"].to_numpy()

    print("=" * 80)
    print("TEMPERATURE SWEEP")
    print("=" * 80)
    print(f"{'tau':>5} {'flip%':>7} {'ECE':>7} | "
          f"{'th=0':>16} {'th=.05':>16} {'th=.10':>16} {'th=.15':>16}")
    print(f"{'':>5} {'':>7} {'':>7} | "
          f"{'picks roi':>16} {'picks roi':>16} {'picks roi':>16} {'picks roi':>16}")
    print("-" * 96)
    for tau in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.85]:
        col = f"p_a_temp_{tau}"
        df[col] = apply_temp(df["p_a_raw"].to_numpy(), tau)
        p_s, y_s = symmetrize(df[col].to_numpy(), y)
        e = ece(p_s, y_s)
        flip = pick_flip_rate(df, tau) * 100
        parts = []
        for th in [0.0, 0.05, 0.10, 0.15]:
            picks, w, l, pl, roi = bets(df, col, th)
            parts.append(f"{picks:>4d} {roi:>+8.4f}")
        print(f"{tau:>5.2f} {flip:>6.1f}% {e:>7.4f} | " + " ".join(f"{p:>15}" for p in parts))

    print()
    print("=" * 80)
    print("EXTENDED THRESHOLD SWEEP FOR RAW MODEL")
    print("=" * 80)
    print(f"{'thresh':>8} {'picks':>6} {'W-L':>10} {'hit%':>7} {'P/L':>9} {'roi':>9}")
    for th in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        picks, w, l, pl, roi = bets(df, "p_a_raw", th)
        print(f"{th:>8.2f} {picks:>6} {w:>4d}-{l:<5d} "
              f"{100*w/max(w+l,1):>6.1f}% {pl:>+9.2f} {roi:>+9.4f}")


if __name__ == "__main__":
    main()
