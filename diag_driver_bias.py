"""Per-driver bias audit on backtest matchups.

For each driver, computes:
  n_matchups      — how many backtest matchups they appear in
  n_favorite      — how many the market made them favorite in
  bias_vs_market  — mean(model_p - market_p) when this driver is side A.
                    Positive = model is systematically HIGHER on them than the
                    market. Negative = model FADES this driver relative to market.
  model_prob_avg  — mean model probability on this driver
  actual_win_rate — how often they actually won those matchups
  calibration     — model_prob_avg - actual_win_rate. Positive = OVERCONFIDENT
                    on this driver. Negative = UNDERCONFIDENT.
  pick_hit_rate   — when model picked this driver (edge > 0), how often correct
  pick_n          — sample size for the pick_hit_rate

Sorted to surface the biggest offenders first: drivers where the model is
systematically wrong in a specific direction and it's costing bets.

Runs against data/processed/backtest_matchups.parquet — the cached matchups
from the honest walk-forward backtest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import american_to_prob

CACHE = Path("data/processed/backtest_matchups.parquet")
MIN_APPEARANCES = 10  # drivers must appear this many times to show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent-races", type=int, default=None,
                    help="Only include the most recent N races (by race_idx). "
                         "Use to check whether historical driver biases still "
                         "apply or if the market/model have adapted.")
    ap.add_argument("--min-appearances", type=int, default=None,
                    help=f"Override MIN_APPEARANCES (default {MIN_APPEARANCES}).")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"No cache at {CACHE}. Run backtest first.")
    df = pd.read_parquet(CACHE)

    if args.recent_races is not None:
        max_idx = df["race_idx"].max()
        threshold = max_idx - args.recent_races + 1
        df = df[df["race_idx"] >= threshold].copy()
        print(f"Filtered to most recent {args.recent_races} races "
              f"({df['race_idx'].nunique()} unique races, {len(df)} matchups)\n")

    min_app = args.min_appearances or MIN_APPEARANCES

    # Compute vig-free market probability from odds.
    ma_raw = df["odds_a"].apply(american_to_prob)
    mb_raw = df["odds_b"].apply(american_to_prob)
    vig = ma_raw + mb_raw
    df["market_p_a"] = ma_raw / vig
    df["edge_a"] = df["p_a_raw"] - df["market_p_a"]
    df["model_picked_a"] = df["edge_a"] > 0

    # Build a driver-level view: each driver contributes both A-side and B-side rows.
    a_rows = df[["a", "p_a_raw", "market_p_a", "outcome_a", "edge_a", "model_picked_a"]].rename(
        columns={
            "a": "driver", "p_a_raw": "model_p", "market_p_a": "market_p",
            "outcome_a": "outcome", "edge_a": "edge", "model_picked_a": "model_picked",
        }
    )
    b_rows = df[["b", "p_a_raw", "market_p_a", "outcome_a", "edge_a", "model_picked_a"]].rename(
        columns={"b": "driver"}
    )
    # Flip perspective to the B-side driver.
    b_rows["model_p"] = 1 - b_rows["p_a_raw"]
    b_rows["market_p"] = 1 - b_rows["market_p_a"]
    b_rows["outcome"] = 1 - b_rows["outcome_a"]
    b_rows["edge"] = b_rows["model_p"] - b_rows["market_p"]
    b_rows["model_picked"] = b_rows["edge"] > 0
    b_rows = b_rows[["driver", "model_p", "market_p", "outcome", "edge", "model_picked"]]

    driver = pd.concat([
        a_rows[["driver", "model_p", "market_p", "outcome", "edge", "model_picked"]],
        b_rows,
    ], ignore_index=True)

    agg = driver.groupby("driver").apply(lambda g: pd.Series({
        "n": len(g),
        "n_market_fav": int((g["market_p"] > 0.5).sum()),
        "bias_vs_market": float((g["model_p"] - g["market_p"]).mean()),
        "model_p_avg": float(g["model_p"].mean()),
        "actual_win_rate": float(g["outcome"].mean()),
        "calibration": float(g["model_p"].mean() - g["outcome"].mean()),
        "pick_hit_rate": float(g.loc[g["model_picked"], "outcome"].mean()) if g["model_picked"].any() else np.nan,
        "pick_n": int(g["model_picked"].sum()),
    })).reset_index()

    agg = agg[agg["n"] >= min_app].copy()
    agg["abs_bias"] = agg["bias_vs_market"].abs()

    print("=" * 88)
    print("DRIVERS THE MODEL DISAGREES WITH MARKET ON MOST (bias_vs_market)")
    print("=" * 88)
    fmt = {
        "bias_vs_market": "{:+.3f}".format,
        "model_p_avg": "{:.3f}".format,
        "actual_win_rate": "{:.3f}".format,
        "calibration": "{:+.3f}".format,
        "pick_hit_rate": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
    }
    print(agg.sort_values("abs_bias", ascending=False).head(15)[[
        "driver", "n", "n_market_fav", "bias_vs_market", "model_p_avg",
        "actual_win_rate", "calibration", "pick_hit_rate", "pick_n",
    ]].to_string(index=False, formatters=fmt))

    print()
    print("=" * 88)
    print("MOST OVERCONFIDENT DRIVERS (positive calibration = model overshoots reality)")
    print("=" * 88)
    print(agg.sort_values("calibration", ascending=False).head(10)[[
        "driver", "n", "model_p_avg", "actual_win_rate", "calibration",
        "pick_hit_rate", "pick_n",
    ]].to_string(index=False, formatters=fmt))

    print()
    print("=" * 88)
    print("MOST UNDERCONFIDENT DRIVERS (negative calibration = model undersells)")
    print("=" * 88)
    print(agg.sort_values("calibration").head(10)[[
        "driver", "n", "model_p_avg", "actual_win_rate", "calibration",
        "pick_hit_rate", "pick_n",
    ]].to_string(index=False, formatters=fmt))

    print()
    print("=" * 88)
    print("DRIVERS MODEL PICKED (edge > 0) BUT LOST — worst pick hit rates")
    print("=" * 88)
    picked = agg[agg["pick_n"] >= 8].copy().sort_values("pick_hit_rate")
    print(picked.head(10)[[
        "driver", "pick_n", "pick_hit_rate", "bias_vs_market", "calibration",
    ]].to_string(index=False, formatters=fmt))

    print()
    print("=" * 88)
    print("DRIVERS MODEL PICKED AND WON MOST — best pick hit rates")
    print("=" * 88)
    print(picked.sort_values("pick_hit_rate", ascending=False).head(10)[[
        "driver", "pick_n", "pick_hit_rate", "bias_vs_market", "calibration",
    ]].to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
