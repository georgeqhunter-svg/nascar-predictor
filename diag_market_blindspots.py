"""Market blind spot investigation.

Slice backtest matchups by dimensions and compute per-bin:
  - n matchups
  - market log-loss (higher = market pricing worse)
  - model log-loss (higher = we're worse)
  - Delta = model_ll - market_ll (negative = WE beat market in this bin)

Reports the bins where:
  A) Market log-loss is highest — market's known blind spots
  B) Δ is most negative — where WE are best relative to market
  C) Both A and B — actionable segments to bet aggressively

Dimensions sliced:
  - Track type
  - Market favorite strength (close 45-55, moderate 55-65, heavy 65-100)
  - Both drivers same team vs different
  - Season half (first 13 races vs playoffs 14+)
  - Race chaos level (cautions, if available from races.parquet)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import american_to_prob

CACHE = Path("data/processed/backtest_matchups.parquet")
MIN_N_PER_BIN = 20  # bins with fewer matchups get suppressed as noise


def _log_clip(p, floor=1e-9):
    return float(np.log(np.clip(p, floor, 1.0 - floor)))


def main():
    if not CACHE.exists():
        raise SystemExit(f"No cache at {CACHE}. Run backtest first.")
    df = pd.read_parquet(CACHE)

    # Vig-free market prob on side A.
    ma = df["odds_a"].apply(american_to_prob)
    mb = df["odds_b"].apply(american_to_prob)
    vig = ma + mb
    df["market_p_a"] = ma / vig
    df["market_fav_p"] = np.maximum(df["market_p_a"], 1 - df["market_p_a"])

    # Per-matchup log-loss for market and model (on the side that won).
    df["market_p_won"] = np.where(df["outcome_a"] == 1, df["market_p_a"], 1 - df["market_p_a"])
    df["model_p_won"] = np.where(df["outcome_a"] == 1, df["p_a_raw"], 1 - df["p_a_raw"])
    df["market_ll"] = -df["market_p_won"].apply(_log_clip)
    df["model_ll"] = -df["model_p_won"].apply(_log_clip)
    df["delta"] = df["model_ll"] - df["market_ll"]

    # Enrich with race + driver metadata.
    races = pd.read_parquet("data/processed/races.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries = pd.read_parquet("data/processed/entries.parquet")

    # Map race name -> track_type + date + n_cautions
    r26 = races[races["season"] == 2026].copy()
    r26["name_key"] = r26["race_name"].str.lower()
    df["name_lower"] = df["race"].str.lower()

    def _first_row(sub):
        return sub.iloc[0] if len(sub) else None

    tt_map, date_map, cau_map = {}, {}, {}
    for name in df["name_lower"].unique():
        # Find the r26 rows whose name_key contains (or is contained in) name.
        matches = r26[
            r26["name_key"].apply(lambda k: (name in k) or (k in name))
        ]
        row = _first_row(matches)
        if row is None:
            continue
        tt_map[name] = row["track_type"]
        date_map[name] = row["date"]
        try:
            cau_map[name] = float(row["cautions"]) if pd.notna(row["cautions"]) else np.nan
        except (TypeError, ValueError):
            cau_map[name] = np.nan

    df["track_type"] = df["name_lower"].map(tt_map)
    df["race_date"] = df["name_lower"].map(date_map)
    df["cautions"] = df["name_lower"].map(cau_map)

    # Same-team matchup indicator.
    ent = entries[["driver", "team", "race_id_short"]].dropna()
    driver_race_team = {(str(row["driver"]).lower(), row["race_id_short"]): row["team"]
                        for _, row in ent.iterrows()}
    # We don't have race_id_short in cache; use driver-level majority team as proxy.
    driver_team_mode = (ent.groupby("driver")["team"]
                       .agg(lambda s: s.mode().iloc[0] if len(s) else "")
                       .to_dict())
    df["team_a"] = df["a"].map(driver_team_mode)
    df["team_b"] = df["b"].map(driver_team_mode)
    df["same_team"] = df["team_a"] == df["team_b"]

    # Playoff indicator (2026 playoffs start ~race #27, but we're using the 23-race
    # backtest span. Rough proxy: last 6 races of the season = playoff-adjacent).
    df["race_order"] = df.groupby("race_date").ngroup()
    n_races = df["race_order"].max() + 1
    df["playoff_adj"] = df["race_order"] >= n_races - 7

    # Market favorite strength bins.
    df["fav_bin"] = pd.cut(
        df["market_fav_p"], [0.50, 0.55, 0.60, 0.65, 0.75, 1.01],
        labels=["50-55%", "55-60%", "60-65%", "65-75%", "75%+"],
        include_lowest=True,
    )

    # Caution bins (chaos proxy)
    if df["cautions"].notna().any():
        df["cautions_bin"] = pd.cut(
            df["cautions"], [-0.1, 4, 8, 12, 50],
            labels=["low (0-4)", "med (5-8)", "high (9-12)", "chaos (13+)"],
        )

    def bin_summary(grouper_name, sort_col="delta"):
        rows = []
        for label, sub in df.groupby(grouper_name, observed=True):
            if len(sub) < MIN_N_PER_BIN: continue
            rows.append({
                grouper_name: str(label),
                "n": len(sub),
                "market_ll": float(sub["market_ll"].mean()),
                "model_ll": float(sub["model_ll"].mean()),
                "delta": float(sub["delta"].mean()),
                "model_hit%": 100 * float(
                    ((sub["p_a_raw"] > 0.5) == sub["outcome_a"].astype(bool)).mean()
                ),
                "market_hit%": 100 * float(
                    ((sub["market_p_a"] > 0.5) == sub["outcome_a"].astype(bool)).mean()
                ),
            })
        if not rows: return
        out = pd.DataFrame(rows).sort_values(sort_col)
        print(out.to_string(index=False, formatters={
            "market_ll": "{:.4f}".format, "model_ll": "{:.4f}".format,
            "delta": "{:+.4f}".format,
            "model_hit%": "{:.1f}%".format, "market_hit%": "{:.1f}%".format,
        }))

    print("=" * 90)
    print("BY TRACK TYPE (sorted by our Delta; negative = we beat market)")
    print("=" * 90)
    bin_summary("track_type")

    print("\n" + "=" * 90)
    print("BY MARKET FAVORITE STRENGTH (close matchups often = market's blind spot)")
    print("=" * 90)
    bin_summary("fav_bin")

    print("\n" + "=" * 90)
    print("SAME-TEAM VS DIFFERENT-TEAM (books may misprice internal teammate dynamics)")
    print("=" * 90)
    bin_summary("same_team")

    print("\n" + "=" * 90)
    print("REGULAR SEASON VS PLAYOFF-ADJACENT (playoff pressure mispricing?)")
    print("=" * 90)
    bin_summary("playoff_adj")

    if "cautions_bin" in df.columns:
        print("\n" + "=" * 90)
        print("BY CAUTION COUNT (proxies race chaos)")
        print("=" * 90)
        bin_summary("cautions_bin")

    # Cross-tabs for the most interesting single-dim finding
    print("\n" + "=" * 90)
    print("CROSS: track_type × favorite strength (find specific actionable segments)")
    print("=" * 90)
    rows = []
    for (tt, fav), sub in df.groupby(["track_type", "fav_bin"], observed=True):
        if len(sub) < 15: continue
        rows.append({
            "track_type": tt, "fav_bin": str(fav),
            "n": len(sub),
            "market_ll": float(sub["market_ll"].mean()),
            "model_ll": float(sub["model_ll"].mean()),
            "delta": float(sub["delta"].mean()),
        })
    if rows:
        cx = pd.DataFrame(rows).sort_values("delta")
        print(cx.to_string(index=False, formatters={
            "market_ll": "{:.4f}".format, "model_ll": "{:.4f}".format,
            "delta": "{:+.4f}".format,
        }))


if __name__ == "__main__":
    main()
