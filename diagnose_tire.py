"""Diagnose whether tire_decay is a bug or already-captured signal.

Checks:
  1. Coverage: what fraction of (race, driver) rows have non-NaN tire_decay?
     By track_type. If short/road are mostly NaN, our thresholds are broken.
  2. Distribution: min/mean/max of tire_decay_type_10 by track_type. Small
     range on a track type = noise.
  3. Correlation with avg_finish_10 (are we double-counting?).
  4. Correlation with actual finish position on the target race.
     If |rho| ~ 0, the feature doesn't predict finishes at all.
"""
import numpy as np
import pandas as pd

from src.features.build_features import build_features


def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    f = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    # Only rows with actual race outcomes.
    fx = f[f["finish_pos"] > 0].copy()

    print("=" * 70)
    print("1) COVERAGE — non-NaN fraction of tire_decay_type_10 by track_type")
    print("=" * 70)
    cov = fx.groupby("track_type").apply(
        lambda g: pd.Series({
            "n_rows": len(g),
            "non_nan_pct": g["tire_decay_type_10"].notna().mean() * 100,
            "mean_races_type": g["tire_races_type"].mean(),
            "median_races_type": g["tire_races_type"].median(),
        })
    )
    print(cov.round(2).to_string())

    print("\n" + "=" * 70)
    print("2) DISTRIBUTION of tire_decay_type_10 by track_type (non-NaN only)")
    print("=" * 70)
    dist = fx.dropna(subset=["tire_decay_type_10"]).groupby("track_type")[
        "tire_decay_type_10"
    ].describe()
    print(dist.round(4).to_string())

    print("\n" + "=" * 70)
    print("3) CORRELATION of tire_decay_type_10 with existing rolling features")
    print("=" * 70)
    corrs = fx[[
        "tire_decay_type_10", "avg_finish_5", "avg_finish_10", "avg_finish_20",
        "pl_effective", "avg_finish_at_type_last_10",
    ]].corr(numeric_only=True)["tire_decay_type_10"].drop("tire_decay_type_10")
    print(corrs.round(3).to_string())
    print("\n(High positive corr with avg_finish = we're re-encoding "
          "finish position. High negative corr with pl_effective = we're "
          "re-encoding driver quality.)")

    print("\n" + "=" * 70)
    print("4) PREDICTIVE POWER — corr of tire_decay_type_10 with actual finish_pos")
    print("=" * 70)
    for tt, sub in fx.dropna(subset=["tire_decay_type_10"]).groupby("track_type"):
        rho = sub[["tire_decay_type_10", "finish_pos"]].corr().iloc[0, 1]
        print(f"  {tt:15s}  n={len(sub):5d}  rho(tire_decay, finish_pos)={rho:+.3f}")
    print("\n(Positive rho = higher decay -> worse finish, as expected. "
          "Near-zero rho = feature is noise for that track type.)")


if __name__ == "__main__":
    main()
