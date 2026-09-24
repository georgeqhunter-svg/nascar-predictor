"""Rank features by LightGBM importance across the whole ensemble.

Trains the committed model on all historical data and dumps every feature's
importance in two flavors:
  - split_count: number of times each feature was used as a split point.
                 Bottom of this list = features the model never touches.
  - gain: cumulative gain from all splits on this feature. Bottom of this
          list = features that split but contribute nothing to log-loss.

We flag suspect features (known-noisy candidates + very-low-importance ones)
so you have a "consider dropping" shortlist for ablation testing.

Runs in ~30 seconds (fits the primary model, no CV loop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import TIGHT_REG, ALPHA


# Features I flagged as likely-noisy or redundant on inspection:
SUSPECT_FEATURES = {
    "best_finish_at_track": "single career extreme, noisy vs avg_finish",
    "best_finish_at_track_last_10": "single window extreme, bounded but still noisy",
    "season_wins_ytd": "one lucky race +1 all season; poor generalization",
    "practice_laps_z": "teams strategically skip practice; noisy signal",
    "has_practice_data": "data-availability flag, may split on track identity",
    "practice_5lap_avg_z": "possibly redundant with practice_best_speed_z",
    "practice_10lap_avg_z": "possibly redundant with practice_best_speed_z",
    "practice_consistency_z": "narrow window, may correlate with laps run",
    "practice_laps_run_z": "not necessarily performance-linked",
    "team_pit_n_stops_10": "guard/sample-size feature, may not add signal",
    "lr_prior_seasons_used": "count of LR seasons; likely dominated by other LR features",
    "manuf_finish_std_at_type_10": "variance signal may be noisy at low n",
    "wx_hot_races": "count feature, may not matter given rolling avgs",
    "wx_cool_races": "count feature, may not matter given rolling avgs",
    "wx_windy_races": "count feature, may not matter given rolling avgs",
    "wx_wet_races": "count feature, may not matter given rolling avgs",
    "drv_manuf_type_races_10": "count feature, may not add over drv_manuf_type_avg",
    "races_at_track_last_10": "count feature",
    "races_at_type_last_10": "count feature",
    "team_races_at_type": "count feature",
    "restart_count_10": "count feature",
    "loop_quality_passes_5": "possibly redundant with loop_quality_passes_10",
    "loop_top15_laps_10": "narrow segmentation of running position",
    "pit_gain_std_10": "variance signal, may be noisy",
    "restart_gain_std_10": "variance signal, may be noisy",
    "team_pit_gain_std_5": "variance signal, may be noisy",
    "team_pit_gain_std_10": "variance signal, may be noisy",
}


def main():
    from src.features.build_features import build_features
    from src.models.gbm_ranker import GBMEnsemble, FEATURES, CATEGORICAL

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...")
    features = build_features(races, entries, sessions,
                              loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    train = features[features["finish_pos"] > 0].copy()
    print(f"Training on {len(train)} driver-races.")

    model = GBMEnsemble()
    model.fit(train, n_estimators=5, **TIGHT_REG)  # 5 boosters, fast

    # Aggregate importance across all boosters.
    all_features = FEATURES + CATEGORICAL
    splits = np.zeros(len(all_features), dtype=float)
    gains = np.zeros(len(all_features), dtype=float)
    for booster in model.boosters:
        booster_feats = booster.feature_name()
        s = booster.feature_importance(importance_type="split")
        g = booster.feature_importance(importance_type="gain")
        # Map by feature name (order can vary across boosters).
        for i, fname in enumerate(booster_feats):
            if fname in all_features:
                idx = all_features.index(fname)
                splits[idx] += s[i]
                gains[idx] += g[i]
    n_boosters = len(model.boosters)
    splits /= n_boosters
    gains /= n_boosters

    df = pd.DataFrame({
        "feature": all_features,
        "avg_splits": splits,
        "avg_gain": gains,
    })
    df["gain_rank"] = df["avg_gain"].rank(method="min", ascending=False).astype(int)
    df["split_rank"] = df["avg_splits"].rank(method="min", ascending=False).astype(int)
    df["suspect"] = df["feature"].apply(lambda f: "*" if f in SUSPECT_FEATURES else "")

    # -- Top by gain (most valuable features) --
    print("\n" + "=" * 88)
    print("TOP 20 FEATURES BY GAIN")
    print("=" * 88)
    top = df.sort_values("avg_gain", ascending=False).head(20)
    print(top[["feature", "avg_splits", "avg_gain", "gain_rank"]].to_string(
        index=False, formatters={"avg_splits": "{:.1f}".format,
                                 "avg_gain": "{:.1f}".format}))

    # -- Bottom by gain (candidates to drop) --
    print("\n" + "=" * 88)
    print("BOTTOM 30 FEATURES BY GAIN — candidates to drop (starred = pre-flagged suspect)")
    print("=" * 88)
    bot = df.sort_values("avg_gain").head(30)
    print(bot[["feature", "suspect", "avg_splits", "avg_gain", "split_rank"]].to_string(
        index=False, formatters={"avg_splits": "{:.1f}".format,
                                 "avg_gain": "{:.1f}".format}))

    # -- Suspect features specifically --
    print("\n" + "=" * 88)
    print("PRE-FLAGGED SUSPECT FEATURES — where they rank")
    print("=" * 88)
    suspects = df[df["feature"].isin(SUSPECT_FEATURES)].sort_values("gain_rank")
    total = len(all_features)
    for _, row in suspects.iterrows():
        reason = SUSPECT_FEATURES[row["feature"]]
        pct = 100 * (total - row["gain_rank"] + 1) / total
        print(f"  {row['feature']:<36} rank {row['gain_rank']:>3}/{total} "
              f"(top {pct:>4.1f}%)  gain {row['avg_gain']:>7.1f}   {reason}")

    # -- Never-split features (zero splits across all boosters) --
    dead = df[df["avg_splits"] == 0]
    if len(dead):
        print("\n" + "=" * 88)
        print(f"DEAD FEATURES ({len(dead)} — never used as a split in any tree)")
        print("=" * 88)
        for f in dead["feature"].tolist():
            print(f"  {f}")


if __name__ == "__main__":
    main()
