# apply_validation_and_run.py — no new files to write; just runs three overfitting checks
# with the tight-reg config that won the sweep.
import numpy as np, pandas as pd
from src.eval.backtest_gbm_walkforward import backtest_gbm_walkforward, summarize_wf

TIGHT_REG = {"num_leaves": 31, "min_data_in_leaf": 25, "lambda_l2": 2.0}
ALPHA = 0.85

races = pd.read_parquet("data/processed/races.parquet").sort_values("date").reset_index(drop=True)
entries = pd.read_parquet("data/processed/entries.parquet")
sessions = pd.read_parquet("data/processed/sessions.parquet")
loopstats = pd.read_parquet("data/processed/loopstats.parquet")

# =========================================================================
# C. Per-season breakdown of the walk-forward run (does 2024 hit ~= 2026?)
# =========================================================================
print("=" * 70)
print("C. PER-SEASON BREAKDOWN (walk-forward, tight-reg, alpha=0.85)")
print("=" * 70)
out = backtest_gbm_walkforward(
    races, entries, sessions, loopstats,
    initial_train_races=36, retrain_every_n=10,
    calib_window=25, recalibrate_every_n=10,
    ensemble_alpha=ALPHA, n_samples=3000,
    gbm_fit_kwargs=TIGHT_REG,
)
df = out["metrics"].copy()
df["season"] = pd.to_datetime(df["date"]).dt.year
by_season = df.groupby("season").agg(
    n_races=("race_id", "count"),
    matchup_ll=("matchup_logloss", "mean"),
    winner_ll=("winner_logloss", "mean"),
    spearman=("spearman_rho", "mean"),
    top5=("top5_accuracy", "mean"),
).round(4)
print(by_season.to_string())

# =========================================================================
# B. Bootstrap 95% CI on matchup log-loss over the full walk-forward test set
# =========================================================================
print()
print("=" * 70)
print("B. BOOTSTRAP CI on walk-forward matchup LL (1000 resamples)")
print("=" * 70)
rng = np.random.default_rng(0)
losses = df["matchup_logloss"].to_numpy()
n = len(losses)
boot = np.array([rng.choice(losses, size=n, replace=True).mean() for _ in range(1000)])
print(f"n_test_races: {n}")
print(f"point estimate: {losses.mean():.4f}")
print(f"95% CI:         [{np.quantile(boot, 0.025):.4f}, {np.quantile(boot, 0.975):.4f}]")
print(f"50% CI:         [{np.quantile(boot, 0.25):.4f}, {np.quantile(boot, 0.75):.4f}]")
frac_below_640 = float(np.mean(boot < 0.64))
frac_below_620 = float(np.mean(boot < 0.62))
print(f"P(true LL < 0.640) ~ {frac_below_640:.2%}")
print(f"P(true LL < 0.620) ~ {frac_below_620:.2%}")

# =========================================================================
# A. TRUE HOLDOUT: initial train covers all races prior to 2026-01-01,
#     never retrain, test only on 2026 races.
# =========================================================================
print()
print("=" * 70)
print("A. TRUE HOLDOUT: train on 2022-2025 (frozen), evaluate on 2026 only")
print("=" * 70)
races["date"] = pd.to_datetime(races["date"])
n_pre_2026 = int((races["date"] < pd.Timestamp("2026-01-01")).sum())
print(f"Initial train window: {n_pre_2026} races")
out2 = backtest_gbm_walkforward(
    races, entries, sessions, loopstats,
    initial_train_races=n_pre_2026,
    retrain_every_n=10_000,           # effectively never retrain
    calib_window=25, recalibrate_every_n=10,
    ensemble_alpha=ALPHA, n_samples=3000,
    gbm_fit_kwargs=TIGHT_REG,
)
df2 = out2["metrics"].copy()
df2["season"] = pd.to_datetime(df2["date"]).dt.year
print(f"Test races (should be 2026 only): {len(df2)}, seasons: {sorted(df2['season'].unique())}")
holdout_row = df2[["winner_logloss","matchup_logloss","spearman_rho","top5_accuracy"]].mean()
print("2026-only holdout metrics:")
print(holdout_row.round(4).to_string())

wf_2026 = df[df["season"] == 2026][["winner_logloss","matchup_logloss","spearman_rho","top5_accuracy"]].mean()
print("\nSame 2026 races under WALK-FORWARD (with more recent retrains):")
print(wf_2026.round(4).to_string())
print("\nDelta (holdout - walkforward):")
print((holdout_row - wf_2026).round(4).to_string())

print("\nDone.")
