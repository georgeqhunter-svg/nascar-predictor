"""Diagnose isotonic calibration for matchup probs."""
import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import TIGHT_REG, ALPHA, N_SAMPLES, HAZARD

from src.features.build_features import build_features
from src.models import distribution as dist
from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
from src.models.gbm_ranker import GBMEnsemble
from src.models.product_calibration import fit_product_calibrators

races = pd.read_parquet("data/processed/races.parquet")
entries = pd.read_parquet("data/processed/entries.parquet")
sessions = pd.read_parquet("data/processed/sessions.parquet")
loopstats = pd.read_parquet("data/processed/loopstats.parquet")
laptimes = pd.read_parquet("data/processed/laptimes.parquet")
races["date"] = pd.to_datetime(races["date"])
entries["date"] = pd.to_datetime(entries["date"])

print("Building features...")
features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
features["date"] = pd.to_datetime(features["date"])

# Target: COTA
target_ts = pd.Timestamp("2026-03-01")
r = races[(races["date"] == target_ts) & (races["season"] == 2026)]
rid = r.iloc[0]["race_id_short"]
target_date_ts = r.iloc[0]["date"]
tt = r.iloc[0]["track_type"]

train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
target = features[features["race_id_short"] == rid].reset_index(drop=True)

model = GBMEnsemble()
model.fit(train, n_estimators=15, **TIGHT_REG)

# Score temp calibration
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
T_by_type = find_best_temperature_by_type(val_races, tt_list, default_T=1.0, n_samples=1500)
T = T_by_type.get(tt, 1.0)

print(f"T={T}")
print("Fitting calibrators...")
calibrators = fit_product_calibrators(val_races, T, n_samples=5000)
c_match = calibrators["matchup"]
print(f"Matchup calibrator fitted: {c_match.fitted}")

# Try transforming a range of values
probes = np.array([0.05, 0.15, 0.30, 0.45, 0.50, 0.55, 0.70, 0.85, 0.95])
mapped = c_match.transform(probes)
print("\nProbes → mapped:")
for p, m in zip(probes, mapped):
    print(f"  {p:.2f} → {m:.4f}")

# Sample target race and compute what happens to matchup probs
raw = model.predict_scores(target)
gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
        / (target["pl_effective"].std() + 1e-9))
blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
blended = blended / max(blended.std(), 1e-6)
haz = np.full(len(target), HAZARD.get(tt, 0.08))
rng = np.random.default_rng(42)
positions = dist.sample_finishing_orders(blended / T, haz, n_samples=N_SAMPLES, rng=rng)
matchup_mtx = dist.matchup_probs(positions)

print(f"\nBefore isotonic:")
print(f"  matchup_mtx shape: {matchup_mtx.shape}")
print(f"  min: {matchup_mtx.min():.4f}, max: {matchup_mtx.max():.4f}")
print(f"  sample values (upper triangle): {[float(matchup_mtx[0, j]) for j in range(1, 6)]}")

matchup_cal = c_match.transform(matchup_mtx.flatten()).reshape(matchup_mtx.shape)
print(f"\nAfter isotonic:")
print(f"  min: {matchup_cal.min():.4f}, max: {matchup_cal.max():.4f}")
print(f"  sample values (same positions): {[float(matchup_cal[0, j]) for j in range(1, 6)]}")

# Check symmetry: does p[i,j] + p[j,i] = 1?
p_ij = matchup_cal[2, 5]
p_ji = matchup_cal[5, 2]
print(f"\nSymmetry check: p[2,5]={p_ij:.4f}, p[5,2]={p_ji:.4f}, sum={p_ij+p_ji:.4f}")
