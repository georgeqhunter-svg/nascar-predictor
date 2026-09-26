"""Was the model unusually bad at Atlanta BEFORE 2026? (no betting lines needed)

Leak-free block walk-forward:
  - train on all races before 2024-01-01 -> score every 2024 race
  - train on all races before 2025-01-01 -> score every 2025 race

For each race, compute pairwise ranking accuracy: over all driver pairs, the
fraction where the model's score order matches actual finish order. Compared
with two simple baselines scored on the same races:
  - start_pos order
  - prior 10-race average finish order
"model edge" = model accuracy minus the better baseline. If Atlanta shows a
much lower model edge than other track groups, that's independent (pre-2026)
evidence of an Atlanta-specific flaw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import TIGHT_REG, ALPHA


def pairwise_acc(score_higher_is_better: np.ndarray, finish: np.ndarray) -> float:
    s = np.asarray(score_higher_is_better, float)
    f = np.asarray(finish, float)
    ok = np.isfinite(s) & np.isfinite(f) & (f > 0)
    s, f = s[ok], f[ok]
    n = len(s)
    if n < 10:
        return np.nan
    ds = s[:, None] - s[None, :]
    df = f[None, :] - f[:, None]  # >0 when i finished ahead of j
    mask = np.triu(np.ones((n, n), bool), 1) & (ds != 0) & (df != 0)
    return float(((ds > 0) == (df > 0))[mask].mean())


def main():
    from src.features.build_features import build_features
    from src.features.tracks import resolve_track_type
    from src.models.gbm_ranker import GBMEnsemble

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...", flush=True)
    feats = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    feats["date"] = pd.to_datetime(feats["date"])

    races["grp"] = races.apply(
        lambda r: "ATLANTA" if "Atlanta" in str(r["track_name"])
        else ("DAY/TAL" if any(k in str(r["track_name"]) for k in ("Daytona", "Talladega"))
              else resolve_track_type(r["track_name"], r["track_type"])), axis=1)
    grp = races.set_index("race_id_short")["grp"].to_dict()

    rows = []
    for season in (2024, 2025):
        cutoff = pd.Timestamp(f"{season}-01-01")
        train = feats[(feats["date"] < cutoff) & (feats["finish_pos"] > 0)]
        test = feats[(feats["season"] == season) & (feats["finish_pos"] > 0)]
        print(f"Season {season}: training on {train['race_id_short'].nunique()} races...", flush=True)
        m = GBMEnsemble()
        m.fit(train, n_estimators=5, **TIGHT_REG)
        for rid, sub in test.groupby("race_id_short"):
            raw = m.predict_scores(sub)
            gz = (raw - raw.mean()) / (raw.std() + 1e-9)
            pl = sub["pl_effective"].to_numpy(float)
            pz = (pl - pl.mean()) / (pl.std() + 1e-9)
            score = ALPHA * gz + (1 - ALPHA) * pz
            fin = sub["finish_pos"].to_numpy(float)
            rows.append({
                "race": rid, "grp": grp.get(rid, "?"),
                "model": pairwise_acc(score, fin),
                "start": pairwise_acc(-sub["start_pos"].to_numpy(float), fin),
                "form": pairwise_acc(-sub["avg_finish_10"].to_numpy(float), fin),
            })

    df = pd.DataFrame(rows)
    df["best_baseline"] = df[["start", "form"]].max(axis=1)
    df["model_edge"] = df["model"] - df["best_baseline"]
    print()
    print("Pairwise ranking accuracy, 2024-2025 races (0.50 = coin flip)")
    print("=" * 78)
    summ = df.groupby("grp").agg(
        races=("race", "count"),
        model=("model", "mean"), start=("start", "mean"), form=("form", "mean"),
        model_edge=("model_edge", "mean"))
    print(summ.round(3).to_string())
    print()
    print("Atlanta races individually:")
    print(df[df["grp"] == "ATLANTA"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
