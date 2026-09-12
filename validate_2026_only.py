"""Walk-forward with 2026-only validation split.

Question: our internal walk-forward reports matchup log-loss ~0.618, but the
OddsLogic backtest on real 2026 lines shows 0.79+. If we compute the same
internal metric on 2026 races only (using all possible driver pairs, weighted
by market-like variance), do we see the same gap?

If YES: the model actually is worse on 2026 and our full-history walk-forward
average has been hiding it. The 0.618 headline was a per-year mix.
If NO: OddsLogic-line drivers are a special subset (top-half) where the model
is worst; the internal-average is honest but not representative.

Output: per-year matchup log-loss (all-vs-all pairs) so we can see the trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import TIGHT_REG, ALPHA, N_SAMPLES, HAZARD


def compute_all_pair_ll(matchup_mtx: np.ndarray, finishes: np.ndarray) -> tuple[float, int]:
    """For every pair (i,j) with i<j, compute -log P(actual winner beats loser).
    Skip pairs where both DNF-tied. Returns (mean_ll, n)."""
    n = len(finishes)
    lls = []
    for i in range(n):
        for j in range(i + 1, n):
            if finishes[i] == 0 or finishes[j] == 0: continue
            if finishes[i] == finishes[j]: continue
            i_won = finishes[i] < finishes[j]
            p = matchup_mtx[i, j] if i_won else 1 - matchup_mtx[i, j]
            lls.append(-float(np.log(np.clip(p, 1e-9, 1 - 1e-9))))
    return (float(np.mean(lls)) if lls else float("nan"), len(lls))


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])
    features["season"] = features["date"].dt.year

    # Walk forward one race at a time. For efficiency, retrain every 5 races
    # from race 30 onward, holding model fixed between retrains.
    all_race_ids = features["race_id_short"].drop_duplicates().tolist()
    race_dates = features.groupby("race_id_short")["date"].first().sort_values()
    ordered_ids = race_dates.index.tolist()

    results = []
    model: GBMEnsemble | None = None
    last_train_size = 0
    T_by_type: dict = {}

    for i, rid in enumerate(ordered_ids):
        if i < 30: continue  # need training pool

        target = features[features["race_id_short"] == rid].reset_index(drop=True)
        if target.empty or (target["finish_pos"] > 0).sum() == 0:
            continue
        tt = target["track_type"].iloc[0]
        season = int(target["season"].iloc[0])
        date = target["date"].iloc[0]

        # Retrain every 5 races or first time.
        train = features[(features["date"] < date) & (features["finish_pos"] > 0)]
        if model is None or (len(train) - last_train_size) >= 5 * 40:
            model = GBMEnsemble()
            model.fit(train, n_estimators=5, **TIGHT_REG)  # 5 boosters for speed
            last_train_size = len(train)

            # Recalibrate on last ~30 training races.
            recent_ids = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()[-30:]
            val_races, tt_list = [], []
            for vrid in recent_ids:
                sub = train[train["race_id_short"] == vrid]
                if len(sub) < 5: continue
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
            if val_races:
                T_by_type = find_best_temperature_by_type(val_races, tt_list, default_T=1.0, n_samples=800)

        T = T_by_type.get(tt, 1.0)
        raw = model.predict_scores(target)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
                / (target["pl_effective"].std() + 1e-9))
        blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
        blended = blended / max(blended.std(), 1e-6)
        haz = np.full(len(target), HAZARD.get(tt, 0.08))
        rng = np.random.default_rng(42)
        positions = dist.sample_finishing_orders(blended / T, haz, n_samples=8000, rng=rng)
        matchup_mtx = dist.matchup_probs(positions)

        ll, n_pairs = compute_all_pair_ll(matchup_mtx, target["finish_pos"].to_numpy())
        results.append({"race_id": rid, "date": date, "season": season, "tt": tt,
                        "T": T, "n_pairs": n_pairs, "matchup_ll": ll})

    df = pd.DataFrame(results)

    print("\n=== Matchup log-loss by season (all-pairs, walk-forward) ===")
    by_season = df.groupby("season").apply(
        lambda g: pd.Series({
            "races": len(g),
            "pairs": g["n_pairs"].sum(),
            "matchup_ll": (g["matchup_ll"] * g["n_pairs"]).sum() / g["n_pairs"].sum(),
        })
    )
    print(by_season.round(4).to_string())

    print("\n=== Matchup log-loss by (season, track_type) ===")
    by_st = df.groupby(["season", "tt"]).apply(
        lambda g: pd.Series({
            "races": len(g),
            "pairs": g["n_pairs"].sum(),
            "matchup_ll": (g["matchup_ll"] * g["n_pairs"]).sum() / g["n_pairs"].sum(),
        })
    )
    print(by_st.round(4).to_string())

    print("\n=== Overall (all races) ===")
    total_pairs = df["n_pairs"].sum()
    total_ll = (df["matchup_ll"] * df["n_pairs"]).sum() / total_pairs
    print(f"Races: {len(df)}   Pairs: {total_pairs}   Weighted matchup LL: {total_ll:.4f}")

    print("\n=== 2026-only per-race ===")
    d26 = df[df["season"] == 2026].sort_values("date")
    if not d26.empty:
        print(d26[["date", "tt", "T", "n_pairs", "matchup_ll"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
