"""GBM backtest with time-based train/eval split.

Train the ranker on all races up to `split_date`, then predict every race
after. For each test race we convert LightGBM scores into a finishing
distribution via the same Gumbel-max sampler used for the PL baseline, so the
downstream metrics (winner LL, matchup LL, Spearman, top-5) are apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..features.build_features import build_features
from ..models import distribution as dist
from ..models.gbm_ranker import GBMRanker


HAZARD_DEFAULT = {
    "superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
    "road": 0.04, "unique": 0.07,
}


def _log_clip(p: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, floor, 1.0 - floor))


def backtest_gbm(
    races: pd.DataFrame,
    entries: pd.DataFrame,
    sessions: pd.DataFrame | None = None,
    split_date: str | pd.Timestamp = "2024-01-01",
    n_samples: int = 3000,
    hazard_by_type: dict[str, float] | None = None,
    score_scale: float = 1.0,
) -> tuple[GBMRanker, pd.DataFrame, pd.DataFrame]:
    """Train GBM on races before `split_date`, evaluate on races after.

    Returns (model, per_race_metrics, feature_matrix).
    """
    if hazard_by_type is None:
        hazard_by_type = HAZARD_DEFAULT

    features = build_features(races, entries, sessions)
    features["date"] = pd.to_datetime(features["date"])
    split = pd.Timestamp(split_date)

    train_feat = features[features["date"] < split].copy()
    test_feat = features[features["date"] >= split].copy()

    if train_feat.empty or test_feat.empty:
        raise RuntimeError(f"empty split — train={len(train_feat)}, test={len(test_feat)}")

    model = GBMRanker()
    model.fit(train_feat)

    rng = np.random.default_rng(0)
    results = []
    for race_id, e in test_feat.groupby("race_id_short", sort=False):
        e = e.copy()
        scores = model.predict_scores(e)
        # Center (Gumbel-max is shift-invariant) but preserve the model's own
        # spread. A scalar multiplier lets us calibrate sharpness vs. noise.
        scores = score_scale * (scores - scores.mean())

        tt = e["track_type"].iloc[0]
        hazards = np.full(len(e), hazard_by_type.get(tt, 0.08))
        finishes = e["finish_pos"].to_numpy()
        is_dnf = e["is_dnf"].to_numpy()

        positions = dist.sample_finishing_orders(scores, hazards, n_samples=n_samples, rng=rng)
        winp = dist.win_probs(positions)
        M = dist.matchup_probs(positions)

        winner_idx = int(np.argmin(finishes))
        winner_ll = -float(_log_clip(np.array([winp[winner_idx]]))[0])

        n = len(e)
        pair_ll = 0.0
        n_pairs = 0
        for a in range(n):
            if is_dnf[a]:
                continue
            for b in range(a + 1, n):
                if is_dnf[b]:
                    continue
                a_ahead = finishes[a] < finishes[b]
                p = M[a, b] if a_ahead else M[b, a]
                pair_ll += -float(_log_clip(np.array([p]))[0])
                n_pairs += 1
        matchup_ll = pair_ll / max(n_pairs, 1)

        exp_finish = positions.mean(axis=0)
        rho = spearmanr(exp_finish, finishes).statistic
        rho = 0.0 if np.isnan(rho) else float(rho)

        top5_pred = np.argsort(-winp)[:5]
        actual_top5 = set(np.where(finishes <= 5)[0])
        top5_acc = len(set(top5_pred.tolist()) & actual_top5) / 5.0

        results.append({
            "race_id": race_id,
            "date": e["date"].iloc[0],
            "track_type": tt,
            "n_drivers": n,
            "winner_logloss": winner_ll,
            "matchup_logloss": matchup_ll,
            "spearman_rho": rho,
            "top5_accuracy": top5_acc,
        })

    return model, pd.DataFrame(results), features


def summarize_gbm(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("track_type").agg(
        n_races=("race_id", "count"),
        winner_logloss=("winner_logloss", "mean"),
        matchup_logloss=("matchup_logloss", "mean"),
        spearman_rho=("spearman_rho", "mean"),
        top5_accuracy=("top5_accuracy", "mean"),
    )
    total = df.agg({
        "winner_logloss": "mean",
        "matchup_logloss": "mean",
        "spearman_rho": "mean",
        "top5_accuracy": "mean",
    })
    total_row = pd.DataFrame(
        [{"n_races": len(df), **total.to_dict()}], index=["ALL"],
    )
    total_row.index.name = "track_type"
    return pd.concat([grouped, total_row])
