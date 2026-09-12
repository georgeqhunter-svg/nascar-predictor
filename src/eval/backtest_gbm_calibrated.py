"""GBM backtest with a train / val / test split and post-hoc calibration.

Workflow:
    1. Train GBM on races before `train_end`.
    2. Score val races (train_end <= date < val_end); grid-search temperature T.
    3. Score test races (>= val_end) applying T. Report metrics.
    4. Optionally fit isotonic on winner probs from val, apply to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..features.build_features import build_features
from ..models import distribution as dist
from ..models.calibrate import (
    RaceScoresGT,
    apply_winner_isotonic,
    find_best_temperature,
    fit_isotonic_winner,
)
from ..models.gbm_ranker import GBMRanker


HAZARD_DEFAULT = {
    "superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
    "road": 0.04, "unique": 0.07,
}


def _log_clip(p: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, floor, 1.0 - floor))


def _prep_race_gt(sub: pd.DataFrame, scores: np.ndarray, hazard_by_type: dict) -> RaceScoresGT:
    tt = sub["track_type"].iloc[0]
    return RaceScoresGT(
        scores=scores,
        finishes=sub["finish_pos"].to_numpy(),
        is_dnf=sub["is_dnf"].to_numpy(),
        hazards=np.full(len(sub), hazard_by_type.get(tt, 0.08)),
    )


def _score_race(model: GBMRanker, sub: pd.DataFrame) -> np.ndarray:
    s = model.predict_scores(sub)
    return s - s.mean()  # shift-invariant; center for numerical stability


def _race_metrics(scores: np.ndarray, race_gt: RaceScoresGT, T: float,
                  iso_fn=None, n_samples: int = 3000, rng=None) -> dict:
    strengths = scores / T
    positions = dist.sample_finishing_orders(strengths, race_gt.hazards,
                                             n_samples=n_samples, rng=rng)
    winp = dist.win_probs(positions)
    if iso_fn is not None:
        winp = apply_winner_isotonic(winp, iso_fn)
    M = dist.matchup_probs(positions)

    winner_idx = int(np.argmin(race_gt.finishes))
    winner_ll = -float(_log_clip(np.array([winp[winner_idx]]))[0])

    n = len(race_gt.finishes)
    pair_ll = 0.0
    n_pairs = 0
    for a in range(n):
        if race_gt.is_dnf[a]:
            continue
        for b in range(a + 1, n):
            if race_gt.is_dnf[b]:
                continue
            a_ahead = race_gt.finishes[a] < race_gt.finishes[b]
            p = M[a, b] if a_ahead else M[b, a]
            pair_ll += -float(_log_clip(np.array([p]))[0])
            n_pairs += 1
    matchup_ll = pair_ll / max(n_pairs, 1)

    exp_finish = positions.mean(axis=0)
    rho = spearmanr(exp_finish, race_gt.finishes).statistic
    rho = 0.0 if np.isnan(rho) else float(rho)
    top5_pred = np.argsort(-winp)[:5]
    actual_top5 = set(np.where(race_gt.finishes <= 5)[0])
    top5_acc = len(set(top5_pred.tolist()) & actual_top5) / 5.0

    return {
        "winner_logloss": winner_ll,
        "matchup_logloss": matchup_ll,
        "spearman_rho": rho,
        "top5_accuracy": top5_acc,
    }


def backtest_gbm_calibrated(
    races: pd.DataFrame,
    entries: pd.DataFrame,
    sessions: pd.DataFrame | None,
    train_end: str = "2023-01-01",
    val_end: str = "2024-01-01",
    n_samples: int = 3000,
    hazard_by_type: dict[str, float] | None = None,
    T_grid: np.ndarray | None = None,
    loopstats: pd.DataFrame | None = None,
) -> dict:
    """Full pipeline. Returns dict with 'best_T', 'T_grid_losses', 'test_metrics',
    'test_metrics_iso', 'model'."""
    if hazard_by_type is None:
        hazard_by_type = HAZARD_DEFAULT

    features = build_features(races, entries, sessions, loopstats=loopstats)
    features["date"] = pd.to_datetime(features["date"])
    train = features[features["date"] < pd.Timestamp(train_end)]
    val = features[(features["date"] >= pd.Timestamp(train_end)) &
                   (features["date"] < pd.Timestamp(val_end))]
    test = features[features["date"] >= pd.Timestamp(val_end)]

    print(f"train races: {train['race_id_short'].nunique()}  "
          f"val races: {val['race_id_short'].nunique()}  "
          f"test races: {test['race_id_short'].nunique()}")

    model = GBMRanker()
    model.fit(train)

    # --- 1. score val races and grid-search T --- #
    val_gt = []
    for _, sub in val.groupby("race_id_short", sort=False):
        val_gt.append(_prep_race_gt(sub, _score_race(model, sub), hazard_by_type))
    best_T, T_table = find_best_temperature(val_gt, T_grid=T_grid, n_samples=n_samples)
    print(f"\nTemperature grid (val loss):")
    print(T_table.round(4).to_string(index=False))
    print(f"\nBest T: {best_T}")

    # --- 2. fit isotonic on val winner probs (using best_T) --- #
    rng = np.random.default_rng(0)
    val_win_probs = []
    val_win_labels = []
    for r in val_gt:
        positions = dist.sample_finishing_orders(r.scores / best_T, r.hazards,
                                                 n_samples=n_samples, rng=rng)
        wp = dist.win_probs(positions)
        val_win_probs.extend(wp.tolist())
        was_win = (r.finishes == 1).astype(int)
        val_win_labels.extend(was_win.tolist())
    iso_fn = fit_isotonic_winner(np.array(val_win_probs), np.array(val_win_labels))

    # --- 3. evaluate on test --- #
    rng = np.random.default_rng(1)
    raw_rows = []
    iso_rows = []
    for race_id, sub in test.groupby("race_id_short", sort=False):
        scores = _score_race(model, sub)
        gt = _prep_race_gt(sub, scores, hazard_by_type)
        raw = _race_metrics(scores, gt, T=best_T, iso_fn=None, n_samples=n_samples, rng=rng)
        iso = _race_metrics(scores, gt, T=best_T, iso_fn=iso_fn, n_samples=n_samples, rng=rng)
        base = {"race_id": race_id, "date": sub["date"].iloc[0],
                "track_type": sub["track_type"].iloc[0], "n_drivers": len(sub)}
        raw_rows.append({**base, **raw})
        iso_rows.append({**base, **iso})

    return {
        "best_T": best_T,
        "T_grid_losses": T_table,
        "test_metrics": pd.DataFrame(raw_rows),
        "test_metrics_iso": pd.DataFrame(iso_rows),
        "model": model,
    }


def summarize_test(df: pd.DataFrame) -> pd.DataFrame:
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
    total_row = pd.DataFrame([{"n_races": len(df), **total.to_dict()}], index=["ALL"])
    total_row.index.name = "track_type"
    return pd.concat([grouped, total_row])
