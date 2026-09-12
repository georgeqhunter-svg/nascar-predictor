"""Walk-forward GBM backtest with periodic retraining, rolling calibration,
and optional PL ensemble.

Improvements over backtest_gbm_calibrated:
  * GBM is retrained every N races on all available history, so training data
    grows through the test period instead of being fixed at 36 races.
  * Temperature T is recalibrated on a trailing window of predictions.
  * `ensemble_alpha` (0..1) blends GBM strengths with PL strengths per race:
    final = alpha * z(gbm) + (1 - alpha) * z(pl_effective).
    PL is naturally calibrated on matchups, so this usually helps matchup LL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..features.build_features import build_features
from ..models import distribution as dist
from ..models.calibrate import RaceScoresGT, find_best_temperature, find_best_temperature_by_type
from ..models.gbm_ranker import GBMEnsemble, GBMRanker

log = logging.getLogger(__name__)


HAZARD_DEFAULT = {
    "superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
    "road": 0.04, "unique": 0.07,
}


def _log_clip(p: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, floor, 1.0 - floor))


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def _race_metrics(strengths, hazards, finishes, is_dnf, n_samples, rng):
    positions = dist.sample_finishing_orders(strengths, hazards, n_samples=n_samples, rng=rng)
    winp = dist.win_probs(positions)
    M = dist.matchup_probs(positions)

    winner_idx = int(np.argmin(finishes))
    winner_ll = -float(_log_clip(np.array([winp[winner_idx]]))[0])

    n = len(finishes)
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

    return dict(winner_logloss=winner_ll, matchup_logloss=matchup_ll,
                spearman_rho=rho, top5_accuracy=top5_acc)


def backtest_gbm_walkforward(
    races: pd.DataFrame,
    entries: pd.DataFrame,
    sessions: pd.DataFrame | None,
    loopstats: pd.DataFrame | None,
    initial_train_races: int = 36,
    retrain_every_n: int = 5,
    calib_window: int = 20,
    recalibrate_every_n: int = 5,      # T search is expensive; do it periodically
    ensemble_alpha: float = 1.0,       # 1.0 = pure GBM, 0.0 = pure PL
    n_samples: int = 3000,
    T_grid: np.ndarray | None = None,
    per_type_T: bool = False,          # if True, calibrate T separately per track_type
    drop_features: list[str] | None = None,
    gbm_fit_kwargs: dict | None = None,
    per_driver_dnf: bool = False,      # scale per-driver DNF hazard by dnf_rate_10
    dnf_scale_range: tuple[float, float] = (0.4, 2.5),  # clip factor
    laptimes: pd.DataFrame | None = None,
    n_ensemble: int = 1,               # >1 = train N GBMs with different seeds, average
) -> dict:
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    race_order = (
        features.groupby("race_id_short", sort=False)["date"].first().sort_values().index.tolist()
    )
    if len(race_order) <= initial_train_races + 5:
        raise RuntimeError("not enough races for walk-forward")

    per_race_dfs = {r: features[features["race_id_short"] == r] for r in race_order}

    # Initial train on the first `initial_train_races` races.
    train_races = race_order[:initial_train_races]
    train_data = pd.concat([per_race_dfs[r] for r in train_races], ignore_index=True)
    _fit_kwargs = gbm_fit_kwargs or {}
    def _new_model():
        if n_ensemble > 1:
            m = GBMEnsemble(drop_features=drop_features)
            m.fit(train_data if False else train_data,
                  n_estimators=n_ensemble, **_fit_kwargs)
            return m
        m = GBMRanker(drop_features=drop_features)
        m.fit(train_data, **_fit_kwargs)
        return m
    model = _new_model()
    log.info("initial fit on %s races (%s rows)", len(train_races), len(train_data))

    T = 3.0
    T_by_type: dict[str, float] = {}
    rng = np.random.default_rng(0)
    predictions_buffer: list[RaceScoresGT] = []
    tt_buffer: list[str] = []
    metrics = []

    for i, race_id in enumerate(race_order[initial_train_races:], start=initial_train_races):
        # Retrain every N races (use all history up to but not including race i).
        if (i - initial_train_races) % retrain_every_n == 0 and i > initial_train_races:
            all_prior = pd.concat([per_race_dfs[r] for r in race_order[:i]], ignore_index=True)
            train_data = all_prior
            if n_ensemble > 1:
                model = GBMEnsemble(drop_features=drop_features)
                model.fit(all_prior, n_estimators=n_ensemble, **_fit_kwargs)
            else:
                model = GBMRanker(drop_features=drop_features)
                model.fit(all_prior, **_fit_kwargs)

        race_feat = per_race_dfs[race_id]
        gbm_scores = model.predict_scores(race_feat)
        gbm_z = _zscore(np.asarray(gbm_scores))
        pl_z = _zscore(race_feat["pl_effective"].to_numpy())
        blended = ensemble_alpha * gbm_z + (1.0 - ensemble_alpha) * pl_z
        # Scale so distribution isn't trivially wide/narrow — T handles calibration.
        blended = blended / max(blended.std(), 1e-6)

        finishes = race_feat["finish_pos"].to_numpy()
        is_dnf = race_feat["is_dnf"].to_numpy()
        tt = race_feat["track_type"].iloc[0]
        base_hazard = HAZARD_DEFAULT.get(tt, 0.08)
        if per_driver_dnf and "dnf_rate_10" in race_feat.columns:
            drv_rate = race_feat["dnf_rate_10"].to_numpy(dtype=float)
            # Field mean of the observed rolling DNF rate (fallback to base_hazard).
            observed_mean = np.nanmean(drv_rate)
            if not np.isfinite(observed_mean) or observed_mean <= 0:
                observed_mean = base_hazard
            factor = np.where(np.isnan(drv_rate), 1.0, drv_rate / observed_mean)
            factor = np.clip(factor, dnf_scale_range[0], dnf_scale_range[1])
            hazards = np.clip(base_hazard * factor, 0.005, 0.6)
        else:
            hazards = np.full(len(race_feat), base_hazard)

        # Recalibrate T periodically from trailing window of past predictions.
        cal_step = i - initial_train_races
        if (len(predictions_buffer) >= calib_window
                and cal_step % recalibrate_every_n == 0):
            if per_type_T:
                T_by_type = find_best_temperature_by_type(
                    predictions_buffer[-calib_window:], tt_buffer[-calib_window:],
                    T_grid=T_grid, n_samples=1000, default_T=T,
                )
            else:
                best_T, _ = find_best_temperature(
                    predictions_buffer[-calib_window:],
                    T_grid=T_grid, n_samples=1000,
                )
                T = best_T

        active_T = T_by_type.get(tt, T) if per_type_T else T
        strengths = blended / active_T
        m = _race_metrics(strengths, hazards, finishes, is_dnf, n_samples, rng)
        m.update(dict(race_id=race_id, date=race_feat["date"].iloc[0],
                      track_type=tt, n_drivers=len(race_feat), T_used=active_T))
        metrics.append(m)

        predictions_buffer.append(RaceScoresGT(
            scores=blended, finishes=finishes, is_dnf=is_dnf, hazards=hazards,
        ))
        tt_buffer.append(tt)

    return {"metrics": pd.DataFrame(metrics), "model": model}


def summarize_wf(df: pd.DataFrame) -> pd.DataFrame:
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
