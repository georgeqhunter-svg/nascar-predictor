"""Walk-forward backtest of the Plackett-Luce baseline.

For each race in chronological order:
  1. Predict using current ratings state:
       - per-driver win probability
       - pairwise matchup probability matrix
  2. Score against observed outcome:
       - winner log-loss
       - matchup log-loss (over all C(n,2) pairs, non-DNFs only)
       - Spearman rank correlation of predicted mean finish vs. actual
  3. Update ratings with the race result.

Reports rolling metrics + final holdout metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..features.qualifying import qual_zscore_for_race, qualifying_best_by_driver
from ..models import distribution as dist
from ..models.plackett_luce import (
    RaceRanking,
    Ratings,
    entry_strengths,
    update_from_race,
)

log = logging.getLogger(__name__)


@dataclass
class RaceResult:
    race_id: str
    date: pd.Timestamp
    track_type: str
    n_drivers: int
    winner_logloss: float
    matchup_logloss: float
    spearman_rho: float
    top5_accuracy: float


def _log_clip(p: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, floor, 1.0 - floor))


def _dnf_hazards(track_type: str, hazard_by_type: dict[str, float]) -> float:
    return hazard_by_type.get(track_type, 0.08)


def backtest(
    races: pd.DataFrame,
    entries: pd.DataFrame,
    ratings: Ratings | None = None,
    hazard_by_type: dict[str, float] | None = None,
    n_samples: int = 3000,
    burn_in_races: int = 30,
    sessions: pd.DataFrame | None = None,
    qual_weight: float = 0.0,
) -> tuple[Ratings, pd.DataFrame]:
    """Run walk-forward. `entries` must have columns: race_id_short, driver, team,
    finish_pos, is_dnf, track_type. `races` provides ordering (date).

    If `sessions` is provided and `qual_weight` > 0, per-race effective driver
    strength is augmented by `qual_weight * z(qual_speed)`. z=0 for missing data
    so drivers without qualifying data get no adjustment (neutral).
    """
    if ratings is None:
        ratings = Ratings()
    if hazard_by_type is None:
        hazard_by_type = {
            "superspeedway": 0.20, "intermediate": 0.05, "short": 0.06,
            "road": 0.04, "unique": 0.07,
        }

    # Precompute per-race qualifying lookup once (fast subsequent slicing).
    qual_by_driver = (
        qualifying_best_by_driver(sessions)
        if (sessions is not None and qual_weight != 0.0)
        else None
    )

    rng = np.random.default_rng(0)

    ordered = races.sort_values("date").reset_index(drop=True)
    results: list[RaceResult] = []

    for i, race_row in ordered.iterrows():
        race_id = race_row["race_id_short"]
        tt = race_row["track_type"]
        e = (
            entries[entries["race_id_short"] == race_id]
            .sort_values("finish_pos")
            .reset_index(drop=True)
        )
        if len(e) < 5:
            log.warning("skip race %s (%s): only %d entries in parquet",
                        race_id, race_row.get("race_name", ""), len(e))
            continue
        drivers = e["driver"].tolist()
        teams = e["team"].tolist()
        finishes = e["finish_pos"].to_numpy()
        is_dnf = e["is_dnf"].to_numpy()

        # ---- predict ---- #
        strengths = entry_strengths(ratings, drivers, teams, tt)
        if qual_by_driver is not None:
            z = qual_zscore_for_race(qual_by_driver, race_id, drivers)
            strengths = strengths + qual_weight * z
        hazards = np.full(len(drivers), _dnf_hazards(tt, hazard_by_type))
        positions = dist.sample_finishing_orders(
            strengths, hazards, n_samples=n_samples, rng=rng
        )
        winp = dist.win_probs(positions)
        M = dist.matchup_probs(positions)  # M[i, j] = P(i finishes ahead of j)

        # ---- score ---- #
        winner_idx = int(np.argmin(finishes))
        winner_ll = -float(_log_clip(np.array([winp[winner_idx]]))[0])

        # Matchup loss over informative pairs: both non-DNF.
        pair_ll = 0.0
        n_pairs = 0
        n = len(drivers)
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

        # Expected finish under the sampled positions.
        expected_finish = positions.mean(axis=0)
        rho = spearmanr(expected_finish, finishes).statistic
        rho = 0.0 if np.isnan(rho) else float(rho)

        top5_pred = np.argsort(-winp)[:5]
        actual_top5 = set(np.where(finishes <= 5)[0])
        top5_acc = len(set(top5_pred.tolist()) & actual_top5) / 5.0

        results.append(RaceResult(
            race_id=race_id,
            date=race_row["date"],
            track_type=tt,
            n_drivers=len(drivers),
            winner_logloss=winner_ll,
            matchup_logloss=matchup_ll,
            spearman_rho=rho,
            top5_accuracy=top5_acc,
        ))

        # ---- update ---- #
        # Truncate ranking-loss updates at first DNF position.
        first_dnf = int(np.argmax(is_dnf)) if is_dnf.any() else len(drivers)
        race = RaceRanking(
            drivers=drivers,
            teams=teams,
            track_type=tt,
            dnf_at=first_dnf,
        )
        update_from_race(ratings, race)

    df = pd.DataFrame([r.__dict__ for r in results])
    df["is_burn_in"] = df.index < burn_in_races
    return ratings, df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Group post-burn-in metrics by track type."""
    live = df[~df["is_burn_in"]]
    grouped = live.groupby("track_type").agg(
        n_races=("race_id", "count"),
        winner_logloss=("winner_logloss", "mean"),
        matchup_logloss=("matchup_logloss", "mean"),
        spearman_rho=("spearman_rho", "mean"),
        top5_accuracy=("top5_accuracy", "mean"),
    )
    total = live.agg({
        "winner_logloss": "mean",
        "matchup_logloss": "mean",
        "spearman_rho": "mean",
        "top5_accuracy": "mean",
    })
    total_row = pd.DataFrame(
        [{"n_races": len(live), **total.to_dict()}],
        index=["ALL"],
    )
    total_row.index.name = "track_type"
    return pd.concat([grouped, total_row])
