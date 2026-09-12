"""Plackett-Luce ratings for NASCAR finishing orders.

The model:
    Effective strength of driver i in race r =
        theta_driver[i]                       # overall driver skill
      + theta_team[team[i]]                   # team equipment strength
      + theta_track[i, track_type[r]]         # driver's per-track-type adjustment
    P(driver_i finishes first in race r) = exp(strength_i) / sum_j exp(strength_j)

Fitting is by online gradient ascent on the ranking log-likelihood after each
race. This is the standard incremental Plackett-Luce update:

    for each position k from top down:
        residual = 1{driver at position k} - exp(strength) / sum_of_remaining_exp
        strength[i] += lr * residual   # for all remaining drivers i

We treat DNFs as a censored tail: finishing positions among DNFs carry little
information, so we truncate the ranking loss at the first-DNF position.

All state is a plain dict; save/load is JSON. Deliberately small dependencies.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..features.tracks import TrackType

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
TRACK_TYPES: tuple[TrackType, ...] = (
    "superspeedway", "intermediate", "short", "road", "unique",
)


@dataclass
class Ratings:
    """Mutable rating state. All strengths start at 0 (neutral)."""

    driver: dict[str, float] = field(default_factory=dict)
    team: dict[str, float] = field(default_factory=dict)
    # driver -> track_type -> per-type adjustment.
    driver_track: dict[str, dict[str, float]] = field(default_factory=dict)
    n_updates: int = 0

    # Regularization anchors: driver ratings shrink toward 0 (league average),
    # team toward 0, and driver-track toward 0.
    lr_driver: float = 0.15
    lr_team: float = 0.08
    lr_driver_track: float = 0.05
    l2_driver: float = 0.003
    l2_team: float = 0.003
    l2_driver_track: float = 0.02

    # --- accessors --------------------------------------------------------- #
    def strength(self, driver: str, team: str, track_type: TrackType) -> float:
        d = self.driver.get(driver, 0.0)
        t = self.team.get(team, 0.0)
        dt = self.driver_track.get(driver, {}).get(track_type, 0.0)
        return d + t + dt

    def bulk_strengths(
        self,
        drivers: Sequence[str],
        teams: Sequence[str],
        track_type: TrackType,
    ) -> np.ndarray:
        return np.array(
            [self.strength(d, t, track_type) for d, t in zip(drivers, teams)],
            dtype=np.float64,
        )

    # --- serialize --------------------------------------------------------- #
    def to_json(self) -> dict:
        return {
            "driver": self.driver,
            "team": self.team,
            "driver_track": self.driver_track,
            "n_updates": self.n_updates,
            "hyperparams": {
                "lr_driver": self.lr_driver,
                "lr_team": self.lr_team,
                "lr_driver_track": self.lr_driver_track,
                "l2_driver": self.l2_driver,
                "l2_team": self.l2_team,
                "l2_driver_track": self.l2_driver_track,
            },
        }

    @classmethod
    def from_json(cls, blob: dict) -> "Ratings":
        hp = blob.get("hyperparams", {})
        r = cls(
            driver=dict(blob.get("driver", {})),
            team=dict(blob.get("team", {})),
            driver_track={k: dict(v) for k, v in blob.get("driver_track", {}).items()},
            n_updates=int(blob.get("n_updates", 0)),
            **{k: float(v) for k, v in hp.items()},
        )
        return r

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Ratings":
        return cls.from_json(json.loads(path.read_text()))


# --------------------------------------------------------------------------- #
# Update
# --------------------------------------------------------------------------- #
@dataclass
class RaceRanking:
    """One race's ranked outcome, ready to feed the PL updater.

    `positions_order` is a list of driver ids in finishing order (best first).
    `dnf_at` is the index into that list at which DNFs begin — the update loop
    stops before that index (ranks among DNFs are noisy).
    """

    drivers: list[str]
    teams: list[str]
    track_type: TrackType
    dnf_at: int


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()


def update_from_race(ratings: Ratings, race: RaceRanking) -> None:
    """Online Plackett-Luce gradient update over the ranked positions of one race."""
    n = len(race.drivers)
    if n < 2 or race.dnf_at < 2:
        return  # nothing to learn from

    strengths = ratings.bulk_strengths(race.drivers, race.teams, race.track_type)

    # Consider positions 0..dnf_at-1 as informative rankings.
    stop = min(race.dnf_at, n - 1)
    remaining = np.ones(n, dtype=bool)

    for k in range(stop):
        if not remaining.any():
            break
        rem_idx = np.where(remaining)[0]
        rem_str = strengths[rem_idx]
        probs = _softmax(rem_str)  # P(each remaining finishes next)

        # winner at this rank is remaining index 0 (drivers list is in finish order).
        winner_local = np.where(rem_idx == k)[0][0]
        target = np.zeros_like(probs)
        target[winner_local] = 1.0
        grad = target - probs  # positive for actual winner, negative for others

        # Distribute the gradient back to the (driver, team, driver_track) params.
        for local_i, global_i in enumerate(rem_idx):
            drv = race.drivers[global_i]
            tm = race.teams[global_i]
            g = float(grad[local_i])

            ratings.driver[drv] = (
                ratings.driver.get(drv, 0.0) * (1 - ratings.l2_driver)
                + ratings.lr_driver * g
            )
            ratings.team[tm] = (
                ratings.team.get(tm, 0.0) * (1 - ratings.l2_team)
                + ratings.lr_team * g
            )
            dt = ratings.driver_track.setdefault(drv, {})
            dt[race.track_type] = (
                dt.get(race.track_type, 0.0) * (1 - ratings.l2_driver_track)
                + ratings.lr_driver_track * g
            )

        remaining[k] = False

    ratings.n_updates += 1


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def entry_strengths(
    ratings: Ratings,
    drivers: Sequence[str],
    teams: Sequence[str],
    track_type: TrackType,
) -> np.ndarray:
    """Vector of effective strengths for the field entered in an upcoming race."""
    return ratings.bulk_strengths(drivers, teams, track_type)


def race_log_likelihood(
    ratings: Ratings,
    race: RaceRanking,
) -> float:
    """Sum of log-P for the informative positions of a race under current ratings."""
    strengths = ratings.bulk_strengths(race.drivers, race.teams, race.track_type)
    ll = 0.0
    remaining = np.ones(len(race.drivers), dtype=bool)
    stop = min(race.dnf_at, len(race.drivers) - 1)
    for k in range(stop):
        rem_idx = np.where(remaining)[0]
        rem_str = strengths[rem_idx]
        probs = _softmax(rem_str)
        winner_local = np.where(rem_idx == k)[0][0]
        ll += math.log(max(probs[winner_local], 1e-12))
        remaining[k] = False
    return ll
