"""Post-hoc calibration for the GBM ranker.

Two techniques:

* Temperature scaling — a single scalar T applied as strength = score / T.
  T > 1 flattens the distribution (less confident); T < 1 sharpens it.
  Chosen to minimize a validation loss (matchup log-loss by default).

* Isotonic regression on winner probabilities — fits a monotonic mapping from
  sample-derived P(winner) to observed winner frequency. Applied post-sampling.

The two are complementary: temperature fixes global over/under-confidence
before sampling; isotonic corrects the head-of-distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import distribution as dist


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
@dataclass
class RaceScoresGT:
    scores: np.ndarray
    finishes: np.ndarray
    is_dnf: np.ndarray
    hazards: np.ndarray


def _log_clip(p: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, floor, 1.0 - floor))


def _matchup_ll_for_race(sample_M: np.ndarray, finishes: np.ndarray, is_dnf: np.ndarray) -> float:
    n = len(finishes)
    ll = 0.0
    n_pairs = 0
    for a in range(n):
        if is_dnf[a]:
            continue
        for b in range(a + 1, n):
            if is_dnf[b]:
                continue
            a_ahead = finishes[a] < finishes[b]
            p = sample_M[a, b] if a_ahead else sample_M[b, a]
            ll += -float(_log_clip(np.array([p]))[0])
            n_pairs += 1
    return ll / max(n_pairs, 1)


def find_best_temperature(
    val_races: list[RaceScoresGT],
    T_grid: np.ndarray | None = None,
    n_samples: int = 2000,
    seed: int = 0,
    dispersion_per_race: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Grid-search T minimizing mean matchup log-loss over val races.

    If dispersion_per_race is provided (aligned with val_races), val sampling
    uses correlated-DNF frailty consistent with what production sampling will
    use. Otherwise defaults to independent hazards.

    Returns (best_T, per-T loss table).
    """
    if T_grid is None:
        T_grid = np.array([0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0])
    rng = np.random.default_rng(seed)

    rows = []
    for T in T_grid:
        losses = []
        winner_losses = []
        for ridx, r in enumerate(val_races):
            strengths = r.scores / T
            disp = (dispersion_per_race[ridx]
                    if dispersion_per_race is not None else 0.0)
            positions = dist.sample_finishing_orders(
                strengths, r.hazards, n_samples=n_samples, rng=rng,
                dnf_dispersion=disp,
            )
            M = dist.matchup_probs(positions)
            winp = dist.win_probs(positions)
            losses.append(_matchup_ll_for_race(M, r.finishes, r.is_dnf))
            winner_idx = int(np.argmin(r.finishes))
            winner_losses.append(-float(_log_clip(np.array([winp[winner_idx]]))[0]))
        rows.append({"T": float(T), "matchup_ll": float(np.mean(losses)),
                     "winner_ll": float(np.mean(winner_losses))})
    df = pd.DataFrame(rows)
    best_T = float(df.loc[df["matchup_ll"].idxmin(), "T"])
    return best_T, df


def find_best_temperature_by_type(
    val_races: list[RaceScoresGT],
    track_types: list[str],
    T_grid: np.ndarray | None = None,
    n_samples: int = 1500,
    default_T: float = 3.0,
    seed: int = 0,
    dnf_dispersion_by_type: dict[str, float] | None = None,
) -> dict[str, float]:
    """Search T separately per track_type. Falls back to a global search if a
    type has too few val races.

    dnf_dispersion_by_type is passed through so val sampling uses the same
    correlated-DNF frailty that production sampling will use.
    """
    if len(val_races) != len(track_types):
        raise ValueError("length mismatch between val_races and track_types")
    by_type: dict[str, list[RaceScoresGT]] = {}
    for r, tt in zip(val_races, track_types):
        by_type.setdefault(tt, []).append(r)

    result: dict[str, float] = {}
    for tt, sub in by_type.items():
        if len(sub) < 4:
            result[tt] = default_T
            continue
        disp = 0.0
        if dnf_dispersion_by_type is not None:
            disp = dnf_dispersion_by_type.get(tt, 0.0)
        # All races in `sub` share the same track type, so a single dispersion.
        disp_per_race = [disp] * len(sub) if disp > 0 else None
        best_T, _ = find_best_temperature(
            sub, T_grid=T_grid, n_samples=n_samples, seed=seed,
            dispersion_per_race=disp_per_race,
        )
        result[tt] = best_T
    return result


# --------------------------------------------------------------------------- #
# Isotonic on winner probs
# --------------------------------------------------------------------------- #
def fit_isotonic_winner(win_probs: np.ndarray, was_winner: np.ndarray):
    """Fit monotonic map: predicted P(winner) -> calibrated P(winner).

    Returns a callable that takes an array of raw probs and returns calibrated
    ones. Uses sklearn's IsotonicRegression which needs to be re-normalized
    per race (so probs still sum to 1 within a race)."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        raise ImportError("scikit-learn is required for isotonic calibration") from e

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(win_probs, was_winner.astype(float))
    return iso.predict


def apply_winner_isotonic(win_probs_race: np.ndarray, iso_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply isotonic map to one race's win probs; renormalize to sum to 1."""
    calibrated = iso_fn(win_probs_race)
    s = calibrated.sum()
    if s <= 0:
        return win_probs_race  # fallback: return uncalibrated
    return calibrated / s
