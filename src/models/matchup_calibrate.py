"""Post-sampling matchup-probability calibration.

The Gumbel-max sim outputs a matrix of pairwise probabilities p_ij =
P(driver_i beats driver_j). Those probabilities inherit whatever mis-
calibration the underlying scores have, and they've been shown to be
systematically overconfident on close calls (diagnostics show model
disagrees with market at 35% hit rate, but the matchup log-loss stays
high because the wrong picks are also confident).

This module fits a single scalar T_match on the training window's
matchup outcomes and applies it as a logit-space temperature:

    logit_new = logit(p) / T_match
    p_new     = sigmoid(logit_new)

T_match > 1 pulls all probabilities toward 0.5 (reduces confidence).
T_match < 1 sharpens them.

The fit target is log-loss on every pair (i, j) with i < j from prior races
where both drivers finished. We ignore matchups involving DNFs on either
side because they're determined by race variance rather than pairwise skill.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_matchup_temperature(p: np.ndarray, T: float) -> np.ndarray:
    """Temperature-scale probabilities in logit space."""
    z = logit(p) / T
    return 1.0 / (1.0 + np.exp(-z))


def fit_matchup_temperature(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    T_bounds: tuple[float, float] = (0.3, 5.0),
) -> float:
    """Return the T that minimizes binary log-loss on the training pairs.

    predicted_probs: shape (n_pairs,), model's P(A beats B).
    outcomes: shape (n_pairs,), 1 if A actually beat B else 0.
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)

    def loss(T):
        p_cal = apply_matchup_temperature(predicted_probs, T)
        p_cal = np.clip(p_cal, 1e-9, 1 - 1e-9)
        return -float(np.mean(outcomes * np.log(p_cal)
                              + (1 - outcomes) * np.log(1 - p_cal)))

    result = minimize_scalar(loss, bounds=T_bounds, method="bounded")
    return float(result.x)
