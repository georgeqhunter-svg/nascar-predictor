"""Sample joint finishing orders from Plackett-Luce strengths + DNF hazards.

The two market queries we care about - P(winner) and P(A finishes ahead of B) -
both fall out of a Monte Carlo simulation of the finishing distribution.

Sampling a Plackett-Luce order given strengths s_1..s_n is a well-known trick:
draw g_i = exp(s_i) / Exp(1) = -ln(u_i) / exp(s_i), sort by g ascending. The
resulting permutation is exactly a PL sample. Vectorizes cleanly in numpy.

DNFs are added as a mixture: with hazard h_i, each driver DNFs and their
position falls to the tail. Phase 2: hazards share a race-level frailty so
DNFs CORRELATE within a sampled race (the big one at plate tracks). The
frailty c ~ Gamma(mean=1, var=v) multiplies every driver's hazard for that
sample; v is the per-track-type dispersion (0 = independent, old behavior).
"""
from __future__ import annotations

import numpy as np


def sample_finishing_orders(
    strengths: np.ndarray,
    dnf_hazards: np.ndarray,
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
    dnf_dispersion: float = 0.0,
    damage_hazards: np.ndarray | None = None,
    damage_penalty: float = 3.0,
) -> np.ndarray:
    """Return an (n_samples, n_drivers) integer array; entry [k, i] = finish
    position of driver i in sample k (1 = winner).

    Three outcome buckets per driver per sample:
      - Clean race → score from Plackett-Luce Gumbel
      - Damage (car takes contact / penalty / limps home): score gets a
        large negative bump so the driver sorts near the bottom BUT NOT AT
        THE BACK. Produces the realistic "P5 pace, finished P22" outcomes.
      - DNF: score dumped to -1e6, driver sorts last.

    dnf_dispersion > 0 correlates DNFs within each sampled race via a shared
    Gamma frailty (mean 1, variance = dnf_dispersion) multiplying all hazards.
    Exactly 0.0 reproduces the old independent-hazard behavior (same draws).

    damage_hazards: per-driver probability of a damage-day (typically 5-15%
    depending on track type). Empirical values in fit_damage.py.
    damage_penalty: how many score standard deviations to subtract when
    damaged. 3.0 produces roughly a P20-P25 outcome for a P5-pace driver.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_drivers = len(strengths)
    assert dnf_hazards.shape == strengths.shape

    # Plackett-Luce sampling by Gumbel-max trick.
    u = rng.random((n_samples, n_drivers))
    u = np.clip(u, 1e-12, 1.0)
    gumbel = -np.log(-np.log(u))
    scores = strengths[None, :] + gumbel

    # Damage injection — happens BEFORE DNF. Damaged-but-alive drivers get a
    # large but finite score penalty so they sort low without being absolute
    # last. Uniform penalty in [1x, 2x] * damage_penalty gives some variance
    # (some damage days end P18, some end P30).
    if damage_hazards is not None:
        assert damage_hazards.shape == strengths.shape
        damage_draw = rng.random((n_samples, n_drivers))
        damaged = damage_draw < damage_hazards[None, :]
        if damaged.any():
            penalty = rng.uniform(1.0, 2.0, size=(n_samples, n_drivers)) * damage_penalty
            scores = np.where(damaged, scores - penalty, scores)

    # DNF injection with optional race-level frailty.
    if dnf_dispersion > 0.0:
        shape = 1.0 / dnf_dispersion
        c = rng.gamma(shape, dnf_dispersion, size=(n_samples, 1))  # mean 1, var v
        haz = np.clip(dnf_hazards[None, :] * c, 0.0, 0.95)
    else:
        haz = dnf_hazards[None, :]
    dnf = rng.random((n_samples, n_drivers)) < haz

    tail_jitter = rng.random((n_samples, n_drivers)) * 0.01
    scores = np.where(dnf, -1e6 + tail_jitter, scores)

    # Rank each row: highest score = position 1.
    order = np.argsort(-scores, axis=1)
    positions = np.empty_like(order)
    ar = np.arange(n_drivers)
    for k in range(n_samples):
        positions[k, order[k]] = ar + 1
    return positions


def win_probs(positions: np.ndarray) -> np.ndarray:
    """P(driver_i finishes first) from sampled positions array."""
    return (positions == 1).mean(axis=0)


def matchup_probs(positions: np.ndarray) -> np.ndarray:
    """(n_drivers, n_drivers) matrix M[i, j] = P(driver_i finishes ahead of driver_j)."""
    less = positions[:, :, None] < positions[:, None, :]
    return less.mean(axis=0)


def topn_probs(positions: np.ndarray, n: int) -> np.ndarray:
    """P(driver_i finishes in the top-n) from sampled positions."""
    return (positions <= n).mean(axis=0)