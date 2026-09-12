"""Sample joint finishing orders from Plackett-Luce strengths + DNF hazards.

The two market queries we care about — P(winner) and P(A finishes ahead of B) —
both fall out of a Monte Carlo simulation of the finishing distribution.

Sampling a Plackett-Luce order given strengths s_1..s_n is a well-known trick:
draw g_i = exp(s_i) / Exp(1) = -ln(u_i) / exp(s_i), sort by g ascending. The
resulting permutation is exactly a PL sample. Vectorizes cleanly in numpy.

DNFs are added as a mixture: with hazard h_i, each driver DNFs independently
and, if they DNF, their position is drawn uniformly from the bottom third of
the field. This is a rough phase-1 approximation; a proper hazard-adjusted
finish model comes later.
"""
from __future__ import annotations

import numpy as np


def sample_finishing_orders(
    strengths: np.ndarray,
    dnf_hazards: np.ndarray,
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return an (n_samples, n_drivers) integer array; entry [k, i] = finish
    position of driver i in sample k (1 = winner)."""
    if rng is None:
        rng = np.random.default_rng()
    n_drivers = len(strengths)
    assert dnf_hazards.shape == strengths.shape

    # Plackett-Luce sampling by inverse-CDF trick.
    u = rng.random((n_samples, n_drivers))
    # Numerical guard: never 0.
    u = np.clip(u, 1e-12, 1.0)
    # Larger score -> earlier finish. Draw score = strength - log(-log u).
    # This is equivalent to Gumbel-max sampling from softmax(strengths).
    gumbel = -np.log(-np.log(u))
    scores = strengths[None, :] + gumbel

    # DNF injection: for each sample, roll independent DNFs, then push those
    # drivers to the tail with a shuffled tail order.
    dnf = rng.random((n_samples, n_drivers)) < dnf_hazards[None, :]
    # Give DNFed drivers a large negative score so they sort to the back;
    # jitter among themselves so they don't tie.
    tail_jitter = rng.random((n_samples, n_drivers)) * 0.01
    scores = np.where(dnf, -1e6 + tail_jitter, scores)

    # Rank each row: highest score = position 1.
    # argsort ascending gives worst first, so reverse via [::-1].
    order = np.argsort(-scores, axis=1)  # order[k, p] = driver at position p (0-indexed)
    positions = np.empty_like(order)
    ar = np.arange(n_drivers)
    for k in range(n_samples):
        positions[k, order[k]] = ar + 1  # positions[k, driver] = 1..n
    return positions


def win_probs(positions: np.ndarray) -> np.ndarray:
    """P(driver_i finishes first) from sampled positions array."""
    return (positions == 1).mean(axis=0)


def matchup_probs(positions: np.ndarray) -> np.ndarray:
    """(n_drivers, n_drivers) matrix M[i, j] = P(driver_i finishes ahead of driver_j)."""
    # positions[k, i] < positions[k, j] means driver i finished ahead of j in sample k.
    # Use broadcasting for vectorized pairwise comparison.
    less = positions[:, :, None] < positions[:, None, :]  # (samples, n, n)
    return less.mean(axis=0)


def topn_probs(positions: np.ndarray, n: int) -> np.ndarray:
    """P(driver_i finishes in the top-n) from sampled positions."""
    return (positions <= n).mean(axis=0)
