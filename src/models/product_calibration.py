"""Per-product calibration harness.

Fits isotonic calibrators for each product type using validation-race data.
Products supported:
  - matchup: pairwise "does A beat B" probability
  - top5:    per-driver "finish in top 5" probability
  - top10:   per-driver "finish in top 10" probability
  - winner:  per-driver "finish 1st" probability

Fit once on the validation set (races used for temperature calibration).
Apply at prediction time to whichever product is being consumed.
"""
from __future__ import annotations

import numpy as np

from . import distribution as dist
from .isotonic_calibrate import ProbabilityCalibrator


def fit_product_calibrators(
    val_races,     # list[RaceScoresGT]
    T: float,      # score temperature (already fit)
    n_samples: int = 5000,
    tt_list: list[str] | None = None,
) -> dict[str, ProbabilityCalibrator | dict]:
    """Sample val races at temperature T, then fit isotonic calibrators for
    matchup, top-5, top-10, and winner probs against actual outcomes.

    If tt_list is provided, ALSO fit per-track-type isotonic calibrators for
    matchups — those get returned as {"matchup_by_type": {tt: cal, ...}}.
    """
    matchup_p, matchup_o = [], []
    matchup_by_type_p: dict[str, list] = {}
    matchup_by_type_o: dict[str, list] = {}
    top5_p, top5_o = [], []
    top10_p, top10_o = [], []
    winner_p, winner_o = [], []

    for k, vr in enumerate(val_races):
        haz_v = np.full(len(vr.scores), 0.08)
        rng_v = np.random.default_rng(0)
        pos_v = dist.sample_finishing_orders(
            vr.scores / max(T, 0.1), haz_v, n_samples=n_samples, rng=rng_v
        )
        m_v = dist.matchup_probs(pos_v)
        f = vr.finishes
        n_v = len(f)

        # Matchup pairs — add BOTH directions of each pair because our
        # validation data comes sorted by finish position, so upper-triangle
        # alone has outcome=1 for every entry and breaks isotonic calibration.
        this_tt = tt_list[k] if tt_list is not None else None
        if this_tt is not None:
            matchup_by_type_p.setdefault(this_tt, [])
            matchup_by_type_o.setdefault(this_tt, [])
        for ii in range(n_v):
            for jj in range(n_v):
                if ii == jj: continue
                if f[ii] <= 0 or f[jj] <= 0: continue
                if f[ii] == f[jj]: continue
                p = float(m_v[ii, jj])
                o = int(f[ii] < f[jj])
                matchup_p.append(p)
                matchup_o.append(o)
                if this_tt is not None:
                    matchup_by_type_p[this_tt].append(p)
                    matchup_by_type_o[this_tt].append(o)

        # Per-driver top-N and winner.
        p_top5 = (pos_v <= 5).sum(axis=0) / n_samples
        p_top10 = (pos_v <= 10).sum(axis=0) / n_samples
        p_win = (pos_v == 1).sum(axis=0) / n_samples
        for i in range(n_v):
            if f[i] <= 0:
                continue
            top5_p.append(float(p_top5[i]))
            top5_o.append(int(f[i] <= 5))
            top10_p.append(float(p_top10[i]))
            top10_o.append(int(f[i] <= 10))
            winner_p.append(float(p_win[i]))
            winner_o.append(int(f[i] == 1))

    calibrators: dict = {}
    calibrators["matchup"] = ProbabilityCalibrator(min_samples=200).fit(
        np.asarray(matchup_p), np.asarray(matchup_o)
    )
    calibrators["top5"] = ProbabilityCalibrator(min_samples=200).fit(
        np.asarray(top5_p), np.asarray(top5_o)
    )
    calibrators["top10"] = ProbabilityCalibrator(min_samples=200).fit(
        np.asarray(top10_p), np.asarray(top10_o)
    )
    calibrators["winner"] = ProbabilityCalibrator(min_samples=200).fit(
        np.asarray(winner_p), np.asarray(winner_o)
    )
    # Per-track-type matchup calibrators. Need thick sample (500+ pairs) to
    # trust the type-specific fit; else caller should fall back to the pooled
    # calibrator above.
    calibrators["matchup_by_type"] = {}
    for tt, ps in matchup_by_type_p.items():
        os_ = matchup_by_type_o[tt]
        if len(ps) < 500:
            continue
        calibrators["matchup_by_type"][tt] = ProbabilityCalibrator(
            min_samples=500).fit(np.asarray(ps), np.asarray(os_))
    return calibrators
