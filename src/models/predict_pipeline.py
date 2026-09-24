"""Shared prediction pipeline.

Matches the honest calibration setup in backtest_oddslogic_v5.py:
  1. Leave-one-race-out CV to fit T per track type (val races are scored by
     an ensemble refit WITHOUT them — no leakage).
  2. Per-driver hazards on both validation and target races (using empirical
     dnf_rate_at_type_10 + crash bump).
  3. Single calibrated temperature T handles both matchup and top-N. No T_m
     layer, no T_top_n widening — the honest T is the honest T.

Both predict_next.py and ev_next.py should call `calibrate_and_sample`
so they can't drift from the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SampledRace:
    T: float
    T_effective: float          # kept for compatibility; equals T
    blended: np.ndarray         # per-driver blended scores
    hazards: np.ndarray         # per-driver hazard rates
    positions: np.ndarray       # (N_SAMPLES, n_drivers), 1-indexed
    matchup_mtx: np.ndarray     # (n_drivers, n_drivers)
    val_pool_size: int          # for logging / guardrail


def calibrate_and_sample(
    train: pd.DataFrame,
    target: pd.DataFrame,
    track_type: str,
    *,
    tight_reg: dict,
    alpha: float,
    n_samples: int,
    hazard_lookup: dict,
    per_driver_hazards_fn,
    n_estimators: int = 15,
    cv_n_estimators: int | None = None,
    val_start: int = 30,
    val_cap_per_type: int | None = None,
    seed: int = 42,
    verbose: bool = True,
    dnf_dispersion_by_type: dict[str, float] | None = None,
    per_driver_damage_hazards_fn=None,   # signature: (target_df, tt) -> np.ndarray | None
    damage_penalty: float = 3.0,
    damage_penalty_by_type: dict[str, float] | None = None,
) -> SampledRace:
    """Fit model + honest T (leave-one-race-out CV) + sample.

    hazard_lookup:      per-track-type baseline dict (HAZARD)
    per_driver_hazards_fn: the function that blends per-driver DNF/crash rates

    Speedup knobs (mirror the ones in backtest_oddslogic_v5.py):
      cv_n_estimators:    n_estimators for the LOO CV refits. Defaults to
                          n_estimators (identical to primary model). Setting
                          to a smaller value (e.g. 5) trades ~1-2% of T-fit
                          precision for 3x wall-time reduction. T is a
                          1-parameter fit — doesn't need ensemble precision.
      val_cap_per_type:   max val races per track type. Keeps the MOST RECENT
                          races per type. Defaults to None (all val races).
                          A cap of ~30 per type is 5-10x oversampled for a
                          1-parameter fit and cuts wall time proportionally.

    Returns everything the caller needs to compute matchup probs and top-N.
    """
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble

    # ---- Fit primary model on all training data ----
    model = GBMEnsemble()
    model.fit(train, n_estimators=n_estimators, **tight_reg)

    # ---- Validation pool: races chronologically after `val_start` ----
    race_order = (
        train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
    )
    val_ids = race_order[val_start:]
    if len(val_ids) < 4 and verbose:
        print(f"WARNING: val pool has only {len(val_ids)} races "
              f"(of {len(race_order)}); calibration may default.")

    # ---- Optional per-track-type val cap: keep the MOST RECENT races per type ----
    if val_cap_per_type is not None and val_ids:
        tt_by_rid = train.groupby("race_id_short")["track_type"].first().to_dict()
        per_type: dict[str, list] = {}
        for rid in val_ids:
            per_type.setdefault(tt_by_rid.get(rid), []).append(rid)
        keep = set()
        for _tt_key, rids in per_type.items():
            keep.update(rids[-val_cap_per_type:])
        val_ids = [rid for rid in val_ids if rid in keep]

    cv_ne = cv_n_estimators if cv_n_estimators is not None else n_estimators

    # ---- Honest CV: refit ensemble WITHOUT each val race, score that race ----
    val_races, tt_list = [], []
    for rid in val_ids:
        sub = train[train["race_id_short"] == rid]
        train_minus = train[train["race_id_short"] != rid]
        m_cv = GBMEnsemble()
        m_cv.fit(train_minus, n_estimators=cv_ne, **tight_reg)
        raw = m_cv.predict_scores(sub)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                / (sub["pl_effective"].std() + 1e-9))
        b = alpha * gbm_z + (1 - alpha) * pl_z
        b = b / max(b.std(), 1e-6)
        v_tt = sub["track_type"].iloc[0]
        val_races.append(RaceScoresGT(
            scores=b,
            finishes=sub["finish_pos"].to_numpy(),
            is_dnf=sub["is_dnf"].to_numpy(),
            hazards=per_driver_hazards_fn(sub, v_tt),
        ))
        tt_list.append(v_tt)

    T_by_type = find_best_temperature_by_type(
        val_races, tt_list, default_T=1.0, n_samples=1500,
        dnf_dispersion_by_type=dnf_dispersion_by_type,
    )
    T = T_by_type.get(track_type, 1.0)

    # ---- Score target race ----
    raw = model.predict_scores(target)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
            / (target["pl_effective"].std() + 1e-9))
    blended = alpha * gbm_z + (1 - alpha) * pl_z
    blended = blended / max(blended.std(), 1e-6)
    haz = per_driver_hazards_fn(target, track_type)

    T_effective = T
    if verbose:
        print(f"Track type: {track_type}, T={T:.3f} (single-layer, T_effective=T)")

    rng = np.random.default_rng(seed)
    disp = (dnf_dispersion_by_type.get(track_type, 0.0)
            if dnf_dispersion_by_type else 0.0)
    dam = (per_driver_damage_hazards_fn(target, track_type)
           if per_driver_damage_hazards_fn is not None else None)
    dam_pen = (damage_penalty_by_type.get(track_type, damage_penalty)
               if damage_penalty_by_type else damage_penalty)
    positions = dist.sample_finishing_orders(
        blended / T_effective, haz, n_samples=n_samples, rng=rng,
        dnf_dispersion=disp,
        damage_hazards=dam,
        damage_penalty=dam_pen,
    )
    matchup_mtx = dist.matchup_probs(positions)

    return SampledRace(
        T=T,
        T_effective=T_effective,
        blended=blended,
        hazards=haz,
        positions=positions,
        matchup_mtx=matchup_mtx,
        val_pool_size=len(val_ids),
    )
