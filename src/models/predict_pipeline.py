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


# ---------------------------------------------------------------------------
# Score spread. "per_race" (old): z-score GBM + PL within each race and rescale
# the blend to std 1, so EVERY race gets the same confidence — a rainout looks
# as certain as a full practice/qualifying weekend. "global": centre within the
# race but divide by constants fit on the out-of-sample CV races (median
# per-race std), so races where the model's scores are compressed stay
# compressed. The median race keeps std ~1, so the T grid is unchanged.
# ---------------------------------------------------------------------------
SCORE_SPREAD_MODE = "global"   # "global" | "per_race"


def fit_blend_scales(pairs: list[tuple[np.ndarray, np.ndarray]], alpha: float) -> dict | None:
    """pairs = [(gbm_raw, pl_effective), ...] from out-of-sample CV races."""
    if SCORE_SPREAD_MODE != "global" or not pairs:
        return None
    s_g = float(np.median([np.std(r) for r, _ in pairs]))
    s_p = float(np.median([np.std(p) for _, p in pairs]))
    s_g, s_p = max(s_g, 1e-6), max(s_p, 1e-6)
    b_std = [np.std(alpha * (r - r.mean()) / s_g + (1 - alpha) * (p - p.mean()) / s_p)
             for r, p in pairs]
    return {"gbm": s_g, "pl": s_p, "blend": max(float(np.median(b_std)), 1e-6)}


def blend_scores(raw: np.ndarray, pl: np.ndarray, alpha: float,
                 scales: dict | None) -> np.ndarray:
    raw = np.asarray(raw, dtype=float); pl = np.asarray(pl, dtype=float)
    if scales is None:   # per-race (old behaviour)
        g = (raw - raw.mean()) / (raw.std() + 1e-9)
        p = (pl - pl.mean()) / (pl.std() + 1e-9)
        b = alpha * g + (1 - alpha) * p
        return b / max(b.std(), 1e-6)
    g = (raw - raw.mean()) / scales["gbm"]
    p = (pl - pl.mean()) / scales["pl"]
    return (alpha * g + (1 - alpha) * p) / scales["blend"]


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
    cv_out = []   # (sub, raw) — scales need ALL val races before blending
    for rid in val_ids:
        sub = train[train["race_id_short"] == rid]
        train_minus = train[train["race_id_short"] != rid]
        m_cv = GBMEnsemble()
        m_cv.fit(train_minus, n_estimators=cv_ne, **tight_reg)
        cv_out.append((sub, m_cv.predict_scores(sub)))
    scales = fit_blend_scales(
        [(raw, sub["pl_effective"].to_numpy()) for sub, raw in cv_out], alpha)
    if verbose and scales:
        print(f"Score spread: global scales gbm={scales['gbm']:.3f} "
              f"pl={scales['pl']:.3f} blend={scales['blend']:.3f}")

    val_races, tt_list = [], []
    for sub, raw in cv_out:
        b = blend_scores(raw, sub["pl_effective"].to_numpy(), alpha, scales)
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
    # Rainout (formula grid): same rule as backtest_oddslogic_v5.run_race —
    # predict with a GBM trained without qualifying-derived features (and
    # without practice features if there was no practice). Formula grid =
    # no qualifying speeds for the field; superspeedways excluded.
    pred_model = model
    try:
        from backtest_oddslogic_v5 import (RAINOUT_NOSTART_MODEL, RAINOUT_DROP_QUAL,
                                           RAINOUT_DROP_PRACTICE)
    except ImportError:
        RAINOUT_NOSTART_MODEL = False
    if RAINOUT_NOSTART_MODEL and track_type != "superspeedway":
        qz = pd.to_numeric(target.get("qual_z"), errors="coerce").fillna(0).abs().sum()
        if qz == 0:
            drop = list(RAINOUT_DROP_QUAL)
            no_prac = (pd.to_numeric(target.get("practice_z"), errors="coerce").fillna(0).abs().sum() == 0
                       and pd.to_numeric(target.get("has_practice_data"), errors="coerce").fillna(0).sum() == 0)
            if no_prac:
                drop += RAINOUT_DROP_PRACTICE
            pred_model = GBMEnsemble(drop_features=drop)
            pred_model.fit(train, n_estimators=n_estimators, **tight_reg)
            if verbose:
                print(f"RAINOUT: formula grid detected -> no-qualifying model "
                      f"({len(drop)} features dropped{', no practice' if no_prac else ''})")
    raw = pred_model.predict_scores(target)
    blended = blend_scores(raw, target["pl_effective"].to_numpy(), alpha, scales)
    if verbose and scales:
        print(f"Target race blend std = {blended.std():.3f} (1.0 = median val race)")
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
