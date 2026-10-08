"""Pseudo-grid for rainout (formula-set) starting lineups.

When practice + qualifying are rained out, NASCAR sets the grid by formula
(mostly prior-race results + owner points). That start position carries NO
information about this weekend's car speed — yet the model learned start_pos
on qualified races, where ~2/3 of its value IS the fast-car signal.

Measured (diag_start_decomp.py, non-SS, finish ~ start + form, pct scale):
    qualified races        start +0.31 ± 0.02   (128 races)
    formula races <=2025   start +0.10 ± 0.08   (7 races)
    formula races 2026     start +0.14 ± 0.08   (5 races)
=> pure track-position value ≈ 1/3 of a qualified start.

Fix: on formula grids, replace start_pos (model input only) with an effective
start, shrunk toward mid-pack (pct scale, 0 = pole):
    eff = 0.5 + W * (start_pct - 0.5) + B * (form_pct - field_mean_form)
    start_pos = 1 + eff * (field - 1)          (continuous, not re-ranked)
Linear matching: a qualified-race model weights start by 0.31; feeding it eff
gives start 0.31*W ≈ 0.10 and extra form 0.31*B ≈ 0.10 — the formula-race
coefficients. W, B are pre-2026 estimates, NOT tuned on the backtest.
The original grid is kept in `start_pos_raw`.

Superspeedways and dirt are left alone: start position matters little there
and their "missing qual speed" races are mostly duel-set / data gaps, not
formula grids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tracks import resolve_track_type

# OFF after 2026-10-01 backtest: Δ +0.0288 (wash). Rainout races net flat
# (N.Wilkesboro/Darlington better, Bristol much worse). Per-race score
# z-scoring in predict_pipeline re-expands the compressed spread, so only the
# reordering survives. Re-test after the score-spread fix.
# 2026-10-02 re-test on top of global score spread: Δ +0.0287 -> +0.0298.
# Same pattern both times: N.Wilkesboro/Darlington ~2 bps better each,
# Bristol ~6 bps worse. Not restricting it to intermediates — that would be
# tuning on one race (Bristol). Off for good unless new rainout data says otherwise.
PSEUDO_GRID_ENABLED = False
PSEUDO_GRID_W = 1 / 3  # start slope: 0.103 / 0.305 (pre-2026 formula vs qualified)
FORM_B = 1 / 3          # form slope: (0.532 - 0.431) / 0.305 (pre-2026)
FORM_WINDOW = 10
FORM_MIN = 3
_SKIP_TYPES = {"superspeedway"}


def formula_grid_races(entries: pd.DataFrame, races: pd.DataFrame) -> set[str]:
    """Race ids whose grid was set by formula (no qualifying speeds at all),
    excluding superspeedways and dirt."""
    if "qual_speed" not in entries.columns:
        return set()
    qs = pd.to_numeric(entries["qual_speed"], errors="coerce")
    has_speed = (qs > 0).groupby(entries["race_id_short"]).mean()
    sp = pd.to_numeric(entries["start_pos"], errors="coerce")
    has_start = (sp > 0).groupby(entries["race_id_short"]).mean()
    cand = set(has_speed.index[(has_speed < 0.2) & (has_start.reindex(has_speed.index) > 0.8)])
    tn = races.set_index("race_id_short")["track_name"].astype(str)
    out = set()
    for rid in cand:
        name = tn.get(rid, "")
        if "Dirt" in name or resolve_track_type(name) in _SKIP_TYPES:
            continue
        out.add(rid)
    return out


def apply_pseudo_grid(entries: pd.DataFrame, races: pd.DataFrame,
                      w: float = PSEUDO_GRID_W) -> pd.DataFrame:
    """Return a copy of entries with start_pos replaced by the pseudo-grid on
    formula-grid races. Adds `start_pos_raw` and `formula_grid` columns."""
    e = entries.copy()
    e["start_pos_raw"] = e["start_pos"]
    # Float so fractional effective starts fit (parquet stores nullable Int64).
    e["start_pos"] = pd.to_numeric(e["start_pos"], errors="coerce").astype(float)
    fg = formula_grid_races(e, races)
    e["formula_grid"] = e["race_id_short"].isin(fg).astype(int)
    if not fg or not PSEUDO_GRID_ENABLED:
        return e

    # Walk-forward form: mean finish pct over the driver's prior N races.
    order = races[["race_id_short", "date"]].copy()
    order["date"] = pd.to_datetime(order["date"])
    tmp = e[["race_id_short", "driver", "finish_pos"]].merge(order, on="race_id_short", how="left")
    fp = pd.to_numeric(tmp["finish_pos"], errors="coerce")
    fp = fp.where(fp > 0)
    n = fp.groupby(tmp["race_id_short"]).transform("count")
    tmp["fin_pct"] = (fp - 1) / (n - 1).where(n > 1)
    tmp = tmp.sort_values(["date", "race_id_short"], kind="stable")
    tmp["form_pct"] = tmp.groupby("driver")["fin_pct"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=FORM_MIN).mean()
    )
    form = tmp.set_index(["race_id_short", "driver"])["form_pct"]

    for rid in fg:
        idx = e.index[(e["race_id_short"] == rid)]
        sp = pd.to_numeric(e.loc[idx, "start_pos"], errors="coerce")
        valid = sp > 0
        if valid.sum() < 2:
            continue
        m = int(valid.sum())
        st_pct = (sp - 1) / (m - 1)
        fpct = pd.Series([form.get((rid, d), np.nan) for d in e.loc[idx, "driver"]], index=idx)
        fpct = fpct.fillna(fpct[valid].mean())  # no history -> field-average form
        # SHRINK, don't re-rank: re-ranking keeps the full P1..P36 spread, so
        # the model would still read it as a strong fast-car signal.
        eff = 0.5 + w * (st_pct - 0.5) + FORM_B * (fpct - fpct[valid].mean())
        eff = eff.clip(0.0, 1.0)
        e.loc[idx[valid], "start_pos"] = (1 + eff[valid] * (m - 1)).values
    return e
