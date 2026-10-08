"""Track-level profiles WITHIN track type (walk-forward, emit-before-ingest).

Track type is too coarse: among intermediates, field DNF rate runs from
Kansas 8.7% to Charlotte 22.6%; start->finish Spearman from Texas 0.21 to
Pocono 0.50. These columns let the model and the hazard layer see the
specific track, shrunk toward the (walk-forward) track-type mean because most
tracks have only 3-10 Next Gen races.

Columns (one row per race x driver; same value for the whole field):
  type_dnf_base             walk-forward mean field DNF fraction at this type
  track_dnf_base            same at this track, shrunk to type (K_DNF races)
  type_damage_base          walk-forward damaged-but-finished fraction at type
  track_damage_base         same at this track, shrunk to type (K_DAMAGE races)
  start_stickiness_at_track Spearman(start, finish) at this track, qualified
                            races only, shrunk to the type mean (K_STICK races)

Damage = finished (not DNF) more than 12 spots behind qualifying position,
the same definition fit_damage.py used for DAMAGE_HAZARD.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .tracks import resolve_track_type

K_DNF = 4.0
K_DAMAGE = 4.0
K_STICK = 4.0
DAMAGE_GAP = 12


def _shrink(vals: list[float], prior: float, k: float) -> float:
    if not np.isfinite(prior):
        return float(np.mean(vals)) if vals else np.nan
    return (float(np.sum(vals)) + k * prior) / (len(vals) + k)


def compute_track_profile(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    r = races[["race_id_short", "date", "track_name"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values(["date", "race_id_short"], kind="stable")
    r["tt"] = r["track_name"].astype(str).map(resolve_track_type)

    e = entries.copy()
    for c in ("finish_pos", "start_pos", "qual_pos", "qual_speed"):
        if c in e.columns:
            e[c] = pd.to_numeric(e[c], errors="coerce")
    by_race = {rid: g for rid, g in e.groupby("race_id_short")}

    trk_dnf, trk_dmg, trk_rho = defaultdict(list), defaultdict(list), defaultdict(list)
    typ_dnf, typ_dmg, typ_rho = defaultdict(list), defaultdict(list), defaultdict(list)

    rows = []
    for rr in r.itertuples(index=False):
        g = by_race.get(rr.race_id_short)
        if g is None:
            continue
        tname, tt = str(rr.track_name), rr.tt
        t_dnf = float(np.mean(typ_dnf[tt])) if typ_dnf[tt] else np.nan
        t_dmg = float(np.mean(typ_dmg[tt])) if typ_dmg[tt] else np.nan
        t_rho = float(np.mean(typ_rho[tt])) if typ_rho[tt] else np.nan
        prof = {
            "type_dnf_base": t_dnf,
            "track_dnf_base": _shrink(trk_dnf[tname], t_dnf, K_DNF),
            "type_damage_base": t_dmg,
            "track_damage_base": _shrink(trk_dmg[tname], t_dmg, K_DAMAGE),
            "start_stickiness_at_track": _shrink(trk_rho[tname], t_rho, K_STICK),
        }
        # --- emit (pre-race) ---
        for d in g["driver"]:
            rows.append({"race_id_short": rr.race_id_short, "driver": d, **prof})

        # --- ingest (post-race), only if the race has been run ---
        ran = g[g["finish_pos"] > 0]
        if len(ran) < 10:
            continue
        dnf = ran["is_dnf"].astype(bool)
        frac_dnf = float(dnf.mean())
        q = ran["qual_pos"] if "qual_pos" in ran.columns else ran["start_pos"]
        q = q.where(q > 0, ran["start_pos"])
        ok = q > 0
        dmg = (~dnf) & ok & (ran["finish_pos"] > q + DAMAGE_GAP)
        frac_dmg = float(dmg.sum() / max(ok.sum(), 1))
        trk_dnf[tname].append(frac_dnf); typ_dnf[tt].append(frac_dnf)
        trk_dmg[tname].append(frac_dmg); typ_dmg[tt].append(frac_dmg)
        # Stickiness only from QUALIFIED grids (formula grids carry no speed
        # info and would understate how much track position matters).
        qs = ran["qual_speed"] if "qual_speed" in ran.columns else pd.Series(dtype=float)
        if (qs > 0).mean() >= 0.5 and (ran["start_pos"] > 0).sum() >= 10:
            sub = ran[ran["start_pos"] > 0]
            rho = sub["start_pos"].rank().corr(sub["finish_pos"].rank())
            if np.isfinite(rho):
                trk_rho[tname].append(float(rho)); typ_rho[tt].append(float(rho))
    return pd.DataFrame(rows)
