"""Per-driver damage tendency at the track type, ADJUSTED for qualifying position.

Damage event = finished (not DNF) more than 12 positions behind qualifying —
same definition as fit_damage.py / DAMAGE_HAZARD. Raw damage rates are
dominated by where a driver starts (2022-25: P1-5 24.9%, P6-10 17.8%,
P11-15 13.1%, P16-20 4.5%, P21+ ~1%), so a raw driver rate mostly says "this
driver qualifies up front". We instead track each driver's RESIDUAL: actual
damage minus the walk-forward expected rate for his (track type, qual bucket).
diag_sim_assumptions.py: residual persistence 0.136 +/- 0.052 (raw was 0.27).

Columns (hazard layer only, not GBM features):
  drv_dmg_resid_type_10  mean residual over last 10 races at the type
  drv_dmg_n_type_10      number of those races
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .tracks import resolve_track_type

WINDOW = 10
GAP = 12
QUAL_BINS = [0, 5, 10, 15, 20, 25, 99]


def _qbin(q: float) -> int:
    for i in range(len(QUAL_BINS) - 1):
        if QUAL_BINS[i] < q <= QUAL_BINS[i + 1]:
            return i
    return len(QUAL_BINS) - 2


def compute_driver_damage_rate(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    e = entries.merge(races[["race_id_short", "track_name"]], on="race_id_short", how="left") \
        if "track_name" not in entries.columns else entries.copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values(["date", "race_id_short"], kind="stable")
    e["tt"] = e["track_name"].astype(str).map(resolve_track_type)
    fp = pd.to_numeric(e["finish_pos"], errors="coerce")
    q = pd.to_numeric(e.get("qual_pos"), errors="coerce")
    sp = pd.to_numeric(e.get("start_pos"), errors="coerce")
    e["_q"] = q.where(q > 0, sp)
    e["_fp"] = fp
    e["_dmg"] = ((~e["is_dnf"].astype(bool)) & (fp > 0) & (e["_q"] > 0) & (fp > e["_q"] + GAP)).astype(float)

    hist: dict = defaultdict(lambda: deque(maxlen=WINDOW))
    cnt = defaultdict(lambda: [0.0, 0.0])          # (tt, qbin) -> [damage events, starts]
    rows = []
    for (rid, tt), g in e.groupby(["race_id_short", "tt"], sort=False):
        for drv in g["driver"]:                                    # emit
            h = hist[(drv, tt)]
            rows.append({"race_id_short": rid, "driver": drv,
                         "drv_dmg_resid_type_10": float(np.mean(h)) if h else np.nan,
                         "drv_dmg_n_type_10": len(h)})
        upd = []
        for drv, dmg, qq, f in zip(g["driver"], g["_dmg"], g["_q"], g["_fp"]):   # ingest
            if not (qq > 0 and f > 0):
                continue
            key = (tt, _qbin(float(qq)))
            ev, n = cnt[key]
            if n >= 20:                                             # need a baseline first
                hist[(drv, tt)].append(float(dmg) - ev / n)
            upd.append((key, dmg))
        for key, dmg in upd:                                        # baseline after all residuals
            cnt[key][0] += dmg; cnt[key][1] += 1
    return pd.DataFrame(rows)
