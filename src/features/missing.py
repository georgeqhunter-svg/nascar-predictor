"""Missing weekend data -> NaN instead of 0.0 (audit item #3).

The qualifying / practice z-score helpers fill a missing driver (or a whole
race with no session data) with 0.0 — which the GBM reads as "exactly average
car". Cases where that's wrong:
  * LapRaptor practice laps only exist from ~mid-2024: every 2022-23 and
    early-2024 race shows the whole field at 0.0 on the practice_* features.
  * Rainouts: qual_z / practice_z are 0.0 for the whole field.
  * Individual drivers who skipped a session or had no clean lap.
LightGBM handles NaN natively (learns a separate branch for "missing"), so
setting these to NaN lets the model tell "no information" from "average".

Applied as a post-process on the finished feature frame so the per-feature
helpers stay untouched. Availability is judged from the raw session tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Tested 2026-10-06: Δ +0.0285 -> +0.0294; paired +0.9 bp (SE 1.2, t=+0.75),
# 9/27 races better. Rainouts mixed (Charlotte/Nashville better, others worse). Off.
MISSING_AS_NAN = False
MIN_DRIVERS = 3         # same threshold the z-score helpers use


def _available(tbl: pd.DataFrame, speed_col: str) -> tuple[set, set]:
    """(race, driver) pairs with a real speed, and races with >= MIN_DRIVERS of them."""
    if tbl is None or tbl.empty or speed_col not in tbl.columns:
        return set(), set()
    ok = tbl[pd.to_numeric(tbl[speed_col], errors="coerce") > 0]
    pairs = set(zip(ok["race_id_short"], ok["driver_name"]))
    cnt = ok.groupby("race_id_short").size()
    races = set(cnt.index[cnt >= MIN_DRIVERS])
    return pairs, races


def apply_missing_as_nan(out: pd.DataFrame, qual: pd.DataFrame, practice: pd.DataFrame,
                         practice_depth: pd.DataFrame) -> pd.DataFrame:
    if not MISSING_AS_NAN or out.empty:
        return out
    out = out.copy()
    key = list(zip(out["race_id_short"], out["driver"]))
    rid = out["race_id_short"]

    q_pairs, q_races = _available(qual, "qual_speed")
    p_pairs, p_races = _available(practice, "practice_speed")
    has_q = np.array([(k in q_pairs) and (k[0] in q_races) for k in key])
    has_p = np.array([(k in p_pairs) and (k[0] in p_races) for k in key])

    out.loc[~has_q, "qual_z"] = np.nan
    out.loc[~rid.isin(q_races), "team_teammate_qual_z"] = np.nan
    out.loc[~has_p, ["practice_z", "practice_gap_z", "practice_laps_z"]] = np.nan
    out.loc[~rid.isin(p_races), "team_teammate_practice_z"] = np.nan

    # LapRaptor practice: `has_practice_data` is 1 only when the driver had a
    # matched LapRaptor practice row; otherwise build_features wrote 0.0.
    lr_cols = ["practice_best_speed_z", "practice_5lap_avg_z", "practice_10lap_avg_z",
               "practice_consistency_z", "practice_laps_run_z", "practice_longrun_gap"]
    if "has_practice_data" in out.columns:
        no_lr = out["has_practice_data"].fillna(0).astype(int) == 0
        out.loc[no_lr, [c for c in lr_cols if c in out.columns]] = np.nan

    out.attrs["missing_as_nan"] = {
        "qual_missing": float((~has_q).mean()),
        "practice_missing": float((~has_p).mean()),
        "lapraptor_missing": float(no_lr.mean()) if "has_practice_data" in out.columns else None,
    }
    return out
