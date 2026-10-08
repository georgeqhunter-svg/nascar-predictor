"""DNF-neutral finishes for driver-form features.

Problem (diag_driver_misses.py, 2026-10-05): DNFs are counted TWICE.
  1. Finish-based form features (avg_finish_*, at_type, at_track, h2h, ...)
     treat a DNF as a 30th-40th place finish -> lower GBM score.
  2. The sampler's per-driver DNF hazard (dnf_rate_at_type_10, crash rate)
     then applies DNF risk again on top.
Ryan Blaney 2025-26: 20% DNF (11 of 13 accidents), avg finish 13.6 but 8.2
when running (best in field; Penske teammates 13.5 / 14.7 when running).
Model underrated him ~10 pts at our worst races; he won 22/25 matchups there.

Fix: for FORM features only, replace a DNF's finish_pos with the driver's
walk-forward mean finish over their last N RUNNING races (strictly prior), so
form measures speed and DNF risk lives only in the hazard layer. The real
finish_pos is untouched for the target, PL ratings, is_dnf-based hazards and
everything else.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Tested 2026-10-05, killed after 13/25 races: +3.7 bps worse on 546 matchups
# vs the stickiness run. Atlanta (Blaney's worst miss) got WORSE (+2.3 bps),
# both superspeedways worse -> his misses there aren't a DNF double-count.
DNF_NEUTRAL_FORM = False
DNF_NEUTRAL_WINDOW = 10
DNF_NEUTRAL_MIN = 3


def neutralize_dnf_finishes(entries: pd.DataFrame) -> pd.DataFrame:
    e = entries.copy()
    e["finish_pos_raw"] = e["finish_pos"]
    order = (e.groupby("race_id_short")["date"].first()
             .sort_values(kind="stable").index.tolist())
    by_race = e.groupby("race_id_short").groups
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=DNF_NEUTRAL_WINDOW))
    fp = pd.to_numeric(e["finish_pos"], errors="coerce")
    dnf = e["is_dnf"].astype(bool)
    new_fp = fp.copy()
    n_changed = 0
    for rid in order:
        idx = by_race[rid]
        field = int((fp.loc[idx] > 0).sum())
        # emit: impute DNFs from PRIOR running finishes only
        for i in idx:
            if dnf.loc[i] and fp.loc[i] > 0:
                h = hist[e.at[i, "driver"]]
                if len(h) >= DNF_NEUTRAL_MIN:
                    # >= 2 so an imputed DNF can never count as a win
                    new_fp.loc[i] = int(np.clip(round(float(np.mean(h))), 2, max(field, 2)))
                    n_changed += 1
        # ingest: this race's running finishes
        for i in idx:
            if not dnf.loc[i] and fp.loc[i] > 0:
                hist[e.at[i, "driver"]].append(float(fp.loc[i]))
    e["finish_pos"] = new_fp.fillna(0).astype(int)
    e.attrs["dnf_neutral_changed"] = n_changed
    return e
