"""LapRaptor advanced-metric prior features.

LapRaptor computes advanced driver stats (PFAEz, wPFARP, wARP, GR-LR, SS,
cPOMS) that are the sharp-bettor's edge on close matchups but not in NASCAR's
public loop data. We can't compute them from scratch, but we can pull them
from LapRaptor's per-season aggregate tables by track_type.

Feature construction (walk-forward safe):
  For a race in season S at track_type T, feature = weighted average of the
  driver's LapRaptor stats over seasons [S-3, S-1] restricted to track_type T.
  Weights halve with age so most recent season counts most.

The result is a driver-specific "prior" for how good they are at this track
type, informed by up to 3 prior seasons of walk-forward-safe advanced metrics.

Input files: data/raw/lapraptor_{short,intermediate,superspeedway,road,long}.tsv
Format: exact columns copy-pasted from
   https://www.lapraptor.com/drivers/?series=1&track_type=X&report=all_advanced
   &split_seasons=on&season=...

We only need: Driver, Season, PFAEz, wPFARP, wARP, GR-LR, SS, cPOMS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LR_METRIC_COLS = ["PFAEz", "wPFARP", "wARP", "GR-LR", "SS", "cPOMS"]

# Map our track_type buckets to LapRaptor's file suffixes.
TRACK_TYPE_TO_LR = {
    "short": "short",
    "intermediate": "intermediate",
    "superspeedway": "superspeedway",
    "road": "road",
    "unique": "long",   # LR groups Pocono/Indy under "long"
}


def _load_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["driver", "season"] + LR_METRIC_COLS)
    df = pd.read_csv(path, sep="\t")
    keep = ["Driver", "Season"] + [c for c in LR_METRIC_COLS if c in df.columns]
    df = df[keep].rename(columns={"Driver": "driver", "Season": "season"})
    df["driver"] = df["driver"].str.strip()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df


def load_all(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Return one DataFrame per track_type keyed by our internal track_type."""
    out: dict[str, pd.DataFrame] = {}
    for tt, suffix in TRACK_TYPE_TO_LR.items():
        out[tt] = _load_one(raw_dir / f"lapraptor_{suffix}.tsv")
    return out


def compute_lr_prior(
    entries: pd.DataFrame,
    races: pd.DataFrame,
    raw_dir: Path,
    lookback_seasons: int = 3,
) -> pd.DataFrame:
    """For each (race_id_short, driver), compute a recency-weighted trailing
    average of the driver's LapRaptor stats at the target's track_type,
    using ONLY seasons strictly before the race's season.

    Returns columns: driver, race_id_short, lr_prior_pfaez,
        lr_prior_wpfarp, lr_prior_warp, lr_prior_grlr, lr_prior_ss,
        lr_prior_cpoms, lr_prior_seasons_used
    """
    lr_by_type = load_all(raw_dir)
    df = entries.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "track_type" not in df.columns:
        df = df.merge(races[["race_id_short", "track_type"]],
                      on="race_id_short", how="left")
    df["season"] = df["date"].dt.year.astype("Int64")

    def _weighted_prior(driver: str, tt: str, season: int) -> dict:
        lr = lr_by_type.get(tt, pd.DataFrame())
        if lr.empty:
            return {}
        prior = lr[
            (lr["driver"] == driver)
            & (lr["season"].between(season - lookback_seasons, season - 1))
        ]
        if prior.empty:
            return {}
        # Weight = 2 ** (season - (S - lookback_seasons)) so most recent has
        # highest weight. Normalize.
        wts = np.array([
            2.0 ** (int(s) - (season - lookback_seasons))
            for s in prior["season"]
        ], dtype=float)
        wts /= wts.sum()

        out = {}
        for col in LR_METRIC_COLS:
            if col in prior.columns:
                vals = pd.to_numeric(prior[col], errors="coerce").to_numpy()
                mask = ~np.isnan(vals)
                if mask.sum() == 0:
                    continue
                w = wts[mask]
                w /= w.sum()
                out[col] = float(np.dot(w, vals[mask]))
        out["_seasons_used"] = int(prior["season"].nunique())
        return out

    rows = []
    for _, r in df[["race_id_short", "driver", "track_type", "season"]].drop_duplicates().iterrows():
        p = _weighted_prior(r["driver"], r["track_type"], int(r["season"]))
        rows.append({
            "race_id_short": r["race_id_short"],
            "driver": r["driver"],
            "lr_prior_pfaez": p.get("PFAEz", np.nan),
            "lr_prior_wpfarp": p.get("wPFARP", np.nan),
            "lr_prior_warp": p.get("wARP", np.nan),
            "lr_prior_grlr": p.get("GR-LR", np.nan),
            "lr_prior_ss": p.get("SS", np.nan),
            "lr_prior_cpoms": p.get("cPOMS", np.nan),
            "lr_prior_seasons_used": p.get("_seasons_used", 0),
        })
    return pd.DataFrame(rows)
