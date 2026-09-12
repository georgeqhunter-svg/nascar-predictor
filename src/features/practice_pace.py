"""Forward-looking practice-pace features from LapRaptor practice CSVs.

Each CSV row = one lap update. We compute per-driver features from the
lap-level data directly:

  practice_best_lap_speed   — max of last_lap_speed over green laps
  practice_5lap_avg_speed   — best 5-lap consecutive green-flag average
  practice_10lap_avg_speed  — best 10-lap window (long-run pace)
  practice_lap_consistency  — std of green-flag lap speeds (lower = more
                              consistent car)
  practice_laps_run         — total green laps completed

These get within-race z-scored so higher = faster/more consistent regardless
of track type.

Matching to our races.parquet is by race_date (from lap_timestamp).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRACTICE_DIR = Path("data/raw/practice_logs")

# We only need a subset of columns per row.
NEEDED_COLS = [
    "series", "race_id", "run_name", "driver_id",
    "laps_completed", "last_lap_speed", "last_lap_categorization",
    "lap_timestamp",
]
CUP_SERIES_ID = 1  # Xfinity (2) and Trucks (3) share weekends w/ Cup — filter out.


def _lap_features_for_driver(sub: pd.DataFrame) -> dict:
    """Given all lap rows for one driver in one session, compute pace features."""
    # Filter to green-flag laps only.
    green = sub[
        (sub["last_lap_categorization"] == "at_speed")
        & sub["last_lap_speed"].notna()
        & (sub["last_lap_speed"] > 0)
    ].copy()
    if len(green) < 3:
        return {}
    speeds = green["last_lap_speed"].to_numpy(dtype=float)

    best5 = np.nan
    best10 = np.nan
    if len(speeds) >= 5:
        # Best 5-lap consecutive window average.
        conv = np.convolve(speeds, np.ones(5) / 5, mode="valid")
        best5 = float(np.max(conv))
    if len(speeds) >= 10:
        conv = np.convolve(speeds, np.ones(10) / 10, mode="valid")
        best10 = float(np.max(conv))

    return {
        "practice_best_lap_speed": float(np.max(speeds)),
        "practice_5lap_avg_speed": best5,
        "practice_10lap_avg_speed": best10,
        "practice_lap_consistency": float(np.std(speeds)),
        "practice_laps_run": int(len(speeds)),
    }


def _load_one(path: Path) -> pd.DataFrame:
    """Load one CSV, return per-driver session-summary features."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in NEEDED_COLS, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or "run_name" not in df.columns:
        return pd.DataFrame()
    # Cup Series only.
    if "series" in df.columns:
        df = df[df["series"] == CUP_SERIES_ID]
        if df.empty:
            return pd.DataFrame()
    # Practice sessions only.
    df = df[df["run_name"].str.contains("Practice", case=False, na=False)]
    if df.empty:
        return pd.DataFrame()

    # Race date from lap_timestamp (first row).
    df["lap_timestamp"] = pd.to_datetime(df["lap_timestamp"], errors="coerce", utc=True)
    race_date = df["lap_timestamp"].dt.date.iloc[0] if len(df) else None
    if race_date is None:
        return pd.DataFrame()
    race_id_lr = df["race_id"].iloc[0]

    rows = []
    for drv_id, sub in df.groupby("driver_id"):
        feats = _lap_features_for_driver(sub)
        if not feats:
            continue
        rows.append({"race_id_lr": race_id_lr, "race_date": race_date,
                     "driver_id": drv_id, **feats})
    return pd.DataFrame(rows)


def _zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 3:
        return pd.Series(np.zeros(len(series)), index=series.index)
    mu, sigma = valid.mean(), valid.std()
    if sigma < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - mu) / sigma).fillna(0)


def compute_practice_features(
    entries: pd.DataFrame, races: pd.DataFrame
) -> pd.DataFrame:
    if not PRACTICE_DIR.exists():
        return pd.DataFrame()

    per_session = []
    for path in PRACTICE_DIR.glob("*.csv"):
        one = _load_one(path)
        if not one.empty:
            per_session.append(one)
    if not per_session:
        return pd.DataFrame()
    all_prac = pd.concat(per_session, ignore_index=True)

    # If a race has multiple practice sessions, prefer the LAST one for each driver.
    all_prac = all_prac.sort_values("race_date").groupby(
        ["race_id_lr", "driver_id"], as_index=False
    ).tail(1)

    # Match race_id_lr to race_id_short by date proximity (practice runs 1-3
    # days before the race).
    races_ref = races[["race_id_short", "date"]].copy()
    races_ref["date"] = pd.to_datetime(races_ref["date"]).dt.date
    race_dates = races_ref.set_index("race_id_short")["date"].to_dict()

    def match(prac_date):
        best_rid, best_diff = None, 99
        for rid, rdate in race_dates.items():
            d = (rdate - prac_date).days
            if 0 <= d <= 5 and d < best_diff:  # race must be 0-5 days AFTER practice
                best_rid, best_diff = rid, d
        return best_rid

    all_prac["race_id_short"] = all_prac["race_date"].apply(match)
    matched = all_prac.dropna(subset=["race_id_short"])

    # Within-race z-scores.
    rows = []
    for rid_short, sub in matched.groupby("race_id_short"):
        z_best = _zscore(sub["practice_best_lap_speed"])
        z_5 = _zscore(sub["practice_5lap_avg_speed"])
        z_10 = _zscore(sub["practice_10lap_avg_speed"])
        z_consistency = -_zscore(sub["practice_lap_consistency"])  # lower better -> negate
        z_laps = _zscore(sub["practice_laps_run"])
        for i, drv_id in enumerate(sub["driver_id"].values):
            rows.append({
                "race_id_short": rid_short,
                "driver_id": int(drv_id),
                "practice_best_speed_z": float(z_best.iloc[i]),
                "practice_5lap_avg_z": float(z_5.iloc[i]),
                "practice_10lap_avg_z": float(z_10.iloc[i]),
                "practice_consistency_z": float(z_consistency.iloc[i]),
                "practice_laps_run_z": float(z_laps.iloc[i]),
                "has_practice_data": 1,
            })
    return pd.DataFrame(rows)
