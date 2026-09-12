"""Tire-degradation features inferred from race lap-times.

Between pit stops for each driver = a green-flag stint. Within a stint,
the tire loses grip lap by lap, so lap times drift upward. The slope of
that drift is the driver's tire-management skill relative to the field's
skill on the same rubber compound:

  stint_pace_decay = slope of linear fit through (lap_index_in_stint, lap_time)
                     on GREEN-FLAG laps only. Positive = losing time.
  stint_pace_retention = median lap in final 25% of stint / best lap in stint.
                     >1 always; closer to 1 = holds pace better.

Per race, we average over all stints of >= 8 green laps each. Roll per
driver over the last N races of the SAME track_type (tire deg varies a lot
by track — Bristol vs Talladega are different worlds).

Caveats:
- We filter to laps within [0.90, 1.06] × field median for that lap to
  exclude cautions, spins, and traffic-slowed laps.
- Superspeedways have almost no tire deg, so slopes there are near zero
  and mostly noise; feature will still be built but expected impact is on
  intermediates/short/road tracks.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


PIT_RATIO_THRESHOLD = 1.4    # matches pit.py
GREEN_MIN_RATIO = 0.90       # laps below this vs field median are anomalous (draft/tow)
GREEN_MAX_RATIO = 1.06       # laps above this are cautions or heavy traffic
MIN_STINT_LEN = 8            # green laps needed to fit a stint slope

# Track types where the per-stint slope is signal, not noise. Diagnosed
# empirically: rho(tire_decay, finish_pos) = -0.30 on short and -0.13 on
# intermediate, but ~0 on road/superspeedway/unique (draft/traffic noise).
SIGNAL_TRACK_TYPES: set[str] = {"short", "intermediate"}


def per_race_tire_stats(laptimes_one_race: pd.DataFrame) -> pd.DataFrame:
    """Compute per-driver tire-deg stats for one race."""
    lt = laptimes_one_race.copy()
    lt["lap_time"] = pd.to_numeric(lt["lap_time"], errors="coerce")
    lt = lt.dropna(subset=["lap_time", "lap"])
    if lt.empty:
        return pd.DataFrame(columns=["driver_id", "n_stints",
                                     "pace_decay", "pace_retention"])

    lap_median = lt.groupby("lap")["lap_time"].median()

    stats_rows = []
    for drv, sub in lt.groupby("driver_id"):
        sub = sub.sort_values("lap")
        laps = sub["lap"].to_numpy()
        times = sub["lap_time"].to_numpy()
        # field median at each of this driver's laps
        med = lap_median.reindex(laps).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where((med > 0) & np.isfinite(med), times / med, np.nan)

        # Find stint boundaries at pit laps (ratio > pit threshold).
        pit_flags = ratio > PIT_RATIO_THRESHOLD

        # Iterate stints between pit laps.
        stint_slopes = []
        stint_retentions = []
        stint_starts = [0] + [i + 1 for i, p in enumerate(pit_flags) if p]
        stint_ends = [i for i, p in enumerate(pit_flags) if p] + [len(laps)]
        for s, e in zip(stint_starts, stint_ends):
            if e <= s:
                continue
            stint_ratio = ratio[s:e]
            stint_time = times[s:e]
            green_mask = (stint_ratio >= GREEN_MIN_RATIO) & (stint_ratio <= GREEN_MAX_RATIO)
            green_lap_idx = np.arange(e - s)[green_mask]
            green_time = stint_time[green_mask]
            if len(green_lap_idx) < MIN_STINT_LEN:
                continue
            # Linear fit: time = a + b * lap_index_in_stint
            b = float(np.polyfit(green_lap_idx.astype(float), green_time, 1)[0])
            stint_slopes.append(b)
            # Retention: median of final 25% vs best in stint.
            cut = int(len(green_time) * 0.75)
            best = float(np.min(green_time))
            late = float(np.median(green_time[cut:])) if cut < len(green_time) else best
            if best > 0:
                stint_retentions.append(late / best)

        if stint_slopes:
            stats_rows.append({
                "driver_id": int(drv) if pd.notna(drv) else None,
                "n_stints": len(stint_slopes),
                "pace_decay": float(np.mean(stint_slopes)),
                "pace_retention": float(np.mean(stint_retentions)) if stint_retentions else np.nan,
            })
    return pd.DataFrame(stats_rows).dropna(subset=["driver_id"])


def compute_rolling_tire(
    laptimes: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Walk-forward per (race_id_short, driver_id):

      tire_decay_type_10   — mean per-stint slope over the driver's last 10
                             races at the SAME track_type as this race.
      tire_retention_type_10 — same, for retention.
      tire_races_type       — sample count in that window.

    Rolling values reflect data known BEFORE the race.
    """
    if laptimes.empty:
        return pd.DataFrame(columns=[
            "race_id_short", "driver_id",
            "tire_decay_type_10", "tire_retention_type_10", "tire_races_type",
        ])

    r = races[["race_id_short", "date", "track_type"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values("date")
    tt_by_race = dict(zip(r["race_id_short"], r["track_type"]))
    race_order = r["race_id_short"].tolist()

    # history[(driver_id, track_type)] = deque of (slope, retention)
    history: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=10))

    out_rows = []
    for race_id in race_order:
        tt = tt_by_race.get(race_id)
        lt_race = laptimes[laptimes["race_id_short"] == race_id]
        drivers_this_race = lt_race["driver_id"].dropna().unique()

        # Emit rolling features BEFORE updating with this race.
        # Feature is only informative on short + intermediate; NaN elsewhere
        # so the GBM ignores it via native NaN handling.
        emit = tt in SIGNAL_TRACK_TYPES
        for drv in drivers_this_race:
            drv = int(drv)
            hist = history[(drv, tt)]
            slopes = [s for s, _ in hist]
            rets = [r for _, r in hist if not np.isnan(r)]
            out_rows.append({
                "race_id_short": race_id,
                "driver_id": drv,
                "tire_decay_type_10": (
                    float(np.mean(slopes)) if (slopes and emit) else np.nan
                ),
                "tire_retention_type_10": (
                    float(np.mean(rets)) if (rets and emit) else np.nan
                ),
                "tire_races_type": len(slopes) if emit else 0,
            })

        # Update history with this race's stats.
        stats = per_race_tire_stats(lt_race)
        for _, row in stats.iterrows():
            drv = int(row["driver_id"])
            history[(drv, tt)].append(
                (float(row["pace_decay"]),
                 float(row["pace_retention"]) if pd.notna(row["pace_retention"]) else np.nan)
            )

    return pd.DataFrame(out_rows)
