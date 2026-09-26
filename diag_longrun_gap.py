"""Does long-run vs short-run practice speed predict finish beyond what the
model already knows?

gap = practice_10lap_avg_z - practice_best_speed_z
      (+ = car is relatively better over a run than over one lap)

Test: regress finish_pos on the baseline signals the model already has
(start-position expectation, recent form, green-flag pace, single-lap practice
speed), then ask whether adding `gap` reduces error. Reported overall, by track
type, and by individual track. Also checks whether gap matters more at tracks
with high in-race tire falloff (wear index from laptimes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.new_signals import compute_new_signals
from src.features.practice_pace import compute_practice_features
from src.features.tracks import resolve_track_type


def ols_added_value(df: pd.DataFrame, base: list[str], extra: str):
    """Return (coef, t_stat, r2_base, r2_full, n) for adding `extra` to OLS."""
    d = df[base + [extra, "finish_pos"]].dropna()
    n = len(d)
    if n < 60:
        return (np.nan,) * 4 + (n,)
    y = d["finish_pos"].to_numpy(float)
    Xb = np.column_stack([np.ones(n)] + [d[c].to_numpy(float) for c in base])
    Xf = np.column_stack([Xb, d[extra].to_numpy(float)])

    def fit(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return beta, resid

    _, rb = fit(Xb)
    bf, rf = fit(Xf)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2b, r2f = 1 - (rb ** 2).sum() / ss_tot, 1 - (rf ** 2).sum() / ss_tot
    sigma2 = (rf ** 2).sum() / (n - Xf.shape[1])
    cov = sigma2 * np.linalg.pinv(Xf.T @ Xf)
    t = bf[-1] / np.sqrt(cov[-1, -1])
    return bf[-1], t, r2b, r2f, n


def race_wear_index(laptimes: pd.DataFrame) -> pd.Series:
    """Per race: median within-run lap-time slope (sec/lap) on green laps.
    Higher = more tire falloff over a run."""
    out = {}
    for rid, lt in laptimes.groupby("race_id_short"):
        lt = lt[["lap", "driver_id", "lap_time"]].copy()
        lt["lap_time"] = pd.to_numeric(lt["lap_time"], errors="coerce")
        lt = lt.dropna()
        if lt.empty:
            continue
        med = lt.groupby("lap")["lap_time"].median()
        base = float(np.median(med[med <= med.median()]))
        green_laps = med[med <= base * 1.15].index
        g = med.loc[green_laps].sort_index()
        # split into consecutive green runs; slope of field-median lap time
        runs, cur = [], [g.index[0]] if len(g) else []
        for a, b in zip(g.index[:-1], g.index[1:]):
            if b == a + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        if cur: runs.append(cur)
        slopes = []
        for r in runs:
            if len(r) >= 10:
                x = np.array(r, float); yv = g.loc[r].to_numpy(float)
                slopes.append(np.polyfit(x - x.mean(), yv, 1)[0])
        if slopes:
            out[rid] = float(np.median(slopes))
    return pd.Series(out, name="wear")


def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])
    races["tt"] = races.apply(
        lambda r: resolve_track_type(r.get("track_name", ""), r["track_type"]), axis=1)

    prac = compute_practice_features(entries, races)
    prac = prac.merge(entries[["race_id_short", "driver_id", "driver"]],
                      on=["race_id_short", "driver_id"], how="left")
    prac["gap"] = prac["practice_10lap_avg_z"] - prac["practice_best_speed_z"]

    sig = compute_new_signals(entries, races, laptimes)

    e = entries[entries["finish_pos"] > 0].sort_values("date").copy()
    e["avg_finish_10"] = (e.groupby("driver")["finish_pos"]
                          .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()))

    df = (e[["race_id_short", "driver", "finish_pos", "avg_finish_10"]]
          .merge(sig[["race_id_short", "driver", "exp_finish_from_start",
                      "green_pace_pct_10"]], on=["race_id_short", "driver"], how="left")
          .merge(prac[["race_id_short", "driver", "practice_best_speed_z",
                       "practice_10lap_avg_z", "gap"]],
                 on=["race_id_short", "driver"], how="inner")
          .merge(races[["race_id_short", "tt", "track_name"]], on="race_id_short"))

    print(f"Driver-races with practice long/short data: {len(df)} "
          f"across {df['race_id_short'].nunique()} races\n")

    base = ["exp_finish_from_start", "avg_finish_10", "green_pace_pct_10",
            "practice_best_speed_z"]

    def report(label, d):
        c, t, r2b, r2f, n = ols_added_value(d, base, "gap")
        if np.isnan(c):
            print(f"{label:<32} n={n:>5}  (too few)")
            return
        flag = "  <-- signal" if abs(t) >= 2 else ""
        print(f"{label:<32} n={n:>5}  coef={c:+.2f} pos  t={t:+.2f}  "
              f"R2 {r2b:.4f} -> {r2f:.4f}{flag}")

    print("coef < 0 means better long-run-vs-short-run => better (lower) finish\n")
    print("=" * 100)
    report("ALL", df)
    print("=" * 100)
    for tt, d in df.groupby("tt"):
        report(f"type: {tt}", d)
    print("=" * 100)
    for tr, d in df.groupby("track_name"):
        if len(d) >= 100:
            report(f"track: {tr[:24]}", d)

    # Wear interaction
    wear = race_wear_index(laptimes)
    df = df.merge(wear, left_on="race_id_short", right_index=True, how="left")
    if df["wear"].notna().sum() > 200:
        q = df["wear"].quantile([1 / 3, 2 / 3]).to_numpy()
        df["wear_bin"] = np.where(df["wear"] <= q[0], "low wear",
                          np.where(df["wear"] <= q[1], "mid wear", "high wear"))
        print("=" * 100)
        for wb in ["low wear", "mid wear", "high wear"]:
            report(f"tire falloff: {wb}", df[df["wear_bin"] == wb])


if __name__ == "__main__":
    main()
