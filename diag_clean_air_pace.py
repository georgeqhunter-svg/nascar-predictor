"""Pre-registered check (2026-10-10): clean-air, tire-age-corrected race pace.
Per race: green laps (<= 1.10 x race fast-lap ref), skip first 2 laps of each
green run, clean air = gap to car ahead > CLEAN_GAP s (or leading). Lap time
relative to the lap's median clean-air time, minus a per-race tire-age slope.
Driver race pace = median relative lap (lower = faster). Feature = mean over
last 5 races. Test on 2022-2025: predicts finish beyond green_pace(5), loop
rating(5), avg running position(5), form(10)?"""
import numpy as np, pandas as pd, statsmodels.api as sm

CLEAN_GAP, GREEN_MAX, SKIP = 1.0, 1.10, 2
L = pd.read_parquet("data/processed/laptimes.parquet")
L["t"] = pd.to_numeric(L.lap_time, errors="coerce")
L = L[(L.lap >= 1) & (L.t > 0)].sort_values(["race_id_short", "driver_id", "lap"])
L["elapsed"] = L.groupby(["race_id_short", "driver_id"]).t.cumsum()
L["pos"] = pd.to_numeric(L.running_pos, errors="coerce")
rows = []
for rid, g in L.groupby("race_id_short", sort=False):
    ref = g.t.quantile(0.10)
    g = g.assign(green=g.t <= GREEN_MAX * ref)
    # green-run age per driver (resets on any non-green lap)
    g = g.sort_values(["driver_id", "lap"])
    brk = (~g.green) | (g.groupby("driver_id").lap.diff() != 1)
    run_id = brk.groupby(g.driver_id).cumsum()
    g["age"] = g.groupby([g.driver_id, run_id]).cumcount()
    # gap to car ahead on the same lap
    ahead = g[["lap", "pos", "elapsed"]].rename(columns={"elapsed": "el_ahead"})
    ahead["pos"] = ahead.pos + 1
    g = g.merge(ahead, on=["lap", "pos"], how="left")
    g["gap"] = g.elapsed - g.el_ahead
    clean = g.green & (g.age >= SKIP) & ((g.pos == 1) | (g.gap > CLEAN_GAP))
    c = g[clean].copy()
    if len(c) < 200: continue
    c["rel"] = c.t / c.groupby("lap").t.transform("median") - 1
    b = np.polyfit(c.age.clip(upper=60), c.rel, 1)[0]              # pooled tire-age slope
    c["rel_adj"] = c.rel - b * (c.age.clip(upper=60) - c.age.clip(upper=60).mean())
    pace = c.groupby("driver_id").rel_adj.agg(["median", "size"])
    pace = pace[pace["size"] >= 15]
    for did, (med, n) in pace.iterrows():
        rows.append((rid, int(did), med * 100, n))
P = pd.DataFrame(rows, columns=["race_id_short", "driver_id", "ca_pace", "ca_laps"])
print(f"clean-air pace for {P.race_id_short.nunique()} races, median {P.ca_laps.median():.0f} clean laps/driver")

E = pd.read_parquet("data/processed/entries.parquet"); R = pd.read_parquet("data/processed/races.parquet")
Lp = pd.read_parquet("data/processed/loopstats.parquet")
E = E[E.finish_pos > 0].merge(R[["race_id_short", "date"]].rename(columns={"date": "d"}), on="race_id_short")
E["date"] = pd.to_datetime(E.d); E = E.sort_values(["date", "race_id_short"])
n = E.groupby("race_id_short").finish_pos.transform("count"); E["fp"] = (E.finish_pos - 1) / (n - 1)
E = E.merge(P, on=["race_id_short", "driver_id"], how="left")
E = E.merge(Lp[["race_id_short", "driver_id", "rating", "avg_ps"]], on=["race_id_short", "driver_id"], how="left")
E["ps"] = (E.avg_ps - 1) / (n - 1)
# per-race green pace percentile (same idea as green_pace_pct): median lap pct rank among green laps
roll = lambda c, w, mp=2: E.groupby("driver").transform(lambda s: s)[c] if False else E.groupby("driver")[c].transform(lambda s: s.shift(1).rolling(w, min_periods=mp).mean())
E["ca5"] = roll("ca_pace", 5); E["rating5"] = roll("rating", 5); E["ps5"] = roll("ps", 5); E["form10"] = roll("fp", 10, 3)
try:
    from src.features.new_signals import compute_green_pace
    gp = compute_green_pace(L.rename(columns={})[["race_id_short", "driver_id", "lap", "lap_time"]] if False else pd.read_parquet("data/processed/laptimes.parquet"),
                            pd.read_parquet("data/processed/entries.parquet"), R)
    E = E.merge(gp[["race_id_short", "driver", "green_pace_pct_5"]], on=["race_id_short", "driver"], how="left")
except Exception as ex:
    print("green_pace unavailable:", ex); E["green_pace_pct_5"] = np.nan
X = E[(E.date.dt.year.between(2022, 2025))]
print("corr(ca5, existing):", X[["ca5", "green_pace_pct_5", "rating5", "ps5", "form10"]].corr().loc["ca5"].round(2).to_dict())
for ctrl in (["form10"], ["form10", "ps5", "rating5"], ["form10", "ps5", "rating5", "green_pace_pct_5"]):
    s = X.dropna(subset=ctrl + ["ca5"])
    Z = s[ctrl + ["ca5"]]; Z = (Z - Z.mean()) / Z.std()
    f = sm.OLS(s.fp, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": s.race_id_short})
    print(f"finish ~ {' + '.join(ctrl)} + clean-air pace(5): ca5 {f.params['ca5']:+.4f} (t={f.tvalues['ca5']:+.2f})  n={len(s)}"
          + "  | " + "  ".join(f"{c} {f.params[c]:+.3f}" for c in ctrl))
P.to_parquet("data/processed/_clean_air_pace.parquet")
