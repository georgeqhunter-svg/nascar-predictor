"""Decompose start position into 'track position' vs 'fast car' signal.

Qualified races:  finish ~ start + qual_speed_z + form   (start coef = pure positional value)
Formula races:    finish ~ start + form                  (start has no speed info)
All on per-race percentile scale (0 = front, 1 = back). Superspeedways + dirt excluded.
"""
import numpy as np, pandas as pd

e = pd.read_parquet("data/processed/entries.parquet")
r = pd.read_parquet("data/processed/races.parquet")
e = e.merge(r[["race_id_short", "track_name"]], on="race_id_short", how="left")
e = e[(e.start_pos > 0) & (e.finish_pos > 0)].copy()
e["date"] = pd.to_datetime(e["date"])
e = e.sort_values(["date", "race_id_short"])

n = e.groupby("race_id_short").finish_pos.transform("count")
e["fin"] = (e.finish_pos - 1) / (n - 1)
e["st"] = (e.start_pos - 1) / (n - 1)
qs = e.qual_speed.where(e.qual_speed > 0)
g = qs.groupby(e.race_id_short)
e["qz"] = -(qs - g.transform("mean")) / g.transform("std")   # + = slower (same sign as start)
e["formula"] = e.race_id_short.map(g.apply(lambda s: s.notna().mean() < 0.2))
# form: driver's mean finish pct over prior 10 races (walk-forward)
e["form"] = e.groupby("driver").fin.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
ss = e.track_name.str.contains("Daytona|Talladega|Atlanta|Dirt", na=False)
e = e[~ss & e.form.notna() & (e.race_id_short != "2026-5628")]

def ols(d, cols):
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
    y = d.fin.values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    cov = np.linalg.inv(X.T @ X) * res.var(ddof=X.shape[1])
    r2 = 1 - res.var() / y.var()
    return {c: (b[i+1], np.sqrt(cov[i+1, i+1])) for i, c in enumerate(cols)}, r2

Q = e[~e.formula & e.qz.notna()]
F = e[e.formula]
print(f"qualified rows {len(Q)} ({Q.race_id_short.nunique()} races) | formula rows {len(F)} ({F.race_id_short.nunique()} races)")
print(f"corr(start, qual_speed) in qualified races: {Q[['st','qz']].corr().iloc[0,1]:.3f}")
for name, d, cols in [
    ("Qualified: start + form",        Q, ["st", "form"]),
    ("Qualified: qual_speed + form",   Q, ["qz", "form"]),
    ("Qualified: start + qspeed + form", Q, ["st", "qz", "form"]),
    ("Formula:   start + form",        F, ["st", "form"]),
    ("Formula:   form only",           F, ["form"]),
]:
    co, r2 = ols(d, cols)
    print(f"{name:34s} R2={r2:.3f}  " + "  ".join(f"{k}={v[0]:+.3f}±{v[1]:.3f}" for k, v in co.items()))

# How much does the formula grid's start pos agree with prior form?
print(f"\ncorr(start, form) qualified: {Q[['st','form']].corr().iloc[0,1]:.3f}  formula: {F[['st','form']].corr().iloc[0,1]:.3f}")

# Honest split: estimate on <=2025 formula races only (2026 ones are in the backtest).
print("\n-- pre-2026 only (for an untuned weight) --")
for name, d in [("Qualified <=2025", Q[Q.date.dt.year <= 2025]), ("Formula   <=2025", F[F.date.dt.year <= 2025]),
                ("Formula   2026", F[F.date.dt.year == 2026])]:
    co, r2 = ols(d, ["st", "form"])
    print(f"{name:18s} n_races={d.race_id_short.nunique():3d}  st={co['st'][0]:+.3f}±{co['st'][1]:.3f}  form={co['form'][0]:+.3f}±{co['form'][1]:.3f}")
