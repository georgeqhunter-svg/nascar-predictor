"""Granular: where, how and why the market beats the model (fast baseline +0.0195).
Part A  where:  how concentrated is the excess loss?
Part B  who:    which drivers, in which direction (model over/underrates)?
Part C  why:    for each driver signal, how much does the MODEL lean on it
                relative to the market, vs how much the OUTCOME rewards it
                relative to the market. Model leans > truth leans => overweighted."""
import numpy as np, pandas as pd, statsmodels.api as sm
import backtest_oddslogic_v5 as bt
from src.features.tracks import resolve_track_type

d = pd.read_parquet("data/processed/backtest_matchups_fastbase_0195.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
p = d.p_a_raw.clip(1e-4, 1-1e-4); y = d.outcome_a.astype(int)
d["x"] = -(y*np.log(p)+(1-y)*np.log(1-p)) + (y*np.log(d.m)+(1-y)*np.log(1-d.m))
lg = lambda q: np.log(q/(1-q))
d["gap"] = lg(p) - lg(d.m)                 # + = model likes driver a more than market

# ---------------- Part A: concentration ----------------
xs = d.x.sort_values(ascending=False).to_numpy(); tot = xs.sum()
print(f"[A] total excess {tot:.1f} over {len(xs)} matchups (mean {tot/len(xs):+.4f})")
for q in (0.05, 0.10, 0.20):
    k = int(len(xs)*q); print(f"    worst {q:.0%} of matchups ({k}) = {xs[:k].sum()/tot:.0%} of total excess")
print(f"    matchups where model beat market: {(d.x<0).mean():.0%};  sum of model wins {xs[xs<0].sum():.1f}, sum of losses {xs[xs>0].sum():.1f}")
d["absgap"] = d.gap.abs()
d["gapbin"] = pd.cut(d.absgap, [0, .2, .4, .7, 5], labels=["<0.2", "0.2-0.4", "0.4-0.7", ">0.7"])
print("    by size of model-vs-market disagreement (logit units):")
print(d.groupby("gapbin", observed=True).agg(n=("x", "size"), excess_sum=("x", "sum"), excess_mean=("x", "mean")).round(3).to_string())

# ---------------- signals per driver-race (walk-forward) ----------------
R = pd.read_parquet("data/processed/races.parquet"); R["date"] = pd.to_datetime(R.date)
E = pd.read_parquet("data/processed/entries.parquet"); E["date"] = pd.to_datetime(E.date)
L = pd.read_parquet("data/processed/loopstats.parquet")
S = pd.read_parquet("data/processed/sessions.parquet")
d["rid"] = d.race_idx.map({i: R[R.date == pd.Timestamp(dt)].race_id_short.iloc[0] for i, (dt, _, _) in enumerate(bt.RACES)})
E = E.merge(R[["race_id_short", "track_name"]], on="race_id_short", how="left")
E["tt"] = E.track_name.map(resolve_track_type)
E = E.merge(L[["race_id_short", "driver_id", "avg_ps"]], on=["race_id_short", "driver_id"], how="left")
E = E.sort_values(["date", "race_id_short"])
ran = E.finish_pos > 0
n = E[ran].groupby("race_id_short").finish_pos.transform("count")
E.loc[ran, "fp"] = (E.loc[ran, "finish_pos"] - 1) / (n - 1)
E.loc[ran, "ps"] = (E.loc[ran, "avg_ps"] - 1) / (n - 1)
E["dnf"] = E.is_dnf.astype(float)
E["fp_run"] = E.fp.where(E.dnf == 0)
g = E.groupby("driver")
roll = lambda c, w: g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
E["form_fin6"] = roll("fp", 6); E["form_run6"] = roll("fp_run", 6); E["speed_ps6"] = roll("ps", 6)
E["form_fin20"] = roll("fp", 20); E["dnf20"] = roll("dnf", 20)
E["career"] = g.cumcount()
E["trk_hist"] = E.groupby(["driver", "track_name"]).fp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
E["type_hist"] = E.groupby(["driver", "tt"]).fp.transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
E["start"] = pd.to_numeric(E.start_pos, errors="coerce")
q = S[(S.run_type_label == "qualifying") & (S.best_lap_speed > 0)].groupby(["race_id_short", "driver_name"]).best_lap_speed.max()
qz = q.groupby(level=0).transform(lambda v: -(v - v.mean())/v.std())        # + = slower (same sign as positions)
pr = S[(S.run_type_label == "practice") & (S.best_lap_speed > 0)].groupby(["race_id_short", "driver_name"]).best_lap_speed.max()
pz = pr.groupby(level=0).transform(lambda v: -(v - v.mean())/v.std())
K = E.set_index(["race_id_short", "driver"])
sig_cols = {"recent finishes (6)": "form_fin6", "recent finishes, running only (6)": "form_run6",
            "recent running position (6)": "speed_ps6", "longer-term finishes (20)": "form_fin20",
            "DNF rate (20)": "dnf20", "track history": "trk_hist", "track-type history": "type_hist",
            "start position": "start", "experience (Cup starts)": "career"}
X = {}
for lab, c in sig_cols.items():
    va = np.array([K[c].get((r, a), np.nan) for r, a in zip(d.rid, d.a)], float)
    vb = np.array([K[c].get((r, b), np.nan) for r, b in zip(d.rid, d.b)], float)
    X[lab] = va - vb
X["qualifying speed"] = np.array([qz.get((r, a), np.nan) for r, a in zip(d.rid, d.a)]) - np.array([qz.get((r, b), np.nan) for r, b in zip(d.rid, d.b)])
X["practice speed"] = np.array([pz.get((r, a), np.nan) for r, a in zip(d.rid, d.b*0 + d.a)]) - np.array([pz.get((r, b), np.nan) for r, b in zip(d.rid, d.b)])
X = pd.DataFrame(X, index=d.index)
# orient every signal so + = "driver a looks BETTER" (positions/rates: lower is better)
for c in X.columns:
    if c not in ("experience (Cup starts)",):
        X[c] = -X[c]

# ---------------- Part C: model lean vs truth lean, one signal at a time ----------------
print("\n[C] per signal (standardized gap, + = a looks better):")
print(f"    {'signal':36s} {'n':>4s}  model leans beyond mkt   outcome rewards beyond mkt   verdict")
for c in X.columns:
    ok = X[c].notna() & np.isfinite(X[c]) & (X[c] != 0)
    z = (X.loc[ok, c] - X.loc[ok, c].mean()) / X.loc[ok, c].std(); grp = d.race_idx[ok]
    f1 = sm.OLS(d.gap[ok], sm.add_constant(z)).fit(cov_type="cluster", cov_kwds={"groups": grp})
    f2 = sm.Logit(y[ok], sm.add_constant(pd.DataFrame({"mkt": lg(d.m[ok]), "z": z}))).fit(disp=0, cov_type="cluster", cov_kwds={"groups": grp})
    a, ta, b_, tb = f1.params.iloc[1], f1.tvalues.iloc[1], f2.params["z"], f2.tvalues["z"]
    verdict = ""
    if ta > 2 and tb < 1: verdict = "MODEL OVERWEIGHTS"
    elif ta < -2 and tb > -1: verdict = "MODEL UNDERWEIGHTS"
    elif tb > 2: verdict = "market underweights"
    print(f"    {c:36s} {ok.sum():4d}  {a:+.3f} (t={ta:+.1f})           {b_:+.3f} (t={tb:+.1f})            {verdict}")
d.join(X).to_parquet("data/processed/_mvm_tmp.parquet")
