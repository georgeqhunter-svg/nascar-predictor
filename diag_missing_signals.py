"""Candidate signals the model may be missing, tested on the 974 backtest
matchups: does each predict outcomes BEYOND the model and BEYOND the market?
All signals are walk-forward (strictly prior races)."""
import numpy as np, pandas as pd, statsmodels.api as sm
import backtest_oddslogic_v5 as bt
d = pd.read_parquet("data/processed/backtest_matchups_baseline27_0285.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
r = pd.read_parquet("data/processed/races.parquet"); r["date"] = pd.to_datetime(r.date)
d["rid"] = d.race_idx.map({i: r[(r.date == pd.Timestamp(dt))].race_id_short.iloc[0] for i, (dt, _, _) in enumerate(bt.RACES)})

e = pd.read_parquet("data/processed/entries.parquet"); e["date"] = pd.to_datetime(e.date)
ls = pd.read_parquet("data/processed/loopstats.parquet")
e = e.merge(ls[["race_id_short", "driver_id", "rating", "avg_ps", "fast_laps", "top15_laps", "laps"]],
            on=["race_id_short", "driver_id"], how="left")
e = e[e.finish_pos > 0].sort_values(["date", "race_id_short"])
n = e.groupby("race_id_short").finish_pos.transform("count")
e["fin_pct"] = (e.finish_pos - 1) / (n - 1)
e["ps_pct"] = (e.avg_ps - 1) / (n - 1)                    # avg running position, pct
e["t15"] = e.top15_laps / e.laps
# per-race org (team) means of SPEED-type stats
org = (e.groupby(["race_id_short", "date", "team"]).agg(org_rating=("rating", "mean"), org_ps=("ps_pct", "mean"),
                                                         org_fin=("fin_pct", "mean")).reset_index().sort_values("date"))
for c in ("org_rating", "org_ps", "org_fin"):
    for w in (3, 6):
        org[f"{c}_{w}"] = org.groupby("team")[c].transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
e = e.merge(org.drop(columns=["date", "org_rating", "org_ps", "org_fin"]), on=["race_id_short", "team"], how="left")
for c in ("rating", "ps_pct", "fin_pct", "t15"):
    for w in (3, 6):
        e[f"drv_{c}_{w}"] = e.groupby("driver")[c].transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
# luck gap: recent finish worse than recent running position -> "unlucky", expect rebound
e["luck_6"] = e.drv_fin_pct_6 - e.drv_ps_pct_6
k = e.set_index(["race_id_short", "driver"])
sigs = ["drv_rating_3", "drv_rating_6", "drv_ps_pct_3", "drv_ps_pct_6", "drv_fin_pct_6", "drv_t15_6",
        "org_rating_3", "org_rating_6", "org_ps_3", "org_ps_6", "org_fin_6", "luck_6"]
p = d.p_a_raw.clip(1e-4, 1-1e-4); lg = lambda q: np.log(q/(1-q)); y = d.outcome_a
print(f"{'signal (a minus b)':22s} {'n':>4s}  beyond MODEL        beyond MARKET")
for sname in sigs:
    ga = np.array([k[sname].get((rr, a), np.nan) for rr, a in zip(d.rid, d.a)])
    gb = np.array([k[sname].get((rr, b), np.nan) for rr, b in zip(d.rid, d.b)])
    g = pd.Series(ga - gb, index=d.index); ok = g.notna() & (g != 0)
    g = (g[ok] - g[ok].mean()) / g[ok].std()                 # standardized gap
    res = []
    for base in (lg(p), lg(d.m)):
        X = sm.add_constant(pd.DataFrame({"b": base[ok], "g": g}))
        f = sm.Logit(y[ok], X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.race_idx[ok]})
        res.append(f"{f.params['g']:+.3f} (t={f.tvalues['g']:+.2f})")
    print(f"{sname:22s} {ok.sum():4d}  {res[0]:18s}  {res[1]}")

# ---- Is the GBM's complexity just noise? A 4-feature logistic on matchup
# differences, leave-one-race-out across the 27 backtest races. ----
from sklearn.linear_model import LogisticRegression
s = pd.read_parquet("data/processed/sessions.parquet")
q = s[(s.run_type_label == "qualifying") & (s.best_lap_speed > 0)].groupby(["race_id_short", "driver_name"]).best_lap_speed.max()
qz = q.groupby(level=0).transform(lambda v: (v - v.mean()) / v.std())
def side(col, drv): return np.array([k[col].get((rr, x), np.nan) for rr, x in zip(d.rid, d[drv])])
X = pd.DataFrame({
    "ps6": side("drv_ps_pct_6", "a") - side("drv_ps_pct_6", "b"),
    "t15": side("drv_t15_6", "a") - side("drv_t15_6", "b"),
    "fin6": side("drv_fin_pct_6", "a") - side("drv_fin_pct_6", "b"),
    "qual": np.array([qz.get((rr, x), np.nan) for rr, x in zip(d.rid, d.a)]) - np.array([qz.get((rr, x), np.nan) for rr, x in zip(d.rid, d.b)]),
}).fillna(0.0)
yy = d.outcome_a.values; pred = np.zeros(len(d))
for rr in d.race_idx.unique():
    tr = (d.race_idx != rr).values
    m = LogisticRegression(C=0.5).fit(X[tr], yy[tr]); pred[~tr] = m.predict_proba(X[~tr])[:, 1]
def LL(q): q = np.clip(q, 1e-6, 1-1e-6); return -np.mean(yy*np.log(q) + (1-yy)*np.log(1-q))
print(f"\nlog-loss on 974 matchups:  market {LL(d.m.values):.4f}   GBM model {LL(p.values):.4f}   "
      f"4-feature logistic (LORO) {LL(pred):.4f}   coin flip {LL(np.full(len(d), .5)):.4f}")
