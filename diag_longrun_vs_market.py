"""Same test as diag_weekend_speed.py, but with LapRaptor long-run practice pace."""
import numpy as np, pandas as pd, statsmodels.api as sm
from src.features.practice_pace import compute_practice_features
d = pd.read_parquet("data/processed/_market_edges_tmp.parquet")
e = pd.read_parquet("data/processed/entries.parquet"); r = pd.read_parquet("data/processed/races.parquet")
pr = compute_practice_features(e, r)
did = e.set_index(["race_id_short", "driver"]).driver_id
cols = ["practice_10lap_avg_z", "practice_5lap_avg_z", "practice_best_speed_z", "practice_consistency_z"]
pr = pr.set_index(["race_id_short", "driver_id"])
for c in cols:
    for side in ("a", "b"):
        vals = []
        for rid, n in zip(d.rid, d[side]):
            i = did.get((rid, n), np.nan)
            v = pr[c].get((rid, int(i)), np.nan) if pd.notna(i) else np.nan
            vals.append(v)
        d[f"{c}_{side}"] = vals
    d[f"{c}_gap"] = d[f"{c}_a"] - d[f"{c}_b"]
p = d.p_a_raw.clip(1e-4, 1-1e-4); lg = lambda q: np.log(q/(1-q))
d["mm"] = lg(d.m) - lg(p); y = d.outcome_a
print("matchups with long-run practice data:", int(d["practice_10lap_avg_z_gap"].notna().sum()), "of", len(d))
for c in cols:
    sub = d.dropna(subset=[f"{c}_gap"]); sub = sub[sub[f"{c}_gap"] != 0]
    g = sub[f"{c}_gap"]; cl = {"groups": sub.race_idx}
    f1 = sm.OLS(sub.mm, sm.add_constant(g)).fit(cov_type="cluster", cov_kwds=cl)
    fm = sm.Logit(y.loc[sub.index], sm.add_constant(pd.DataFrame({"b": lg(p).loc[sub.index], "g": g}))).fit(disp=0, cov_type="cluster", cov_kwds=cl)
    fk = sm.Logit(y.loc[sub.index], sm.add_constant(pd.DataFrame({"b": lg(d.m).loc[sub.index], "g": g}))).fit(disp=0, cov_type="cluster", cov_kwds=cl)
    print(f"{c:24s} n={len(sub):4d}  mkt-vs-model t={f1.tvalues.iloc[1]:+.2f}  "
          f"beyond model coef {fm.params['g']:+.3f} t={fm.tvalues['g']:+.2f}  "
          f"beyond market coef {fk.params['g']:+.3f} t={fk.tvalues['g']:+.2f}")
