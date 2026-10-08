"""Does the market use this weekend's practice/qualifying speed more than the model?
For each matchup: z-scored qual speed + practice speed (NASCAR sessions, within race).
Regress (market - model) and outcome on the speed gaps."""
import numpy as np, pandas as pd
d = pd.read_parquet("data/processed/_market_edges_tmp.parquet")
s = pd.read_parquet("data/processed/sessions.parquet")
r = pd.read_parquet("data/processed/races.parquet")
def z(df, col):
    g = df.groupby("race_id_short")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")
best = (s[s.best_lap_speed > 0].groupby(["race_id_short", "run_type_label", "driver_name"])
        .best_lap_speed.max().reset_index())
out = {}
for lab in ("qualifying", "practice"):
    b = best[best.run_type_label == lab].copy()
    b["z"] = z(b, "best_lap_speed")
    out[lab] = b.set_index(["race_id_short", "driver_name"]).z
for lab, ser in out.items():
    for side in ("a", "b"):
        d[f"{lab}_{side}"] = [ser.get((rid, n), np.nan) for rid, n in zip(d.rid, d[side])]
    d[f"{lab}_gap"] = d[f"{lab}_a"] - d[f"{lab}_b"]          # + = driver a faster
p = d.p_a_raw.clip(1e-4, 1-1e-4)
lg = lambda q: np.log(q/(1-q))
d["mkt_minus_model"] = lg(d.m) - lg(p)      # + = market likes a more than model does
d["y"] = d.outcome_a
print("coverage:", {k: round(d[f'{k}_gap'].notna().mean(), 2) for k in out})
import statsmodels.api as sm
for lab in ("qualifying", "practice"):
    sub = d.dropna(subset=[f"{lab}_gap"])
    X = sm.add_constant(sub[[f"{lab}_gap"]])
    fit = sm.OLS(sub.mkt_minus_model, X).fit(cov_type="cluster", cov_kwds={"groups": sub.race_idx})
    print(f"\n[{lab}] market-minus-model logit on speed gap: coef {fit.params.iloc[1]:+.3f} "
          f"(t={fit.tvalues.iloc[1]:+.2f}, n={len(sub)})")
    # does the speed gap predict outcomes BEYOND the model?  BEYOND the market?
    for base, name in ((lg(p), "model"), (lg(d.m), "market")):
        Xo = pd.DataFrame({"base": base.loc[sub.index], "gap": sub[f"{lab}_gap"]})
        lo = sm.Logit(sub.y, sm.add_constant(Xo)).fit(disp=0, cov_type="cluster",
                                                       cov_kwds={"groups": sub.race_idx})
        print(f"   outcome ~ {name} + {lab}_gap: gap coef {lo.params['gap']:+.3f} (t={lo.tvalues['gap']:+.2f})")
both = d.dropna(subset=["qualifying_gap", "practice_gap"])
X = sm.add_constant(both[["qualifying_gap", "practice_gap"]])
f2 = sm.OLS(both.mkt_minus_model, X).fit(cov_type="cluster", cov_kwds={"groups": both.race_idx})
print("\nmarket-minus-model on both gaps:\n", f2.params.round(3).to_string(), "\n t:", f2.tvalues.round(2).to_dict(), " R2", round(f2.rsquared, 3))
