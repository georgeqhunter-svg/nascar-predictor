"""Pre-registered test (2026-10-09): does a driver's PRE-2022 record at the
mapped track predict 2026 matchup outcomes beyond the current model (+0.0080)?
Mapping decided before looking: unchanged tracks -> own history; Daytona <->
Talladega pooled; Atlanta/Texas/Bristol/Roval/Chicago/new venues -> no prior."""
import numpy as np, pandas as pd, statsmodels.api as sm
import backtest_oddslogic_v5 as bt

H = "data/processed/history/"
hr = pd.read_parquet(H + "races.parquet"); he = pd.read_parquet(H + "entries.parquet"); hl = pd.read_parquet(H + "loopstats.parquet")
ALIAS = {"ISM Raceway": "Phoenix Raceway", "Dover International Speedway": "Dover Motor Speedway"}
UNCHANGED = ["Darlington Raceway", "Martinsville Speedway", "Richmond Raceway", "Phoenix Raceway", "Dover Motor Speedway",
             "New Hampshire Motor Speedway", "Pocono Raceway", "Michigan International Speedway", "Las Vegas Motor Speedway",
             "Kansas Speedway", "Homestead-Miami Speedway", "Charlotte Motor Speedway", "Watkins Glen International", "Sonoma Raceway"]
def group(tn):
    tn = ALIAS.get(tn, tn)
    if tn in ("Daytona International Speedway", "Talladega Superspeedway"): return "DAY/TAL"
    return tn if tn in UNCHANGED else None
he = he.merge(hr[["race_id_short", "track_name"]], on="race_id_short")
he["grp"] = he.track_name.map(group)
he = he[he.finish_pos > 0].copy()
n = he.groupby("race_id_short").finish_pos.transform("count")
he["fp"] = (he.finish_pos - 1) / (n - 1)
he["fp_run"] = he.fp.where(~he.is_dnf.astype(bool))
he = he.merge(hl[["race_id_short", "driver_id", "avg_ps"]], on=["race_id_short", "driver_id"], how="left")
he["ps"] = (he.avg_ps - 1) / (n - 1)
pri = he.dropna(subset=["grp"]).groupby(["driver", "grp"]).agg(
    hist_run=("fp_run", "mean"), hist_ps=("ps", "mean"), n_hist=("fp", "size")).reset_index()
pri = pri[pri.n_hist >= 2]
P = pri.set_index(["driver", "grp"])
print(f"history: {he.race_id_short.nunique()} races; priors for {pri.driver.nunique()} drivers x {pri.grp.nunique()} track groups")

d = pd.read_parquet("data/processed/backtest_matchups_fulldata_0080.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
R = pd.read_parquet("data/processed/races.parquet"); R["date"] = pd.to_datetime(R.date)
d["track"] = d.race_idx.map({i: R[R.date == pd.Timestamp(dt)].track_name.iloc[0] for i, (dt, _, _) in enumerate(bt.RACES)})
d["grp"] = d.track.map(group)
lg = lambda q: np.log(q/(1-q)); p = d.p_a_raw.clip(1e-4, 1-1e-4); y = d.outcome_a.astype(int)
for col, lab in [("hist_run", "pre-2022 running finishes at track"), ("hist_ps", "pre-2022 avg running position (loop, 2019-21)")]:
    va = np.array([P[col].get((a, g), np.nan) if g else np.nan for a, g in zip(d.a, d.grp)])
    vb = np.array([P[col].get((b, g), np.nan) if g else np.nan for b, g in zip(d.b, d.grp)])
    gap = pd.Series(vb - va, index=d.index)                    # + = a historically better (lower pct)
    ok = gap.notna()
    z = (gap[ok] - gap[ok].mean()) / gap[ok].std()
    out = []
    for base, nm in ((lg(p), "model"), (lg(d.m), "market")):
        f = sm.Logit(y[ok], sm.add_constant(pd.DataFrame({"b": base[ok], "z": z}))).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d.race_idx[ok]})
        out.append(f"beyond {nm}: {f.params['z']:+.3f} (t={f.tvalues['z']:+.2f})")
    print(f"{lab:48s} n={ok.sum():4d} matchups in {d.grp[ok].nunique()} track groups | " + " | ".join(out))

# Secondary (bigger sample, not market-relative): in 2022-2025 Next Gen races at
# mapped tracks, does the pre-2022 record predict finish BEYOND the driver's
# recent Next Gen form and his Next Gen record at that track?
E = pd.read_parquet("data/processed/entries.parquet").merge(R[["race_id_short", "track_name"]], on="race_id_short")
E["date"] = pd.to_datetime(E.date); E = E[E.finish_pos > 0].sort_values("date")
n2 = E.groupby("race_id_short").finish_pos.transform("count"); E["fp"] = (E.finish_pos - 1) / (n2 - 1)
E["grp"] = E.track_name.map(group)
E["form10"] = E.groupby("driver").fp.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
E["ng_trk"] = E.groupby(["driver", "grp"]).fp.transform(lambda s: s.shift(1).expanding().mean())
X = E[(E.date.dt.year <= 2025) & E.grp.notna()].copy()
for col in ("hist_run", "hist_ps"):
    X[col] = [P[col].get((dv, g), np.nan) for dv, g in zip(X.driver, X.grp)]
for col in ("hist_run", "hist_ps"):
    s = X.dropna(subset=[col, "form10"])
    for ctrl in (["form10"], ["form10", "ng_trk"]):
        ss = s.dropna(subset=ctrl)
        Z = ss[ctrl + [col]]; Z = (Z - Z.mean()) / Z.std()
        f = sm.OLS(ss.fp, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": ss.race_id_short})
        print(f"2022-25: finish ~ {'+'.join(ctrl)} + {col}: coef {f.params[col]:+.4f} (t={f.tvalues[col]:+.2f}), n={len(ss)}")

print("\nby season (finish ~ form10 + Next Gen track record + pre-2022 running position):")
s = X.dropna(subset=["hist_ps", "form10", "ng_trk"])
for yr, ss in s.groupby(s.date.dt.year):
    Z = ss[["form10", "ng_trk", "hist_ps"]]; Z = (Z - Z.mean()) / Z.std()
    f = sm.OLS(ss.fp, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": ss.race_id_short})
    print(f"  {yr}: coef {f.params['hist_ps']:+.4f} (t={f.tvalues['hist_ps']:+.2f}), n={len(ss)}")
