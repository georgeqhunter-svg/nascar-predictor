"""Worst vs best backtest races on the CURRENT model (fast baseline +0.0195):
race-level attributes that might explain where the model struggles."""
import numpy as np, pandas as pd
import backtest_oddslogic_v5 as bt
from src.features.tracks import resolve_track_type
from src.features.formula_grid import formula_grid_races

d = pd.read_parquet("data/processed/backtest_matchups_fastbase_0195.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
p = d.p_a_raw.clip(1e-4, 1-1e-4); y = d.outcome_a
d["llm"] = -(y*np.log(p)+(1-y)*np.log(1-p)); d["llk"] = -(y*np.log(d.m)+(1-y)*np.log(1-d.m))
d["x"] = d.llm - d.llk
d["conf"] = (p-.5).abs(); d["mconf"] = (d.m-.5).abs()
d["dis"] = (p-.5)*(d.m-.5) < 0
d["fav_won"] = np.where(d.m >= .5, y == 1, y == 0)

R = pd.read_parquet("data/processed/races.parquet"); R["date"] = pd.to_datetime(R.date)
E = pd.read_parquet("data/processed/entries.parquet"); E["date"] = pd.to_datetime(E.date)
dates = {i: pd.Timestamp(dt) for i, (dt, _, _) in enumerate(bt.RACES)}
d["rid"] = d.race_idx.map({i: R[R.date == dt].race_id_short.iloc[0] for i, dt in dates.items()})
fg = formula_grid_races(E, R)

E = E.sort_values("date"); n = E.groupby("race_id_short").finish_pos.transform("count")
E["fp"] = (E.finish_pos - 1) / (n - 1)
E["form10"] = E[E.finish_pos > 0].groupby("driver").fp.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())

rows = []
for rid, g in d.groupby("rid"):
    rr = R.set_index("race_id_short").loc[rid]; e = E[(E.race_id_short == rid) & (E.finish_pos > 0)]
    # Did the matchup drivers (the ones the market prices) have chaotic days?
    md = set(g.a) | set(g.b); em = e[e.driver.isin(md)]
    rows.append(dict(
        race=g.race.iloc[0], track=rr.track_name.replace(" Speedway", "").replace(" Raceway", "")[:22],
        tt=resolve_track_type(rr.track_name), n=len(g), excess=g.x.mean(), mkt_ll=g.llk.mean(),
        fav_won=g.fav_won.mean(), model_conf=g.conf.mean(), mkt_conf=g.mconf.mean(), disagree=g.dis.mean(),
        rainout=rid in fg, cautions=rr.cautions, lead_chg=rr.lead_changes,
        dnf_field=e.is_dnf.astype(bool).mean(), dnf_matchup_drv=em.is_dnf.astype(bool).mean(),
        form_rho=e[["form10", "fp"]].corr("spearman").iloc[0, 1],
        start_rho=e[["start_pos", "fp"]].corr("spearman").iloc[0, 1],
        playoff=int(rr.get("playoff_round", 0) or 0) > 0,
    ))
t = pd.DataFrame(rows).sort_values("excess", ascending=False)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
num = ["mkt_ll", "fav_won", "model_conf", "mkt_conf", "disagree", "cautions", "lead_chg", "dnf_field", "dnf_matchup_drv", "form_rho", "start_rho"]
print("\nSpearman with per-race excess (27 races):")
print(t[num + ["excess"]].corr("spearman")["excess"].drop("excess").round(2).sort_values().to_string())
w, b = t.head(9), t.tail(9)
print("\nworst 9 vs best 9 (means):")
print(pd.DataFrame({"worst9": w[num + ["rainout", "playoff"]].astype(float).mean(),
                    "best9": b[num + ["rainout", "playoff"]].astype(float).mean()}).round(3).to_string())
