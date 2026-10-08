"""Where does the market beat the model? Slices excess log-loss (model - market)
across matchup situations. Uses the cached per-matchup backtest results."""
import sys, numpy as np, pandas as pd
from src.features.tracks import resolve_track_type
from src.features.formula_grid import formula_grid_races

f = sys.argv[1] if len(sys.argv) > 1 else "data/processed/backtest_matchups_stickiness_0282.parquet"
d = pd.read_parquet(f)
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
p = d.p_a_raw.clip(1e-4, 1-1e-4); y = d.outcome_a
d["x"] = -(y*np.log(p)+(1-y)*np.log(1-p)) + (y*np.log(d.m)+(1-y)*np.log(1-d.m))

# --- attach race + driver context ---
r = pd.read_parquet("data/processed/races.parquet"); r["date"] = pd.to_datetime(r.date)
e = pd.read_parquet("data/processed/entries.parquet"); e["date"] = pd.to_datetime(e.date)
import backtest_oddslogic_v5 as bt
dates = [pd.Timestamp(dt) for dt, _, _ in bt.RACES]
d["date"] = d.race_idx.map(dict(enumerate(dates)))
rid_by_date = r[r.season == 2026].set_index("date").race_id_short
d["rid"] = d.date.map(rid_by_date)
d["tt"] = d.rid.map(r.set_index("race_id_short").track_name).map(resolve_track_type)
fg = formula_grid_races(e, r); d["rainout"] = d.rid.isin(fg)

e = e.sort_values("date")
e["career"] = e.groupby("driver").cumcount()   # races BEFORE this one
e["fin"] = pd.to_numeric(e.finish_pos, errors="coerce")
e["form10"] = e.groupby("driver").fin.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
key = e.set_index(["race_id_short", "driver"])
def look(col, drv_col):
    return [key[col].get((rid, drv), np.nan) for rid, drv in zip(d.rid, d[drv_col])]
for side in ("a", "b"):
    d[f"start_{side}"] = pd.to_numeric(pd.Series(look("start_pos", side)), errors="coerce").values
    d[f"career_{side}"] = look("career", side)
    d[f"form_{side}"] = look("form10", side)
    d[f"team_{side}"] = look("team", side)
    d[f"make_{side}"] = look("make", side)

d["fav_strength"] = (d.m - .5).abs()
d["disagree"] = (p - .5) * (d.m - .5) < 0
d["start_gap"] = (d.start_a - d.start_b).abs()
# does the model side with the better STARTER more than the market does?
better_start_a = d.start_a < d.start_b
d["model_leans_grid"] = np.where(better_start_a, p - d.m, d.m - p) > 0
d["min_career"] = d[["career_a", "career_b"]].min(axis=1)
d["avg_form"] = d[["form_a", "form_b"]].mean(axis=1)
d["teammates"] = d.team_a == d.team_b
d["same_make"] = d.make_a == d.make_b

def show(col, bins=None, labels=None, title=None):
    s = pd.cut(d[col], bins, labels=labels) if bins is not None else d[col]
    g = d.groupby(s, observed=True).agg(n=("x", "size"), excess=("x", "mean"),
                                         se=("x", lambda v: v.std()/np.sqrt(len(v))))
    g["excess_bps"] = (g.excess*1e4).round(0); g["±se"] = (g.se*1e4).round(0)
    print(f"\n-- {title or col} --"); print(g[["n", "excess_bps", "±se"]].to_string())

print(f"overall: n={len(d)}  excess={d.x.mean()*1e4:+.0f} bps (model - market log-loss; + = market better)")
show("tt", title="track type")
show("rainout", title="formula-grid rainout")
show("fav_strength", [0, .05, .12, .2, .5], ["pick'em <55%", "55-62%", "62-70%", "70%+"], "how lopsided the market line is")
show("disagree", title="model picks the market underdog")
show("start_gap", [-1, 3, 8, 15, 40], ["0-3", "4-8", "9-15", "16+"], "starting-position gap")
show("model_leans_grid", title="model leans toward better starter more than market")
show("min_career", [-1, 50, 150, 1000], ["<50 Cup starts", "50-150", "150+"], "least-experienced driver in matchup")
show("avg_form", [0, 10, 15, 20, 45], ["elite (<10)", "10-15", "15-20", "20+"], "avg 10-race form of the pair")
show("teammates", title="teammates")
show("same_make", title="same manufacturer")
d.to_parquet("data/processed/_market_edges_tmp.parquet")
