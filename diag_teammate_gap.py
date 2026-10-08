"""Does the model drag drivers who outperform their teammates toward the team average?"""
import numpy as np, pandas as pd
import backtest_oddslogic_v5 as bt
from src.models.plackett_luce import RaceRanking, Ratings, update_from_race
from src.features.tracks import resolve_track_type

d = pd.read_parquet("data/processed/backtest_matchups_baseline27_0285.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
rows = []
for _, r in d.iterrows():
    rows.append((r.a, r.p_a_raw, r.m, r.outcome_a == 1)); rows.append((r.b, 1-r.p_a_raw, 1-r.m, r.outcome_a == 0))
s = pd.DataFrame(rows, columns=["driver", "model", "mkt", "won"])
per = s.groupby("driver").agg(n=("won", "size"), model_vs_mkt=("model", lambda v: 0), won=("won", "mean"), mkt=("mkt", "mean"))
per["model_vs_mkt"] = s.assign(g=s.model - s.mkt).groupby("driver").g.mean()
per["won_vs_mkt"] = per.won - per.mkt
per = per[per.n >= 30]

# 2026 teammate gap: driver's avg running finish pct minus teammates' (same team, same races)
e = pd.read_parquet("data/processed/entries.parquet")
e = e[(e.season == 2026) & (e.finish_pos > 0)].copy()
n = e.groupby("race_id_short").finish_pos.transform("count"); e["fp"] = (e.finish_pos - 1)/(n - 1)
run = e[~e.is_dnf.astype(bool)]
gap = {}
for drv, g in run.groupby("driver"):
    team = g.team.mode().iloc[0]
    tm = run[(run.team == team) & (run.driver != drv) & run.race_id_short.isin(g.race_id_short)]
    if len(tm) >= 8:
        gap[drv] = (g.fp.mean() - tm.fp.mean(), team)
per["tm_gap"] = [gap.get(i, (np.nan, ""))[0] for i in per.index]     # negative = beats teammates
per["team"] = [gap.get(i, (np.nan, ""))[1] for i in per.index]
x = per.dropna(subset=["tm_gap"])
print(x.sort_values("tm_gap")[["team", "n", "tm_gap", "model_vs_mkt", "won_vs_mkt"]].round(3).to_string())
print("\nSpearman  tm_gap vs model_vs_mkt:", round(x[["tm_gap", "model_vs_mkt"]].corr("spearman").iloc[0, 1], 2),
      "   tm_gap vs won_vs_mkt:", round(x[["tm_gap", "won_vs_mkt"]].corr("spearman").iloc[0, 1], 2), f"  (n={len(x)} drivers)")

# PL decomposition at the latest race: driver vs team component
E = pd.read_parquet("data/processed/entries.parquet"); R = pd.read_parquet("data/processed/races.parquet")
R = R.sort_values("date"); rat = Ratings()
for _, rr in R.iterrows():
    g = E[(E.race_id_short == rr.race_id_short) & (E.finish_pos > 0)].sort_values("finish_pos")
    if len(g) < 2: continue
    tt = resolve_track_type(rr.track_name); dn = g.is_dnf.astype(bool).tolist()
    first = next((i for i, v in enumerate(dn) if v), len(g))
    update_from_race(rat, RaceRanking(drivers=g.driver.tolist(), teams=g.team.tolist(), track_type=tt, dnf_at=first))
print("\nPlackett-Luce ratings after Las Vegas (intermediate):")
for drv, team in [("Ryan Blaney", "Team Penske"), ("Joey Logano", "Team Penske"), ("Austin Cindric", "Team Penske"),
                  ("Kyle Larson", "Hendrick Motorsports"), ("Denny Hamlin", "Joe Gibbs Racing"), ("Christopher Bell", "Joe Gibbs Racing")]:
    dv, tv = rat.driver.get(drv, 0), rat.team.get(team, 0); dt = rat.driver_track.get(drv, {}).get("intermediate", 0)
    print(f"  {drv:18s} driver {dv:+.2f}  team {tv:+.2f}  drv@type {dt:+.2f}  total {dv+tv+dt:+.2f}")
