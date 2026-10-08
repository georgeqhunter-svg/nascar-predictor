"""Which drivers does the model misjudge, and where? Attributes each matchup's
excess log-loss (model - market) to both drivers, signed by whether the model
rated the driver ABOVE (overrated) or BELOW (underrated) the market."""
import numpy as np, pandas as pd
d = pd.read_parquet("data/processed/backtest_matchups.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
p = d.p_a_raw.clip(1e-4, 1-1e-4); y = d.outcome_a
d["x"] = -(y*np.log(p)+(1-y)*np.log(1-p)) + (y*np.log(d.m)+(1-y)*np.log(1-d.m))  # excess LL
d["gap"] = p - d.m   # + = model likes driver a more than market
rows = []
for _, r in d.iterrows():
    for drv, g, won in ((r.a, r.gap, r.outcome_a == 1), (r.b, -r.gap, r.outcome_a == 0)):
        rows.append({"race": r.race, "driver": drv, "gap": g, "won": won, "x": r.x / 2})
t = pd.DataFrame(rows)
WORST = ["Autotrader", "Quaker State", "Toyota Save Mart", "Goodyear 400", "Window World",
         "Great American", "Dollar Tree"]
t["worst"] = t.race.str.contains("|".join(WORST))
t["dir"] = np.where(t.gap > 0, "over", "under")

def table(sub, title, k=12):
    g = sub.groupby("driver").agg(n=("x", "size"), excess=("x", "sum"), avg_gap=("gap", "mean"),
                                  win_rate=("won", "mean"))
    print(f"\n== {title} — biggest excess log-loss contributors ==")
    print(g.sort_values("excess", ascending=False).head(k).round(3).to_string())
table(t, "ALL 25 races")
table(t[t.worst], "WORST 7 races (Atlanta x2, Sonoma, Darlington spr, N.Wilkesboro, Pocono, Loudon)")
for name in ["Autotrader", "Quaker State", "Toyota Save Mart", "Goodyear 400"]:
    table(t[t.race.str.contains(name)], name, k=6)
print("\nExcess by direction (worst races):")
print(t[t.worst].groupby("dir").agg(n=("x", "size"), excess=("x", "sum"), win_rate=("won", "mean")).round(3))
print("\nExcess by direction (all races):")
print(t.groupby("dir").agg(n=("x", "size"), excess=("x", "sum"), win_rate=("won", "mean")).round(3))
