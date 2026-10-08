"""Where is the MARKET itself consistently off? Model-free: compares vig-free
closing probabilities to outcomes, and flat-$1 ROI at the actual posted odds."""
import numpy as np, pandas as pd
d = pd.read_parquet("data/processed/_market_edges_tmp.parquet")
e = pd.read_parquet("data/processed/entries.parquet"); e["date"] = pd.to_datetime(e.date)
e = e.sort_values("date"); e["fin"] = pd.to_numeric(e.finish_pos, errors="coerce")
e["last_fin"] = e.groupby("driver").fin.shift(1)
e["last_dnf"] = e.groupby("driver").is_dnf.shift(1)
k = e.set_index(["race_id_short", "driver"])
for s in ("a", "b"):
    d[f"last_{s}"] = [k.last_fin.get((r, n), np.nan) for r, n in zip(d.rid, d[s])]
    d[f"lastdnf_{s}"] = [k.last_dnf.get((r, n), np.nan) for r, n in zip(d.rid, d[s])]

def dec(o): return o/100 + 1 if o > 0 else 100/abs(o) + 1
# One row per SIDE so we can ask "what if you bet every driver with property X"
rows = []
for _, r in d.iterrows():
    for me, op, o, mp, won, st, so, fo, la, ld, mk, tm in (
        (r.a, r.b, r.odds_a, r.m, r.outcome_a == 1, r.start_a, r.start_b, r.form_a, r.last_a, r.lastdnf_a, r.make_a, r.team_a),
        (r.b, r.a, r.odds_b, 1 - r.m, r.outcome_a == 0, r.start_b, r.start_a, r.form_b, r.last_b, r.lastdnf_b, r.make_b, r.team_b)):
        rows.append(dict(race=r.race_idx, tt=r.tt, rainout=r.rainout, driver=me, opp=op, odds=o, mkt=mp,
                         won=won, profit=(dec(o) - 1) if won else -1.0, start=st, opp_start=so,
                         form=fo, last=la, lastdnf=ld, make=mk, team=tm))
t = pd.DataFrame(rows)

def summ(g, title, min_n=40):
    a = g.agg(n=("won", "size"), priced=("mkt", "mean"), won=("won", "mean"), roi=("profit", "mean"),
              se_roi=("profit", lambda v: v.std()/np.sqrt(len(v))))
    a = a[a.n >= min_n]; a["edge"] = a.won - a.priced; a["t"] = a.roi / a.se_roi
    print(f"\n== {title} ==")
    print(a[["n", "priced", "won", "edge", "roi", "t"]].round(3).to_string())

t["price_band"] = pd.cut(t.mkt, [0, .3, .38, .45, .5, .55, .62, .7, 1])
summ(t.groupby("price_band", observed=True), "market calibration by price (favorite-longshot)")
summ(t.groupby(["tt", t.mkt > .5], observed=True), "favorites vs dogs by track type", 30)
t["start_adv"] = np.sign(t.opp_start - t.start)        # +1 = starts ahead
summ(t.groupby("start_adv"), "starts ahead of opponent?")
t["last_band"] = pd.cut(t["last"], [0, 5, 15, 25, 45], labels=["top5 last week", "6-15", "16-25", "26+"])
summ(t.groupby("last_band", observed=True), "last race finish (recency)")
summ(t.groupby(t.lastdnf.astype("boolean")), "DNF'd last race?")
summ(t.groupby("make"), "manufacturer")
summ(t.groupby("driver"), "by driver (min 40 sides)", 40)
t.drop(columns=[c for c in t.columns if c.endswith("band")]).to_parquet("data/processed/_market_sides_tmp.parquet")
