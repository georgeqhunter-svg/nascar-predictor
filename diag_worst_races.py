"""What do our worst backtest races have in common? Uses the cached matchups."""
import os, numpy as np, pandas as pd
from src.features.track_profile import compute_track_profile
from src.features.formula_grid import formula_grid_races
from src.features.tracks import resolve_track_type

d = pd.read_parquet("data/processed/backtest_matchups.parquet")
def ip(o): return 100/(o+100) if o > 0 else -o/(-o+100)
ma, mb = d.odds_a.map(ip), d.odds_b.map(ip); d["m"] = ma/(ma+mb)
p = d.p_a_raw.clip(1e-4, 1-1e-4); y = d.outcome_a
d["llm"] = -(y*np.log(p)+(1-y)*np.log(1-p)); d["llk"] = -(y*np.log(d.m)+(1-y)*np.log(1-d.m))
d["overconf"] = (p-.5).abs() - (d.m-.5).abs()          # + = model more confident than market
d["disagree"] = (p-.5)*(d.m-.5) < 0                      # picks the other side
d["fav_m"] = np.where(d.m >= .5, d.a, d.b)

r = pd.read_parquet("data/processed/races.parquet"); r["date"] = pd.to_datetime(r.date)
e = pd.read_parquet("data/processed/entries.parquet")
r26 = r[r.season == 2026]
def rid_for(name):
    m = r26[r26.race_name.str.contains(name.split()[0], case=False, regex=False)]
    return m.race_id_short.iloc[0] if len(m) == 1 else None
names = d.drop_duplicates("race_idx")[["race_idx", "race"]]
# disambiguate by order: backtest races are chronological
r26s = r26.sort_values("date").reset_index(drop=True)
prof = compute_track_profile(e, r).drop_duplicates("race_id_short").set_index("race_id_short")
fg = formula_grid_races(e, r)
lr_ids = {int(f.split("_")[0]) for f in os.listdir("data/raw/practice_logs")}
q = pd.to_numeric(e.qual_speed, errors="coerce")
has_q = (q > 0).groupby(e.race_id_short).mean()

rows = []
for _, nm in names.iterrows():
    first = nm.race.split()[0]
    cand = r26s[r26s.race_name.str.contains(first, case=False, regex=False)]
    rid = cand.race_id_short.iloc[-1] if len(cand) else None
    if first.lower() in ("coca-cola", "coke"):
        cand = r26s[r26s.race_name.str.contains(nm.race, case=False, regex=False)]
        rid = cand.race_id_short.iloc[0] if len(cand) else rid
    g = d[d.race_idx == nm.race_idx]
    rr = r.set_index("race_id_short").loc[rid] if rid else None
    rows.append({
        "race": nm.race, "track": rr.track_name if rid else "?", "tt": resolve_track_type(rr.track_name) if rid else "?",
        "n": len(g), "delta": (g.llm - g.llk).mean(), "overconf": g.overconf.mean(),
        "disagree%": g.disagree.mean(), "mdl_acc": ((g.p_a_raw > .5) == (g.outcome_a == 1)).mean(),
        "mkt_acc": ((g.m > .5) == (g.outcome_a == 1)).mean(),
        "formula": rid in fg, "lr_practice": (int(rr.race_id_nascar) in lr_ids) if rid else None,
        "qual_speeds": has_q.get(rid, np.nan) > .5,
        "stick_trk": prof.loc[rid, "start_stickiness_at_track"] if rid in prof.index else np.nan,
        "dnf_trk": prof.loc[rid, "track_dnf_base"] if rid in prof.index else np.nan,
        "cautions": rr.cautions if rid else np.nan, "lead_chg": rr.lead_changes if rid else np.nan,
    })
t = pd.DataFrame(rows).sort_values("delta", ascending=False)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
num = ["overconf", "disagree%", "stick_trk", "dnf_trk", "cautions", "lead_chg"]
print("\nSpearman with per-race delta:")
print(t[num + ["delta"]].corr("spearman")["delta"].drop("delta").round(2).to_string())
w, b = t.head(8), t.tail(8)
print("\nworst 8 vs best 8:\n", pd.DataFrame({"worst": w[num + ['formula','lr_practice']].astype(float).mean(),
                                            "best": b[num + ['formula','lr_practice']].astype(float).mean()}).round(3))
# where the losses come from: disagree vs agree, confidence bucket
d["delta"] = d.llm - d.llk
print("\nLoss by model-vs-market stance (all races):")
print(d.groupby("disagree").agg(n=("delta", "size"), delta=("delta", "mean"), mdl_acc=("outcome_a", lambda s: 0)).drop(columns="mdl_acc").round(4))
d["oc_bin"] = pd.cut(d.overconf, [-1, -.1, -.03, .03, .1, 1])
print(d.groupby("oc_bin", observed=True).agg(n=("delta", "size"), delta=("delta", "mean")).round(4))
