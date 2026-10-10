"""Pre-registered surface test (2026-10-09). 2022-2025 data only.
Does a driver's record on the SAME SURFACE CLASS predict his finish beyond
recent form and track-type history? Ovals only, excl. Daytona/Talladega.
Repave years: dailydownforce.com list (Jan 2024) + user: N. Wilkesboro 2024,
no repaves since 2024; uncovered tracks are asphalt (age unknown -> excluded)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from src.features.tracks import resolve_track_type

REPAVE = {  # track_name in races.parquet -> last full repave year
    "Atlanta Motor Speedway": 2022, "Sonoma Raceway": 2024, "North Wilkesboro Speedway": 2024,
    "Texas Motor Speedway": 2017, "World Wide Technology Raceway": 2017, "Watkins Glen International": 2016,
    "Kansas Speedway": 2012, "Michigan International Speedway": 2012, "Pocono Raceway": 2012,
    "Phoenix Raceway": 2011, "Daytona International Speedway": 2011, "Darlington Raceway": 2008,
    "Las Vegas Motor Speedway": 2007, "Charlotte Motor Speedway": 2006, "Talladega Superspeedway": 2006,
    "New Hampshire Motor Speedway": 2005, "Richmond Raceway": 2004, "Martinsville Speedway": 2004,
    "Indianapolis Motor Speedway": 2004, "Homestead-Miami Speedway": 2003, "Iowa Speedway": 2006,
}
CONCRETE = {"Dover Motor Speedway", "Nashville Superspeedway", "Bristol Motor Speedway", "Martinsville Speedway"}
NWB_PRE2024_OLD = True     # pre-2024 North Wilkesboro surface was decades old

def surface_class(track, year):
    if track in CONCRETE: return "concrete"
    if track == "North Wilkesboro Speedway" and year < 2024: return "old"
    if track == "Sonoma Raceway" and year < 2024: return "old"
    y = REPAVE.get(track)
    if y is None: return None
    age = year - y
    return "fresh" if age < 5 else ("mid" if age < 15 else "old")

E = pd.read_parquet("data/processed/entries.parquet"); R = pd.read_parquet("data/processed/races.parquet")
E = E[E.finish_pos > 0].merge(R[["race_id_short", "track_name"]], on="race_id_short")
E["date"] = pd.to_datetime(E.date); E = E.sort_values(["date", "race_id_short"])
E["tt"] = E.track_name.map(resolve_track_type)
n = E.groupby("race_id_short").finish_pos.transform("count"); E["fp"] = (E.finish_pos - 1) / (n - 1)
E["cls"] = [surface_class(t, d.year) for t, d in zip(E.track_name, E.date)]
oval = (~E.tt.isin(["road", "superspeedway"]))
E.loc[~oval, "cls"] = None
print(E[oval & E.cls.notna()].drop_duplicates("race_id_short").groupby("cls").size().rename("races (2022+)").to_string())
roll = lambda keys, w, mp: E.groupby(keys).fp.transform(lambda s: s.shift(1).rolling(w, min_periods=mp).mean())
E["form10"] = roll("driver", 10, 3)
E["type10"] = roll(["driver", "tt"], 10, 3)
E["trk5"] = E.groupby(["driver", "track_name"]).fp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
S = E[E.cls.notna()].copy()
S["cls10"] = S.groupby(["driver", "cls"]).fp.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
E = E.join(S["cls10"])
X = E[(E.date.dt.year <= 2025) & E.cls.notna()]
for ctrl in (["form10", "type10"], ["form10", "type10", "trk5"]):
    s = X.dropna(subset=ctrl + ["cls10"])
    Z = s[ctrl + ["cls10"]]; Z = (Z - Z.mean()) / Z.std()
    f = sm.OLS(s.fp, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": s.race_id_short})
    print(f"\nfinish ~ {' + '.join(ctrl)} + SAME-SURFACE-CLASS history (n={len(s)}, {s.race_id_short.nunique()} races)")
    for c in ctrl + ["cls10"]:
        print(f"   {c:8s} {f.params[c]:+.4f} (t={f.tvalues[c]:+.2f})")
    # by class
    for cl, ss in s.groupby("cls"):
        if len(ss) < 150: continue
        Zc = ss[ctrl + ["cls10"]]; Zc = (Zc - Zc.mean()) / Zc.std()
        fc = sm.OLS(ss.fp, sm.add_constant(Zc)).fit(cov_type="cluster", cov_kwds={"groups": ss.race_id_short})
        print(f"     class {cl:8s} n={len(ss):4d}  same-class coef {fc.params['cls10']:+.4f} (t={fc.tvalues['cls10']:+.2f})")
