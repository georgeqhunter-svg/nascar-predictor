"""Check race-simulation assumptions directly against 2022-2025 data (2026 = backtest, untouched).
Persistence = slope of next-race outcome on a past rate's deviation from its baseline.
1.0 = past rate fully carries forward, 0 = pure noise."""
import numpy as np, pandas as pd
from src.features.tracks import resolve_track_type
import backtest_oddslogic_v5 as bt

E = pd.read_parquet("data/processed/entries.parquet"); R = pd.read_parquet("data/processed/races.parquet")
E = E[E.finish_pos > 0].merge(R[["race_id_short", "track_name"]], on="race_id_short")
E["date"] = pd.to_datetime(E.date); E = E.sort_values(["date", "race_id_short"])
E["tt"] = E.track_name.map(resolve_track_type)
E = E[E.date.dt.year <= 2025]
q = pd.to_numeric(E.qual_pos, errors="coerce").where(lambda s: s > 0, pd.to_numeric(E.start_pos, errors="coerce"))
E["dnf"] = E.is_dnf.astype(float)
E["dmg"] = ((E.dnf == 0) & (E.finish_pos > q + 12)).astype(float)
st = E.status.astype(str).str.lower()
E["crash"] = ((E.dnf == 1) & st.str.contains("accident|crash|dvp|damage")).astype(float)
E["mech"] = ((E.dnf == 1) & ~st.str.contains("accident|crash|dvp|damage")).astype(float)

def persistence(df, key, col, window=10, min_n=8, base_by="tt"):
    df = df.copy()
    df["past"] = df.groupby(key)[col].transform(lambda s: s.shift(1).rolling(window, min_periods=min_n).mean())
    df["base"] = df.groupby(base_by)[col].transform(lambda s: s.expanding().mean().shift(1))
    x = df.dropna(subset=["past", "base"]); x = x[x.date.dt.year >= 2023]
    dev, res = x.past - x.base, x[col] - x.base
    b = (dev*res).sum() / (dev*dev).sum()
    se = np.sqrt(((res - b*dev)**2).sum() / (dev*dev).sum() / len(x))
    return b, se, len(x)

print("== DRIVER-level persistence (window 10 races at the track type, >= 8) ==")
for col, lab in [("dnf", "any DNF"), ("crash", "crash DNF"), ("mech", "mechanical DNF"), ("dmg", "damaged but finished")]:
    b, se, n = persistence(E, ["driver", "tt"], col)
    print(f"  {lab:22s} persistence {b:+.3f} ± {se:.3f}   (n={n})")
print(f"  model currently: DNF weight up to 10/(10+K) = {10/(10+bt.HAZARD_SHRINK_K):.3f} (K={bt.HAZARD_SHRINK_K}); "
      f"crash 'bump' added on top UNSHRUNK before blending; damage: no driver term")

print("\n== TRACK-level persistence (race-level rates, last 4 races at the track vs type) ==")
rl = E.groupby(["race_id_short", "date", "track_name", "tt"]).agg(dnf=("dnf", "mean"), dmg=("dmg", "mean")).reset_index().sort_values("date")
for col in ("dnf", "dmg"):
    b, se, n = persistence(rl, "track_name", col, window=4, min_n=2)
    print(f"  {col:4s} persistence {b:+.3f} ± {se:.3f}  (n={n} races)")

print("\n== Baselines: simulation constants vs 2022-2025 data ==")
emp = E.groupby("tt").agg(dnf=("dnf", "mean"), dmg=("dmg", "mean"), n=("dnf", "size"))
emp["HAZARD (model)"] = emp.index.map(bt.HAZARD.get); emp["DAMAGE_HAZARD (model)"] = emp.index.map(bt.DAMAGE_HAZARD.get)
print(emp.round(3).to_string())

print("\n== Field DNF-count overdispersion vs model DNF_DISPERSION ==")
for tt, g in rl.groupby("tt"):
    nper = E[E.tt == tt].groupby("race_id_short").size().mean(); p = g.dnf.mean()
    binom_var = p*(1-p)/nper; obs = g.dnf.var()
    # Gamma frailty with var v on a per-race multiplier: Var(frac) ~ binom + p^2 v
    v = max(0.0, (obs - binom_var) / p**2)
    print(f"  {tt:13s} races={len(g):3d}  DNF frac {p:.3f}  implied dispersion {v:.2f}  model {bt.DNF_DISPERSION.get(tt)}")

print("\n== Damaged drivers: positions lost vs qualifying (empirical) ==")
dm = E[E.dmg == 1]
print("  median positions lost by type:", (dm.finish_pos - q[dm.index]).groupby(dm.tt).median().round(1).to_dict())
