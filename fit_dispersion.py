"""Estimate per-track-type DNF overdispersion (frailty variance v) from history.

Under independent DNFs with per-type mean hazard p, a race with n cars has
Var(dnf_count) = n*p*(1-p). Observed variance is larger because wrecks
correlate. The excess maps to the Gamma frailty variance:
    v ~= (Var_obs - Var_indep) / (mean(n^2) * p^2)
Fit on <=2025 so the 2026 backtest races never inform this constant.
"""
import numpy as np
import pandas as pd

entries = pd.read_parquet("data/processed/entries.parquet")

# entries already carries `season`; only pull track_type from races if missing.
df = entries
if "track_type" not in df.columns:
    races = pd.read_parquet("data/processed/races.parquet")[["race_id_short", "track_type"]]
    df = df.merge(races.drop_duplicates("race_id_short"), on="race_id_short", how="left")

df = df[df["season"] <= 2025]
if "is_dnf" not in df.columns:
    raise SystemExit(f"no is_dnf column; entries columns are: {list(df.columns)}")

print("track_type        p_dnf  var_obs  var_indep   v_hat  races")
out = {}
for tt, g in sorted(df.groupby("track_type")):
    per_race = g.groupby("race_id_short")["is_dnf"].agg(["sum", "count"])
    X = per_race["sum"].to_numpy(float)
    n = per_race["count"].to_numpy(float)
    p = X.sum() / n.sum()
    var_obs = X.var(ddof=1)
    var_indep = float(np.mean(n * p * (1 - p)))
    v = max(0.0, (var_obs - var_indep) / (float(np.mean(n ** 2)) * p ** 2))
    out[tt] = v
    print(f"{tt:15s}  {p:.3f}   {var_obs:7.2f}   {var_indep:7.2f}   {v:.3f}   {len(X)}")

print("\nPaste into backtest_oddslogic_v5.py next to HAZARD:")
print("HAZARD_DISP = {")
for tt, v in out.items():
    print(f'    "{tt}": {v:.3f},')
print("}")