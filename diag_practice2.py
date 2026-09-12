"""Check driver_id and race_id_short compatibility between practice and entries."""
import pandas as pd
from src.features.practice_pace import compute_practice_features

r = pd.read_parquet("data/processed/races.parquet")
e = pd.read_parquet("data/processed/entries.parquet")

p = compute_practice_features(e, r)
print("Practice DF driver_id dtype:", p["driver_id"].dtype)
print("Practice DF race_id_short sample:", p["race_id_short"].iloc[0])
print()

# Does 2026-5598 appear in practice DF?
cota = p[p["race_id_short"] == "2026-5598"]
print(f"2026-5598 in practice DF: {len(cota)} rows")
if len(cota) > 0:
    print("  Some driver_ids:", cota["driver_id"].head(5).tolist())
print()

# What about entries for that race?
e_cota = e[e["race_id_short"] == "2026-5598"]
print(f"2026-5598 in entries: {len(e_cota)} rows")
print("Entries driver_id dtype:", e["driver_id"].dtype)
print("  Some driver_ids:", e_cota["driver_id"].head(5).tolist())
print()

# Try the exact set intersection
prac_ids = set(cota["driver_id"].astype(int).tolist())
ent_ids = set(e_cota["driver_id"].dropna().astype(int).tolist())
print(f"Practice driver_ids: {sorted(prac_ids)[:10]}")
print(f"Entries driver_ids:  {sorted(ent_ids)[:10]}")
print(f"Intersection size: {len(prac_ids & ent_ids)}")
