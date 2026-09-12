"""Repro exactly what build_features does for COTA."""
import pandas as pd
from src.features.practice_pace import compute_practice_features

r = pd.read_parquet("data/processed/races.parquet")
e = pd.read_parquet("data/processed/entries.parquet")

practice = compute_practice_features(e, r).set_index(["race_id_short", "driver_id"])

# Simulate the loop
race_id = "2026-5598"  # this is what race_row["race_id_short"] returns for COTA
e_cota = e[e["race_id_short"] == race_id]
driver_ids = e_cota["driver_id"].astype("Int64").tolist()

hits = 0
misses = 0
for _drv_id in driver_ids:
    if _drv_id is None or pd.isna(_drv_id):
        misses += 1
        continue
    key = (race_id, int(_drv_id))
    if key in practice.index:
        hits += 1
    else:
        misses += 1

print(f"race_id: '{race_id}' (type={type(race_id).__name__})")
print(f"index name: {practice.index.names}, dtypes: {practice.index.get_level_values(0).dtype}, {practice.index.get_level_values(1).dtype}")
print(f"Total drivers: {len(driver_ids)}, hits: {hits}, misses: {misses}")
print()

# Manual check
print("First practice index entries for 2026-5598:")
matching = practice[practice.index.get_level_values("race_id_short") == "2026-5598"]
print(matching.head(3).index.tolist())
print()
print("First driver_ids from entries for 2026-5598:")
print([int(d) for d in driver_ids[:3] if pd.notna(d)])
