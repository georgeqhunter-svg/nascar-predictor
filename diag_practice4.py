"""Trace exactly what happens inside build_features for COTA."""
import pandas as pd
from src.features.build_features import build_features
from src.features.practice_pace import compute_practice_features

r = pd.read_parquet("data/processed/races.parquet")
e = pd.read_parquet("data/processed/entries.parquet")

# Call compute_practice_features exactly as build_features does
p = compute_practice_features(e, r)
print(f"compute_practice_features returned: {len(p)} rows, empty={p.empty}")
print()

# Index it exactly as build_features does
p_indexed = p.set_index(["race_id_short", "driver_id"]) if not p.empty else None
print(f"After set_index: type={type(p_indexed).__name__}")
if p_indexed is not None:
    print(f"  Index: {p_indexed.index.names}")
    print(f"  Level 0 sample: {p_indexed.index.get_level_values(0)[:3].tolist()}")
    print(f"  Level 1 sample: {p_indexed.index.get_level_values(1)[:3].tolist()}")
print()

# Emulate the loop for COTA
race_id = "2026-5598"
sub_e = e[e["race_id_short"] == race_id].sort_values("finish_pos")
driver_ids = sub_e["driver_id"].astype("Int64").tolist()

print(f"race_id={race_id!r}")
print(f"First 3 driver_ids: {driver_ids[:3]}")
print(f"First driver_id types: {[type(d).__name__ for d in driver_ids[:3]]}")
print()

for i, _drv_id in enumerate(driver_ids[:5]):
    print(f"driver_ids[{i}] = {_drv_id}, is None: {_drv_id is None}, "
          f"notna: {pd.notna(_drv_id)}")
    if _drv_id is None or not pd.notna(_drv_id):
        continue
    key = (race_id, int(_drv_id))
    in_idx = key in p_indexed.index
    print(f"  key={key}, in index: {in_idx}")
