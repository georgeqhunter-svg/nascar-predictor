"""Fresh check: does build_features now actually populate practice features?"""
import pandas as pd
from src.features.build_features import build_features

r = pd.read_parquet("data/processed/races.parquet")
e = pd.read_parquet("data/processed/entries.parquet")
s = pd.read_parquet("data/processed/sessions.parquet")
lt = pd.read_parquet("data/processed/laptimes.parquet")

f = build_features(r, e, s, laptimes=lt)
cota = f[f["race_id_short"] == "2026-5598"]
print(f"COTA rows: {len(cota)}")
print(f"has_practice_data sum: {cota['has_practice_data'].sum()} / {len(cota)}")
print(f"practice_best_speed_z nonzero: {(cota['practice_best_speed_z'] != 0).sum()}")
print()
print(cota[["driver", "has_practice_data", "practice_best_speed_z",
            "practice_5lap_avg_z", "practice_10lap_avg_z"]].head(15).to_string(index=False))
