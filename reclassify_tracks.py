"""Rerun track_type classification over races.parquet and entries.parquet
after the _norm_track_name bugfix. In place."""
import pandas as pd
from src.scrape.nascar_com import track_type_for

for path in ("data/processed/races.parquet", "data/processed/entries.parquet"):
    df = pd.read_parquet(path)
    if "track_name" not in df.columns:
        print(f"{path}: no track_name, skip")
        continue
    old = df["track_type"].copy() if "track_type" in df.columns else None
    df["track_type"] = df["track_name"].map(track_type_for)
    df.to_parquet(path, index=False)
    if old is not None:
        changed = (old != df["track_type"]).sum()
        print(f"{path}: {changed} rows changed")

r = pd.read_parquet("data/processed/races.parquet")
print("\nNew distribution:")
print(r["track_type"].value_counts())

print("\nCharlotte + Indy sanity check:")
sub = r[r["track_name"].str.contains("Charlotte|Indianapolis", na=False)]
print(sub[["date", "race_name", "track_name", "track_type"]].to_string(index=False))
