"""Diagnose why practice features aren't matching to 2026-5598 (COTA)."""
import pandas as pd
from pathlib import Path

# 1. Check if CSV files exist for LapRaptor race_id 5598
print("=" * 60)
print("Files matching 5598_*:")
files = list(Path("data/raw/practice_logs").glob("5598_*.csv"))
for path in files:
    df = pd.read_csv(path, usecols=["series", "run_name", "lap_timestamp"], nrows=1)
    ts = pd.to_datetime(df["lap_timestamp"].iloc[0], utc=True, errors="coerce")
    print(f"  {path.name}: series={df['series'].iloc[0]}, "
          f"run={df['run_name'].iloc[0]}, date={ts.date() if pd.notna(ts) else None}")
print()

# 2. Check what date COTA is in races.parquet
print("=" * 60)
print("Races.parquet lookup for COTA (2026-5598):")
r = pd.read_parquet("data/processed/races.parquet")
sub = r[r["race_id_short"] == "2026-5598"]
print(sub[["race_id_short", "date", "race_name"]].to_string(index=False))
print()

# 3. Show all races on the same weekend
print("=" * 60)
print("All 2026 races in Feb-Mar:")
r["date"] = pd.to_datetime(r["date"])
sub = r[(r["season"] == 2026) & (r["date"].dt.month.isin([2, 3]))]
print(sub[["race_id_short", "date", "race_name"]].to_string(index=False))
