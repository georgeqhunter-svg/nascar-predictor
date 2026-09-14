"""Manually inject Illinois 2026 qualifying results into entries.parquet.
NASCAR.com's cacher hasn't published them yet, so we patch by hand.

Source: https://www.nascar.com/live-results/nascar-cup-series/2026-enjoy-illinois-300/
"""
from pathlib import Path
import pandas as pd


QUALIFYING = [
    (1,  "Joey Logano",         137.124),
    (2,  "Kyle Larson",         135.980),
    (3,  "Ryan Blaney",         135.898),
    (4,  "Christopher Bell",    135.767),
    (5,  "Chase Briscoe",       135.546),
    (6,  "Carson Hocevar",      135.395),
    (7,  "Bubba Wallace",       135.355),
    (8,  "Chase Elliott",       135.302),
    (9,  "Ty Gibbs",            135.294),
    (10, "Denny Hamlin",        135.086),
    (11, "Alex Bowman",         134.702),
    (12, "Zane Smith",          134.650),
    (13, "Austin Cindric",      134.565),
    (14, "Ryan Preece",         134.381),
    (15, "Daniel Suárez",       134.360),
    (16, "Josh Berry",          134.316),
    (17, "AJ Allmendinger",     134.140),
    (18, "Tyler Reddick",       134.100),
    (19, "Connor Zilisch",      134.040),
    (20, "Ross Chastain",       133.996),
    (21, "Chris Buescher",      133.901),
    (22, "Austin Dillon",       133.833),
    (23, "Brad Keselowski",     133.575),
    (24, "William Byron",       133.531),
    (25, "Erik Jones",          133.480),
    (26, "Cole Custer",         133.432),
    (27, "Austin Hill",         133.262),
    (28, "John H. Nemechek",    133.097),
    (29, "Michael McDowell",    132.830),
    (30, "Cody Ware",           132.642),
    (31, "Noah Gragson",        132.610),
    (32, "Ty Dillon",           132.575),
    (33, "Riley Herbst",        132.112),
    (34, "Todd Gilliland",      132.100),
    (35, "Shane Van Gisbergen", 131.633),
    (36, "Ricky Stenhouse Jr",  131.521),
]

path = Path("data/processed/entries.parquet")
entries = pd.read_parquet(path)
entries["date"] = pd.to_datetime(entries["date"])

# Find Illinois race_id_short.
target = entries[
    (entries["date"] == pd.Timestamp("2026-09-13"))
    & (entries["season"] == 2026)
]
if target.empty:
    print("Illinois race not found — is date 2026-09-13?")
    raise SystemExit(1)
rid = target["race_id_short"].iloc[0]
print(f"Patching {rid}, {len(target)} drivers")

# Cast to floats for qual_speed if needed.
if "qual_speed" in entries.columns:
    entries["qual_speed"] = entries["qual_speed"].astype(float)

# Update rows.
patched = 0
unmatched = []
for pos, driver, speed in QUALIFYING:
    mask = (entries["race_id_short"] == rid) & (entries["driver"] == driver)
    if not mask.any():
        unmatched.append(driver)
        continue
    entries.loc[mask, "qual_pos"] = pos
    entries.loc[mask, "qual_speed"] = speed
    entries.loc[mask, "start_pos"] = pos  # single-round qual, no stage penalties applied
    patched += 1

entries.to_parquet(path, index=False)
print(f"Patched {patched}/{len(QUALIFYING)} drivers")
if unmatched:
    print(f"NOT FOUND in entries: {unmatched}")
    print("These names may be spelled differently in entries.parquet")
