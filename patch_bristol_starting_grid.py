"""Patch Bristol Sept 20 2026 starting positions into entries.parquet.

Practice and qualifying both cancelled due to rain. NASCAR set the lineup by
formula (owner points + recent performance). qual_pos = start_pos = grid
position. qual_speed stays 0/null (no actual qualifying happened).

Source: NASCAR Cup Series starting lineup by row, Bristol Night Race,
provided by NASCAR Statistics 9/18/2026.
"""
from pathlib import Path
import pandas as pd


STARTING_GRID = [
    (1,  "Kyle Larson"),
    (2,  "Joey Logano"),
    (3,  "Denny Hamlin"),
    (4,  "William Byron"),
    (5,  "Carson Hocevar"),
    (6,  "Alex Bowman"),
    (7,  "Ross Chastain"),
    (8,  "Austin Cindric"),
    (9,  "Austin Dillon"),
    (10, "Bubba Wallace"),
    (11, "Brad Keselowski"),
    (12, "Christopher Bell"),
    (13, "Ty Gibbs"),
    (14, "Erik Jones"),
    (15, "Connor Zilisch"),
    (16, "Ryan Preece"),
    (17, "Cole Custer"),
    (18, "AJ Allmendinger"),
    (19, "John H. Nemechek"),  # entries.parquet uses "John H. Nemechek"
    (20, "Tyler Reddick"),
    (21, "Michael McDowell"),
    (22, "Shane Van Gisbergen"),
    (23, "Chris Buescher"),
    (24, "Todd Gilliland"),
    (25, "Austin Hill"),
    (26, "Chase Briscoe"),
    (27, "Chase Elliott"),
    (28, "Daniel Suárez"),  # entries.parquet uses accented "Suárez"
    (29, "Ryan Blaney"),
    (30, "Cody Ware"),
    (31, "Ricky Stenhouse Jr"),
    (32, "Josh Berry"),
    (33, "Ty Dillon"),
    (34, "Riley Herbst"),
    (35, "Zane Smith"),
    (36, "Noah Gragson"),
    (37, "Josh Bilicki"),
]

path = Path("data/processed/entries.parquet")
entries = pd.read_parquet(path)
entries["date"] = pd.to_datetime(entries["date"])

# Bristol Bass Pro Shops Night Race.
target = entries[
    (entries["date"] == pd.Timestamp("2026-09-19"))
    & (entries["season"] == 2026)
]
if target.empty:
    print("Bristol race not found — check date 2026-09-19")
    raise SystemExit(1)
rid = target["race_id_short"].iloc[0]
print(f"Patching {rid}, {len(target)} drivers on file")

# Rain-out: qual_speed remains 0 (no actual qualifying). Only start_pos +
# qual_pos get populated.
patched, unmatched = 0, []
for pos, driver in STARTING_GRID:
    mask = (entries["race_id_short"] == rid) & (entries["driver"] == driver)
    if not mask.any():
        unmatched.append(driver)
        continue
    entries.loc[mask, "start_pos"] = pos
    entries.loc[mask, "qual_pos"] = pos  # grid order treated as "qualifying" for the model
    patched += 1

entries.to_parquet(path, index=False)
print(f"Patched {patched}/{len(STARTING_GRID)} drivers")
if unmatched:
    print(f"NOT FOUND: {unmatched}")
    print("These names may be spelled differently in entries.parquet")
