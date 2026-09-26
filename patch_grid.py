"""Patch a race's starting grid into entries.parquet (reusable every week).

Usage:
    python patch_grid.py 2026-5628 grids/kansas_2026-09-27.txt
    python patch_grid.py 2026-5628 grids/kansas_2026-09-27.txt --speeds

Grid file: one driver per line, "position, driver name" (optionally ", speed").
Lines starting with # are ignored. Driver names are accent- and middle-name-
insensitive ("Daniel Suarez" matches "Daniel Suárez", "John Hunter Nemechek"
matches "John H. Nemechek").

Sets start_pos and qual_pos to the grid position. With --speeds, also sets
qual_speed from the third column (use when real qualifying happened; leave off
for rainout / formula grids so qual_speed stays 0).

Only needed when NASCAR.com's cacher hasn't posted qualifying yet — try
`python -m src.cli.backfill --seasons 2026` first; if start_pos comes through,
you don't need this.
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd


def _norm(s: str) -> str:
    n = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower()
    parts = n.split()
    if len(parts) >= 3:
        n = f"{parts[0]} {' '.join(m.rstrip('.')[0] for m in parts[1:-1] if m)} {parts[-1]}"
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("race_id", help="race_id_short, e.g. 2026-5628")
    ap.add_argument("grid", help="grid file path")
    ap.add_argument("--speeds", action="store_true", help="also write qual_speed (col 3)")
    args = ap.parse_args()

    grid = []
    for line in Path(args.grid).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        pos, drv = int(parts[0]), parts[1]
        spd = float(parts[2]) if args.speeds and len(parts) > 2 else None
        grid.append((pos, drv, spd))

    path = Path("data/processed/entries.parquet")
    e = pd.read_parquet(path)
    in_race = e["race_id_short"] == args.race_id
    if not in_race.any():
        raise SystemExit(f"No entries for {args.race_id} — run backfill first.")
    e["qual_speed"] = e["qual_speed"].astype(float)
    key = e.loc[in_race, "driver"].map(_norm)
    idx_by_key = dict(zip(key, e.index[in_race]))

    patched, missing = 0, []
    for pos, drv, spd in grid:
        i = idx_by_key.get(_norm(drv))
        if i is None:
            missing.append(drv)
            continue
        e.at[i, "start_pos"] = pos
        e.at[i, "qual_pos"] = pos
        if spd is not None:
            e.at[i, "qual_speed"] = spd
        patched += 1

    e.to_parquet(path, index=False)
    print(f"Patched {patched}/{len(grid)} drivers for {args.race_id} "
          f"({int(in_race.sum())} on entry list)")
    if missing:
        print("NOT on entry list:", missing)
    unfilled = set(e.loc[in_race & (e["start_pos"] <= 0), "driver"])
    if unfilled:
        print("Entry-list drivers with no grid spot:", sorted(unfilled))


if __name__ == "__main__":
    main()
