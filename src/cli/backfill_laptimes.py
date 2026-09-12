"""Incremental lap-times backfill.

Reads existing data/processed/races.parquet, fetches lap-times per race,
writes data/processed/laptimes.parquet (long format, one row per driver-lap).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..scrape.nascar_laptimes import fetch_laptimes

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def backfill_laptimes(sleep: float = 0.3) -> pd.DataFrame:
    races = pd.read_parquet(PROCESSED / "races.parquet")
    all_rows = []
    for _, row in tqdm(races.iterrows(), total=len(races), desc="laptimes", unit="race"):
        try:
            df = fetch_laptimes(int(row["season"]), int(row["race_id_nascar"]))
        except Exception as e:
            log.warning("laptimes failed for %s: %s", row.get("race_name"), e)
            continue
        if df.empty:
            continue
        df["race_id_short"] = row["race_id_short"]
        df["season"] = row["season"]
        all_rows.append(df)
        time.sleep(sleep)

    if not all_rows:
        log.warning("no laptimes rows fetched")
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED / "laptimes.parquet", index=False)
    log.info("Wrote %s laptime rows across %s races (%s MB)",
             f"{len(out):,}", out["race_id_short"].nunique(),
             int((PROCESSED / "laptimes.parquet").stat().st_size / 1024 / 1024))
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--sleep", type=float, default=0.3)
    args = p.parse_args(argv)
    backfill_laptimes(args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
