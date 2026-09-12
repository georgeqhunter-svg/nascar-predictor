"""Incremental loopstats backfill.

Reads existing data/processed/races.parquet, fetches loopstats for every race,
writes data/processed/loopstats.parquet with one row per (race, driver).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..scrape.nascar_loop import fetch_loopstats

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def backfill_loop(sleep: float = 0.3) -> pd.DataFrame:
    races = pd.read_parquet(PROCESSED / "races.parquet")
    all_rows = []
    for _, row in tqdm(races.iterrows(), total=len(races), desc="loopstats", unit="race"):
        try:
            df = fetch_loopstats(int(row["season"]), int(row["race_id_nascar"]))
        except Exception as e:
            log.warning("loopstats failed for %s: %s", row.get("race_name"), e)
            continue
        if df.empty:
            continue
        df["race_id_short"] = row["race_id_short"]
        df["season"] = row["season"]
        df["date"] = pd.Timestamp(row["date"])
        all_rows.append(df)
        time.sleep(sleep)

    if not all_rows:
        log.warning("no loopstats rows fetched")
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED / "loopstats.parquet", index=False)
    log.info("Wrote %s loopstats rows across %s races", len(out), out["race_id_short"].nunique())
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--sleep", type=float, default=0.3)
    args = p.parse_args(argv)
    backfill_loop(args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
