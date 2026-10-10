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


def backfill_laptimes(sleep: float = 0.3, refetch: bool = False) -> pd.DataFrame:
    races = pd.read_parquet(PROCESSED / "races.parquet")
    path = PROCESSED / "laptimes.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    # INCREMENTAL by default: only races not already on disk. The old version
    # re-downloaded everything and OVERWROTE the file, so any race whose fetch
    # failed (e.g. a 403) silently vanished from history.
    if not refetch and not existing.empty:
        have = set(existing["race_id_short"].unique())
        races = races[~races["race_id_short"].isin(have)]
    log.info("Fetching laptimes for %s races (refetch=%s)", len(races), refetch)
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

    new = pd.concat(all_rows, ignore_index=True)
    if not existing.empty:
        keep = existing[~existing["race_id_short"].isin(set(new["race_id_short"]))]
        out = pd.concat([keep, new], ignore_index=True)
    else:
        out = new
    log.info("Fetched %s new races; merged total %s races", new["race_id_short"].nunique(),
             out["race_id_short"].nunique())
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
    p.add_argument("--refetch", action="store_true", help="re-download every race (merge, never drop)")
    args = p.parse_args(argv)
    backfill_laptimes(args.sleep, refetch=args.refetch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
