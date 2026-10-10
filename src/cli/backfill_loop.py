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


def backfill_loop(sleep: float = 0.3, refetch: bool = False, base: Path = PROCESSED) -> pd.DataFrame:
    races = pd.read_parquet(base / "races.parquet")
    path = base / "loopstats.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    # INCREMENTAL by default: only races not already on disk. The old version
    # re-downloaded everything and OVERWROTE the file, so any race whose fetch
    # failed (e.g. a 403) silently vanished from history.
    if not refetch and not existing.empty:
        have = set(existing["race_id_short"].unique())
        races = races[~races["race_id_short"].isin(have)]
    log.info("Fetching loopstats for %s races (refetch=%s)", len(races), refetch)
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

    new = pd.concat(all_rows, ignore_index=True)
    if not existing.empty:
        keep = existing[~existing["race_id_short"].isin(set(new["race_id_short"]))]
        out = pd.concat([keep, new], ignore_index=True)
    else:
        out = new
    log.info("Fetched %s new races; merged total %s races", new["race_id_short"].nunique(),
             out["race_id_short"].nunique())
    base.mkdir(parents=True, exist_ok=True)
    out.to_parquet(base / "loopstats.parquet", index=False)
    log.info("Wrote %s loopstats rows across %s races", len(out), out["race_id_short"].nunique())
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--refetch", action="store_true", help="re-download every race (merge, never drop)")
    p.add_argument("--history", action="store_true", help="use data/processed/history/ (pre-2022)")
    args = p.parse_args(argv)
    backfill_loop(args.sleep, refetch=args.refetch,
                  base=(PROCESSED / "history") if args.history else PROCESSED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
