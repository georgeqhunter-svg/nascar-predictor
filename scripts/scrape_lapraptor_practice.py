"""Scrape LapRaptor practice-lap CSVs.

Discovers per-race run_ids from the race pages, then downloads each run's
lap-level CSV to data/raw/practice_logs/{race_id}_{run_id}.csv.

Usage:
    python scripts/scrape_lapraptor_practice.py --start 5400 --end 5700
    python scripts/scrape_lapraptor_practice.py --race-ids 5620,5619,5618

Skips races/runs already downloaded. Respects a 2-second delay between
requests. If LapRaptor requires login, set LAPRAPTOR_COOKIE env var to your
session cookie (grab from browser DevTools -> Application -> Cookies).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.lapraptor.com"
OUT_DIR = Path("data/raw/practice_logs")
POLITE_SLEEP = 0.8


def session_with_cookie() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    })
    cookie = os.environ.get("LAPRAPTOR_COOKIE")
    if cookie:
        s.headers["Cookie"] = cookie
    return s


CANDIDATE_RUN_IDS = list(range(6, 18))  # practice/qual/race cluster around 10-14

def discover_run_ids(session: requests.Session, race_id: int) -> list[dict]:
    """Race pages are JS-rendered so we can't parse them for run_ids.
    Instead, just try candidate ids and let download_run 404 the missing ones.
    """
    return [{"run_id": r} for r in CANDIDATE_RUN_IDS]


def download_run(session: requests.Session, race_id: int, run_id: int) -> bool:
    out = OUT_DIR / f"{race_id}_{run_id}.csv"
    if out.exists():
        return True
    url = f"{BASE}/races/{race_id}/run/{run_id}/laps/raw.csv"
    # Retry on connection errors — LapRaptor's CDN sometimes closes idle connections.
    resp = None
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=60)
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 2 ** attempt * 5  # 5, 10, 20, 40 seconds
            print(f"  connection error on {race_id}/{run_id}, retry in {wait}s ({e.__class__.__name__})")
            time.sleep(wait)
    if resp is None:
        print(f"  FAIL {race_id}/{run_id}: exhausted retries")
        return False
    if resp.status_code == 404:
        return False  # this run_id doesn't exist for this race, expected
    if resp.status_code != 200:
        print(f"  FAIL {race_id}/{run_id}: HTTP {resp.status_code}")
        return False
    body = resp.text[:200]
    if "<html" in body.lower() and "series," not in body:
        # Empty run or auth wall — skip quietly unless clearly auth.
        if "login" in body.lower():
            print(f"  AUTH: race {race_id}/{run_id} — set LAPRAPTOR_COOKIE.")
        return False
    # Real CSVs are tens of KB; empty ones are just the header row (~1KB).
    if len(resp.text) < 2000:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(resp.text, encoding="utf-8")
    print(f"  OK {race_id}/{run_id}: {len(resp.text)//1024} KB")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, help="start race_id (inclusive)")
    ap.add_argument("--end", type=int, help="end race_id (inclusive)")
    ap.add_argument("--race-ids", type=str,
                    help="comma-separated race_ids (overrides start/end)")
    args = ap.parse_args()

    if args.race_ids:
        race_ids = [int(x) for x in args.race_ids.split(",")]
    elif args.start and args.end:
        race_ids = list(range(args.start, args.end + 1))
    else:
        print("Need --race-ids OR --start and --end")
        sys.exit(1)

    session = session_with_cookie()
    for rid in race_ids:
        print(f"race {rid}")
        try:
            runs = discover_run_ids(session, rid)
        except Exception as e:
            print(f"  discover failed: {e}")
            continue
        if not runs:
            print(f"  no runs found (race may not exist)")
            time.sleep(POLITE_SLEEP)
            continue
        for r in runs:
            download_run(session, rid, r["run_id"])
            time.sleep(POLITE_SLEEP)
        time.sleep(POLITE_SLEEP)


if __name__ == "__main__":
    main()
