"""One-time (or refresh) fetch of Open-Meteo weather for every race in races.parquet.

Writes data/processed/weather.parquet. Safe to re-run — only fetches races not
already cached. Pass --refresh to re-fetch everything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.weather import load_or_fetch_weather


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore existing cache and re-fetch all rows.")
    args = ap.parse_args()

    races = pd.read_parquet("data/processed/races.parquet")
    races["date"] = pd.to_datetime(races["date"])
    print(f"Priming weather cache for {len(races)} races...")
    df = load_or_fetch_weather(races, refresh=args.refresh)
    print(f"\nTotal cached rows: {len(df)}")
    print(f"Rows with valid temp:     {df['temp_max_f'].notna().sum()}")
    print(f"Rows with valid wind:     {df['wind_max_mph'].notna().sum()}")
    print(f"Rows with valid precip:   {df['precip_sum_in'].notna().sum()}")
    print(f"\nSample:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
