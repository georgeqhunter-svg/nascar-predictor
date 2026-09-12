"""Run the OddsLogic backtest for a single race.

Usage:
    python backtest_one.py                    # runs the LAST race in RACES
    python backtest_one.py "Coca-Cola"        # runs the race whose name matches
    python backtest_one.py 2026-05-24         # runs the race on this date
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import RACES, run_race


def pick(arg: str | None):
    if not arg:
        return RACES[0]  # most recently added is at the top of RACES
    for entry in RACES:
        date, name, _ = entry
        if arg.lower() in name.lower() or arg == date:
            return entry
    print(f"No race matched '{arg}'. Available:")
    for date, name, _ in RACES:
        print(f"  {date}  {name}")
    sys.exit(1)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    date, name, matchups = pick(arg)

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    from src.features.build_features import build_features
    print("Building features...")
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    print(f"Running: {date}  {name}  ({len(matchups)} matchups)\n")
    result = run_race(date, name, matchups, races, entries, sessions,
                      loopstats, laptimes, features=features)
    if result:
        print(f"\nMarket: log-loss {result['market_ll']:.4f}, "
              f"correct {result['market_correct']}/{result['n']} "
              f"({result['market_correct']/result['n']*100:.1f}%)")
        print(f"Model:  log-loss {result['model_ll']:.4f}, "
              f"correct {result['model_correct']}/{result['n']} "
              f"({result['model_correct']/result['n']*100:.1f}%)")
        print(f"Delta:  {result['model_ll'] - result['market_ll']:+.4f}")


if __name__ == "__main__":
    main()
