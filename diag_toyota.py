from diagnose_v5 import run_one
from backtest_oddslogic_v5 import RACES
import pandas as pd

races = pd.read_parquet("data/processed/races.parquet")
entries = pd.read_parquet("data/processed/entries.parquet")
sessions = pd.read_parquet("data/processed/sessions.parquet")
loopstats = pd.read_parquet("data/processed/loopstats.parquet")
laptimes = pd.read_parquet("data/processed/laptimes.parquet")
races["date"] = pd.to_datetime(races["date"])
entries["date"] = pd.to_datetime(entries["date"])

for date, name, matchups in RACES:
    if ("Toyota" in name) or ("Save Mart" in name):
        run_one(date, name, matchups, races, entries, sessions, loopstats, laptimes)
