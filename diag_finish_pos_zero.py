"""How many finish_pos=0 rows exist in entries.parquet, and where?

If any race has a finish_pos=0 row that sorts before the real winner,
that race's row went into update_from_race as the winner, corrupting
Plackett-Luce ratings for every subsequent race.

Reports:
  - count and per-season breakdown of finish_pos <= 0 rows among "run" races
  - which races were corrupted (a finish_pos=0 that sorts BEFORE finish_pos=1)
  - whether the corrupted races are in training/backtest ranges
"""
from __future__ import annotations

import pandas as pd

entries = pd.read_parquet("data/processed/entries.parquet")
races = pd.read_parquet("data/processed/races.parquet")
races["date"] = pd.to_datetime(races["date"])
entries["date"] = pd.to_datetime(entries["date"])

# Which races have been "run" (any driver finished)?
race_max_fp = entries.groupby("race_id_short")["finish_pos"].max()
run_races = set(race_max_fp[race_max_fp > 0].index)
print(f"Races with any finished driver: {len(run_races)} of {entries['race_id_short'].nunique()}")

# All rows with finish_pos <= 0 in run races (these are the suspect rows).
suspect = entries[
    (entries["race_id_short"].isin(run_races))
    & (entries["finish_pos"] <= 0)
].copy()

print(f"\nfinish_pos <= 0 rows in run races: {len(suspect)}")
if len(suspect) == 0:
    print("No corrupted rows found — PL ratings are clean.")
    raise SystemExit(0)

# By season.
print("\nBy season:")
by_season = suspect.groupby("season").size().rename("suspect_rows")
print(by_season.to_string())

# Which races are impacted? A race is impacted if it has a finish_pos<=0
# AND a finish_pos==1 (real winner). The 0-row sorts before the 1-row, so
# the 0-row's driver becomes "winner" in PL updates.
impacted = []
for rid, sub in suspect.groupby("race_id_short"):
    all_rows = entries[entries["race_id_short"] == rid]
    has_real_winner = (all_rows["finish_pos"] == 1).any()
    if has_real_winner:
        race_info = races[races["race_id_short"] == rid]
        if race_info.empty:
            continue
        rr = race_info.iloc[0]
        impacted.append({
            "race_id": rid,
            "date": rr["date"].date(),
            "season": int(rr["season"]),
            "race_name": rr["race_name"],
            "track_type": rr["track_type"],
            "n_suspect_rows": len(sub),
            "suspect_drivers": ", ".join(sub["driver"].tolist()[:3])
                + (f" (+{len(sub)-3} more)" if len(sub) > 3 else ""),
        })

print(f"\n{len(impacted)} races had corrupted PL updates:")
if impacted:
    df = pd.DataFrame(impacted).sort_values("date")
    print(df.to_string(index=False))
    print(f"\nEarliest corruption: {df['date'].min()}")
    print(f"Latest corruption:   {df['date'].max()}")
    print(f"\nEvery race AFTER {df['date'].min()} has been trained on")
    print("a Plackett-Luce state that includes the fake-winner update(s).")

    # Any of these in the 2026 backtest window?
    backtest_2026 = df[df["season"] == 2026]
    if len(backtest_2026):
        print(f"\n*** {len(backtest_2026)} corrupted races in the 2026 backtest sample ***")
