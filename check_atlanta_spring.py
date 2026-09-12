"""Verify Atlanta Autotrader 400 (Feb 22 2026) is in the parquets before
adding it to RACES. Prints the race row and driver list with finish positions."""
import pandas as pd

races = pd.read_parquet("data/processed/races.parquet")
entries = pd.read_parquet("data/processed/entries.parquet")
races["date"] = pd.to_datetime(races["date"])

target = races[
    (races["date"] == pd.Timestamp("2026-02-22"))
    & (races["season"] == 2026)
]
if target.empty:
    print("NOT FOUND in races.parquet — need to scrape this race first")
    print("\nAll Feb 2026 races on file:")
    print(races[(races["date"] >= "2026-02-01") & (races["date"] < "2026-03-01")]
          [["date", "race_name", "track_name"]].to_string(index=False))
else:
    row = target.iloc[0]
    print(f"Found: {row['race_name']}")
    print(f"  race_id_short: {row['race_id_short']}")
    print(f"  track: {row['track_name']} ({row['track_type']})")
    print(f"  date: {row['date'].date()}")

    e = entries[entries["race_id_short"] == row["race_id_short"]]
    e = e[e["finish_pos"] > 0].sort_values("finish_pos")
    print(f"\n  {len(e)} drivers with finish positions:")
    print(e[["finish_pos", "driver", "make", "status"]].head(45).to_string(index=False))
