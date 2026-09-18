"""How often do we predict on unseen categorical values in the CV / backtest?

For each backtest race:
  - `train` = features before target date (what the primary model trains on)
  - `train_minus` = train excluding a given val race (what the CV refit sees)

Checks:
  1. Which target-race drivers/teams/manufacturers are NOT in `train`?
  2. Across all backtest races, how many unique (race, driver) pairs is that?
  3. For CV refits: are val-race drivers ever absent from `train_minus`?
     (This is the mechanism kimi flagged for T fitting on wrong codes.)
  4. Impact test: swap an unseen driver for a random veteran and diff the
     predicted score. A big delta = LightGBM is genuinely using the raw
     categorical code (bad); ~0 delta = the categorical feature is inert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import RACES, TIGHT_REG


def main():
    from src.features.build_features import build_features
    from src.models.gbm_ranker import GBMEnsemble

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...")
    features = build_features(races, entries, sessions,
                              loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    print("\n" + "=" * 70)
    print("Backtest target races: unseen driver / team / manufacturer counts")
    print("=" * 70)
    total_unseen = []
    for date, name, _ in RACES:
        r = races[
            ((races["date"] == pd.Timestamp(date))
             | (races["race_name"].str.contains(name, case=False, na=False)))
            & (races["season"] == 2026)
        ]
        if r.empty:
            continue
        rid = r.iloc[0]["race_id_short"]
        tdate = r.iloc[0]["date"]
        train = features[(features["date"] < tdate) & (features["finish_pos"] > 0)]
        target = features[features["race_id_short"] == rid]

        train_drivers = set(train["driver"].unique())
        train_teams = set(train["team"].unique())
        train_makes = set(train["manufacturer"].unique()) if "manufacturer" in train else set()

        unseen_drv = set(target["driver"].unique()) - train_drivers
        unseen_team = set(target["team"].unique()) - train_teams
        unseen_make = (set(target["manufacturer"].unique()) - train_makes) if "manufacturer" in target else set()
        total_unseen.append({
            "race": name,
            "unseen_drivers": len(unseen_drv),
            "unseen_teams": len(unseen_team),
            "unseen_makes": len(unseen_make),
            "example_drivers": ", ".join(list(unseen_drv)[:3]),
        })

    df = pd.DataFrame(total_unseen)
    print(df.to_string(index=False))
    print(f"\nSum across 23 races: {df['unseen_drivers'].sum()} unseen driver-race pairs, "
          f"{df['unseen_teams'].sum()} unseen team-race, "
          f"{df['unseen_makes'].sum()} unseen manuf-race.")

    # -------- CV refit check: unseen at train_minus time --------
    print("\n" + "=" * 70)
    print("CV refits: drivers in val_race but NOT in train_minus")
    print("=" * 70)
    cv_unseen_total = 0
    n_val_races_checked = 0
    for date, name, _ in RACES[:5]:  # spot check first 5 backtest races
        r = races[
            ((races["date"] == pd.Timestamp(date))
             | (races["race_name"].str.contains(name, case=False, na=False)))
            & (races["season"] == 2026)
        ]
        if r.empty:
            continue
        tdate = r.iloc[0]["date"]
        train = features[(features["date"] < tdate) & (features["finish_pos"] > 0)]
        race_order = (train.groupby("race_id_short")["date"].first()
                      .sort_values().index.tolist())
        val_ids = race_order[30:]
        for vid in val_ids[:20]:  # spot check first 20 val races per backtest race
            sub = train[train["race_id_short"] == vid]
            train_minus = train[train["race_id_short"] != vid]
            unseen = set(sub["driver"].unique()) - set(train_minus["driver"].unique())
            if unseen:
                cv_unseen_total += len(unseen)
            n_val_races_checked += 1
    print(f"Val races spot-checked: {n_val_races_checked}, unseen-in-train_minus: {cv_unseen_total}")

    # -------- Impact test: does driver categorical actually carry weight? --------
    print("\n" + "=" * 70)
    print("Impact test: swap driver categorical, measure predicted-score delta")
    print("=" * 70)
    # Use the last-race target as the impact test case.
    date, name, _ = RACES[-1]
    r = races[
        ((races["date"] == pd.Timestamp(date))
         | (races["race_name"].str.contains(name, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    rid = r.iloc[0]["race_id_short"]
    tdate = r.iloc[0]["date"]
    train = features[(features["date"] < tdate) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == rid].reset_index(drop=True)
    model = GBMEnsemble()
    model.fit(train, n_estimators=5, **TIGHT_REG)  # small ensemble for speed
    raw_scores = model.predict_scores(target)
    # Pick the top-3 drivers by predicted score and swap them in the row.
    picks = target["driver"].tolist()
    for row_idx in [0, len(target) // 2, len(target) - 1]:
        original_driver = picks[row_idx]
        original_score = float(raw_scores[row_idx])
        # Swap this row's driver to a well-known veteran.
        swap_veteran = "Denny Hamlin" if original_driver != "Denny Hamlin" else "Kyle Larson"
        swapped = target.copy()
        swapped.loc[row_idx, "driver"] = swap_veteran
        swapped_score = float(model.predict_scores(swapped)[row_idx])
        print(f"  row {row_idx} ({original_driver:>22}): "
              f"score {original_score:+.3f}  swap->{swap_veteran}: {swapped_score:+.3f}  "
              f"delta {swapped_score - original_score:+.3f}")

    print("\nInterpretation:")
    print("  |delta| > 0.5  = categorical carries meaningful weight (worth fixing)")
    print("  |delta| < 0.1  = feature is largely inert, categorical fix is hygiene only")


if __name__ == "__main__":
    main()
