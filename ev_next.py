"""EV analysis for posted matchups on the next race.

Uses the shared honest-calibration pipeline in src.models.predict_pipeline —
identical to the backtest, so model probabilities here are the same
probabilities the backtest measures. No divergence between "what we bet"
and "what we backtested."
"""
from __future__ import annotations

import pandas as pd

from backtest_oddslogic_v5 import (
    TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, DNF_DISPERSION, DAMAGE_PENALTY,
    american_to_prob, per_driver_hazards, per_driver_damage_hazards,
)
from src.models.predict_pipeline import calibrate_and_sample


# Bristol Motor Speedway — Bass Pro Shops Night Race (playoff, first round cutoff).
TARGET_NAME = "Bass Pro Shops"
TARGET_DATE = "2026-09-20"


# Posted matchups: (driver_a, driver_b, odds_a, odds_b)
# Update per weekend from the sportsbook you're pricing against.
MATCHUPS: list[tuple[str, str, int, int]] = [
    # Paste FanDuel (or preferred sharp book) matchups here once posted.
]


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def main():
    from src.features.build_features import build_features

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features...")
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    target_ts = pd.Timestamp(TARGET_DATE)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(TARGET_NAME, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    if r.empty:
        print("Race not found — check TARGET_NAME/TARGET_DATE.")
        return
    rid = r.iloc[0]["race_id_short"]
    target_date_ts = r.iloc[0]["date"]
    from src.features.tracks import resolve_track_type
    tt = resolve_track_type(r.iloc[0].get("track_name", ""),
                            fallback=r.iloc[0]["track_type"])
    print(f"Target: {r.iloc[0]['race_name']}  ({tt}, {target_date_ts.date()})")

    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == rid].reset_index(drop=True)
    if target.empty:
        print("No entry list yet — scrape first.")
        return

    print("Training + calibrating (leave-one-race-out CV)...")
    sr = calibrate_and_sample(
        train, target, tt,
        tight_reg=TIGHT_REG, alpha=ALPHA, n_samples=N_SAMPLES,
        hazard_lookup=HAZARD,
        per_driver_hazards_fn=per_driver_hazards,
        dnf_dispersion_by_type=DNF_DISPERSION,
        per_driver_damage_hazards_fn=per_driver_damage_hazards,
        damage_penalty=DAMAGE_PENALTY,
    )

    driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
    rows, unresolved = [], []
    for a, b, oa, ob in MATCHUPS:
        if a not in driver_to_idx or b not in driver_to_idx:
            unresolved.append((a, b))
            continue
        i, j = driver_to_idx[a], driver_to_idx[b]
        p_a = float(sr.matchup_mtx[i, j]); p_b = 1 - p_a

        ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma, mb = ma_raw / vig, mb_raw / vig

        b_dec_a = american_to_decimal(oa) - 1
        b_dec_b = american_to_decimal(ob) - 1
        ev_a = p_a * b_dec_a - (1 - p_a)
        ev_b = p_b * b_dec_b - (1 - p_b)

        if ev_a > ev_b:
            pick, pick_p, pick_odds, pick_ev, pick_market = "A", p_a, oa, ev_a, ma
        else:
            pick, pick_p, pick_odds, pick_ev, pick_market = "B", p_b, ob, ev_b, mb
        pick_driver = a if pick == "A" else b
        other_driver = b if pick == "A" else a
        edge = pick_p - pick_market

        rows.append({
            "pick": pick_driver,
            "against": other_driver,
            "odds": pick_odds,
            "model_p": pick_p,
            "market_p_vigfree": pick_market,
            "edge": edge,
            "ev_per_dollar": pick_ev,
        })

    df = pd.DataFrame(rows).sort_values("ev_per_dollar", ascending=False)
    fmt = {
        "model_p": "{:.1%}".format,
        "market_p_vigfree": "{:.1%}".format,
        "edge": "{:+.1%}".format,
        "ev_per_dollar": "{:+.4f}".format,
    }

    print("\n" + "=" * 90)
    print("BEST-EV MATCHUPS")
    print("=" * 90)
    print(df.to_string(index=False, formatters=fmt))

    print("\n" + "=" * 90)
    print("BETS TO TAKE (edge >= 30% — backtest-validated threshold)")
    print("=" * 90)
    plus = df[df["edge"] >= 0.30]
    if len(plus) > 0:
        print(plus.to_string(index=False, formatters=fmt))
        print(f"\n{len(plus)} bets with >=30% edge, mean EV = ${plus['ev_per_dollar'].mean():.4f}/$")
    else:
        print("No matchups clear the 30% edge threshold.")

    print("\n" + "=" * 90)
    print("Marginal (10-30% edge — do NOT bet these; shown for context)")
    print("=" * 90)
    mid = df[(df["edge"] >= 0.10) & (df["edge"] < 0.30)]
    if len(mid) > 0:
        print(mid.to_string(index=False, formatters=fmt))

    if unresolved:
        print(f"\n{len(unresolved)} matchups unresolved (driver not in entry list):")
        for a, b in unresolved:
            print(f"  {a} vs {b}")


if __name__ == "__main__":
    main()
