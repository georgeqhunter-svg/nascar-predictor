"""EV analysis for posted matchups on the next race.

Uses the shared honest-calibration pipeline in src.models.predict_pipeline —
identical to the backtest, so model probabilities here are the same
probabilities the backtest measures. No divergence between "what we bet"
and "what we backtested."
"""
from __future__ import annotations

import pandas as pd

from backtest_oddslogic_v5 import (
    TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, DNF_DISPERSION,
    DAMAGE_PENALTY, DAMAGE_PENALTY_BY_TYPE,
    CV_N_ESTIMATORS, VAL_CAP_PER_TYPE,
    american_to_prob, per_driver_hazards, per_driver_damage_hazards,
)
from src.models.predict_pipeline import calibrate_and_sample


# Kansas Speedway — Hollywood Casino 400 (2026-09-27).
TARGET_NAME = "Hollywood Casino 400"
TARGET_DATE = "2026-09-27"


# Circa Sports matchups for Bristol Bass Pro Shops Night Race (2026-09-19).
# Sharp book — edges will be smaller than FanDuel. Our +25% edge threshold
# from the backtest is calibrated against OddsLogic closes (sharp-book quality),
# so Circa edges going through it are honest.
# Format: (driver_a, driver_b, odds_a, odds_b)
MATCHUPS: list[tuple[str, str, int, int]] = [
    # Kansas 2026-09-27 — Circa: (driver_a, driver_b, odds_a, odds_b)
    ("Kyle Larson", "Denny Hamlin", -115, -115),
    ("Kyle Larson", "Christopher Bell", -140, 110),
    ("Kyle Larson", "Tyler Reddick", -160, 130),
    ("Kyle Larson", "Ryan Blaney", -260, 210),
    ("Denny Hamlin", "Christopher Bell", -140, 110),
    ("Denny Hamlin", "Tyler Reddick", -160, 130),
    ("Denny Hamlin", "Ryan Blaney", -250, 200),
    ("Christopher Bell", "Tyler Reddick", -140, 110),
    ("Christopher Bell", "Ryan Blaney", -220, 180),
    ("Tyler Reddick", "Ryan Blaney", -200, 165),
    ("Ty Gibbs", "Chase Briscoe", -130, 100),
    ("Ty Gibbs", "Bubba Wallace", -130, 100),
    ("Ty Gibbs", "Joey Logano", -150, 120),
    ("Ty Gibbs", "William Byron", -130, 100),
    ("Chase Briscoe", "William Byron", -120, -110),
    ("Chase Briscoe", "Bubba Wallace", -120, -110),
    ("Chase Briscoe", "Joey Logano", -130, 100),
    ("William Byron", "Bubba Wallace", -115, -115),
    ("William Byron", "Joey Logano", -120, -110),
    ("Bubba Wallace", "Joey Logano", -120, -110),
    ("Chase Elliott", "Chris Buescher", -185, 150),
    ("Chase Elliott", "Carson Hocevar", -110, -120),
    ("Chase Elliott", "Brad Keselowski", -190, 155),
    ("Chase Elliott", "Ross Chastain", -240, 195),
    ("Chris Buescher", "Carson Hocevar", 150, -185),
    ("Chris Buescher", "Brad Keselowski", -160, 130),
    ("Chris Buescher", "Ross Chastain", -170, 140),
    ("Carson Hocevar", "Brad Keselowski", -200, 165),
    ("Carson Hocevar", "Ross Chastain", -230, 185),
    ("Brad Keselowski", "Ross Chastain", -160, 130),
]

# Frozen betting rule (set before Kansas lines): edge >= 20%, flat small stakes.
EDGE_THRESHOLD = 0.20


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
        damage_penalty_by_type=DAMAGE_PENALTY_BY_TYPE,
        cv_n_estimators=CV_N_ESTIMATORS,
        val_cap_per_type=VAL_CAP_PER_TYPE,
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
    pct = int(EDGE_THRESHOLD * 100)
    print(f"BETS TO TAKE (edge >= {pct}% — frozen rule)")
    print("=" * 90)
    plus = df[df["edge"] >= EDGE_THRESHOLD]
    if len(plus) > 0:
        print(plus.to_string(index=False, formatters=fmt))
        print(f"\n{len(plus)} bets with >={pct}% edge, mean EV = ${plus['ev_per_dollar'].mean():.4f}/$")
    else:
        print(f"No matchups clear the {pct}% edge threshold.")

    print("\n" + "=" * 90)
    print(f"Marginal (10-{pct}% edge — do NOT bet these; shown for context)")
    print("=" * 90)
    mid = df[(df["edge"] >= 0.10) & (df["edge"] < EDGE_THRESHOLD)]
    if len(mid) > 0:
        print(mid.to_string(index=False, formatters=fmt))

    if unresolved:
        print(f"\n{len(unresolved)} matchups unresolved (driver not in entry list):")
        for a, b in unresolved:
            print(f"  {a} vs {b}")


if __name__ == "__main__":
    main()
