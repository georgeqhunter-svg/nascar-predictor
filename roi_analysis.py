"""ROI analysis instead of log-loss.

The right question is not "is our log-loss below the closing line's" but
"if we bet our picks at standard -110 lines, do we make money?"

Break-even at -110 vig is a 52.38% hit rate. Anything above that is profit,
before considering that we could bet at BETTER-than-closing lines earlier
in the week (positive closing-line value).

This script re-runs the OddsLogic backtest but instead of log-loss, it
computes:

  1. Raw ROI if we bet every matchup at -110 on our preferred side.
  2. Filtered ROI: only bet when our probability exceeds the vig-free
     market implied probability by an EDGE_THRESHOLD.
  3. Same but at multiple thresholds so we can see the frontier.
  4. Kelly-fraction ROI for comparison.

Uses the same features / model / calibration as backtest_oddslogic_v5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, american_to_prob, per_driver_hazards,
)


VIG_ODDS = -110  # standard book price
BREAK_EVEN_PROB = 110 / (110 + 100)  # 0.5238 at -110

EDGE_THRESHOLDS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]


def payout_per_dollar_at_odds(odds: int) -> float:
    """For a $1 stake at American odds, net profit if the bet wins."""
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble

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
    print(f"Done. {len(features)} rows.\n")

    all_bets = []  # list of dicts, one per matchup

    for date, name, matchups in RACES:
        target_ts = pd.Timestamp(date)
        r = races[
            ((races["date"] == target_ts)
             | (races["race_name"].str.contains(name, case=False, na=False)))
            & (races["season"] == 2026)
        ]
        if r.empty:
            continue
        rid = r.iloc[0]["race_id_short"]
        target_date_ts = r.iloc[0]["date"]
        tt = r.iloc[0]["track_type"]

        train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
        target = features[features["race_id_short"] == rid].reset_index(drop=True)
        if target.empty:
            continue

        model = GBMEnsemble()
        model.fit(train, n_estimators=15, **TIGHT_REG)

        race_order = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
        val_ids = race_order[30:]
        val_races, tt_list = [], []
        for vrid in val_ids:
            sub = train[train["race_id_short"] == vrid]
            raw = model.predict_scores(sub)
            gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
            pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                    / (sub["pl_effective"].std() + 1e-9))
            b = ALPHA * gbm_z + (1 - ALPHA) * pl_z
            b = b / max(b.std(), 1e-6)
            val_races.append(RaceScoresGT(
                scores=b, finishes=sub["finish_pos"].to_numpy(),
                is_dnf=sub["is_dnf"].to_numpy(),
                hazards=np.full(len(sub), 0.08),  # baseline; per-driver reverted after calibration issue
            ))
            tt_list.append(sub["track_type"].iloc[0])
        T_by_type = find_best_temperature_by_type(val_races, tt_list, default_T=1.0, n_samples=1500)
        T = T_by_type.get(tt, 1.0)

        raw = model.predict_scores(target)
        gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
        pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
                / (target["pl_effective"].std() + 1e-9))
        blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
        blended = blended / max(blended.std(), 1e-6)
        haz = per_driver_hazards(target, tt)
        rng = np.random.default_rng(42)
        positions = dist.sample_finishing_orders(blended / T, haz, n_samples=N_SAMPLES, rng=rng)
        matchup_mtx = dist.matchup_probs(positions)

        driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
        actual = entries[entries["race_id_short"] == rid][
            ["driver", "finish_pos"]
        ].set_index("driver")

        for a, b, oa, ob in matchups:
            if a not in driver_to_idx or b not in driver_to_idx: continue
            if a not in actual.index or b not in actual.index: continue

            ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
            vig = ma_raw + mb_raw
            ma, mb = ma_raw / vig, mb_raw / vig

            i, j = driver_to_idx[a], driver_to_idx[b]
            p_a = float(matchup_mtx[i, j])

            fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
            a_won = fa < fb

            # If model likes A more than market does, bet A. Otherwise B.
            if p_a >= 0.5:
                pick, pick_p, market_p, won = a, p_a, ma, a_won
            else:
                pick, pick_p, market_p, won = b, 1 - p_a, mb, not a_won

            edge = pick_p - market_p  # positive = model sees more value than market
            all_bets.append({
                "race": name, "tt": tt,
                "pick": pick, "won": int(won),
                "model_p": pick_p, "market_p": market_p, "edge": edge,
            })

        print(f"  processed {name}: {len([b for b in all_bets if b['race']==name])} bets")

    df = pd.DataFrame(all_bets)
    print(f"\nTotal bets: {len(df)}")

    print("\n" + "=" * 70)
    print("FLAT ROI at -110 lines (all bets at $1 stake)")
    print("=" * 70)
    net = df["won"] * payout_per_dollar_at_odds(VIG_ODDS) - (1 - df["won"])
    total_bets = len(df)
    total_wins = df["won"].sum()
    total_profit = net.sum()
    print(f"  n={total_bets}  hits={total_wins}/{total_bets} ({total_wins/total_bets*100:.1f}%)  "
          f"profit=${total_profit:.2f}  ROI={total_profit/total_bets*100:+.2f}%")

    print("\n" + "=" * 70)
    print("ROI vs edge threshold (only bet when model_p - market_p >= threshold)")
    print("=" * 70)
    rows = []
    for th in EDGE_THRESHOLDS:
        sub = df[df["edge"] >= th]
        if sub.empty:
            continue
        n = len(sub)
        wins = sub["won"].sum()
        net = sub["won"] * payout_per_dollar_at_odds(VIG_ODDS) - (1 - sub["won"])
        profit = net.sum()
        rows.append({
            "min_edge": th, "n_bets": n, "hit_rate_pct": wins / n * 100,
            "profit_dollars": profit, "roi_pct": profit / n * 100,
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n" + "=" * 70)
    print("Kelly-fraction ROI (fractional-Kelly stake proportional to edge)")
    print("=" * 70)
    # Simple half-Kelly for stability
    kelly_stake = np.maximum(
        0.5 * (df["model_p"] * (payout_per_dollar_at_odds(VIG_ODDS) + 1) - 1)
        / payout_per_dollar_at_odds(VIG_ODDS), 0
    )
    kelly_return = (df["won"] * payout_per_dollar_at_odds(VIG_ODDS) - (1 - df["won"])) * kelly_stake
    print(f"  Total half-Kelly stake: ${kelly_stake.sum():.2f}")
    print(f"  Total profit:           ${kelly_return.sum():.2f}")
    if kelly_stake.sum() > 0:
        print(f"  Kelly ROI:              {kelly_return.sum() / kelly_stake.sum() * 100:+.2f}%")

    print("\n" + "=" * 70)
    print("Per-track-type breakdown at 0% edge threshold")
    print("=" * 70)
    for tt, sub in df.groupby("tt"):
        n = len(sub); wins = sub["won"].sum()
        net = sub["won"] * payout_per_dollar_at_odds(VIG_ODDS) - (1 - sub["won"])
        profit = net.sum()
        print(f"  {tt:15s}  n={n:4d}  hits={wins}/{n} ({wins/n*100:5.1f}%)  "
              f"profit=${profit:+6.2f}  ROI={profit/n*100:+.2f}%")


if __name__ == "__main__":
    main()
