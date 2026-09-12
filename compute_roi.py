"""Compute expected ROI under flat-unit and quarter-Kelly staking.

Runs the full backtest, then for each matchup:
  - Model probability (our estimate)
  - Vig-free market probability
  - Actual outcome
  - Both sides' American odds

For each bet we ONLY place when model probability > vig-free market probability
(i.e., we have positive expected value at the offered price).

Staking:
  Flat: 1 unit on underdog side (positive odds), or enough units to win 1
        on favorite side. Standard "1 unit to win 1 unit" convention.
  Quarter-Kelly: stake = 0.25 * (p_model * (b+1) - 1) / b, where b = decimal
                 odds - 1. Skip if Kelly is negative.

Reports total staked, total profit, ROI, per-race breakdown, and extrapolation
to a full 36-race Cup season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob,
)


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def american_to_stake_flat(odds: int) -> float:
    """Return the stake needed to win exactly 1 unit.
    Underdog (positive odds): stake 1, win odds/100 (we bet 1 unit).
    Favorite (negative odds): stake abs(odds)/100.
    Convention we're using: on underdog we bet 1 unit; on favorite we bet
    enough to win exactly 1 unit.
    """
    if odds > 0:
        return 1.0  # bet 1u, win odds/100 units
    else:
        return abs(odds) / 100.0  # bet this much, win 1u


def payout_win(odds: int, stake: float) -> float:
    """Profit if bet wins (not including original stake)."""
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))


def main():
    from src.features.build_features import build_features
    from src.models import distribution as dist
    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
    from src.models.gbm_ranker import GBMEnsemble
    from src.models.matchup_calibrate import (
        apply_matchup_temperature, fit_matchup_temperature,
    )

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

    all_bets = []

    for date, name, matchups in RACES:
        r = races[
            ((races["date"] == pd.Timestamp(date))
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

        race_ids = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
        val_ids = race_ids[30:]
        val_races, tt_list = [], []
        for vid in val_ids:
            sub = train[train["race_id_short"] == vid]
            raw = model.predict_scores(sub)
            gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
            pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                    / (sub["pl_effective"].std() + 1e-9))
            b = ALPHA * gbm_z + (1 - ALPHA) * pl_z
            b = b / max(b.std(), 1e-6)
            val_races.append(RaceScoresGT(
                scores=b, finishes=sub["finish_pos"].to_numpy(),
                is_dnf=sub["is_dnf"].to_numpy(),
                hazards=np.full(len(sub), HAZARD.get(sub["track_type"].iloc[0], 0.08)),
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
        haz = np.full(len(target), HAZARD.get(tt, 0.08))
        rng = np.random.default_rng(42)
        positions = dist.sample_finishing_orders(blended / T, haz, n_samples=N_SAMPLES, rng=rng)
        matchup_mtx = dist.matchup_probs(positions)

        # Per-track-type calibration (as in main backtest).
        cal_probs_by_type, cal_out_by_type = {}, {}
        for vr, v_tt in zip(val_races, tt_list):
            haz_v = np.full(len(vr.scores), 0.08)
            rng_v = np.random.default_rng(0)
            pos_v = dist.sample_finishing_orders(
                vr.scores / max(T, 0.1), haz_v, n_samples=5000, rng=rng_v
            )
            m_v = dist.matchup_probs(pos_v)
            f = vr.finishes
            n_v = len(f)
            cal_probs_by_type.setdefault(v_tt, [])
            cal_out_by_type.setdefault(v_tt, [])
            for ii in range(n_v):
                for jj in range(ii + 1, n_v):
                    if f[ii] <= 0 or f[jj] <= 0: continue
                    if f[ii] == f[jj]: continue
                    cal_probs_by_type[v_tt].append(float(m_v[ii, jj]))
                    cal_out_by_type[v_tt].append(int(f[ii] < f[jj]))
        all_p = [p for lst in cal_probs_by_type.values() for p in lst]
        all_o = [o for lst in cal_out_by_type.values() for o in lst]
        T_match_global = fit_matchup_temperature(np.asarray(all_p), np.asarray(all_o)) if all_p else 1.0
        if cal_probs_by_type.get(tt) and len(cal_probs_by_type[tt]) >= 200:
            T_match = fit_matchup_temperature(
                np.asarray(cal_probs_by_type[tt]),
                np.asarray(cal_out_by_type[tt]),
            )
        else:
            T_match = T_match_global
        matchup_mtx = apply_matchup_temperature(matchup_mtx, T_match)

        driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
        actual = entries[entries["race_id_short"] == rid][["driver", "finish_pos"]].set_index("driver")

        for a, b, oa, ob in matchups:
            if a not in driver_to_idx or b not in driver_to_idx: continue
            if a not in actual.index or b not in actual.index: continue

            ma_raw = american_to_prob(oa); mb_raw = american_to_prob(ob)
            vig = ma_raw + mb_raw
            ma, mb = ma_raw / vig, mb_raw / vig
            i, j = driver_to_idx[a], driver_to_idx[b]
            p_model_a = float(matchup_mtx[i, j])
            fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
            a_won = fa < fb

            # Decide which side to bet: whichever has positive EV per our model.
            # side_odds = American odds we'd take on that side
            side_a_ev = p_model_a - ma  # positive means we like A more than market
            side_b_ev = (1 - p_model_a) - mb
            if max(side_a_ev, side_b_ev) <= 0:
                continue  # no edge either way, skip

            if side_a_ev > side_b_ev:
                p_model = p_model_a
                odds = oa
                won = a_won
            else:
                p_model = 1 - p_model_a
                odds = ob
                won = not a_won

            b_dec = american_to_decimal(odds) - 1  # decimal odds - 1
            all_bets.append({
                "race": name, "tt": tt,
                "p_model": p_model, "odds": odds, "b_dec": b_dec, "won": int(won),
            })

    df = pd.DataFrame(all_bets)
    print(f"\nTotal bets with positive-EV: {len(df)}\n")

    # ---- Flat unit betting ----
    flat_stake = df["odds"].apply(american_to_stake_flat)
    flat_return = df.apply(
        lambda r: payout_win(r["odds"], american_to_stake_flat(r["odds"])) if r["won"]
        else -american_to_stake_flat(r["odds"]),
        axis=1
    )
    flat_total_staked = flat_stake.sum()
    flat_total_profit = flat_return.sum()
    flat_roi = flat_total_profit / flat_total_staked if flat_total_staked else 0

    # ---- Quarter-Kelly betting ----
    def kelly_frac(p, b):
        # (p*(b+1) - 1) / b = (p*b + p - 1) / b
        f = (p * (b + 1) - 1) / b
        return max(0, f)

    df["kelly_full"] = df.apply(lambda r: kelly_frac(r["p_model"], r["b_dec"]), axis=1)
    df["kelly_quarter"] = df["kelly_full"] * 0.25
    df["kelly_return"] = df.apply(
        lambda r: r["kelly_quarter"] * r["b_dec"] if r["won"]
        else -r["kelly_quarter"],
        axis=1
    )
    kelly_total_staked = df["kelly_quarter"].sum()
    kelly_total_profit = df["kelly_return"].sum()
    kelly_roi = kelly_total_profit / kelly_total_staked if kelly_total_staked else 0

    print(f"{'=' * 60}")
    print(f"FLAT UNIT BETTING (1u to win 1u)")
    print(f"{'=' * 60}")
    print(f"  Bets placed:         {len(df)}")
    print(f"  Wins:                {int(df['won'].sum())} ({df['won'].mean()*100:.1f}%)")
    print(f"  Total units staked:  {flat_total_staked:.2f}")
    print(f"  Total profit:        {flat_total_profit:+.2f} units")
    print(f"  ROI (per unit staked): {flat_roi*100:+.2f}%")
    print(f"  Profit per bet:      {flat_total_profit/len(df):+.4f} units")
    print()
    print(f"{'=' * 60}")
    print(f"QUARTER-KELLY BETTING")
    print(f"{'=' * 60}")
    print(f"  Bets placed:         {len(df)}")
    print(f"  Wins:                {int(df['won'].sum())} ({df['won'].mean()*100:.1f}%)")
    print(f"  Total units staked:  {kelly_total_staked:.2f}")
    print(f"  Total profit:        {kelly_total_profit:+.2f} units")
    print(f"  ROI (per unit staked): {kelly_roi*100:+.2f}%")
    print(f"  Avg quarter-Kelly bet: {df['kelly_quarter'].mean():.4f} units")

    print()
    print(f"{'=' * 60}")
    print(f"EXTRAPOLATION to full 36-race Cup season")
    print(f"{'=' * 60}")
    # Our backtest has ~22 races; full 2026 Cup season is 36 races.
    scale = 36 / 22
    print(f"  Estimated bets:      {int(len(df) * scale)}")
    print(f"  Flat profit (units): {flat_total_profit * scale:+.1f}")
    print(f"  Quarter-Kelly profit (units): {kelly_total_profit * scale:+.1f}")


if __name__ == "__main__":
    main()
