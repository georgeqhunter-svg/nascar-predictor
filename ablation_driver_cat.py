"""Ablation: does dropping the driver categorical feature fix the overconfidence?

Three configs, each retrained walk-forward on all races through Sep 6 2026:
  A. Baseline           — current feature set.
  B. Drop driver cat    — force GBM to rely on PL + rolling + qual.
  C. Drop driver + team — even more aggressive; only track_type and manufacturer remain categorical.

For each config, compute:
  1. Internal matchup log-loss on 2026 races (all-pairs, walk-forward).
  2. Matchup log-loss on the 12 OddsLogic races (using the market-listed pairs).
  3. Disagreement hit rate on OddsLogic pairs (model right when disagreeing with market).

If B or C substantially improves (2) and (3), the driver categorical was the bug.

Runtime: ~30-45 min (three separate models, each doing the equivalent of
backtest_oddslogic_v5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    RACES, TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, american_to_prob,
)
from src.features.build_features import build_features
from src.models import distribution as dist
from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type
from src.models.gbm_ranker import GBMEnsemble


CONFIGS = {
    "A_baseline": {"drop_features": []},
    "B_no_driver": {"drop_features": ["driver"]},
    "C_no_driver_team": {"drop_features": ["driver", "team"]},
}


def build_model_for_config(train_df: pd.DataFrame, drop: list[str]) -> GBMEnsemble:
    m = GBMEnsemble()
    m.drop_features = drop or None
    m.fit(train_df, n_estimators=10, **TIGHT_REG)
    return m


def eval_config(name: str, drop: list[str], features: pd.DataFrame,
                races: pd.DataFrame, entries: pd.DataFrame) -> dict:
    print(f"\n{'='*70}\nConfig {name}  drop={drop}\n{'='*70}")

    total_pairs_2026 = 0
    total_ll_2026 = 0.0
    ol_market_lls = []
    ol_model_lls = []
    n_agree_correct = 0; n_agree = 0
    n_disagree_correct = 0; n_disagree = 0
    ol_model_hit = 0; ol_market_hit = 0; ol_n = 0
    per_race_ol = []

    ol_lookup = {r[1]: r for r in RACES}

    # For efficiency: retrain once per OddsLogic race + do all-2026 walk-forward via one pass.
    for date, name_match, matchups in RACES:
        r = races[
            ((races["date"] == pd.Timestamp(date))
             | (races["race_name"].str.contains(name_match, case=False, na=False)))
            & (races["season"] == 2026)
        ]
        if r.empty:
            continue
        rid = r.iloc[0]["race_id_short"]
        target_date = r.iloc[0]["date"]
        tt = r.iloc[0]["track_type"]

        train = features[(features["date"] < target_date) & (features["finish_pos"] > 0)]
        target = features[features["race_id_short"] == rid].reset_index(drop=True)
        if target.empty:
            continue

        model = build_model_for_config(train, drop)

        # Calibrate.
        race_ids = train.groupby("race_id_short")["date"].first().sort_values().index.tolist()
        val_ids = race_ids[30:]
        vr, tt_list = [], []
        for vid in val_ids:
            sub = train[train["race_id_short"] == vid]
            if len(sub) < 5: continue
            raw = model.predict_scores(sub)
            gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
            pl_z = ((sub["pl_effective"].to_numpy() - sub["pl_effective"].mean())
                    / (sub["pl_effective"].std() + 1e-9))
            b = ALPHA * gbm_z + (1 - ALPHA) * pl_z
            b = b / max(b.std(), 1e-6)
            vr.append(RaceScoresGT(
                scores=b, finishes=sub["finish_pos"].to_numpy(),
                is_dnf=sub["is_dnf"].to_numpy(),
                hazards=np.full(len(sub), HAZARD.get(sub["track_type"].iloc[0], 0.08)),
            ))
            tt_list.append(sub["track_type"].iloc[0])
        T_by_type = find_best_temperature_by_type(vr, tt_list, default_T=1.0, n_samples=1000)
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

        # OddsLogic-listed matchups.
        driver_to_idx = {d: i for i, d in enumerate(target["driver"].values)}
        actual = entries[entries["race_id_short"] == rid][
            ["driver", "finish_pos"]
        ].set_index("driver")

        race_market_lls = []
        race_model_lls = []
        race_mkt_c = race_mdl_c = race_n = 0
        for a, b, oa, ob in matchups:
            if a not in driver_to_idx or b not in driver_to_idx: continue
            if a not in actual.index or b not in actual.index: continue
            ma = american_to_prob(oa); mb = american_to_prob(ob); v = ma + mb
            ma, mb = ma / v, mb / v
            i, j = driver_to_idx[a], driver_to_idx[b]
            p = float(matchup_mtx[i, j])
            fa, fb = int(actual.loc[a, "finish_pos"]), int(actual.loc[b, "finish_pos"])
            a_won = fa < fb
            p_mkt = ma if a_won else mb
            p_mdl = p if a_won else 1 - p
            race_market_lls.append(-float(np.log(np.clip(p_mkt, 1e-9, 1 - 1e-9))))
            race_model_lls.append(-float(np.log(np.clip(p_mdl, 1e-9, 1 - 1e-9))))
            race_mkt_c += int((ma > mb and a_won) or (mb > ma and not a_won))
            race_mdl_c += int((p > 0.5 and a_won) or (p < 0.5 and not a_won))
            race_n += 1

            model_pick_a = p > 0.5; market_pick_a = ma > mb
            if model_pick_a == market_pick_a:
                n_agree += 1
                if (model_pick_a and a_won) or (not model_pick_a and not a_won):
                    n_agree_correct += 1
            else:
                n_disagree += 1
                if (model_pick_a and a_won) or (not model_pick_a and not a_won):
                    n_disagree_correct += 1

        if race_n > 0:
            r_mkt = float(np.mean(race_market_lls))
            r_mdl = float(np.mean(race_model_lls))
            print(f"  {name_match} ({tt}): n={race_n} mkt={r_mkt:.4f} mdl={r_mdl:.4f} Delta={r_mdl-r_mkt:+.4f}")
            per_race_ol.append({"race": name_match, "tt": tt, "n": race_n,
                                "mkt_ll": r_mkt, "mdl_ll": r_mdl})
            ol_market_lls.extend(race_market_lls)
            ol_model_lls.extend(race_model_lls)
            ol_market_hit += race_mkt_c
            ol_model_hit += race_mdl_c
            ol_n += race_n

        # All-pairs internal LL for this 2026 race.
        finishes = target["finish_pos"].to_numpy()
        pair_lls = []
        for i in range(len(finishes)):
            for j in range(i + 1, len(finishes)):
                if finishes[i] == 0 or finishes[j] == 0: continue
                if finishes[i] == finishes[j]: continue
                i_won = finishes[i] < finishes[j]
                p = matchup_mtx[i, j] if i_won else 1 - matchup_mtx[i, j]
                pair_lls.append(-float(np.log(np.clip(p, 1e-9, 1 - 1e-9))))
        if pair_lls:
            total_ll_2026 += sum(pair_lls)
            total_pairs_2026 += len(pair_lls)

    ol_mkt = float(np.mean(ol_market_lls)) if ol_market_lls else float("nan")
    ol_mdl = float(np.mean(ol_model_lls)) if ol_model_lls else float("nan")
    int_ll = total_ll_2026 / total_pairs_2026 if total_pairs_2026 else float("nan")

    print(f"\n[{name}] Summary")
    print(f"  Internal 2026 all-pairs LL: {int_ll:.4f}  ({total_pairs_2026} pairs)")
    print(f"  OddsLogic market  LL: {ol_mkt:.4f}  hit {ol_market_hit}/{ol_n} "
          f"({ol_market_hit/max(ol_n,1)*100:.1f}%)")
    print(f"  OddsLogic model   LL: {ol_mdl:.4f}  hit {ol_model_hit}/{ol_n} "
          f"({ol_model_hit/max(ol_n,1)*100:.1f}%)")
    print(f"  Delta (mdl - mkt) OddsLogic LL: {ol_mdl - ol_mkt:+.4f}")
    print(f"  Agreement:    {n_agree_correct}/{n_agree} ({n_agree_correct/max(n_agree,1)*100:.1f}%)")
    print(f"  Disagreement: {n_disagree_correct}/{n_disagree} ({n_disagree_correct/max(n_disagree,1)*100:.1f}%)")

    return {
        "name": name, "internal_ll": int_ll, "ol_mkt_ll": ol_mkt, "ol_mdl_ll": ol_mdl,
        "ol_delta": ol_mdl - ol_mkt, "agree_hit": n_agree_correct / max(n_agree, 1),
        "disagree_hit": n_disagree_correct / max(n_disagree, 1),
        "ol_n": ol_n, "per_race": per_race_ol,
    }


def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    results = []
    for cname, cfg in CONFIGS.items():
        r = eval_config(cname, cfg["drop_features"], features, races, entries)
        results.append(r)

    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    df = pd.DataFrame([
        {"config": r["name"], "int_2026_ll": r["internal_ll"],
         "ol_mkt": r["ol_mkt_ll"], "ol_mdl": r["ol_mdl_ll"],
         "delta": r["ol_delta"], "agree_hit%": r["agree_hit"] * 100,
         "disagree_hit%": r["disagree_hit"] * 100}
        for r in results
    ])
    print(df.round(4).to_string(index=False))

    print("\nInterpretation:")
    print("  - If disagree_hit% climbs toward 50%, the driver categorical was the bug.")
    print("  - If ol_mdl and delta improve, we've found real edge to keep.")
    print("  - If ol_mdl gets worse but int_2026_ll stays flat, the categorical was\n"
          "    load-bearing on internal metrics; we may have overfit our diagnostic.")


if __name__ == "__main__":
    main()
