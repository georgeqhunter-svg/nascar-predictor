"""Comprehensive prediction for Enjoy Illinois 300 (Gateway, Sep 13 2026).

Outputs from a single Gumbel-max simulation:
  1. Winner probabilities (P(finish == 1))
  2. Top 5 probabilities (P(finish <= 5))
  3. Top 10 probabilities (P(finish <= 10))
  4. Winning manufacturer probabilities
  5. Probability each driver is the highest-finishing driver of their make
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import TIGHT_REG, ALPHA, N_SAMPLES, HAZARD


TARGET_NAME = "Enjoy Illinois"
TARGET_DATE = "2026-09-13"


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

    # Find target race.
    target_ts = pd.Timestamp(TARGET_DATE)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(TARGET_NAME, case=False, na=False)))
        & (races["season"] == 2026)
    ]
    if r.empty:
        print(f"Race not found in parquet — check TARGET_NAME/TARGET_DATE")
        return
    rid = r.iloc[0]["race_id_short"]
    target_date_ts = r.iloc[0]["date"]
    tt = r.iloc[0]["track_type"]
    print(f"Target: {r.iloc[0]['race_name']}  ({tt}, date={target_date_ts.date()})")

    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == rid].reset_index(drop=True)
    if target.empty:
        print("No entry list for target race yet — need to scrape entry list first.")
        return
    print(f"Field size: {len(target)} drivers")

    print("Training model...")
    model = GBMEnsemble()
    model.fit(train, n_estimators=15, **TIGHT_REG)

    # Calibrate score temperature.
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
    print(f"Track type: {tt}, Score temperature T={T}")

    # Score target race.
    raw = model.predict_scores(target)
    gbm_z = (raw - raw.mean()) / (raw.std() + 1e-9)
    pl_z = ((target["pl_effective"].to_numpy() - target["pl_effective"].mean())
            / (target["pl_effective"].std() + 1e-9))
    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z
    blended = blended / max(blended.std(), 1e-6)
    from backtest_oddslogic_v5 import per_driver_hazards
    haz = per_driver_hazards(target, tt)
    # Preliminary sample for matchup calibration fit.
    rng_prelim = np.random.default_rng(0)
    prelim_positions = dist.sample_finishing_orders(
        blended / T, haz, n_samples=8000, rng=rng_prelim
    )
    matchup_mtx = dist.matchup_probs(prelim_positions)

    # Matchup temperature calibration.
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
    # Isotonic calibration was tried and abandoned — reliability diagram on the
    # backtest showed that any post-hoc calibration inverted bet directions
    # and cratered ROI. Raw sampler outputs are what we bet on. Display probs
    # are slightly overconfident on the tails; treat "80% top-10" as ~65%.
    print(f"Score temperature T={T}, matchup temperature T_m={T_match:.2f}\n")

    # Two sampling passes:
    #   - Matchup sampling at T * T_m (matchup-calibrated).
    #   - Winner/top-N/manufacturer at 1.5x that. Top-N is a MARGINAL over the
    #     joint distribution and needs a wider temperature than pairwise
    #     comparisons: the sharp-tail concentration that calibrates matchups
    #     leaves top-N systematically overconfident. Empirically ~1.5x brings
    #     the favorite's implied win prob into line with market observed.
    T_effective = T * T_match
    # Widen top-N sampling only when T_effective is "sharp" (< 1.0). If T
    # already fit at 1.0+ during calibration, no extra widening — the model
    # is already at natural width and further flattening starves the leaders
    # of top-10 probability. Cap widening at 1.5x, applied only to the amount
    # T_effective is BELOW 1.0.
    T_top_n = max(T_effective * 1.5, 1.0) if T_effective < 1.0 else T_effective
    print(f"T_effective (matchup) = {T_effective:.3f}, T_top_n = {T_top_n:.3f}\n")

    rng = np.random.default_rng(42)
    positions = dist.sample_finishing_orders(
        blended / T_effective, haz, n_samples=N_SAMPLES, rng=rng
    )
    rng2 = np.random.default_rng(43)
    positions_top_n = dist.sample_finishing_orders(
        blended / T_top_n, haz, n_samples=N_SAMPLES, rng=rng2
    )
    matchup_mtx = dist.matchup_probs(positions)

    # -------- Extract predictions from the positions array --------
    drivers = target["driver"].tolist()
    makes = target["manufacturer"].tolist() if "manufacturer" in target.columns else \
            target["make"].tolist() if "make" in target.columns else ["Unknown"] * len(drivers)
    n_drivers = len(drivers)

    # positions is (N_SAMPLES, n_drivers) — 1-indexed (1=winner).
    # Winner/top-N/manuf use the wider-temperature sample.
    win_prob = (positions_top_n == 1).sum(axis=0) / N_SAMPLES
    top5_prob = (positions_top_n <= 5).sum(axis=0) / N_SAMPLES
    top10_prob_raw = (positions_top_n <= 10).sum(axis=0) / N_SAMPLES

    # Top-10 was the only product miscalibrated on backtest (reliability
    # diagram showed S-curve stretching). Apply the pre-fit isotonic
    # calibrator if available.
    import pickle
    from pathlib import Path
    _cal_path = Path("data/processed/top10_calibrator.pkl")
    if _cal_path.exists():
        with open(_cal_path, "rb") as _f:
            _top10_cal = pickle.load(_f)
        top10_prob = _top10_cal.transform(top10_prob_raw)
        print(f"Applied top-10 isotonic calibration")
    else:
        top10_prob = top10_prob_raw
        print(f"[warning] No top-10 calibrator at {_cal_path}; using raw probs")

    # 4. Winning manufacturer: whose make finished in position 1 each sample.
    make_win_counts = defaultdict(int)
    for s in range(N_SAMPLES):
        winner_idx = int(np.argmin(positions_top_n[s]))
        make_win_counts[makes[winner_idx]] += 1
    manuf_win = {m: c / N_SAMPLES for m, c in make_win_counts.items()}

    # 5. Top-of-manufacturer: for each driver, fraction of samples where they
    #    are the best-finishing driver of their make.
    by_make_indices = defaultdict(list)
    for i, m in enumerate(makes):
        by_make_indices[m].append(i)
    top_of_make = np.zeros(n_drivers)
    for m, idxs in by_make_indices.items():
        if len(idxs) == 1:
            top_of_make[idxs[0]] = 1.0
            continue
        sub_pos = positions_top_n[:, idxs]
        best_within = np.argmin(sub_pos, axis=1)
        for k, i in enumerate(idxs):
            top_of_make[i] = float(np.mean(best_within == k))

    # -------- Print reports --------
    df = pd.DataFrame({
        "driver": drivers,
        "make": makes,
        "win_prob": win_prob,
        "top5_prob": top5_prob,
        "top10_prob": top10_prob,
        "top_of_make": top_of_make,
    })

    def american_from_prob(p):
        if p <= 0: return "—"
        if p >= 1: return "-inf"
        dec = 1 / p
        if dec >= 2:
            return f"+{int((dec - 1) * 100)}"
        return f"-{int(100 / (dec - 1))}"

    df["win_odds"] = df["win_prob"].apply(american_from_prob)

    print("=" * 90)
    print("WINNER PROBABILITIES (Top 20)")
    print("=" * 90)
    top_win = df.sort_values("win_prob", ascending=False).head(20)
    print(top_win[["driver", "make", "win_prob", "win_odds"]].to_string(
        index=False, formatters={"win_prob": "{:.3%}".format}))
    print()

    print("=" * 90)
    print("TOP 5 PROBABILITIES (Top 20 drivers)")
    print("=" * 90)
    top5 = df.sort_values("top5_prob", ascending=False).head(20)
    print(top5[["driver", "make", "top5_prob"]].to_string(
        index=False, formatters={"top5_prob": "{:.1%}".format}))
    print()

    print("=" * 90)
    print("TOP 10 PROBABILITIES (Top 20 drivers)")
    print("=" * 90)
    top10 = df.sort_values("top10_prob", ascending=False).head(20)
    print(top10[["driver", "make", "top10_prob"]].to_string(
        index=False, formatters={"top10_prob": "{:.1%}".format}))
    print()

    print("=" * 90)
    print("WINNING MANUFACTURER PROBABILITIES")
    print("=" * 90)
    manuf_df = pd.DataFrame([
        {"make": m, "prob": p} for m, p in sorted(manuf_win.items(), key=lambda x: -x[1])
    ])
    print(manuf_df.to_string(index=False, formatters={"prob": "{:.1%}".format}))
    print()

    print("=" * 90)
    print("TOP-OF-MANUFACTURER PROBABILITIES (per make, sorted)")
    print("=" * 90)
    for make in sorted(df["make"].unique()):
        sub = df[df["make"] == make].sort_values("top_of_make", ascending=False)
        print(f"\n--- {make} ---")
        print(sub[["driver", "top_of_make"]].to_string(
            index=False, formatters={"top_of_make": "{:.1%}".format}))


if __name__ == "__main__":
    main()
