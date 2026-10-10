"""Why does the model like (or dislike) a driver?  Feature-by-feature SHAP audit.

Trains the GBM ensemble ONCE on all races before 2026 (same settings as the
backtest, DNF-excluded target), then scores every 2026 race and splits each
driver's score into per-feature contributions (LightGBM pred_contrib = exact
TreeSHAP). Scores are centred within each race, because only relative scores
matter for ranking: a contribution of +0.10 means "this feature pushes him
0.10 above the field average in that race".

Output:
  * per driver: avg centred score, avg centred contribution of every feature
    -> data/processed/driver_shap_2026.csv
  * printed: the drivers in DRIVERS, top features lifting / dropping them,
    plus how their actual 2026 running results compare.

Usage:  python diag_driver_shap.py                 (default drivers)
        python diag_driver_shap.py "Joey Logano" "Ryan Blaney"
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import TIGHT_REG
from src.features.build_features import build_features
from src.models.gbm_ranker import GBMEnsemble, _prepare_x, GBM_EXCLUDE_DNF

DRIVERS = sys.argv[1:] or ["Joey Logano", "Ryan Blaney", "Austin Cindric", "Kyle Larson",
                            "Denny Hamlin", "William Byron", "Christopher Bell"]
N_EST = 5
CUTOFF = pd.Timestamp("2026-01-01")


def main() -> None:
    P = "data/processed/"
    races = pd.read_parquet(P + "races.parquet")
    entries = pd.read_parquet(P + "entries.parquet")
    sessions = pd.read_parquet(P + "sessions.parquet")
    loop = pd.read_parquet(P + "loopstats.parquet")
    lap = pd.read_parquet(P + "laptimes.parquet")
    print("Building features (slow part)...")
    f = build_features(races, entries, sessions, loopstats=loop, laptimes=lap)
    f["date"] = pd.to_datetime(f["date"])

    train = f[(f.date < CUTOFF) & (f.finish_pos > 0)]
    test = f[(f.date >= CUTOFF) & (f.finish_pos > 0)].reset_index(drop=True)
    print(f"Training on {train.race_id_short.nunique()} races (< {CUTOFF.date()}), "
          f"explaining {test.race_id_short.nunique()} 2026 races. GBM_EXCLUDE_DNF={GBM_EXCLUDE_DNF}")
    m = GBMEnsemble()
    m.fit(train, n_estimators=N_EST, **TIGHT_REG)

    x = _prepare_x(test, drop=m.drop_features)
    contrib = np.mean([b.predict(x, pred_contrib=True) for b in m.boosters], axis=0)
    cols = list(x.columns) + ["_bias"]
    C = pd.DataFrame(contrib, columns=cols).drop(columns="_bias")
    C["score"] = C.sum(axis=1)
    C["race_id_short"] = test["race_id_short"].values
    C["driver"] = test["driver"].values
    # centre within race
    num = [c for c in C.columns if c not in ("race_id_short", "driver")]
    C[num] = C[num] - C.groupby("race_id_short")[num].transform("mean")

    # actual 2026 running results (finish pct among finishers) for context
    t = test.copy()
    n = t.groupby("race_id_short").finish_pos.transform("count")
    t["fin_pct"] = (t.finish_pos - 1) / (n - 1)
    t["run_fin_pct"] = t.fin_pct.where(~t.is_dnf.astype(bool))
    actual = t.groupby("driver").agg(races=("fin_pct", "size"), run_fin_pct=("run_fin_pct", "mean"),
                                     start_pct=("start_pos", "mean"))

    per = C.groupby("driver")[num].mean()
    per = per.join(actual, how="left")
    per.to_csv(P + "driver_shap_2026.csv")
    print(f"Wrote {P}driver_shap_2026.csv ({len(per)} drivers)\n")

    # field-wide: how well does each driver's model score track his actual running results?
    reg = per.dropna(subset=["run_fin_pct"])
    reg = reg[reg.races >= 10]
    b = np.polyfit(reg.run_fin_pct, reg.score, 1)
    reg["score_vs_results"] = reg.score - np.polyval(b, reg.run_fin_pct)  # + = model likes him more than his 2026 results justify
    print("Drivers the model likes MORE than their 2026 running results justify (score residual):")
    print(reg.sort_values("score_vs_results", ascending=False)[["races", "run_fin_pct", "score", "score_vs_results"]]
          .head(8).round(3).to_string())
    print("\n...and LESS:")
    print(reg.sort_values("score_vs_results")[["races", "run_fin_pct", "score", "score_vs_results"]]
          .head(5).round(3).to_string())

    feats = [c for c in num if c != "score"]
    for drv in DRIVERS:
        if drv not in per.index:
            print(f"\n{drv}: not found"); continue
        row = per.loc[drv]
        s = row[feats].sort_values()
        print(f"\n=== {drv}: avg centred score {row['score']:+.3f} | 2026 running finish pct "
              f"{row['run_fin_pct']:.3f} | residual vs results "
              f"{reg['score_vs_results'].get(drv, np.nan):+.3f}")
        print("  lifting him:  " + ", ".join(f"{k} {v:+.3f}" for k, v in s[::-1].head(8).items()))
        print("  dropping him: " + ", ".join(f"{k} {v:+.3f}" for k, v in s.head(5).items()))

    # which features drive the residual across ALL drivers? (contribution vs residual)
    print("\nFeatures whose contribution best explains 'model likes him more than results' across drivers:")
    corr = {c: np.corrcoef(reg[c], reg["score_vs_results"])[0, 1] for c in feats if reg[c].std() > 1e-9}
    top = sorted(corr.items(), key=lambda kv: -kv[1])[:10]
    for k, v in top:
        print(f"  {k:36s} corr {v:+.2f}   (Logano {per.loc['Joey Logano', k]:+.3f})"
              if "Joey Logano" in per.index else f"  {k:36s} corr {v:+.2f}")


if __name__ == "__main__":
    main()
