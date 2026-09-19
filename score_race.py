"""score_race.py - LOCKBOX v2 scorer for any race, with a running ledger.

Usage:
    python score_race.py "Illinois" 2026-09-13 matchups_illinois.txt
    python score_race.py "Las Vegas" 2026-10-11 matchups_vegas.txt --season 2026

Matchups file: one matchup per line, comma-separated:
    Driver A, Driver B, oddsA, oddsB
Blank lines and lines starting with # are ignored. Odds are American
("+106" and "-134" both fine). Driver names are accent-insensitive.

Scores the committed model on the posted matchups, graded against actual
finishes. Reports per-matchup and aggregate log-loss / Brier vs the vig-free
market, settles the model's EV picks at flat $1 stakes (plus the >=30%-edge
"official bet" subset), then appends to two running ledgers:

    lockbox_ledger.csv    - one row per graded race
    lockbox_matchups.csv  - one row per graded matchup (powers the running CI)

Re-running the same race_id replaces its ledger rows (idempotent re-grade).
Both ledgers print running pooled stats after every race, including a 95% CI
on the per-matchup log-loss delta vs the market.

LOCKBOX v2 CHANGES vs v1:
    - Uses src.models.predict_pipeline.calibrate_and_sample, which does
      leave-one-race-out CV to fit T (val races are scored by ensembles
      refit WITHOUT them — no leakage).
    - Killed T_match: single calibrated T handles both matchup and top-N.
      T_match in v1 was fit against the same val races T was fit on, so it
      was optimizing against a bias that shouldn't have existed.
    - Per-driver hazards on both val and target races.
    - Matches backtest_oddslogic_v5.py's honest walk-forward setup exactly.
"""
from __future__ import annotations

import argparse
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_oddslogic_v5 import (
    TIGHT_REG, ALPHA, N_SAMPLES, HAZARD, DNF_DISPERSION, DAMAGE_PENALTY,
    american_to_prob, per_driver_hazards, per_driver_damage_hazards,
)
from src.models.predict_pipeline import calibrate_and_sample


LOCKBOX_VERSION = "v2"  # honest leave-one-race-out CV, no T_match


def _norm(s: str) -> str:
    # Accent-fold and lower.
    norm = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower()
    # Canonicalize middle-name variants: "john hunter nemechek" and
    # "john h. nemechek" both -> "john h nemechek". Same rule as
    # backtest_oddslogic_v5.py so cache and lockbox agree.
    parts = norm.split()
    if len(parts) >= 3:
        first, middles, last = parts[0], parts[1:-1], parts[-1]
        middles_short = " ".join(m.rstrip(".")[0] for m in middles if m)
        norm = f"{first} {middles_short} {last}".strip()
    return norm


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def _log_clip(p: float, floor: float = 1e-9) -> float:
    return float(np.log(min(max(p, floor), 1.0 - floor)))


def parse_matchups(path: str) -> list[tuple[str, str, int, int]]:
    out = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip().strip('"').strip("'") for p in line.replace("\t", ",").split(",")]
        parts = [p for p in parts if p]
        if len(parts) < 4:
            raise SystemExit(f"{path}:{ln}: need 'Driver A, Driver B, oddsA, oddsB' - got {line!r}")
        a, b = parts[0], parts[1]
        try:
            oa, ob = int(parts[2]), int(parts[3])
        except ValueError:
            raise SystemExit(f"{path}:{ln}: odds must be integers - got {line!r}")
        out.append((a, b, oa, ob))
    if not out:
        raise SystemExit(f"{path}: no matchups parsed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Lockbox v2: grade the committed (honest-calibration) model vs the market.")
    ap.add_argument("name", help="race name fragment, e.g. 'Illinois'")
    ap.add_argument("date", help="race date YYYY-MM-DD")
    ap.add_argument("matchups", help="path to matchups file")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--ledger", default="lockbox_ledger.csv")
    ap.add_argument("--detail", default="lockbox_matchups.csv")
    ap.add_argument("--edge", type=float, default=0.30, help="official-bet edge threshold")
    args = ap.parse_args()

    matchups = parse_matchups(args.matchups)
    print(f"Loaded {len(matchups)} matchups from {args.matchups}")

    from src.features.build_features import build_features

    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")
    races["date"] = pd.to_datetime(races["date"])
    entries["date"] = pd.to_datetime(entries["date"])

    print("Building features (slow part)...")
    features = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    features["date"] = pd.to_datetime(features["date"])

    target_ts = pd.Timestamp(args.date)
    r = races[
        ((races["date"] == target_ts)
         | (races["race_name"].str.contains(args.name, case=False, na=False)))
        & (races["season"] == args.season)
    ]
    if len(r) == 0:
        raise SystemExit(f"No {args.season} race found matching date={args.date} name~{args.name!r}")
    if len(r) > 1:
        r = r[races["date"] == target_ts] if (races["date"] == target_ts).any() else r
    rid = r.iloc[0]["race_id_short"]
    race_name = r.iloc[0].get("race_name", args.name)
    target_date_ts = r.iloc[0]["date"]
    from src.features.tracks import resolve_track_type
    tt = resolve_track_type(r.iloc[0].get("track_name", ""),
                            fallback=r.iloc[0]["track_type"])
    print(f"Target: {race_name} ({rid}, {tt})")

    train = features[(features["date"] < target_date_ts) & (features["finish_pos"] > 0)]
    target = features[features["race_id_short"] == rid].reset_index(drop=True)

    print("Calibrating + sampling (leave-one-race-out CV - this is the slow part)...")
    sr = calibrate_and_sample(
        train, target, tt,
        tight_reg=TIGHT_REG, alpha=ALPHA, n_samples=N_SAMPLES,
        hazard_lookup=HAZARD,
        per_driver_hazards_fn=per_driver_hazards,
        dnf_dispersion_by_type=DNF_DISPERSION,
        per_driver_damage_hazards_fn=per_driver_damage_hazards,
        damage_penalty=DAMAGE_PENALTY,
    )
    matchup_mtx = sr.matchup_mtx

    name_to_idx = {_norm(d): i for i, d in enumerate(target["driver"].values)}
    actual = entries[entries["race_id_short"] == rid][["driver", "finish_pos"]].copy()
    actual["nkey"] = actual["driver"].map(_norm)
    finish = dict(zip(actual["nkey"], actual["finish_pos"]))

    rows, unresolved, dnf_skipped = [], [], []
    for a, b, oa, ob in matchups:
        ka, kb = _norm(a), _norm(b)
        if ka not in name_to_idx or kb not in name_to_idx:
            unresolved.append((a, b))
            continue
        i, j = name_to_idx[ka], name_to_idx[kb]
        p_a = float(matchup_mtx[i, j])
        ma_raw, mb_raw = american_to_prob(oa), american_to_prob(ob)
        vig = ma_raw + mb_raw
        ma = ma_raw / vig
        fa, fb = finish.get(ka), finish.get(kb)
        if fa is None or fb is None:
            unresolved.append((a, b))
            continue
        if fa <= 0 or fb <= 0:
            dnf_skipped.append((a, b))
            continue
        a_won = fa < fb
        p_model_won = p_a if a_won else 1 - p_a
        p_market_won = ma if a_won else 1 - ma
        # EV pick: side with higher expected value per $1
        ev_a = p_a * (american_to_decimal(oa) - 1) - (1 - p_a)
        ev_b = (1 - p_a) * (american_to_decimal(ob) - 1) - p_a
        if ev_a >= ev_b:
            pick, pick_odds, pick_won, pick_ev, edge = a, oa, a_won, ev_a, p_a - ma
        else:
            pick, pick_odds, pick_won, pick_ev, edge = b, ob, not a_won, ev_b, (1 - p_a) - (1 - ma)
        profit = (american_to_decimal(pick_odds) - 1) if pick_won else -1.0
        rows.append({
            "a": a, "b": b, "odds_a": oa, "odds_b": ob,
            "model_p_a": p_a, "mkt_p_a": ma,
            "winner": a if a_won else b,
            "model_ll": -_log_clip(p_model_won), "mkt_ll": -_log_clip(p_market_won),
            "delta": -_log_clip(p_model_won) + _log_clip(p_market_won),
            "model_brier": (p_a - float(a_won)) ** 2, "mkt_brier": (ma - float(a_won)) ** 2,
            "model_right": int((p_a > 0.5) == a_won), "mkt_right": int((ma > 0.5) == a_won),
            "pick": pick, "pick_odds": pick_odds, "pick_ev": pick_ev, "edge": edge,
            "pick_won": int(pick_won), "profit": profit,
        })

    df = pd.DataFrame(rows)
    print()
    print("=" * 100)
    print(f"LOCKBOX {LOCKBOX_VERSION}: {race_name} ({rid}, {tt})  T={sr.T:.3f}  "
          f"n={len(df)} matchups")
    print("=" * 100)
    if len(df) == 0:
        print("No matchups could be graded - ledger NOT updated.")
    else:
        show = df[["a", "b", "model_p_a", "mkt_p_a", "winner", "model_ll", "mkt_ll"]].copy()
        print(show.to_string(index=False, formatters={
            "model_p_a": "{:.3f}".format, "mkt_p_a": "{:.3f}".format,
            "model_ll": "{:.3f}".format, "mkt_ll": "{:.3f}".format}))
        print()
        print(f"log-loss : model {df['model_ll'].mean():.4f}  market {df['mkt_ll'].mean():.4f}  "
              f"Delta {df['model_ll'].mean() - df['mkt_ll'].mean():+.4f}")
        print(f"Brier    : model {df['model_brier'].mean():.4f}  market {df['mkt_brier'].mean():.4f}")
        print(f"correct  : model {df['model_right'].sum()}/{len(df)}  market {df['mkt_right'].sum()}/{len(df)}")
        print()
        print("EV picks (higher-EV side of every matchup, flat $1):")
        print(f"  {df['pick_won'].sum()}/{len(df)} won, {df['profit'].sum():+.2f} units")
        bets = df[df["edge"] >= args.edge]
        if len(bets):
            print(f"Official bets (edge >= {args.edge:.0%}):")
            print(bets[["pick", "pick_ev", "edge", "pick_won", "profit"]].to_string(
                index=False, formatters={"pick_ev": "{:+.3f}".format, "edge": "{:+.1%}".format}))
            print(f"  {bets['pick_won'].sum()}/{len(bets)} won, {bets['profit'].sum():+.2f} units")
        else:
            print(f"No matchup cleared the {args.edge:.0%} edge threshold.")
    if dnf_skipped:
        print("Not graded (unclassified/DNF finish):")
        for a, b in dnf_skipped:
            print(f"  {a} vs {b}")
    if unresolved:
        print("Unresolved (name/finish not matched):")
        for a, b in unresolved:
            print(f"  {a} vs {b}")

    if len(df) == 0:
        return

    # --- ledgers -----------------------------------------------------------
    race_row = {
        "graded_at": datetime.now().isoformat(timespec="seconds"),
        "lockbox_version": LOCKBOX_VERSION,
        "race_id": str(rid), "race_name": race_name,
        "date": str(pd.Timestamp(target_date_ts).date()), "season": args.season,
        "track_type": tt, "T": sr.T,
        "n": len(df),
        "model_ll": df["model_ll"].mean(), "mkt_ll": df["mkt_ll"].mean(),
        "delta": df["model_ll"].mean() - df["mkt_ll"].mean(),
        "brier_model": df["model_brier"].mean(), "brier_mkt": df["mkt_brier"].mean(),
        "model_correct": int(df["model_right"].sum()), "mkt_correct": int(df["mkt_right"].sum()),
        "ev_won": int(df["pick_won"].sum()), "ev_n": len(df),
        "ev_profit": df["profit"].sum(),
        "bets_n": int((df["edge"] >= args.edge).sum()),
        "bets_won": int(df.loc[df["edge"] >= args.edge, "pick_won"].sum()),
        "bets_profit": df.loc[df["edge"] >= args.edge, "profit"].sum(),
        "matchups_file": args.matchups,
    }
    ledger_path = Path(args.ledger)
    if ledger_path.exists():
        led = pd.read_csv(ledger_path, dtype={"race_id": str})
        led = led[led["race_id"] != str(rid)]
    else:
        led = pd.DataFrame()
    led = pd.concat([led, pd.DataFrame([race_row])], ignore_index=True)
    led = led.sort_values(["date", "race_id"]).reset_index(drop=True)
    led.to_csv(ledger_path, index=False)

    det = df.copy()
    det.insert(0, "date", str(pd.Timestamp(target_date_ts).date()))
    det.insert(0, "race_id", str(rid))
    det.insert(0, "lockbox_version", LOCKBOX_VERSION)
    detail_path = Path(args.detail)
    if detail_path.exists():
        old = pd.read_csv(detail_path, dtype={"race_id": str})
        old = old[old["race_id"] != str(rid)]
    else:
        old = pd.DataFrame()
    det = pd.concat([old, det], ignore_index=True)
    det = det.sort_values(["date", "race_id"]).reset_index(drop=True)
    det.to_csv(detail_path, index=False)

    # --- running totals ----------------------------------------------------
    print()
    print("=" * 100)
    print(f"RUNNING LOCKBOX TOTALS  ({args.ledger}, {args.detail})")
    print("=" * 100)
    n_races = len(led)
    tot_matchups = int(led["n"].sum())
    w = led["n"] / led["n"].sum()
    pooled_model = float((led["model_ll"] * w).sum())
    pooled_mkt = float((led["mkt_ll"] * w).sum())
    print(f"races graded      : {n_races}  ({tot_matchups} matchups)")
    print(f"pooled log-loss   : model {pooled_model:.4f}  market {pooled_mkt:.4f}  "
          f"Delta {pooled_model - pooled_mkt:+.4f} per matchup")
    d = det["delta"].to_numpy()
    if len(d) >= 2:
        sd = float(d.std(ddof=1))
        se = sd / np.sqrt(len(d))
        lo, hi = float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)
        print(f"per-matchup delta : {d.mean():+.4f}  (95% CI [{lo:+.4f}, {hi:+.4f}], n={len(d)})")
        if lo > 0:
            print("  -> market is SIGNIFICANTLY better (CI entirely above 0)")
        elif hi < 0:
            print("  -> model is SIGNIFICANTLY better (CI entirely below 0)")
        else:
            print("  -> CI spans 0: no significant difference yet - keep collecting races")
    print(f"EV picks overall  : {int(led['ev_won'].sum())}/{int(led['ev_n'].sum())} won, "
          f"{led['ev_profit'].sum():+.2f} units")
    if led["bets_n"].sum() > 0:
        print(f"official bets     : {int(led['bets_won'].sum())}/{int(led['bets_n'].sum())} won, "
              f"{led['bets_profit'].sum():+.2f} units")
    print()
    print(led[["date", "race_name", "track_type", "n", "model_ll", "mkt_ll", "delta",
               "ev_profit"]].to_string(index=False, formatters={
        "model_ll": "{:.4f}".format, "mkt_ll": "{:.4f}".format,
        "delta": "{:+.4f}".format, "ev_profit": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
