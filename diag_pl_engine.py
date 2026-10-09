"""Plackett-Luce engine check: how well do ratings predict the NEXT race?
Replays 2022+ history (DNFs excluded, as adopted) and scores each race BEFORE
updating on it (walk-forward). Compares the current update (L2 shrink applied
at every position step) with a per-race update (gradients summed over the
race, shrink applied once per race)."""
import math, numpy as np, pandas as pd
from collections import defaultdict
from src.models.plackett_luce import Ratings, RaceRanking, update_from_race, race_log_likelihood
from src.features.tracks import resolve_track_type

E = pd.read_parquet("data/processed/entries.parquet")
R = pd.read_parquet("data/processed/races.parquet"); R["date"] = pd.to_datetime(R.date); R = R.sort_values("date")
races = []
for _, rr in R.iterrows():
    g = E[(E.race_id_short == rr.race_id_short) & (E.finish_pos > 0) & (~E.is_dnf.astype(bool))].sort_values("finish_pos")
    if len(g) >= 10:
        races.append((rr.race_id_short, rr.date, rr.track_name,
                      RaceRanking(drivers=g.driver.tolist(), teams=g.team.tolist(),
                                  track_type=resolve_track_type(rr.track_name), dnf_at=len(g))))

def update_per_race(rt, race, l2d, l2t, l2dt):
    """Same gradient as update_from_race, but summed over the race and shrunk once."""
    n = len(race.drivers)
    s = rt.bulk_strengths(race.drivers, race.teams, race.track_type)
    gd = np.zeros(n); remaining = np.ones(n, bool)
    for k in range(n - 1):
        idx = np.where(remaining)[0]; p = np.exp(s[idx] - s[idx].max()); p /= p.sum()
        t = np.zeros_like(p); t[np.where(idx == k)[0][0]] = 1
        gd[idx] += t - p; remaining[k] = False
    tg = defaultdict(float)
    for i, d in enumerate(race.drivers):
        tg[race.teams[i]] += gd[i]
    for i, d in enumerate(race.drivers):
        rt.driver[d] = rt.driver.get(d, 0) * (1 - l2d) + rt.lr_driver * gd[i]
        dt = rt.driver_track.setdefault(d, {})
        dt[race.track_type] = dt.get(race.track_type, 0) * (1 - l2dt) + rt.lr_driver_track * gd[i]
    for tm, g in tg.items():
        rt.team[tm] = rt.team.get(tm, 0) * (1 - l2t) + rt.lr_team * g

def run(mode, **kw):
    rt = Ratings(); out = []
    for rid, dt, tn, race in races:
        if dt.year >= 2023:
            ll = race_log_likelihood(rt, race)
            out.append((rid, tn, race.track_type, ll / (len(race.drivers) - 1)))
        if mode == "current": update_from_race(rt, race)
        else: update_per_race(rt, race, **kw)
    return pd.DataFrame(out, columns=["rid", "track", "tt", "ll"]), rt

base = None
for name, mode, kw in [("current (shrink every step)", "current", {}),
                       ("per-race, same per-race l2 (0.003/0.003/0.02)", "per", dict(l2d=.003, l2t=.003, l2dt=.02)),
                       ("per-race, l2 0.01/0.01/0.03", "per", dict(l2d=.01, l2t=.01, l2dt=.03))]:
    df, rt = run(mode, **kw)
    if base is None: base = df
    diff = (df.ll - base.ll).values
    print(f"{name:48s} mean next-race LL/position {df.ll.mean():.4f}  vs current {diff.mean()*1e3:+.2f} (per 1e-3)  better in {(diff>0).mean():.0%} of races")
    print("    by type:", df.groupby("tt").ll.mean().round(4).to_dict())
    print("    drv@type spread (std of all driver-track adj):",
          round(np.std([v for d in rt.driver_track.values() for v in d.values()]), 3),
          "| driver spread:", round(np.std(list(rt.driver.values())), 3))

print("\n-- variants of the CURRENT update --")
def run_cur(**hp):
    rt = Ratings(**hp); out = []
    for rid, dt, tn, race in races:
        if dt.year >= 2023:
            out.append((tn, race.track_type, race_log_likelihood(rt, race) / (len(race.drivers) - 1)))
        update_from_race(rt, race)
    return pd.DataFrame(out, columns=["track", "tt", "ll"])
for name, hp in [("no driver-track term", dict(lr_driver_track=0.0)),
                 ("no team term", dict(lr_team=0.0)),
                 ("driver only", dict(lr_driver_track=0.0, lr_team=0.0)),
                 ("lr_driver 0.10", dict(lr_driver=0.10)),
                 ("lr_driver 0.25", dict(lr_driver=0.25))]:
    df = run_cur(**hp); diff = (df.ll - base.ll).values
    print(f"{name:24s} {df.ll.mean():.4f}  vs current {diff.mean()*1e3:+.2f}  better in {(diff>0).mean():.0%}  "
          f"| ss {df[df.tt=='superspeedway'].ll.mean()-base[base.tt=='superspeedway'].ll.mean():+.4f}  "
          f"Atlanta {df[df.track.str.contains('Atlanta')].ll.mean()-base[base.track.str.contains('Atlanta')].ll.mean():+.4f}")

print("\n-- PRE-2026 ONLY (selection set; 2026 = backtest, untouched) --")
def run_sel(**hp):
    rt = Ratings(**hp); out = []
    for rid, dt, tn, race in races:
        if 2023 <= dt.year <= 2025:
            out.append((race.track_type, race_log_likelihood(rt, race) / (len(race.drivers) - 1)))
        update_from_race(rt, race)
    return pd.DataFrame(out, columns=["tt", "ll"])
b0 = run_sel()
for name, hp in [("no team term", dict(lr_team=0.0)), ("driver only", dict(lr_driver_track=0.0, lr_team=0.0)),
                 ("lr_driver 0.10", dict(lr_driver=0.10)), ("no team + lr 0.10", dict(lr_team=0.0, lr_driver=0.10)),
                 ("driver only + lr 0.10", dict(lr_team=0.0, lr_driver_track=0.0, lr_driver=0.10)),
                 ("no team + lr 0.07", dict(lr_team=0.0, lr_driver=0.07))]:
    df = run_sel(**hp); dd = (df.ll - b0.ll).values
    print(f"{name:26s} vs current {dd.mean()*1e3:+.2f}  better in {(dd>0).mean():.0%} (n={len(dd)})")
