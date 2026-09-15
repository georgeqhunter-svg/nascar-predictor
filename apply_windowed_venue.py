"""apply_windowed_venue.py - window the career-long venue accumulators in rolling.py.

avg_finish_at_type_ytd / avg_finish_at_track / team_avg_finish_at_type are currently
CAREER averages (accumulators never reset). This caps them at recent windows:
  driver-at-type  last 8, driver-at-track last 5, team-at-type last 30.
Career COUNTS are kept as experience features. Feature names unchanged.
"""
from pathlib import Path
import shutil
import sys

R = [
("""    type_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "sum_finish": 0})""",
"""    type_hist: dict[tuple, deque] = defaultdict(deque)    # last-8 finishes at this track type
    type_count: dict[tuple, int] = defaultdict(int)       # career races at this track type"""),

("""    # NEW: per-driver-per-track history (avg finish and count at THIS specific track).
    track_stats: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "sum_finish": 0, "best": 999}
    )""",
"""    # Per-driver-per-track history: career count/best + last-5 finishes window.
    track_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "best": 999})
    track_hist: dict[tuple, deque] = defaultdict(deque)   # last-5 finishes at this track"""),

("""    # NEW: per-TEAM-per-track-type rolling avg finish (all their drivers).
    team_type_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "sum_finish": 0})""",
"""    # Per-TEAM-per-track-type: career count + last-30 team finishes window.
    team_type_hist: dict[tuple, deque] = defaultdict(deque)
    team_type_count: dict[tuple, int] = defaultdict(int)"""),

("""            ss = season_stats[(driver, season)]
            ts = type_stats[(driver, track_type)]
            trk = track_stats[(driver, track_id)]
            team = team_lookup.get(driver, "")
            tt_team = team_type_stats[(team, track_type)]""",
"""            ss = season_stats[(driver, season)]
            th = type_hist[(driver, track_type)]
            trk = track_stats[(driver, track_id)]
            tkh = track_hist[(driver, track_id)]
            team = team_lookup.get(driver, "")
            tth = team_type_hist[(team, track_type)]"""),

("""                "races_at_type_ytd": ts["n"],
                "avg_finish_at_type_ytd": (ts["sum_finish"] / ts["n"]) if ts["n"] else np.nan,
                "races_at_track": trk["n"],
                "avg_finish_at_track": (trk["sum_finish"] / trk["n"]) if trk["n"] else np.nan,
                "best_finish_at_track": trk["best"] if trk["n"] else np.nan,
                "team_avg_finish_at_type": (tt_team["sum_finish"] / tt_team["n"]) if tt_team["n"] else np.nan,
                "team_races_at_type": tt_team["n"],""",
"""                "races_at_type_ytd": type_count[(driver, track_type)],
                "avg_finish_at_type_ytd": _avg_last(th, 8),
                "races_at_track": trk["n"],
                "avg_finish_at_track": _avg_last(tkh, 5),
                "best_finish_at_track": trk["best"] if trk["n"] else np.nan,
                "team_avg_finish_at_type": _avg_last(tth, 30),
                "team_races_at_type": team_type_count[(team, track_type)],"""),

("""            ts = type_stats[(driver, track_type)]
            ts["n"] += 1
            ts["sum_finish"] += finish""",
"""            thu = type_hist[(driver, track_type)]
            thu.append(finish)
            if len(thu) > 8:
                thu.popleft()
            type_count[(driver, track_type)] += 1"""),

("""            trk = track_stats[(driver, track_id)]
            trk["n"] += 1
            trk["sum_finish"] += finish
            if finish < trk["best"]:
                trk["best"] = finish""",
"""            trk = track_stats[(driver, track_id)]
            trk["n"] += 1
            if finish < trk["best"]:
                trk["best"] = finish
            tku = track_hist[(driver, track_id)]
            tku.append(finish)
            if len(tku) > 5:
                tku.popleft()"""),

("""            team = team_lookup.get(driver, "")
            tt_team = team_type_stats[(team, track_type)]
            tt_team["n"] += 1
            tt_team["sum_finish"] += finish""",
"""            team = team_lookup.get(driver, "")
            ttu = team_type_hist[(team, track_type)]
            ttu.append(finish)
            if len(ttu) > 30:
                ttu.popleft()
            team_type_count[(team, track_type)] += 1"""),
]

p = Path("src/features/rolling.py")
if not p.exists():
    sys.exit("Run from the project root.")
src = p.read_text(encoding="utf-8")
if "type_hist" in src:
    sys.exit("Already patched (type_hist present). Nothing to do.")
for old, new in R:
    n = src.count(old)
    if n != 1:
        sys.exit("ABORT: anchor found %dx (expected 1): %r" % (n, old[:60]))
    src = src.replace(old, new, 1)
shutil.copy2(str(p), str(p) + ".winbak")
p.write_text(src, encoding="utf-8")
import py_compile
py_compile.compile(str(p), doraise=True)
print("Patched OK: src/features/rolling.py (compiles). Run: python backtest_oddslogic_v5.py")
