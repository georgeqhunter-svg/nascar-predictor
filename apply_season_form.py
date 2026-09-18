"""apply_season_form.py - add season-to-date form features to rolling.py + FEATURES.

season_stats already accumulates wins/top5/top10 per (driver, season) but never
records avg finish. Add n + sum_finish, emit races_ytd and avg_finish_ytd, and
register both in gbm_ranker.FEATURES. Walk-forward safe (update after emission).
"""
from pathlib import Path
import shutil
import sys

R_ROLLING = [
("""    season_stats: dict[tuple, dict] = defaultdict(lambda: {"wins": 0, "top5": 0, "top10": 0})""",
"""    season_stats: dict[tuple, dict] = defaultdict(
        lambda: {"wins": 0, "top5": 0, "top10": 0, "n": 0, "sum_finish": 0}
    )"""),

("""                "season_wins_ytd": ss["wins"],""",
"""                "races_ytd": ss["n"],
                "avg_finish_ytd": (ss["sum_finish"] / ss["n"]) if ss["n"] else np.nan,
                "season_wins_ytd": ss["wins"],"""),

("""            ss = season_stats[(driver, season)]
            if finish == 1:""",
"""            ss = season_stats[(driver, season)]
            ss["n"] += 1
            ss["sum_finish"] += finish
            if finish == 1:"""),
]

R_GBM = [
("""    "practice_consistency_z", "practice_laps_run_z", "has_practice_data",""",
"""    "practice_consistency_z", "practice_laps_run_z", "has_practice_data",
    "races_ytd", "avg_finish_ytd","""),
]


def patch(path, replacements, backup_suffix):
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    for old, new in replacements:
        n = src.count(old)
        if n != 1:
            sys.exit("ABORT [%s]: anchor found %dx (expected 1): %r" % (path, n, old[:60]))
        src = src.replace(old, new, 1)
    shutil.copy2(str(p), str(p) + backup_suffix)
    p.write_text(src, encoding="utf-8")
    print("Patched OK: " + path)


bf = Path("src/features/rolling.py")
if not bf.exists():
    sys.exit("Run from the project root.")
if "avg_finish_ytd" in bf.read_text(encoding="utf-8"):
    sys.exit("Already patched (avg_finish_ytd present). Nothing to do.")

patch("src/features/rolling.py", R_ROLLING, ".formbak")
patch("src/models/gbm_ranker.py", R_GBM, ".formbak")

import py_compile
py_compile.compile("src/features/rolling.py", doraise=True)
py_compile.compile("src/models/gbm_ranker.py", doraise=True)
print("Both compile. Run: python backtest_oddslogic_v5.py")
