"""apply_guardrails.py - one-time patch for backtest_oddslogic_v5.py."""
from pathlib import Path
import shutil
import sys

TARGET = Path("backtest_oddslogic_v5.py")
if not TARGET.exists():
    sys.exit("Run this from the project root (backtest_oddslogic_v5.py not found here).")

src = TARGET.read_text(encoding="utf-8")

if "_EVALUATED_RIDS" in src:
    print("Already patched (_EVALUATED_RIDS present). Nothing to do.")
    sys.exit(0)

REPLACEMENTS = [
    (
        "def run_race(",
        """# Race IDs this backtest evaluates. Filled as each race is processed and
# excluded from every calibration pool, so no evaluated race ever helps
# calibrate another (walk-forward purity across the whole backtest).
_EVALUATED_RIDS: set = set()


def run_race(""",
    ),
    (
        "    target_rid = r.iloc[0][\"race_id_short\"]",
        """    target_rid = r.iloc[0]["race_id_short"]
    _EVALUATED_RIDS.add(target_rid)""",
    ),
    (
        "    val_ids = race_order[30:]",
        """    # Guardrail: never calibrate on races this backtest evaluates, and
    # never silently fall back to default temperatures on an empty pool.
    val_ids = [rid for rid in race_order[30:] if rid not in _EVALUATED_RIDS]
    if len(val_ids) < 4:
        print(f"WARNING [{name_match}]: validation pool has only {len(val_ids)} "
              f"races (of {len(race_order)} train races) - calibration falls back "
              f"to T=1.0/Tm=1.0. Check training-data span!")""",
    ),
    (
        'print("SUMMARY (six races vs OddsLogic closing lines)")',
        'print(f"SUMMARY ({len(summary)} races vs OddsLogic closing lines)")',
    ),
]

for old, new in REPLACEMENTS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor found {n} times (expected 1), not patching: {old[:60]!r}")
    src = src.replace(old, new, 1)

shutil.copy2(TARGET, TARGET.with_suffix(TARGET.suffix + ".bak"))
TARGET.write_text(src, encoding="utf-8")
print("Patched OK. Backup at backtest_oddslogic_v5.py.bak")