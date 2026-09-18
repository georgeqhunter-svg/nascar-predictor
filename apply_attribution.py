"""apply_attribution.py - per-driver log-loss cost attribution in the backtest.

Adds a module-level accumulator charged in the matchup loop (each participant gets
the matchup's model_ll and market_ll), and prints a cost leaderboard at the end of
main(): who accounts for the model's excess nats vs the market.
"""
from pathlib import Path
import shutil
import sys

R = [
("_EVALUATED_RIDS: set = set()",
"""_EVALUATED_RIDS: set = set()
from collections import defaultdict as _dd
DRIVER_ATTRIB = _dd(lambda: [0.0, 0.0, 0])  # driver -> [model_ll_sum, market_ll_sum, n]"""),

("""        market_ll.append(-_log_clip(p_market_won))
        model_ll.append(-_log_clip(p_model_won))""",
"""        _kl = -_log_clip(p_market_won)
        _ml = -_log_clip(p_model_won)
        market_ll.append(_kl)
        model_ll.append(_ml)
        DRIVER_ATTRIB[a][0] += _ml; DRIVER_ATTRIB[a][1] += _kl; DRIVER_ATTRIB[a][2] += 1
        DRIVER_ATTRIB[b][0] += _ml; DRIVER_ATTRIB[b][1] += _kl; DRIVER_ATTRIB[b][2] += 1"""),

("""    total_mkt_c = sum(s["market_correct"] for s in summary)
    total_mdl_c = sum(s["model_correct"] for s in summary)""",
"""    total_mkt_c = sum(s["market_correct"] for s in summary)
    total_mdl_c = sum(s["model_correct"] for s in summary)

    attrib = sorted(
        ((d, v[0] - v[1], v[2]) for d, v in DRIVER_ATTRIB.items()),
        key=lambda x: -x[1],
    )
    print()
    print("DRIVER COST ATTRIBUTION (excess nats vs market; each matchup charges both drivers):")
    for d, xs, k in attrib[:15]:
        print(f"  {d:24s} excess={xs:+7.2f}  n={k:3d}  per-matchup={xs/max(k,1):+.4f}")
    print("  ... best vs market:")
    for d, xs, k in attrib[-5:]:
        print(f"  {d:24s} excess={xs:+7.2f}  n={k:3d}  per-matchup={xs/max(k,1):+.4f}")
    tot_excess = sum(v[0] - v[1] for v in DRIVER_ATTRIB.values()) / 2
    top5 = sum(x[1] for x in attrib[:5]) / 2
    if tot_excess > 1e-9:
        print(f"  TOTAL excess={tot_excess:+.2f} nats; top-5 drivers hold {top5/tot_excess*100:.0f}% of it")
    else:
        print(f"  TOTAL excess={tot_excess:+.2f} nats (model at or ahead of market)")"""),
]

p = Path("backtest_oddslogic_v5.py")
if not p.exists():
    sys.exit("Run from the project root.")
src = p.read_text(encoding="utf-8")
if "DRIVER_ATTRIB" in src:
    sys.exit("Already patched (DRIVER_ATTRIB present). Nothing to do.")
for old, new in R:
    n = src.count(old)
    if n != 1:
        sys.exit("ABORT: anchor found %dx (expected 1): %r" % (n, old[:60]))
    src = src.replace(old, new, 1)
shutil.copy2(str(p), str(p) + ".attribbak")
p.write_text(src, encoding="utf-8")
import py_compile
py_compile.compile(str(p), doraise=True)
print("Patched OK: backtest_oddslogic_v5.py (compiles). Run: python backtest_oddslogic_v5.py")
