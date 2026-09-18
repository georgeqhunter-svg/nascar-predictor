"""apply_hazard_cal.py - make calibration use the same measured hazards as scoring.

Backtest currently fits temperatures with flat HAZARD/0.08 validation hazards, then
scores target races with per_driver_hazards(). This aligns validation/calibration
sampling with final scoring.

Patches backtest_oddslogic_v5.py only. Aborts if anchors are not unique.
"""
from pathlib import Path


def patch(path, repls):
    p = Path(path)
    raw = p.read_bytes()
    t = raw.decode("utf-8")
    nl = "\r\n" if "\r\n" in t else "\n"
    for i, (old, new) in enumerate(repls, 1):
        o = old.replace("\n", nl)
        n = new.replace("\n", nl)
        cnt = t.count(o)
        if cnt != 1:
            raise SystemExit(f"{path}: anchor {i} found {cnt}x (expected 1) - NOT written")
        t = t.replace(o, n)
    p.write_bytes(t.encode("utf-8"))
    print(f"patched {path} ({len(repls)} edits)")


B = []
B.append((
    '            hazards=np.full(len(sub), HAZARD.get(sub["track_type"].iloc[0], 0.08)),',
    '            hazards=per_driver_hazards(sub, sub["track_type"].iloc[0]),',
))
B.append((
    '        haz_v = np.full(len(vr.scores), 0.08)',
    '        haz_v = vr.hazards',
))

patch("backtest_oddslogic_v5.py", B)
print("all patches applied")
