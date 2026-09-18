"""apply_alpha_by_type.py - per-track-type GBM/PL blend weight on the calibration pool.

Same matchup-LL objective as the temperature search; calibrate.py untouched;
walk-forward safe (past races only). Thin types (<4) fall back to global ALPHA.
"""
from pathlib import Path
import shutil
import sys

R = [('    val_races, tt_list = [], []', '    val_races, tt_list = [], []\n    val_parts = []'), ('        b = ALPHA * gbm_z + (1 - ALPHA) * pl_z\n        b = b / max(b.std(), 1e-6)', '        b = ALPHA * gbm_z + (1 - ALPHA) * pl_z\n        b = b / max(b.std(), 1e-6)\n        val_parts.append((gbm_z, pl_z, sub["finish_pos"].to_numpy(),\n                          sub["is_dnf"].to_numpy(),\n                          np.full(len(sub), HAZARD.get(sub["track_type"].iloc[0], 0.08))))'), ('    T_by_type = find_best_temperature_by_type(val_races, tt_list, default_T=1.0, n_samples=1500)', '    # ---- Per-track-type ALPHA (GBM/PL blend) + temperature, one objective ----\n    ALPHA_GRID = (0.50, 0.65, 0.75, 0.85, 0.925, 1.00)\n    alpha_by_type: dict = {}\n    T_by_type: dict = {}\n    _by_type: dict = {}\n    for _comp, _t in zip(val_parts, tt_list):\n        _by_type.setdefault(_t, []).append(_comp)\n    for _t, _races_t in _by_type.items():\n        if len(_races_t) < 4:\n            continue\n        _best = (float("inf"), ALPHA, 1.0)  # (matchup_ll, alpha, T)\n        for _a in ALPHA_GRID:\n            _vr = []\n            for _gz, _pz, _fin, _dnf, _hz in _races_t:\n                _b = _a * _gz + (1.0 - _a) * _pz\n                _b = _b / max(_b.std(), 1e-6)\n                _vr.append(RaceScoresGT(scores=_b, finishes=_fin, is_dnf=_dnf, hazards=_hz))\n            _T_a, _tbl = find_best_temperature(_vr, n_samples=1500)\n            _ll_a = float(_tbl["matchup_ll"].min())\n            if _ll_a < _best[0] - 1e-4:\n                _best = (_ll_a, _a, _T_a)\n        alpha_by_type[_t] = _best[1]\n        T_by_type[_t] = _best[2]\n    # Rebuild pool scores under each race\'s own type-ALPHA so downstream\n    # matchup-temperature calibration sees consistent strengths.\n    for _i, (_comp, _t) in enumerate(zip(val_parts, tt_list)):\n        _gz, _pz, _fin, _dnf, _hz = _comp\n        _a = alpha_by_type.get(_t, ALPHA)\n        _b = _a * _gz + (1.0 - _a) * _pz\n        val_races[_i].scores = _b / max(_b.std(), 1e-6)'), ('    T = T_by_type.get(tt, 1.0)', '    T = T_by_type.get(tt, 1.0)\n    alpha_t = alpha_by_type.get(tt, ALPHA)'), ('    blended = ALPHA * gbm_z + (1 - ALPHA) * pl_z', '    blended = alpha_t * gbm_z + (1 - alpha_t) * pl_z'), ('    from src.models.calibrate import RaceScoresGT, find_best_temperature_by_type', '    from src.models.calibrate import (RaceScoresGT, find_best_temperature,\n                                      find_best_temperature_by_type)'), ('({tt}, T={T}, Tm={T_match:.2f})', '({tt}, T={T}, a={alpha_t:.2f}, Tm={T_match:.2f})')]

p = Path("backtest_oddslogic_v5.py")
if not p.exists():
    sys.exit("Run from the project root.")
src = p.read_text(encoding="utf-8")
if "alpha_by_type" in src:
    sys.exit("Already patched (alpha_by_type present). Nothing to do.")
for old, new in R:
    n = src.count(old)
    if n != 1:
        sys.exit("ABORT: anchor found %dx (expected 1): %r" % (n, old[:60]))
    src = src.replace(old, new, 1)
shutil.copy2(str(p), str(p) + ".alphabak")
p.write_text(src, encoding="utf-8")
import py_compile
py_compile.compile(str(p), doraise=True)
print("Patched OK: backtest_oddslogic_v5.py (compiles). NOTE: calibration is ~6x slower now.")
print("Run: python backtest_oddslogic_v5.py")
