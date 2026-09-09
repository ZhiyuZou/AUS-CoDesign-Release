# -*- coding: utf-8 -*-
"""Find environmental conditions that yield target Vt values."""
import numpy as np

# === Auto-added path setup (project reorganization) ===
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# === End path setup ===

from plume_calculator import effective_volume


Q0 = 0.0016667  # m^3/s per nozzle


def Vt_of_uc(uc, m=1, tide=0.0):
    Vt, info = effective_volume(uc, u_next=uc, tide=tide, m=m, Q0=Q0)
    return Vt, info


def find_uc(target, m=1, tide=0.0, lo=0.02, hi=0.25):
    """Binary search for u_c that gives target Vt (steady current)."""
    for _ in range(100):
        mid = (lo + hi) / 2
        v, _ = Vt_of_uc(mid, m, tide)
        if v > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


targets = [132.313, 107.125]
for tgt in targets:
    uc = find_uc(tgt)
    Vt, info = Vt_of_uc(uc)
    print(f'--- Target Vt = {tgt} m3/h ---')
    print(f'  u_c = u_next = {uc:.4f} m/s')
    print(f'  tide = 0.0 m, m = 1, Q0 = {Q0*1000:.2f} L/s')
    print(f'  Zm  = {info["Zm"]:.3f} m   (target 8.0 m)')
    print(f'  xs  = {info["xs"]:.3f} m   xd = {info["xd"]:.1f} m')
    print(f'  Qd  = {info["Qd"]:.4f} m3/s')
    print(f'  Qt  = {info["Qt"]:.4f} m3/s')
    print(f'  kata= {info["kata"]:.4f}')
    qk = info["Qt"] * info["kata"]
    print(f'  Qt*kata = {qk:.5f} m3/s = {qk*3600:.3f} m3/h')
    print(f'  Vt  = {Vt:.3f} m3/h')
    print()
