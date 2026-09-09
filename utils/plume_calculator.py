# -*- coding: utf-8 -*-
"""
Standalone bubble-entrained plume (BEP) calculator for air-injection artificial upwelling.
Extracted from fast_refined.py -> AUS_Environment.calc_Vt().

Given cross-flow velocity and tide level, computes:
  Zd  : bubble/plume separation height          [m]
  Zm  : maximum plume rise height               [m]
  Bd  : plume half-width at separation          [m]
  b_m : plume half-width at maximum rise height [m]
  reach : whether Zm reaches the cultivation layer (Zm >= Z_s + tide)

Optionally (with u_next) also computes the effective hourly nutrient-water volume Vt.

Usage:
  1) Run demo:        python plume_calculator.py
  2) Import:          from plume_calculator import plume_geometry, effective_volume
  3) Batch (arrays):  see __main__ demo below
"""
import numpy as np

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

# ==============================================================================
# Physical constants (defaults from Aoshan Bay AUS, same as fast_refined.py)
# ==============================================================================
RHO_W      = 1025.19   # ambient seawater density [kg/m^3]
RHO_DELTA  = 0.04      # density anomaly of bottom water [kg/m^3]
RHO_A      = 1018.0    # ambient density used in centerline coefficient [kg/m^3]
V_S        = 0.3       # bubble slip velocity [m/s]
G          = 9.81      # gravitational acceleration [m/s^2]
H0         = 10.4      # atmospheric pressure head [m]
D0         = 0.8       # nozzle diameter [m]
Z_S        = 8.0       # cultivation layer depth [m]
X_D        = 45.0      # lateral distance from nozzle to farm boundary [m]
FARM_LEN   = 120.0     # farm length along the dominant current direction [m]
DT         = 1.0       # time step [h]
BETA_JET   = 0.17      # entrainment/expansion coefficient of the negative-buoyancy jet


# ==============================================================================
# Core geometry
# ==============================================================================
def plume_geometry(u_c, Q0=0.002, d0=D0, v_s=V_S, rho_w=RHO_W, rho_delta=RHO_DELTA,
                   rho_a=RHO_A, g=G, H0=H0):
    """
    Compute bubble-plume geometry for a single nozzle in cross-flow.

    Parameters
    ----------
    u_c  : float or array - cross-current velocity [m/s]
    Q0   : float - air injection rate per nozzle [m^3/s]
    d0   : float - nozzle diameter [m]
    v_s  : float - bubble slip velocity [m/s]
    rho_w: float - ambient water density [kg/m^3]
    rho_delta : float - bottom-water density anomaly [kg/m^3]
    rho_a: float - ambient density for centerline coefficient [kg/m^3]
    g    : float - gravity [m/s^2]
    H0   : float - pressure head [m]

    Returns
    -------
    dict with keys: Zd, alpha, A, delta_z, Bd, vd, Qd, v0, Q0_calc,
                    rho_d, vmd, xi, Zm, b_m   (all scalars or arrays matching u_c)
    """
    u = np.maximum(np.abs(u_c), 1e-8)

    # --- Bubble-plume stage (before separation) ---
    # Separation height (Socolofsky & Adams, 2002)
    Zd = 5.1 * Q0 / ((v_s ** 2.4) * u) ** 0.88 * g

    # Entrainment coefficient (Qiang et al., 2018)
    alpha = 0.082 * np.tanh((g * Q0 / H0) ** (1.0 / 3.0) / v_s) ** (3.0 / 8.0)

    # Centerline velocity coefficient (Qiang et al., 2018, Eq. 32)
    A = 1.02 * alpha ** (-1) * (g * 1.2 * Q0 * 1.25 / np.pi / rho_a) ** (1.0 / 3.0)

    # Virtual origin offset
    delta_z = d0 / (2.4 * alpha)

    # Plume half-width, velocity, volume flux at separation
    Bd = 1.2 * (Zd + delta_z) * alpha
    vd = A * (Zd + delta_z) ** (-1.0 / 3.0)
    Qd = np.pi * Bd ** 2 * np.sqrt(np.asarray(u_c, dtype=float) ** 2 + vd ** 2)

    # Initial plume flux at the nozzle
    v0 = A * delta_z ** (-1.0 / 3.0)
    Q0_calc = np.pi * d0 ** 2 * np.sqrt(np.asarray(u_c, dtype=float) ** 2 + v0 ** 2)

    # Density of the separated plume (mixing of entrained ambient + dense bottom water)
    rho_d = ((Qd - Q0_calc) * rho_w + Q0_calc * (rho_w + rho_delta)) / Qd

    # Average plume velocity at separation
    vmd = vd * np.sqrt(2.0 * np.pi) / 6.0

    # Buoyancy ratio xi = rho_d / (rho_d - rho_a)
    xi = rho_d / (rho_d - rho_w)

    # --- Negative-buoyancy jet stage (after separation) ---
    # Maximum rise height (Eq. 42 with x -> infinity)
    Zm = (np.sqrt((Bd / BETA_JET) ** 2
                  + 4.0 / 3.0 * xi * Bd * vmd ** 2 / (BETA_JET * g))
          - Bd / BETA_JET + Zd)

    # Plume half-width at maximum rise height: b(z) = Bd + beta*(z - Zd)
    b_m = Bd + BETA_JET * (Zm - Zd)

    return dict(Zd=Zd, alpha=alpha, A=A, delta_z=delta_z, Bd=Bd, vd=vd, Qd=Qd,
                v0=v0, Q0_calc=Q0_calc, rho_d=rho_d, vmd=vmd, xi=xi, Zm=Zm, b_m=b_m)


# ==============================================================================
# Current-effect coefficient kata (Zhang et al., 2023, Eq. 12)
# ==============================================================================
def kata_coefficient(u_c, u_next, xs, xd, farm_len=FARM_LEN, dt=DT, clip=True):
    """
    Compute the tidal current-effect coefficient kata.

    Accounts for advection of the upwelled plume relative to the farm
    boundaries, including current reversal between consecutive time steps.
    kata acts as an efficiency multiplier on the plume volume flux; a value
    of 1 means all upwelled water is retained within the farm, while 0 means
    it is fully advected beyond the boundary.

    Parameters
    ----------
    u_c      : float - current velocity at this step [m/s]
    u_next   : float - current velocity at next step [m/s]
    xs       : float - horizontal displacement of the plume centreline when it
                       reaches the cultivation layer [m]
    xd       : float - lateral distance from nozzle to the down-current farm
                       boundary [m]
    farm_len : float - farm length along the dominant current direction [m]
    dt       : float - time step [h]
    clip     : bool  - if True, clip kata to [0, 1]

    Returns
    -------
    kata : float - dimensionless current-effect coefficient
    xaq  : float - effective boundary distance after considering reversal [m]
    """
    u = max(abs(u_c), 1e-8)
    u_n = max(abs(u_next), 1e-8)
    u_dt = u * dt * 3600.0          # distance travelled per step [m]
    denom = max(xd - xs, 1e-6)

    # Current reversal between steps -> plume can exploit the full farm length
    xaq = farm_len if ((u_next > 0 and u_c < 0) or (u_next < 0 and u_c > 0)) else xd

    kata = ((xd - xs) / u_dt) ** 2 * (
        (u_dt - xd) / denom
        + (u / u_n) * ((xaq - xd) / denom + 0.5)
        + 0.5
    )
    if clip:
        kata = float(np.clip(kata, 0.0, 1.0))
    return kata, xaq


# ==============================================================================
# Reach check + effective volume (full Vt model, needs u_next)
# ==============================================================================
def effective_volume(u_c, u_next, tide=0.0, m=1, Q0=0.002, d0=D0, v_s=V_S,
                     rho_w=RHO_W, rho_delta=RHO_DELTA, rho_a=RHO_A, g=G, H0=H0,
                     Z_s=Z_S, x_d=X_D, farm_len=FARM_LEN, dt=DT):
    """
    Compute the effective nutrient-rich water volume delivered to the farm in
    one time step, replicating AUS_Environment.calc_Vt() from fast_refined.py.

    Parameters
    ----------
    u_c    : float - current velocity at this step [m/s]
    u_next : float - current velocity at next step [m/s]
    tide   : float - tide level added to cultivation depth [m]
    m      : int   - number of active compressors/nozzles
    Q0     : float - air injection rate per nozzle [m^3/s]
    (others : physical/geometric parameters, see plume_geometry)

    Returns
    -------
    Vt  : float - effective volume per hour for m nozzles [m^3]
    info: dict  - geometry + reach flags
    """
    geom = plume_geometry(u_c, Q0, d0, v_s, rho_w, rho_delta, rho_a, g, H0)
    Zd, Zm, Bd, alpha, A = geom['Zd'], geom['Zm'], geom['Bd'], geom['alpha'], geom['A']
    delta_z, xi, vmd = geom['delta_z'], geom['xi'], geom['vmd']
    Q0_calc = geom['Q0_calc']

    z_target = Z_s + tide
    u = max(abs(u_c), 1e-8)
    u_n = max(abs(u_next), 1e-8)

    # Lateral distance to farm boundary (direction-dependent)
    xd = x_d if u_c > 0 else farm_len - x_d

    # Reach check
    reaches_layer = bool(Zm >= z_target)
    Vt = 0.0
    xs = np.nan
    within_farm = False
    kata = 0.0
    xaq = xd

    if reaches_layer:
        # Horizontal displacement at cultivation-layer height
        if Zd > z_target:
            xs = (((z_target + delta_z) ** (4.0 / 3.0) - delta_z ** (4.0 / 3.0))
                  * 3.0 / 4.0 / A) * u
        else:
            term = (1.0
                    - 3.0 * BETA_JET * g / (4.0 * xi * Bd * vmd ** 2)
                    * (z_target - Zd) * (z_target + 2.0 * Bd / BETA_JET - Zd))
            term = max(term, 0.0)
            xs = xi * vmd * u / g * (1.0 - term ** (2.0 / 3.0)) + (
                    ((Zd + delta_z) ** (4.0 / 3.0) - delta_z ** (4.0 / 3.0))
                    * 3.0 / 4.0 / A) * u

        within_farm = bool(xs < xd)

        if within_farm:
            # Current-effect coefficient (Zhang et al., 2023)
            kata, xaq = kata_coefficient(u_c, u_next, xs, xd, farm_len, dt)
            Vavg = kata * Q0_calc * dt * 3600.0
            Vt = m * Vavg

    info = dict(Zd=Zd, Zm=Zm, Bd=Bd, b_m=geom['b_m'], xs=xs, xd=xd,
                xaq=xaq, kata=kata, Q0_calc=Q0_calc, Qd=geom['Qd'],
                Qt=m * Q0_calc, Q_eff=kata * m * Q0_calc,
                reaches_layer=reaches_layer, within_farm=within_farm)
    return Vt, info


# ==============================================================================
# Demo
# ==============================================================================
if __name__ == '__main__':
    import pandas as pd


    Q0 = 0.0016667  # m^3/s per nozzle (typical)

    print('=' * 78)
    print(f'Bubble-plume geometry  |  Q0 = {Q0*1000:.1f} L/s per nozzle, '
          f'd0 = {D0} m, Z_s = {Z_S} m')
    print('=' * 78)

    # ---- 1) Geometry vs current velocity (tide = 0) ----
    u_list = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows = []
    for uc in u_list:
        g = plume_geometry(uc, Q0=Q0)
        rows.append({
            'u_c [m/s]': uc,
            'Zd [m]': round(g['Zd'], 2),
            'Zm [m]': round(g['Zm'], 2),
            'Bd [m]': round(g['Bd'], 2),
            'b_m [m]': round(g['b_m'], 2),
            'reach Zs': 'YES' if g['Zm'] >= Z_S else 'no',
        })
    df = pd.DataFrame(rows)
    print('\n--- Plume geometry vs cross-flow velocity (tide = 0 m) ---')
    print(df.to_string(index=False))

    # ---- 2) Zm vs tide level at fixed current ----
    print('\n--- Maximum rise height Zm vs tide level (u_c = 0.10 m/s) ---')
    for tide in [0.0, 0.5, 1.0, 1.5, 2.0]:
        g = plume_geometry(0.10, Q0=Q0)
        reach = 'YES' if g['Zm'] >= Z_S + tide else 'no'
        print(f'  tide = {tide:.1f} m | target depth = {Z_S+tide:.1f} m | '
              f'Zm = {g["Zm"]:.2f} m | b_m = {g["b_m"]:.2f} m | reach = {reach}')

    # ---- 3) Effective volume + kata example (full model) ----
    print('\n--- Effective hourly volume Vt and kata (single nozzle, m=1) ---')
    print(f'{"u_c [m/s]":>10} {"Zm [m]":>8} {"xs [m]":>8} {"Qd [m3/s]":>10} '
          f'{"Qt [m3/s]":>10} {"kata":>7} {"Vt [m3/h]":>11}')
    for uc in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
        Vt, info = effective_volume(uc, u_next=uc, tide=0.0, m=1, Q0=Q0)
        print(f'{uc:>10.2f} {info["Zm"]:>8.2f} {info["xs"]:>8.1f} '
              f'{info["Qd"]:>10.4f} {info["Qt"]:>10.4f} '
              f'{info["kata"]:>7.3f} {Vt:>11.1f}')

    # ---- 4) kata sensitivity: same xs, varying current (steady, no reversal) ----
    print('\n--- kata vs current velocity at fixed plume displacement xs = 10 m ---')
    for uc in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        k, _ = kata_coefficient(uc, u_next=uc, xs=10.0, xd=X_D)
        u_dt = uc * 3600.0
        print(f'  u_c = {uc:.2f} m/s (u*dt = {u_dt:5.0f} m/step) | kata = {k:.3f}')

    # ---- 5) Quick single query ----
    print('\n--- Quick query ---')
    uc_in = float(input('  Enter current velocity u_c [m/s] (default 0.1): ') or 0.1)
    un_in = float(input('  Enter next-step velocity u_next [m/s] (default = u_c): ') or uc_in)
    tide_in = float(input('  Enter tide level [m] (default 0): ') or 0.0)
    g = plume_geometry(uc_in, Q0=Q0)
    print(f'  Zd  = {g["Zd"]:.3f} m')
    print(f'  Zm  = {g["Zm"]:.3f} m   (target = {Z_S+tide_in:.1f} m, '
          f'reach = {"YES" if g["Zm"] >= Z_S+tide_in else "no"})')
    print(f'  Bd  = {g["Bd"]:.3f} m')
    print(f'  b_m = {g["b_m"]:.3f} m')
    Vt, info = effective_volume(uc_in, un_in, tide=tide_in, m=1, Q0=Q0)
    print(f'  xs  = {info["xs"]:.2f} m   xd = {info["xd"]:.1f} m')
    print(f'  Qd  = {info["Qd"]:.4f} m^3/s  (plume flux at separation)')
    print(f'  Qt  = {info["Qt"]:.4f} m^3/s  (total plume flux, m=1)')
    print(f'  kata= {info["kata"]:.3f}')
    print(f'  Vt  = {Vt:.1f} m^3/h (per nozzle)')
