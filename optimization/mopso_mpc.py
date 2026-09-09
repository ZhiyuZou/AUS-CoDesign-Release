# ==============================================================================
# ==============================================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
from typing import Optional
import time
from numba import jit
from scipy.optimize import minimize

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

# ==============================================================================
# ==============================================================================
@jit(nopython=True, cache=True)
def jit_calc_Vt(m_t, u_c_t, u_next, tide, Q0, v_s, g, d0, x_d, Z_s, rho_w, rho_delta):
    if m_t <= 1e-3: return 0.0
    u_safe = max(abs(u_c_t), 1e-8)
    u_next_safe = max(abs(u_next), 1e-8)

    term1 = (v_s ** 2.4) * u_safe
    Zd = 5.1 * Q0 / (term1 ** 0.88) * g
    term_alpha = (g * Q0 / 10.4) ** (1 / 3) / v_s
    alpha = 0.082 * np.tanh(term_alpha) ** (3 / 8)
    term_A = (g * 1.2 * Q0 * 1.25 / 3.14 / 1018) ** (1 / 3)
    A = 1.02 * (alpha ** (-1)) * term_A
    delta_z = d0 / (2.4 * alpha)
    Bd = 1.2 * (Zd + delta_z) * alpha
    vd = A * ((Zd + delta_z) ** (-1 / 3))
    v0 = A * (delta_z ** (-1 / 3))
    plume_flux_Q0 = 3.14 * (d0 ** 2) * np.sqrt(u_c_t ** 2 + v0 ** 2)
    Qd = 3.14 * (Bd ** 2) * np.sqrt(u_c_t ** 2 + vd ** 2)
    rho_d = ((Qd - plume_flux_Q0) * rho_w + plume_flux_Q0 * (rho_w + rho_delta)) / Qd
    vmd = vd * np.sqrt(2 * 3.14) / 6
    jieta = rho_d / (rho_d - rho_w)
    beta_param = 0.17
    term_sqrt = (Bd / beta_param) ** 2 + (4 / 3 * jieta * Bd * (vmd ** 2) / (beta_param * g))

    if term_sqrt < 0: return 0.0

    Zm = np.sqrt(term_sqrt) - (Bd / beta_param) + Zd
    xd_val = x_d if u_c_t > 0 else 120 - x_d
    if Zm < Z_s + tide: return 0.0
    if Zd > Z_s + tide:
        term_zs = (Z_s + tide + delta_z) ** (4 / 3)
        term_dz = delta_z ** (4 / 3)
        xs = ((term_zs - term_dz) * 3 / 4 / A) * u_safe
    else:
        term_B = 1 - 3 * beta_param * g / (4 * jieta * Bd * vmd ** 2) * (Z_s + tide - Zd) * (
                Z_s + tide + 2 * Bd / beta_param - Zd)
        if term_B < 0: term_B = 0
        term_td = ((Zd + delta_z) ** (4 / 3) - delta_z ** (4 / 3))
        td = term_td * 3 / 4 / A
        xd1 = td * u_safe
        xs = jieta * vmd * u_safe / g * (1 - term_B ** (2 / 3)) + xd1
    if xs >= xd_val: return 0.0
    cond = ((u_next > 0) and (u_c_t < 0)) or ((u_next < 0) and (u_c_t > 0))
    xaq = 120.0 if cond else xd_val
    denom = xd_val - xs
    if abs(denom) < 1e-6: denom = 1e-6
    term_kata = ((xd_val - xs) / (u_safe * 3600)) ** 2
    inner_kata = ((u_safe * 3600 - xd_val) / denom) + (u_safe / u_next_safe) * ((xaq - xd_val) / denom + 0.5) + 0.5
    kata = term_kata * inner_kata
    Vavg = kata * plume_flux_Q0 * 1.0 * 3600
    res = m_t * Vavg
    if np.isnan(res) or res < 0: return 0.0
    return res

@jit(nopython=True, cache=True)
def jit_calc_f2(T_t, I_t, I_s, T_opt, T_min, T_max):
    I_illu = I_t * 0.168
    f_I = min(I_illu / I_s, 2.0)
    if T_t <= T_opt:
        T_x = T_min
    else:
        T_x = T_max
    if abs(T_x - T_opt) < 1e-6:
        f_T = 1.0
    else:
        f_T = np.exp(1 - f_I - 2.3 * ((T_t - T_opt) / (T_x - T_opt)) ** 2)
    return max(f_I * f_T, 0.0)

@jit(nopython=True, cache=True)
def jit_get_solar(I_t, P_AZ, K):
    g_t = (I_t * 1.0 * P_AZ * K) / 1e6
    limit = P_AZ / 1000 * K
    if g_t > limit: g_t = limit
    if g_t < 0: g_t = 0
    return g_t

@jit(nopython=True, cache=True)
def jit_predict_cost(u_seq, init_E, weather_data, params_phys, params_cfg):
    Q0, v_s, g, d0, x_d, Z_s, rho_w, rho_delta, I_s, T_opt, T_min, T_max, K, eta_c, eta_d, c_max, d_max, b, dt, beta = params_phys
    P_AZ, E_max, E_min, p0, M = params_cfg

    total_cost = 0.0
    penalty_E = 0.0
    curr_E = init_E
    H = len(u_seq)

    for k in range(H):
        m_k = u_seq[k]
        T_t = weather_data[k, 0]
        I_t = weather_data[k, 1]
        Z_t = weather_data[k, 2]
        u_t = weather_data[k, 3]
        u_next = weather_data[k + 1, 3] if k < H - 1 else u_t

        Vt = jit_calc_Vt(m_k, u_t, u_next, Z_t, Q0, v_s, g, d0, x_d, Z_s, rho_w, rho_delta)
        f2_t = jit_calc_f2(T_t, I_t, I_s, T_opt, T_min, T_max)

        raw_nti = f2_t * Vt
        step_reward = beta * raw_nti

        penalty_shutdown = 0.0
        if curr_E <= E_min + 1.0 and m_k > 0.1: penalty_shutdown = 1000.0 * m_k
        penalty_ineffective = 0.0
        if m_k > 0.5:
            if Vt < 1e-6 or f2_t < 1e-3: penalty_ineffective = 1.0 * m_k

        g_t = jit_get_solar(I_t, P_AZ, K)
        q_t = m_k * p0
        surplus = g_t - (b + q_t)

        c_t, d_t = 0.0, 0.0
        if surplus > 0:
            c_t = min(surplus, c_max)
        else:
            d_t = min(-surplus, d_max)

        next_E = curr_E + (eta_c * c_t - d_t / eta_d) * dt

        if next_E < E_min: penalty_E += 500.0 * (E_min - next_E) ** 2
        if next_E > E_max: penalty_E += 500.0 * (next_E - E_max) ** 2

        step_cost = -step_reward + penalty_ineffective + penalty_E + penalty_shutdown
        total_cost += step_cost

        if next_E < E_min: next_E = E_min
        if next_E > E_max: next_E = E_max
        curr_E = next_E

    total_cost -= 0.5 * curr_E
    return total_cost

# ==============================================================================
# ==============================================================================
class AUS_Config:
    def __init__(self, data_dir, config_dict):
        self.data_dir = data_dir
        self.dt = 1
        self.total_time = 2880
        self.eta_c, self.eta_d = 0.95, 0.95
        self.c_max, self.d_max = 30.0, 30.0
        self.b = 0.2
        self.Z_s, self.x_d = 8.0, 45.0
        self.rho_delta, self.rho_w = 0.04, 1025.19
        self.H0, self.v_s, self.g = 10.4, 0.3, 9.81
        self.I_s = 180.0 * 0.217
        self.T_opt, self.T_min, self.T_max = 10.0, 0.5, 20.0
        self.beta = 0.001
        self.d0 = 0.8
        self.K = 0.8
        self.config_ranges = {"p_az": (0, 60000), "e_max": (0.0, 120.0), "m": (1, 24), "q0": (0.001, 0.003)}
        self.set_hardware(config_dict)

    def set_hardware(self, config):
        self.P_AZ = config["p_az"]
        self.E_max = config["e_max"]
        self.E_min = 0.05 * self.E_max
        self.M = int(config["m"])
        self.Q0 = config["q0"]
        self.p0 = ((self.Q0 * 60 * 1000 + 58.691) / 0.1442) / 1000
        self.phys_params = np.array([
            self.Q0, self.v_s, self.g, self.d0, self.x_d, self.Z_s,
            self.rho_w, self.rho_delta, self.I_s, self.T_opt, self.T_min, self.T_max,
            self.K, self.eta_c, self.eta_d, self.c_max, self.d_max, self.b, self.dt, self.beta
        ], dtype=np.float64)
        self.cfg_params = np.array([self.P_AZ, self.E_max, self.E_min, self.p0, self.M], dtype=np.float64)

    def get_solar_power(self, I_t):
        return jit_get_solar(I_t, self.P_AZ, self.K)

    def calc_Vt(self, m_t, u_c_t, u_next, tide):
        if np.isscalar(m_t) and np.isscalar(u_c_t):
            return jit_calc_Vt(m_t, u_c_t, u_next, tide, self.Q0, self.v_s, self.g, self.d0, self.x_d, self.Z_s,
                               self.rho_w, self.rho_delta)
        else:
            return np.vectorize(jit_calc_Vt, excluded=[4, 5, 6, 7, 8, 9, 10, 11])(m_t, u_c_t, u_next, tide, self.Q0,
                                                                                  self.v_s, self.g, self.d0, self.x_d,
                                                                                  self.Z_s, self.rho_w, self.rho_delta)

    def calc_f2(self, T_t, I_t):
        if np.isscalar(T_t):
            return jit_calc_f2(T_t, I_t, self.I_s, self.T_opt, self.T_min, self.T_max)
        else:
            return np.vectorize(jit_calc_f2, excluded=[2, 3, 4, 5])(T_t, I_t, self.I_s, self.T_opt, self.T_min,
                                                                    self.T_max)

class AUS_Environment:
    def __init__(self, cfg_obj, preloaded_data=None):
        self.cfg = cfg_obj
        self.train_years = [2021, 2022, 2023, 2024]
        self.year_data_dict = preloaded_data if preloaded_data else {}
        self.current_step = 0
        self.current_state = None
        self.env_data = None
        if not preloaded_data: self.load_all_year_data()

        self.state_mean = np.zeros(10)
        self.state_std = np.ones(10)

    def load_all_year_data(self):
        for y in self.train_years:
            try:
                df = pd.read_excel(os.path.join(self.cfg.data_dir, f"{y}.xlsx"), header=0, index_col=0,
                                   engine="openpyxl")
                df = df.reset_index(drop=True).astype(float).head(self.cfg.total_time).ffill().bfill()
                df.iloc[:, 2] = df.iloc[:, 2] / 100  # Unit fix
                if len(df) < self.cfg.total_time:
                    df = pd.concat([df, pd.DataFrame([df.iloc[-1]] * (self.cfg.total_time - len(df)))],
                                   ignore_index=True)
                self.year_data_dict[y] = df.fillna(0).values
            except:
                pass

    def normalize_state(self, s):
        return (s - self.state_mean) / self.state_std

    def denormalize_state(self, s):
        return s * self.state_std + self.state_mean

    def reset(self, y):
        self.env_data = self.year_data_dict[y]
        self.current_step = 0
        init_E = np.random.uniform(self.cfg.E_min, self.cfg.E_max)
        raw = np.array([init_E, self.env_data[0, 0], self.env_data[0, 1], self.env_data[0, 2], self.env_data[0, 3], 0,
                        self.cfg.P_AZ, self.cfg.E_max, self.cfg.M, self.cfg.Q0])
        self.current_state = self.normalize_state(raw)
        return self.current_state, init_E

    def step(self, act, curr_E):
        m = act

        T, I, Z, u = self.env_data[min(self.current_step, 2879)]
        self.current_step += 1
        d = self.current_step >= 2880
        u_next = self.env_data[min(self.current_step, 2879), 3]

        g = self.cfg.get_solar_power(I)
        q = m * self.cfg.p0
        surplus = g - (self.cfg.b + q)

        c, f = 0, 1
        if surplus > 0:
            c = min(surplus, self.cfg.c_max, (self.cfg.E_max - curr_E) / self.cfg.eta_c)
            E_n = curr_E + c * self.cfg.eta_c
        elif -surplus <= (curr_E - self.cfg.E_min) * self.cfg.eta_d:
            E_n = curr_E + surplus / self.cfg.eta_d
        else:
            m, f, E_n = 0, 0, curr_E

        E_n = np.clip(E_n, self.cfg.E_min, self.cfg.E_max)
        Vt = self.cfg.calc_Vt(m, u, u_next, Z)
        f2 = self.cfg.calc_f2(T, I)

        r = self.cfg.beta * Vt * f2 if (f == 1 and (m == 0 or (Vt > 1e-6 and f2 > 1e-2))) else -1

        info = {"Vt": Vt, "f2_t": f2, "energy_waste": max(surplus - c, 0), "flag": f}

        raw_n = np.array(
            [E_n, self.env_data[min(self.current_step, 2879), 0], self.env_data[min(self.current_step, 2879), 1],
             self.env_data[min(self.current_step, 2879), 2], self.env_data[min(self.current_step, 2879), 3],
             self.current_step, self.cfg.P_AZ, self.cfg.E_max, self.cfg.M, self.cfg.Q0])
        self.current_state = self.normalize_state(raw_n)

        return self.current_state, r, d, info, E_n

# ==============================================================================
# ==============================================================================
class MPC_Controller:
    def __init__(self, cfg: AUS_Config, horizon=24):
        self.cfg, self.H, self.prev_solution = cfg, horizon, None

    def solve(self, current_E, env_future_data):
        H_act = len(env_future_data) - 1
        if H_act == 0: return 0.0
        w_arr = np.array(env_future_data, dtype=np.float64)

        if self.prev_solution is not None and len(self.prev_solution) == H_act:
            x0 = np.roll(self.prev_solution, -1)
            x0[-1] = x0[-2]
        else:
            x0 = np.zeros(H_act)
            for k in range(H_act):
                if w_arr[k, 1] > 100: x0[k] = self.cfg.M / 2

        bounds = [(0, self.cfg.M) for _ in range(H_act)]

        res = minimize(fun=jit_predict_cost, x0=x0, args=(current_E, w_arr, self.cfg.phys_params, self.cfg.cfg_params),
                       method='SLSQP', bounds=bounds, options={'ftol': 1e-5, 'disp': False, 'maxiter': 100})

        if res.success:
            self.prev_solution = res.x
            return int(round(res.x[0]))
        else:
            self.prev_solution = None
            return 0

# ==============================================================================
# ==============================================================================
class CostCalculator:
    def __init__(self):
        self.C_EGS_per_kw = 95.9
        self.C_ESS_per_kwh = 49.2
        self.C_Other = 140.0
        self.C_Compressor_base = 116.8
        self.C_Nozzle = 217.0
        self.C_Pipe = 70.1

    def calculate_alcc(self, config):
        p_az_kw = config['p_az'] / 1000.0
        cost_egs = p_az_kw * self.C_EGS_per_kw
        cost_ess = config['e_max'] * self.C_ESS_per_kwh
        q0_lmin = config['q0'] * 60 * 1000
        unit_capacity_factor = q0_lmin / 100.0
        cost_compressor_unit = (unit_capacity_factor * self.C_Compressor_base) + self.C_Nozzle + self.C_Pipe
        cost_air_system = config['m'] * cost_compressor_unit
        total_alcc = cost_egs + cost_ess + cost_air_system + self.C_Other
        return total_alcc

# ==============================================================================
# ==============================================================================
def evaluate_single_particle_mpc(position, bounds, env_data_dir, preloaded_data, fast_mode=False):
    def decode_pos(pos, b):
        p_az = np.clip(pos[0], b[0][0], b[0][1])
        e_max = np.clip(pos[1], b[1][0], b[1][1])
        m = int(np.clip(round(pos[2]), b[2][0], b[2][1]))
        q0 = np.clip(pos[3], b[3][0], b[3][1])
        return {"p_az": p_az, "e_max": e_max, "m": m, "q0": q0}

    config = decode_pos(position, bounds)
    cost_calc = CostCalculator()
    alcc = cost_calc.calculate_alcc(config)

    cfg_obj = AUS_Config(env_data_dir, config)
    env = AUS_Environment(cfg_obj, preloaded_data=preloaded_data)
    mpc_controller = MPC_Controller(cfg_obj, horizon=24)

    total_nti = 0

    if fast_mode:
        years = [2024]
    else:
        years = env.train_years

    for year in years:
        if year in env.year_data_dict:
            _, curr_E = env.reset(year)
            done = False
            year_nti = 0

            mpc_controller.prev_solution = None

            while not done:
                end_idx = min(env.current_step + 25, 2880)
                future_data = env.env_data[env.current_step: end_idx]

                optimal_machines = mpc_controller.solve(curr_E, future_data)

                _, _, done, info, next_E = env.step(optimal_machines, curr_E)

                if info['flag']:
                    year_nti += info['f2_t'] * info['Vt']

                curr_E = next_E

            total_nti += year_nti

    avg_nti = total_nti / len(years)
    return np.array([alcc, -avg_nti])

# ==============================================================================
# ==============================================================================
class Particle:
    def __init__(self, bounds):
        self.position = np.array([random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.objectives = np.array([float('inf'), float('inf')])
        self.best_objectives = np.array([float('inf'), float('inf')])

class AUS_MOPSO_Parallel_MPC:
    def __init__(self, data_dir, bounds, num_particles=50, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, fast_mode=False):
        self.data_dir = data_dir
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.fast_mode = fast_mode

        print("Preloading environmental data...")
        temp_cfg = AUS_Config(data_dir, {"p_az": 48000, "e_max": 80, "m": 16, "q0": 0.002})
        temp_env = AUS_Environment(temp_cfg)
        self.preloaded_data = temp_env.year_data_dict

        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.repository = []
        self.history = {'iter': [], 'alcc_min': [], 'alcc_avg': [], 'alcc_max': [],
                        'nti_min': [], 'nti_avg': [], 'nti_max': []}

    def _dominates(self, obj_a, obj_b):
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def update_repository(self):
        candidates = self.repository + [copy.deepcopy(p) for p in self.swarm]
        unique_candidates = {}
        for p in candidates:
            objs = tuple(np.round(p.objectives, 4))
            if objs not in unique_candidates:
                unique_candidates[objs] = p

        candidates = list(unique_candidates.values())
        non_dominated = []
        for i, p1 in enumerate(candidates):
            is_dominated = False
            for j, p2 in enumerate(candidates):
                if i != j and self._dominates(p2.objectives, p1.objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(p1)

        max_repo_size = 100
        if len(non_dominated) > max_repo_size:
            non_dominated.sort(key=lambda x: x.objectives[0])
            indices = np.linspace(0, len(non_dominated) - 1, max_repo_size, dtype=int)
            self.repository = [non_dominated[i] for i in indices]
        else:
            self.repository = non_dominated

    def select_leader(self):
        if not self.repository: return self.swarm[0].best_position
        leader = random.choice(self.repository)
        return leader.position

    def optimize(self):
        print(f"Starting MPC-MOPSO (particles={self.num_particles}, iterations={self.max_iter})...")
        print("Note: MPC is slow (4-year average); progress is reported sequentially...")

        _ = jit_calc_Vt(10.0, 0.5, 0.5, 1.0, 0.003, 0.3, 9.81, 0.8, 45.0, 8.0, 1025.0, 0.04)

        eval_args_base = (self.bounds, self.data_dir, self.preloaded_data, self.fast_mode)

        for it in range(self.max_iter):
            start_t = time.time()

            # Sequential particle evaluation (process pool removed).
            results = []
            for i, p in enumerate(self.swarm):
                try:
                    results.append(evaluate_single_particle_mpc(p.position, *eval_args_base))
                except Exception as e:
                    print(f"  [Error] Particle {i} failed: {e}")
                    results.append(np.array([float('inf'), float('inf')]))

            current_alcc_list = []
            current_nti_list = []

            for i, p in enumerate(self.swarm):
                current_objs = results[i]
                p.objectives = current_objs

                if np.isinf(current_objs[0]): continue

                current_alcc_list.append(current_objs[0])
                current_nti_list.append(-current_objs[1])

                if self._dominates(current_objs, p.best_objectives):
                    p.best_position = p.position.copy()
                    p.best_objectives = current_objs.copy()
                elif not self._dominates(p.best_objectives, current_objs):
                    if random.random() < 0.5:
                        p.best_position = p.position.copy()
                        p.best_objectives = current_objs.copy()

            self.update_repository()

            if len(current_alcc_list) > 0:
                alcc_min, alcc_max, alcc_avg = np.min(current_alcc_list), np.max(current_alcc_list), np.mean(
                    current_alcc_list)
                nti_min, nti_max, nti_avg = np.min(current_nti_list), np.max(current_nti_list), np.mean(
                    current_nti_list)
            else:
                alcc_min = alcc_max = alcc_avg = nti_min = nti_max = nti_avg = 0

            self.history['iter'].append(it)
            self.history['alcc_min'].append(alcc_min)
            self.history['alcc_avg'].append(alcc_avg)
            self.history['alcc_max'].append(alcc_max)
            self.history['nti_min'].append(nti_min)
            self.history['nti_avg'].append(nti_avg)
            self.history['nti_max'].append(nti_max)

            iter_time = time.time() - start_t
            print(f"Iter {it + 1} Summary: RepoSize={len(self.repository)} | "
                  f"CostMin={alcc_min:.0f} | NTIMax={nti_max:.0f} | Time={iter_time:.1f}s")

            if it < self.max_iter - 1:
                for p in self.swarm:
                    leader_pos = self.select_leader()
                    r1, r2 = random.random(), random.random()
                    p.velocity = (self.w * p.velocity +
                                  self.c1 * r1 * (p.best_position - p.position) +
                                  self.c2 * r2 * (leader_pos - p.position))
                    p.position += p.velocity
                    for d in range(len(self.bounds)):
                        p.position[d] = np.clip(p.position[d], self.bounds[d][0], self.bounds[d][1])

        return self.repository

# ==============================================================================
# ==============================================================================
if __name__ == "__main__":
    DATA_DIR = _DATA_DIR

    bounds = [
        (0, 60000),  # PV
        (0.0, 120.0),  # Battery
        (1, 24),  # M
        (0.001, 0.003)  # Q0
    ]

    mopso = AUS_MOPSO_Parallel_MPC(
        DATA_DIR, bounds,
        num_particles=50,
        max_iter=100,
        fast_mode=False,
    )

    pareto_front = mopso.optimize()

    def decode_helper(pos):
        return {"p_az": pos[0], "e_max": pos[1], "m": int(round(pos[2])), "q0": pos[3]}

    results = []
    for p in pareto_front:
        cfg = decode_helper(p.position)
        cfg['ALCC'] = p.objectives[0]
        cfg['NTI'] = -p.objectives[1]
        results.append(cfg)

    df_res = pd.DataFrame(results).sort_values(by="ALCC")
    save_path = os.path.join(_RESULT_DIR, "MPC_MOPSO_Pareto_4YearAvg.xlsx")
    df_res.to_excel(save_path, index=False)
    print(f"\nOptimization finished, saved to: {save_path}")

    # ==========================================================================
    # ==========================================================================
    if len(mopso.history['iter']) > 0:
        history = mopso.history
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        ax2 = ax1.twinx()
        l1, = ax1.plot(history['iter'], history['alcc_min'], 'g-', label='Min Cost')
        ax1.plot(history['iter'], history['alcc_avg'], 'g--', alpha=0.5)
        l2, = ax2.plot(history['iter'], history['nti_max'], 'b-', label='Max NTI (MPC-4Yr)')
        ax2.plot(history['iter'], history['nti_avg'], 'b--', alpha=0.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost (CNY)', color='g')
        ax2.set_ylabel('MPC Optimal NTI', color='b')
        ax1.set_title('MPC-MOPSO Convergence (4-Year Avg)')
        ax1.legend([l1, l2], ['Min Cost', 'Max NTI'], loc='center right')
        ax1.grid(True, alpha=0.3)

        axes[1].scatter(df_res['ALCC'], df_res['NTI'], c='purple', edgecolor='k', s=50, label='MPC Pareto')
        axes[1].set_xlabel('Annualized Cost (CNY)')
        axes[1].set_ylabel('Nutrient Transport Index (NTI)')
        axes[1].set_title('Pareto Frontier (MPC 4-Year Avg)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()