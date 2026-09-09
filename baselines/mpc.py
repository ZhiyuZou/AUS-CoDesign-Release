import numpy as np
import pandas as pd
import time
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numba import jit

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

    # Zd
    term1 = (v_s ** 2.4) * u_safe
    Zd = 5.1 * Q0 / (term1 ** 0.88) * g

    # Alpha
    term_alpha = (g * Q0 / 10.4) ** (1 / 3) / v_s
    alpha = 0.082 * np.tanh(term_alpha) ** (3 / 8)

    # A
    term_A = (g * 1.2 * Q0 * 1.25 / 3.14 / 1018) ** (1 / 3)
    A = 1.02 * (alpha ** (-1)) * term_A

    delta_z = d0 / (2.4 * alpha)
    Bd = 1.2 * (Zd + delta_z) * alpha
    vd = A * ((Zd + delta_z) ** (-1 / 3))

    # v0, plume_flux
    v0 = A * (delta_z ** (-1 / 3))
    plume_flux_Q0 = 3.14 * (d0 ** 2) * np.sqrt(u_c_t ** 2 + v0 ** 2)

    Qd = 3.14 * (Bd ** 2) * np.sqrt(u_c_t ** 2 + vd ** 2)
    rho_d = ((Qd - plume_flux_Q0) * rho_w + plume_flux_Q0 * (rho_w + rho_delta)) / Qd

    vmd = vd * np.sqrt(2 * 3.14) / 6
    jieta = rho_d / (rho_d - rho_w)
    beta_param = 0.17

    # Zm
    term_sqrt = (Bd / beta_param) ** 2 + (4 / 3 * jieta * Bd * (vmd ** 2) / (beta_param * g))
    Zm = np.sqrt(term_sqrt) - (Bd / beta_param) + Zd

    xd_val = x_d if u_c_t > 0 else 120 - x_d

    if Zm < Z_s + tide: return 0.0

    # xs
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

    # xaq
    cond = ((u_next > 0) and (u_c_t < 0)) or ((u_next < 0) and (u_c_t > 0))
    xaq = 120.0 if cond else xd_val

    denom = xd_val - xs
    if abs(denom) < 1e-6: denom = 1e-6

    term_kata = ((xd_val - xs) / (u_safe * 3600)) ** 2
    inner_kata = ((u_safe * 3600 - xd_val) / denom) + (u_safe / u_next_safe) * ((xaq - xd_val) / denom + 0.5) + 0.5
    kata = term_kata * inner_kata

    Vavg = kata * plume_flux_Q0 * 1.0 * 3600  # dt=1 hardcoded or passed
    return m_t * Vavg

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
    # dt=1
    g_t = (I_t * 1.0 * P_AZ * K) / 1e6
    limit = P_AZ / 1000 * K
    if g_t > limit: g_t = limit
    if g_t < 0: g_t = 0
    return g_t

@jit(nopython=True, cache=True)
def jit_predict_cost(u_seq, init_E, weather_data,
                     params_phys, params_cfg):
    
    Q0, v_s, g, d0, x_d, Z_s, rho_w, rho_delta, I_s, T_opt, T_min, T_max, K, eta_c, eta_d, c_max, d_max, b, dt, beta = params_phys
    P_AZ, E_max, E_min, p0, M = params_cfg

    cost = 0.0
    curr_E = init_E

    H = len(u_seq)

    for k in range(H):
        m_k = u_seq[k]

        T_t = weather_data[k, 0]
        I_t = weather_data[k, 1]
        Z_t = weather_data[k, 2]
        u_t = weather_data[k, 3]

        if k < H - 1:
            u_next = weather_data[k + 1, 3]
        else:
            u_next = u_t

        Vt = jit_calc_Vt(m_k, u_t, u_next, Z_t, Q0, v_s, g, d0, x_d, Z_s, rho_w, rho_delta)
        f2_t = jit_calc_f2(T_t, I_t, I_s, T_opt, T_min, T_max)

        g_t = jit_get_solar(I_t, P_AZ, K)
        q_t = m_k * p0
        surplus = g_t - (b + q_t)

        c_t = 0.0
        d_t = 0.0

        if surplus > 0:
            c_t = surplus
            if c_t > c_max: c_t = c_max
        else:
            d_t = -surplus
            if d_t > d_max: d_t = d_max

        next_E = curr_E + (eta_c * c_t - d_t / eta_d) * dt

        penalty_E = 0.0
        if next_E < E_min:
            penalty_E += 500.0 * (E_min - next_E) ** 2
        if next_E > E_max:
            penalty_E += 500.0 * (next_E - E_max) ** 2

        penalty_shutdown = 0.0
        if curr_E <= E_min + 1.0 and m_k > 0.1:
            penalty_shutdown = 1000.0 * m_k

        penalty_ineffective = 0.0
        if m_k > 0.5:
            if Vt < 1e-6 or f2_t < 1e-3:
                penalty_ineffective = 1.0 * m_k

        raw_nti = f2_t * Vt
        step_reward = beta * raw_nti

        step_cost = -step_reward + penalty_ineffective + penalty_E + penalty_shutdown
        cost += step_cost

        # Clip E
        if next_E < E_min: next_E = E_min
        if next_E > E_max: next_E = E_max
        curr_E = next_E

    cost -= 0.5 * curr_E
    return cost

# ==============================================================================
# ==============================================================================
class AUS_Config:
    def __init__(self, data_dir, config_dict):
        self.data_dir = data_dir
        self.dt = 1
        self.total_time = 2880

        self.eta_c = 0.95
        self.eta_d = 0.95
        self.c_max = 30.0
        self.d_max = 30.0
        self.b = 0.2

        self.Z_s = 8.0
        self.x_d = 45.0
        self.rho_delta = 0.04
        self.rho_w = 1025.19
        self.H0 = 10.4
        self.v_s = 0.3
        self.g = 9.81
        self.I_s = 180.0 * 0.217
        self.T_opt = 10.0
        self.T_min = 0.5
        self.T_max = 20.0
        self.beta = 0.001
        self.d0 = 0.8
        self.K = 0.8

        self.P_AZ = config_dict["p_az"]
        self.E_max = config_dict["e_max"]
        self.E_min = 0.05 * self.E_max
        self.M = config_dict["m"]
        self.Q0 = config_dict["q0"]
        self.p0 = ((self.Q0 * 60 * 1000 + 58.691) / 0.1442) / 1000

        self.phys_params = np.array([
            self.Q0, self.v_s, self.g, self.d0, self.x_d, self.Z_s,
            self.rho_w, self.rho_delta, self.I_s, self.T_opt, self.T_min, self.T_max,
            self.K, self.eta_c, self.eta_d, self.c_max, self.d_max, self.b, self.dt, self.beta
        ], dtype=np.float64)

        self.cfg_params = np.array([
            self.P_AZ, self.E_max, self.E_min, self.p0, self.M
        ], dtype=np.float64)

    def get_solar_power(self, I_t):
        return jit_get_solar(I_t, self.P_AZ, self.K)

    def calc_Vt(self, m_t, u_c, u_n, tide):
        return jit_calc_Vt(m_t, u_c, u_n, tide, self.Q0, self.v_s, self.g, self.d0,
                           self.x_d, self.Z_s, self.rho_w, self.rho_delta)

    def calc_f2(self, T_t, I_t):
        return jit_calc_f2(T_t, I_t, self.I_s, self.T_opt, self.T_min, self.T_max)

# ==============================================================================
# ==============================================================================
class MPC_Controller:
    def __init__(self, cfg: AUS_Config, horizon=24):
        self.cfg = cfg
        self.H = horizon
        self.prev_solution = None

    def solve(self, current_E, env_future_data):
        H_actual = len(env_future_data) - 1
        if H_actual == 0: return 0.0

        weather_arr = np.array(env_future_data, dtype=np.float64)

        # Warm Start
        if self.prev_solution is not None and len(self.prev_solution) == H_actual:
            x0 = np.roll(self.prev_solution, -1)
            x0[-1] = x0[-2]
        else:
            x0 = np.zeros(H_actual)
            for k in range(H_actual):
                if weather_arr[k, 1] > 100:
                    x0[k] = self.cfg.M / 2

        bounds = [(0, self.cfg.M) for _ in range(H_actual)]

        res = minimize(
            fun=jit_predict_cost,
            x0=x0,
            args=(current_E, weather_arr, self.cfg.phys_params, self.cfg.cfg_params),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-4, 'disp': False, 'maxiter': 50}
        )

        if res.success:
            self.prev_solution = res.x
        else:
            self.prev_solution = None

        return res.x[0]

# ==============================================================================
# ==============================================================================
class AUS_Environment:
    def __init__(self, cfg: AUS_Config):
        self.cfg = cfg
        self.year_data_dict = {}
        self.load_all_year_data()
        self.current_step = 0
        self.current_E = 0.0
        self.env_data = None

    def load_all_year_data(self):
        train_years = [2021, 2022, 2023, 2024]
        for year in train_years:
            file_path = f"{self.cfg.data_dir}/{year}.xlsx"
            try:
                df = pd.read_excel(file_path, header=0, index_col=0, engine="openpyxl").reset_index(drop=True)
                df = df.astype(float)
                if len(df) > 0 and df.shape[1] >= 3:
                    df.iloc[:, 2] = df.iloc[:, 2] / 100
                if len(df) < self.cfg.total_time:
                    last_row = df.iloc[-1]
                    df = pd.concat([df, pd.DataFrame([last_row] * (self.cfg.total_time - len(df)))], ignore_index=True)
                df.columns = ["temperature", "light", "tide_height", "current_speed"]
                df = df.head(self.cfg.total_time).ffill().bfill()
                self.year_data_dict[year] = df.values
            except Exception:
                pass

    def reset(self, year=2021):
        if year not in self.year_data_dict:
            year = list(self.year_data_dict.keys())[0]
        self.env_data = self.year_data_dict[year]
        self.current_step = 0
        self.current_E = np.random.uniform(self.cfg.E_min, self.cfg.E_max)
        return self._get_state()

    def _get_state(self):
        safe_idx = min(self.current_step, len(self.env_data) - 1)
        row = self.env_data[safe_idx]
        return {"E": self.current_E, "T": row[0], "I": row[1], "Z": row[2], "u": row[3]}

    def step(self, action_m):
        actual_m = max(0, min(int(action_m), self.cfg.M))

        row = self.env_data[self.current_step]
        T_t, I_t, Z_t, u_t = row[0], row[1], row[2], row[3]

        self.current_step += 1
        done = self.current_step >= self.cfg.total_time

        safe_next_idx = min(self.current_step, len(self.env_data) - 1)
        u_next = self.env_data[safe_next_idx, 3]

        g_t = self.cfg.get_solar_power(I_t)
        q_t = actual_m * self.cfg.p0
        total_load = self.cfg.b + q_t
        energy_surplus = g_t - total_load

        c_t = 0.0
        flag = 1
        E_t = self.current_E

        if energy_surplus > 0:
            c_t = min(energy_surplus, self.cfg.c_max, (self.cfg.E_max - E_t) / (self.cfg.eta_c * self.cfg.dt))
            E_next = E_t + self.cfg.eta_c * c_t * self.cfg.dt
        elif -energy_surplus < (E_t - self.cfg.E_min) * self.cfg.eta_d / self.cfg.dt:
            E_next = E_t + energy_surplus * self.cfg.dt / self.cfg.eta_d
        else:
            actual_m = 0
            flag = 0
            energy_surplus_new = g_t - self.cfg.b
            if energy_surplus_new > 0:
                c_t = min(energy_surplus_new, self.cfg.c_max, (self.cfg.E_max - E_t) / (self.cfg.eta_c * self.cfg.dt))
                E_next = E_t + self.cfg.eta_c * c_t * self.cfg.dt
            else:
                E_next = E_t + energy_surplus_new * self.cfg.dt / self.cfg.eta_d

        self.current_E = np.clip(E_next, self.cfg.E_min, self.cfg.E_max)

        Vt = self.cfg.calc_Vt(actual_m, u_t, u_next, Z_t)
        f2_t = self.cfg.calc_f2(T_t, I_t)

        raw_nti_metric = f2_t * Vt
        reward_val = 0.0
        if flag == 1:
            if actual_m == 0 or (Vt >= 1e-6 and f2_t >= 1e-2):
                reward_val += self.cfg.beta * raw_nti_metric
            else:
                reward_val -= 1.0

        energy_waste = max(energy_surplus - c_t, 0.0)

        info = {
            "Raw_NTI": raw_nti_metric,
            "Reward": reward_val,
            "energy_waste": energy_waste,
            "actual_m": actual_m,
            "E": self.current_E
        }

        return self._get_state(), reward_val, done, info

def run_mpc_simulation(data_dir, fixed_config):
    print("=== Starting Numba-accelerated MPC simulation ===")
    cfg = AUS_Config(data_dir, fixed_config)

    print("Warming up the JIT compiler...")
    _ = jit_calc_Vt(10.0, 0.5, 0.5, 1.0, 0.003, 0.3, 9.81, 0.8, 45.0, 8.0, 1025.0, 0.04)

    env = AUS_Environment(cfg)
    if not env.year_data_dict:
        print("Error: data not loaded")
        return

    mpc = MPC_Controller(cfg, horizon=24)
    state = env.reset(year=2024)

    total_reward = 0
    total_raw_nti = 0

    start_time = time.time()

    for t in range(cfg.total_time):
        end_idx = min(t + mpc.H + 1, cfg.total_time)
        future_data = env.env_data[t: end_idx]

        if len(future_data) < 2:
            action_raw = 0
        else:
            action_raw = mpc.solve(state["E"], future_data)

        action_m = int(round(action_raw))
        next_state, reward, done, info = env.step(action_m)

        total_reward += reward
        total_raw_nti += info["Raw_NTI"]
        state = next_state

        if t % 500 == 0:
            print(f"Step {t} | Act: {action_m} | Rew: {reward:.2f}")

    print(f"\nSimulation finished. Elapsed: {(time.time() - start_time):.1f}s")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Raw NTI: {total_raw_nti:.0f}")

if __name__ == "__main__":
    DATA_DIR = _DATA_DIR
    TARGET_CONFIG = {"p_az": 48000, "e_max": 86.4, "m": 16, "q0": 0.0016667}
    run_mpc_simulation(DATA_DIR, TARGET_CONFIG)