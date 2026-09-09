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

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

# ==============================================================================
# ==============================================================================
class AUS_Environment:
    def __init__(self, data_dir: str, preloaded_data=None):
        self.dt = 1
        self.total_time = 2880
        self.eta_c = 0.95
        self.eta_d = 0.95
        self.c_max = 30.0
        self.d_max = 30.0
        self.K = 0.8
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

        self.config_ranges = {
            "p_az": (0, 60000), "e_max": (0.0, 120.0),
            "m": (1, 24), "q0": (0.001, 0.003)
        }

        self.data_dir = data_dir
        self.train_years = [2021, 2022, 2023, 2024]
        self.year_data_dict = {}

        if preloaded_data is not None:
            self.year_data_dict = preloaded_data
        else:
            self.load_all_year_data()

    def load_all_year_data(self):
        for year in self.train_years:
            try:
                xlsx_path = os.path.join(self.data_dir, f"{year}.xlsx")
                if os.path.exists(xlsx_path):
                    df = pd.read_excel(xlsx_path, header=0, index_col=0, engine="openpyxl").reset_index(drop=True)
                    df = df.astype(float)
                    if len(df) > 0 and df.shape[1] >= 3:
                        df.iloc[:, 2] = df.iloc[:, 2] / 100
                    df = df.head(self.total_time).ffill().bfill()
                    if len(df) < self.total_time:
                        last_row = df.iloc[-1]
                        df = pd.concat([df, pd.DataFrame([last_row] * (self.total_time - len(df)))], ignore_index=True)
                    df.columns = ["temperature", "light", "tide_height", "current_speed"]
                    valid_mask = (df.notna().all(axis=1)) & (df["light"] >= 0)
                    df = df[valid_mask].reindex(range(self.total_time), method='ffill')
                    self.year_data_dict[year] = df.values
            except Exception as e:
                print(f"[Warn] Failed to load year {year}: {e}")

    def get_solar_power(self, I_t: float, P_AZ: float) -> float:
        return np.clip((I_t * self.dt * P_AZ * self.K) / 1e6, 0.0, P_AZ / 1000 * self.K)

# ==============================================================================
# ==============================================================================
class DP_Optimizer:
    def __init__(self, env: AUS_Environment, config: dict, num_battery_states=100):
        self.env = env
        self.cfg = config

        self.P_AZ = config['p_az']
        self.E_max = config['e_max']
        self.E_min = 0.05 * self.E_max
        self.M = int(config['m'])
        self.Q0 = config['q0']

        self.p0 = ((self.Q0 * 60 * 1000 + 58.691) / 0.1442) / 1000
        self.num_states = num_battery_states
        self.E_grid = np.linspace(self.E_min, self.E_max, num_battery_states)
        self.actions = np.arange(self.M + 1)

        self.action_power_costs = self.env.b + self.actions * self.p0
        self.action_power_costs[0] = self.env.b

    def calc_Vt_vectorized(self, actions, u_c_t, u_next, tide):
        u_safe = max(abs(u_c_t), 1e-8)
        u_next_safe = max(abs(u_next), 1e-8)
        v_s = self.env.v_s
        g = self.env.g

        Zd = 5.1 * self.Q0 / ((v_s ** 2.4) * u_safe) ** 0.88 * g
        alpha = 0.082 * np.tanh((g * self.Q0 / 10.4) ** (1 / 3) / v_s) ** (3 / 8)
        A = 1.02 * alpha ** (-1) * (g * 1.2 * self.Q0 * 1.25 / 3.14 / 1018) ** (1 / 3)
        delta_z = self.env.d0 / (2.4 * alpha)
        Bd = 1.2 * (Zd + delta_z) * alpha
        vd = A * (Zd + delta_z) ** (-1 / 3)
        v0 = A * delta_z ** (-1 / 3)
        plume_flux_Q0 = 3.14 * self.env.d0 ** 2 * np.sqrt(u_c_t ** 2 + v0 ** 2)
        Qd = 3.14 * Bd ** 2 * np.sqrt(u_c_t ** 2 + vd ** 2)
        rho_d = ((Qd - plume_flux_Q0) * self.env.rho_w + plume_flux_Q0 * (self.env.rho_w + self.env.rho_delta)) / Qd
        vmd = vd * np.sqrt(2 * 3.14) / 6
        jieta = rho_d / (rho_d - self.env.rho_w)
        beta_param = 0.17
        Zm = np.sqrt((Bd / beta_param) ** 2 + 4 / 3 * jieta * Bd * vmd ** 2 / (beta_param * g)) - Bd / beta_param + Zd
        xd = self.env.x_d if u_c_t > 0 else 120 - self.env.x_d

        is_valid = True
        if Zm < self.env.Z_s + tide:
            is_valid = False
        else:
            if Zd > self.env.Z_s + tide:
                delta_term = (self.env.Z_s + tide + delta_z) ** (4 / 3) - delta_z ** (4 / 3)
                xs = (delta_term * 3 / 4 / A) * u_safe
            else:
                term = 1 - 3 * beta_param * g / (4 * jieta * Bd * vmd ** 2) * (self.env.Z_s + tide - Zd) * (
                        self.env.Z_s + tide + 2 * Bd / beta_param - Zd)
                if term < 0: term = 0
                td = ((Zd + delta_z) ** (4 / 3) - delta_z ** (4 / 3)) * 3 / 4 / A
                xd1 = td * u_safe
                xs = jieta * vmd * u_safe / g * (1 - term ** (2 / 3)) + xd1
            if xs >= xd: is_valid = False

        if not is_valid:
            return np.zeros_like(actions, dtype=float)

        xaq = 120 if ((u_next > 0 and u_c_t < 0) or (u_next < 0 and u_c_t > 0)) else xd
        denom = xd - xs
        if abs(denom) < 1e-6: denom = 1e-6
        kata = ((xd - xs) / (u_safe * 3600)) ** 2 * (
                (u_safe * 3600 - xd) / denom + u_safe / u_next_safe * ((xaq - xd) / denom + 0.5) + 0.5
        )
        Vavg = kata * plume_flux_Q0 * self.env.dt * 3600
        return actions * Vavg

    def calc_f2_scalar(self, T_t, I_t):
        I_illu = I_t * 0.168
        f_I = min(I_illu / self.env.I_s, 2.0)
        T_x = self.env.T_min if T_t <= self.env.T_opt else self.env.T_max
        if abs(T_x - self.env.T_opt) < 1e-6:
            f_T = 1.0
        else:
            f_T = np.exp(1 - f_I - 2.3 * ((T_t - self.env.T_opt) / (T_x - self.env.T_opt)) ** 2)
        return max(f_I * f_T, 0.0)

    def solve(self, env_data):
        T_steps = min(len(env_data), self.env.total_time)
        V = np.zeros(self.num_states)
        Policy = np.zeros((T_steps, self.num_states), dtype=int)

        # Backward Induction
        for t in range(T_steps - 1, -1, -1):
            row = env_data[t]
            T_t, I_t, Z_t, u_t = row[0], row[1], row[2], row[3]
            u_next = env_data[t + 1][3] if t < T_steps - 1 else u_t

            g_t = self.env.get_solar_power(I_t, self.P_AZ)
            Vt_vec = self.calc_Vt_vectorized(self.actions, u_t, u_next, Z_t)
            f2_t = self.calc_f2_scalar(T_t, I_t)
            current_rewards = self.env.beta * Vt_vec * f2_t

            E_curr_col = self.E_grid.reshape(-1, 1)
            energy_surplus = g_t - self.action_power_costs.reshape(1, -1)

            # Constraints
            max_discharge_capacity = (E_curr_col - self.E_min) * self.env.eta_d / self.env.dt + 1e-5
            discharge_needed = -energy_surplus
            violation_mask = (energy_surplus < 0) & (discharge_needed > max_discharge_capacity)

            mask_charge = energy_surplus > 0
            c_t = np.minimum(energy_surplus, self.env.c_max)
            d_t = np.minimum(-energy_surplus, self.env.d_max)
            delta_E = (mask_charge * (c_t * self.env.eta_c) - (~mask_charge) * (d_t / self.env.eta_d)) * self.env.dt
            E_next_matrix = np.clip(E_curr_col + delta_E, self.E_min, self.E_max)

            V_next_flat = np.interp(E_next_matrix.ravel(), self.E_grid, V)
            V_next_matrix = V_next_flat.reshape(self.num_states, len(self.actions))

            Q_matrix = current_rewards.reshape(1, -1) + V_next_matrix
            Q_matrix[violation_mask] = -1e9

            best_actions_idx = np.argmax(Q_matrix, axis=1)
            V = np.max(Q_matrix, axis=1)
            Policy[t] = best_actions_idx

        # Forward Simulation for Real NTI
        total_real_nti = 0.0
        curr_E = (self.E_max + self.E_min) / 2

        for t in range(T_steps):
            state_idx = (np.abs(self.E_grid - curr_E)).argmin()
            best_act_idx = Policy[t, state_idx]
            best_action = self.actions[best_act_idx]

            row = env_data[t]
            T_t, I_t, Z_t, u_t = row[0], row[1], row[2], row[3]
            u_next = env_data[t + 1][3] if t < T_steps - 1 else u_t

            Vt = self.calc_Vt_vectorized(np.array([best_action]), u_t, u_next, Z_t)[0]
            f2_t = self.calc_f2_scalar(T_t, I_t)
            total_real_nti += (Vt * f2_t)

            g_t = self.env.get_solar_power(I_t, self.P_AZ)
            power_cost = self.env.b + best_action * self.p0
            surplus = g_t - power_cost

            if surplus > 0:
                c_t = min(surplus, self.env.c_max)
                curr_E += c_t * self.env.eta_c * self.env.dt
            else:
                d_t = min(-surplus, self.env.d_max)
                curr_E -= d_t / self.env.eta_d * self.env.dt
            curr_E = np.clip(curr_E, self.E_min, self.E_max)

        return total_real_nti

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
def evaluate_single_particle_dp(position, bounds, env_data_dir, preloaded_data, fast_mode=False):
    def decode_pos(pos, b):
        p_az = np.clip(pos[0], b[0][0], b[0][1])
        e_max = np.clip(pos[1], b[1][0], b[1][1])
        m = int(np.clip(round(pos[2]), b[2][0], b[2][1]))
        q0 = np.clip(pos[3], b[3][0], b[3][1])
        return {"p_az": p_az, "e_max": e_max, "m": m, "q0": q0}

    config = decode_pos(position, bounds)
    cost_calc = CostCalculator()
    alcc = cost_calc.calculate_alcc(config)

    env = AUS_Environment(env_data_dir, preloaded_data=preloaded_data)
    dp_solver = DP_Optimizer(env, config, num_battery_states=100)

    total_nti = 0
    years = env.train_years if not fast_mode else [2024]

    for year in years:
        if year in env.year_data_dict:
            env_data = env.year_data_dict[year]
            year_nti = dp_solver.solve(env_data)
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

class AUS_MOPSO_Parallel_DP:
    def __init__(self, data_dir, bounds, num_particles=50, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, fast_mode=False):
        self.data_dir = data_dir
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.fast_mode = fast_mode

        print("Preloading environmental data...")
        temp_env = AUS_Environment(data_dir)
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
        print(f"Starting DP-MOPSO (particles={self.num_particles}, iterations={self.max_iter})...")
        eval_args_base = (self.bounds, self.data_dir, self.preloaded_data, self.fast_mode)

        for it in range(self.max_iter):
            start_t = time.time()
            # Parallel call with error handling
            try:
                results = [evaluate_single_particle_dp(p.position, *eval_args_base) for p in self.swarm]
            except Exception as e:
                print(f"Evaluation error: {e}")
                print("Reduce memory footprint and retry.")
                return self.repository

            current_alcc_list = []
            current_nti_list = []

            for i, p in enumerate(self.swarm):
                current_objs = results[i]
                p.objectives = current_objs
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

            alcc_min, alcc_max, alcc_avg = np.min(current_alcc_list), np.max(current_alcc_list), np.mean(
                current_alcc_list)
            nti_min, nti_max, nti_avg = np.min(current_nti_list), np.max(current_nti_list), np.mean(current_nti_list)

            self.history['iter'].append(it)
            self.history['alcc_min'].append(alcc_min)
            self.history['alcc_avg'].append(alcc_avg)
            self.history['alcc_max'].append(alcc_max)
            self.history['nti_min'].append(nti_min)
            self.history['nti_avg'].append(nti_avg)
            self.history['nti_max'].append(nti_max)

            iter_time = time.time() - start_t
            print(f"{it + 1:<5} | {len(self.repository):<4} | "
                  f"{alcc_min:.0f} / {alcc_avg:.0f} | "
                  f"{nti_max:.0f} / {nti_avg:.0f} | {iter_time:.1f}s")

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

    mopso = AUS_MOPSO_Parallel_DP(
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
    save_path = os.path.join(_RESULT_DIR, "DP_MOPSO_Pareto_Baseline_Corrected.xlsx")
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
        l2, = ax2.plot(history['iter'], history['nti_max'], 'b-', label='Max NTI (Oracle)')
        ax2.plot(history['iter'], history['nti_avg'], 'b--', alpha=0.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost (CNY)', color='g')
        ax2.set_ylabel('DP Optimal NTI', color='b')
        ax1.set_title('DP-MOPSO Convergence (Theoretical Bound)')
        ax1.legend([l1, l2], ['Min Cost', 'Max NTI'], loc='center right')
        ax1.grid(True, alpha=0.3)

        axes[1].scatter(df_res['ALCC'], df_res['NTI'], c='purple', edgecolor='k', s=50, label='DP Pareto (Bound)')
        axes[1].set_xlabel('Annualized Cost (CNY)')
        axes[1].set_ylabel('Nutrient Transport Index (NTI)')
        axes[1].set_title('Theoretical Pareto Frontier (Oracle)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()