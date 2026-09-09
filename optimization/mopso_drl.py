import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
import copy
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

# ==============================================================================
# ==============================================================================
class Vectorized_AUS_Environment:
    def __init__(self, data_dir: str, num_envs: int, preloaded_data=None):
        self.num_envs = num_envs

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

        self.fixed_action_dim = 24
        self.dynamic_dim = 6
        self.static_dim = 4
        self.state_dim = self.dynamic_dim + self.static_dim

        self.data_dir = data_dir
        self.train_years = [2021, 2022, 2023, 2024]
        self.year_data_dict = {}

        if preloaded_data is not None:
            self.year_data_dict = preloaded_data
        else:
            self.load_all_year_data()

        self.calculate_statistics()

        self.current_states = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        self.configs = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.p0_vec = np.zeros(self.num_envs, dtype=np.float32)
        self.E_min_vec = np.zeros(self.num_envs, dtype=np.float32)

        self.current_step = 0
        self.env_data = None

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

    def calculate_statistics(self):
        all_data_list = list(self.year_data_dict.values())
        if not all_data_list:
            self.state_mean = np.zeros(self.state_dim)
            self.state_std = np.ones(self.state_dim)
            return

        all_data = np.concatenate(all_data_list)
        e_mean_est = (self.config_ranges["e_max"][1] + self.config_ranges["e_max"][0]) / 4

        dyn_mean = np.array([e_mean_est, np.mean(all_data[:, 0]), np.mean(all_data[:, 1]), np.mean(all_data[:, 2]),
                             np.mean(all_data[:, 3]), self.total_time / 2], dtype=np.float32)
        dyn_std = np.array(
            [20.0, np.std(all_data[:, 0]), np.std(all_data[:, 1]), np.std(all_data[:, 2]), np.std(all_data[:, 3]),
             self.total_time / 4], dtype=np.float32)
        stat_mean = np.array([np.mean(self.config_ranges["p_az"]), np.mean(self.config_ranges["e_max"]),
                              np.mean(self.config_ranges["m"]), np.mean(self.config_ranges["q0"])], dtype=np.float32)
        stat_std = np.array([(self.config_ranges["p_az"][1] - self.config_ranges["p_az"][0]) / 10,
                             (self.config_ranges["e_max"][1] - self.config_ranges["e_max"][0]) / 10,
                             (self.config_ranges["m"][1] - self.config_ranges["m"][0]) / 10,
                             (self.config_ranges["q0"][1] - self.config_ranges["q0"][0]) / 10], dtype=np.float32)

        self.state_mean = np.concatenate([dyn_mean, stat_mean])
        self.state_std = np.concatenate([dyn_std, stat_std])
        self.state_std[self.state_std < 1e-6] = 1e-6

    def reset(self, configs: np.ndarray, year: int):
        
        self.current_step = 0
        self.env_data = self.year_data_dict[year]

        self.configs = configs.astype(np.float32)
        self.P_AZ_vec = self.configs[:, 0]
        self.E_max_vec = self.configs[:, 1]
        self.M_vec = self.configs[:, 2].astype(int)
        self.Q0_vec = self.configs[:, 3]

        self.E_min_vec = 0.05 * self.E_max_vec
        self.p0_vec = ((self.Q0_vec * 60 * 1000 + 58.691) / 0.1442) / 1000

        init_E = np.random.uniform(self.E_min_vec, self.E_max_vec)

        init_temp = self.env_data[0, 0]
        init_light = self.env_data[0, 1]
        init_tide = self.env_data[0, 2]
        init_current = self.env_data[0, 3]

        self.current_states[:, 0] = init_E
        self.current_states[:, 1] = init_temp
        self.current_states[:, 2] = init_light
        self.current_states[:, 3] = init_tide
        self.current_states[:, 4] = init_current
        self.current_states[:, 5] = 0.0
        self.current_states[:, 6:] = self.configs

        masks = self._get_action_mask_vectorized(init_E, init_light)

        # ==========================================
        # ==========================================
        self.current_states = self.normalize_state(self.current_states)

        return self.current_states, masks

    def _get_action_mask_vectorized(self, E_t_vec, I_t):
        
        solar_power_vec = np.clip((I_t * self.dt * self.P_AZ_vec * self.K) / 1e6, 0.0, self.P_AZ_vec / 1000 * self.K)
        max_discharge_vec = np.minimum(self.d_max, (np.maximum(0, E_t_vec - self.E_min_vec) * self.eta_d) / self.dt)
        total_power = solar_power_vec + max_discharge_vec  # (N,)

        # actions 0..23
        actions = np.arange(self.fixed_action_dim)  # (24,)
        ratios = actions / (self.fixed_action_dim - 1)  # (24,)

        # M_vec: (N,) -> (N, 1) * (1, 24) -> (N, 24) actual_m_matrix
        actual_m_matrix = np.round(self.M_vec[:, None] * ratios[None, :])

        # p0_vec: (N,) -> (N, 1)
        power_req_matrix = self.b + (actual_m_matrix * self.p0_vec[:, None])

        # (N, 1) >= (N, 24) -> (N, 24) boolean
        masks = (total_power[:, None] >= power_req_matrix).astype(np.float32)

        no_valid = (masks.sum(axis=1) == 0)
        masks[no_valid, 0] = 1.0

        return masks

    def calc_Vt_vectorized(self, m_t_vec, u_c_t, u_next, tide):
        

        N = self.num_envs
        u_safe = max(abs(u_c_t), 1e-5)
        u_next_safe = max(abs(u_next), 1e-5)

        # Q0_vec: (N,)
        # Zd: (N,)
        term1 = (self.v_s ** 2.4) * u_safe
        Zd = 5.1 * self.Q0_vec / (term1 ** 0.88) * self.g

        # alpha: (N,)
        term_alpha = (self.g * self.Q0_vec / 10.4) ** (1 / 3) / self.v_s
        alpha = 0.082 * np.tanh(term_alpha) ** (3 / 8)

        # A: (N,)
        term_A = (self.g * 1.2 * self.Q0_vec * 1.25 / 3.14 / 1018) ** (1 / 3)
        A = 1.02 * (alpha ** (-1)) * term_A

        delta_z = self.d0 / (2.4 * alpha)

        # td: (N,)
        term_z_pow = (Zd + delta_z) ** (4 / 3)
        delta_z_pow = delta_z ** (4 / 3)
        td = (term_z_pow - delta_z_pow) * 3 / 4 / A

        xd1 = td * u_safe
        Bd = 1.2 * (Zd + delta_z) * alpha
        vd = A * ((Zd + delta_z) ** (-1 / 3))

        # Qd, Q0_calc, rho_d
        Qd = 3.14 * (Bd ** 2) * np.sqrt(u_c_t ** 2 + vd ** 2)
        v0 = A * (delta_z ** (-1 / 3))
        Q0_calc = 3.14 * (self.d0 ** 2) * np.sqrt(u_c_t ** 2 + v0 ** 2)

        rho_d = ((Qd - Q0_calc) * self.rho_w + Q0_calc * (self.rho_w + self.rho_delta)) / Qd

        vmd = vd * np.sqrt(2 * 3.14) / 6
        jieta = rho_d / (rho_d - self.rho_w)
        beta_const = 0.17

        # Zm: (N,)
        term_sqrt = (Bd / beta_const) ** 2 + (4 / 3 * jieta * Bd * (vmd ** 2) / (beta_const * self.g))
        Zm = np.sqrt(term_sqrt) - (Bd / beta_const) + Zd

        xd = self.x_d if u_c_t > 0 else 120 - self.x_d

        # Logic Masks
        # Condition 1: Zm < Z_s + tide
        cond1 = Zm < (self.Z_s + tide)

        # Condition 2: Zd > Z_s + tide -> calc xs type A
        cond2 = Zd > (self.Z_s + tide)

        # xs calculation
        xs = np.zeros(N, dtype=np.float32)

        # Case A (cond2 True)
        term_zs_pow = (self.Z_s + tide + delta_z) ** (4 / 3)
        xs[cond2] = ((term_zs_pow - delta_z_pow)[cond2] * 3 / 4 / A[cond2]) * u_safe

        # Case B (cond2 False)
        term_B = 1 - 3 * beta_const * self.g / (4 * jieta * Bd * vmd ** 2) * (self.Z_s + tide - Zd) * (
                    self.Z_s + tide + 2 * Bd / beta_const - Zd)
        term_B = np.maximum(term_B, 0)  # clip
        xs[~cond2] = (jieta * vmd * u_safe / self.g * (1 - term_B ** (2 / 3)))[~cond2] + xd1[~cond2]

        # Condition 3: xs >= xd
        cond3 = xs >= xd

        # xaq calculation
        cond_curr = ((u_next > 0) & (u_c_t < 0)) | ((u_next < 0) & (u_c_t > 0))
        xaq = 120.0 if cond_curr else xd

        denom = xd - xs
        denom[np.abs(denom) < 1e-6] = 1e-6  # avoid div 0

        term_kata = ((xd - xs) / (u_safe * 3600)) ** 2
        inner_kata = ((u_safe * 3600 - xd) / denom) + (u_safe / u_next_safe) * ((xaq - xd) / denom + 0.5) + 0.5
        kata = term_kata * inner_kata

        Vavg = kata * Q0_calc * self.dt * 3600

        Vt = m_t_vec * Vavg

        # Apply failure conditions
        Vt[cond1] = 0.0
        Vt[cond3] = 0.0
        Vt[m_t_vec == 0] = 0.0  # base condition

        # Handle exceptions (NaNs)
        Vt = np.nan_to_num(Vt, nan=0.0)

        return Vt

    def step(self, actions_idx: np.ndarray):
        
        N = self.num_envs
        # (N,)
        ratios = actions_idx / (self.fixed_action_dim - 1)
        actual_compressors = np.round(ratios * self.M_vec).astype(np.float32)

        E_t = self.current_states[:, 0] * self.state_std[0] + self.state_mean[0]

        idx = min(self.current_step, self.total_time - 1)
        next_idx = min(self.current_step + 1, self.total_time - 1)

        T_t, I_t, Z_t, u_t = self.env_data[idx]
        u_next = self.env_data[next_idx, 3]

        solar_power = np.clip((I_t * self.dt * self.P_AZ_vec * self.K) / 1e6, 0.0, self.P_AZ_vec / 1000 * self.K)
        q_t = actual_compressors * self.p0_vec
        total_load = self.b + q_t
        energy_surplus = solar_power - total_load

        E_next = np.zeros(N, dtype=np.float32)
        flag = np.ones(N, dtype=np.float32)  # 1=Valid, 0=Invalid

        # Case 1: Surplus > 0 (Charge)
        mask_charge = energy_surplus > 0
        c_t = np.minimum(energy_surplus, self.c_max)
        c_t = np.minimum(c_t, (self.E_max_vec - E_t) / (self.eta_c * self.dt))
        E_next[mask_charge] = E_t[mask_charge] + self.eta_c * c_t[mask_charge] * self.dt

        # Case 2: Surplus <= 0 (Discharge)
        # Check if discharge is possible
        max_d = (E_t - self.E_min_vec) * self.eta_d / self.dt
        mask_discharge_ok = (~mask_charge) & (-energy_surplus <= max_d)
        E_next[mask_discharge_ok] = E_t[mask_discharge_ok] + energy_surplus[mask_discharge_ok] * self.dt / self.eta_d

        # Case 3: Impossible (Blackout/Violation)
        mask_fail = (~mask_charge) & (~mask_discharge_ok)
        E_next[mask_fail] = E_t[mask_fail]
        actual_compressors[mask_fail] = 0  # Force shutdown calculation for NTI
        flag[mask_fail] = 0.0

        # Clip E_next
        E_next = np.clip(E_next, self.E_min_vec, self.E_max_vec)

        Vt = self.calc_Vt_vectorized(actual_compressors, u_t, u_next, Z_t)

        # f2 (Scalar, shared across particles)
        I_illu = I_t * 0.168
        f_I = min(I_illu / self.I_s, 2.0)
        T_x = self.T_min if T_t <= self.T_opt else self.T_max
        if abs(T_x - self.T_opt) < 1e-6:
            f_T = 1.0
        else:
            f_T = np.exp(1 - f_I - 2.3 * ((T_t - self.T_opt) / (T_x - self.T_opt)) ** 2)
        f2_t = max(f_I * f_T, 0.0)

        self.current_step += 1
        done = self.current_step >= self.total_time

        # Update dynamic states
        # Normalized Update
        E_next_norm = (E_next - self.state_mean[0]) / self.state_std[0]

        # Update Matrix
        self.current_states[:, 0] = E_next_norm

        next_env_data = self.env_data[next_idx]  # (4,)
        # Batch update env states (all rows same)
        self.current_states[:, 1] = (next_env_data[0] - self.state_mean[1]) / self.state_std[1]
        self.current_states[:, 2] = (next_env_data[1] - self.state_mean[2]) / self.state_std[2]
        self.current_states[:, 3] = (next_env_data[2] - self.state_mean[3]) / self.state_std[3]
        self.current_states[:, 4] = (next_env_data[3] - self.state_mean[4]) / self.state_std[4]
        self.current_states[:, 5] = (self.current_step - self.state_mean[5]) / self.state_std[5]

        # 6. Next Mask
        next_I = next_env_data[1] if not done else 0.0
        next_masks = self._get_action_mask_vectorized(E_next, next_I)

        infos = {
            "Vt": Vt,
            "f2_t": f2_t,
            "flag": flag,
            "action_mask": next_masks
        }

        return self.current_states, None, done, infos

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

# ==============================================================================
# ==============================================================================
class Optimized_Interaction_Network(nn.Module):
    def __init__(self, dynamic_dim, static_dim, action_dim, num_quantiles=200):
        super().__init__()
        self.action_dim, self.num_quantiles = action_dim, num_quantiles
        self.dynamic_net = nn.Sequential(nn.Linear(dynamic_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.static_net = nn.Sequential(nn.Linear(static_dim, 256), nn.ReLU(), nn.Linear(256, 1024))
        self.fusion_net = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, num_quantiles))
        self.advantage_stream = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),
                                              nn.Linear(512, action_dim * num_quantiles))

    def forward(self, x):
        dyn_dim = self.dynamic_net[0].in_features
        dyn_out = self.dynamic_net(x[:, :dyn_dim])
        stat_raw = self.static_net(x[:, dyn_dim:])
        scale, shift = torch.chunk(stat_raw, 2, dim=1)
        features = self.fusion_net((dyn_out * torch.sigmoid(scale)) + shift)
        v = self.value_stream(features).unsqueeze(1)
        a = self.advantage_stream(features).view(-1, self.action_dim, self.num_quantiles)
        return v + a - a.mean(dim=1, keepdim=True)

# ==============================================================================
# ==============================================================================
class BatchAgentEvaluator:
    def __init__(self, model_class, model_state_dict, dynamic_dim, static_dim, action_dim, device='cpu'):
        self.device = torch.device(device)
        self.net = model_class(dynamic_dim, static_dim, action_dim).to(self.device)
        self.net.load_state_dict(model_state_dict)
        self.net.eval()

    def batch_select_action(self, states, masks):
        
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Forward pass (Batch)
            q_dist = self.net(states_t)  # (N, Action, Quantiles)
            q_values = q_dist.mean(dim=-1)  # (N, Action)

            # Apply Mask (Batch)
            q_values[masks_t == 0] = -float('inf')

            # Argmax
            actions = q_values.argmax(dim=1)  # (N,)

        return actions.cpu().numpy()

# ==============================================================================
# ==============================================================================
class VectorizedCostCalculator:
    def __init__(self):
        self.C_EGS_per_kw = 95.9
        self.C_ESS_per_kwh = 49.2
        self.C_Other = 140.0
        self.C_Compressor_base = 116.8
        self.C_Nozzle = 217.0
        self.C_Pipe = 70.1

    def calculate_alcc_batch(self, configs):
        """
        configs: (N, 4) -> [p_az, e_max, m, q0]
        """
        p_az_kw = configs[:, 0] / 1000.0
        e_max = configs[:, 1]
        m = configs[:, 2]
        q0 = configs[:, 3]

        cost_egs = p_az_kw * self.C_EGS_per_kw
        cost_ess = e_max * self.C_ESS_per_kwh
        q0_lmin = q0 * 60 * 1000
        unit_capacity_factor = q0_lmin / 100.0

        cost_compressor_unit = (unit_capacity_factor * self.C_Compressor_base) + self.C_Nozzle + self.C_Pipe
        cost_air_system = m * cost_compressor_unit

        total_alcc = cost_egs + cost_ess + cost_air_system + self.C_Other
        return total_alcc

# ==============================================================================
# ==============================================================================
class Particle:
    def __init__(self, bounds):
        self.position = np.array([random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.objectives = np.array([float('inf'), float('inf')])
        self.best_objectives = np.array([float('inf'), float('inf')])

class AUS_MOPSO_Vectorized:
    def __init__(self, data_dir, model_path, bounds, num_particles=50, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, fast_mode=False):
        self.data_dir = data_dir
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.fast_mode = fast_mode

        print("Initializing vectorized environment...")
        self.env = Vectorized_AUS_Environment(data_dir, num_envs=num_particles)

        print("Loading agent model...")
        model_state_dict = torch.load(model_path, map_location='cpu')
        self.agent = BatchAgentEvaluator(
            Optimized_Interaction_Network, model_state_dict,
            self.env.dynamic_dim, self.env.static_dim, self.env.fixed_action_dim,
            device='cpu'
        )

        self.cost_calc = VectorizedCostCalculator()
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.repository = []

        self.history = {
            'iter': [], 'alcc_min': [], 'alcc_avg': [], 'nti_max': [], 'nti_avg': []
        }

    def decode_configs(self, swarm):
        raw_pos = np.array([p.position for p in swarm])
        configs = np.zeros_like(raw_pos)

        for i, b in enumerate(self.bounds):
            configs[:, i] = np.clip(raw_pos[:, i], b[0], b[1])

        configs[:, 2] = np.round(configs[:, 2])
        return configs

    def evaluate_swarm(self):
        
        configs = self.decode_configs(self.swarm)  # (N, 4)

        alcc_scores = self.cost_calc.calculate_alcc_batch(configs)

        nti_scores = np.zeros(self.num_particles)
        years = self.env.train_years if not self.fast_mode else [2024]

        for year in years:
            # Reset all particles for this year
            states, masks = self.env.reset(configs, year)
            done = False

            while not done:
                # Batch Inference
                actions = self.agent.batch_select_action(states, masks)

                # Batch Physics Step
                next_states, _, done, infos = self.env.step(actions)

                # Accumulate NTI (Vectorized)
                # flag=1 (Valid) & Vt > epsilon
                valid_mask = (infos['flag'] == 1)
                step_nti = infos['f2_t'] * infos['Vt']
                nti_scores[valid_mask] += step_nti[valid_mask]

                states = next_states
                masks = infos['action_mask']

        # Average NTI over years
        nti_scores /= len(years)

        return alcc_scores, nti_scores

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

        # Pruning
        max_repo = 100
        if len(non_dominated) > max_repo:
            non_dominated.sort(key=lambda x: x.objectives[0])
            keep = [non_dominated[0], non_dominated[-1]]
            if max_repo > 2:
                keep.extend(random.sample(non_dominated[1:-1], max_repo - 2))
            self.repository = keep
        else:
            self.repository = non_dominated

    def optimize(self):
        print(f"Starting vectorized MOPSO (particles={self.num_particles}, iterations={self.max_iter})...")
        print(f"{'Iter':<5} | {'Repo':<4} | {'ALCC (Min/Avg)':<20} | {'NTI (Max/Avg)':<20}")
        print("-" * 65)

        for it in range(self.max_iter):
            alcc_vec, nti_vec = self.evaluate_swarm()

            current_alcc_list = []
            current_nti_list = []

            for i, p in enumerate(self.swarm):
                # Objectives: [ALCC, -NTI] (Minimize both)
                current_objs = np.array([alcc_vec[i], -nti_vec[i]])
                p.objectives = current_objs

                current_alcc_list.append(current_objs[0])
                current_nti_list.append(-current_objs[1])

                # Update pBest
                if self._dominates(current_objs, p.best_objectives):
                    p.best_position = p.position.copy()
                    p.best_objectives = current_objs.copy()
                elif not self._dominates(p.best_objectives, current_objs):
                    if random.random() < 0.5:
                        p.best_position = p.position.copy()
                        p.best_objectives = current_objs.copy()

            self.update_repository()

            # Stats & Log
            alcc_min, alcc_avg = np.min(current_alcc_list), np.mean(current_alcc_list)
            nti_max, nti_avg = np.max(current_nti_list), np.mean(current_nti_list)

            self.history['iter'].append(it)
            self.history['alcc_min'].append(alcc_min)
            self.history['alcc_avg'].append(alcc_avg)
            self.history['nti_max'].append(nti_max)
            self.history['nti_avg'].append(nti_avg)

            print(f"{it + 1:<5} | {len(self.repository):<4} | "
                  f"{alcc_min:.0f} / {alcc_avg:.0f} | "
                  f"{nti_max:.0f} / {nti_avg:.0f}")

            # Update Velocity & Position
            if not self.repository: leader_pos = self.swarm[0].best_position

            for p in self.swarm:
                leader = random.choice(self.repository) if self.repository else p
                leader_pos = leader.position

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
    MODEL_PATH = os.path.join(_MODEL_DIR, "proposed_agent.pth")

    bounds = [
        (0, 60000),  # PV
        (0.0, 120.0),  # Battery
        (1, 24),  # M
        (0.001, 0.003)  # Q0
    ]

    mopso = AUS_MOPSO_Vectorized(
        DATA_DIR, MODEL_PATH, bounds,
        num_particles=50,
        max_iter=100,
        fast_mode=False
    )

    pareto_front = mopso.optimize()

    results = []
    for p in pareto_front:
        res = {
            "p_az": p.position[0],
            "e_max": p.position[1],
            "m": int(round(p.position[2])),
            "q0": p.position[3],
            "ALCC": p.objectives[0],
            "NTI": -p.objectives[1]
        }
        results.append(res)

    df_res = pd.DataFrame(results).sort_values(by="ALCC")
    save_path = os.path.join(_RESULT_DIR, "MOPSO_Vectorized_Results.xlsx")
    df_res.to_excel(save_path, index=False)
    print(f"\nOptimization finished, saved to: {save_path}")

    hist = mopso.history
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.plot(hist['iter'], hist['alcc_min'], 'g-', label='Min ALCC')
    ax1.plot(hist['iter'], hist['alcc_avg'], 'g--', alpha=0.3)
    ax2.plot(hist['iter'], hist['nti_max'], 'b-', label='Max NTI')
    ax2.plot(hist['iter'], hist['nti_avg'], 'b--', alpha=0.3)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (CNY)', color='g')
    ax2.set_ylabel('NTI', color='b')
    ax1.set_title('Convergence')

    axes[1].scatter(df_res['ALCC'], df_res['NTI'], c='r')
    axes[1].set_xlabel('Cost')
    axes[1].set_ylabel('NTI')
    axes[1].set_title('Pareto Frontier')

    plt.tight_layout()
    plt.show()