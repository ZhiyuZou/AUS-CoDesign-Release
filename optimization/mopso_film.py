import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
import copy
import matplotlib.pyplot as plt
from typing import Optional

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

        self.current_config = None
        self.current_step = 0
        self.current_state = None
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

    def _get_action_mask(self, E_t: float, I_t: float) -> np.ndarray:
        solar_power = self.get_solar_power(I_t)
        max_discharge = min(self.d_max, (max(0, E_t - self.E_min) * self.eta_d) / self.dt)
        total_power_available = solar_power + max_discharge

        mask = np.zeros(self.fixed_action_dim, dtype=np.float32)
        for action in range(self.fixed_action_dim):
            ratio = action / (self.fixed_action_dim - 1)
            actual_m = round(ratio * self.M)
            power_req = self.b + (actual_m * self.p0)
            if power_req <= total_power_available:
                mask[action] = 1.0
        if np.sum(mask) == 0: mask[0] = 1.0
        return mask

    def reset(self, random_year: bool = True, fixed_config: Optional[dict] = None, fixed_year: Optional[int] = None):
        if fixed_config:
            self.current_config = fixed_config.copy()
            self.current_config["q0"] = np.clip(self.current_config.get("q0", 0.002),
                                                self.config_ranges["q0"][0], self.config_ranges["q0"][1])
        else:
            self.current_config = self._random_config()

        self.P_AZ = self.current_config["p_az"]
        self.E_max = self.current_config["e_max"]
        self.E_min = 0.05 * self.E_max
        self.M = int(self.current_config["m"])
        self.Q0 = self.current_config["q0"]
        self.p0 = ((self.Q0 * 60 * 1000 + 58.691) / 0.1442) / 1000

        if fixed_year is not None:
            if fixed_year in self.year_data_dict:
                self.current_year = fixed_year
                self.env_data = self.year_data_dict[fixed_year]
            else:
                self.current_year = list(self.year_data_dict.keys())[0]
                self.env_data = self.year_data_dict[self.current_year]
        elif random_year:
            self.current_year = random.choice(list(self.year_data_dict.keys()))
            self.env_data = self.year_data_dict[self.current_year]
        else:
            self.current_year = list(self.year_data_dict.keys())[0]
            self.env_data = self.year_data_dict[self.current_year]

        self.current_step = 0
        init_E = np.random.uniform(self.E_min, self.E_max)
        init_temp = self.env_data[0, 0]
        init_light = self.env_data[0, 1]
        init_tide = self.env_data[0, 2]
        init_current = self.env_data[0, 3]

        raw_state = np.array([
            init_E, init_temp, init_light, init_tide, init_current, self.current_step,
            self.P_AZ, self.E_max, self.M, self.Q0
        ], dtype=np.float32)

        self.current_state = self.normalize_state(raw_state)
        mask = self._get_action_mask(init_E, init_light)
        return self.current_state, mask

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state_norm: np.ndarray) -> np.ndarray:
        return state_norm * self.state_std + self.state_mean

    def _random_config(self) -> dict:
        return {k: np.random.uniform(*v) if isinstance(v[0], float) else np.random.randint(*v)
                for k, v in self.config_ranges.items()}

    def _map_action_to_compressors(self, action: int) -> int:
        ratio = action / (self.fixed_action_dim - 1)
        return int(round(ratio * self.M))

    def get_solar_power(self, I_t: float) -> float:
        return np.clip((I_t * self.dt * self.P_AZ * self.K) / 1e6, 0.0, self.P_AZ / 1000 * self.K)

    def calc_Vt(self, m_t: int, u_c_t: float, u_next: float, tide: float) -> float:
        if m_t == 0: return 0.0
        u_safe = max(abs(u_c_t), 1e-8)
        u_next_safe = max(abs(u_next), 1e-8)
        try:
            Zd = 5.1 * self.Q0 / ((self.v_s ** 2.4) * u_safe) ** 0.88 * self.g
            alpha = 0.082 * np.tanh((self.g * self.Q0 / 10.4) ** (1 / 3) / self.v_s) ** (3 / 8)
            A = 1.02 * alpha ** (-1) * (self.g * 1.2 * self.Q0 * 1.25 / 3.14 / 1018) ** (1 / 3)
            delta_z = self.d0 / (2.4 * alpha)
            td = ((Zd + delta_z) ** (4 / 3) - delta_z ** (4 / 3)) * 3 / 4 / A
            xd1 = td * u_safe
            Bd = 1.2 * (Zd + delta_z) * alpha
            vd = A * (Zd + delta_z) ** (-1 / 3)
            Qd = 3.14 * Bd ** 2 * np.sqrt(u_c_t ** 2 + vd ** 2)
            v0 = A * delta_z ** (-1 / 3)
            Q0_calc = 3.14 * self.d0 ** 2 * np.sqrt(u_c_t ** 2 + v0 ** 2)
            rho_d = ((Qd - Q0_calc) * self.rho_w + Q0_calc * (self.rho_w + self.rho_delta)) / Qd
            vmd = vd * np.sqrt(2 * 3.14) / 6
            jieta = rho_d / (rho_d - self.rho_w)
            beta = 0.17
            Zm = np.sqrt((Bd / beta) ** 2 + 4 / 3 * jieta * Bd * vmd ** 2 / (beta * self.g)) - Bd / beta + Zd
            xd = self.x_d if u_c_t > 0 else 120 - self.x_d

            if Zm < self.Z_s + tide: return 0.0
            if Zd > self.Z_s + tide:
                xs = (((self.Z_s + tide + delta_z) ** (4 / 3) - delta_z ** (4 / 3)) * 3 / 4 / A) * u_safe
            else:
                term = 1 - 3 * beta * self.g / (4 * jieta * Bd * vmd ** 2) * (self.Z_s + tide - Zd) * (
                        self.Z_s + tide + 2 * Bd / beta - Zd)
                if term < 0: term = 0
                xs = jieta * vmd * u_safe / self.g * (1 - term ** (2 / 3)) + xd1
            if xs >= xd: return 0.0

            xaq = 120 if ((u_next > 0 and u_c_t < 0) or (u_next < 0 and u_c_t > 0)) else xd
            denom = xd - xs
            if abs(denom) < 1e-6: denom = 1e-6
            kata = ((xd - xs) / (u_safe * 3600)) ** 2 * (
                    (u_safe * 3600 - xd) / denom + u_safe / u_next_safe * ((xaq - xd) / denom + 0.5) + 0.5
            )
            Vavg = kata * Q0_calc * self.dt * 3600
            return m_t * Vavg
        except:
            return 0.0

    def calc_f2(self, T_t: float, I_t: float) -> float:
        I_illu = I_t * 0.168
        f_I = min(I_illu / self.I_s, 2.0)
        T_x = self.T_min if T_t <= self.T_opt else self.T_max
        if abs(T_x - self.T_opt) < 1e-6:
            f_T = 1.0
        else:
            f_T = np.exp(1 - f_I - 2.3 * ((T_t - self.T_opt) / (T_x - self.T_opt)) ** 2)
        return max(f_I * f_T, 0.0)

    def step(self, action: int):
        actual_compressors = self._map_action_to_compressors(action)
        state = self.denormalize_state(self.current_state)
        E_t = state[0]

        idx = min(self.current_step, self.total_time - 1)
        T_t, I_t, Z_t, u_t = self.env_data[idx]

        self.current_step += 1
        next_idx = min(self.current_step, self.total_time - 1)
        u_next = self.env_data[next_idx, 3]

        g_t = self.get_solar_power(I_t)
        q_t = actual_compressors * self.p0
        total_load = self.b + q_t
        energy_surplus = g_t - total_load

        c_t, d_t, flag = 0.0, 0.0, 1
        if energy_surplus > 0:
            c_t = min(energy_surplus, self.c_max, (self.E_max - E_t) / (self.eta_c * self.dt))
            E_next = E_t + self.eta_c * c_t * self.dt
        elif -energy_surplus <= (E_t - self.E_min) * self.eta_d / self.dt:
            E_next = E_t + energy_surplus * self.dt / self.eta_d
        else:
            actual_compressors, q_t, flag = 0, 0, 0
            E_next = E_t

        E_next = np.clip(E_next, self.E_min, self.E_max)
        Vt = self.calc_Vt(actual_compressors, u_t, u_next, Z_t)
        f2_t = self.calc_f2(T_t, I_t)

        reward = 0.0
        if flag == 1:
            if actual_compressors == 0 or (Vt >= 1e-6 and f2_t >= 1e-2):
                reward += self.beta * f2_t * Vt
            else:
                reward -= 1

        energy_waste = max(energy_surplus - c_t, 0.0)
        if energy_waste > 20: reward -= 0

        done = self.current_step >= self.total_time

        raw_next_state = np.array([
            E_next, self.env_data[next_idx, 0], self.env_data[next_idx, 1],
            self.env_data[next_idx, 2], self.env_data[next_idx, 3],
            self.current_step,
            self.P_AZ, self.E_max, self.M, self.Q0
        ], dtype=np.float32)

        self.current_state = self.normalize_state(raw_next_state)
        if not done:
            next_I = self.env_data[self.current_step, 1] if self.current_step < self.total_time else 0
            next_mask = self._get_action_mask(E_next, next_I)
        else:
            next_mask = np.ones(self.fixed_action_dim, dtype=np.float32)

        info = {"Vt": Vt, "f2_t": f2_t, "energy_waste": energy_waste, "flag": flag, "action_mask": next_mask}
        return self.current_state, reward, done, info

# ==============================================================================
# ==============================================================================
class StandardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=200):
        super().__init__()
        self.feature_net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, num_quantiles))
        self.advantage_stream = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),
                                              nn.Linear(512, action_dim * num_quantiles))
        self.action_dim, self.num_quantiles = action_dim, num_quantiles

    def forward(self, x):
        features = self.feature_net(x)
        v = self.value_stream(features).unsqueeze(1)
        a = self.advantage_stream(features).view(-1, self.action_dim, self.num_quantiles)
        return v + a - a.mean(dim=1, keepdim=True)

class DuelingQR_DQNNetwork(StandardNetwork): pass

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
class AgentEvaluator:
    def __init__(self, model_class, model_state_dict, state_dim, action_dim, dynamic_dim=None, static_dim=None,
                 force_no_mask=False, device='cpu'):
        self.device = torch.device(device)
        self.force_no_mask = force_no_mask

        if model_class == Optimized_Interaction_Network:
            self.net = model_class(dynamic_dim, static_dim, action_dim).to(self.device)
            self.use_mask = not force_no_mask
        else:
            self.net = model_class(state_dim, action_dim).to(self.device)
            self.use_mask = (not force_no_mask) if model_class == DuelingQR_DQNNetwork else False

        try:
            self.net.load_state_dict(model_state_dict)
        except Exception as e:
            print(f"Failed to load model weights: {e}")
        self.net.eval()

    def select_action(self, state, mask):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(state_t).mean(dim=-1)
            if self.use_mask: q[torch.tensor(mask, device=self.device).unsqueeze(0) == 0] = -float('inf')
        return q.argmax().item()

# ==============================================================================
# ==============================================================================
class CostCalculator:
    def __init__(self):
        self.C_EGS_per_kw = 95.9
        self.C_ESS_per_kwh = 36.6
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
def evaluate_single_particle_static(position, bounds, env_data_dir, preloaded_data,
                                    model_class, model_state_dict,
                                    state_dim, action_dim, dyn_dim, stat_dim,
                                    fast_mode=False):
    

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

    agent = AgentEvaluator(
        model_class, model_state_dict, state_dim, action_dim,
        dynamic_dim=dyn_dim, static_dim=stat_dim, device='cpu',
        force_no_mask=True  # Improved = FiLM No Mask
    )

    total_nti = 0
    years = env.train_years if not fast_mode else [2024]

    for year in years:
        state, mask = env.reset(random_year=False, fixed_config=config, fixed_year=year)
        done = False
        year_nti = 0
        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done, info = env.step(action)

            if info['flag']:
                year_nti += info['f2_t'] * info['Vt']

            state, mask = next_state, info['action_mask']
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

class AUS_MOPSO_Parallel:
    def __init__(self, data_dir, model_path, bounds, num_particles=50, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, fast_mode=False, model_type='film'):
        self.data_dir = data_dir
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.fast_mode = fast_mode

        print("Preloading environmental data...")
        temp_env = AUS_Environment(data_dir)
        self.preloaded_data = temp_env.year_data_dict
        self.env_dims = (temp_env.state_dim, temp_env.fixed_action_dim,
                         temp_env.dynamic_dim, temp_env.static_dim)

        print("Loading model weights...")
        state_dict = torch.load(model_path, map_location='cpu')
        self.model_state_dict = state_dict

        if model_type == 'standard':
            self.model_class = StandardNetwork
        elif model_type == 'dueling':
            self.model_class = DuelingQR_DQNNetwork
        else:
            self.model_class = Optimized_Interaction_Network  # Default to FiLM

        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.repository = []

        self.history = {
            'iter': [],
            'alcc_min': [], 'alcc_avg': [], 'alcc_max': [],
            'nti_min': [], 'nti_avg': [], 'nti_max': []
        }

    def _dominates(self, obj_a, obj_b):
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def update_repository(self):
        for particle in self.swarm:
            self.repository.append(copy.deepcopy(particle))

        non_dominated = []
        for i, p1 in enumerate(self.repository):
            is_dominated = False
            for j, p2 in enumerate(self.repository):
                if i != j and self._dominates(p2.objectives, p1.objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(p1)

        max_repo_size = 100
        if len(non_dominated) > max_repo_size:
            non_dominated.sort(key=lambda x: x.objectives[0])
            keep = [non_dominated[0], non_dominated[-1]]
            keep.extend(random.sample(non_dominated[1:-1], max_repo_size - 2))
            self.repository = keep
        else:
            self.repository = non_dominated

    def select_leader(self):
        if not self.repository: return self.swarm[0].best_position
        leader = random.choice(self.repository)
        return leader.position

    def optimize(self):
        print(f"Starting MOPSO (particles={self.num_particles}, iterations={self.max_iter})...")
        print(f"Using Model Class: {self.model_class.__name__}")
        print(f"{'Iter':<5} | {'Repo':<4} | {'ALCC (Min/Avg)':<20} | {'NTI (Max/Avg)':<20}")
        print("-" * 65)

        eval_args_base = (self.bounds, self.data_dir, self.preloaded_data,
                          self.model_class, self.model_state_dict,
                          *self.env_dims, self.fast_mode)

        for it in range(self.max_iter):
            results = [evaluate_single_particle_static(p.position, *eval_args_base) for p in self.swarm]

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

            print(f"{it + 1:<5} | {len(self.repository):<4} | "
                  f"{alcc_min:.0f} / {alcc_avg:.0f} | "
                  f"{nti_max:.0f} / {nti_avg:.0f}")

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

    MODEL_PATH = os.path.join(_MODEL_DIR, "film_agent.pth")

    bounds = [
        (0, 60000),  # PV
        (0.0, 120.0),  # Battery
        (1, 24),  # M
        (0.001, 0.003)  # Q0
    ]

    mopso = AUS_MOPSO_Parallel(
        DATA_DIR, MODEL_PATH, bounds,
        num_particles=50,
        max_iter=100,
        fast_mode=False,
        model_type='film'
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
    save_path = os.path.join(_RESULT_DIR, "MOPSO_Pareto_Results_FiLM.xlsx")
    df_res.to_excel(save_path, index=False)
    print(f"\nOptimization finished, saved to: {save_path}")

    # ==========================================================================
    # ==========================================================================
    history = mopso.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax2 = ax1.twinx()

    l1, = ax1.plot(history['iter'], history['alcc_min'], 'g-', label='Min ALCC (Cost)')
    ax1.plot(history['iter'], history['alcc_avg'], 'g--', alpha=0.5)

    l2, = ax2.plot(history['iter'], history['nti_max'], 'b-', label='Max NTI (Effect)')
    ax2.plot(history['iter'], history['nti_avg'], 'b--', alpha=0.5)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Annualized Cost (CNY)', color='g')
    ax2.set_ylabel('Nutrient Transport Index (NTI)', color='b')
    ax1.set_title('FiLM Agent Optimization Process')

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
    ax1.grid(True, alpha=0.3)

    axes[1].scatter(df_res['ALCC'], df_res['NTI'], c='orange', edgecolor='k', s=50, label='Pareto Optimal (FiLM)')
    axes[1].set_xlabel('Annualized Life-Cycle Cost (CNY)')
    axes[1].set_ylabel('Nutrient Transport Index (NTI)')
    axes[1].set_title('Pareto Frontier (FiLM Agent)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()