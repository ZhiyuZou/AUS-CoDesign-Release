import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
from typing import Optional, Dict
import os
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
class Optimized_Interaction_Network(nn.Module):
    def __init__(self, dynamic_dim, static_dim, action_dim, num_quantiles=200):
        super().__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.hidden_dim = 512

        self.dynamic_net = nn.Sequential(
            nn.Linear(dynamic_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim * 2)
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # 4. Heads
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_quantiles)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * num_quantiles)
        )

    def forward(self, x):
        dynamic_dim = self.dynamic_net[0].in_features
        dynamic_x = x[:, :dynamic_dim]
        static_x = x[:, dynamic_dim:]

        dyn_out = self.dynamic_net(dynamic_x)  # [Batch, 512]

        stat_raw = self.static_net(static_x)  # [Batch, 1024]
        scale_raw, shift_raw = torch.chunk(stat_raw, 2, dim=1)  # [Batch, 512] * 2

        scale = torch.sigmoid(scale_raw)
        shift = shift_raw

        modulated_features = (dyn_out * scale) + shift

        features = self.fusion_net(modulated_features)

        v = self.value_stream(features).unsqueeze(1)
        a = self.advantage_stream(features).view(-1, self.action_dim, self.num_quantiles)
        q_quantiles = v + a - a.mean(dim=1, keepdim=True)

        return q_quantiles

    def get_expectation(self, quantiles):
        return torch.mean(quantiles, dim=-1)

# ==============================================================================
# ==============================================================================
class Improved_DuelingQR_DQNAgent:
    def __init__(self, dynamic_dim, static_dim, action_dim=24, num_quantiles=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.tau = torch.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles).to(self.device)

        self.current_net = Optimized_Interaction_Network(dynamic_dim, static_dim, action_dim, num_quantiles).to(
            self.device)
        self.target_net = Optimized_Interaction_Network(dynamic_dim, static_dim, action_dim, num_quantiles).to(
            self.device)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.current_net.parameters(), lr=0.00025)
        # Configuration-partitioned replay (32 sub-buffers, Sec. 2.2.2).
        self.n_groups = 32
        self.sub_buffers = [deque(maxlen=2_000_000 // self.n_groups) for _ in range(self.n_groups)]
        self.global_buffer = deque(maxlen=2_000_000)
        self.batch_size = 256
        self.gamma = 0.99
        self.target_update_freq = 10000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 200000
        self.train_steps = 0
        self.huber_loss = nn.HuberLoss(delta=1.0)

    def select_action(self, states, is_train=True):
        
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch_size = state_tensor.shape[0]

        if is_train:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-self.train_steps / self.epsilon_decay)
            self.train_steps += batch_size

            random_vals = np.random.rand(batch_size)
            random_actions = np.random.randint(0, self.action_dim, size=batch_size)

            with torch.no_grad():
                quantiles = self.current_net(state_tensor)
                q_expect = self.current_net.get_expectation(quantiles)
                greedy_actions = q_expect.argmax(dim=1).cpu().numpy()

            final_actions = np.where(random_vals < epsilon, random_actions, greedy_actions)
            return final_actions[0] if batch_size == 1 else final_actions
        else:
            with torch.no_grad():
                quantiles = self.current_net(state_tensor)
                q_expect = self.current_net.get_expectation(quantiles)
            final_actions = q_expect.argmax(dim=1).cpu().numpy()
            return final_actions[0] if batch_size == 1 else final_actions

    def add_experience(self, state, action, reward, next_state, done, group=0):
        transition = (state, action, reward, next_state, done)
        self.sub_buffers[group].append(transition)
        self.global_buffer.append(transition)

    def _stratified_batch(self):
        batch = []
        for g in range(self.n_groups):
            buf = self.sub_buffers[g]
            k = self.batch_size // self.n_groups
            if len(buf) >= k:
                batch.extend(random.sample(buf, k))
            elif len(buf) > 0:
                batch.extend(random.choices(buf, k=k))
        shortfall = self.batch_size - len(batch)
        if shortfall > 0 and len(self.global_buffer) > 0:
            pool = self.global_buffer
            batch.extend(random.sample(pool, min(shortfall, len(pool)))
                         if len(pool) >= shortfall else random.choices(pool, k=shortfall))
        return batch

    def learn(self):
        if len(self.global_buffer) < self.batch_size: return 0.0

        batch = self._stratified_batch()
        state_batch = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(np.array([e[1] for e in batch]), dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(np.array([e[2] for e in batch]), dtype=torch.float32, device=self.device).unsqueeze(
            1)
        next_state_batch = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(np.array([e[4] for e in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_quantiles = self.target_net(next_state_batch)
            next_q_expect = self.target_net.get_expectation(next_quantiles)

            next_action = next_q_expect.argmax(dim=1, keepdim=True)

            next_quantiles_selected = next_quantiles.gather(1, next_action.unsqueeze(2).expand(-1, -1,
                                                                                               self.num_quantiles)).squeeze(
                1)
            target_quantiles = reward_batch + self.gamma * (1 - done_batch) * next_quantiles_selected
            target_quantiles = target_quantiles.unsqueeze(1).expand(-1, self.num_quantiles, -1)

        current_quantiles = self.current_net(state_batch)
        current_quantiles_selected = current_quantiles.gather(1, action_batch.unsqueeze(2).expand(-1, -1,
                                                                                                  self.num_quantiles)).squeeze(
            1)
        current_quantiles_selected = current_quantiles_selected.unsqueeze(2).expand(-1, -1, self.num_quantiles)

        td_error = target_quantiles - current_quantiles_selected
        tau = self.tau.unsqueeze(0).expand(self.batch_size, -1).unsqueeze(2)
        weight = torch.abs(tau - (td_error < 0).float())
        loss = (self.huber_loss(current_quantiles_selected, target_quantiles) * weight).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_steps % self.target_update_freq < self.batch_size:
            self.target_net.load_state_dict(self.current_net.state_dict())
        return loss.item()

# ==============================================================================
# ==============================================================================
class AUS_Environment:
    def __init__(self, data_dir: str):
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

        self.state_mean = None
        self.state_std = None
        self.year_data_dict = {}

        self.load_all_year_data()
        self.calculate_statistics()

        self.current_config = None
        self.current_step = 0
        self.current_state = None
        self.env_data = None

    def load_all_year_data(self):
        loaded_count = 0
        for year in self.train_years:
            try:
                xlsx_path = os.path.join(self.data_dir, f"{year}.xlsx")
                df = pd.read_excel(xlsx_path, header=0, index_col=0, engine="openpyxl").reset_index(drop=True)
                df.columns = ["temperature", "light", "tide_height", "current_speed"]
                df = df.astype(float)
                if len(df) > 0 and df.shape[1] >= 3:
                    df.iloc[:, 2] = df.iloc[:, 2] / 100
                df = df.head(self.total_time).ffill().bfill()
                if len(df) < self.total_time:
                    last_row = df.iloc[-1] if len(df) > 0 else pd.Series([0, 0, 0, 0], index=df.columns)
                    df = pd.concat([df, pd.DataFrame([last_row] * (self.total_time - len(df)))], ignore_index=True)

                valid_mask = (df.notna().all(axis=1)) & (df["light"] >= 0)
                df = df[valid_mask].reindex(range(self.total_time), method='ffill')
                self.year_data_dict[year] = df.values
                loaded_count += 1
            except Exception as e:
                print(f"[Warn] Failed to load year {year}: {e}")

        if loaded_count == 0:
            raise FileNotFoundError(f"Fatal: no valid yearly data found under {self.data_dir}")

    def calculate_statistics(self):
        available_years = list(self.year_data_dict.keys())
        all_data = np.concatenate([self.year_data_dict[y] for y in available_years])
        e_mean_est = (self.config_ranges["e_max"][1] + self.config_ranges["e_max"][0]) / 4

        dyn_mean = np.array([e_mean_est, np.mean(all_data[:, 0]), np.mean(all_data[:, 1]),
                             np.mean(all_data[:, 2]), np.mean(all_data[:, 3]), self.total_time / 2], dtype=np.float32)
        dyn_std = np.array([20.0, np.std(all_data[:, 0]), np.std(all_data[:, 1]),
                            np.std(all_data[:, 2]), np.std(all_data[:, 3]), self.total_time / 4], dtype=np.float32)

        stat_mean = np.array([np.mean(self.config_ranges["p_az"]), np.mean(self.config_ranges["e_max"]),
                              np.mean(self.config_ranges["m"]), np.mean(self.config_ranges["q0"])], dtype=np.float32)
        stat_std = np.array([(self.config_ranges["p_az"][1] - self.config_ranges["p_az"][0]) / 10,
                             (self.config_ranges["e_max"][1] - self.config_ranges["e_max"][0]) / 10,
                             (self.config_ranges["m"][1] - self.config_ranges["m"][0]) / 10,
                             (self.config_ranges["q0"][1] - self.config_ranges["q0"][0]) / 10], dtype=np.float32)

        self.state_mean = np.concatenate([dyn_mean, stat_mean])
        self.state_std = np.concatenate([dyn_std, stat_std])
        self.state_std[self.state_std < 1e-6] = 1e-6

    def reset(self, random_year: bool = True, fixed_config: Optional[dict] = None):
        if fixed_config:
            self.current_config = fixed_config.copy()
            self.current_config["q0"] = np.clip(self.current_config.get("q0", 0.002),
                                                self.config_ranges["q0"][0], self.config_ranges["q0"][1])
        else:
            self.current_config = self._random_config()

        self.P_AZ = self.current_config["p_az"]
        self.E_max = self.current_config["e_max"]
        self.E_min = 0.05 * self.E_max
        self.M = self.current_config["m"]
        self.Q0 = self.current_config["q0"]
        self.p0 = ((self.Q0 * 60 * 1000 + 58.691) / 0.1442) / 1000

        available = list(self.year_data_dict.keys())
        self.current_year = random.choice(available) if random_year else available[0]
        self.env_data = self.year_data_dict[self.current_year]
        self.current_step = 0

        init_E = np.random.uniform(self.E_min, self.E_max)
        raw_state = np.array(
            [init_E, self.env_data[0, 0], self.env_data[0, 1], self.env_data[0, 2], self.env_data[0, 3],
             self.current_step, self.P_AZ, self.E_max, self.M, self.Q0], dtype=np.float32)

        self.current_state = self.normalize_state(raw_state)
        return self.current_state

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
        T_t, I_t, Z_t, u_t = self.env_data[self.current_step, 0:4]

        self.current_step += 1
        u_next = self.env_data[self.current_step, 3] if self.current_step < self.total_time else u_t

        g_t = self.get_solar_power(I_t)
        q_t = actual_compressors * self.p0
        total_load = self.b + q_t
        energy_surplus = g_t - total_load

        c_t, flag = 0.0, 1

        if energy_surplus > 0:
            c_t = min(energy_surplus, self.c_max, (self.E_max - E_t) / (self.eta_c * self.dt))
            E_next = E_t + self.eta_c * c_t * self.dt
        elif -energy_surplus <= (E_t - self.E_min) * self.eta_d / self.dt:
            E_next = E_t + energy_surplus * self.dt / self.eta_d
        else:
            actual_compressors, q_t = 0, 0
            flag = 0
            energy_surplus = g_t - self.b
            if energy_surplus > 0:
                c_t = min(energy_surplus, self.c_max, (self.E_max - E_t) / (self.eta_c * self.dt))
                E_next = E_t + self.eta_c * c_t * self.dt
            else:
                E_next = E_t + energy_surplus * self.dt / self.eta_d

        E_next = np.clip(E_next, self.E_min, self.E_max)

        Vt = self.calc_Vt(actual_compressors, u_t, u_next, Z_t)
        f2_t = self.calc_f2(T_t, I_t)

        reward = 0.0
        if flag == 1:
            if actual_compressors > 0 and (Vt < 1e-6 or f2_t < 1e-2):
                reward -= 1
            else:
                reward += self.beta * f2_t * Vt

        energy_waste = max(energy_surplus - c_t, 0.0)
        done = self.current_step >= self.total_time

        raw_next = np.array([E_next,
                             self.env_data[min(self.current_step, self.total_time - 1), 0],
                             self.env_data[min(self.current_step, self.total_time - 1), 1],
                             self.env_data[min(self.current_step, self.total_time - 1), 2],
                             self.env_data[min(self.current_step, self.total_time - 1), 3],
                             self.current_step, self.P_AZ, self.E_max, self.M, self.Q0], dtype=np.float32)

        self.current_state = self.normalize_state(raw_next)

        info = {
            "Vt": Vt, "f2_t": f2_t, "energy_waste": energy_waste, "flag": flag,
            "current_config": self.current_config
        }
        return self.current_state, reward, done, info

# ==============================================================================
# ==============================================================================

# Configuration signature -> sub-buffer index (equal-weight normalized config).
def config_group(cfg, ranges, n_groups=32):
    def norm(key):
        lo, hi = ranges[key]
        v = (cfg[key] - lo) / (hi - lo)
        return min(max(v, 0.0), 1.0 - 1e-9)
    signature = 0.25 * (norm('p_az') + norm('e_max') + norm('m') + norm('q0'))
    return int(1e6 * signature) % n_groups

# Sequential single-process training loop.
def train(data_dir, target_episodes=5000, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"=== Training film (single env, batch=256, 32 sub-buffers) | {timestamp} ===")
    env = AUS_Environment(data_dir)
    agent = Improved_DuelingQR_DQNAgent(env.dynamic_dim, env.static_dim, env.fixed_action_dim)
    train_log = []

    for ep in range(1, target_episodes + 1):
        state = env.reset(random_year=True)
        done = False
        ep_reward = ep_nti = ep_waste = 0.0
        loss_sum, loss_n = 0.0, 0
        while not done:
            action = int(agent.select_action(state[None, :], is_train=True))
            next_state, reward, done, info = env.step(action)
            group = config_group(env.current_config, env.config_ranges)
            agent.add_experience(state, action, reward, next_state, done, group)
            state = next_state
            ep_reward += reward
            if info['flag']:
                ep_nti += info['Vt'] * info['f2_t']
            ep_waste += info['energy_waste']
            if len(agent.global_buffer) > agent.batch_size:
                loss_sum += agent.learn()
                loss_n += 1
        cfg = env.current_config
        avg_loss = loss_sum / loss_n if loss_n else 0.0
        print(f"Episode [{ep}/{target_episodes}] reward={ep_reward:.2f} "
              f"loss={avg_loss:.4f} NTI={ep_nti:.0f} waste={ep_waste:.2f} "
              f"P={cfg['p_az']:.0f} M={cfg['m']} E={cfg['e_max']:.1f} Q={cfg['q0']:.6f}")
        train_log.append({"episode": ep, "total_reward": ep_reward, "avg_loss": avg_loss,
                          "total_nti": ep_nti, "total_energy_waste": ep_waste,
                          "p_az": cfg['p_az'], "e_max": cfg['e_max'],
                          "m": cfg['m'], "q0": cfg['q0']})
        if ep % 2500 == 0:
            torch.save(agent.current_net.state_dict(),
                       os.path.join(_MODEL_DIR, f"film_seed{seed}_checkpoint_ep{ep}.pth"))
        pd.DataFrame(train_log).to_excel(
            os.path.join(_RESULT_DIR, f"film_seed{seed}_log_{timestamp}.xlsx"), index=False)

    pd.DataFrame(train_log).to_excel(
        os.path.join(_RESULT_DIR, f"film_seed{seed}_log_{timestamp}.xlsx"), index=False)
    torch.save(agent.current_net.state_dict(),
               os.path.join(_MODEL_DIR, f"film_seed{seed}_final_{timestamp}.pth"))
    print(f"Saved final film model and training log.")

if __name__ == "__main__":
    # Five independent training replicates (seeds 0-4). Evaluate each trained
    # model on the fixed 100 configurations and save the long-format tables to
    # results/seeds/{Agent}_seed{k}.xlsx for statistical_tests/run_seed_variance.py.
    for _seed in range(5):
        train(_DATA_DIR, 5000, seed=_seed)

