import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

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

        self.P_AZ = config_dict["p_az"]
        self.E_max = config_dict["e_max"]
        self.E_min = 0.05 * self.E_max
        self.M = config_dict["m"]
        self.Q0 = config_dict["q0"]
        self.p0 = ((self.Q0 * 60 * 1000 + 58.691) / 0.1442) / 1000

    def get_solar_power(self, I_t):
        H_t = I_t * self.dt
        g_t = (H_t * self.P_AZ * 0.8) / (1e6)
        return np.clip(g_t, 0.0, self.P_AZ / 1000 * 0.8)

    def calc_Vt(self, m_t_continuous, u_c_t, u_next, tide):
        
        if m_t_continuous <= 1e-3: return 0.0

        u_safe = max(abs(u_c_t), 1e-8)
        u_next_safe = max(abs(u_next), 1e-8)

        Zd = 5.1 * self.Q0 / ((self.v_s ** 2.4) * u_safe) ** 0.88 * self.g
        alpha = 0.082 * np.tanh((self.g * self.Q0 / 10.4) ** (1 / 3) / self.v_s) ** (3 / 8)
        A = 1.02 * alpha ** (-1) * (self.g * 1.2 * self.Q0 * 1.25 / 3.14 / 1018) ** (1 / 3)
        delta_z = self.d0 / (2.4 * alpha)

        Bd = 1.2 * (Zd + delta_z) * alpha
        vd = A * (Zd + delta_z) ** (-1 / 3)

        v0 = A * delta_z ** (-1 / 3)
        plume_flux_Q0 = 3.14 * self.d0 ** 2 * np.sqrt(u_c_t ** 2 + v0 ** 2)

        Qd = 3.14 * Bd ** 2 * np.sqrt(u_c_t ** 2 + vd ** 2)
        rho_d = ((Qd - plume_flux_Q0) * self.rho_w + plume_flux_Q0 * (self.rho_w + self.rho_delta)) / Qd

        vmd = vd * np.sqrt(2 * 3.14) / 6
        jieta = rho_d / (rho_d - self.rho_w)
        beta_param = 0.17

        Zm = np.sqrt(
            (Bd / beta_param) ** 2 + 4 / 3 * jieta * Bd * vmd ** 2 / (beta_param * self.g)) - Bd / beta_param + Zd
        xd = self.x_d if u_c_t > 0 else 120 - self.x_d

        if Zm < self.Z_s + tide: return 0.0

        if Zd > self.Z_s + tide:
            delta_term = (self.Z_s + tide + delta_z) ** (4 / 3) - delta_z ** (4 / 3)
            xs = (delta_term * 3 / 4 / A) * u_safe
        else:
            term = 1 - 3 * beta_param * self.g / (4 * jieta * Bd * vmd ** 2) * (self.Z_s + tide - Zd) * (
                    self.Z_s + tide + 2 * Bd / beta_param - Zd)
            if term < 0: term = 0
            td = ((Zd + delta_z) ** (4 / 3) - delta_z ** (4 / 3)) * 3 / 4 / A
            xd1 = td * u_safe
            xs = jieta * vmd * u_safe / self.g * (1 - term ** (2 / 3)) + xd1

        if xs >= xd: return 0.0

        xaq = 120 if ((u_next > 0 and u_c_t < 0) or (u_next < 0 and u_c_t > 0)) else xd
        denom = xd - xs
        if abs(denom) < 1e-6: denom = 1e-6

        kata = ((xd - xs) / (u_safe * 3600)) ** 2 * (
                (u_safe * 3600 - xd) / denom + u_safe / u_next_safe * ((xaq - xd) / denom + 0.5) + 0.5)

        Vavg = kata * plume_flux_Q0 * self.dt * 3600
        return m_t_continuous * Vavg

    def calc_f2(self, T_t, I_t):
        I_illu = I_t * 0.168
        f_I = min(I_illu / self.I_s, 2.0)
        T_x = self.T_min if T_t <= self.T_opt else self.T_max
        if abs(T_x - self.T_opt) < 1e-6:
            f_T = 1.0
        else:
            temp_term = 2.3 * ((T_t - self.T_opt) / (T_x - self.T_opt)) ** 2
            f_T = np.exp(1 - f_I - temp_term)
        return max(f_I * f_T, 0.0)

# ==============================================================================
# ==============================================================================
class DP_Optimizer:
    def __init__(self, cfg: AUS_Config, num_battery_states=300):
        self.cfg = cfg
        self.num_states = num_battery_states
        self.E_grid = np.linspace(self.cfg.E_min, self.cfg.E_max, num_battery_states)
        self.actions = np.arange(self.cfg.M + 1)

    def solve_global_optimal(self, env_data, init_E):
        
        T_steps = min(len(env_data), self.cfg.total_time)
        print(f"\n--- [DP] Start global optimization (baseline) ---")
        print(f"    Horizon: {T_steps} steps | Battery Grid: {self.num_states} nodes | Actions: 0-{self.cfg.M}")

        V = np.zeros(self.num_states)

        Policy = np.zeros((T_steps, self.num_states), dtype=int)

        start_time = time.time()

        # ==================================================
        # ==================================================
        print("    Stage 1: Backward Induction...", end="")
        for t in range(T_steps - 1, -1, -1):
            row = env_data[t]
            T_t, I_t, Z_t, u_t = row[0], row[1], row[2], row[3]
            u_next = env_data[t + 1][3] if t < T_steps - 1 else u_t

            g_t = self.cfg.get_solar_power(I_t)

            rewards_actions = np.zeros(len(self.actions))
            power_costs = np.zeros(len(self.actions))

            for idx, a in enumerate(self.actions):
                Vt = self.cfg.calc_Vt(a, u_t, u_next, Z_t)
                f2_t = self.cfg.calc_f2(T_t, I_t)
                raw_nti = f2_t * Vt

                rewards_actions[idx] = self.cfg.beta * raw_nti

                if a > 0 and raw_nti < 1e-9:
                    rewards_actions[idx] = -1.0

                power_costs[idx] = self.cfg.b + a * self.cfg.p0

            E_curr_col = self.E_grid.reshape(-1, 1)  # (num_states, 1)

            # energy_surplus: (num_states, num_actions)
            energy_surplus = g_t - power_costs.reshape(1, -1)

            max_discharge_capacity = (E_curr_col - self.cfg.E_min) * self.cfg.eta_d / self.cfg.dt + 1e-5

            discharge_needed = -energy_surplus
            violation_mask = (energy_surplus < 0) & (discharge_needed > max_discharge_capacity)

            mask_charge = energy_surplus > 0
            c_t = np.minimum(energy_surplus, self.cfg.c_max)
            d_t = np.minimum(-energy_surplus, self.cfg.d_max)
            delta_E = (mask_charge * (c_t * self.cfg.eta_c) - (~mask_charge) * (d_t / self.cfg.eta_d)) * self.cfg.dt
            E_next_matrix = E_curr_col + delta_E
            E_next_matrix = np.clip(E_next_matrix, self.cfg.E_min, self.cfg.E_max)

            V_next_flat = np.interp(E_next_matrix.ravel(), self.E_grid, V)
            V_next_matrix = V_next_flat.reshape(self.num_states, len(self.actions))

            Q_matrix = rewards_actions.reshape(1, -1) + V_next_matrix

            Q_matrix[violation_mask] = -1e9

            best_actions_idx = np.argmax(Q_matrix, axis=1)
            V = np.max(Q_matrix, axis=1)
            Policy[t] = best_actions_idx

        print(f" Done ({time.time() - start_time:.2f}s)")

        # ==================================================
        # ==================================================
        print("    Stage 2: Forward Recovery...", end="")
        optimal_trajectory = []
        curr_E = init_E
        total_reward = 0
        total_raw_nti = 0

        for t in range(T_steps):
            state_idx = (np.abs(self.E_grid - curr_E)).argmin()
            best_act_idx = Policy[t, state_idx]
            best_action = self.actions[best_act_idx]

            row = env_data[t]
            T_t, I_t, Z_t, u_t = row[0], row[1], row[2], row[3]
            u_next = env_data[t + 1][3] if t < T_steps - 1 else u_t

            g_t = self.cfg.get_solar_power(I_t)
            q_t = best_action * self.cfg.p0
            surplus = g_t - (self.cfg.b + q_t)

            c_t_real = 0.0
            d_t_real = 0.0

            if surplus > 0:
                c_t_real = min(surplus, self.cfg.c_max)
                curr_E += c_t_real * self.cfg.eta_c * self.cfg.dt
            else:
                d_t_real = min(-surplus, self.cfg.d_max)
                curr_E -= d_t_real / self.cfg.eta_d * self.cfg.dt

            curr_E = np.clip(curr_E, self.cfg.E_min, self.cfg.E_max)

            Vt = self.cfg.calc_Vt(best_action, u_t, u_next, Z_t)
            f2_t = self.cfg.calc_f2(T_t, I_t)
            raw_nti = f2_t * Vt

            r_val = self.cfg.beta * raw_nti
            if best_action > 0 and raw_nti < 1e-9: r_val = -1.0

            total_reward += r_val
            total_raw_nti += raw_nti

            optimal_trajectory.append({
                "Step": t,
                "Action_Machine": best_action,
                "Battery_SoC": curr_E,
                "Solar_Power": g_t,
                "Load_Power": self.cfg.b + q_t,
                "Raw_NTI": raw_nti,
                "Current_Speed": u_t,
                "Irradiance": I_t,
                "Violation": "Yes" if (surplus < 0 and -surplus > (
                            curr_E + d_t_real - self.cfg.E_min) * self.cfg.eta_d / self.cfg.dt + 1.0) else "No"
            })
        print(" Done")

        return optimal_trajectory, total_reward, total_raw_nti

# ==============================================================================
# ==============================================================================
class AUS_Environment:
    def __init__(self, cfg: AUS_Config):
        self.cfg = cfg
        self.year_data_dict = {}
        self.load_all_year_data()
        self.env_data = None

    def load_all_year_data(self):
        train_years = [2024]
        for year in train_years:
            file_path = f"{self.cfg.data_dir}/{year}.xlsx"
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
            try:
                df = pd.read_excel(file_path, header=0, index_col=0, engine="openpyxl").reset_index(drop=True)
                df = df.astype(float)
                if len(df) > 0 and df.shape[1] >= 3:
                    df.iloc[:, 2] = df.iloc[:, 2] / 100
                if len(df) < self.cfg.total_time:
                    last_row = df.iloc[-1]
                    df = pd.concat([df, pd.DataFrame([last_row] * (self.cfg.total_time - len(df)))], ignore_index=True)
                df.columns = ["T", "I", "Z", "u"]
                df = df.head(self.cfg.total_time).ffill().bfill()
                self.year_data_dict[year] = df.values
            except Exception as e:
                print(f"Error loading {year}: {e}")

    def reset(self, year=2024):
        if year not in self.year_data_dict:
            return None
        self.env_data = self.year_data_dict[year]
        return (self.cfg.E_max + self.cfg.E_min) / 2

# ==============================================================================
# ==============================================================================
def run_dp_simulation(data_dir, config):
    print("========================================================")
    print("   AUS Global Optimal Trajectory (DP) Generator")
    print("========================================================")

    cfg = AUS_Config(data_dir, config)
    env = AUS_Environment(cfg)

    init_E = env.reset(year=2024)
    if env.env_data is None: return

    all_data = env.env_data  # (T, 4)

    dp_solver = DP_Optimizer(cfg, num_battery_states=300)
    traj, total_reward, total_nti = dp_solver.solve_global_optimal(all_data, init_E)

    df_results = pd.DataFrame(traj)

    print("\n========================================================")
    print("   FINAL RESULTS SUMMARY (Global Optimal)")
    print("========================================================")
    print(f"Total Reward (Obj): {total_reward:.2f}")
    print(f"Total Raw NTI     : {total_nti:.0f}")

    save_path = os.path.join(_RESULT_DIR, "DP_Global_Optimal_Trajectory_Corrected.xlsx")
    df_results.to_excel(save_path, index=False)
    print(f"\nDetailed trajectory saved to: {save_path}")

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df_results['Step'], df_results['Action_Machine'], label='Optimal Action', color='blue',
             drawstyle='steps-post')
    plt.ylabel('Compressors On')
    plt.title('DP Global Optimal Policy (Pure NTI Maximization)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(df_results['Step'], df_results['Battery_SoC'], label='Battery Energy (kWh)', color='green')
    plt.axhline(y=cfg.E_min, color='r', linestyle='--', label='Min Energy')
    plt.axhline(y=cfg.E_max, color='r', linestyle='--', label='Max Energy')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(df_results['Step'], df_results['Raw_NTI'], label='Raw NTI', color='orange', alpha=0.8)
    ax2.plot(df_results['Step'], df_results['Current_Speed'], label='Current Speed', color='cyan', alpha=0.3,
             linestyle=':')
    ax1.set_ylabel('NTI', color='orange')
    ax2.set_ylabel('Current Speed (m/s)', color='cyan')
    ax1.set_xlabel('Time Step (Minute)')
    plt.title('Physical Output vs Environment')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_PATH = _DATA_DIR

    TEST_CONFIG = {"p_az": 48000, "e_max": 86.4, "m": 16, "q0": 0.0016667}

    run_dp_simulation(DATA_PATH, TEST_CONFIG)