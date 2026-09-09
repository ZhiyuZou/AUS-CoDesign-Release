import numpy as np
import pandas as pd
from typing import Optional
import os

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

# ------------------------------------------------------
# ------------------------------------------------------
class AUS_Environment:
    def __init__(self, data_dir: str):
        self.dt = 1
        self.total_time = 2880
        self.E_max = 86.4
        self.E_min = 4.0
        self.eta_c = 0.95
        self.eta_d = 0.95
        self.c_max = 30.0
        self.d_max = 30.0
        self.P_AZ = 48000
        self.K = 0.8
        self.b = 0.2
        self.M = 16
        self.p0 = 1.0

        self.Q0 = 0.00166667
        self.Z_s = 8.0
        self.x_d = 45.0
        self.rho_delta = 0.04
        self.rho_w = 1025.19
        self.H0 = 10.4
        self.v_s = 0.3
        self.g = 9.81
        self.d0 = 0.8

        self.I_s = 180.0 * 0.217
        self.T_opt = 10.0
        self.T_min = 0.5
        self.T_max = 20.0
        self.beta = 0.001

        self.data_dir = data_dir
        self.year_data_dict = {}
        self.current_step = 0
        self.current_state = None
        self.env_data = None
        self.current_year = None

    def load_year_data(self, year: int):
        
        file_path = os.path.join(self.data_dir, f"{year}.xlsx")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_excel(file_path, header=0, index_col=0, engine="openpyxl").reset_index(drop=True)
            df = df.astype(float)
            if len(df) > 0 and df.shape[1] >= 3:
                df.iloc[:, 2] = df.iloc[:, 2] / 100
            df = df.head(self.total_time).ffill().bfill()
            if len(df) < self.total_time:
                last_row = df.iloc[-1] if len(df) > 0 else pd.Series([0, 0, 0, 0])
                df = pd.concat([df, pd.DataFrame([last_row] * (self.total_time - len(df)))], ignore_index=True)
            df.columns = ["temperature", "light", "tide_height", "current_speed"]
            valid_mask = (df.notna().all(axis=1)) & (df["light"] >= 0)
            df = df[valid_mask].reindex(range(self.total_time), method='ffill')
            self.year_data_dict[year] = df.values
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def reset(self, year: int):
        
        if year not in self.year_data_dict:
            self.load_year_data(year)
        self.current_year = year
        self.env_data = self.year_data_dict[year]
        self.current_step = 0

        init_E = np.random.uniform(self.E_min, self.E_max)
        init_temp = self.env_data[0, 0]
        init_light = self.env_data[0, 1]
        init_tide = self.env_data[0, 2]
        init_current = self.env_data[0, 3]

        self.current_state = np.array([init_E, init_temp, init_light, init_tide, init_current, 0], dtype=np.float32)
        return self.current_state

    def get_solar_power(self, I_t: float) -> float:
        
        H_t = I_t * self.dt
        g_t = (H_t * self.P_AZ * self.K) / 1e6
        return np.clip(g_t, 0.0, self.P_AZ / 1000 * self.K)

    def calc_Vt(self, m_t: int, u_c_t: float, u_next: float, tide: float) -> float:
        
        if m_t == 0:
            return 0.0

        u_safe = max(abs(u_c_t), 1e-8)
        u_next_safe = max(abs(u_next), 1e-8)

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
        Q0 = 3.14 * self.d0 ** 2 * np.sqrt(u_c_t ** 2 + v0 ** 2)
        rho_d = ((Qd - Q0) * self.rho_w + Q0 * (self.rho_w + self.rho_delta)) / Qd
        vmd = vd * np.sqrt(2 * 3.14) / 6
        jieta = rho_d / (rho_d - self.rho_w)
        beta = 0.17
        Zm = np.sqrt((Bd / beta) ** 2 + 4 / 3 * jieta * Bd * vmd ** 2 / (beta * self.g)) - Bd / beta + Zd
        xd = self.x_d if u_c_t > 0 else 120 - self.x_d

        if Zm < self.Z_s + tide:
            return 0.0
        if Zd > self.Z_s + tide:
            xs = (((self.Z_s + tide + delta_z) ** (4 / 3) - delta_z ** (4 / 3)) * 3 / 4 / A) * u_safe
        else:
            term = 1 - 3 * beta * self.g / (4 * jieta * Bd * vmd ** 2) * (self.Z_s + tide - Zd) * (
                    self.Z_s + tide + 2 * Bd / beta - Zd)
            xs = jieta * vmd * u_safe / self.g * (1 - term ** (2 / 3)) + xd1
        if xs >= xd:
            return 0.0
        xaq = 120 if ((u_next > 0 and u_c_t < 0) or (u_next < 0 and u_c_t > 0)) else xd
        kata = ((xd - xs) / (u_safe * 3600)) ** 2 * (
                (u_safe * 3600 - xd) / (xd - xs) + u_safe / u_next_safe * ((xaq - xd) / (xd - xs) + 0.5) + 0.5
        )
        Vavg = kata * Q0 * self.dt * 3600
        return m_t * Vavg

    def calc_f2(self, T_t: float, I_t: float) -> float:
        
        I_illu = I_t * 0.168
        f_I = min(I_illu / self.I_s, 2.0)

        T_x = self.T_min if T_t <= self.T_opt else self.T_max
        if abs(T_x - self.T_opt) < 1e-6:
            f_T = 1.0
        else:
            temp_term = 2.3 * ((T_t - self.T_opt) / (T_x - self.T_opt)) ** 2
            f_T = np.exp(1 - f_I - temp_term)

        return max(f_I * f_T, 0.0)

    def step(self, action: int):
        
        E_t = self.current_state[0]
        T_t = self.env_data[self.current_step, 0]
        I_t = self.env_data[self.current_step, 1]
        Z_t = self.env_data[self.current_step, 2]
        u_t = self.env_data[self.current_step, 3]

        self.current_step += 1
        u_next = self.env_data[self.current_step, 3] if self.current_step < self.total_time - 1 else u_t

        g_t = self.get_solar_power(I_t)
        q_t = action * self.p0
        total_load = self.b + q_t
        energy_surplus = g_t - total_load

        reward = 0.0
        c_t, d_t, flag = 0.0, 0.0, 1
        if energy_surplus > 0:
            c_t = min(energy_surplus, self.c_max, (self.E_max - E_t) / (self.eta_c * self.dt))
            E_next = E_t + self.eta_c * c_t * self.dt
        else:
            required_discharge = -energy_surplus * self.dt / self.eta_d
            if E_t - required_discharge >= self.E_min:
                E_next = E_t - required_discharge
            else:
                E_next = self.E_min
                reward -= 1
                flag = 0

        E_next = np.clip(E_next, self.E_min, self.E_max)

        Vt = self.calc_Vt(action, u_t, u_next, Z_t)
        f2_t = self.calc_f2(T_t, I_t)

        if flag == 1:
            reward += self.beta * f2_t * Vt if (action == 0 or (Vt >= 1e-6 and f2_t >= 1e-6)) else reward - 1

        energy_waste = max(energy_surplus - c_t, 0.0)
        if energy_waste > 20:
            reward -= 0.01

        self.current_state = np.array([E_next, T_t, I_t, Z_t, u_t, self.current_step], dtype=np.float32)

        done = self.current_step >= self.total_time - 1

        info = {
            "Vt": Vt, "f2_t": f2_t, "energy_waste": energy_waste,
            "solar_power": g_t, "total_load": total_load, "ESS": E_next
        }
        return self.current_state, reward, done, info

# ------------------------------------------------------
# ------------------------------------------------------
def run_rule_based_system(data_dir: str, target_year: int):
    
    results_dir = os.path.join(_RESULT_DIR, "rule_based")
    os.makedirs(results_dir, exist_ok=True)

    env = AUS_Environment(data_dir)
    env.reset(year=target_year)

    F2_THRESHOLD = 0.2
    VT_THRESHOLD = 0
    ENERGY_RATIO = 0.2

    log_data = []
    total_nti = 0.0
    total_energy_waste = 0.0
    total_operation_hours = 0
    total_reward = 0.0

    print(f"Starting rule-based operation (year: {target_year})...")
    print(f"Rule: turn on 16 compressors when f2_t > {F2_THRESHOLD}, Vt > {VT_THRESHOLD} and energy sufficiency >= {ENERGY_RATIO * 100}%")

    current_state = env.current_state
    done = False

    while not done:
        step = env.current_step

        E_t = current_state[0]
        T_t = env.env_data[step, 0]
        I_t = env.env_data[step, 1]
        Z_t = env.env_data[step, 2]
        u_t = env.env_data[step, 3]
        u_next = env.env_data[step + 1, 3] if step + 1 < env.total_time else u_t

        f2_t = env.calc_f2(T_t, I_t)
        solar_power = env.get_solar_power(I_t)
        total_load = env.b + env.M * env.p0

        max_discharge_power = min(env.d_max, (E_t - env.E_min) * env.eta_d / env.dt)  # kW
        available_energy = solar_power + max_discharge_power

        potential_vt = env.calc_Vt(env.M, u_t, u_next, Z_t)

        energy_sufficient = available_energy >= ENERGY_RATIO * total_load
        if energy_sufficient and f2_t > F2_THRESHOLD:
            action = env.M
            total_operation_hours += 1
        else:
            action = 0

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        total_nti += info["f2_t"] * info["Vt"]
        total_energy_waste += info["energy_waste"]

        if step % 1 == 0 or step == env.total_time - 1:
            log_data.append({
                "timestep_h": step,
                "battery_kWh": round(E_t, 2),
                "temperature_C": round(T_t, 2),
                "light": round(I_t, 2),
                "tide_height_m": round(Z_t, 2),
                "current_speed": round(u_t, 2),
                "compressors_on": action,
                "f_temp": round(f2_t, 4),
                "upwelled_volume_m3": round(info["Vt"], 2),
                "solar_power_kW": round(solar_power, 2),
                "available_power_kW": round(available_energy, 2),
                "total_load_kW": round(total_load, 2),
                "energy_wasted_kWh": round(info["energy_waste"], 2),
                "instant_reward": round(reward, 4)
            })

        current_state = next_state

        if step % 500 == 0:
            print(f"Progress: {step}/{env.total_time} h ({step / env.total_time * 100:.1f}%)")

    log_df = pd.DataFrame(log_data)
    log_path = os.path.join(results_dir, f"run_log_{target_year}.xlsx")
    log_df.to_excel(log_path, index=False, engine="openpyxl")

    print("\n" + "=" * 60)
    print(f"Rule-based operation report (year: {target_year})")
    print("=" * 60)
    print(f"Total operating time: {total_operation_hours} h ({total_operation_hours / env.total_time * 100:.2f}%)")
    print(f"Total Nutrient Transport Index (NTI): {total_nti:.2f}")
    print(f"Total energy wasted: {total_energy_waste:.2f} kWh")
    print(f"Average hourly energy waste: {total_energy_waste / env.total_time:.2f} kWh")
    print(f"Episode total reward: {total_reward:.4f}")
    print("=" * 60)

    print(f"\nResults saved to: {results_dir}")
    return {
        "total_NTI": total_nti,
        "total_energy_wasted_kWh": total_energy_waste,
        "runtime_h": total_operation_hours,
        "episode_total_reward": total_reward,
        "log_path": log_path
    }

# ------------------------------------------------------
# ------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = _DATA_DIR
    TARGET_YEAR = 2025

    required_file = f"{TARGET_YEAR}.xlsx"
    if not os.path.exists(os.path.join(DATA_DIR, required_file)):
        raise FileNotFoundError(f"Required data file not found: {os.path.join(DATA_DIR, required_file)}")

    run_results = run_rule_based_system(
        data_dir=DATA_DIR,
        target_year=TARGET_YEAR
    )