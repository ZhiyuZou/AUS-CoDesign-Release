import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# === Auto-added path setup (project reorganization) ===
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PROJECT_ROOT, "data")
_MODEL_DIR = _os.path.join(_PROJECT_ROOT, "models")
_RESULT_DIR = _os.path.join(_PROJECT_ROOT, "results")
# === End path setup ===

DATA_DIR = _DATA_DIR
SOURCE_FILE = "2025.xlsx"
NOISE_LEVELS = [0.10, 0.20, 0.30]
PLOT_STEPS = 200
NOISE_ALPHA = 0.3
# =========================================

ENV_VARIABLES = {
    0: "Temperature (°C)",
    1: "Light Intensity",
    2: "Tidal Height",
    3: "Current Velocity"
}

def generate_noisy_datasets():
    source_path = os.path.join(_DATA_DIR, SOURCE_FILE)
    if not os.path.exists(source_path):
        print(f"Error: source file not found {source_path}")
        return None

    print(f"Reading source file: {source_path} ...")
    df_source = pd.read_excel(source_path, header=0, index_col=0).astype(float)

    global_std = df_source.std(axis=0).values
    print("\nGlobal standard deviation of each variable:")
    for col_idx, var_name in ENV_VARIABLES.items():
        print(f"  {var_name}: {global_std[col_idx]:.4f}")

    comparison_data = {}
    for col_idx, var_name in ENV_VARIABLES.items():
        comparison_data[var_name] = {
            'Original': df_source.iloc[:, col_idx].values[:PLOT_STEPS]
        }

    for level in NOISE_LEVELS:
        print(f"\n--- Processing {int(level * 100)}% noise data ---")
        df_noisy = df_source.copy()

        for col_idx in range(df_source.shape[1]):
            relative_noise = np.random.normal(0.0, level, size=len(df_noisy))
            relative_component = df_noisy.iloc[:, col_idx] * relative_noise

            absolute_noise = np.random.normal(0.0, level, size=len(df_noisy))
            absolute_component = global_std[col_idx] * absolute_noise

            total_noise = (NOISE_ALPHA * relative_component +
                           (1 - NOISE_ALPHA) * absolute_component)

            df_noisy.iloc[:, col_idx] += total_noise

        df_noisy.iloc[:, 0] = df_noisy.iloc[:, 0].clip(lower=0)
        df_noisy.iloc[:, 1] = df_noisy.iloc[:, 1].clip(lower=0)

        percentage_str = f"{int(level * 100)}%"
        new_filename = f"2025-{percentage_str}.xlsx"
        save_path = os.path.join(_DATA_DIR, new_filename)
        df_noisy.index.name = "datetime"
        df_noisy.to_excel(save_path, header=True, index=True)
        print(f"Saved: {new_filename}")

        for col_idx, var_name in ENV_VARIABLES.items():
            comparison_data[var_name][f'{percentage_str} Noise'] = df_noisy.iloc[:, col_idx].values[:PLOT_STEPS]

    print("\nAll files generated.")
    return comparison_data, global_std

def plot_all_comparisons(data_dict, global_std):
    
    n_vars = len(ENV_VARIABLES)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars), sharex=True)
    fig.suptitle(f'Improved Mixed Noise Model (α={NOISE_ALPHA}) - First {PLOT_STEPS} Steps',
                 fontsize=16, y=0.95)

    x = range(PLOT_STEPS)
    colors = {'Original': 'k-', '10% Noise': 'b--', '20% Noise': 'g-.', '30% Noise': 'r:'}
    linewidths = {'Original': 2, '10% Noise': 1.5, '20% Noise': 1.5, '30% Noise': 1.5}
    alphas = {'Original': 1.0, '10% Noise': 0.7, '20% Noise': 0.7, '30% Noise': 0.7}

    for i, (var_name, var_data) in enumerate(data_dict.items()):
        ax = axes[i]

        for label, values in var_data.items():
            ax.plot(x, values, colors[label],
                    linewidth=linewidths[label],
                    alpha=alphas[label],
                    label=label)

        ax.set_ylabel(f"{var_name}\n(Global Std: {global_std[i]:.3f})")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        if var_name == "Current Velocity":
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.text(0.02, 0.95, 'Zero-crossing region', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

if __name__ == "__main__":
    np.random.seed(99)
    comp_data, global_std = generate_noisy_datasets()

    if comp_data:
        plot_all_comparisons(comp_data, global_std)