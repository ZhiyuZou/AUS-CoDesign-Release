# Intelligent Co-design and Operation Optimization of Artificial Upwelling Systems — Code & Data Release

This repository contains the single-process, fully reproducible source code, the trained
models and the processed environmental data for the co-design (hardware sizing +
operational control) framework of an off-grid, solar/battery-powered artificial upwelling
system (AUS) for seaweed mariculture in Aoshan Bay.

All scripts resolve `data/`, `models/` and `results/` relative to the repository root, so the
folder can be copied anywhere and run without editing paths. **All multiprocessing / parallel
execution has been removed; every script runs sequentially in a single process.**

## 1. Repository layout

| Path | Role | Manuscript element |
|---|---|---|
| `training/train_standard.py` | Dueling QR-DQN baseline agent (no mask, no FiLM) | Sec. 3.2, ablation "Standard" |
| `training/train_masking.py` | QR-DQN + invalid-action masking | ablation "Masking" |
| `training/train_film.py` | QR-DQN + FiLM hardware conditioning (no mask) | ablation "FiLM" |
| `training/train_proposed.py` | QR-DQN + FiLM + invalid-action masking (proposed) | proposed controller |
| `baselines/rule_based.py` | Rule-based heuristic controller (RBH) | Sec. 3.3 baseline |
| `baselines/mpc.py` | Model predictive control, horizon H=24, maxiter=100 | Sec. 3.3 baseline |
| `baselines/dp.py` | Perfect-foresight dynamic program, single state (battery energy), NS=200 | Sec. 3.3 upper bound |
| `optimization/mopso_*.py` | Multi-objective PSO outer loop (standard/film/drl/mpc/dp evaluator) | Sec. 3.4 / Fig. 9 |
| `utils/plume_calculator.py` | Bubble-plume / effective-volume (`z_m`, `b_m`, `Q_t`, `kata`) calculator | Sec. 3.1 / App. A |
| `utils/data_noise.py` | Generate noisy observation datasets | robustness inputs |
| `utils/complexity.py`, `utils/find_conditions.py` | Complexity illustration and plume-condition finder (stand-alone plotters) | App. / Fig. |
| `data/` | Processed hourly environmental data 2021–2025 and noisy 2025 variants | inputs |
| `models/` | Four trained network weights (`*.pth`) | trained agents |
| `results/` | Empty output folder; training logs and optimization results are written here | outputs |

## 2. Environment

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Tested with Python 3.12.7 on Windows; package versions are pinned in `requirements.txt`.
A CUDA GPU is used automatically when available, but all scripts also run on CPU.

## 3. Reproduction workflow

Run each script from its own folder (paths are resolved automatically).

1. **Train the four controllers** — each `training/train_*.py` loops over seeds 0–4
   (five independent replicates of 5000 episodes each) and writes per-seed weights to
   `models/`, with training logs written to `results/`. To reproduce the released single
   weights, train with seed 0 and rename the output to `{agent}_agent.pth`.
2. **Outer-loop hardware optimization** — run any `optimization/mopso_*.py`. MOPSO uses
   50 particles, 100 iterations, inertia weight w=0.7, c1=c2=1.5.
3. **Baseline controllers** — `baselines/rule_based.py`, `baselines/mpc.py` and
   `baselines/dp.py` implement the three non-learned benchmarks used for comparison.

## 4. Key implementation settings (matching the manuscript)

- **Reward / NTI.** The reward is the biological response `f(I)·f(T)` multiplied by the
  transported plume volume, plus a fixed idle penalty of -0.1 and reward-scaling factor
  beta=0.1. NTI is retained as the operational objective because ambient nutrient
  concentration and internal nutrient quota are not observable in real time.
- **Stratified experience replay.** A continuous hardware configuration
  `(P_AZ, E_max, M, Q0)` is min–max normalized and mapped to one of 32 configuration groups
  by a hash signature; every mini-batch (batch size 256) draws 8 transitions from each group
  (shortfalls back-filled from the global buffer), spanning the whole design space per update.
- **Network / optimization.** Adam, lr=2.5e-4; gamma=0.99; 200 quantiles; target hard update
  every 1e4 steps; epsilon 1.0→0.01 with decay constant 2e5; replay capacity 2e6;
  hidden width 512; one learning update per environment step; 5000 episodes of 2880 hourly
  steps (~14.4e6 steps).
- **DP.** Perfect foresight over a single internal state (battery state of energy) discretized
  into NS=200 grid points with linear interpolation; complexity O(T·NS·NA·c_phys), linear in
  grid resolution.
- **MPC.** Receding horizon H=24 h, solver maxiter=100.
- **Communication delay is not modeled**: the controller acts hourly while field telemetry
  latency (seconds–minutes) is two orders of magnitude shorter and state changes over that
  scale are negligible; this effect is folded into observation noise.

## 5. Data

Every workbook in `data/` is a labeled hourly time series: the first column is the row label
`datetime` and the first row gives the column names. Each file holds 2880 rows, i.e. 120
operational days from 1 Feb 00:00 to 31 May 23:00 of the indicated year (for leap year 2024
the 2880 operational hours end on 30 May).

| Column | Meaning | Unit |
|---|---|---|
| `datetime` | Hour timestamp (row label) | yyyy-mm-dd HH:00 |
| `temperature_C` | Sea-surface temperature | °C |
| `irradiance_W_m2` | Surface solar irradiance (the "light" input) | W/m² |
| `tide_height_cm` | Tidal level; the loaders divide by 100 to obtain metres | cm |
| `current_speed_m_s` | Tidal current velocity | m/s |

`data/2021.xlsx`–`2025.xlsx` are the five yearly series; `data/2025-{5,10,20,30}%.xlsx` and the
corresponding `*_mixed_noise.xlsx` files are observation-uncertainty variants of the 2025
series. On loading, the `datetime` column is used as the row label and then dropped
(`reset_index`), so every controller receives the same positional four-column array; missing
entries (two trailing temperature samples) are forward/backward filled, and the tidal level is
converted from cm to m, exactly as in the manuscript.

## 6. Notes and scope

- `results/` is intentionally empty; re-running a script writes its outputs there.
- All randomness is seeded: training replicates use seeds 0–4, and the fixed
  configuration set uses seed 42.
