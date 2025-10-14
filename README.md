````markdown
# FiredroneRL — Drone exploration and fire-avoidance (UOAR)

This repository contains an environment, agent, training loop and analysis tools for Unmanned Online Area Reconnaissance (UOAR) with drones. The project includes a Gym-compatible environment, a TD3-like continuous-action agent, logging, plotting utilities, and small helper scripts.

Contents
--------
- `UOAR-based-drone/UOAR_drone_rl.py` — Main environment (`DroneEnv`), agent (`TD3Agent`), training loop and utilities.
- `analysis/plot_all_figures.py` — Plotting/analysis utilities for training logs and decision maps.
- `logs/` — Default location for runtime logs (CSV + exported artifacts).
- `outputs/plots/` — Generated PNGs from the analysis scripts.

Quickstart
----------
1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run the training/interactive loop:

```powershell
python UOAR-based-drone/UOAR_drone_rl.py
```

The script is interactive by default and will prompt you to choose scenarios, save agents, and optionally run analysis plots.

Non-interactive smoke run example:

```powershell
$env:SMOKE_RUN=10; python UOAR-based-drone/UOAR_drone_rl.py
```

What changed recently
---------------------
- Observations now include the last-seen fire location for each drone. Observation shape per drone is now 10 values: `[x, y, z, fire_below, other_x, other_y, other_z, fire_x, fire_y, fire_z]`.
- The export decision-map code was updated to construct observations that match the current env observation size. If you see a shape mismatch error while saving decision maps (e.g., earlier 7 vs 10), update custom scripts to use the env's `observation_space`.

Saving and decision maps
------------------------
- When you save agents, the run will attempt to export a decision map for agent 0. The exporter builds a grid of deterministic actor outputs over (x,y) positions and stores it as a NumPy `.npy` file containing `{'xs': xs, 'ys': ys, 'grid': grid}`.
- If exporting fails due to dimension mismatch, ensure the agent's actor expects the same observation length as `env.observation_space.shape[1]`.

Analysis
--------
Run the analysis script to generate plots from `logs/`:

```powershell
python analysis/plot_all_figures.py
```

This script reads `logs/training_metrics.csv` and `logs/steps.csv` and writes PNGs to `outputs/plots/`.

Cleaning up docs
----------------
- Duplicate README files were consolidated into this single root README. If you maintain separate documentation for analysis, reference this README and link the `analysis/` directory.

Troubleshooting
---------------
- Error: "could not broadcast input array from shape (10,) into shape (7,)"
	- Cause: code was still assuming the old 7-dim observation format. Fix: update any code constructing observations (decision-map or tests) to use the env's `observation_space.shape[1]`.

- Error while exporting decision map: "mat1 and mat2 shapes cannot be multiplied"
	- Cause: actor network expected a different input dimension than provided observation. Fix: rebuild actor with correct `obs_dim` or adapt the decision map to feed the actor a matching sized vector.

Contact / Contributing
----------------------
Open issues or PRs on the repository. For quick fixes or bug reports, include the traceback and the line that constructs observations or initializes agents.

````