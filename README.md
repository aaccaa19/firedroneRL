# FiredroneRL — Drone exploration and fire-avoidance (UOAR)

This repository contains an environment, agent, training loop and analysis tools for Unmanned Online Area Reconnaissance (UOAR) with drones. The project includes a Gym-compatible environment, a TD3-like continuous-action agent, logging, plotting utilities, and small helper scripts.

## Contents

- `drl-based-drone/UOAR_drone_rl.py` — Main environment (`DroneEnv`), agent (`TD3Agent`), training loop and utilities.
- `analysis/plot_all_figures.py` — Plotting/analysis utilities for training logs and decision maps.
- `logs/` — Default location for runtime logs (CSV + exported artifacts).
- `outputs/plots/` — Generated PNGs from the analysis scripts.
- `wrapper/` — High-level API wrapper for programmatic access.

## Installation

### From Source (Recommended)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/aaccaa19/firedroneRL.git
cd firedroneRL
pip install -e .
```

### Standard Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- Python ≥3.8
- gymnasium
- numpy
- torch
- matplotlib
- pandas
- seaborn

## Quick Start

### Interactive Training

Run the main training loop with interactive prompts:

```bash
python drl-based-drone/UOAR_drone_rl.py
```

The script will prompt you to:
- Choose ablation mode (automated parameter sweep) or standard training
- Select curriculum episodes and max steps per episode
- Configure which reward components to enable
- Select training scenarios (Static Fire Line, Spreading Fire, etc.)
- Save trained agents and generate analysis plots

### Non-Interactive Smoke Run

Run 10 episodes quickly without prompts (useful for testing):

```bash
# Linux/Mac
SMOKE_RUN=10 python drl-based-drone/UOAR_drone_rl.py

# Windows PowerShell
$env:SMOKE_RUN=10; python drl-based-drone/UOAR_drone_rl.py
```

### Using as a Python Package

Import and use the environment and agent directly in your code:

```python
from drl_based_drone.UOAR_drone_rl import DroneEnv, TD3Agent

# Create environment
env = DroneEnv(curriculum_level=0, scenario=1, area_size=20)
obs, _ = env.reset()

# Create agent
agent = TD3Agent(
    area_size=env.area_size,
    drone_radius=env.drone_radius,
    fire_line=env.fire_line,
    fire_radius=env.fire_radius,
    safety_margin=env.safety_margin,
    max_step_size=env.max_step_size,
    obs_dim=env.observation_space.shape[1],
    act_dim=env.action_space.shape[1],
    x_max=env.x_max,
    y_max=env.y_max,
    z_min=env.z_min,
    z_max=env.z_max
)

# Run episode
done = [False, False]
steps = 0
while not all(done) and steps < 300:
    actions = [agent.select_action(obs[i], deterministic=False) for i in range(env.n_drones)]
    obs, rewards, done, _, info = env.step(actions)
    steps += 1

env.close()
```

### Command-Line Entry Point

After installation, use the command-line entry point:

```bash
firedrone-rl
```

This runs the main interactive training script.

## Training Configuration

The training loop supports multiple scenarios and curriculum learning:

### Scenarios

1. **Static Fire Line** — Fixed fire line on the map
2. **Spreading Fire** — Fire spawns dynamically during episode
3. **Line of Fires** — Multiple fires arranged in a line
4. **Random Static Fires** — 5 randomly placed static fires
5. **Impassable Fire Wall** — Fire wall with random crash risk
6. **Central Fire** — Large central fire obstruction

### Curriculum Learning

Training uses 5 curriculum levels with progressively tighter safety margins (larger = easier):
- Level 0: Most difficult (smallest safety margin)
- Level 4: Easiest (largest safety margin)

Configure episodes per curriculum level via interactive prompt.

### Reward Components

Enable/disable components to study their impact:
- Proximity rewards & penalties
- Exploration cell rewards
- Per-step energy penalty
- Edge/wall penalty
- Movement bonus
- Stationary penalty
- Fire discovery bonus
- Lateral/patrol bonus
- Optimal proximity bonus
- Cell reward regeneration

## Output Files

All outputs are saved in `logs/` and `outputs/plots/`:

### Training Logs (CSV format)

**training_metrics.csv** — Per-episode statistics:
- `episode`, `total_reward`, `length`, `success`, `collisions`
- `avg_goal_dist`, `fires_discovered`, `fires_viewed`, `boxes_explored`
- Reward components: `proximity_component`, `exploration_component`, `energy_component`
- Normalized values: `norm_prox_mean`, `norm_expl_mean`, `norm_energy_mean`
- Network diagnostics: `actor_loss`, `critic_loss`, `td_error`, `action_saturation_fraction`

**steps.csv** — Per-step detailed data:
- `episode`, `step`, `x`, `y`, `z` — Drone position
- `action0`, `action1`, `action2` — Actions taken
- `reward`, `normalized_reward` — Rewards
- `exploration_reward`, `proximity_reward`, `energy` — Component breakdown
- `td_error`, `action_saturation`, `rule_override`

**updates.csv** — Agent training update diagnostics:
- `episode`, `step`, `agent_id`, `total_it`
- `actor_loss`, `critic_loss`, `td_error`
- `q_min`, `q_max`, `normalized_reward`
- `actor_grad_norm`, `critic_grad_norm`, `replay_fill`

**eval_metrics.csv** — Periodic deterministic policy evaluations (every 5 episodes):
- `episode`, `scenario`, `curriculum_level`
- `eval_steps`, `eval_total_reward`, `eval_avg_reward`, `eval_success`

### Decision Map

**decision_map.npy** — Spatial grid of learned policy:
- NumPy dict with keys: `xs`, `ys`, `grid`
- `grid` contains deterministic actor actions for each (x,y) position
- Useful for visualization and analysis of learned behavior

### Generated Plots (PNG)

Run analysis to generate publication-ready plots:

```bash
python analysis/plot_all_figures.py
```

Outputs include:
- `training_curve.png` — Reward over episodes
- `length_and_success.png` — Episode length and success rate
- `actor_loss.png`, `critic_loss.png` — Network loss curves
- `td_error_hist.png` — TD-error distribution
- `state_visitation.png` — Heatmap of visited states
- `trajectories.png` — 10 sample drone trajectories evenly distributed through training
- `avg_fire_distance.png` — Distance to fire over episodes
- `q_value_heatmap.png` — Learned Q-value surface
- `decision_region.png` — Policy decision regions

## Agent Saving and Loading

Agents are saved as pickle files with format: `td3_agent_s{scenario}_agent_{id}.pkl`

### Save Agent

Agents are auto-saved after training or via interactive prompt. The saved checkpoint includes:
- Actor and critic network weights
- Replay buffer state
- Training configuration (learning rates, noise levels, etc.)

### Load Agent

Agents are automatically loaded if matching checkpoint files exist:

```python
agent = TD3Agent(...)
agent.load('td3_agent_s1_agent_0.pkl')
```

The loader is defensive — it skips incompatible parameters and reports warnings.

## Environment Details

### Observation Space

Per-drone observation (10 values):
```
[x, y, z, fire_below, other_x, other_y, other_z, fire_x, fire_y, fire_z]
```

- `x, y, z` — Drone position
- `fire_below` — Boolean: fire directly below (within radius)
- `other_x, other_y, other_z` — Other drone's position
- `fire_x, fire_y, fire_z` — Last seen fire location

### Action Space

Per-drone action (3 continuous values, normalized to [-1, 1]):
```
[delta_x, delta_y, delta_z]
```

Actions are scaled by `max_step_size` and clipped to environment boundaries.

### Constraints

- Area: 20 × 10 units
- Altitude: 5–10 units (z)
- Number of drones: 2
- Fire radius: 0.25–0.5 units (scenario dependent)
- Drone radius: 0.25 units

## Analysis

### Plotting Analysis

Generate diagnostic plots from training logs:

```bash
python analysis/plot_all_figures.py
```

If log files don't exist, synthetic example data is generated for demonstration.

### Smoke Actor Check

Validate network initialization and diagnostics:

```bash
python analysis/smoke_actor_check.py
```

Prints:
- Q-value statistics (mean, std)
- Actor loss and critic statistics
- Parameter norms and learning rates
- Policy delay and gradient clipping settings

## Rule-Based Baseline

A simple rule-based agent is provided for baseline comparison:

```python
from rule_based_drone.rulebaseddrone import RuleBasedDrone

drone = RuleBasedDrone()
action = drone.compute_action(obs)
```

The rule-based agent uses heuristics like:
- Move toward unexplored cells
- Maintain safe distance from fire
- Avoid edge collisions

## Troubleshooting

### Shape Mismatch on Decision Map Save

If you see `AttributeError: 'TD3Agent' object has no attribute 'z_min'`:
- Ensure all agent instantiations pass `x_max`, `y_max`, `z_min`, `z_max` from the environment
- Update to the latest version with boundary parameter support

### Missing Log Files

The analysis script will synthesize example data if logs are missing. To generate real logs:
1. Run training: `python drl-based-drone/UOAR_drone_rl.py`
2. Accept prompts and complete at least one episode
3. Run analysis: `python analysis/plot_all_figures.py`

### Out of Memory

If training runs out of VRAM:
- Reduce `batch_size` in `TD3Agent.__init__()` (default 256)
- Reduce `buffer_size` (default 200000)
- Reduce curriculum episodes per level

## Recent Changes

- Observations now include last-seen fire location (`fire_x, fire_y, fire_z`)
- Observation shape is now standardized to 10 values per drone
- Agent initialization requires boundary parameters (`x_max`, `y_max`, `z_min`, `z_max`)
- Decision map export handles flexible observation dimensions
- Trajectory visualization now plots 10 evenly-distributed samples instead of first 20 episodes

## Contributing

This project is part of academic research on multi-agent RL for wildfire response. Please cite appropriately if using this code.

## License

See LICENSE file for details.

## References

- TD3: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (ICML 2018)
- Gymnasium: https://gymnasium.farama.org/

