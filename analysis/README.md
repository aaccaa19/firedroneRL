Analysis plots
=================

This folder contains `plot_all_figures.py` which generates a set of diagnostic plots and simple diagrams
useful for explaining and presenting the RL training and agent behavior.

Usage
-----

1. Place your training logs in `logs/training_metrics.csv` and per-step logs in `logs/steps.csv`.
   - `training_metrics.csv` should contain columns: episode,total_reward,length,success,collisions,goal_dist
   - `steps.csv` should contain columns: episode,step,x,y,z,action0,action1,action2,reward,td_error,rule_override

2. Run the script from the repository root:

```bash
python analysis/plot_all_figures.py
```

3. PNGs will be written to `outputs/plots/`.

If log files are missing the script will synthesize example data so you can preview the plots immediately.

Files generated
---------------

- 01_training_curve.png
- 02_length_and_success.png
- 03_losses.png
- 04_td_error_hist.png
- 05_action_distributions.png
- 06_state_visitation.png
- 07_trajectories.png
- 08_reward_components.png
- 09_action_vs_state_slice.png
- 10_q_value_heatmap.png
- 11_rule_vs_rl_timeline.png
- 12_decision_region.png
- A_agent_environment_loop.png
- B_system_architecture.png
- C_sequence_diagram.png
- D_nn_architecture.png
- E_mdp_schematic.png
- F_pipeline_diagram.png
