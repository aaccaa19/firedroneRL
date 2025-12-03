"""Standalone test-runner for section-2 (final/test scenario).
This script imports the environment and agent classes from the main training script
(`UOAR_drone_rl.py`) by loading the file as a module at runtime. It then loads
pickled agent files and runs deterministic test episodes (section 2 behavior).

Usage: python final_environment.py --scenario 2 --episodes 1 --steps 300
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt



def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description='Run final test scenario using pickled agents')
    parser.add_argument('--scenario', type=int, default=2, help='Scenario number to test (default: 2)')
    parser.add_argument('--episodes', type=int, default=1, help='Episodes per curriculum level to run')
    parser.add_argument('--steps', type=int, default=300, help='Max steps per episode')
    parser.add_argument('--agent-pattern', type=str, default='td3_agent_s{scenario}_agent_{i}.pkl', help='Pattern for agent pickle filenames')
    parser.add_argument('--pickles-dir', type=str, default='.', help='Directory where agent pickles are stored')
    args = parser.parse_args()

    workspace = Path(__file__).parent
    uoa_path = workspace / 'UOAR_drone_rl.py'
    if not uoa_path.exists():
        print(f"Error: expected {uoa_path} to exist (source of DroneEnv/TD3Agent).")
        return 1

    # Load the training script as a module to reuse DroneEnv and TD3Agent classes
    uoar = load_module_from_path(uoa_path, 'uoa_main')

    # Import helper spawn functions from the earlier-created helper file (if present)
    helper_path = workspace / 'final_enviroment.py'
    fe = None
    if helper_path.exists():
        fe = load_module_from_path(helper_path, 'fe_helpers')

    Scenario = args.scenario

    # Keep storage for the final episode's states so we can visualize once after all runs
    last_episode_states = None
    last_episode_fire_centers = None

    # Offer an interactive scenario choice to the user (console prompt)
    # Prefer the canonical scenario names from the training script to ensure exact mapping.
    scenario_names = getattr(uoar, 'scenario_names', None)
    if not isinstance(scenario_names, dict):
        scenario_names = {1: "Static Fire Line", 2: "Spreading Fire", 3: "Line of Fires", 4: "Random Fires", 5: "Impassable Fire Wall (w/ Random Crash)", 6: "Central Fire"}
    print("Available scenarios:")
    for k in sorted(scenario_names.keys()):
        print(f"  {k}: {scenario_names[k]}")
    user_input = input(f"Enter scenario number to run (default {args.scenario}): ").strip()
    if user_input and user_input.lstrip('+-').isdigit():
        si = int(user_input)
        if si in scenario_names:
            args.scenario = si
        else:
            print(f"Invalid scenario '{si}' - using default {args.scenario}.")

    # Prompt for episodes and steps interactively (allow defaults from CLI)
    ep_input = input(f"Enter number of episodes per curriculum level (default {args.episodes}): ").strip()
    if ep_input:
        if ep_input.lstrip('+-').isdigit():
            args.episodes = max(1, int(ep_input))
        else:
            print(f"Invalid episodes '{ep_input}', keeping default {args.episodes}.")
    steps_input = input(f"Enter max steps per episode (default {args.steps}): ").strip()
    if steps_input:
        if steps_input.lstrip('+-').isdigit():
            args.steps = max(1, int(steps_input))
        else:
            print(f"Invalid steps '{steps_input}', keeping default {args.steps}.")

    # Ensure the chosen scenario value (possibly updated by the interactive prompt)
    # is used when loading pickles below.
    Scenario = args.scenario

    # Create a temporary environment to infer observation/action dims and drone count,
    # then create agents once and load their pickles a single time up-front.
    # If a later environment uses a different drone count we will recreate agents for that env.
    temp_env = uoar.DroneEnv(curriculum_level=0, scenario=Scenario)
    obs_dim = temp_env.observation_space.shape[1]
    act_dim = temp_env.action_space.shape[1]
    n_drones = temp_env.n_drones

    # Create agents once using the inferred dimensions
    agents = [uoar.TD3Agent(temp_env.area_size, temp_env.drone_radius, temp_env.fire_line, temp_env.fire_radius, temp_env.safety_margin, temp_env.max_step_size, obs_dim=obs_dim, act_dim=act_dim) for _ in range(n_drones)]

    # Load pickles for agents once at startup (skip empty/corrupted files)
    pickles_dir = Path(args.pickles_dir)
    for i in range(len(agents)):
        fname = pickles_dir / args.agent_pattern.format(scenario=Scenario, i=i)
        if fname.exists() and fname.stat().st_size > 0:
            try:
                agents[i].load(str(fname))
                print(f"Loaded agent {i} from {fname}")
            except EOFError:
                print(f"Warning: pickle for agent {i} appears truncated (EOF) - skipping: {fname}")
            except Exception as e:
                print(f"Warning: failed to load agent {i} from {fname}: {e}")
        else:
            if fname.exists():
                print(f"Warning: pickle for agent {i} is empty (size=0): {fname}")
            else:
                print(f"Agent pickle not found: {fname} (running with uninitialized agent {i})")

    # Run the requested number of episodes (no curriculum sweep). Each episode uses
    # the standard final/test scenario (curriculum_level 0). This makes
    # args.episodes control the total number of episodes executed.
    for ep in range(args.episodes):
        curriculum_level = 0
        print(f"[final_environment] Scenario={Scenario} Episode={ep+1}/{args.episodes}")
        env = uoar.DroneEnv(curriculum_level=curriculum_level, scenario=Scenario)
        # If this environment has a different number of drones, recreate agents and try to load
        if env.n_drones != len(agents):
            print(f"Warning: env.n_drones ({env.n_drones}) != loaded agents ({len(agents)}). Recreating agents for this env.")
            obs_dim = env.observation_space.shape[1]
            act_dim = env.action_space.shape[1]
            agents = [uoar.TD3Agent(env.area_size, env.drone_radius, env.fire_line, env.fire_radius, env.safety_margin, env.max_step_size, obs_dim=obs_dim, act_dim=act_dim) for _ in range(env.n_drones)]
            pickles_dir = Path(args.pickles_dir)
            for i in range(len(agents)):
                fname = pickles_dir / args.agent_pattern.format(scenario=Scenario, i=i)
                if fname.exists() and fname.stat().st_size > 0:
                    try:
                        agents[i].load(str(fname))
                        print(f"Loaded agent {i} from {fname} (recreated)")
                    except EOFError:
                        print(f"Warning: pickle for agent {i} appears truncated (EOF) - skipping: {fname}")
                    except Exception as e:
                        print(f"Warning: failed to load agent {i} from {fname}: {e}")
                else:
                    if fname.exists():
                        print(f"Warning: pickle for agent {i} is empty (size=0): {fname}")
                    else:
                        print(f"Agent pickle not found: {fname} (running with uninitialized agent {i})")

        # Initialize section-2 specific environment: start with one fire at (5,5,7.5) and empty scheduled spawns
        env.fire_centers = [np.array([5.0, 5.0, 7.5])]
        env._scheduled_spawns = []

        obs, _ = env.reset()
        # Collect last-episode states for visualization
        episode_states = []
        episode_fire_centers = []
        done = [False] * env.n_drones
        steps = 0
        total_reward = [0.0] * env.n_drones

        while not all(done) and steps < args.steps:
            actions = np.zeros((env.n_drones, act_dim))
            # record pre-step drone positions (x,y,z)
            try:
                episode_states.append(np.copy(obs[:, :3]))
            except Exception:
                # fallback to flatten per-drone positions
                try:
                    episode_states.append(np.copy(obs))
                except Exception:
                    episode_states.append(None)
            # record current fire centers snapshot
            try:
                episode_fire_centers.append([fc.copy() for fc in env.fire_centers])
            except Exception:
                episode_fire_centers.append(None)
            for i in range(env.n_drones):
                if not done[i]:
                    try:
                        actions[i] = agents[i].select_action(obs[i], deterministic=True)
                    except Exception:
                        # If the agent is not ready, use zero action
                        actions[i] = np.zeros((act_dim,))

            next_obs, rewards, done, _, info = env.step(actions, seen_grids=None)
            for i in range(env.n_drones):
                total_reward[i] += float(np.clip(rewards[i], -uoar.REWARD_CLIP, uoar.REWARD_CLIP))
            obs = next_obs
            steps += 1

            # Use helper functions for spawn logic if available
            if fe is not None:
                try:
                    fe.process_scheduled_spawns(env, steps)
                    fe.spawn_random_fire_periodic(env, steps)
                except Exception:
                    pass

        print(f"[final_environment] Finished episode: steps={steps} total_reward={total_reward}")

        # Save the episode states if this is the final episode
        if episode_states and ep == args.episodes - 1:
            last_episode_states = episode_states
            last_episode_fire_centers = episode_fire_centers

    # After all runs, play the final episode visualization once (if available)
    if last_episode_states:
        print("Playing final episode visualization (auto)...")
        plots_dir = workspace / 'plots'
        plots_dir.mkdir(exist_ok=True)
        mid_saved = False
        mid_idx = max(0, len(last_episode_states) // 2)
        for t in range(len(last_episode_states)):
            st = last_episode_states[t]
            fc = last_episode_fire_centers[t] if last_episode_fire_centers and t < len(last_episode_fire_centers) else None
            try:
                env.render(drone_pos=st, fire_centers=fc)
            except Exception:
                pass
            print(f"Step {t}: Drone positions: {np.array(st).round(2) if st is not None else 'N/A'}")
            if plt is not None and (not mid_saved) and t == mid_idx:
                fname = plots_dir / f'visualization_s{args.scenario}_final_ep_step{t}.png'
                try:
                    plt.savefig(fname, dpi=500)
                    print(f"Saved visualization frame to {fname}")
                    mid_saved = True
                except Exception:
                    pass

        # Prompt to save the mid-episode frame (guarded for non-interactive runs)
        if plt is not None:
            try:
                save_input = input('Save mid-episode visualization as PNG? (y/n): ').strip().lower()
            except EOFError:
                save_input = 'n'
            if save_input == 'y':
                if not mid_saved:
                    fname = plots_dir / f'visualization_s{args.scenario}_final_ep_step{mid_idx}.png'
                    try:
                        plt.savefig(fname, dpi=500)
                        print(f"Saved visualization frame to {fname}")
                    except Exception as e:
                        print('Failed to save visualization:', e)
                else:
                    print('Visualization already saved.')
        else:
            print('matplotlib not available; cannot save visualization frames.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
