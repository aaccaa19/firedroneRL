import argparse
import numpy as np
from pathlib import Path

from UOAR_drone_rl import DroneEnv, TD3Agent


def run_eval(weights, scenario, curriculum_level=4, n_episodes=20, steps_per_episode=300):
    weights = Path(weights)
    if not weights.exists():
        print(f"Weights file not found: {weights}")
        return 1
    env = DroneEnv(curriculum_level=curriculum_level, scenario=scenario)
    print(f"Env created: scenario={scenario}, curriculum_level={curriculum_level}")
    print(f"Initial fire_centers (count): {len(getattr(env, 'fire_centers', []))}")
    agents = [TD3Agent(env.area_size, env.drone_radius, env.fire_line, env.fire_radius, env.safety_margin, env.max_step_size, obs_dim=env.observation_space.shape[1], act_dim=env.action_space.shape[1], x_max=env.x_max, y_max=env.y_max, z_min=env.z_min, z_max=env.z_max) for _ in range(env.n_drones)]
    for a in agents:
        a.load(str(weights))
    print(f"Loaded weights into {len(agents)} agents from {weights}")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = [False] * env.n_drones
        steps = 0
        env.fires_visited = [set() for _ in range(env.n_drones)]
        seen_grids = [set() for _ in range(env.n_drones)]
        total_reward = [0.0 for _ in range(env.n_drones)]
        crashed = [False for _ in range(env.n_drones)]
        print('\n=== Episode', ep+1, '===')
        print('Initial drone positions:', env.drone_pos)
        print('Fire centers count:', len(getattr(env, 'fire_centers', [])))
        while not all(done) and steps < steps_per_episode:
            actions = np.zeros((env.n_drones, env.action_space.shape[1]))
            for i in range(env.n_drones):
                if not done[i]:
                    actions[i] = agents[i].select_action(obs[i], deterministic=True)
            next_obs, rewards, done, _, info = env.step(actions, seen_grids=seen_grids)
            for i in range(env.n_drones):
                total_reward[i] += float(rewards[i])
                if rewards[i] <= -50:  # heuristic threshold near collision penalty
                    crashed[i] = True
            obs = next_obs
            steps += 1
        all_fires = set().union(*env.fires_visited) if hasattr(env, 'fires_visited') else set()
        print(f'Episode {ep+1}: steps={steps}, total_reward={total_reward}, crashed={crashed}, fires_visited_per_drone={env.fires_visited}, total_unique_fires={len(all_fires)}')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', '-w', required=True)
    p.add_argument('--scenario', '-s', type=int, default=2)
    p.add_argument('--curriculum', '-c', type=int, default=4)
    p.add_argument('--episodes', '-n', type=int, default=20)
    args = p.parse_args()
    run_eval(args.weights, args.scenario, curriculum_level=args.curriculum, n_episodes=args.episodes)
