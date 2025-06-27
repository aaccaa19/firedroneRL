# This file contains the UOAREnv environment and RuleAgent in a single file for a flat structure.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import collections

class UOAREnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=10, fire_prob=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.fire_prob = fire_prob
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(grid_size, grid_size), dtype=np.int8
        )
        self.fig, self.ax = None, None
        self.im = None
        plt.ion()  # Enable interactive mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.fire = np.random.rand(self.grid_size, self.grid_size) < self.fire_prob
        self.visited = np.zeros_like(self.grid)
        self.agent_pos = [0, 0]
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
        x, y = self.agent_pos
        if action == 0 and x > 0: x -= 1      # up
        elif action == 1 and x < self.grid_size - 1: x += 1  # down
        elif action == 2 and y > 0: y -= 1    # left
        elif action == 3 and y < self.grid_size - 1: y += 1  # right
        self.agent_pos = [x, y]
        reward = 0
        if self.fire[x, y]:
            reward = -10
            self.done = True
        elif not self.visited[x, y]:
            reward = 1
            self.visited[x, y] = 1
        else:
            reward = -1  # Penalize revisiting explored space
        if np.all(self.visited | self.fire):
            self.done = True
        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        # The agent observes a 3x3 grid centered on itself (with out-of-bounds as -2)
        obs = np.full((3, 3), -2, dtype=np.int8)  # -2 for out-of-bounds
        x, y = self.agent_pos
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if [nx, ny] == self.agent_pos:
                        obs[dx+1, dy+1] = 2  # Mark agent
                    elif self.fire[nx, ny]:
                        obs[dx+1, dy+1] = -1  # Fire
                    elif self.visited[nx, ny]:
                        obs[dx+1, dy+1] = -1   # Visited (now -1)
                    else:
                        obs[dx+1, dy+1] = 0   # Unvisited
        return obs

    def render(self):
        img = np.copy(self.visited)
        img[self.fire] = -1
        img[self.agent_pos[0], self.agent_pos[1]] = 2
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(img, cmap='hot', interpolation='nearest')
            plt.title("UOAR Drone Exploration")
        else:
            self.im.set_data(img)
        plt.draw()
        plt.pause(0.1)

class RuleAgent:
    def __init__(self, action_space, grid_size=10):
        self.action_space = action_space
        self.grid_size = grid_size
        self.moves = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right
        self.reset_memory()

    def reset_memory(self):
        self.known_map = np.full((self.grid_size, self.grid_size), None)
        self.position = [0, 0]

    def update_memory(self, obs, pos):
        x, y = pos
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.known_map[nx, ny] = obs[dx+1, dy+1]
        self.position = [x, y]

    def bfs_to_unvisited(self, start):
        # BFS to find the shortest path to any unvisited (0) cell
        queue = collections.deque()
        queue.append((start, []))
        visited = set()
        visited.add(tuple(start))
        while queue:
            (x, y), path = queue.popleft()
            for idx, (dx, dy) in enumerate(self.moves):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) in visited:
                        continue
                    val = self.known_map[nx, ny]
                    if val == 0:
                        return path + [idx]
                    if val != -1 and val is not None:
                        queue.append(((nx, ny), path + [idx]))
                        visited.add((nx, ny))
        return None

    def act(self, observation):
        x, y = self.position
        self.update_memory(observation, [x, y])
        # Prefer unvisited, non-fire neighbors
        for idx, (dx, dy) in enumerate(self.moves):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                val = self.known_map[nx, ny]
                if val == 0:
                    return idx
        # Backtracking: plan path to nearest unvisited cell
        path = self.bfs_to_unvisited([x, y])
        if path:
            return path[0]  # Take the first step along the path
        # Otherwise, prefer any safe neighbor
        for idx, (dx, dy) in enumerate(self.moves):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                val = self.known_map[nx, ny]
                if val != -1 and val is not None:
                    return idx
        return self.action_space.sample()  # fallback: random

def main():
    env = UOAREnv()
    agent = RuleAgent(env.action_space, grid_size=env.grid_size)
    num_episodes = 100
    episode_rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        agent.reset_memory()
        agent.position = [0, 0]
        done = False
        total_reward = 0
        while not done:
            action = agent.act(obs)
            # Update agent's position for memory
            x, y = agent.position
            if action == 0 and x > 0: x -= 1
            elif action == 1 and x < env.grid_size - 1: x += 1
            elif action == 2 and y > 0: y -= 1
            elif action == 3 and y < env.grid_size - 1: y += 1
            agent.position = [x, y]
            obs, reward, done, truncated, info = env.step(action)
            if episode == num_episodes - 1:
                env.render()
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total Points = {total_reward}")
    plt.ioff()
    plt.show()
    # Plot statistics after all episodes
    plt.figure()
    plt.plot(range(1, num_episodes+1), episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Points')
    plt.title('Rule Drone: Total Points per Episode')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
