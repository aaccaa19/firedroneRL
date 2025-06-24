import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

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

        # Move agent
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

        if np.all(self.visited | self.fire):
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        obs = np.copy(self.visited)
        obs[self.agent_pos[0], self.agent_pos[1]] = 2  # Mark agent
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