# UOAR Drone Exploration

This project implements a Gymnasium environment for Unmanned Online Area Reconnaissance (UOAR) with drones. The goal is for drones to explore the maximum area while avoiding fire zones.

## Features

- Custom Gymnasium environment (`UOAREnv`)
- Drones receive rewards for exploring new areas and penalties for entering fire zones
- Simple agent for demonstration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import gymnasium as gym
from src.env.uoar_env import UOAREnv

env = UOAREnv()
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
```

## Requirements

- gymnasium
- numpy
- matplotlib (for rendering)

### Step 1: Set Up Your Project

1. **Create a Project Directory**:
   ```bash
   mkdir drone_exploration
   cd drone_exploration
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   ```bash
   pip install gym numpy matplotlib
   ```

### Step 2: Define the Environment

Create a new file named `drone_env.py` to define the custom Gym environment for the drone exploration task.

```python
import gym
from gym import spaces
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), fire_locations=None):
        super(DroneEnv, self).__init__()
        self.grid_size = grid_size
        self.fire_locations = fire_locations if fire_locations else []
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32)
        self.reset()

    def reset(self):
        self.drone_position = [0, 0]  # Start at the top-left corner
        self.visited = np.zeros(self.grid_size)
        return self._get_observation()

    def step(self, action):
        if action == 0:  # Up
            self.drone_position[0] = max(0, self.drone_position[0] - 1)
        elif action == 1:  # Down
            self.drone_position[0] = min(self.grid_size[0] - 1, self.drone_position[0] + 1)
        elif action == 2:  # Left
            self.drone_position[1] = max(0, self.drone_position[1] - 1)
        elif action == 3:  # Right
            self.drone_position[1] = min(self.grid_size[1] - 1, self.drone_position[1] + 1)

        # Check if the drone is in a fire location
        done = tuple(self.drone_position) in self.fire_locations
        if done:
            reward = -100  # Penalty for hitting a fire
        else:
            self.visited[self.drone_position[0], self.drone_position[1]] = 1
            reward = 1  # Reward for exploring new area

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = self.visited.copy()
        obs[tuple(self.drone_position)] = 1  # Mark the drone's current position
        return obs

    def render(self, mode='human'):
        grid = np.zeros(self.grid_size)
        for fire in self.fire_locations:
            grid[fire] = -1  # Mark fire locations
        grid[tuple(self.drone_position)] = 1  # Mark drone position
        print(grid)
```

### Step 3: Implement the UOAR Algorithm

Create a new file named `uoar.py` to implement the UOAR algorithm.

```python
import numpy as np
import random

class UOAR:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        return random.choice(range(self.env.action_space.n))

    def run(self, episodes=100):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action()
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
```

### Step 4: Main Script to Run the Simulation

Create a new file named `main.py` to run the simulation.

```python
import numpy as np
import gym
from drone_env import DroneEnv
from uoar import UOAR

def main():
    fire_locations = [(2, 2), (3, 3), (5, 5)]  # Example fire locations
    env = DroneEnv(grid_size=(10, 10), fire_locations=fire_locations)
    uoar_agent = UOAR(env)

    uoar_agent.run(episodes=10)

if __name__ == "__main__":
    main()
```

### Step 5: Run the Project

Run the main script to see the UOAR algorithm in action.

```bash
python main.py
```

### Conclusion

This project sets up a basic framework for a drone exploration task using the UOAR algorithm in a custom Gym environment. You can further enhance the UOAR algorithm by implementing more sophisticated strategies for action selection, such as Q-learning or other reinforcement learning techniques. Additionally, you can improve the environment by adding more features, such as obstacles or varying rewards.