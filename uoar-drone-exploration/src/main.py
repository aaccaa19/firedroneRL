import matplotlib
matplotlib.use('TkAgg')
from env.uoar_env import UOAREnv
from agents.uoar_agent import RandomAgent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = UOAREnv()
    agent = RandomAgent(env.action_space)
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    plt.ioff()
    plt.show()  # Keeps the window open after the episode ends