import time
import sys

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

sys.path.append('gym_novel_gridworlds/envs')
from novel_gridworld_v7_env import NovelGridworldV7Env


env = NovelGridworldV7Env()
env.goal_env = 0

# Load the trained agent
model = PPO2.load('NovelGridworld-v7_200000_testing_goal_env0_best_model')

for i_episode in range(10):
    print("EPISODE STARTS")
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)
        # End the episode if agent is dead
        if done:
            print("Episode #: "+str(i_episode)+" finished after "+str(i)+" timesteps\n")
            time.sleep(1)
            break
