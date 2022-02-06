import os
import sys

import gym
import time
import numpy as np
# import gym_novel_gridworlds
import TurtleBot_v0

# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
# from SimpleDQN import SimpleDQN
from SimpleA2C import a2c
import matplotlib.pyplot as plt



def CheckTrainingDoneCallback(reward_array, done_array, env):

	done_cond = False
	reward_cond = False
	if len(done_array) > 30:
		if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
			if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
				done_cond = True

		if done_cond == True:
			if env < 3:
				if np.mean(reward_array[-10:]) > 950:
					reward_cond = True
			else:
				if np.mean(reward_array[-10:]) > 950:
					reward_cond = True

		if done_cond == True and reward_cond == True:
			return 1
		else:
			return 0
	else:
		return 0



if __name__ == "__main__":

	no_of_environmets = 4

	width_array = [2,2.5,3.4,4]
	height_array = [2,2.5,3.4,4]
	no_trees_array = [1,1,3,4]
	no_rocks_array = [0,1,2,2]
	crafting_table_array = [0,0,1,1]
	starting_trees_array = [0,0,0,0]
	starting_rocks_array = [0,0,0,0]
	type_of_env_array = [0,1,2,2]

	# width_array = [11]
	# height_array = [11]
	# no_trees_array = [4]
	# no_rocks_array = [2]
	# crafting_table_array = [1]
	# starting_trees_array = [0]
	# starting_rocks_array = [0]
	# type_of_env_array = [2]

	total_timesteps_array = []
	total_reward_array = []
	avg_reward_array = []
	final_timesteps_array = []
	final_reward_array = []
	final_avg_reward_array = []
	curr_task_completion_array = []
	final_task_completion_array = []

	actionCnt = 5
	D = 83 #8 beams x 4 items lidar + 5 inventory items
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 1

	# agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	# agent.set_explore_epsilon(MAX_EPSILON)
	action_space = ['W','A','D','U','C']
	total_episodes_arr = []

	for i in range(no_of_environmets):
		# print("Environment: ", i)
		# i = 2

		width = width_array[i]
		height = height_array[i]
		no_trees = no_trees_array[i]
		no_rocks = no_rocks_array[i]
		crafting_table = crafting_table_array[i]
		starting_trees = starting_trees_array[i]
		starting_rocks = starting_rocks_array[i]
		type_of_env = type_of_env_array[i]

		final_status = False

		if i == no_of_environmets-1:
			final_status = True

		env_id = 'TurtleBot-v0'
		env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'pogo_stickstone_axe':0},
			initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'stone_axe':0}, goal_env = type_of_env, is_final = final_status)

		a2c(env, i)